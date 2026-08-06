import os
import sys
from typing import Any

from dotenv import load_dotenv
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnablePassthrough

from .multi_rep_utils import restore_original_content
from .providers import ProviderConfig, get_embeddings, get_llm, get_reranker
from .query_processor import QueryAnalyzer, RoutingRetriever, SearchQuery, compute_rrf
from .user_config import load_user_config
from .workspace import Workspace


def post_filter_documents(docs: list, query: SearchQuery) -> list:
    """Post-filtering layer to ensure retrieved documents strictly match constraints."""
    filtered_docs = []
    for doc in docs:
        # Check file_type
        if query.file_type and doc.metadata.get("file_type") != query.file_type:
            continue
        # Check year
        if query.publish_year and doc.metadata.get("year") != query.publish_year:
            continue
        # Check page number
        if query.page_number and doc.metadata.get("page") != query.page_number:
            continue
        # Check data_source
        if query.data_source and doc.metadata.get("data_source") != query.data_source:
            continue
        filtered_docs.append(doc)
    return filtered_docs


class CustomEnsembleRetriever(BaseRetriever):
    retrievers: list
    weights: list

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list:
        retrieved_lists = [r.invoke(query) for r in self.retrievers]
        return compute_rrf(retrieved_lists)


class CustomCompressionRetriever(BaseRetriever):
    base_retriever: BaseRetriever
    base_compressor: Any

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list:
        docs = self.base_retriever.invoke(query)
        if not docs:
            return []
        return self.base_compressor.compress_documents(docs, query)


# Load environment variables
load_dotenv()

def setup_pipeline(
    workspace: Workspace | None = None,
    config: ProviderConfig | None = None,
):
    """Build the retrieval pipeline for one workspace.

    `workspace` scopes which user's index is loaded; `config` decides which models
    build it. Both default to the shared single-user values, so the CLI and any
    pre-auth caller keep working unchanged.
    """
    ws = workspace or Workspace.for_user(None)
    cfg = config or load_user_config(ws)

    # 1. Validation Checks
    # Provider selection, key presence, and (for Ollama) daemon reachability are all
    # validated inside the factory, so a misconfiguration fails here with a clear message.
    print(f"🔌 [Providers] {cfg.describe()}")

    db_path = str(ws.index_dir)
    if not ws.has_index:
        raise FileNotFoundError(
            f"FAISS index for workspace '{ws.user_id}' is empty or does not exist "
            f"({db_path}). Upload documents and build the index first."
        )

    # 2. Check and announce LangSmith integration
    if os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true":
        project = os.getenv("LANGCHAIN_PROJECT", "rag-local-assistant")
        print(f"📊 [LangSmith] Tracing enabled for project: '{project}'")

    # 3. Load the existing FAISS database from disk
    print(f"Loading database for workspace '{ws.user_id}'...")
    embeddings = get_embeddings(cfg)
    vectorstore = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)

    # 3. Setup Keyword (BM25) and Semantic (FAISS) Retrievers
    print("Initializing hybrid retrieval (BM25 + FAISS)...")
    # Seed the BM25 index safely using all documents from the vector store
    if hasattr(vectorstore, "docstore") and hasattr(vectorstore.docstore, "_dict"):
        documents = list(vectorstore.docstore._dict.values())
    else:
        documents = []

    if not documents:
        raise ValueError(
            "No documents loaded from vector store database to build BM25 search index."
        )

    # Initialize keyword search on exact document contents
    bm25_retriever = BM25Retriever.from_documents(documents)
    # Retrieve a broader pool of candidates (k=8) to allow the reranker to prune
    bm25_retriever.k = 8

    # Initialize semantic/vector search (k=8)
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

    # Initialize basic retriever (k=3) for the Fast Path bypass
    basic_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Combine retrievers using Reciprocal Rank Fusion (RRF)
    ensemble_retriever = CustomEnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever], weights=[0.5, 0.5]
    )

    # 4. Setup Reranker (Flashrank or Cohere)
    compressor = get_reranker(cfg)

    compression_retriever = CustomCompressionRetriever(
        base_compressor=compressor, base_retriever=ensemble_retriever
    )

    # 5. Setup LLM
    llm = get_llm(cfg)
    query_analyzer = QueryAnalyzer(llm)

    # 6. Setup Routing Retriever (Dynamic Translation Router)
    print("Initializing Dynamic Query Translation Router...")
    # Routing comes from this user's config, not the process environment.
    routing_method = cfg.routing_method

    routing_retriever = RoutingRetriever(
        base_retriever=compression_retriever,
        llm=llm,
        embeddings=embeddings,
        routing_method=routing_method,
    )
    routing_retriever.initialize_router()

    # 7. Define Prompts for Conversational Flow

    # Prompt to reformulate follow-up questions to be standalone
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # Prompt for final response generation
    # Retrieval always returns *something*: FAISS hands back the k nearest vectors
    # whether or not they are relevant, and the cross-encoder is unreliable
    # out-of-domain (measured: an unrelated question scored 0.99 against an
    # unrelated chunk). So relevance cannot be decided before generation — the
    # model has to read the context and judge. This prompt lets it do that and
    # answer anyway, rather than replying "no information provided" to every
    # question the documents happen not to cover.
    qa_system_prompt = (
        "You are a helpful AI assistant. Use the following context to answer the user's question. "
        "The context may include excerpts from the user's uploaded documents or DuckDuckGo web search results. "
        "If the context does not contain the answer, draw on your general knowledge and clearly indicate "
        "when you are doing so. "
        "Never invent details about the documents themselves.\n\n"
        "Format for reading, in Markdown. Use short paragraphs; a bullet list when "
        "you give several facts; a numbered list only when order actually matters; "
        "**bold** for key figures and terms. Keep it as short as the question allows "
        "and do not pad the answer with restatements of the question.\n\n"
        "Context:\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # 8. Build the Chains (LCEL implementations replacing classic chains)

    # Standalone query re-writer chain
    query_rewriter = contextualize_q_prompt | llm | StrOutputParser()

    # Routing helper to bypass re-writer when history is empty
    def route_retriever_input(inputs):
        if inputs.get("chat_history"):
            return query_rewriter.invoke(inputs)
        return inputs["input"]

    # History-aware retriever chain (replaces create_history_aware_retriever)
    fast_history_retriever = RunnablePassthrough() | route_retriever_input | basic_retriever

    # Helper function to format docs for prompt injection
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Document QA response generator (replaces create_stuff_documents_chain)
    question_answer_chain = (
        RunnablePassthrough.assign(context=lambda x: format_docs(x["context"]))
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    # Complete retrieval chain linking retrieval and QA (replaces create_retrieval_chain)
    fast_rag_chain = RunnablePassthrough.assign(
        context=fast_history_retriever
    ) | RunnablePassthrough.assign(answer=question_answer_chain)

    return {
        "llm": llm,
        "embeddings": embeddings,
        "vectorstore": vectorstore,
        "vector_retriever": vector_retriever,
        "basic_retriever": basic_retriever,
        "ensemble_retriever": ensemble_retriever,
        "compression_retriever": compression_retriever,
        "query_analyzer": query_analyzer,
        "routing_retriever": routing_retriever,
        "contextualize_q_prompt": contextualize_q_prompt,
        "question_answer_chain": question_answer_chain,
        "fast_rag_chain": fast_rag_chain,
    }
