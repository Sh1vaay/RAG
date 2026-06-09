import os
import sys
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from typing import Any
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from query_processor import RoutingRetriever, QueryAnalyzer, SearchQuery, compute_rrf
from multi_rep_utils import restore_original_content

def post_filter_documents(docs: list, query: SearchQuery) -> list:
    """Bulletproof post-filtering layer to ensure all retrieved documents strictly match constraints."""
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

def setup_pipeline():
    # 1. Validation Checks
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_openai_api_key_here":
        raise ValueError("OPENAI_API_KEY is not set or is still the default placeholder value in .env.")

    # Check if vectorstore exists
    db_path = "./faiss_db"
    if not os.path.exists(db_path) or not os.listdir(db_path):
        raise FileNotFoundError(f"FAISS database directory '{db_path}' is empty or does not exist.")

    # 2. Check and announce LangSmith integration
    if os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true":
        project = os.getenv("LANGCHAIN_PROJECT", "rag-local-assistant")
        print(f"📊 [LangSmith] Tracing enabled for project: '{project}'")

    # 3. Load the existing FAISS database from disk
    print("Loading database...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.load_local(
        db_path,
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    # 3. Setup Keyword (BM25) and Semantic (FAISS) Retrievers
    print("Initializing hybrid retrieval (BM25 + FAISS)...")
    # Extract all documents currently persisted inside the vector store to seed the BM25 index safely
    if hasattr(vectorstore, "docstore") and hasattr(vectorstore.docstore, "_dict"):
        documents = list(vectorstore.docstore._dict.values())
    else:
        documents = []
        
    if not documents:
        raise ValueError("No documents loaded from vector store database to build BM25 search index.")

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
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5]
    )

    # 4. Setup Reranker (Flashrank or Cohere)
    reranker_provider = os.getenv("RERANKER_PROVIDER", "flashrank").strip().lower()
    if reranker_provider == "cohere":
        cohere_api_key = os.getenv("COHERE_API_KEY")
        if not cohere_api_key or cohere_api_key == "your_cohere_api_key_here":
            print("[WARNING] COHERE_API_KEY is not set. Falling back to local Flashrank.", file=sys.stderr)
            compressor = FlashrankRerank(top_n=3)
            print("Initializing local Cross-Encoder reranking (Flashrank)...")
        else:
            from langchain_cohere import CohereRerank
            compressor = CohereRerank(top_n=3)
            print("Initializing cloud-based Cohere reranking...")
    else:
        compressor = FlashrankRerank(top_n=3)
        print("Initializing local Cross-Encoder reranking (Flashrank)...")

    compression_retriever = CustomCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=ensemble_retriever
    )

    # 5. Setup LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    query_analyzer = QueryAnalyzer(llm)

    # 6. Setup Routing Retriever (Dynamic Translation Router)
    print("Initializing Dynamic Query Translation Router...")
    routing_method = os.getenv("ROUTING_METHOD", "semantic").strip().lower()
    if routing_method not in ("semantic", "llm"):
        routing_method = "semantic"
        
    routing_retriever = RoutingRetriever(
        base_retriever=compression_retriever,
        llm=llm,
        embeddings=embeddings,
        routing_method=routing_method
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
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    # Prompt for final response generation
    qa_system_prompt = (
        "Answer the question based only on the following context:\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    # 8. Build the Chains (LCEL implementations replacing classic chains)
    
    # Standalone query re-writer chain
    query_rewriter = contextualize_q_prompt | llm | StrOutputParser()

    # Routing helper to bypass re-writer when history is empty
    def route_retriever_input(inputs):
        if inputs.get("chat_history"):
            return query_rewriter.invoke(inputs)
        return inputs["input"]

    # History-aware retriever chain (replaces create_history_aware_retriever)
    fast_history_retriever = (
        RunnablePassthrough()
        | route_retriever_input
        | basic_retriever
    )

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
    fast_rag_chain = (
        RunnablePassthrough.assign(context=fast_history_retriever)
        | RunnablePassthrough.assign(answer=question_answer_chain)
    )

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

def main():
    try:
        pipeline = setup_pipeline()
    except Exception as e:
        print(f"[ERROR] Pipeline setup failed: {e}", file=sys.stderr)
        return
        
    llm = pipeline["llm"]
    embeddings = pipeline["embeddings"]
    vectorstore = pipeline["vectorstore"]
    vector_retriever = pipeline["vector_retriever"]
    basic_retriever = pipeline["basic_retriever"]
    ensemble_retriever = pipeline["ensemble_retriever"]
    compression_retriever = pipeline["compression_retriever"]
    query_analyzer = pipeline["query_analyzer"]
    routing_retriever = pipeline["routing_retriever"]
    contextualize_q_prompt = pipeline["contextualize_q_prompt"]
    question_answer_chain = pipeline["question_answer_chain"]
    fast_rag_chain = pipeline["fast_rag_chain"]

    # Initialize in-memory session chat history
    chat_history = []

    # 9. Execution Loop
    print("\n=======================================================")
    print("🚀 Conversational Multi-Query Hybrid RAG Ready.")
    print("Type 'exit' or 'quit' to close.")
    print("=======================================================")
    
    try:
        while True:
            try:
                query = input("\nAsk a question: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nExiting. Goodbye!")
                break
                
            if not query:
                continue
                
            if query.lower() in ('exit', 'quit'):
                print("Goodbye!")
                break
                
            print("Thinking...")
            try:
                # 1. Contextualize query if history exists
                if chat_history:
                    contextualize_chain = contextualize_q_prompt | llm
                    standalone_q = contextualize_chain.invoke({
                        "input": query,
                        "chat_history": chat_history
                    }).content.strip()
                else:
                    standalone_q = query
                
                # 2. Check Semantic Router immediately for Fast Path
                route, _ = routing_retriever.determine_route(standalone_q)
                
                if route == "simple":
                    print("⚡ [Fast Path Triggered] Simple query detected. Bypassing heavy pipeline...")
                    fast_result = fast_rag_chain.invoke({
                        "input": standalone_q,
                        "chat_history": chat_history
                    })
                    answer = fast_result["answer"]
                    context_docs = fast_result.get("context", [])
                    
                    # Display Output & Update History
                    print(f"\nAnswer:\n{answer}")
                    if context_docs:
                        print("\n📚 Top Sources (Fast Retrieval):")
                        for idx, doc in enumerate(context_docs[:3], 1):
                            source_url = doc.metadata.get("source", "Unknown Source")
                            snippet = doc.page_content[:100].strip().replace('\n', ' ')
                            print(f"  [{idx}] {source_url} - \"{snippet}...\"")
                            
                    chat_history.append(HumanMessage(content=query))
                    chat_history.append(AIMessage(content=answer))
                    continue

                # 3. Proceed with Heavy Pipeline
                # Extract query filters/constraints
                structured_query = query_analyzer.analyze(standalone_q)
                
                # Build database filter dictionary
                db_filters = {}
                if structured_query.file_type:
                    db_filters["file_type"] = structured_query.file_type
                if structured_query.publish_year:
                    db_filters["year"] = structured_query.publish_year
                if structured_query.page_number:
                    db_filters["page"] = structured_query.page_number
                if structured_query.data_source:
                    db_filters["data_source"] = structured_query.data_source
                
                # Dynamically inject filter to the retriever
                vector_retriever.search_kwargs["filter"] = db_filters if db_filters else None
                if db_filters:
                    print(f"🎯 [Query Analyzer] Extracted constraints: {db_filters}")
                
                # 4. Execute Route
                if route == "decomposition":
                    from decomposition_graph import create_decomposition_graph
                    # Build and execute the cyclic LangGraph decomposition agent
                    graph = create_decomposition_graph(compression_retriever, llm)

                    state = graph.invoke({
                        "main_question": structured_query.content_search,
                        "sub_questions": [],
                        "current_index": 0,
                        "sub_answers": [],
                        "retrieved_docs": [],
                        "final_answer": ""
                    })

                    answer = state["final_answer"]
                    context_docs = restore_original_content(state["retrieved_docs"])

                elif route in ("standard",) and (
                    any(kw in structured_query.content_search.lower() for kw in
                        ("compare", "versus", "difference", "evaluate", "analyse", "analyze",
                         "pros and cons", "tradeoff", "contrast", "critique", "assessment"))
                ):
                    # ── Agentic CRAG + Self-RAG path (complex analytical queries) ──
                    print("🤖 [Agentic RAG] Complex query detected. Activating CRAG + Self-RAG loop...")
                    from agentic_graph import create_agentic_graph
                    agentic = create_agentic_graph(compression_retriever, llm)
                    agentic_state = agentic.invoke({
                        "question": structured_query.content_search,
                        "rewritten_question": "",
                        "retrieved_docs": [],
                        "relevant_docs": [],
                        "answer": "",
                        "reflection_passed": False,
                        "answer_relevant": False,
                        "retry_count": 0,
                    })
                    answer = agentic_state["answer"]
                    context_docs = restore_original_content(
                        agentic_state["relevant_docs"] or agentic_state["retrieved_docs"]
                    )

                else:
                    # Retrieve documents using chosen translation strategy
                    context_docs = routing_retriever.retrieve_for_route(structured_query.content_search, route)

                    # Apply final post-filtering layer to ensure perfect matches (especially for BM25)
                    context_docs = post_filter_documents(context_docs, structured_query)

                    # Restore original content (swapped out during Multi-Rep indexing)
                    context_docs = restore_original_content(context_docs)

                    # Generate final answer using context
                    answer = question_answer_chain.invoke({
                        "context": context_docs,
                        "input": structured_query.content_search,
                        "chat_history": chat_history
                    })
                
                # 4. Display Output
                print(f"\nAnswer:\n{answer}")
                
                if context_docs:
                    print("\n📚 Top Sources (De-duplicated & Re-ranked):")
                    # De-duplicate context docs for print preview
                    seen_sources = set()
                    unique_sources = []
                    for doc in context_docs:
                        key = (doc.page_content[:100], doc.metadata.get("source", ""))
                        if key not in seen_sources:
                            seen_sources.add(key)
                            unique_sources.append(doc)
                            
                    for idx, doc in enumerate(unique_sources[:5], 1):
                        source_url = doc.metadata.get("source", "Unknown Source")
                        title = doc.metadata.get("title", "Document")
                        snippet = doc.page_content[:150].strip().replace('\n', ' ')
                        print(f"  [{idx}] {title} - {source_url}")
                        print(f"      Snippet preview: \"{snippet}...\"")
                
                # 5. Update Conversation History
                chat_history.append(HumanMessage(content=query))
                chat_history.append(AIMessage(content=answer))
                
            except Exception as e:
                print(f"\n[ERROR] An error occurred while invoking the model: {e}", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred in loop: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
