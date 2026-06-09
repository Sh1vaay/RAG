import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field


# 1. Structured Output Schema for LLM-based Router
class RouteSelection(BaseModel):
    route: str = Field(
        description=(
            "The selected routing method. Must be one of: "
            "'standard', 'multi_query', 'rag_fusion', 'step_back', "
            "'decomposition', 'hyde'"
        )
    )
    reason: str = Field(
        description=(
            "The detailed engineering reason for selecting this specific "
            "route over the others."
        )
    )


# 2. Prompts for LLM Router
ROUTER_SYSTEM_PROMPT = (
    "You are an expert query routing assistant in a retrieval-augmented generation (RAG) system.\n"
    "Analyze the user's standalone question and select the optimal query "
    "translation strategy from the following options:\n\n"
    "1. 'standard': For direct, simple, single-topic keyword searches or "
    "specific fact lookup questions.\n"
    "2. 'multi_query': For generic, ambiguous, or synonym-dependent queries "
    "where searching from 3 different perspectives improves retrieval.\n"
    "3. 'rag_fusion': For multi-faceted queries where merging and ranking "
    "candidate documents using Reciprocal Rank Fusion (RRF) provides the "
    "best results.\n"
    "4. 'decomposition': For complex, multi-hop, or comparative questions "
    "(e.g. comparing A vs B, differences, list of sequential steps) that "
    "must be broken down into sub-questions.\n"
    "5. 'step_back': For highly technical, troubleshooting, coding, or "
    "concept-based questions that benefit from abstracting to a broader "
    "general principle first.\n"
    "6. 'hyde': For conceptual queries or broad definitions (e.g. 'What is "
    "X?') where generating a hypothetical answer helps search matching.\n\n"
    "Analyze the question carefully, make a routing choice, and provide a clear reason."
)
ROUTER_PROMPT = ChatPromptTemplate.from_messages(
    [("system", ROUTER_SYSTEM_PROMPT), ("human", "{question}")]
)


# 3. Custom Zero-Dependency Embedding Semantic Router
class SemanticRouter:
    """Computes cosine similarity to route queries in milliseconds."""

    def __init__(self, embeddings: OpenAIEmbeddings):
        self.embeddings = embeddings
        self.routes_samples = {
            "hyde": [
                "What is task decomposition?",
                "Explain the concept of semantic chunking.",
                "What does Reciprocal Rank Fusion mean?",
                "Define hypothetical document embeddings.",
                "What is Flashrank?",
                "What is the definition of RAG?",
                "Explain the theory behind cross encoders.",
                "Define BM25 retrieval.",
            ],
            "step_back": [
                "Why did my system throw a Connection Timeout during ingestion?",
                "How do I fix a RateLimitError when calling OpenAI?",
                "My vector store throws an index out of bounds error.",
                "How to resolve a SQLite database lock?",
                "Why does the model return empty source documents?",
                "Troubleshoot API keys error.",
                "What is the general concept behind this rate limiting error?",
            ],
            "decomposition": [
                "What is the difference between chain of thought and step back prompting?",
                "Compare Chroma and FAISS in terms of search speed.",
                "What are the pros and cons of semantic chunking versus fixed size splitting?",
                "How does BM25 compare to vector search?",
                "List the steps to configure the pipeline and run evaluation.",
                "Contrast dense and sparse retrieval methods.",
                "Give me a comparison between RAG and fine tuning.",
            ],
            "rag_fusion": [
                "How does RAG compare to fine tuning in terms of latency, accuracy, and cost?",
                "What are the best retrieval practices for multi-faceted enterprise search?",
                "Provide a comparative analysis of retrieval methods.",
                "Evaluate different prompt engineering techniques in production.",
            ],
            "multi_query": [
                "Tell me about Lilian Weng's research.",
                "Who wrote the article on autonomous agents?",
                "Where can I find the agent framework details?",
                "Give me information about prompt engineering.",
            ],
            "simple": [
                "Hello",
                "Hi there",
                "What is FAISS?",
                "Who made this?",
                "What is the capital of France?",
                "How are you?",
                "Thanks!",
                "What is the definition of RAG?",
            ],
        }
        self.route_embeddings = {}
        self._initialize_embeddings()

    def _initialize_embeddings(self):
        print("Embedding semantic routing reference samples...")
        for route, samples in self.routes_samples.items():
            self.route_embeddings[route] = self.embeddings.embed_documents(samples)

    def route(self, query: str, threshold: float = 0.40) -> Tuple[str, float]:
        query_vector = self.embeddings.embed_query(query)

        best_route = "standard"
        best_score = -1.0

        # Compute Cosine Similarity (dot product of normalized OpenAI vectors)
        for route, embeddings in self.route_embeddings.items():
            for emb in embeddings:
                score = float(np.dot(query_vector, emb))
                if score > best_score:
                    best_score = score
                    best_route = route

        if best_score < threshold:
            # Below confidence threshold, fallback to standard direct lookup
            return "standard", best_score

        return best_route, best_score


# 4. Define Translation Prompts with Technical Few-Shot Examples
MULTI_QUERY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an AI assistant tasked with generating three different versions of "
            "the given user question to retrieve the most relevant documents from a "
            "vector database. Provide these alternative questions "
            "separated by newlines. Do not add numbering, bullet points, or introductory text.\n\n"
            "Example:\n"
            "User Query: What is semantic chunking?\n"
            "Alternative Questions:\n"
            "How does semantic similarity sentence splitting work?\n"
            "What is the difference between semantic chunking and character chunking?\n"
            "Explain semantic boundary detection in document ingestion.",
        ),
        ("human", "{question}"),
    ]
)

STEP_BACK_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are an expert technical assistant. Given a specific user question, "
                "generate a broader, more abstract step-back question about the underlying "
                "general principles or concepts. Output only the step-back question.\n\n"
                "Example 1:\n"
                "User Query: Why did my Tesla Model 3 battery die at 20% in the snow?\n"
                "Step-Back: How does extreme cold affect lithium-ion battery degradation?\n\n"
                "Example 2:\n"
                "User Query: Why does Chroma throw a sqlite3.OperationalError: "
                "table already exists?\n"
                "Step-Back: What are the common causes of database schema conflicts in "
                "SQLite persistence?"
            ),
        ),
        ("human", "{question}"),
    ]
)

DECOMPOSITION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are an AI assistant. Decompose the given user question into 2 or 3 smaller, "
                "sequential sub-questions needed to formulate the final answer. Output only the "
                "sub-questions, one per line. Do not add numbering or bullet points.\n\n"
                "Example:\n"
                "User Query: Compare chain of thought prompting and self consistency.\n"
                "Sub-questions:\n"
                "What is chain of thought prompting?\n"
                "What is self consistency in language models?\n"
                "How do chain of thought and self consistency compare in methodology and accuracy?"
            ),
        ),
        ("human", "{question}"),
    ]
)

HYDE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "Write a short, hypothetical document or textbook paragraph answering "
                "the user's question. Do not write introductions or meta-commentary, "
                "just write the factual passage directly.\n\n"
                "Example:\n"
                "User Query: What is task decomposition?\n"
                "Hypothetical Answer:\n"
                "Task decomposition is a method used to break down a complex task into "
                "smaller, manageable sub-tasks. In the context of AI agents, techniques "
                "like Chain of Thought (CoT) or Tree of Thoughts are applied to split "
                "multi-step problems into sequential "
                "reasoning steps, allowing the LLM to process each part individually."
            ),
        ),
        ("human", "{question}"),
    ]
)


# 5. Helper Algorithms
def compute_rrf(doc_lists: List[List[Document]], k: int = 60, top_n: int = 8) -> List[Document]:
    """Applies Reciprocal Rank Fusion (RRF) to score and combine document lists."""
    rrf_scores: Dict[Tuple[str, str], float] = {}
    doc_map: Dict[Tuple[str, str], Document] = {}

    for doc_list in doc_lists:
        for rank, doc in enumerate(doc_list):
            key = (doc.page_content, doc.metadata.get("source", ""))
            doc_map[key] = doc

            # Score formula: sum( 1 / (k + rank) )
            score = 1.0 / (k + rank)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + score

    # Sort documents by highest RRF score
    sorted_keys = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
    return [doc_map[key] for key, score in sorted_keys[:top_n]]


def merge_and_deduplicate(docs: List[Document]) -> List[Document]:
    """Merges and de-duplicates a list of documents based on content and source."""
    seen = set()
    unique_docs = []
    for doc in docs:
        key = (doc.page_content, doc.metadata.get("source", ""))
        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)
    return unique_docs


def retrieve_parallel(retriever: BaseRetriever, queries: List[str]) -> List[List[Document]]:
    """Concurrently executes base retriever lookups for all queries using a thread pool."""
    with ThreadPoolExecutor(max_workers=len(queries)) as executor:
        results = list(executor.map(retriever.invoke, queries))
    return results


# 6. Custom Routing Retriever Class
class RoutingRetriever(BaseRetriever):
    """LangChain BaseRetriever wrapper that dynamically classifies and translates queries.

    Supports two routing methods:
    - 'semantic': (Default) Local embedding-based similarity classifier (fast & cheap).
    - 'llm': LLM-based structured classifier using OpenAI gpt-4o-mini.
    """

    base_retriever: BaseRetriever
    llm: ChatOpenAI
    embeddings: OpenAIEmbeddings
    routing_method: str = "semantic"

    _semantic_router: Any = None

    class Config:
        arbitrary_types_allowed = True

    def initialize_router(self):
        """Pre-embeds routing reference samples during program startup."""
        if self.routing_method == "semantic" and self._semantic_router is None:
            self._semantic_router = SemanticRouter(self.embeddings)

    def determine_route(self, query: str) -> Tuple[str, float]:
        """Classifies the query intent to select the translation route."""
        if self.routing_method == "llm":
            # Invoke LLM structured router
            router_chain = ROUTER_PROMPT | self.llm.with_structured_output(RouteSelection)
            try:
                decision = router_chain.invoke({"question": query})
                route = decision.route.strip().lower()
                reason = decision.reason
            except Exception as e:
                print(
                    f"[WARNING] LLM routing failed: {e}. Falling back to standard route.",
                    file=sys.stderr,
                )
                route = "standard"
                reason = "LLM routing engine encountered a validation error."

            # Logging LLM Routing Telemetry
            print("\n" + "=" * 65)
            print(f"🔮 [LLM ROUTER] Selected Route: {route.upper()}")
            print(f"💬 [Reasoning] {reason}")
            print("=" * 65 + "\n")
            return route, 1.0
        else:
            # Default: Embedding Semantic Router
            if self._semantic_router is None:
                self.initialize_router()
            try:
                route, score = self._semantic_router.route(query)
            except Exception as e:
                print(
                    f"[WARNING] Semantic routing failed: {e}. Falling back to standard route.",
                    file=sys.stderr,
                )
                route = "standard"
                score = 0.0

            # Logging Semantic Routing Telemetry
            print("\n" + "=" * 65)
            print(f"🔮 [SEMANTIC ROUTER] Selected Route: {route.upper()}")
            print(f"🎯 [Confidence Score] {score:.4f}")
            print("=" * 65 + "\n")
            return route, score

    def retrieve_for_route(self, query: str, route: str) -> List[Document]:
        """Runs the translation strategy logic and retrieves source documents."""
        # Route 1: Hypothetical Document Embeddings (HyDE) with similarity threshold check
        if route == "hyde":
            hyde_chain = HYDE_PROMPT | self.llm
            try:
                hypothetical_answer = hyde_chain.invoke({"question": query}).content.strip()

                # Check Cosine Similarity of query vs hypothetical answer to catch hallucination
                query_vector = self.embeddings.embed_query(query)
                hyde_vector = self.embeddings.embed_query(hypothetical_answer)
                similarity = float(np.dot(query_vector, hyde_vector))

                # Safety similarity threshold (standard for OpenAI text-embedding-3-small)
                threshold = 0.60

                print(f"⚖️ [HyDE Similarity Check] Score: {similarity:.4f} (Threshold: {threshold})")

                if similarity >= threshold:
                    print(f'📝 [HyDE Accepted & Querying]:\n"{hypothetical_answer[:180]}..."\n')
                    return self.base_retriever.invoke(hypothetical_answer)
                else:
                    print(
                        "[WARNING] HyDE similarity below threshold. Hallucination drift "
                        "detected. Discarding passage and falling back to original query.\n"
                    )
                    return self.base_retriever.invoke(query)
            except Exception as e:
                print(f"[ERROR] HyDE pipeline failed: {e}", file=sys.stderr)
                return self.base_retriever.invoke(query)

        elif route == "step_back":
            sb_chain = STEP_BACK_PROMPT | self.llm
            try:
                step_back_q = sb_chain.invoke({"question": query}).content.strip()
                print(f'↩️ [Step-Back Query Generated]: "{step_back_q}"\n')

                retrieved_lists = retrieve_parallel(self.base_retriever, [query, step_back_q])
                return merge_and_deduplicate(retrieved_lists[0] + retrieved_lists[1])
            except Exception as e:
                print(f"[ERROR] Step-Back pipeline failed: {e}", file=sys.stderr)
                return self.base_retriever.invoke(query)

        elif route == "decomposition":
            # Direct/naive retrieval fallback
            dec_chain = DECOMPOSITION_PROMPT | self.llm
            try:
                sub_q_text = dec_chain.invoke({"question": query}).content.strip()
                sub_queries = [q.strip() for q in sub_q_text.split("\n") if q.strip()]
                print("🧩 [Decomposed Sub-Questions]:")
                for idx, q in enumerate(sub_queries, 1):
                    print(f"  {idx}. {q}")
                print()

                retrieved_lists = retrieve_parallel(self.base_retriever, sub_queries)
                flat_docs = []
                for doc_list in retrieved_lists:
                    flat_docs.extend(doc_list)
                return merge_and_deduplicate(flat_docs)
            except Exception as e:
                print(f"[ERROR] Decomposition pipeline failed: {e}", file=sys.stderr)
                return self.base_retriever.invoke(query)

        elif route == "rag_fusion":
            mq_chain = MULTI_QUERY_PROMPT | self.llm
            try:
                mq_text = mq_chain.invoke({"question": query}).content.strip()
                queries = [query] + [q.strip() for q in mq_text.split("\n") if q.strip()]
                print("🔥 [RAG-Fusion Multi-Queries Generated]:")
                for idx, q in enumerate(queries, 1):
                    print(f"  {idx}. {q}")
                print()

                retrieved_lists = retrieve_parallel(self.base_retriever, queries)
                return compute_rrf(retrieved_lists, k=60, top_n=8)
            except Exception as e:
                print(f"[ERROR] RAG-Fusion pipeline failed: {e}", file=sys.stderr)
                return self.base_retriever.invoke(query)

        elif route == "multi_query":
            mq_chain = MULTI_QUERY_PROMPT | self.llm
            try:
                mq_text = mq_chain.invoke({"question": query}).content.strip()
                queries = [query] + [q.strip() for q in mq_text.split("\n") if q.strip()]
                print("🔄 [Multi-Query Perspectives Generated]:")
                for idx, q in enumerate(queries, 1):
                    print(f"  {idx}. {q}")
                print()

                retrieved_lists = retrieve_parallel(self.base_retriever, queries)
                flat_docs = []
                for doc_list in retrieved_lists:
                    flat_docs.extend(doc_list)
                return merge_and_deduplicate(flat_docs)
            except Exception as e:
                print(f"[ERROR] Multi-Query pipeline failed: {e}", file=sys.stderr)
                return self.base_retriever.invoke(query)

        else:
            print("⚡ [Direct Route]: Standard query lookup.\n")
            return self.base_retriever.invoke(query)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        route, _ = self.determine_route(query)
        return self.retrieve_for_route(query, route)


# 7. Query Analyzer & Schema for Metadata Filtering
class SearchQuery(BaseModel):
    """Structured query analysis output holding core search term and optional filters."""

    content_search: str = Field(..., description="The core semantic query for similarity search.")
    file_type: Optional[str] = Field(
        None, description="Filter by file type. Must be one of: 'pdf', 'csv', 'docx', 'txt', 'web'."
    )
    publish_year: Optional[int] = Field(
        None,
        description="Specific year the document was published/created (four digits, e.g. 2023).",
    )
    page_number: Optional[int] = Field(
        None, description="Specific 1-indexed page number to extract (only for PDFs)."
    )
    data_source: Optional[Literal["web_blogs", "academic_papers", "internal_docs"]] = Field(
        None,
        description=(
            "Filter by data source index. 'web_blogs' for blogs/URLs, "
            "'academic_papers' for research/scientific papers, "
            "'internal_docs' for general local documentation files."
        ),
    )


class QueryAnalyzer:
    """Invokes structured parsing to separate search terms from metadata constraints."""

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.structured_llm = self.llm.with_structured_output(SearchQuery)

    def analyze(self, question: str) -> SearchQuery:
        current_date_str = datetime.now().strftime("%Y-%m-%d")
        system_prompt = (
            "You are an expert query analyzer. Your task is to split a user's "
            "natural language question into a core semantic search query "
            "(content_search) and explicit metadata filters.\n\n"
            f"The current date is {current_date_str}.\n"
            "Use this date to resolve relative dates mentioned in the question "
            "(like 'within the last year', 'published in the last 6 months', "
            "'this year') into a specific year value.\n\n"
            "Metadata Schema:\n"
            "- file_type: 'pdf', 'csv', 'docx', 'txt', 'web'\n"
            "- publish_year: four digit year (e.g. 2023)\n"
            "- page_number: page number to fetch (only if user explicitly says "
            "page 2, page 5, etc.)\n"
            "- data_source: 'web_blogs' (for blogs/web pages), 'academic_papers' "
            "(for research papers/journals), 'internal_docs' (for general local "
            "documentation)\n\n"
            "Be precise. If any constraint is not explicitly mentioned or "
            "clearly implied, return null for that field."
        )
        prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("human", "{question}")]
        )
        chain = prompt | self.structured_llm
        try:
            return chain.invoke({"question": question})
        except Exception as e:
            print(
                f"[WARNING] Query analyzer failed: {e}. Falling back to default search.",
                file=sys.stderr,
            )
            return SearchQuery(content_search=question)
