import os
import sys
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# 1. Structured Output Schema for LLM-based Router
class RouteSelection(BaseModel):
    route: str = Field(
        description="The selected routing method. Must be one of: 'standard', 'multi_query', 'rag_fusion', 'step_back', 'decomposition', 'hyde'"
    )
    reason: str = Field(
        description="The detailed engineering reason for selecting this specific route over the others."
    )

# 2. Prompts for LLM Router
ROUTER_SYSTEM_PROMPT = (
    "You are an expert query routing assistant in a retrieval-augmented generation (RAG) system.\n"
    "Analyze the user's standalone question and select the optimal query translation strategy from the following options:\n\n"
    "1. 'standard': For direct, simple, single-topic keyword searches or specific fact lookup questions.\n"
    "2. 'multi_query': For generic, ambiguous, or synonym-dependent queries where searching from 3 different perspectives improves retrieval.\n"
    "3. 'rag_fusion': For multi-faceted queries where merging and ranking candidate documents using Reciprocal Rank Fusion (RRF) provides the best results.\n"
    "4. 'decomposition': For complex, multi-hop, or comparative questions (e.g. comparing A vs B, differences, list of sequential steps) that must be broken down into sub-questions.\n"
    "5. 'step_back': For highly technical, troubleshooting, coding, or concept-based questions that benefit from abstracting to a broader general principle first.\n"
    "6. 'hyde': For conceptual queries or broad definitions (e.g. 'What is X?') where generating a hypothetical answer helps search matching.\n\n"
    "Analyze the question carefully, make a routing choice, and provide a clear reason."
)
ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", ROUTER_SYSTEM_PROMPT),
    ("human", "{question}")
])

# 3. Custom Zero-Dependency Embedding Semantic Router
class SemanticRouter:
    """Computes cosine similarity between user query and reference samples to route queries in milliseconds."""
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
                "Define BM25 retrieval."
            ],
            "step_back": [
                "Why did my system throw a Connection Timeout during ingestion?",
                "How do I fix a RateLimitError when calling OpenAI?",
                "My vector store throws an index out of bounds error.",
                "How to resolve a SQLite database lock?",
                "Why does the model return empty source documents?",
                "Troubleshoot API keys error.",
                "What is the general concept behind this rate limiting error?"
            ],
            "decomposition": [
                "What is the difference between chain of thought and step back prompting?",
                "Compare Chroma and FAISS in terms of search speed.",
                "What are the pros and cons of semantic chunking versus fixed size splitting?",
                "How does BM25 compare to vector search?",
                "List the steps to configure the pipeline and run evaluation.",
                "Contrast dense and sparse retrieval methods.",
                "Give me a comparison between RAG and fine tuning."
            ],
            "rag_fusion": [
                "How does RAG compare to fine tuning in terms of latency, accuracy, and cost?",
                "What are the best retrieval practices for multi-faceted enterprise search?",
                "Provide a comparative analysis of retrieval methods.",
                "Evaluate different prompt engineering techniques in production."
            ],
            "multi_query": [
                "Tell me about Lilian Weng's research.",
                "Who wrote the article on autonomous agents?",
                "Where can I find the agent framework details?",
                "Give me information about prompt engineering."
            ]
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

# 4. Define Translation Prompts
MULTI_QUERY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are an AI assistant tasked with generating three different versions of the given user question "
               "to retrieve the most relevant documents from a vector database. Provide these alternative questions "
               "separated by newlines. Do not add numbering, bullet points, or introductory text."),
    ("human", "{question}")
])

STEP_BACK_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are an expert technical assistant. Given a specific user question, generate a broader, more abstract "
               "step-back question about the underlying general principles or concepts. Output only the step-back question."),
    ("human", "{question}")
])

DECOMPOSITION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are an AI assistant. Decompose the given user question into 2 or 3 smaller, sequential sub-questions "
               "needed to formulate the final answer. Output only the sub-questions, one per line. Do not add numbering or bullet points."),
    ("human", "{question}")
])

HYDE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Write a short, hypothetical document or textbook paragraph answering the user's question. "
               "Do not write introductions or meta-commentary, just write the factual passage directly."),
    ("human", "{question}")
])

# 5. Helper Algorithms
def compute_rrf(doc_lists: List[List[Document]], k: int = 60, top_n: int = 8) -> List[Document]:
    """Applies Reciprocal Rank Fusion (RRF) to score and combine multiple lists of retrieved documents."""
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
        """Pre-embeds routing reference samples during program startup (only for semantic routing)."""
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
                print(f"[WARNING] LLM routing failed: {e}. Falling back to standard route.", file=sys.stderr)
                route = "standard"
                reason = "LLM routing engine encountered a validation error."

            # Logging LLM Routing Telemetry
            print("\n" + "="*65)
            print(f"🔮 [LLM ROUTER] Selected Route: {route.upper()}")
            print(f"💬 [Reasoning] {reason}")
            print("="*65 + "\n")
            return route, 1.0
        else:
            # Default: Embedding Semantic Router
            if self._semantic_router is None:
                self.initialize_router()
            try:
                route, score = self._semantic_router.route(query)
            except Exception as e:
                print(f"[WARNING] Semantic routing failed: {e}. Falling back to standard route.", file=sys.stderr)
                route = "standard"
                score = 0.0

            # Logging Semantic Routing Telemetry
            print("\n" + "="*65)
            print(f"🔮 [SEMANTIC ROUTER] Selected Route: {route.upper()}")
            print(f"🎯 [Confidence Score] {score:.4f}")
            print("="*65 + "\n")
            return route, score

    def retrieve_for_route(self, query: str, route: str) -> List[Document]:
        """Runs the translation strategy logic and retrieves source documents."""
        if route == "hyde":
            hyde_chain = HYDE_PROMPT | self.llm
            try:
                hypothetical_answer = hyde_chain.invoke({"question": query}).content.strip()
                print(f"📝 [HyDE Passage Generated]:\n\"{hypothetical_answer[:180]}...\"\n")
                return self.base_retriever.invoke(hypothetical_answer)
            except Exception as e:
                print(f"[ERROR] HyDE pipeline failed: {e}", file=sys.stderr)
                return self.base_retriever.invoke(query)

        elif route == "step_back":
            sb_chain = STEP_BACK_PROMPT | self.llm
            try:
                step_back_q = sb_chain.invoke({"question": query}).content.strip()
                print(f"↩️ [Step-Back Query Generated]: \"{step_back_q}\"\n")
                
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
                print(f"🧩 [Decomposed Sub-Questions]:")
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
                print(f"🔥 [RAG-Fusion Multi-Queries Generated]:")
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
                print(f"🔄 [Multi-Query Perspectives Generated]:")
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
            print(f"⚡ [Direct Route]: Standard query lookup.\n")
            return self.base_retriever.invoke(query)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        route, _ = self.determine_route(query)
        return self.retrieve_for_route(query, route)
