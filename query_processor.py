import os
import sys
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_openai import ChatOpenAI

# 1. Define the Structured Pydantic Model for Query Routing
class RouteSelection(BaseModel):
    route: str = Field(
        description="The selected routing method. Must be one of: 'standard', 'multi_query', 'rag_fusion', 'step_back', 'decomposition', 'hyde'"
    )
    reason: str = Field(
        description="The detailed engineering reason for selecting this specific route over the others."
    )

# 2. Define the Routing Prompt
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

# 3. Define Translation Prompts
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

# 4. Helper Algorithms
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

# 5. Custom Routing Retriever Class
class RoutingRetriever(BaseRetriever):
    """LangChain BaseRetriever wrapper that dynamically classifies and translates queries."""
    base_retriever: BaseRetriever
    llm: ChatOpenAI
    
    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        # Invoke Structured Classification Router
        router_chain = ROUTER_PROMPT | self.llm.with_structured_output(RouteSelection)
        try:
            decision = router_chain.invoke({"question": query})
            route = decision.route.strip().lower()
            reason = decision.reason
        except Exception as e:
            print(f"[WARNING] Query routing failed: {e}. Falling back to standard route.", file=sys.stderr)
            route = "standard"
            reason = "Routing engine failed or encountered a validation error."

        # Logging Routing Telemetry to Console
        print("\n" + "="*65)
        print(f"🔮 [QUERY ROUTER] Selected Route: {route.upper()}")
        print(f"💬 [Reasoning] {reason}")
        print("="*65 + "\n")

        # Route 1: Hypothetical Document Embeddings (HyDE)
        if route == "hyde":
            hyde_chain = HYDE_PROMPT | self.llm
            try:
                hypothetical_answer = hyde_chain.invoke({"question": query}).content.strip()
                print(f"📝 [HyDE Passage Generated]:\n\"{hypothetical_answer[:180]}...\"\n")
                return self.base_retriever.invoke(hypothetical_answer)
            except Exception as e:
                print(f"[ERROR] HyDE pipeline failed: {e}", file=sys.stderr)
                return self.base_retriever.invoke(query)

        # Route 2: Step-Back Prompting
        elif route == "step_back":
            sb_chain = STEP_BACK_PROMPT | self.llm
            try:
                step_back_q = sb_chain.invoke({"question": query}).content.strip()
                print(f"↩️ [Step-Back Query Generated]: \"{step_back_q}\"\n")
                
                # Retrieve concurrently for both original and abstracted queries
                retrieved_lists = retrieve_parallel(self.base_retriever, [query, step_back_q])
                return merge_and_deduplicate(retrieved_lists[0] + retrieved_lists[1])
            except Exception as e:
                print(f"[ERROR] Step-Back pipeline failed: {e}", file=sys.stderr)
                return self.base_retriever.invoke(query)

        # Route 3: Query Decomposition
        elif route == "decomposition":
            dec_chain = DECOMPOSITION_PROMPT | self.llm
            try:
                sub_q_text = dec_chain.invoke({"question": query}).content.strip()
                sub_queries = [q.strip() for q in sub_q_text.split("\n") if q.strip()]
                print(f"🧩 [Decomposed Sub-Questions]:")
                for idx, q in enumerate(sub_queries, 1):
                    print(f"  {idx}. {q}")
                print()
                
                # Retrieve concurrently for all sub-questions
                retrieved_lists = retrieve_parallel(self.base_retriever, sub_queries)
                flat_docs = []
                for doc_list in retrieved_lists:
                    flat_docs.extend(doc_list)
                return merge_and_deduplicate(flat_docs)
            except Exception as e:
                print(f"[ERROR] Decomposition pipeline failed: {e}", file=sys.stderr)
                return self.base_retriever.invoke(query)

        # Route 4: RAG-Fusion (Multi-Query + Reciprocal Rank Fusion)
        elif route == "rag_fusion":
            mq_chain = MULTI_QUERY_PROMPT | self.llm
            try:
                mq_text = mq_chain.invoke({"question": query}).content.strip()
                queries = [query] + [q.strip() for q in mq_text.split("\n") if q.strip()]
                print(f"🔥 [RAG-Fusion Multi-Queries Generated]:")
                for idx, q in enumerate(queries, 1):
                    print(f"  {idx}. {q}")
                print()
                
                # Retrieve concurrently for all query variations
                retrieved_lists = retrieve_parallel(self.base_retriever, queries)
                return compute_rrf(retrieved_lists, k=60, top_n=8)
            except Exception as e:
                print(f"[ERROR] RAG-Fusion pipeline failed: {e}", file=sys.stderr)
                return self.base_retriever.invoke(query)

        # Route 5: Multi-Query
        elif route == "multi_query":
            mq_chain = MULTI_QUERY_PROMPT | self.llm
            try:
                mq_text = mq_chain.invoke({"question": query}).content.strip()
                queries = [query] + [q.strip() for q in mq_text.split("\n") if q.strip()]
                print(f"🔄 [Multi-Query Perspectives Generated]:")
                for idx, q in enumerate(queries, 1):
                    print(f"  {idx}. {q}")
                print()
                
                # Retrieve concurrently for all variations
                retrieved_lists = retrieve_parallel(self.base_retriever, queries)
                flat_docs = []
                for doc_list in retrieved_lists:
                    flat_docs.extend(doc_list)
                return merge_and_deduplicate(flat_docs)
            except Exception as e:
                print(f"[ERROR] Multi-Query pipeline failed: {e}", file=sys.stderr)
                return self.base_retriever.invoke(query)

        # Default Route: Standard Lookup
        else:
            print(f"⚡ [Direct Route]: Standard query lookup.\n")
            return self.base_retriever.invoke(query)
