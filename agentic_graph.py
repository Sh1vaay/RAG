"""
agentic_graph.py
────────────────
Corrective RAG (CRAG) + Self-RAG implemented as a LangGraph cyclic graph.

This module is imported lazily from main.py only when the Semantic Router
classifies a query as complex. All existing fast-path routes are unaffected.

Graph flow:
  retrieve → grade_documents
               ├─ relevant   → generate → reflect
               │                            ├─ grounded   → END
               │                            └─ hallucination → generate (retry, max 2)
               └─ irrelevant → rewrite_query → web_search → generate → reflect
"""

import sys
from typing import TypedDict, List, Annotated
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END


# ── State ─────────────────────────────────────────────────────────────────────

class AgenticState(TypedDict):
    question: str
    rewritten_question: str
    retrieved_docs: List[Document]
    relevant_docs: List[Document]
    answer: str
    reflection_passed: bool
    answer_relevant: bool
    retry_count: int


# ── Node Prompts ──────────────────────────────────────────────────────────────

_GRADE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a relevance grader. Given a user question and a retrieved document, "
        "decide if the document is relevant to answering the question.\n"
        "Reply with exactly one word: 'yes' or 'no'. No other text.",
    ),
    ("human", "Question: {question}\n\nDocument:\n{document}"),
])

_REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a query rewriter. The initial retrieval returned no useful documents. "
        "Rewrite the question to be more general and better suited for vector search. "
        "Output only the rewritten question — no preamble.",
    ),
    ("human", "Original question: {question}"),
])

_GENERATE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful expert assistant. Answer the question strictly based on the "
        "provided context documents. If the context does not contain enough information, "
        "say so explicitly. Do not fabricate information.\n\nContext:\n{context}",
    ),
    ("human", "{question}"),
])

_REFLECT_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a hallucination detector. Given an answer and the source documents used "
        "to produce it, decide if the answer contains any claims NOT supported by the documents.\n"
        "Reply with exactly one word: 'grounded' if fully supported, 'hallucination' if not.",
    ),
    ("human", "Answer:\n{answer}\n\nSource Documents:\n{context}"),
])

_ANSWER_GRADE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an answer relevance grader. Given a user question and a generated answer, "
        "decide if the answer actually addresses (answers) the original question directly.\n"
        "Reply with exactly one word: 'yes' or 'no'. No other text.",
    ),
    ("human", "User Question: {question}\n\nGenerated Answer:\n{answer}"),
])


# ── Graph Factory ─────────────────────────────────────────────────────────────

def create_agentic_graph(retriever: BaseRetriever, llm: ChatOpenAI):
    """
    Returns a compiled LangGraph for CRAG + Self-RAG.

    Parameters
    ----------
    retriever : BaseRetriever
        The existing compression_retriever (BM25 + FAISS + Flashrank). Reused as-is.
    llm : ChatOpenAI
        The shared gpt-4o-mini instance from main.py.
    """

    grade_chain        = _GRADE_PROMPT        | llm
    rewrite_chain      = _REWRITE_PROMPT      | llm
    gen_chain          = _GENERATE_PROMPT     | llm
    reflect_chain      = _REFLECT_PROMPT      | llm
    answer_grade_chain = _ANSWER_GRADE_PROMPT | llm

    # ── Nodes ─────────────────────────────────────────────────────────────────

    def retrieve(state: AgenticState) -> dict:
        q = state.get("rewritten_question") or state["question"]
        print(f"🔍 [Agentic: Retrieve] Query: \"{q}\"")
        try:
            docs = retriever.invoke(q)
        except Exception as exc:
            print(f"[WARNING] Retrieval failed: {exc}", file=sys.stderr)
            docs = []
        return {"retrieved_docs": docs}

    def grade_documents(state: AgenticState) -> dict:
        print("⚖️  [Agentic: Grade Documents]")
        question = state.get("rewritten_question") or state["question"]
        relevant = []
        for doc in state["retrieved_docs"]:
            try:
                verdict = grade_chain.invoke({
                    "question": question,
                    "document": doc.page_content[:800],
                }).content.strip().lower()
            except Exception:
                verdict = "yes"  # assume relevant on error
            if verdict == "yes":
                relevant.append(doc)
        print(f"   {len(relevant)}/{len(state['retrieved_docs'])} documents graded as relevant.")
        return {"relevant_docs": relevant}

    def rewrite_query(state: AgenticState) -> dict:
        print("✏️  [Agentic: Rewrite Query]")
        try:
            new_q = rewrite_chain.invoke({"question": state["question"]}).content.strip()
        except Exception as exc:
            print(f"[WARNING] Rewrite failed: {exc}", file=sys.stderr)
            new_q = state["question"]
        print(f"   Rewritten: \"{new_q}\"")
        return {"rewritten_question": new_q}

    def web_search(state: AgenticState) -> dict:
        print("🌐 [Agentic: Web Search]")
        search_query = state.get("rewritten_question") or state["question"]
        try:
            from langchain_community.tools import DuckDuckGoSearchRun
            search = DuckDuckGoSearchRun()
            search_result = search.run(search_query)
            # Create a Document wrapping the search results
            web_doc = Document(
                page_content=search_result,
                metadata={"source": "duckduckgo", "title": "Web Search Result"}
            )
            return {"relevant_docs": [web_doc]}
        except Exception as exc:
            print(f"[ERROR] Web search failed: {exc}", file=sys.stderr)
            return {"relevant_docs": []}

    def generate(state: AgenticState) -> dict:
        print("💬 [Agentic: Generate Answer]")
        docs = state["relevant_docs"] or state["retrieved_docs"]
        context = "\n\n".join(
            f"Source: {d.metadata.get('source', 'Unknown')}\n{d.page_content}"
            for d in docs
        )
        try:
            answer = gen_chain.invoke({
                "question": state.get("rewritten_question") or state["question"],
                "context": context,
            }).content.strip()
        except Exception as exc:
            print(f"[ERROR] Generation failed: {exc}", file=sys.stderr)
            answer = "I was unable to generate an answer."
        retry = state.get("retry_count", 0)
        return {"answer": answer, "retry_count": retry}

    def reflect(state: AgenticState) -> dict:
        print("🪞 [Agentic: Self-Reflect]")
        docs = state["relevant_docs"] or state["retrieved_docs"]
        context = "\n\n".join(d.page_content for d in docs)
        try:
            verdict = reflect_chain.invoke({
                "answer": state["answer"],
                "context": context,
            }).content.strip().lower()
        except Exception:
            verdict = "grounded"
        passed = verdict == "grounded"
        print(f"   Reflection verdict: {'✅ grounded' if passed else '⚠️  hallucination detected'}")
        return {
            "reflection_passed": passed,
            "retry_count": state.get("retry_count", 0) + (0 if passed else 1),
        }

    def grade_answer(state: AgenticState) -> dict:
        print("⚖️  [Agentic: Grade Answer Relevance]")
        question = state["question"]
        answer = state["answer"]
        try:
            verdict = answer_grade_chain.invoke({
                "question": question,
                "answer": answer,
            }).content.strip().lower()
        except Exception:
            verdict = "yes"  # assume relevant on error
        passed = verdict == "yes"
        print(f"   Answer relevance verdict: {'✅ addresses question' if passed else '⚠️  does not address question'}")
        return {
            "answer_relevant": passed,
            "retry_count": state.get("retry_count", 0) + (0 if passed else 1),
        }

    # ── Routing Conditions ────────────────────────────────────────────────────

    def after_grade(state: AgenticState) -> str:
        if state["relevant_docs"]:
            return "generate"
        return "rewrite_query"

    def after_reflect(state: AgenticState) -> str:
        if state["reflection_passed"]:
            return "grade_answer"
        # Avoid infinite loops — max 2 reflection retries
        if state.get("retry_count", 0) >= 2:
            print("   ⚠️  Max retries reached. Returning best available answer.")
            return END
        return "generate"

    def after_grade_answer(state: AgenticState) -> str:
        if state["answer_relevant"]:
            return END
        # Avoid infinite loops
        if state.get("retry_count", 0) >= 2:
            print("   ⚠️  Max retries reached. Returning best available answer.")
            return END
        return "rewrite_query"

    # ── Build Graph ───────────────────────────────────────────────────────────

    workflow = StateGraph(AgenticState)
    workflow.add_node("retrieve",       retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("rewrite_query",  rewrite_query)
    workflow.add_node("web_search",     web_search)
    workflow.add_node("generate",       generate)
    workflow.add_node("reflect",        reflect)
    workflow.add_node("grade_answer",   grade_answer)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        after_grade,
        {"generate": "generate", "rewrite_query": "rewrite_query"},
    )
    workflow.add_edge("rewrite_query", "web_search")
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", "reflect")
    workflow.add_conditional_edges(
        "reflect",
        after_reflect,
        {"generate": "generate", "grade_answer": "grade_answer", END: END},
    )
    workflow.add_conditional_edges(
        "grade_answer",
        after_grade_answer,
        {"rewrite_query": "rewrite_query", END: END},
    )

    return workflow.compile()


