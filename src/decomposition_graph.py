import sys
from typing import Any, Dict, List, TypedDict

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph


# 1. State Definition
class DecompositionState(TypedDict):
    main_question: str
    sub_questions: List[str]
    current_index: int
    sub_answers: List[Dict[str, str]]
    retrieved_docs: List[Document]
    final_answer: str


# 2. Graph Construction Function
def create_decomposition_graph(retriever: BaseRetriever, llm: ChatOpenAI):
    """Compiles a LangGraph cyclic workflow to answer multi-hop questions sequentially."""

    def generate_sub_questions(state: DecompositionState) -> Dict[str, Any]:
        print("🤖 [LangGraph: Node - Generate Sub-questions]")
        dec_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an AI assistant. Decompose the given user question "
                    "into 2 or 3 smaller, sequential sub-questions needed to "
                    "formulate the final answer. Output only the sub-questions, "
                    "one per line. Do not add numbering or bullet points.",
                ),
                ("human", "{question}"),
            ]
        )
        dec_chain = dec_prompt | llm
        try:
            sub_q_text = dec_chain.invoke({"question": state["main_question"]}).content.strip()
            sub_queries = [q.strip() for q in sub_q_text.split("\n") if q.strip()]
            print("🧩 Decomposed Sub-Questions:")
            for idx, q in enumerate(sub_queries, 1):
                print(f"  {idx}. {q}")
            return {
                "sub_questions": sub_queries,
                "current_index": 0,
                "sub_answers": [],
                "retrieved_docs": [],
            }
        except Exception as e:
            print(f"[ERROR] Failed to decompose question: {e}", file=sys.stderr)
            # Fallback to main question as a single sub-question
            return {
                "sub_questions": [state["main_question"]],
                "current_index": 0,
                "sub_answers": [],
                "retrieved_docs": [],
            }

    def answer_sub_question(state: DecompositionState) -> Dict[str, Any]:
        idx = state["current_index"]
        current_q = state["sub_questions"][idx]
        print(
            f"\n🤖 [LangGraph: Node - Answer Sub-question {idx + 1}/{len(state['sub_questions'])}]"
        )
        print(f'❓ Sub-question: "{current_q}"')

        # Compile previous QA context as memory
        previous_context = ""
        if state["sub_answers"]:
            previous_context = "\n".join(
                [
                    f"Sub-question: {qa['question']}\nAnswer: {qa['answer']}"
                    for qa in state["sub_answers"]
                ]
            )

        # Retrieve documents for the current sub-question
        search_query = current_q
        if state["sub_answers"]:
            # Contextualize query with key details from previous sub-answer
            search_query = f"{current_q} (Context: {state['sub_answers'][-1]['answer'][:120]})"

        print(f'🔍 Retrieving context for: "{search_query}"...')
        docs = retriever.invoke(search_query)

        # Answer the sub-question using LLM and context
        ans_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert technical assistant. Answer the current sub-question "
                    "based on the retrieved documents and the context of previously answered "
                    "sub-questions.\n\n"
                    "--- START PREVIOUS CONTEXT ---\n"
                    "{previous_context}\n"
                    "--- END PREVIOUS CONTEXT ---\n\n"
                    "Retrieved Documents:\n"
                    "{documents}",
                ),
                ("human", "Current Sub-question: {question}"),
            ]
        )

        # Format documents content
        docs_text = "\n\n".join(
            [
                f"Source: {doc.metadata.get('source', 'Unknown')}\nSnippet: {doc.page_content}"
                for doc in docs
            ]
        )

        ans_chain = ans_prompt | llm
        try:
            answer = ans_chain.invoke(
                {
                    "previous_context": previous_context or "None.",
                    "documents": docs_text or "No relevant documents found.",
                    "question": current_q,
                }
            ).content.strip()
            snippet = answer.replace("\n", " ")[:120]
            print(f'💡 Sub-answer {idx + 1}: "{snippet}..."')
        except Exception as e:
            print(f"[ERROR] Failed to answer sub-question: {e}", file=sys.stderr)
            answer = "Error generating answer for this sub-question."

        # Update state fields
        new_sub_answers = list(state["sub_answers"])
        new_sub_answers.append({"question": current_q, "answer": answer})

        new_docs = list(state["retrieved_docs"])
        new_docs.extend(docs)

        return {
            "sub_answers": new_sub_answers,
            "retrieved_docs": new_docs,
            "current_index": idx + 1,
        }

    def generate_final_answer(state: DecompositionState) -> Dict[str, Any]:
        print("\n🤖 [LangGraph: Node - Generate Final Answer]")

        # Compile all sub-answers
        sub_qa_text = "\n\n".join(
            [
                f"Sub-question: {qa['question']}\nAnswer: {qa['answer']}"
                for qa in state["sub_answers"]
            ]
        )

        final_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert technical assistant. Write a comprehensive "
                    "final answer to the user's main question based on the "
                    "resolved sub-questions and answers.\n\n"
                    "--- RESOLVED SUB-QUESTIONS & ANSWERS ---\n"
                    "{sub_qa_text}\n"
                    "--- END RESOLVED CONTEXT ---\n\n"
                    "Synthesize these answers into a cohesive, structured response that directly "
                    "answers the main question. Citations from source documents are already "
                    "integrated in the sub-answers context, make sure to preserve them.",
                ),
                ("human", "Main Question: {main_question}"),
            ]
        )

        final_chain = final_prompt | llm
        try:
            final_ans = final_chain.invoke(
                {"sub_qa_text": sub_qa_text, "main_question": state["main_question"]}
            ).content.strip()
        except Exception as e:
            print(f"[ERROR] Failed to generate final answer: {e}", file=sys.stderr)
            final_ans = "Error generating final synthesized answer."

        return {"final_answer": final_ans}

    # 3. Routing Condition
    def should_continue(state: DecompositionState) -> str:
        if state["current_index"] < len(state["sub_questions"]):
            return "answer_sub_question"
        return "generate_final_answer"

    # 4. Construct workflow
    workflow = StateGraph(DecompositionState)

    workflow.add_node("generate_sub_questions", generate_sub_questions)
    workflow.add_node("answer_sub_question", answer_sub_question)
    workflow.add_node("generate_final_answer", generate_final_answer)

    workflow.set_entry_point("generate_sub_questions")
    workflow.add_edge("generate_sub_questions", "answer_sub_question")

    workflow.add_conditional_edges(
        "answer_sub_question",
        should_continue,
        {
            "answer_sub_question": "answer_sub_question",
            "generate_final_answer": "generate_final_answer",
        },
    )
    workflow.add_edge("generate_final_answer", END)

    return workflow.compile()
