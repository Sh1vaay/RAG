#!/usr/bin/env python3
"""
evaluate.py
───────────
Automated RAG Evaluation Suite using RAGAS.
Calculates Context Precision, Context Recall, Answer Relevance, and Faithfulness.
"""

import os
import sys
from dotenv import load_dotenv
import pandas as pd
from datasets import Dataset

# Ragas imports
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevance,
    context_precision,
    context_recall,
)

# Load pipeline component setup from main.py
from main import setup_pipeline, post_filter_documents
from multi_rep_utils import restore_original_content


# ── Evaluation dataset ────────────────────────────────────────────────────────

EVAL_DATASET = [
    {
        "question": "What are the key components of an LLM agent?",
        "ground_truth": "The key components of an LLM-based agent system are planning (including task decomposition and self-reflection), memory (short-term and long-term), and tool use."
    },
    {
        "question": "Explain task decomposition in the context of LLM agents.",
        "ground_truth": "Task decomposition involves breaking down complex tasks into smaller, more manageable sub-tasks. This can be achieved through LLM prompting like Chain-of-Thought, task-specific instructions, or human inputs."
    },
    {
        "question": "How does the ReAct framework integrate planning and tool use?",
        "ground_truth": "The ReAct framework integrates action execution and reasoning generation alternately. It allows the model to generate reasoning thoughts and execute actions (like calling search or calculator tools) and receive feedback to plan the next steps."
    },
    {
        "question": "What is the difference between short-term and long-term memory in LLM agents?",
        "ground_truth": "Short-term memory is managed via chat history or in-context learning constraints, while long-term memory is managed via external vector databases for fast retrieval of historical information over time."
    },
    {
        "question": "What are some common limitations of LLM agents?",
        "ground_truth": "Common limitations include finite context window constraints, difficulty in planning over long-horizon tasks, and the reliability of natural language interfaces when executing tool calls."
    }
]


def run_pipeline_query(pipeline: dict, query: str) -> dict:
    """Runs a single standalone query through the exact RAG system routing/generation steps."""
    llm = pipeline["llm"]
    vector_retriever = pipeline["vector_retriever"]
    compression_retriever = pipeline["compression_retriever"]
    query_analyzer = pipeline["query_analyzer"]
    routing_retriever = pipeline["routing_retriever"]
    question_answer_chain = pipeline["question_answer_chain"]
    fast_rag_chain = pipeline["fast_rag_chain"]

    # Evaluated queries are standalone (no previous chat history)
    chat_history = []

    # 1. Determine Route
    route, _ = routing_retriever.determine_route(query)

    if route == "simple":
        fast_result = fast_rag_chain.invoke({
            "input": query,
            "chat_history": chat_history
        })
        answer = fast_result["answer"]
        context_docs = fast_result.get("context", [])
    else:
        # 2. Proceed with Heavy Pipeline
        # Extract query filters/constraints
        structured_query = query_analyzer.analyze(query)
        
        # Build database filter dictionary (if any)
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
        
        # 3. Execute Route
        if route == "decomposition":
            from decomposition_graph import create_decomposition_graph
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
            from agentic_graph import create_agentic_graph
            agentic = create_agentic_graph(compression_retriever, llm)
            agentic_state = agentic.invoke({
                "question": structured_query.content_search,
                "rewritten_question": "",
                "retrieved_docs": [],
                "relevant_docs": [],
                "answer": "",
                "reflection_passed": False,
                "retry_count": 0,
            })
            answer = agentic_state["answer"]
            context_docs = restore_original_content(
                agentic_state["relevant_docs"] or agentic_state["retrieved_docs"]
            )
        else:
            # Retrieve documents using chosen translation strategy
            context_docs = routing_retriever.retrieve_for_route(structured_query.content_search, route)
            # Apply final post-filtering layer
            context_docs = post_filter_documents(context_docs, structured_query)
            # Restore original content
            context_docs = restore_original_content(context_docs)
            # Generate final answer using context
            answer = question_answer_chain.invoke({
                "context": context_docs,
                "input": structured_query.content_search,
                "chat_history": chat_history
            })

    # Return answer along with raw page contents of context documents
    return {
        "answer": answer,
        "contexts": [doc.page_content for doc in context_docs]
    }


def main():
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[ERROR] OPENAI_API_KEY environment variable is missing. Cannot run RAGAS evaluation.", file=sys.stderr)
        sys.exit(1)

    print("🔧 Setting up RAG pipeline components...")
    try:
        pipeline = setup_pipeline()
    except Exception as exc:
        print(f"[ERROR] Failed to setup RAG pipeline: {exc}", file=sys.stderr)
        sys.exit(1)

    print("\n🚀 Executing evaluation queries on RAG pipeline...")
    answers = []
    contexts = []

    for idx, item in enumerate(EVAL_DATASET, 1):
        q = item["question"]
        print(f"[{idx}/{len(EVAL_DATASET)}] Query: \"{q}\"")
        try:
            res = run_pipeline_query(pipeline, q)
            answers.append(res["answer"])
            contexts.append(res["contexts"])
        except Exception as exc:
            print(f"[WARNING] Query execution failed: {exc}", file=sys.stderr)
            answers.append("Execution failed.")
            contexts.append([])

    print("\n📊 Preparing dataset for RAGAS evaluation...")
    data_dict = {
        "question": [item["question"] for item in EVAL_DATASET],
        "answer": answers,
        "contexts": contexts,
        "ground_truth": [item["ground_truth"] for item in EVAL_DATASET]
    }
    
    # Ragas evaluation requires a HF Dataset
    dataset = Dataset.from_dict(data_dict)

    print("⚖️  Evaluating metrics (Faithfulness, Answer Relevance, Context Precision, Context Recall)...")
    try:
        result = evaluate(
            dataset=dataset,
            metrics=[
                faithfulness,
                answer_relevance,
                context_precision,
                context_recall
            ]
        )
    except Exception as exc:
        print(f"[ERROR] RAGAS evaluation failed: {exc}", file=sys.stderr)
        sys.exit(1)

    # Convert results to DataFrame for nice visualization and saving
    results_df = result.to_pandas()
    
    print("\n=======================================================")
    print("📈 RAGAS EVALUATION METRICS REPORT")
    print("=======================================================")
    print(results_df[["question", "faithfulness", "answer_relevance", "context_precision", "context_recall"]].to_string(index=False))
    print("-------------------------------------------------------")
    print("AVERAGE SCORES:")
    for metric, score in result.items():
        print(f"  - {metric.capitalize()}: {score:.4f}")
    print("=======================================================")

    report_path = "evaluation_report.csv"
    results_df.to_csv(report_path, index=False)
    print(f"💾 Detailed report saved to: {report_path}\n")


if __name__ == "__main__":
    main()
