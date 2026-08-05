"""
Evaluation harness - adapted for Aether AI.

This script evaluates the Aether AI pipeline against the two golden sets
from the internship project.

Faithfulness here = word-overlap between the retrieved chunk actually used
and the golden answer (a crude proxy for what an LLM judge would check).
"""

import json
import os
import sys
from typing import Dict

# Add current directory to path to import Aether AI components
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.main import setup_pipeline
from src.multi_rep_utils import restore_original_content

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
GOLDEN_V1 = os.path.join(PROJECT_ROOT, "eval", "golden_v1_flawed.json")
GOLDEN_V2 = os.path.join(PROJECT_ROOT, "eval", "golden_v2_fixed.json")


def word_overlap_faithfulness(retrieved_text: str, golden_answer: str) -> float:
    """Crude faithfulness proxy: what fraction of the golden answer's
    meaningful words are actually present in the retrieved chunk?"""
    stop = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "to",
        "of",
        "for",
        "and",
        "in",
        "on",
        "at",
        "with",
        "we",
        "our",
        "you",
        "your",
        "this",
        "that",
        "it",
        "be",
        "as",
    }
    golden_words = {w.strip(".,?!").lower() for w in golden_answer.split()} - stop
    retrieved_words = {w.strip(".,?!").lower() for w in retrieved_text.split()} - stop
    if not golden_words:
        return 1.0
    overlap = golden_words & retrieved_words
    return len(overlap) / len(golden_words)


def run_eval(golden_path: str, pipeline) -> Dict:
    with open(golden_path) as f:
        golden_set = json.load(f)

    results = []

    query_analyzer = pipeline["query_analyzer"]
    routing_retriever = pipeline["routing_retriever"]
    question_answer_chain = pipeline["question_answer_chain"]

    for item in golden_set:
        query = item["question"]
        golden_answer = item["golden_answer"]

        # 1. Analyze and Route (simulate Aether AI's heavy path)
        structured_query = query_analyzer.analyze(query)
        route, _ = routing_retriever.determine_route(query)

        if route == "simple":
            # Force standard retrieval for rigorous evaluation
            route = "standard"

        # 2. Retrieve Documents
        context_docs = routing_retriever.retrieve_for_route(structured_query.content_search, route)
        context_docs = restore_original_content(context_docs)

        # Combine retrieved context for the faithfulness check
        retrieved_text = " ".join([doc.page_content for doc in context_docs])

        # 3. Generate Answer
        answer = question_answer_chain.invoke(
            {
                "context": context_docs,
                "input": structured_query.content_search,
                "chat_history": [],
            }
        )

        # 4. Calculate Faithfulness
        faithfulness = word_overlap_faithfulness(retrieved_text, golden_answer)
        results.append(
            {
                "question": query,
                "golden_answer": golden_answer,
                "actual_answer": answer,
                "faithfulness": round(faithfulness, 3),
                "route_used": route,
            }
        )

    avg_faithfulness = sum(r["faithfulness"] for r in results) / len(results)
    return {"results": results, "avg_faithfulness": round(avg_faithfulness, 3)}


def main():
    print("Initializing Aether AI Pipeline for Evaluation...")
    try:
        pipeline = setup_pipeline()
    except Exception as e:
        print(f"Failed to load pipeline: {e}")
        print("Please make sure the dataset is in ./documents.")
        print("Then run: python -m src.ingest")
        return

    print("\n" + "=" * 70)
    print("EVAL RUN 1: golden_v1_flawed.json (loosely-worded reference answers)")
    print("=" * 70)
    report_v1 = run_eval(GOLDEN_V1, pipeline)
    for r in report_v1["results"]:
        flag = "OK" if r["faithfulness"] >= 0.5 else "LOW"
        print(
            "  [{flag}] faithfulness={faith:.2f} | Route: {route} | {q}".format(
                flag=flag, faith=r["faithfulness"], route=r["route_used"], q=r["question"]
            )
        )
    print(f"\n  AVERAGE FAITHFULNESS (v1): {report_v1['avg_faithfulness']:.2f}")

    print("\n" + "=" * 70)
    print("EVAL RUN 2: golden_v2_fixed.json (grounded strictly in source docs)")
    print("=" * 70)
    report_v2 = run_eval(GOLDEN_V2, pipeline)
    for r in report_v2["results"]:
        flag = "OK" if r["faithfulness"] >= 0.5 else "LOW"
        print(
            "  [{flag}] faithfulness={faith:.2f} | Route: {route} | {q}".format(
                flag=flag, faith=r["faithfulness"], route=r["route_used"], q=r["question"]
            )
        )
    print(f"\n  AVERAGE FAITHFULNESS (v2): {report_v2['avg_faithfulness']:.2f}")

    print("\n" + "=" * 70)
    print(
        f"RESULT: {report_v1['avg_faithfulness']:.2f} -> {report_v2['avg_faithfulness']:.2f} "
        f"after fixing the golden set (not the retrieval/guardrail logic)."
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
