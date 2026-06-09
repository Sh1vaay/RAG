"""
multi_rep_utils.py
──────────────────
Multi-Representation Indexing utilities.

Instead of embedding raw chunks (which are often noisy), we embed clean
one-line LLM summaries and store the original chunk text inside metadata.
At retrieval time we swap the summary back for the full original content
so the LLM still sees rich context when generating the answer.
"""

import sys
from typing import List
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


# ── Prompt used to summarise each chunk ──────────────────────────────────────
_SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a technical summariser. Write a single concise sentence "
        "(max 30 words) that captures the core idea of the following text. "
        "Do NOT start with 'This document' or 'This text'. "
        "Output the sentence only — no preamble, no punctuation at the end.",
    ),
    ("human", "{text}"),
])


def generate_summaries(splits: List[Document], llm: ChatOpenAI) -> List[Document]:
    """
    For every chunk in `splits`:
      1. Generate a one-line summary via the LLM (batched in one API call).
      2. Return a new Document whose page_content is the summary and whose
         metadata contains the original chunk text under 'original_content'.

    The returned list is what gets embedded and stored in the vector database.
    """
    if not splits:
        return []

    print(f"🔖 [Multi-Rep] Generating summaries for {len(splits)} chunks (batched)...")

    # Build one prompt message per chunk
    chain = _SUMMARY_PROMPT | llm
    inputs = [{"text": doc.page_content} for doc in splits]

    try:
        # .batch() fires all requests concurrently — much faster than a loop
        responses = chain.batch(inputs, config={"max_concurrency": 10})
    except Exception as exc:
        print(
            f"[WARNING] Multi-Rep summary batch failed: {exc}. "
            "Falling back to raw chunks.",
            file=sys.stderr,
        )
        return splits  # graceful fallback — use raw chunks as-is

    summary_docs: List[Document] = []
    for doc, response in zip(splits, responses):
        summary_text = response.content.strip()
        # Carry all original metadata forward; add original_content
        new_meta = dict(doc.metadata)
        new_meta["original_content"] = doc.page_content
        new_meta["indexed_as"] = "summary"
        summary_docs.append(Document(page_content=summary_text, metadata=new_meta))

    print(f"✅ [Multi-Rep] {len(summary_docs)} summary documents ready.")
    return summary_docs


def restore_original_content(docs: List[Document]) -> List[Document]:
    """
    After retrieval, swap the summary page_content back to the full original
    chunk so the answer-generation LLM receives rich context, not the short
    summary.

    Documents that were NOT indexed via Multi-Rep (no 'original_content' key)
    are returned unchanged, so this function is always safe to call.
    """
    restored: List[Document] = []
    for doc in docs:
        original = doc.metadata.get("original_content")
        if original:
            new_meta = {k: v for k, v in doc.metadata.items() if k != "original_content"}
            restored.append(Document(page_content=original, metadata=new_meta))
        else:
            restored.append(doc)
    return restored
