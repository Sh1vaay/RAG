"""
Retrieval + guardrail layer.

`retrieve()` demonstrates both the bug and the fix from the project write-up:
- `use_category_filter=False` reproduces the original bug (pure semantic
  similarity confuses FTTH and Electrical installation content).
- `use_category_filter=True` (the default / fixed behavior) detects the
  likely category from the query first and filters retrieval to it.

`answer()` assembles a guardrailed response: it only uses retrieved context,
and explicitly declines when confidence is too low, instead of guessing.
"""

import os
import pickle
from dataclasses import dataclass
from typing import List, Optional

from .category_detector import detect_category
from .vectorstore import VectorStore

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PERSIST_DIR = os.path.join(PROJECT_ROOT, "data", "chroma_store")

LOW_CONFIDENCE_THRESHOLD = 0.15  # similarity below this triggers the fallback


@dataclass
class RetrievedChunk:
    text: str
    category: str
    source: str
    score: float  # similarity score, higher = more relevant


@dataclass
class Answer:
    text: str
    sources: List[str]
    confident: bool
    chunks_used: List[RetrievedChunk]


class Retriever:
    def __init__(self, persist_dir: str = PERSIST_DIR):
        self.store = VectorStore(persist_dir)
        with open(os.path.join(persist_dir, "embedder.pkl"), "rb") as f:
            self.embedder = pickle.load(f)

    def retrieve(
        self, query: str, k: int = 5, use_category_filter: bool = True
    ) -> List[RetrievedChunk]:
        query_embedding = self.embedder.embed_one(query)

        where = None
        if use_category_filter:
            category = detect_category(query)
            if category:
                where = {"category": category}

        result = self.store.query(query_embedding, n_results=k, where=where)

        chunks = []
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        for doc, meta, dist in zip(docs, metas, distances):
            # ChromaDB returns squared-L2 distance for these embeddings; convert
            # to a rough 0-1 "similarity" score for readability (not a true cosine sim).
            score = 1.0 / (1.0 + dist)
            chunks.append(
                RetrievedChunk(
                    text=doc,
                    category=meta.get("category", ""),
                    source=meta.get("source", ""),
                    score=score,
                )
            )
        return chunks


def assemble_answer(query: str, chunks: List[RetrievedChunk]) -> Answer:
    """
    Guardrailed answer assembly. This is a template-based stand-in for an
    LLM call (see llm_client.py for the pluggable real-LLM path) so the
    whole pipeline runs without an API key, while still enforcing the same
    guardrail: never answer beyond what was actually retrieved.
    """
    if not chunks or chunks[0].score < LOW_CONFIDENCE_THRESHOLD:
        return Answer(
            text="I don't have this information - please check with your supervisor.",
            sources=[],
            confident=False,
            chunks_used=[],
        )

    top = chunks[0]
    # Extract the "A:" portion if this came from an FAQ chunk; otherwise use the raw text.
    text = top.text
    if "\nA: " in text:
        text = text.split("\nA: ", 1)[1]

    return Answer(
        text=text.strip(),
        sources=[f"{c.source}" for c in chunks[:2]],
        confident=True,
        chunks_used=chunks,
    )


def ask(
    query: str, k: int = 5, use_category_filter: bool = True, retriever: Optional[Retriever] = None
) -> Answer:
    retriever = retriever or Retriever()
    chunks = retriever.retrieve(query, k=k, use_category_filter=use_category_filter)
    return assemble_answer(query, chunks)
