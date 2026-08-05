"""
Pluggable LLM client for generating the final answer from retrieved context.

- If ANTHROPIC_API_KEY is set, calls the real Claude API with a strict
  guardrail system prompt ("only answer from the provided context").
- Otherwise, falls back to the extractive method in retriever.assemble_answer()
  so the whole project remains runnable and testable without any API key -
  this is the mode used by the test suite and the offline demo.
"""

import os
from typing import List

from .retriever import RetrievedChunk

SYSTEM_PROMPT = (
    "You are an internal support assistant. Only answer using the provided "
    "context below. If the answer is not clearly supported by the context, "
    'say: "I don\'t have this information - please check with your supervisor." '
    "Keep answers to 2-3 sentences. Never invent details not present in the context."
)


def _build_context_block(chunks: List[RetrievedChunk]) -> str:
    return "\n\n---\n\n".join(f"[Source: {c.source}]\n{c.text}" for c in chunks)


def generate_answer(query: str, chunks: List[RetrievedChunk]) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        # Offline fallback: identical guardrail logic, just extractive instead
        # of LLM-generated. See retriever.assemble_answer for the shared logic.
        from .retriever import assemble_answer

        return assemble_answer(query, chunks).text

    try:
        import anthropic
    except ImportError:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is set but the 'anthropic' package isn't installed. "
            "Run: pip install anthropic"
        )

    client = anthropic.Anthropic(api_key=api_key)
    context_block = _build_context_block(chunks)
    message = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=300,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Context:\n{context_block}\n\nQuestion: {query}",
            }
        ],
    )
    return message.content[0].text
