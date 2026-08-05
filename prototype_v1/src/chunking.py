"""
A lightweight recursive character text splitter (mirrors what a LangChain
RecursiveCharacterTextSplitter does, without the extra dependency).
Splits on paragraph breaks first, then sentences, then words, until each
chunk is under `chunk_size` characters, with `overlap` characters shared
between consecutive chunks to avoid losing context at chunk boundaries.
"""

from typing import List

DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " "]


def _split_on_separator(text: str, separator: str) -> List[str]:
    if separator == "":
        return list(text)
    return text.split(separator)


def _split_no_overlap(text: str, chunk_size: int, separators: List[str]) -> List[str]:
    """Recursively splits text into raw chunks with NO overlap applied.
    Overlap is applied exactly once, by recursive_split(), after this
    returns - applying it at every recursion level was the original bug
    (overlap text compounding each time an oversized part got re-split)."""
    text = text.strip()
    if len(text) <= chunk_size:
        return [text] if text else []

    separator = separators[0] if separators else ""
    remaining_separators = separators[1:] if len(separators) > 1 else []

    parts = _split_on_separator(text, separator)
    chunks: List[str] = []
    current = ""

    for part in parts:
        candidate = (current + separator + part) if current else part
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current.strip())
            if len(part) > chunk_size:
                chunks.extend(_split_no_overlap(part, chunk_size, remaining_separators))
                current = ""
            else:
                current = part

    if current:
        chunks.append(current.strip())

    return [c for c in chunks if c]


def recursive_split(
    text: str, chunk_size: int = 800, overlap: int = 100, separators: List[str] = None
) -> List[str]:
    separators = separators if separators is not None else DEFAULT_SEPARATORS
    chunks = _split_no_overlap(text, chunk_size, separators)

    if overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tail = chunks[i - 1][-overlap:]
            overlapped.append((prev_tail + " " + chunks[i]).strip())
        chunks = overlapped

    return chunks
