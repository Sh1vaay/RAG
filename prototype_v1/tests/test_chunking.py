from src.chunking import recursive_split


def test_short_text_returns_single_chunk():
    text = "This is a short sentence."
    chunks = recursive_split(text, chunk_size=800, overlap=100)
    assert chunks == [text]


def test_empty_text_returns_no_chunks():
    assert recursive_split("", chunk_size=800, overlap=100) == []


def test_long_text_splits_into_multiple_chunks():
    paragraph = "This is a sentence that repeats several times to build up length. " * 40
    chunks = recursive_split(paragraph, chunk_size=200, overlap=20)
    assert len(chunks) > 1
    for c in chunks:
        # allow a little slack for overlap being appended
        assert len(c) <= 200 + 40


def test_no_chunk_is_empty():
    text = "Paragraph one.\n\nParagraph two.\n\nParagraph three is a bit longer than the others."
    chunks = recursive_split(text, chunk_size=30, overlap=5)
    assert all(c.strip() for c in chunks)


def test_overlap_shares_content_between_consecutive_chunks():
    paragraph = "Alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima. " * 10
    chunks = recursive_split(paragraph, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    # the tail of chunk[i] should reappear at the start of chunk[i+1]
    for i in range(1, len(chunks)):
        tail = chunks[i - 1][-20:]
        assert tail[:10] in chunks[i]
