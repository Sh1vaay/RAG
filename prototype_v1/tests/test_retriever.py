from src.retriever import Retriever, ask, assemble_answer

_retriever = None


def get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


def test_retrieval_returns_results():
    chunks = get_retriever().retrieve("What does the CCTV AMC plan include?", k=3)
    assert len(chunks) > 0
    assert chunks[0].category == "CCTV"


def test_ftth_query_with_clear_keyword_retrieves_ftth_content():
    """Regression test for the original bug: a clearly-worded FTTH query
    must retrieve FTTH content, not electrical installation content."""
    chunks = get_retriever().retrieve(
        "How long does FTTH installation take?", k=3, use_category_filter=True
    )
    assert chunks[0].category == "FTTH"
    assert "fiber" in chunks[0].text.lower() or "ftth" in chunks[0].text.lower()


def test_category_filter_narrows_results_when_category_detected():
    get_retriever().retrieve("CCTV camera resolution options", k=5, use_category_filter=False)
    chunks_filtered = get_retriever().retrieve(
        "CCTV camera resolution options", k=5, use_category_filter=True
    )
    # filtered results should all be CCTV; unfiltered may include other categories
    assert all(c.category == "CCTV" for c in chunks_filtered)


def test_guardrail_declines_on_empty_retrieval():
    answer = assemble_answer("irrelevant query", [])
    assert answer.confident is False
    assert "don't have this information" in answer.text


def test_ask_end_to_end_returns_sourced_answer():
    answer = ask(
        "What is the standard BPO support desk operating hours?", retriever=get_retriever()
    )
    assert answer.confident is True
    assert len(answer.sources) > 0
    assert answer.text  # non-empty
