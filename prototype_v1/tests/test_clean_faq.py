import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from datetime import date

from clean_faq import dedupe, flag_stale, similarity


def test_similarity_identical_strings_is_one():
    assert similarity("hello world", "hello world") == 1.0


def test_similarity_different_strings_is_low():
    assert similarity("hello world", "completely different text") < 0.5


def test_dedupe_removes_near_duplicate_within_same_category():
    rows = [
        {"category": "CCTV", "question": "What does the AMC plan include?", "answer": "a"},
        {"category": "CCTV", "question": "What does the AMC plan include?!", "answer": "b"},
        {"category": "FTTH", "question": "What does the AMC plan include?", "answer": "c"},
    ]
    kept, removed = dedupe(rows, threshold=0.85)
    assert removed == 1
    # the FTTH row survives even though the question text is identical,
    # because it's a different category (not a real duplicate)
    assert len(kept) == 2
    assert any(r["category"] == "FTTH" for r in kept)


def test_dedupe_keeps_distinct_questions():
    rows = [
        {"category": "CCTV", "question": "What does the AMC plan include?", "answer": "a"},
        {"category": "CCTV", "question": "How fast is the outage response time?", "answer": "b"},
    ]
    kept, removed = dedupe(rows)
    assert removed == 0
    assert len(kept) == 2


def test_flag_stale_flags_old_entries():
    rows = [
        {"category": "X", "question": "q1", "last_updated": "2020-01-01"},
        {"category": "X", "question": "q2", "last_updated": date.today().isoformat()},
    ]
    stale = flag_stale(rows, today=date.today())
    assert len(stale) == 1
    assert stale[0]["question"] == "q1"


def test_flag_stale_handles_bad_dates_gracefully():
    rows = [{"category": "X", "question": "q1", "last_updated": "not-a-date"}]
    # should not raise
    stale = flag_stale(rows)
    assert stale == []
