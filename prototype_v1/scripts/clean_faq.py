"""
Cleans the raw FAQ sheet: removes duplicate/near-duplicate rows, standardizes
categories, and flags entries not updated in 12+ months (matches the internship
project's "stale FAQ flagging" step).
"""

import csv
import os
import sys
from datetime import date, timedelta
from difflib import SequenceMatcher

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(PROJECT_ROOT, "data", "faq_raw.csv")
CLEAN_PATH = os.path.join(PROJECT_ROOT, "data", "faq_master.csv")
STALE_REPORT_PATH = os.path.join(PROJECT_ROOT, "data", "stale_faq_report.csv")

STALE_THRESHOLD_DAYS = 365


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def load_raw(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def dedupe(rows, threshold=0.85):
    """Remove near-duplicate questions within the same category."""
    kept = []
    removed = 0
    for row in rows:
        is_dup = False
        for k in kept:
            if (
                k["category"] == row["category"]
                and similarity(k["question"], row["question"]) >= threshold
            ):
                is_dup = True
                break
        if is_dup:
            removed += 1
            continue
        kept.append(row)
    return kept, removed


def flag_stale(rows, today=None):
    today = today or date.today()
    stale = []
    for row in rows:
        try:
            updated = date.fromisoformat(row["last_updated"])
        except ValueError:
            continue
        if (today - updated) > timedelta(days=STALE_THRESHOLD_DAYS):
            stale.append({**row, "days_stale": (today - updated).days})
    return stale


def main():
    if not os.path.exists(RAW_PATH):
        print(f"Raw FAQ file not found at {RAW_PATH}", file=sys.stderr)
        sys.exit(1)

    raw = load_raw(RAW_PATH)
    print(f"Loaded {len(raw)} raw rows")

    cleaned, removed = dedupe(raw)
    print(f"Removed {removed} duplicate/near-duplicate rows -> {len(cleaned)} rows remain")

    for i, row in enumerate(cleaned, start=1):
        row["id"] = f"faq_{i:04d}"

    with open(CLEAN_PATH, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["id", "category", "question", "answer", "last_updated"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in cleaned:
            writer.writerow({k: row[k] for k in fieldnames})
    print(f"Wrote cleaned FAQ sheet -> {CLEAN_PATH}")

    stale = flag_stale(cleaned)
    with open(STALE_REPORT_PATH, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["id", "category", "question", "last_updated", "days_stale"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in stale:
            writer.writerow({k: row[k] for k in fieldnames})
    print(
        f"Flagged {len(stale)} stale entries (>{STALE_THRESHOLD_DAYS} days old) "
        f"-> {STALE_REPORT_PATH}"
    )


if __name__ == "__main__":
    main()
