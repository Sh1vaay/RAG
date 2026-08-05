"""
Ingestion pipeline: reads the cleaned FAQ sheet and all source PDFs,
chunks them, embeds every chunk, and loads everything into the vector store
with category metadata attached (used later for filtered retrieval).

Run directly: `python -m src.ingest`
"""

import csv
import os
from typing import Any, Dict, List

from pypdf import PdfReader

from .chunking import recursive_split
from .embeddings import Embedder
from .vectorstore import VectorStore

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAQ_PATH = os.path.join(PROJECT_ROOT, "data", "faq_master.csv")
DOCS_DIR = os.path.join(PROJECT_ROOT, "data", "docs")
PERSIST_DIR = os.path.join(PROJECT_ROOT, "data", "chroma_store")

# Filename -> category, since our sample PDFs are named after their service line.
FILENAME_CATEGORY = {
    "cctv_amc_brochure.pdf": "CCTV",
    "ftth_installation_guide.pdf": "FTTH",
    "electrical_installation_guide.pdf": "Electrical",
    "wifi_solutions_brochure.pdf": "WiFi",
    "smart_city_case_study.pdf": "SmartCity",
    "finops_control_overview.pdf": "Cloud",
    "bpo_service_sop.pdf": "BPO",
    "datacenter_colocation_brochure.pdf": "DataCenter",
    "it_infrastructure_amc_guide.pdf": "ITInfra",
    "structured_cabling_certification_guide.pdf": "StructuredCabling",
    "solar_maintenance_guide.pdf": "Solar",
    "access_control_security_brochure.pdf": "AccessControl",
}


def load_faq_chunks() -> List[Dict[str, Any]]:
    """FAQ rows are treated as atomic chunks - one row, one chunk (no splitting)."""
    chunks = []
    if not os.path.exists(FAQ_PATH):
        return chunks
    with open(FAQ_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            text = f"Q: {row['question']}\nA: {row['answer']}"
            chunks.append(
                {
                    "id": f"faq-{row['id']}",
                    "text": text,
                    "category": row["category"],
                    "source": "faq_master.csv",
                }
            )
    return chunks


def extract_pdf_text(path: str) -> str:
    reader = PdfReader(path)
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def load_pdf_chunks(chunk_size: int = 800, overlap: int = 100) -> List[Dict[str, Any]]:
    chunks = []
    if not os.path.isdir(DOCS_DIR):
        return chunks
    for filename in sorted(os.listdir(DOCS_DIR)):
        if not filename.lower().endswith(".pdf"):
            continue
        path = os.path.join(DOCS_DIR, filename)
        text = extract_pdf_text(path)
        category = FILENAME_CATEGORY.get(filename, "General")
        pieces = recursive_split(text, chunk_size=chunk_size, overlap=overlap)
        for i, piece in enumerate(pieces):
            chunks.append(
                {
                    "id": f"pdf-{filename}-{i}",
                    "text": piece,
                    "category": category,
                    "source": filename,
                }
            )
    return chunks


def run_ingest(reset: bool = True) -> VectorStore:
    faq_chunks = load_faq_chunks()
    pdf_chunks = load_pdf_chunks()
    all_chunks = faq_chunks + pdf_chunks

    if not all_chunks:
        raise RuntimeError("No chunks found - run scripts/clean_faq.py and check data/docs/ first.")

    print(
        f"Loaded {len(faq_chunks)} FAQ chunks and {len(pdf_chunks)} PDF chunks "
        f"({len(all_chunks)} total)"
    )

    embedder = Embedder()
    embedder.fit([c["text"] for c in all_chunks])
    embeddings = embedder.embed([c["text"] for c in all_chunks])

    store = VectorStore(PERSIST_DIR, reset=reset)
    store.add(
        ids=[c["id"] for c in all_chunks],
        embeddings=embeddings,
        documents=[c["text"] for c in all_chunks],
        metadatas=[{"category": c["category"], "source": c["source"]} for c in all_chunks],
    )
    print(f"Indexed {store.count()} chunks into vector store at {PERSIST_DIR}")

    # Persist the fitted embedder so retrieval can embed new queries consistently
    import pickle

    with open(os.path.join(PERSIST_DIR, "embedder.pkl"), "wb") as f:
        pickle.dump(embedder, f)

    return store


if __name__ == "__main__":
    run_ingest()
