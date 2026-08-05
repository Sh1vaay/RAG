import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PERSIST_DIR = os.path.join(PROJECT_ROOT, "data", "chroma_store")
FAQ_MASTER = os.path.join(PROJECT_ROOT, "data", "faq_master.csv")


@pytest.fixture(scope="session", autouse=True)
def ensure_index_built():
    """Integration tests need a real, populated vector store. Build it once
    per test session if it doesn't already exist, rather than mocking it -
    this matches the project's real integration-test strategy of exercising
    the retriever against a small real ChromaDB collection."""
    if not os.path.exists(FAQ_MASTER):
        import subprocess
        subprocess.run([sys.executable, os.path.join(PROJECT_ROOT, "scripts", "clean_faq.py")], check=True)
    if not os.path.exists(PERSIST_DIR) or not os.listdir(PERSIST_DIR):
        from src.ingest import run_ingest
        run_ingest(reset=True)
    yield
