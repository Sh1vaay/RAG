import argparse
import glob
import os
import re
import sys

import bs4
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
    WebBaseLoader,
)
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_experimental.text_splitter import SemanticChunker

from .multi_rep_utils import generate_summaries
from .providers import ConfigError, ProviderConfig, get_embeddings, get_llm
from .user_config import load_user_config
from .workspace import InvalidUserIdError, Workspace

# Load environment variables from .env file
load_dotenv()


# ── RAPTOR: Cluster + summarise helper ────────────────────────────────────────
def build_raptor_layer(splits: list, embeddings: Embeddings, llm: BaseChatModel) -> list:
    """
    Recursively builds a tree of cluster summaries following the RAPTOR paper:
    Level 1: Clusters raw chunks -> Generates Cluster Summaries.
    Level 2: Clusters the Level 1 Summaries -> Generates Root Summaries.
    This process repeats until the number of nodes is too small to cluster.
    All levels of summaries are returned to be indexed alongside raw chunks.
    """
    try:
        import numpy as np
        from langchain_core.prompts import ChatPromptTemplate
        from sklearn.mixture import GaussianMixture
    except ImportError:
        print(
            "[WARNING] scikit-learn not installed. Skipping RAPTOR. Run: uv sync", file=sys.stderr
        )
        return []

    all_summaries = []
    current_docs = splits
    level = 1
    max_levels = 3

    while level <= max_levels:
        if len(current_docs) < 4:
            print(
                f"🌲 [RAPTOR] Level {level}: Too few nodes ({len(current_docs)}) "
                "to cluster. Stopping tree growth."
            )
            break

        print(f"🌲 [RAPTOR] Level {level}: Embedding {len(current_docs)} nodes for clustering...")
        texts = [d.page_content for d in current_docs]
        try:
            vectors = embeddings.embed_documents(texts)
            vectors_np = np.array(vectors)
        except Exception as exc:
            print(f"[ERROR] RAPTOR embedding extraction failed: {exc}", file=sys.stderr)
            break

        # Choose components: square root heuristic
        n_clusters = min(max(2, int(len(current_docs) ** 0.5)), 10)
        if n_clusters >= len(current_docs):
            n_clusters = max(1, len(current_docs) - 1)
            if n_clusters < 2:
                break

        print(f"🌲 [RAPTOR] Level {level}: Fitting {n_clusters} clusters...")
        try:
            gm = GaussianMixture(n_components=n_clusters, random_state=42)
            labels = gm.fit_predict(vectors_np)
        except Exception as exc:
            print(f"[ERROR] RAPTOR GMM fitting failed: {exc}", file=sys.stderr)
            break

        cluster_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a technical summariser. Given a group of related text passages, "
                    "write a concise 2-3 sentence thematic summary that captures the shared topic. "
                    "Output only the summary, no preamble.",
                ),
                ("human", "{passages}"),
            ]
        )
        cluster_chain = cluster_prompt | llm

        level_summaries = []
        for cluster_id in range(n_clusters):
            indices = [i for i, lbl in enumerate(labels) if lbl == cluster_id]
            if not indices:
                continue
            passages = "\n\n".join(current_docs[i].page_content[:400] for i in indices[:8])
            try:
                summary = cluster_chain.invoke({"passages": passages}).text.strip()
            except Exception as exc:
                print(
                    f"[WARNING] RAPTOR Level {level} cluster {cluster_id} summary failed: {exc}",
                    file=sys.stderr,
                )
                continue

            level_summaries.append(
                Document(
                    page_content=summary,
                    metadata={
                        "layer": f"raptor_summary_level_{level}",
                        "cluster_id": cluster_id,
                        "source": "raptor",
                        "file_type": "raptor",
                        "year": 0,
                        "page": 0,
                        "row": 0,
                        "data_source": "internal_docs",
                    },
                )
            )

        print(f"✅ [RAPTOR] Level {level} created {len(level_summaries)} summaries.")
        all_summaries.extend(level_summaries)

        # Prepare next level to cluster these summaries
        current_docs = level_summaries
        level += 1

    return all_summaries


def load_single_document(file_path: str):
    """Loads a single document based on its file extension using LangChain loaders."""
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".csv":
        loader = CSVLoader(file_path)
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
    elif ext == ".txt" or ext == ".md":
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        print(f"[WARNING] Unsupported file type: {file_path}. Skipping.")
        return []

    try:
        return loader.load()
    except Exception as e:
        print(f"[ERROR] Failed to load {file_path}: {e}", file=sys.stderr)
        return []


def ingest_data(
    workspace: Workspace | None = None,
    use_raptor: bool = False,
    config: ProviderConfig | None = None,
):
    """Build the vector index for one workspace.

    `workspace` scopes which user's documents are read and where the index is
    written; `config` decides which models do the work. When run as a subprocess
    the config is read from the workspace itself, so no credential ever travels
    through argv (world-readable in a process listing) or an inherited env var.
    """
    ws = (workspace or Workspace.for_user(None)).ensure()
    cfg = config or load_user_config(ws)

    # 1. Validation — surface a bad provider selection before doing any work.
    try:
        cfg.validate()
        print(f"🔌 [Providers] {cfg.describe()}")
    except ConfigError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    documents_dir = str(ws.documents_dir)
    print(f"📁 [Workspace] {ws.user_id} → {documents_dir}")

    # Find all matching files in documents_dir
    files = []
    for ext in ("*.pdf", "*.csv", "*.docx", "*.txt", "*.md"):
        files.extend(glob.glob(os.path.join(documents_dir, ext)))
        files.extend(glob.glob(os.path.join(documents_dir, ext.upper())))

    # Remove duplicates from glob casing match
    files = sorted(list(set(files)))

    docs = []
    if files:
        print(f"Found {len(files)} document(s) in '{documents_dir}'. Loading...")
        for file_path in files:
            print(f"  Loading {os.path.basename(file_path)}...")
            docs.extend(load_single_document(file_path))
    else:
        # Fallback to loading the Weng blog post
        print(f"No documents found in '{documents_dir}'.")
        print("Falling back to loading default web resource...")
        target_url = "https://lilianweng.github.io/posts/2023-06-23-agent/"
        try:
            loader = WebBaseLoader(
                web_paths=(target_url,),
                bs_kwargs=dict(
                    parse_only=bs4.SoupStrainer(
                        class_=("post-content", "post-title", "post-header")
                    )
                ),
            )
            docs = loader.load()
        except Exception as e:
            print(
                f"[ERROR] Failed to fetch or parse web document from {target_url}: {e}",
                file=sys.stderr,
            )
            sys.exit(1)

    if not docs:
        print("[ERROR] No documents loaded. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Enrich document metadata with structured fields before splitting
    print("Enriching document metadata with file_type, year, page, and row info...")
    for doc in docs:
        source = doc.metadata.get("source", "")
        # 1. Determine file_type
        if source.startswith("http"):
            doc.metadata["file_type"] = "web"
        else:
            ext = os.path.splitext(source)[-1].lower()
            if ext == ".pdf":
                doc.metadata["file_type"] = "pdf"
            elif ext == ".csv":
                doc.metadata["file_type"] = "csv"
            elif ext == ".docx":
                doc.metadata["file_type"] = "docx"
            elif ext == ".txt" or ext == ".md":
                doc.metadata["file_type"] = "txt"
            else:
                doc.metadata["file_type"] = "unknown"

        # 2. Extract publication or creation year
        year_match = re.search(r"\b(19\d\d|20\d\d)\b", source)
        if year_match:
            doc.metadata["year"] = int(year_match.group(1))
        else:
            doc.metadata["year"] = 0

        # 3. Ensure page/row standard types (convert page to 1-indexed)
        if "page" in doc.metadata:
            try:
                doc.metadata["page"] = int(doc.metadata["page"]) + 1
            except Exception:
                pass
        else:
            doc.metadata["page"] = 0

        if "row" in doc.metadata:
            try:
                doc.metadata["row"] = int(doc.metadata["row"])
            except Exception:
                pass
        else:
            doc.metadata["row"] = 0

        # 4. Determine data_source corpus categorization
        if source.startswith("http"):
            doc.metadata["data_source"] = "web_blogs"
        elif any(kw in source.lower() for kw in ("paper", "arxiv", "academic", "research")):
            doc.metadata["data_source"] = "academic_papers"
        else:
            doc.metadata["data_source"] = "internal_docs"

    print(f"Splitting documents (Loaded {len(docs)} pages/documents) using Semantic Chunker...")
    try:
        # Explicitly set the embedding model to ensure stability and calculate similarity thresholds
        embeddings = get_embeddings(cfg)
        text_splitter = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")
        splits = text_splitter.split_documents(docs)
        print(f"Created {len(splits)} semantic text chunks.")
    except Exception as e:
        print(f"[ERROR] Failed to split documents semantically: {e}", file=sys.stderr)
        sys.exit(1)

    # ── Multi-Representation Indexing (always on) ──────────────────────────
    llm = get_llm(cfg)
    final_docs = generate_summaries(splits, llm)
    if not final_docs:
        # Graceful fallback: use raw splits if summary generation failed
        print("[WARNING] Multi-Rep summaries empty, falling back to raw chunks.")
        final_docs = splits

    # ── RAPTOR layer (opt-in) ──────────────────────────────────────────────
    if use_raptor:
        print("🌲 [RAPTOR] Building cluster summary tree (--raptor flag detected)...")
        raptor_docs = build_raptor_layer(splits, embeddings, llm)
        final_docs = final_docs + raptor_docs  # merge leaf summaries + cluster summaries
        print(f"📦 Total documents to embed: {len(final_docs)} (leaves + RAPTOR summaries)")

    print("Embedding and saving to FAISS database...")
    try:
        db = FAISS.from_documents(documents=final_docs, embedding=embeddings)
        db.save_local(str(ws.index_dir))
        print(f"Ingestion complete. Database successfully saved to {ws.index_dir}")
    except Exception as e:
        print(
            f"[ERROR] Failed to generate embeddings or write to FAISS database: {e}",
            file=sys.stderr,
        )
        sys.exit(1)


def main() -> None:
    """CLI entrypoint. The API invokes this as a subprocess, passing the *verified*
    user id — never a value taken straight from a request body."""
    parser = argparse.ArgumentParser(description="Build the vector index for a workspace.")
    parser.add_argument(
        "--user",
        default=None,
        help="Workspace/user id to ingest for. Defaults to the shared local workspace.",
    )
    parser.add_argument(
        "--raptor",
        action="store_true",
        help="Also build the RAPTOR hierarchical cluster-summary tree.",
    )
    args = parser.parse_args()

    try:
        workspace = Workspace.for_user(args.user)
    except InvalidUserIdError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(2)

    ingest_data(workspace=workspace, use_raptor=args.raptor)


if __name__ == "__main__":
    main()
