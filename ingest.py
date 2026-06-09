import os
import sys
import glob
import re
import bs4
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFLoader,
    CSVLoader,
    Docx2txtLoader,
    TextLoader
)
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.documents import Document
from multi_rep_utils import generate_summaries

# Load environment variables from .env file
load_dotenv()

# ── RAPTOR: Cluster + summarise helper ────────────────────────────────────────
def build_raptor_layer(splits: list, embeddings: OpenAIEmbeddings, llm: ChatOpenAI) -> list:
    """
    Clusters the document chunks using Gaussian Mixture Models, then
    generates one LLM summary per cluster.  Returns a list of Documents
    tagged with metadata['layer'] = 'raptor_summary' so they can be
    distinguished from raw leaf chunks during retrieval.
    """
    try:
        import numpy as np
        from sklearn.mixture import GaussianMixture
        from langchain_core.prompts import ChatPromptTemplate
    except ImportError:
        print("[WARNING] scikit-learn not installed. Skipping RAPTOR. Run: uv add scikit-learn", file=sys.stderr)
        return []

    if len(splits) < 4:
        print("[WARNING] Too few chunks for RAPTOR clustering. Skipping.", file=sys.stderr)
        return []

    print(f"🌲 [RAPTOR] Embedding {len(splits)} chunks for clustering...")
    texts = [d.page_content for d in splits]
    vectors = embeddings.embed_documents(texts)
    vectors_np = np.array(vectors)

    # Choose number of clusters: sqrt heuristic, capped at 10
    n_clusters = min(max(2, int(len(splits) ** 0.5)), 10)
    print(f"🌲 [RAPTOR] Fitting {n_clusters} clusters...")
    gm = GaussianMixture(n_components=n_clusters, random_state=42)
    labels = gm.fit_predict(vectors_np)

    # Summarise each cluster
    cluster_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a technical summariser. Given a group of related text passages, "
         "write a concise 2-3 sentence thematic summary that captures the shared topic. "
         "Output only the summary, no preamble."),
        ("human", "{passages}")
    ])
    cluster_chain = cluster_prompt | llm

    raptor_docs = []
    for cluster_id in range(n_clusters):
        indices = [i for i, l in enumerate(labels) if l == cluster_id]
        if not indices:
            continue
        passages = "\n\n".join(splits[i].page_content[:400] for i in indices[:8])  # cap to avoid token limits
        try:
            summary = cluster_chain.invoke({"passages": passages}).content.strip()
        except Exception as exc:
            print(f"[WARNING] RAPTOR cluster {cluster_id} summary failed: {exc}", file=sys.stderr)
            continue
        raptor_docs.append(Document(
            page_content=summary,
            metadata={
                "layer": "raptor_summary",
                "cluster_id": cluster_id,
                "source": "raptor",
                "file_type": "raptor",
                "year": 0,
                "page": 0,
                "row": 0,
                "data_source": "internal_docs",
            }
        ))
    print(f"✅ [RAPTOR] Created {len(raptor_docs)} cluster summary documents.")
    return raptor_docs

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

def ingest_data():
    # 1. Validation
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_openai_api_key_here":
        print("[ERROR] OPENAI_API_KEY is not set or is still the default placeholder value in .env.", file=sys.stderr)
        print("Please set your OpenAI API key in the .env file before running ingest.py.", file=sys.stderr)
        sys.exit(1)

    documents_dir = "./documents"
    if not os.path.exists(documents_dir):
        os.makedirs(documents_dir)
        print(f"Created directory: {documents_dir}")

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
            print(f"[ERROR] Failed to fetch or parse web document from {target_url}: {e}", file=sys.stderr)
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
        year_match = re.search(r'\b(19\d\d|20\d\d)\b', source)
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
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        text_splitter = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")
        splits = text_splitter.split_documents(docs)
        print(f"Created {len(splits)} semantic text chunks.")
    except Exception as e:
        print(f"[ERROR] Failed to split documents semantically: {e}", file=sys.stderr)
        sys.exit(1)

    # ── Multi-Representation Indexing (always on) ──────────────────────────
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    final_docs = generate_summaries(splits, llm)
    if not final_docs:
        # Graceful fallback: use raw splits if summary generation failed
        print("[WARNING] Multi-Rep summaries empty, falling back to raw chunks.")
        final_docs = splits

    # ── RAPTOR layer (opt-in via --raptor flag) ────────────────────────────
    use_raptor = "--raptor" in sys.argv
    if use_raptor:
        print("🌲 [RAPTOR] Building cluster summary tree (--raptor flag detected)...")
        raptor_docs = build_raptor_layer(splits, embeddings, llm)
        final_docs = final_docs + raptor_docs  # merge leaf summaries + cluster summaries
        print(f"📦 Total documents to embed: {len(final_docs)} (leaves + RAPTOR summaries)")

    print("Embedding and saving to Chroma database...")
    try:
        Chroma.from_documents(
            documents=final_docs,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )
        print("Ingestion complete. Database successfully saved to ./chroma_db")
    except Exception as e:
        print(f"[ERROR] Failed to generate embeddings or write to Chroma database: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    ingest_data()
