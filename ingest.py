import os
import sys
import glob
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
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# Load environment variables from .env file
load_dotenv()

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
    import re
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

    print("Embedding and saving to Chroma database...")
    try:
        # Initialize Chroma with a persist directory to save data locally
        Chroma.from_documents(
            documents=splits, 
            embedding=embeddings,
            persist_directory="./chroma_db"
        )
        print("Ingestion complete. Database successfully saved to ./chroma_db")
    except Exception as e:
        print(f"[ERROR] Failed to generate embeddings or write to Chroma database: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    ingest_data()
