import os
import sys
import bs4
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# Load environment variables from .env file
load_dotenv()

def ingest_data():
    # 1. Validation
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_openai_api_key_here":
        print("[ERROR] OPENAI_API_KEY is not set or is still the default placeholder value in .env.", file=sys.stderr)
        print("Please set your OpenAI API key in the .env file before running ingest.py.", file=sys.stderr)
        sys.exit(1)

    print("Loading documents...")
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
        if not docs:
            raise ValueError(f"No content loaded from the URL: {target_url}")
    except Exception as e:
        print(f"[ERROR] Failed to fetch or parse document from {target_url}: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Splitting documents (Loaded {len(docs)} document(s))...")
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=300, 
        chunk_overlap=50
    )
    splits = text_splitter.split_documents(docs)
    print(f"Created {len(splits)} text chunks.")

    print("Embedding and saving to Chroma database...")
    try:
        # Explicitly set the embedding model to ensure stability and cost control
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
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
