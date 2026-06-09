import os
import sys

import numpy as np
import tiktoken
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

# Load API keys
load_dotenv()


def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def cosine_similarity(vec1, vec2):
    """Calculates the cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)


if __name__ == "__main__":
    # Validate API key exists
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_openai_api_key_here":
        print(
            "[ERROR] OPENAI_API_KEY is not set or is still the default placeholder value in .env.",
            file=sys.stderr,
        )
        print(
            "Please configure your OpenAI API key in the .env file before running playground.py.",
            file=sys.stderr,
        )
        sys.exit(1)

    question = "What kinds of pets do I like?"
    document = "My favorite pet is a cat."

    # 1. Count Tokens
    tokens = num_tokens_from_string(question)
    print(f"Tokens in question: {tokens}")

    # 2. Generate Embeddings
    print("Generating embeddings...")
    try:
        # Use text-embedding-3-small to align with ingest.py and main.py
        embd = OpenAIEmbeddings(model="text-embedding-3-small")
        query_result = embd.embed_query(question)
        document_result = embd.embed_query(document)

        print(f"Length of embedding vector: {len(query_result)}")

        # 3. Calculate Similarity
        similarity = cosine_similarity(query_result, document_result)
        print(f"Cosine Similarity: {similarity:.4f}")
    except Exception as e:
        print(f"[ERROR] Failed to calculate embedding similarity: {e}", file=sys.stderr)
        sys.exit(1)
