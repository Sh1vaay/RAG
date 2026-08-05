"""
Thin wrapper around ChromaDB matching the internship project's choice of a
free, self-hostable vector store. Stores precomputed embeddings (from
embeddings.py) alongside category metadata for filtered retrieval.
"""
import os
import shutil
from typing import List, Dict, Any
import chromadb


class VectorStore:
    def __init__(self, persist_dir: str, collection_name: str = "company_docs", reset: bool = False):
        if reset and os.path.exists(persist_dir):
            shutil.rmtree(persist_dir)
        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(collection_name)

    def add(self, ids: List[str], embeddings, documents: List[str], metadatas: List[Dict[str, Any]]):
        self.collection.add(
            ids=ids,
            embeddings=embeddings.tolist() if hasattr(embeddings, "tolist") else embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def query(self, query_embedding, n_results: int = 5, where: Dict[str, Any] = None):
        emb = query_embedding.tolist() if hasattr(query_embedding, "tolist") else query_embedding
        kwargs = dict(query_embeddings=[emb], n_results=n_results)
        if where:
            kwargs["where"] = where
        return self.collection.query(**kwargs)

    def count(self) -> int:
        return self.collection.count()
