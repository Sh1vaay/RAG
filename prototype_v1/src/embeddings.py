"""
Embedding backend for the RAG pipeline.

In the real internship project, embeddings came from a hosted embedding API.
This local demo uses a TF-IDF vectorizer (scikit-learn) so the whole project
runs fully offline and deterministically - no API key or network access
required to reproduce it. The interface (`Embedder.fit`, `.embed`) is written
so swapping in a real embedding API later only means changing this one file.
"""
from typing import List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class Embedder:
    def __init__(self, max_features: int = 4096):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            stop_words="english",
        )
        self._fitted = False

    def fit(self, corpus: List[str]) -> None:
        self.vectorizer.fit(corpus)
        self._fitted = True

    def embed(self, texts: List[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Embedder must be fit() on a corpus before embed()")
        matrix = self.vectorizer.transform(texts)
        return matrix.toarray().astype("float32")

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]
