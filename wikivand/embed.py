"""Diff embeddings. MiniLM via sentence-transformers if installed, else a hashing fallback.

The embedder name is stored with every model so serving uses the same one.
"""
from __future__ import annotations

import threading

import numpy as np


class HashingEmbedder:
    name = "hashing-char-2-4-384"
    dim = 384

    def __init__(self):
        from sklearn.feature_extraction.text import HashingVectorizer

        self._vec = HashingVectorizer(analyzer="char_wb", ngram_range=(2, 4), n_features=self.dim,
                                      alternate_sign=False, norm="l2")

    def encode(self, texts: list[str]) -> np.ndarray:
        return self._vec.transform(texts).toarray().astype(np.float32)


class MiniLMEmbedder:
    dim = 384

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self._model = SentenceTransformer(model_name, device="cpu")
        self._lock = threading.Lock()

    def encode(self, texts: list[str]) -> np.ndarray:
        with self._lock:
            out = self._model.encode(texts, batch_size=64, normalize_embeddings=True,
                                     show_progress_bar=False)
        out = np.asarray(out, dtype=np.float32)
        out[[not t for t in texts]] = 0.0  # no diff -> zero vector, same as training
        return out


def load_embedder(name: str | None):
    if name in (None, "", "none"):
        return None
    if name == HashingEmbedder.name:
        return HashingEmbedder()
    try:
        return MiniLMEmbedder(name)
    except ImportError:
        print("sentence-transformers not installed; using hashing embedder")
        return HashingEmbedder()
