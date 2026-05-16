"""
Embedder wrapper around sentence-transformers.

Model: all-MiniLM-L6-v2
  - 22MB on disk (fits <50MB budget)
  - 384-dim embeddings
  - ~30ms per sentence on CPU

TF-IDF fallback exists for environments where sentence-transformers
cannot be installed. Always prefer transformer embeddings in production.
"""

from __future__ import annotations
import numpy as np
from typing import List


class Embedder:
    def __init__(self, use_transformer: bool = True):
        self._model = None
        self._tfidf = None
        self._use_transformer = use_transformer
        self._mode = None
        self._tfidf_fitted = False

    def _load(self):
        if self._model is not None or self._tfidf is not None:
            return

        if self._use_transformer:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer("all-MiniLM-L6-v2")
                self._mode = "transformer"
                print("[embedder] Loaded all-MiniLM-L6-v2")
                return
            except ImportError:
                print("[embedder] sentence-transformers not available, falling back to TF-IDF")

        from sklearn.feature_extraction.text import TfidfVectorizer
        self._tfidf = TfidfVectorizer(max_features=512, ngram_range=(1, 2))
        self._mode = "tfidf"
        print("[embedder] Using TF-IDF fallback (lower quality drift scores)")

    def embed(self, texts: List[str]) -> np.ndarray:
        """Returns (N, D) float32 array."""
        self._load()
        texts = [t.strip() for t in texts if t.strip()]
        if not texts:
            raise ValueError("No non-empty texts provided")

        if self._mode == "transformer":
            vecs = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return np.array(vecs, dtype=np.float32)

        if not self._tfidf_fitted:
            self._tfidf.fit(texts)
            self._tfidf_fitted = True
        mat = self._tfidf.transform(texts).toarray().astype(np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return mat / norms

    def embed_mean(self, texts: List[str]) -> np.ndarray:
        """Average embedding across a list of texts -> (D,) vector."""
        vecs = self.embed(texts)
        return vecs.mean(axis=0)

    @property
    def mode(self) -> str:
        self._load()
        return self._mode


_embedder: Embedder = None

def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder