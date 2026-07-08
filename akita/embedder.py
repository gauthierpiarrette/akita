"""
Pluggable embedders for Akita Private Memory.

The embedder MUST run on the key holder's side: if any server embeds the
text, that server has seen the plaintext and the privacy promise is void.

- MiniLMEmbedder: real semantic embeddings (sentence-transformers), used
  by the demos. ~25 MB quantized equivalents of this model run on phones
  and in browsers, which is the production deployment target.
- HashingEmbedder: deterministic, dependency-free token-hashing
  projection. No semantics worth shipping — it exists so the tests can
  verify the *encrypted pipeline* exactly matches the plaintext pipeline
  without downloading a model.
"""

from __future__ import annotations

import hashlib

import numpy as np


class HashingEmbedder:
    def __init__(self, dim: int = 384):
        self.dim = dim

    def _token_vec(self, token: str) -> np.ndarray:
        seed = int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], "big")
        return np.random.default_rng(seed).standard_normal(self.dim)

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim))
        for i, text in enumerate(texts):
            for token in text.lower().split():
                out[i] += self._token_vec(token)
            norm = np.linalg.norm(out[i])
            if norm > 0:
                out[i] /= norm
        return out


class MiniLMEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        get_dim = getattr(self._model, "get_embedding_dimension", None) \
            or self._model.get_sentence_embedding_dimension
        self.dim = get_dim()

    def encode(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        ).astype(np.float64)
