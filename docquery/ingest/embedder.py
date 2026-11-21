"""Embedding providers behind one interface. All return L2-normalised float32 arrays."""
from __future__ import annotations

import hashlib
import os
import re

import numpy as np


def _normalise(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


class Embedder:
    dim: int
    name: str

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


class SentenceTransformerEmbedder(Embedder):
    # BGE models work best with an instruction prefix on queries only.
    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, model: str, batch_size: int = 32):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model)
        self.batch_size = batch_size
        self.name = model
        get_dim = getattr(self.model, "get_sentence_embedding_dimension", None) or self.model.get_embedding_dimension
        self.dim = int(get_dim())
        self._prefix = self.QUERY_PREFIX if "bge" in model.lower() and "m3" not in model.lower() else ""

    def embed_documents(self, texts):
        return _normalise(self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True))

    def embed_query(self, text):
        return self.embed_documents([self._prefix + text])[0]


class OpenAIEmbedder(Embedder):
    def __init__(self, model: str = "text-embedding-3-small", batch_size: int = 64):
        import httpx

        self.name, self.batch_size = model, batch_size
        self.base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self.client = httpx.Client(headers={"Authorization": f"Bearer {key}"}, timeout=60)
        self.dim = len(self.embed_documents(["dimension probe"])[0])

    def embed_documents(self, texts):
        vecs = []
        for i in range(0, len(texts), self.batch_size):
            r = self.client.post(f"{self.base}/embeddings",
                                 json={"model": self.name, "input": texts[i:i + self.batch_size]})
            r.raise_for_status()
            vecs += [d["embedding"] for d in sorted(r.json()["data"], key=lambda d: d["index"])]
        return _normalise(np.array(vecs))


class HashEmbedder(Embedder):
    """Dependency-free hashed bag-of-words embedder. For tests and offline smoke runs only."""

    def __init__(self, dim: int = 512):
        self.dim, self.name = dim, f"hash-{dim}"

    def embed_documents(self, texts):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for tok in re.findall(r"\w+", text.lower()):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                out[row, h % self.dim] += 1.0 if (h >> 8) & 1 else -1.0
        out[np.all(out == 0, axis=1), 0] = 1.0
        return _normalise(out)


def build_embedder(cfg: dict) -> Embedder:
    provider = cfg.get("provider", "sentence-transformers")
    if provider == "sentence-transformers":
        return SentenceTransformerEmbedder(cfg["model"], int(cfg.get("batch_size", 32)))
    if provider == "openai":
        return OpenAIEmbedder(cfg.get("model", "text-embedding-3-small"))
    if provider == "hash":
        return HashEmbedder(int(cfg.get("dim", 512)))
    raise ValueError(f"Unknown embedding provider: {provider}")
