"""Vector stores: an in-memory NumPy store (zero setup) and Qdrant (embedded or server)."""
from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np


class VectorStore:
    def add(self, ids: list[str], vectors: np.ndarray) -> None: ...
    def search(self, vector: np.ndarray, k: int) -> list[tuple[str, float]]: ...
    def delete(self, ids: list[str]) -> None: ...
    def save(self) -> None: ...
    def __len__(self) -> int: ...


class MemoryStore(VectorStore):
    """Exact cosine search with a matrix product. Fast enough for tens of thousands of chunks."""

    def __init__(self, path: Path, dim: int):
        self.path, self.dim = Path(path), dim
        self.ids: list[str] = []
        self.vectors = np.zeros((0, dim), dtype=np.float32)
        if (self.path / "vectors.npy").exists():
            self.vectors = np.load(self.path / "vectors.npy")
            self.ids = (self.path / "vector_ids.txt").read_text().splitlines()

    def add(self, ids, vectors):
        self.ids += list(ids)
        self.vectors = np.vstack([self.vectors, np.asarray(vectors, dtype=np.float32)])

    def search(self, vector, k):
        if not self.ids:
            return []
        scores = self.vectors @ np.asarray(vector, dtype=np.float32)
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(self.ids[i], float(scores[i])) for i in top]

    def delete(self, ids):
        drop = set(ids)
        keep = [i for i, cid in enumerate(self.ids) if cid not in drop]
        self.ids = [self.ids[i] for i in keep]
        self.vectors = self.vectors[keep]

    def save(self):
        self.path.mkdir(parents=True, exist_ok=True)
        np.save(self.path / "vectors.npy", self.vectors)
        (self.path / "vector_ids.txt").write_text("\n".join(self.ids))

    def __len__(self):
        return len(self.ids)


class QdrantStore(VectorStore):
    COLLECTION = "docquery"

    def __init__(self, path: Path, dim: int, url: str | None = None):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        self.client = QdrantClient(url=url) if url else QdrantClient(path=str(Path(path) / "qdrant"))
        if not self.client.collection_exists(self.COLLECTION):
            self.client.create_collection(self.COLLECTION, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))

    @staticmethod
    def _pid(chunk_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))

    def add(self, ids, vectors):
        from qdrant_client.models import PointStruct

        points = [PointStruct(id=self._pid(c), vector=v.tolist(), payload={"chunk_id": c}) for c, v in zip(ids, vectors)]
        for i in range(0, len(points), 256):
            self.client.upsert(self.COLLECTION, points=points[i:i + 256])

    def search(self, vector, k):
        res = self.client.query_points(self.COLLECTION, query=np.asarray(vector).tolist(), limit=k, with_payload=True)
        return [(p.payload["chunk_id"], float(p.score)) for p in res.points]

    def delete(self, ids):
        from qdrant_client.models import PointIdsList

        self.client.delete(self.COLLECTION, points_selector=PointIdsList(points=[self._pid(c) for c in ids]))

    def save(self):
        pass  # Qdrant persists on write

    def __len__(self):
        return self.client.count(self.COLLECTION).count


def build_store(cfg: dict, index_dir: str, dim: int) -> VectorStore:
    backend = cfg.get("backend", "memory")
    if backend == "memory":
        return MemoryStore(Path(index_dir), dim)
    if backend == "qdrant":
        return QdrantStore(Path(index_dir), dim, cfg.get("qdrant_url"))
    raise ValueError(f"Unknown store backend: {backend}")
