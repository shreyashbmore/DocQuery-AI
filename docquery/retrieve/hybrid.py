"""Retriever: dense, BM25 or hybrid (RRF), followed by optional cross-encoder re-ranking."""
from __future__ import annotations

from docquery.metrics import containment
from docquery.retrieve.bm25 import BM25Index
from docquery.retrieve.fusion import reciprocal_rank_fusion
from docquery.schema import Chunk, Retrieved


class Retriever:
    def __init__(self, chunks: dict[str, Chunk], store, embedder, cfg: dict, reranker=None):
        self.chunks, self.store, self.embedder, self.cfg, self.reranker = chunks, store, embedder, cfg, reranker
        ids = list(chunks)
        self.bm25 = BM25Index(ids, [chunks[i].text for i in ids])

    def retrieve(self, query: str, top_k: int | None = None) -> list[Retrieved]:
        mode = self.cfg.get("mode", "hybrid")
        n = int(self.cfg.get("candidates", 20))
        top_k = top_k or int(self.cfg.get("top_k", 5))
        per_stage: dict[str, dict] = {}

        dense, sparse = [], []
        if mode in ("dense", "hybrid"):
            dense = self.store.search(self.embedder.embed_query(query), n)
            for cid, s in dense:
                per_stage.setdefault(cid, {})["dense"] = s
        if mode in ("bm25", "hybrid"):
            sparse = self.bm25.search(query, n)
            for cid, s in sparse:
                per_stage.setdefault(cid, {})["bm25"] = s

        if mode == "hybrid":
            fused = reciprocal_rank_fusion([[c for c, _ in dense], [c for c, _ in sparse]], int(self.cfg.get("rrf_k", 60)))
            for cid, s in fused:
                per_stage[cid]["rrf"] = s
            ranked = fused[:n]
        else:
            ranked = dense if mode == "dense" else sparse

        results = [Retrieved(self.chunks[c], s, per_stage.get(c, {})) for c, s in ranked if c in self.chunks]

        if self.reranker is not None and results:
            scores = self.reranker.score(query, [r.chunk.text for r in results])
            for r, s in zip(results, scores):
                r.scores["rerank"] = s
                r.score = s
            results.sort(key=lambda r: -r.score)

        threshold = self.cfg.get("dedup_threshold")
        if threshold:
            results = self.deduplicate(results, float(threshold), top_k)
        return results[:top_k]

    @staticmethod
    def deduplicate(results: list[Retrieved], threshold: float, top_k: int) -> list[Retrieved]:
        """Keep a result only if it is not mostly contained in a higher-ranked one.

        Overlapping chunks often retrieve as near-identical pairs; without this they can fill the
        top-k and push out other relevant passages."""
        kept: list[Retrieved] = []
        for r in results:
            if all(containment(r.chunk.text, k.chunk.text) < threshold for k in kept):
                kept.append(r)
            if len(kept) == top_k:
                break
        return kept
