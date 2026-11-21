"""Cross-encoder re-ranking: reads query and passage together for a precise relevance score."""
from __future__ import annotations

import math


class CrossEncoderReranker:
    def __init__(self, model: str):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model)
        self.name = model

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        raw = [float(x) for x in self.model.predict([(query, p) for p in passages])]
        # Newer sentence-transformers already apply a sigmoid for single-label models; older ones return logits.
        if all(0.0 <= x <= 1.0 for x in raw):
            return raw
        return [1.0 / (1.0 + math.exp(-x)) for x in raw]


class LexicalReranker:
    """Cheap fallback used in tests: fraction of query terms present in the passage."""

    name = "lexical"

    def score(self, query, passages):
        from docquery.retrieve.bm25 import tokenize

        q = set(tokenize(query))
        return [len(q & set(tokenize(p))) / max(1, len(q)) for p in passages]


def build_reranker(cfg: dict):
    if not cfg.get("enabled", True):
        return None
    if cfg.get("model") == "lexical":
        return LexicalReranker()
    return CrossEncoderReranker(cfg["model"])
