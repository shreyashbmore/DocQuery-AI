"""Sparse keyword retrieval with BM25 (catches exact terms, course codes, numbers)."""
from __future__ import annotations

import re

from rank_bm25 import BM25Plus

STOPWORDS = set("""a an and are as at be but by for from has have i if in into is it its of on or
that the their there these this to was were what when where which who why will with you your""".split())


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*", text.lower()) if t not in STOPWORDS]


class BM25Index:
    def __init__(self, ids: list[str], texts: list[str]):
        self.ids = ids
        self.token_sets = [set(tokenize(t)) for t in texts]
        # BM25+ keeps IDF positive even for tiny corpora (Okapi gives 0 when a term is in half the docs)
        self.bm25 = BM25Plus([tokenize(t) or ["<empty>"] for t in texts]) if texts else None

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        if self.bm25 is None:
            return []
        q = tokenize(query)
        scores = self.bm25.get_scores(q)
        qset = set(q)
        matched = [i for i in range(len(scores)) if self.token_sets[i] & qset]
        ranked = sorted(matched, key=lambda i: -scores[i])[:k]
        return [(self.ids[i], float(scores[i])) for i in ranked]
