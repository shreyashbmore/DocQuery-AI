"""Evaluation metrics for retrieval and answers."""
from __future__ import annotations

import math
import re
from collections import Counter


def recall_at_k(relevant_flags: list[bool], k: int) -> float:
    """1 if any relevant passage is in the top k (hit rate per question)."""
    return float(any(relevant_flags[:k]))


def mrr_at_k(relevant_flags: list[bool], k: int = 10) -> float:
    for i, rel in enumerate(relevant_flags[:k], start=1):
        if rel:
            return 1.0 / i
    return 0.0


def ndcg_at_k(relevant_flags: list[bool], k: int = 10) -> float:
    dcg = sum(1.0 / math.log2(i + 1) for i, rel in enumerate(relevant_flags[:k], start=1) if rel)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(k, sum(relevant_flags)) + 1))
    return dcg / ideal if ideal else 0.0


def _norm_tokens(text: str) -> list[str]:
    text = re.sub(r"\[\d+\]", " ", text.lower())
    return re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text)


def token_f1(prediction: str, reference: str) -> float:
    p, r = _norm_tokens(prediction), _norm_tokens(reference)
    if not p or not r:
        return float(p == r)
    common = sum((Counter(p) & Counter(r)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(p), common / len(r)
    return 2 * precision * recall / (precision + recall)


def numbers_match(prediction: str, reference: str) -> float | None:
    """For numeric questions: are all numbers in the reference present in the prediction? None if not numeric."""
    ref = set(re.findall(r"\d+(?:\.\d+)?", reference))
    if not ref:
        return None
    pred = set(re.findall(r"\d+(?:\.\d+)?", re.sub(r"\[\d+\]", " ", prediction)))
    return float(ref <= pred)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, math.ceil(pct / 100 * len(s)) - 1))
    return s[idx]


def is_relevant(chunk, item: dict) -> bool:
    """A chunk is relevant if it is from the gold source (and page, when given),
    or if it contains the gold evidence string."""
    evidence = item.get("evidence")
    if evidence and " ".join(evidence.lower().split()) in " ".join(chunk.text.lower().split()):
        return True
    if item.get("source") and chunk.source != item["source"]:
        return False
    pages = item.get("pages")
    if pages:
        return chunk.page in pages
    return bool(item.get("source")) and not evidence
