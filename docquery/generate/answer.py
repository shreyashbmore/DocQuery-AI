"""Answer post-processing: citation parsing, confidence gate and sentence-level support check."""
from __future__ import annotations

import re

import numpy as np

from docquery.generate.prompts import NOT_FOUND
from docquery.ingest.chunkers import split_sentences
from docquery.schema import Citation, Retrieved, SentenceCheck

_CITE = re.compile(r"\[(\d+)\]")


def passes_gate(contexts: list[Retrieved], cfg: dict) -> tuple[bool, str]:
    """Return (ok, reason). Abstain when the best evidence is too weak to answer from."""
    if not contexts:
        return False, "no passages retrieved"
    if not cfg.get("enabled", True):
        return True, "gate disabled"
    best = contexts[0].scores
    if "rerank" in best:
        top = max(r.scores.get("rerank", 0) for r in contexts)
        return top >= cfg["min_rerank_score"], f"top rerank score {top:.3f}"
    dense = [r.scores["dense"] for r in contexts if "dense" in r.scores]
    if dense:
        return max(dense) >= cfg["min_dense_score"], f"top dense score {max(dense):.3f}"
    return True, "no calibrated score available"


def is_abstention(text: str) -> bool:
    return NOT_FOUND.lower().rstrip(".") in text.lower()


def extract_citations(text: str, contexts: list[Retrieved]) -> list[Citation]:
    nums = sorted({int(n) for n in _CITE.findall(text)})
    return [Citation(n, contexts[n - 1].chunk) for n in nums if 1 <= n <= len(contexts)]


def answer_sentences(text: str) -> list[str]:
    """Split an answer into sentences, re-attaching citations that the splitter pushed to the next sentence
    (models often write "... exam. [1] Next sentence")."""
    out: list[str] = []
    for sent in split_sentences(text):
        lead = re.match(r"^((?:\s*\[\d+\])+)\s*(.*)$", sent, re.S)
        if lead and out:
            out[-1] = out[-1] + " " + lead.group(1).strip()
            sent = lead.group(2)
        if sent.strip():
            out.append(sent.strip())
    return out


def support_check(text: str, contexts: list[Retrieved], embedder, min_similarity: float) -> list[SentenceCheck]:
    """Flag sentences with no citation, or whose closest cited passage sentence is not similar enough."""
    checks = []
    sentences = [s for s in answer_sentences(text) if _CITE.sub("", s).strip()]
    if not sentences:
        return checks
    plain = [_CITE.sub("", s).strip() for s in sentences]
    sent_vecs = embedder.embed_documents(plain)
    cache: dict[int, np.ndarray] = {}
    for sent, vec in zip(sentences, sent_vecs):
        nums = [int(n) for n in _CITE.findall(sent) if 1 <= int(n) <= len(contexts)]
        best = 0.0
        for n in nums:
            if n not in cache:
                cache[n] = embedder.embed_documents(split_sentences(contexts[n - 1].chunk.text) or [contexts[n - 1].chunk.text])
            best = max(best, float(np.max(cache[n] @ vec)))
        checks.append(SentenceCheck(sent, nums, bool(nums) and best >= min_similarity, best))
    return checks
