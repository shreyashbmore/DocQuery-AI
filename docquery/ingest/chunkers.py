"""Three chunking strategies compared in the ablation study.

Sizes are measured in words (roughly 0.75 tokens each) to stay tokenizer-agnostic.
"""
from __future__ import annotations

import re
from typing import Callable

import numpy as np

from docquery.schema import Chunk, Section

_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def split_sentences(text: str) -> list[str]:
    sents = []
    for para in re.split(r"\n\s*\n", text):
        para = " ".join(para.split())
        if para:
            sents += [s.strip() for s in _SENT_RE.split(para) if s.strip()]
    return sents


def _words(text: str) -> list[str]:
    return text.split()


# ---------------------------------------------------------------- fixed
def fixed_chunks(text: str, size: int, overlap: int) -> list[str]:
    words = _words(text)
    if not words:
        return []
    step = max(1, size - overlap)
    return [" ".join(words[i:i + size]) for i in range(0, max(1, len(words) - overlap), step)]


# ---------------------------------------------------------------- recursive
_SEPARATORS = ["\n\n", "\n", ". ", " "]


def _recursive_split(text: str, size: int, seps: list[str]) -> list[str]:
    if len(_words(text)) <= size:
        return [text]
    if not seps:
        return fixed_chunks(text, size, 0)
    sep, rest = seps[0], seps[1:]
    pieces = [p for p in text.split(sep) if p.strip()]
    if len(pieces) == 1:
        return _recursive_split(text, size, rest)
    out = []
    for p in pieces:
        p = p + (sep.strip() if sep.strip() else "")
        out += _recursive_split(p, size, rest) if len(_words(p)) > size else [p]
    return out


def recursive_chunks(text: str, size: int, overlap: int) -> list[str]:
    """Split on paragraphs, then lines, then sentences; merge neighbours up to ``size`` words.

    Paragraph breaks are kept so later stages (sentence splitting, citations) see real structure.
    """
    pieces = [p.strip() for p in _recursive_split(text, size, _SEPARATORS) if p.strip()]
    chunks, cur, cur_len = [], [], 0
    for piece in pieces:
        n = len(_words(piece))
        if cur and cur_len + n > size:
            chunks.append("\n\n".join(cur))
            tail = _words(cur[-1])[-overlap:] if overlap else []
            cur, cur_len = ([" ".join(tail)], len(tail)) if tail else ([], 0)
        cur.append(piece)
        cur_len += n
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks


# ---------------------------------------------------------------- semantic
def semantic_chunks(text: str, size: int, embed: Callable[[list[str]], np.ndarray],
                    threshold_pct: float = 0.25) -> list[str]:
    """Break where adjacent sentences are least similar (bottom ``threshold_pct`` of similarities)."""
    sents = split_sentences(text)
    if len(sents) <= 1:
        return [" ".join(sents)] if sents else []
    vecs = embed(sents)
    sims = np.sum(vecs[:-1] * vecs[1:], axis=1)
    cut = np.quantile(sims, threshold_pct)
    chunks, cur, cur_len = [], [sents[0]], len(_words(sents[0]))
    for sent, sim in zip(sents[1:], sims):
        n = len(_words(sent))
        if (sim < cut and cur_len >= size * 0.3) or cur_len + n > size:
            chunks.append(" ".join(cur))
            cur, cur_len = [], 0
        cur.append(sent)
        cur_len += n
    if cur:
        chunks.append(" ".join(cur))
    return chunks


# ---------------------------------------------------------------- entry point
def chunk_sections(sections: list[Section], cfg: dict, embed=None) -> list[Chunk]:
    strategy = cfg.get("strategy", "recursive")
    size, overlap = int(cfg.get("chunk_size", 220)), int(cfg.get("overlap", 40))
    out: list[Chunk] = []
    for sec in sections:
        if strategy == "fixed":
            texts = fixed_chunks(sec.text, size, overlap)
        elif strategy == "recursive":
            texts = recursive_chunks(sec.text, size, overlap)
        elif strategy == "semantic":
            if embed is None:
                raise ValueError("semantic chunking needs an embedding function")
            texts = semantic_chunks(sec.text, size, embed, float(cfg.get("semantic_threshold", 0.25)))
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")
        for i, t in enumerate(texts):
            if len(t.split()) < 5:
                continue
            cid = f"{sec.doc_id}-p{sec.page or 0}-c{i}"
            out.append(Chunk(cid, sec.doc_id, sec.source, t, sec.page, {"strategy": strategy}))
    return out
