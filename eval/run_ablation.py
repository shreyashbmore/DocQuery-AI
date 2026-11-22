"""Ablation study: chunking x overlap x retrieval mode x re-ranker x dedup (x LLM when --with-llm).

  python eval/run_ablation.py --docs data/docs --gold eval/gold_qa.jsonl
  python eval/run_ablation.py --docs data/docs --gold eval/gold_qa.jsonl --with-llm \
      --llm ollama:llama3.1:8b --llm openai:gpt-4o-mini

  # Chunk-overlap study: 0/10/20/30% overlap, word vs sentence-aligned, with and without de-duplication
  python eval/run_ablation.py --docs data/docs --gold eval/gold_qa.jsonl --chunking recursive \
      --overlaps 0 0.1 0.2 0.3 --overlap-modes word sentence --dedup off on --retrieval hybrid+rerank

Each chunking strategy gets its own index (built once); retrieval variants reuse it.
Results go to eval/results/ablation_<timestamp>.csv - paste that table into the report.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from docquery.cli import _load_env  # noqa: E402
from docquery.config import load_config  # noqa: E402
from docquery.ingest.embedder import build_embedder  # noqa: E402
from docquery.ingest.loaders import iter_paths  # noqa: E402
from docquery.pipeline import DocQuery  # noqa: E402
from docquery.retrieve.reranker import build_reranker  # noqa: E402
from eval.run_eval import evaluate, load_gold  # noqa: E402

CHUNKING = ["fixed", "recursive", "semantic"]
RETRIEVAL = {"dense": ("dense", False), "bm25": ("bm25", False), "hybrid": ("hybrid", False),
             "hybrid+rerank": ("hybrid", True)}


def index_tag(docs: str, chunk_cfg: dict, embedder) -> str:
    """Index folder name: changes whenever the documents, chunking settings or embedder change,
    so a stale index is never reused after you add or edit documents."""
    files = sorted((str(p), p.stat().st_size, int(p.stat().st_mtime)) for p in iter_paths([docs]))
    key = json.dumps([files, chunk_cfg, embedder.name], sort_keys=True, default=str)
    digest = hashlib.sha1(key.encode()).hexdigest()[:8]
    return (f"{chunk_cfg['strategy']}_s{chunk_cfg['chunk_size']}_ov{float(chunk_cfg['overlap']):g}_"
            f"{chunk_cfg.get('overlap_mode', 'sentence')}_{digest}")


def run_grid(base: dict, gold: list[dict], docs: str, embedder, reranker, *, chunking=CHUNKING,
             overlaps=None, modes=None, dedup=("on",), retrieval=tuple(RETRIEVAL), llms=None,
             log=print) -> list[dict]:
    """Evaluate every combination and return one summary row per configuration.

    Each (chunking, overlap, overlap_mode) gets its own index, built once and reused by the retrieval
    variants. ``llms`` = list of "provider:model" to also generate answers (slow); None = retrieval only.
    """
    overlaps = list(overlaps or [base["chunking"]["overlap"]])
    modes = list(modes or [base["chunking"].get("overlap_mode", "sentence")])
    dedup_on = base["retrieval"].get("dedup_threshold") or 0.8
    results = []
    for strategy, overlap, omode in itertools.product(chunking, overlaps, modes):
        if strategy != "recursive" and omode != modes[0]:
            continue  # overlap mode only affects the recursive chunker
        if strategy == "semantic" and overlap != overlaps[0]:
            continue  # semantic chunks have no overlap
        chunk_cfg = {**base["chunking"], "strategy": strategy, "overlap": overlap, "overlap_mode": omode}
        tag = index_tag(docs, chunk_cfg, embedder)
        cfg = {**base, "index_dir": str(ROOT / "index" / "ablation" / tag), "chunking": chunk_cfg}
        dq = DocQuery(cfg, embedder=embedder, reranker=None)
        if not dq.chunks:
            log(f"Building index {tag} ...")
            dq.ingest([docs])
        for rname, dd, llm_spec in itertools.product(retrieval, dedup, llms or [None]):
            mode, use_rr = RETRIEVAL[rname]
            dq.cfg = {**cfg, "retrieval": {**cfg["retrieval"], "mode": mode,
                                           "dedup_threshold": dedup_on if dd == "on" else None}}
            dq.reranker = reranker if use_rr else None
            dq._retriever = None
            if llm_spec:
                provider, model = llm_spec.split(":", 1)
                dq.cfg["llm"] = {**cfg["llm"], "provider": provider, "model": model}
                dq._llm = None
            ov = "" if strategy == "semantic" else f" {overlap:g}" + (f" {omode}" if strategy == "recursive" else "")
            name = f"{strategy}{ov} · {rname} · dedup {dd}" + (f" · {llm_spec}" if llm_spec else "")
            log(f"Running {name} ...")
            summary, _ = evaluate(dq, gold, retrieval_only=not llm_spec)
            results.append({"config": name, "chunking": strategy, "overlap": overlap, "overlap_mode": omode,
                            "retrieval": rname, "dedup": dd, "llm": llm_spec or "", "chunks": len(dq.chunks),
                            **summary})
    return results


TABLE_COLS = ["chunks", "recall@1", "recall@5", "mrr@10", "ndcg@10", "dup_rate@5", "token_f1",
              "citation_precision", "abstention_correct", "p50_ms", "p95_ms"]


def markdown_table(results: list[dict], first: str = "config", cols=None) -> str:
    keys = {k for r in results for k in r}
    cols = [c for c in (cols or TABLE_COLS) if c in keys]
    lines = [f"| {first} | " + " | ".join(cols) + " |", "|---|" + "---:|" * len(cols)]
    for r in results:
        lines.append(f"| {r[first]} | " + " | ".join("" if r.get(c) is None else str(r.get(c)) for c in cols) + " |")
    return "\n".join(lines)


def write_csv(rows: list[dict], path: Path) -> None:
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main(argv=None):
    _load_env()
    p = argparse.ArgumentParser()
    p.add_argument("--docs", required=True)
    p.add_argument("--gold", required=True)
    p.add_argument("-c", "--config")
    p.add_argument("-o", "--override", action="append", default=[])
    p.add_argument("--chunking", nargs="+", default=CHUNKING)
    p.add_argument("--overlaps", nargs="+", type=float, help="words (40) or fractions of chunk size (0.2)")
    p.add_argument("--overlap-modes", nargs="+", choices=["word", "sentence"])
    p.add_argument("--dedup", nargs="+", choices=["on", "off"], default=["on"])
    p.add_argument("--retrieval", nargs="+", choices=list(RETRIEVAL), default=list(RETRIEVAL))
    p.add_argument("--with-llm", action="store_true", help="also generate answers (slow)")
    p.add_argument("--llm", action="append", default=[], help="provider:model, repeatable")
    args = p.parse_args(argv)

    base = load_config(args.config, args.override)
    embedder = build_embedder(base["embedding"])             # load models once
    reranker = build_reranker({**base["reranker"], "enabled": True})
    llms = (args.llm or [f"{base['llm']['provider']}:{base['llm']['model']}"]) if args.with_llm else None
    results = run_grid(base, load_gold(args.gold), args.docs, embedder, reranker, chunking=args.chunking,
                       overlaps=args.overlaps, modes=args.overlap_modes, dedup=args.dedup,
                       retrieval=args.retrieval, llms=llms)
    out_dir = ROOT / "eval" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"ablation_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    write_csv(results, out_path)
    print("\n" + markdown_table(results))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
