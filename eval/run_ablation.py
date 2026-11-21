"""Ablation study: chunking x retrieval mode x re-ranker (x LLM when --with-llm).

  python eval/run_ablation.py --docs data/docs --gold eval/gold_qa.jsonl
  python eval/run_ablation.py --docs data/docs --gold eval/gold_qa.jsonl --with-llm \
      --llm ollama:llama3.1:8b --llm openai:gpt-4o-mini

Each chunking strategy gets its own index (built once); retrieval variants reuse it.
Results go to eval/results/ablation_<timestamp>.csv - paste that table into the report.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from docquery.cli import _load_env  # noqa: E402
from docquery.config import load_config  # noqa: E402
from docquery.ingest.embedder import build_embedder  # noqa: E402
from docquery.pipeline import DocQuery  # noqa: E402
from docquery.retrieve.reranker import build_reranker  # noqa: E402
from eval.run_eval import evaluate, load_gold  # noqa: E402

CHUNKING = ["fixed", "recursive", "semantic"]
RETRIEVAL = [("dense", False), ("bm25", False), ("hybrid", False), ("hybrid", True)]


def main(argv=None):
    _load_env()
    p = argparse.ArgumentParser()
    p.add_argument("--docs", required=True)
    p.add_argument("--gold", required=True)
    p.add_argument("-c", "--config")
    p.add_argument("-o", "--override", action="append", default=[])
    p.add_argument("--chunking", nargs="+", default=CHUNKING)
    p.add_argument("--with-llm", action="store_true", help="also generate answers (slow)")
    p.add_argument("--llm", action="append", default=[], help="provider:model, repeatable")
    args = p.parse_args(argv)

    base = load_config(args.config, args.override)
    gold = load_gold(args.gold)
    embedder = build_embedder(base["embedding"])             # load models once
    reranker = build_reranker({**base["reranker"], "enabled": True})
    llms = args.llm or [f"{base['llm']['provider']}:{base['llm']['model']}"]

    out_dir = ROOT / "eval" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"ablation_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    results = []

    for strategy in args.chunking:
        index_dir = str(ROOT / "index" / "ablation" / f"{strategy}_{embedder.name.replace('/', '_')}")
        cfg = {**base, "index_dir": index_dir, "chunking": {**base["chunking"], "strategy": strategy}}
        dq = DocQuery(cfg, embedder=embedder, reranker=None)
        if not dq.chunks:
            print(f"Building {strategy} index...")
            dq.ingest([args.docs])
        for (mode, use_rr), llm_spec in itertools.product(RETRIEVAL, llms if args.with_llm else [None]):
            dq.cfg = {**cfg, "retrieval": {**cfg["retrieval"], "mode": mode}}
            dq.reranker = reranker if use_rr else None
            dq._retriever = None
            if llm_spec:
                provider, model = llm_spec.split(":", 1)
                dq.cfg["llm"] = {**cfg["llm"], "provider": provider, "model": model}
                dq._llm = None
            name = f"{strategy} | {mode}{' + rerank' if use_rr else ''}" + (f" | {llm_spec}" if llm_spec else "")
            print(f"Running {name} ...")
            summary, _ = evaluate(dq, gold, retrieval_only=not args.with_llm)
            results.append({"config": name, "chunking": strategy, "retrieval": mode, "rerank": use_rr,
                            "llm": llm_spec or "", "chunks": len(dq.chunks), **summary})

    keys = list(dict.fromkeys(k for r in results for k in r))
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(results)

    cols = [c for c in ["recall@1", "recall@5", "mrr@10", "ndcg@10", "token_f1", "abstention_correct", "p95_ms"] if c in keys]
    print("\n| Configuration | " + " | ".join(cols) + " |")
    print("|---|" + "---|" * len(cols))
    for r in results:
        print(f"| {r['config']} | " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
