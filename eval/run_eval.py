"""Evaluate one configuration on the gold Q&A set.

  python eval/run_eval.py --gold eval/gold_qa.jsonl                       # retrieval + answers
  python eval/run_eval.py --gold eval/gold_qa.jsonl --retrieval-only      # fast, no LLM calls
  python eval/run_eval.py -o retrieval.mode=dense -o reranker.enabled=false --retrieval-only

Gold file: one JSON object per line
  {"id": "q1", "question": "...", "answer": "...", "source": "rules.pdf", "pages": [4],
   "evidence": "optional exact phrase", "answerable": true, "type": "factual"}
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as stats
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from docquery.cli import _load_env  # noqa: E402
from docquery.config import load_config  # noqa: E402
from docquery.metrics import (is_relevant, mrr_at_k, ndcg_at_k, numbers_match, percentile,  # noqa: E402
                              recall_at_k, token_f1)
from docquery.pipeline import DocQuery  # noqa: E402


def load_gold(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def evaluate(dq: DocQuery, gold: list[dict], retrieval_only: bool = False) -> tuple[dict, list[dict]]:
    rows = []
    for item in gold:
        answerable = item.get("answerable", True)
        row = {"id": item["id"], "type": item.get("type", ""), "answerable": answerable}
        t0 = time.perf_counter()
        retrieved = dq.retriever.retrieve(item["question"], top_k=10)
        row["retrieve_ms"] = (time.perf_counter() - t0) * 1000
        if answerable:
            flags = [is_relevant(r.chunk, item) for r in retrieved]
            row.update({"recall@1": recall_at_k(flags, 1), "recall@5": recall_at_k(flags, 5),
                        "recall@10": recall_at_k(flags, 10), "mrr@10": mrr_at_k(flags, 10),
                        "ndcg@10": ndcg_at_k(flags, 10)})
        if not retrieval_only:
            ans = dq.ask(item["question"])
            row["total_ms"] = ans.timings.get("total", 0) * 1000
            row["abstained"] = ans.abstained
            row["abstention_correct"] = float(ans.abstained == (not answerable))
            if answerable and not ans.abstained:
                row["token_f1"] = token_f1(ans.text, item.get("answer", ""))
                nm = numbers_match(ans.text, item.get("answer", ""))
                if nm is not None:
                    row["numeric_exact"] = nm
                if ans.citations:
                    row["citation_precision"] = sum(is_relevant(c.chunk, item) for c in ans.citations) / len(ans.citations)
                if ans.checks:
                    row["unsupported_rate"] = len(ans.unsupported) / len(ans.checks)
            row["answer"] = ans.text
        rows.append(row)
    return summarise(rows), rows


def summarise(rows: list[dict]) -> dict:
    def mean(key):
        vals = [r[key] for r in rows if key in r]
        return round(stats.mean(vals), 4) if vals else None

    summary = {k: mean(k) for k in ["recall@1", "recall@5", "recall@10", "mrr@10", "ndcg@10", "token_f1",
                                    "numeric_exact", "citation_precision", "unsupported_rate", "abstention_correct"]}
    lat = [r["total_ms"] for r in rows if "total_ms" in r] or [r["retrieve_ms"] for r in rows]
    summary.update({"p50_ms": round(percentile(lat, 50)), "p95_ms": round(percentile(lat, 95)), "n": len(rows)})
    return {k: v for k, v in summary.items() if v is not None}


def main(argv=None):
    _load_env()
    p = argparse.ArgumentParser()
    p.add_argument("--gold", default="eval/gold_qa.jsonl")
    p.add_argument("-c", "--config")
    p.add_argument("-o", "--override", action="append", default=[])
    p.add_argument("--retrieval-only", action="store_true")
    p.add_argument("--out", help="CSV file for per-question results")
    args = p.parse_args(argv)

    cfg = load_config(args.config, args.override)
    dq = DocQuery(cfg)
    if not dq.chunks:
        sys.exit(f"Index {cfg['index_dir']} is empty. Run `docquery ingest <docs>` first.")
    summary, rows = evaluate(dq, load_gold(args.gold), args.retrieval_only)
    print(json.dumps(summary, indent=2))
    if args.out:
        keys = sorted({k for r in rows for k in r})
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    return summary


if __name__ == "__main__":
    main()
