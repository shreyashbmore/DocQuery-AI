"""Command line interface.

  docquery ingest data/docs/                 # index a folder of PDFs / DOCX / TXT / HTML
  docquery ask "What is the minimum attendance requirement?"
  docquery chat                              # multi-turn chat in the terminal
  docquery docs                              # list indexed documents
Common options: -c configs/api_openai.yaml   -o retrieval.mode=dense -o llm.model=qwen2.5:7b
"""
from __future__ import annotations

import argparse
import logging
import sys

from docquery.config import load_config


def _load_env():
    try:
        from pathlib import Path
        import os

        env = Path(__file__).resolve().parent.parent / ".env"
        if env.exists():
            for line in env.read_text().splitlines():
                if "=" in line and not line.lstrip().startswith("#"):
                    k, v = line.split("=", 1)
                    if v.strip():
                        os.environ.setdefault(k.strip(), v.strip())
    except Exception:
        pass


def print_answer(ans, verbose: bool = False):
    print("\n" + ans.text + "\n")
    if ans.standalone_question != ans.question:
        print(f"  (searched for: {ans.standalone_question})")
    for c in ans.citations:
        print(f"  [{c.number}] {c.chunk.label}")
    if ans.unsupported:
        print("\n  Warning: these sentences are weakly supported by their citations:")
        for chk in ans.unsupported:
            print(f"   - {chk.sentence[:110]}  (sim {chk.similarity:.2f})")
    if verbose:
        print("\n  Retrieved passages:")
        for i, r in enumerate(ans.contexts, 1):
            s = ", ".join(f"{k}={v:.3f}" for k, v in r.scores.items())
            print(f"   [{i}] {r.chunk.label}  {s}")
        print("  Timings: " + ", ".join(f"{k}={v * 1000:.0f}ms" for k, v in ans.timings.items()))
    print()


def main(argv=None):
    _load_env()
    p = argparse.ArgumentParser(prog="docquery", description="DocQuery AI: question answering over your documents")
    p.add_argument("-c", "--config", help="YAML config layered over configs/default.yaml")
    p.add_argument("-o", "--override", action="append", default=[], help="key.sub=value override (repeatable)")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)
    pi = sub.add_parser("ingest", help="index files or folders")
    pi.add_argument("paths", nargs="+")
    pi.add_argument("--ocr", action="store_true", help="OCR scanned PDF pages (needs Tesseract)")
    pa = sub.add_parser("ask", help="ask one question")
    pa.add_argument("question")
    sub.add_parser("chat", help="interactive multi-turn chat")
    sub.add_parser("docs", help="list indexed documents")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    from docquery.pipeline import DocQuery

    cfg = load_config(args.config, args.override)
    dq = DocQuery(cfg, reranker=None if args.cmd in ("ingest", "docs") else "auto")

    if args.cmd == "ingest":
        stats = dq.ingest(args.paths, ocr=args.ocr)
        print(f"Indexed {stats['files']} file(s), {stats['chunks']} chunks -> {cfg['index_dir']}")
        if stats["skipped"]:
            print("Skipped:", *stats["skipped"], sep="\n  ")
    elif args.cmd == "docs":
        for d in dq.documents():
            print(f"{d['source']:<50} {d['chunks']:>5} chunks  {d['pages']:>4} pages")
    elif args.cmd == "ask":
        print_answer(dq.ask(args.question), args.verbose)
    elif args.cmd == "chat":
        history: list[dict] = []
        print("DocQuery AI chat. Type 'exit' to quit, 'reset' to clear history.")
        while True:
            try:
                q = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if q in ("exit", "quit"):
                break
            if q == "reset":
                history = []
                continue
            if not q:
                continue
            ans = dq.ask(q, history)
            print_answer(ans, args.verbose)
            history += [{"role": "user", "content": q}, {"role": "assistant", "content": ans.text}]
    return 0


if __name__ == "__main__":
    sys.exit(main())
