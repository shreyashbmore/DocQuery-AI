"""Draft a gold Q&A set from your documents with an LLM. Every item MUST then be checked by hand.

  python eval/draft_gold.py --docs data/docs --n 80 --unanswerable 20 --llm qwen2.5:7b
  python eval/review_gold.py          # accept / edit / delete each draft -> eval/gold_qa.jsonl

How it works: documents are split into passages; a spread of passages (across files and pages) is
sent to the LLM, which writes one question, its answer, and an exact evidence sentence copied from
the passage. Items whose evidence is not found verbatim in the passage are discarded. For
unanswerable items the LLM writes a question on the same topic that the passage does NOT answer.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from docquery.cli import _load_env  # noqa: E402
from docquery.generate.llm import build_llm  # noqa: E402
from docquery.ingest.chunkers import chunk_sections  # noqa: E402
from docquery.ingest.loaders import iter_paths, load_document  # noqa: E402

TYPES = ["factual", "numeric", "reasoning"]

ANSWERABLE_PROMPT = """You write test questions for a document question-answering system.
Read the passage and write ONE question that a student or staff member might really ask, which this passage answers.
Question type: {qtype} ({type_hint}).
If the passage is mainly a practice exercise or quiz item (answer choices, "Quantity A/B"), an answer key,
a figure or table without explanation, a table of contents, an index, or has no self-contained fact,
return {{"skip": true}} instead.
Rules:
- The question must make sense on its own, without seeing the passage (name the subject explicitly).
  Never refer to "the passage", "the text", "question 3", "choice (A)" or "the figure".
- "answer": a short, complete answer taken from the passage.
- "evidence": ONE sentence copied EXACTLY, character for character, from the passage that contains the answer.
Return JSON only: {{"question": "...", "answer": "...", "evidence": "..."}}

Passage (from {source}{page}):
\"\"\"{text}\"\"\""""

UNANSWERABLE_PROMPT = """You write test questions for a document question-answering system.
Read the passage and write ONE realistic question on the SAME topic that the passage does NOT answer
(for example a related fee, date, person or rule that is not mentioned). The system should refuse to answer it.
The question must make sense on its own: never refer to "the passage", "this book" or "the text".
If the passage has no clear topic (index, table of contents, answer key), return {{"skip": true}}.
Return JSON only: {{"question": "..."}}

Passage (from {source}{page}):
\"\"\"{text}\"\"\""""

TYPE_HINTS = {"factual": "who/what/where/which, answered by one fact",
              "numeric": "the answer is a number, amount, percentage, date or count",
              "reasoning": "needs combining two or more sentences, or a condition/exception"}


# Questions that only make sense next to the passage are useless for retrieval evaluation.
CONTEXT_BOUND = re.compile(r"\b(the|this) (passage|text|excerpt|book|chapter|figure|table|question|problem)\b|"
                           r"\bquestion \d+|\bchoice \(?[a-e]\)?|\bquantity [ab]\b|\bstatements? (i|ii|iii)\b",
                           re.I)
QUIZ_MARKERS = re.compile(r"\(([A-E])\)|\bQuantity [AB]\b|\banswer choices?\b", re.I)


def norm(s) -> str:
    return " ".join(str(s or "").lower().split())


def text(v) -> str:
    """LLMs sometimes return numbers or lists instead of strings."""
    if v is None:
        return ""
    if isinstance(v, list):
        v = " ".join(map(str, v))
    return str(v).strip()


def looks_like_quiz(passage: str) -> bool:
    return len(QUIZ_MARKERS.findall(passage)) >= 3


def parse_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def spread_sample(chunks, k, rng):
    """Round-robin over documents so every file contributes questions."""
    by_doc: dict[str, list] = {}
    for c in chunks:
        by_doc.setdefault(c.source, []).append(c)
    for lst in by_doc.values():
        rng.shuffle(lst)
    out = []
    while len(out) < k and any(by_doc.values()):
        for src in list(by_doc):
            if by_doc[src]:
                out.append(by_doc[src].pop())
    return out[:k]


def main(argv=None):
    _load_env()
    p = argparse.ArgumentParser()
    p.add_argument("--docs", default="data/docs")
    p.add_argument("--n", type=int, default=80, help="answerable questions to draft")
    p.add_argument("--unanswerable", type=int, default=20)
    p.add_argument("--llm", default="qwen2.5:7b", help="Ollama model name, or provider:model (openai:gpt-4o-mini)")
    p.add_argument("--out", default="eval/gold_qa.draft.jsonl")
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--resume", action="store_true", help="keep drafts already in --out and add to them")
    args = p.parse_args(argv)

    provider, model = args.llm.split(":", 1) if args.llm.startswith(("openai:", "ollama:")) else ("ollama", args.llm)
    llm = build_llm({"provider": provider, "model": model, "temperature": 0.3, "max_tokens": 400, "timeout": 180})
    rng = random.Random(args.seed)

    sections = [s for path in iter_paths([args.docs]) for s in load_document(path)]
    if not sections:
        sys.exit(f"No supported documents found in {args.docs}")
    all_chunks = [c for c in chunk_sections(sections, {"strategy": "recursive", "chunk_size": 180, "overlap": 0})
                  if len(c.text.split()) >= 40]
    chunks = [c for c in all_chunks if not looks_like_quiz(c.text)]
    if len(all_chunks) - len(chunks):
        print(f"Skipping {len(all_chunks) - len(chunks)} passages that look like practice questions / answer choices")
    if not chunks:
        sys.exit("No usable passages: the documents look like quiz or exercise material only.")
    total_wanted = args.n + args.unanswerable
    pool = spread_sample(chunks, min(len(chunks), int(total_wanted * 1.6) + 5), rng)
    print(f"{len(sections)} sections, {len(chunks)} passages; drafting {args.n} answerable + {args.unanswerable} unanswerable")

    out_path = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    items: list[dict] = []
    if args.resume and out_path.exists():
        items = [json.loads(line) for line in out_path.read_text().splitlines() if line.strip()]
        print(f"Resuming with {len(items)} existing drafts")
    else:
        out_path.write_text("")
    asked = {norm(i["question"]) for i in items}
    idx = max([int(i["id"][1:]) for i in items if i["id"][1:].isdigit()] or [0])
    dropped = skipped = 0

    def save(item: dict) -> None:  # write as we go, so a crash never loses finished drafts
        items.append(item)
        with open(out_path, "a") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    for chunk in pool * 3:  # small document sets: revisit passages for a different question type
        n_ans = sum(i["answerable"] for i in items)
        n_un = len(items) - n_ans
        if n_ans >= args.n and n_un >= args.unanswerable:
            break
        want_answerable = n_ans < args.n and (n_un >= args.unanswerable or rng.random() < args.n / total_wanted)
        page = f", page {chunk.page}" if chunk.page else ""
        if want_answerable:
            qtype = TYPES[n_ans % len(TYPES)]
            prompt = ANSWERABLE_PROMPT.format(qtype=qtype, type_hint=TYPE_HINTS[qtype], source=chunk.source,
                                              page=page, text=chunk.text)
        else:
            prompt = UNANSWERABLE_PROMPT.format(source=chunk.source, page=page, text=chunk.text)
        try:
            data = parse_json(llm.chat([{"role": "user", "content": prompt}], json_mode=True)) or {}
        except Exception as e:
            print(f"  LLM error: {e}")
            dropped += 1
            continue
        if data.get("skip"):
            skipped += 1
            continue
        q = text(data.get("question"))
        if not q or norm(q) in asked or CONTEXT_BOUND.search(q):
            dropped += 1
            continue
        idx += 1
        asked.add(norm(q))
        if want_answerable:
            ev = text(data.get("evidence"))
            answer = text(data.get("answer"))
            if not ev or not answer or norm(ev) not in norm(chunk.text):
                dropped += 1  # evidence not verbatim -> likely hallucinated
                continue
            save({"id": f"q{idx:03d}", "type": qtype, "question": q, "answer": answer,
                  "source": chunk.source, "pages": [chunk.page] if chunk.page else [], "evidence": ev,
                  "answerable": True, "verified": False})
        else:
            save({"id": f"q{idx:03d}", "type": "unanswerable", "question": q, "answer": "",
                  "near_source": chunk.source, "answerable": False, "verified": False})
        print(f"  [{len(items):3d}] {items[-1]['type']:<12} {q[:90]}")

    n_ans = sum(i["answerable"] for i in items)
    print(f"\nWrote {len(items)} drafts ({n_ans} answerable, {len(items) - n_ans} unanswerable) to {out_path}")
    print(f"Discarded {dropped} drafts (evidence not copied exactly, question depends on the passage, "
          f"duplicate, or bad JSON); the model skipped {skipped} unsuitable passages.")
    print("Next: python eval/review_gold.py   (check every item by hand)")


if __name__ == "__main__":
    main()
