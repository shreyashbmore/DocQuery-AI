"""Review drafted questions one by one: accept, edit, or delete.

  python eval/review_gold.py                                   # eval/gold_qa.draft.jsonl -> eval/gold_qa.jsonl
  python eval/review_gold.py --draft my_draft.jsonl --out eval/gold_qa.jsonl

Progress is saved after every decision, so you can stop (q) and resume later.
For unanswerable items, check the documents really do NOT contain the answer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def save(path: Path, items: list[dict]) -> None:
    path.write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items))


def edit_field(item: dict, key: str) -> None:
    new = input(f"  new {key} (enter = keep): ").strip()
    if new:
        item[key] = new


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--draft", default="eval/gold_qa.draft.jsonl")
    p.add_argument("--out", default="eval/gold_qa.jsonl")
    args = p.parse_args(argv)
    draft_path, out_path = ROOT / args.draft, ROOT / args.out

    drafts = load(draft_path)
    done = load(out_path)
    seen = {d["id"] for d in done} | {d["id"] for d in drafts if d.get("rejected")}
    todo = [d for d in drafts if d["id"] not in seen]
    print(f"{len(done)} already verified, {len(todo)} to review. Keys: [a]ccept [e]dit [d]elete [s]kip [q]uit\n")

    for item in todo:
        print("-" * 80)
        print(f"{item['id']}  ({item['type']})  source: {item.get('source') or item.get('near_source')} "
              f"pages: {item.get('pages', '')}")
        print(f"Q: {item['question']}")
        if item["answerable"]:
            print(f"A: {item['answer']}\nEvidence: {item['evidence']}")
        else:
            print("A: (should be refused - confirm the documents do not answer this)")
        while True:
            k = input("> ").strip().lower()
            if k == "a":
                item["verified"] = True
                done.append(item)
                save(out_path, done)
                break
            if k == "e":
                edit_field(item, "question")
                if item["answerable"]:
                    edit_field(item, "answer")
                    edit_field(item, "evidence")
                item["verified"] = True
                item["edited"] = True
                done.append(item)
                save(out_path, done)
                break
            if k == "d":
                item["rejected"] = True
                save(draft_path, drafts)
                break
            if k == "s":
                break
            if k == "q":
                print(f"Saved. {len(done)} verified items in {out_path}")
                return
    n_ans = sum(d["answerable"] for d in done)
    print(f"\nDone. {len(done)} verified ({n_ans} answerable, {len(done) - n_ans} unanswerable) in {out_path}")


if __name__ == "__main__":
    main()
