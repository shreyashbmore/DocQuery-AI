# Running the DocQuery AI experiments

This guide takes you from an empty `data/docs/` folder to a committed experiment report (`docs/EXPERIMENTS.md`).

## 1. One-time setup

```bash
cd docquery-ai
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install Ollama from <https://ollama.com>, open it (or run `ollama serve`), then pull the model:

```bash
ollama pull qwen2.5:7b        # ~4.7 GB. Any other model works too, e.g. qwen3:8b or llama3.1:8b
ollama run qwen2.5:7b "Say OK"   # quick check
```

The first run of the scripts below also downloads the embedding and re-ranker models from Hugging Face (about 400 MB, cached afterwards).

## 2. Add documents

Copy 8–10 PDFs, DOCX or TXT files into `data/docs/`, for example the department rulebook, 2–3 syllabi and 3–4 research papers. Ask your HOD before using any internal documents. Everything runs locally, so nothing leaves your machine.

## 3. Build the gold question set (about 1–2 hours, mostly reviewing)

```bash
python eval/draft_gold.py --docs data/docs --n 80 --unanswerable 20 --llm qwen2.5:7b
python eval/review_gold.py
```

- `draft_gold.py` has Qwen write questions, answers and an exact evidence sentence from your documents. Drafts whose evidence is not copied word for word are thrown away. The output goes to `eval/gold_qa.draft.jsonl`.
- `review_gold.py` shows each draft. Press `a` to accept, `e` to edit, `d` to delete, `s` to skip or `q` to quit. Your progress is saved after every decision, and accepted items go to `eval/gold_qa.jsonl`.
- **Be strict.** Delete vague or trivial questions, fix wrong answers, and for unanswerable questions confirm the documents really don't contain the answer. A good target is 100 verified questions: 80 answerable and 20 unanswerable.

## 4. Run the experiments

Start with a quick check, which takes a few minutes and uses a smaller grid and 10 questions:

```bash
python eval/run_experiments.py --docs data/docs --gold eval/gold_qa.jsonl --llm qwen2.5:7b --quick --max-gen-questions 10
```

If that works, run the full suite:

```bash
python eval/run_experiments.py --docs data/docs --gold eval/gold_qa.jsonl --llm qwen2.5:7b
```

Optionally, compare two local models (or a hosted one; set `OPENAI_API_KEY` in `.env` first):

```bash
python eval/run_experiments.py --docs data/docs --gold eval/gold_qa.jsonl --llm qwen2.5:7b --llm llama3.1:8b
python eval/run_experiments.py --docs data/docs --gold eval/gold_qa.jsonl --llm qwen2.5:7b --llm openai:gpt-4o-mini
```

| Study | What it measures | Typical time on a laptop (100 questions) |
| --- | --- | --- |
| A: retrieval methods | 3 chunking strategies × dense / BM25 / hybrid / hybrid + re-rank | 5–15 min |
| B: chunk overlap | 0 / 10 / 20 / 30% overlap × word / sentence × dedup off / on | 5–15 min |
| C: end-to-end answers | Qwen answers every question with citations | 15–40 min per model |

Other useful flags:

- `--skip-generation` runs only studies A and B, with no LLM.
- `--max-gen-questions N` limits study C to N questions.
- `-o key=value` overrides any config setting, for example `-o chunking.chunk_size=300`.

## 5. What you get

```
docs/EXPERIMENTS.md                      latest report: setup, results tables, failure cases, how to reproduce
docs/experiments/<run_id>.md             a copy of each run's report
eval/results/run_<run_id>/
    run.log                              full log with timestamps
    config.yaml                          exact configuration used
    study_a.csv, study_b.csv             one row per configuration
    generation_<model>.csv               one row per question: retrieval scores, answer, timings
    report.md                            same as the report above
```

Before committing, open `docs/EXPERIMENTS.md` and fill in **Author's observations** at the bottom in your own words.

Then commit:

```bash
git add docs/ eval/results/run_* eval/gold_qa.jsonl
git commit -m "Add experiment results"
```

The `index/` folder (vector indexes) is git-ignored. It is rebuilt automatically whenever your documents or chunking settings change.

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `Cannot reach Ollama` | Open the Ollama app or run `ollama serve` |
| `Ollama models not found` | `ollama pull qwen2.5:7b` |
| Hugging Face download error | Check your internet connection. Models are cached in `~/.cache/huggingface` after the first download |
| Very slow generation | Use `--max-gen-questions 30`, or a smaller model (`qwen2.5:3b`) |
| Many "false refusal" failures | The confidence gate may be too strict for your documents: try `-o gate.min_rerank_score=0.05` and compare |
| Warning that items are not hand-verified | Finish `python eval/review_gold.py` first |
