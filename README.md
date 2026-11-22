# DocQuery AI

**Intelligent Question Answering over Documents.** You ask questions about your PDFs, Word files and web pages, and each answer cites the exact passage it came from.

DocQuery AI is a Retrieval-Augmented Generation (RAG) system built for verifiable answers:

- **Hybrid retrieval:** BM25 keyword search and dense embeddings are fused with Reciprocal Rank Fusion, then a cross-encoder re-ranks the results.
- **Citations on every sentence:** each claim links to its source chunk, and the UI highlights the passage on the PDF page.
- **Hallucination guard:** when the evidence is weak, a score gate abstains instead of guessing. A post-generation support check then flags sentences their citations don't back.
- **Conversational:** follow-up questions are rewritten into standalone queries before retrieval.
- **Open or hosted models:** local LLMs through Ollama, or any OpenAI-compatible API.
- **Measured:** a built-in evaluation harness and ablation runner report Recall@k, MRR, nDCG, abstention accuracy, answer F1 and latency.

## Architecture

```
Offline   documents -> parse (+OCR) -> chunk (fixed | recursive | semantic) -> embed -> vector store (memory | Qdrant)
Online    question (+history) -> rewrite -> BM25 + dense -> RRF -> cross-encoder re-rank -> score gate
          -> LLM (Ollama | OpenAI-compatible) -> citation parsing -> support check -> cited answer
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Local LLM (or skip this step and use configs/api_openai.yaml)
#    Install Ollama from https://ollama.com, then:
ollama pull llama3.1:8b

# 2. Index documents
cp ~/Downloads/*.pdf data/docs/
python -m docquery.cli ingest data/docs

# 3. Ask
python -m docquery.cli ask "What is the minimum attendance requirement?" -v
python -m docquery.cli chat

# 4. Web UI / REST API
streamlit run ui/app.py
uvicorn docquery.api.main:app --reload      # http://localhost:8000/docs
```

The first run downloads `BAAI/bge-small-en-v1.5` (embeddings) and `BAAI/bge-reranker-base` (re-ranker) from Hugging Face. That's about 400 MB, and it's cached after that.

**No-download smoke test:** `python -m docquery.cli -c configs/offline.yaml ingest eval/sample`, then run `ask` with the same `-c` flag. It uses a hashing embedder and an extractive "LLM", so it only checks that everything runs. The answers are not good.

**Hosted models:** copy `.env.example` to `.env`, set `OPENAI_API_KEY` (and optionally `OPENAI_BASE_URL` for Gemini, Groq and similar), then pass `-c configs/api_openai.yaml`.

## Configuration

`configs/default.yaml` sets every option. You can layer a file with `-c` and override single keys with `-o`:

```bash
python -m docquery.cli -o retrieval.mode=dense -o reranker.enabled=false ask "..."
python -m docquery.cli -o llm.model=qwen2.5:7b -o store.backend=qdrant ingest data/docs
```

| Key | Options |
| --- | --- |
| `chunking.strategy` | `fixed`, `recursive`, `semantic` |
| `chunking.overlap` / `overlap_mode` | words or a fraction (`0.2`) / `sentence`, `word` |
| `retrieval.dedup_threshold` | `0.8` (default), `null` to turn off |
| `embedding.provider` | `sentence-transformers`, `openai`, `hash` |
| `store.backend` | `memory`, `qdrant` (embedded, or `store.qdrant_url` for a server; see `docker-compose.yml`) |
| `retrieval.mode` | `dense`, `bm25`, `hybrid` |
| `reranker.enabled` | `true` / `false` (`model`: any cross-encoder) |
| `llm.provider` | `ollama`, `openai`, `extractive` |

Changing the embedder or the store needs a fresh `index_dir` (or delete the old index).

## Evaluation

**One command for everything:** `python eval/run_experiments.py --docs data/docs --gold eval/gold_qa.jsonl --llm qwen2.5:7b` runs all three studies and writes `docs/EXPERIMENTS.md`. See [docs/RUNNING_EXPERIMENTS.md](docs/RUNNING_EXPERIMENTS.md) for the full guide, including how to draft and review the gold question set with `eval/draft_gold.py` and `eval/review_gold.py`.

Individual tools:

1. Write `eval/gold_qa.jsonl`, one question per line. See `eval/gold_qa.sample.jsonl` for the format, including unanswerable questions (`"answerable": false`).
2. Run:

```bash
python eval/run_eval.py --gold eval/gold_qa.jsonl --retrieval-only     # fast
python eval/run_eval.py --gold eval/gold_qa.jsonl --out eval/results/run.csv
python eval/run_ablation.py --docs data/docs --gold eval/gold_qa.jsonl  # chunking x retrieval x re-ranker
python eval/run_ablation.py --docs data/docs --gold eval/gold_qa.jsonl --with-llm \
    --llm ollama:llama3.1:8b --llm openai:gpt-4o-mini                   # open vs hosted LLM
```

The ablation prints a Markdown table you can paste straight into the report. Results are saved in `eval/results/`.

**Chunk-overlap study.** Overlap stops a rule that crosses a chunk boundary from being split in two. Too much overlap, though, creates near-duplicate chunks that crowd the top-k. Common practice is 10–20% of the chunk size. To measure it on your own documents:

```bash
python eval/run_ablation.py --docs data/docs --gold eval/gold_qa.jsonl --chunking recursive \
    --overlaps 0 0.1 0.2 0.3 --overlap-modes word sentence --dedup off on --retrieval hybrid+rerank
```

- `--overlap-modes`: `word` carries a raw word window into the next chunk. `sentence` carries only whole sentences.
- `--dedup`: turns removal of near-duplicate results on or off (`retrieval.dedup_threshold`).
- `dup_rate@5`: the share of the top 5 results that repeat a higher-ranked one.

## Tests

```bash
pytest -q        # runs offline: no model downloads or API keys needed
```

## Project layout

```
docquery/
  ingest/     loaders (PDF/DOCX/TXT/HTML, OCR), chunkers, embedders
  store/      in-memory and Qdrant vector stores
  retrieve/   BM25, RRF fusion, cross-encoder re-ranker, hybrid retriever
  generate/   prompts, LLM providers, citations, score gate, support check
  chat/       conversational query rewriting
  api/        FastAPI app
  pipeline.py DocQuery: ingest() and ask()
  metrics.py  Recall@k, MRR, nDCG, token F1, numeric match
ui/app.py     Streamlit chat with highlighted PDF citations
eval/         evaluation and ablation scripts, sample gold set
tests/        pytest suite
report/       technical report
```

## Roadmap

- [x] Week 1: ingestion, three chunkers, dense baseline, metrics, evaluation harness
- [x] Week 2 (core): BM25 + RRF, re-ranker, local and API LLMs, citations, score gate, query rewriting
- [ ] Write the 100-question gold set on department documents
- [ ] Run the full ablation and the open vs API comparison
- [ ] RAGAS faithfulness / answer-relevancy scores
- [ ] Deploy the demo (Hugging Face Spaces), record a 3-minute video, write the report

## Author

Shreyash More
