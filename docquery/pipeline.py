"""DocQuery AI pipeline: ingest documents, then answer questions with cited, checked answers."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from docquery.chat.rewrite import standalone_question
from docquery.generate.answer import extract_citations, is_abstention, passes_gate, support_check
from docquery.generate.llm import build_llm
from docquery.generate.prompts import NOT_FOUND, answer_messages
from docquery.ingest.chunkers import chunk_sections
from docquery.ingest.embedder import build_embedder
from docquery.ingest.loaders import iter_paths, load_document
from docquery.retrieve.hybrid import Retriever
from docquery.retrieve.reranker import build_reranker
from docquery.schema import Answer, Chunk
from docquery.store.vector_store import build_store

log = logging.getLogger("docquery")


class DocQuery:
    def __init__(self, cfg: dict, embedder=None, llm=None, reranker="auto"):
        self.cfg = cfg
        self.index_dir = Path(cfg["index_dir"])
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder or build_embedder(cfg["embedding"])
        self._check_meta()
        self.store = build_store(cfg["store"], str(self.index_dir), self.embedder.dim)
        self.chunks: dict[str, Chunk] = self._load_chunks()
        self.reranker = build_reranker(cfg["reranker"]) if reranker == "auto" else reranker
        self._llm = llm
        self._retriever = None

    # ------------------------------------------------------------------ persistence
    @property
    def _chunks_path(self) -> Path:
        return self.index_dir / "chunks.jsonl"

    def _check_meta(self):
        meta_path = self.index_dir / "meta.json"
        meta = {"embedder": self.embedder.name, "dim": self.embedder.dim,
                "store": self.cfg["store"]["backend"], "chunking": self.cfg["chunking"]}
        if meta_path.exists():
            old = json.loads(meta_path.read_text())
            if old["embedder"] != meta["embedder"] or old.get("store") != meta["store"]:
                raise RuntimeError(f"Index at {self.index_dir} was built with {old['embedder']} / {old.get('store')}. "
                                   "Use a different index_dir or re-ingest with --reset.")
        else:
            meta_path.write_text(json.dumps(meta, indent=2))

    def _load_chunks(self) -> dict[str, Chunk]:
        if not self._chunks_path.exists():
            return {}
        with open(self._chunks_path) as f:
            return {c.chunk_id: c for c in (Chunk.from_dict(json.loads(line)) for line in f if line.strip())}

    def _save_chunks(self):
        with open(self._chunks_path, "w") as f:
            for c in self.chunks.values():
                f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------ components
    @property
    def llm(self):
        if self._llm is None:
            self._llm = build_llm(self.cfg["llm"])
        return self._llm

    @property
    def retriever(self) -> Retriever:
        if self._retriever is None:
            self._retriever = Retriever(self.chunks, self.store, self.embedder, self.cfg["retrieval"], self.reranker)
        return self._retriever

    def documents(self) -> list[dict]:
        docs: dict[str, dict] = {}
        for c in self.chunks.values():
            d = docs.setdefault(c.doc_id, {"doc_id": c.doc_id, "source": c.source, "chunks": 0, "pages": set()})
            d["chunks"] += 1
            if c.page:
                d["pages"].add(c.page)
        return [{**d, "pages": len(d["pages"])} for d in docs.values()]

    # ------------------------------------------------------------------ ingestion
    def ingest(self, paths: list[str | Path], ocr: bool = False) -> dict:
        files = iter_paths(paths)
        stats = {"files": 0, "chunks": 0, "skipped": []}
        embed_fn = self.embedder.embed_documents
        for path in files:
            try:
                sections = load_document(path, ocr=ocr)
            except Exception as e:  # keep going on a bad file
                log.warning("Skipping %s: %s", path, e)
                stats["skipped"].append(str(path))
                continue
            if not sections:
                stats["skipped"].append(str(path))
                continue
            doc_id = sections[0].doc_id
            stale = [cid for cid, c in self.chunks.items() if c.doc_id == doc_id]
            if stale:  # re-ingesting the same file replaces its chunks
                self.store.delete(stale)
                for cid in stale:
                    del self.chunks[cid]
            new = chunk_sections(sections, self.cfg["chunking"], embed=embed_fn)
            if not new:
                continue
            vectors = embed_fn([c.text for c in new])
            self.store.add([c.chunk_id for c in new], vectors)
            self.chunks.update({c.chunk_id: c for c in new})
            stats["files"] += 1
            stats["chunks"] += len(new)
            log.info("Indexed %s: %d chunks", path.name, len(new))
        self.store.save()
        self._save_chunks()
        self._retriever = None
        return stats

    # ------------------------------------------------------------------ question answering
    def ask(self, question: str, history: list[dict] | None = None) -> Answer:
        history = history or []
        t: dict[str, float] = {}
        t0 = time.perf_counter()

        query = question
        if self.cfg["chat"].get("rewrite_queries", True) and history:
            query = standalone_question(question, history, self.llm, int(self.cfg["chat"].get("max_history_turns", 4)))
        t["rewrite"] = time.perf_counter() - t0

        t1 = time.perf_counter()
        contexts = self.retriever.retrieve(query)
        t["retrieve"] = time.perf_counter() - t1

        ok, reason = passes_gate(contexts, self.cfg["gate"])
        if not ok:
            t["total"] = time.perf_counter() - t0
            log.info("Abstained before generation: %s", reason)
            return Answer(question, query, NOT_FOUND, True, [], contexts, [], t)

        t2 = time.perf_counter()
        text = self.llm.chat(answer_messages(query, contexts))
        t["generate"] = time.perf_counter() - t2

        abstained = is_abstention(text)
        citations = [] if abstained else extract_citations(text, contexts)
        checks = []
        if not abstained and self.cfg["support_check"].get("enabled", True):
            t3 = time.perf_counter()
            checks = support_check(text, contexts, self.embedder, float(self.cfg["support_check"]["min_similarity"]))
            t["support_check"] = time.perf_counter() - t3
        t["total"] = time.perf_counter() - t0
        return Answer(question, query, text, abstained, citations, contexts, checks, t)
