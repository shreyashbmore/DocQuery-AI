"""REST API.  Run:  uvicorn docquery.api.main:app --reload   (docs at http://localhost:8000/docs)"""
from __future__ import annotations

import os
import shutil
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from docquery.config import ROOT, load_config
from docquery.pipeline import DocQuery

app = FastAPI(title="DocQuery AI", version="0.1.0",
              description="Question answering over documents with hybrid retrieval and verifiable citations.")


@lru_cache(maxsize=1)
def get_dq() -> DocQuery:
    return DocQuery(load_config(os.getenv("DOCQUERY_CONFIG")))


class Turn(BaseModel):
    role: str
    content: str


class AskRequest(BaseModel):
    question: str
    history: list[Turn] = []


@app.get("/health")
def health():
    dq = get_dq()
    return {"status": "ok", "chunks": len(dq.chunks), "embedder": dq.embedder.name, "llm": dq.cfg["llm"]["model"]}


@app.get("/documents")
def documents():
    return get_dq().documents()


@app.post("/ingest")
async def ingest(files: list[UploadFile] = File(...)):
    upload_dir = ROOT / "data" / "docs"
    upload_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for f in files:
        dest = upload_dir / Path(f.filename).name
        with open(dest, "wb") as out:
            shutil.copyfileobj(f.file, out)
        saved.append(dest)
    return get_dq().ingest(saved)


@app.post("/ask")
def ask(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(400, "Empty question")
    ans = get_dq().ask(req.question, [t.model_dump() for t in req.history])
    return {
        "answer": ans.text,
        "abstained": ans.abstained,
        "standalone_question": ans.standalone_question,
        "citations": [{"number": c.number, "source": c.chunk.source, "page": c.chunk.page,
                       "chunk_id": c.chunk.chunk_id, "text": c.chunk.text} for c in ans.citations],
        "unsupported_sentences": [c.sentence for c in ans.unsupported],
        "contexts": [{"chunk_id": r.chunk.chunk_id, "source": r.chunk.source, "page": r.chunk.page,
                      "scores": r.scores} for r in ans.contexts],
        "timings_ms": {k: round(v * 1000, 1) for k, v in ans.timings.items()},
    }
