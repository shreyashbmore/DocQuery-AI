import json

from docquery.pipeline import DocQuery
from docquery.ingest.embedder import HashEmbedder
from docquery.retrieve.reranker import LexicalReranker
from tests.conftest import ROOT, SAMPLE


def test_ingest_and_persist(dq, offline_cfg):
    assert len(dq.chunks) > 3
    again = DocQuery(offline_cfg, embedder=HashEmbedder(), reranker=LexicalReranker())
    assert set(again.chunks) == set(dq.chunks)
    assert len(again.store) == len(dq.chunks)


def test_reingest_replaces_chunks(dq):
    n = len(dq.chunks)
    dq.ingest([SAMPLE])
    assert len(dq.chunks) == n
    assert len(dq.store) == n


def test_answer_has_citations(dq):
    ans = dq.ask("What is the minimum attendance required for the end-semester examination?")
    assert not ans.abstained
    assert "75" in ans.text
    assert ans.citations and ans.citations[0].chunk.source == "sample_regulations.md"
    assert ans.checks and all(c.citations for c in ans.checks)


def test_abstains_when_answer_missing(dq):
    ans = dq.ask("What is the hostel fee for first-year students?")
    assert ans.abstained


def test_retrieval_modes(dq):
    q = "How many credits are needed for the degree?"
    for mode in ["dense", "bm25", "hybrid"]:
        dq.cfg["retrieval"]["mode"] = mode
        dq._retriever = None
        top = dq.retriever.retrieve(q)
        assert top, mode


def test_eval_harness(dq):
    from eval.run_eval import evaluate, load_gold

    summary, rows = evaluate(dq, load_gold(ROOT / "eval" / "gold_qa.sample.jsonl"))
    assert summary["recall@5"] >= 0.8
    assert summary["abstention_correct"] >= 0.8
    json.dumps(summary)
