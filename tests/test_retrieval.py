from docquery.metrics import mrr_at_k, ndcg_at_k, recall_at_k, token_f1, numbers_match
from docquery.retrieve.bm25 import BM25Index
from docquery.retrieve.fusion import reciprocal_rank_fusion


def test_rrf_rewards_agreement():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]], k=60)
    assert {fused[0][0], fused[1][0]} == {"a", "b"}
    assert fused[-1][0] in {"c", "d"}


def test_bm25_finds_exact_term():
    idx = BM25Index(["1", "2"], ["course CS-301 covers compilers", "attendance rules for students"])
    assert idx.search("CS-301", 2)[0][0] == "1"


def test_metrics():
    flags = [False, True, False]
    assert recall_at_k(flags, 1) == 0 and recall_at_k(flags, 2) == 1
    assert mrr_at_k(flags) == 0.5
    assert 0 < ndcg_at_k(flags) < 1
    assert token_f1("160 credits [1]", "160 credits") == 1.0
    assert numbers_match("You need 160 credits [2].", "160 credits") == 1.0
    assert numbers_match("text", "no numbers") is None


def test_duplicate_rate_and_dedup():
    from docquery.metrics import containment, duplicate_rate
    from docquery.retrieve.hybrid import Retriever
    from docquery.schema import Chunk, Retrieved

    a = "students need 75 percent attendance to sit the exam"
    b = "rule: students need 75 percent attendance to sit the exam in every course"
    c = "the project carries 12 credits"
    assert containment(a, b) == 1.0
    assert duplicate_rate([a, b, c]) == 1 / 3
    res = [Retrieved(Chunk(str(i), "d", "s", t), 1.0) for i, t in enumerate([a, b, c])]
    kept = Retriever.deduplicate(res, 0.8, top_k=5)
    assert [r.chunk.chunk_id for r in kept] == ["0", "2"]
