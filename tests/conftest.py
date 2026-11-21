import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from docquery.config import load_config  # noqa: E402
from docquery.ingest.embedder import HashEmbedder  # noqa: E402
from docquery.pipeline import DocQuery  # noqa: E402
from docquery.retrieve.reranker import LexicalReranker  # noqa: E402

SAMPLE = ROOT / "eval" / "sample" / "sample_regulations.md"


@pytest.fixture
def offline_cfg(tmp_path):
    """Config that needs no model downloads, network or GPU."""
    return load_config(overrides=[
        f"index_dir={tmp_path / 'index'}", "embedding.provider=hash", "llm.provider=extractive",
        "reranker.model=lexical", "gate.min_rerank_score=0.3", "support_check.min_similarity=0.3",
        "chunking.chunk_size=80", "chunking.overlap=10",
    ])


@pytest.fixture
def dq(offline_cfg):
    d = DocQuery(offline_cfg, embedder=HashEmbedder(), reranker=LexicalReranker())
    d.ingest([SAMPLE])
    return d
