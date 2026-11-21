import numpy as np

from docquery.ingest.chunkers import chunk_sections, fixed_chunks, recursive_chunks, semantic_chunks, split_sentences
from docquery.ingest.embedder import HashEmbedder
from docquery.schema import Section

TEXT = ("Para one sentence one. Para one sentence two is here.\n\n"
        "Second paragraph talks about grading. It has a ten-point scale.\n\n" * 20)


def test_fixed_respects_size_and_overlap():
    chunks = fixed_chunks(" ".join(f"w{i}" for i in range(100)), size=30, overlap=10)
    assert all(len(c.split()) <= 30 for c in chunks)
    assert chunks[1].split()[0] == "w20"  # step = size - overlap
    assert chunks[-1].split()[-1] == "w99"


def test_recursive_keeps_size_bound():
    chunks = recursive_chunks(TEXT, size=50, overlap=5)
    assert len(chunks) > 1
    assert all(len(c.split()) <= 55 for c in chunks)


def test_semantic_chunks_cover_text():
    emb = HashEmbedder()
    chunks = semantic_chunks(TEXT, 60, emb.embed_documents)
    assert sum(len(c.split()) for c in chunks) == len(TEXT.split())


def test_split_sentences():
    assert split_sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]


def test_chunk_sections_ids_and_pages():
    secs = [Section("d1", "a.pdf", TEXT, page=3)]
    chunks = chunk_sections(secs, {"strategy": "recursive", "chunk_size": 50, "overlap": 5})
    assert chunks[0].chunk_id.startswith("d1-p3-c")
    assert all(c.page == 3 for c in chunks)
