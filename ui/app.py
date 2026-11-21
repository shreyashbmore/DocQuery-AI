"""Streamlit chat UI.  Run:  streamlit run ui/app.py"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from docquery.cli import _load_env  # noqa: E402
from docquery.config import load_config  # noqa: E402
from docquery.pipeline import DocQuery  # noqa: E402

_load_env()
st.set_page_config(page_title="DocQuery AI", layout="wide")
DOCS_DIR = ROOT / "data" / "docs"


@st.cache_resource(show_spinner="Loading models...")
def get_dq(config_name: str) -> DocQuery:
    path = None if config_name == "default.yaml" else ROOT / "configs" / config_name
    return DocQuery(load_config(path))


def render_highlighted_page(source: str, page: int, passage: str):
    """Render the cited PDF page with the passage highlighted (PyMuPDF text search)."""
    try:
        import pymupdf as fitz
    except ImportError:
        import fitz

    path = DOCS_DIR / source
    if not path.exists() or path.suffix.lower() != ".pdf" or not page:
        return None
    with fitz.open(path) as pdf:
        pg = pdf[page - 1]
        words = passage.split()
        for i in range(0, min(len(words), 120), 6):  # search short phrases; long ones rarely match exactly
            for quad in pg.search_for(" ".join(words[i:i + 6]), quads=True):
                pg.add_highlight_annot(quad)
        return pg.get_pixmap(dpi=110).tobytes("png")


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("DocQuery AI")
    st.caption("Ask your documents. Every answer cites its source.")
    configs = sorted(p.name for p in (ROOT / "configs").glob("*.yaml"))
    start = os.getenv("DOCQUERY_CONFIG_NAME", "default.yaml")
    config_name = st.selectbox("Configuration", configs, index=configs.index(start) if start in configs else 0)
    dq = get_dq(config_name)
    st.caption(f"Embeddings: `{dq.embedder.name}`  \nLLM: `{dq.cfg['llm']['provider']}/{dq.cfg['llm']['model']}`  \n"
               f"Retrieval: `{dq.cfg['retrieval']['mode']}`" + (" + re-ranker" if dq.reranker else ""))

    uploads = st.file_uploader("Add documents", type=["pdf", "docx", "txt", "md", "html"], accept_multiple_files=True)
    if uploads and st.button("Index uploaded files", type="primary"):
        DOCS_DIR.mkdir(parents=True, exist_ok=True)
        paths = []
        for f in uploads:
            dest = DOCS_DIR / f.name
            dest.write_bytes(f.getbuffer())
            paths.append(dest)
        with st.spinner("Parsing, chunking and embedding..."):
            stats = dq.ingest(paths)
        st.success(f"Indexed {stats['files']} file(s), {stats['chunks']} chunks")

    st.subheader("Indexed documents")
    docs = dq.documents()
    if not docs:
        st.info("No documents yet. Upload some above.")
    for d in docs:
        st.markdown(f"- **{d['source']}** · {d['chunks']} chunks")
    show_debug = st.toggle("Show retrieval details")
    if st.button("Clear chat"):
        st.session_state.messages = []


# ------------------------------------------------------------------ chat
st.session_state.setdefault("messages", [])


def show_answer(ans_payload: dict, key: str):
    st.markdown(ans_payload["text"])
    if ans_payload["unsupported"]:
        st.warning("Weakly supported: " + " / ".join(s[:90] for s in ans_payload["unsupported"]))
    for c in ans_payload["citations"]:
        with st.expander(f"[{c['number']}] {c['source']}" + (f", page {c['page']}" if c["page"] else "")):
            st.markdown(f"> {c['text']}")
            if st.checkbox("Show page", key=f"{key}-{c['number']}"):
                img = render_highlighted_page(c["source"], c["page"], c["text"])
                if img:
                    st.image(img)
    if show_debug and ans_payload.get("debug"):
        st.json(ans_payload["debug"])


for i, m in enumerate(st.session_state.messages):
    with st.chat_message(m["role"]):
        if m["role"] == "assistant":
            show_answer(m["payload"], f"h{i}")
        else:
            st.markdown(m["content"])

if question := st.chat_input("Ask a question about your documents"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]
    with st.chat_message("assistant"):
        with st.spinner("Searching and reading..."):
            ans = dq.ask(question, history)
        payload = {
            "text": ans.text,
            "unsupported": [c.sentence for c in ans.unsupported],
            "citations": [{"number": c.number, "source": c.chunk.source, "page": c.chunk.page, "text": c.chunk.text}
                          for c in ans.citations],
            "debug": {"standalone_question": ans.standalone_question,
                      "timings_ms": {k: round(v * 1000) for k, v in ans.timings.items()},
                      "contexts": [{"source": r.chunk.label, **{k: round(v, 3) for k, v in r.scores.items()}}
                                   for r in ans.contexts]},
        }
        show_answer(payload, f"h{len(st.session_state.messages)}")
    st.session_state.messages.append({"role": "assistant", "content": ans.text, "payload": payload})
