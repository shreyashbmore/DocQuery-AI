"""Document loaders. Each returns a list of Sections, keeping page numbers for citations."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

from docquery.schema import Section

SUPPORTED = {".pdf", ".docx", ".txt", ".md", ".html", ".htm"}


def doc_id_for(path: Path) -> str:
    return hashlib.sha1(path.name.encode()).hexdigest()[:10]


def clean_text(text: str) -> str:
    text = text.replace("­", "")                 # soft hyphens
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)       # de-hyphenate line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pdf(path: Path, ocr: bool = False) -> list[Section]:
    try:
        import pymupdf as fitz
    except ImportError:  # older PyMuPDF
        import fitz

    sections = []
    did = doc_id_for(path)
    with fitz.open(path) as pdf:
        for i, page in enumerate(pdf, start=1):
            text = page.get_text("text")
            if ocr and len(text.strip()) < 20:
                try:  # scanned page -> Tesseract OCR via PyMuPDF (needs tesseract installed)
                    text = page.get_textpage_ocr(full=True).extractText()
                except Exception:
                    pass
            text = clean_text(text)
            if text:
                sections.append(Section(did, path.name, text, page=i))
    return sections


def load_docx(path: Path) -> list[Section]:
    import docx

    d = docx.Document(str(path))
    parts = [p.text for p in d.paragraphs if p.text.strip()]
    for table in d.tables:
        for row in table.rows:
            parts.append(" | ".join(c.text.strip() for c in row.cells))
    return [Section(doc_id_for(path), path.name, clean_text("\n\n".join(parts)))]


def load_html(path: Path) -> list[Section]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(path.read_text(errors="ignore"), "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return [Section(doc_id_for(path), path.name, clean_text(soup.get_text("\n")))]


def load_text(path: Path) -> list[Section]:
    text = path.read_text(errors="ignore")
    if path.suffix.lower() == ".md":
        text = re.sub(r"^#{1,6}\s*", "", text, flags=re.M)        # drop markdown heading marks
    return [Section(doc_id_for(path), path.name, clean_text(text))]


def load_document(path: str | Path, ocr: bool = False) -> list[Section]:
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".pdf":
        return load_pdf(path, ocr=ocr)
    if ext == ".docx":
        return load_docx(path)
    if ext in {".html", ".htm"}:
        return load_html(path)
    if ext in {".txt", ".md"}:
        return load_text(path)
    raise ValueError(f"Unsupported file type: {path.suffix} (supported: {sorted(SUPPORTED)})")


def iter_paths(paths: list[str | Path]) -> list[Path]:
    out = []
    for p in map(Path, paths):
        if p.is_dir():
            out += sorted(f for f in p.rglob("*") if f.suffix.lower() in SUPPORTED)
        elif p.suffix.lower() in SUPPORTED:
            out.append(p)
    return out
