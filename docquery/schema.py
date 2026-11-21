"""Core data structures shared by every stage of the pipeline."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Section:
    """A contiguous piece of a source document (one PDF page, one DOCX, one web page)."""
    doc_id: str
    source: str
    text: str
    page: int | None = None


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    source: str
    text: str
    page: int | None = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Chunk":
        return Chunk(**d)

    @property
    def label(self) -> str:
        return f"{self.source}" + (f", p. {self.page}" if self.page else "")


@dataclass
class Retrieved:
    chunk: Chunk
    score: float
    scores: dict = field(default_factory=dict)  # per-stage scores: dense, bm25, rrf, rerank


@dataclass
class Citation:
    number: int
    chunk: Chunk


@dataclass
class SentenceCheck:
    sentence: str
    citations: list[int]
    supported: bool
    similarity: float


@dataclass
class Answer:
    question: str
    standalone_question: str
    text: str
    abstained: bool
    citations: list[Citation]
    contexts: list[Retrieved]
    checks: list[SentenceCheck] = field(default_factory=list)
    timings: dict = field(default_factory=dict)

    @property
    def unsupported(self) -> list[SentenceCheck]:
        return [c for c in self.checks if not c.supported]
