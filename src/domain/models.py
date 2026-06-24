from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import UUID


class ChunkType(str, Enum):
    THEORY = "theory"
    DEFINITION = "definition"
    FORMULA = "formula"
    EXAMPLE = "example"
    PROBLEM_STATEMENT = "problem_statement"
    SOLUTION = "solution"
    HINT = "hint"
    SUMMARY = "summary"
    FIGURE_CAPTION = "figure_caption"


@dataclass(frozen=True)
class BookScope:
    class_id: UUID
    subject_id: UUID
    book_id: UUID


@dataclass
class RawPage:
    page_number: int
    text: str


@dataclass
class Chunk:
    content: str
    chunk_type: ChunkType
    chunk_index: int = 0
    chapter_number: int | None = None
    section_ref: str | None = None
    problem_number: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    topics: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddedChunk(Chunk):
    embedding: list[float] = field(default_factory=list)
    content_hash: str = ""


@dataclass
class RetrievedChunk:
    id: UUID
    content: str
    chunk_type: ChunkType
    similarity: float
    problem_number: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
