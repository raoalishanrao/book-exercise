from abc import ABC, abstractmethod
from uuid import UUID

from src.domain.models import BookScope, ChunkType, RetrievedChunk


class Retriever(ABC):
    """Single responsibility: fetch relevant chunks for a student query."""

    @abstractmethod
    def search(
        self,
        query: str,
        scope: BookScope,
        *,
        chunk_types: list[ChunkType] | None = None,
        match_count: int = 12,
        text_query: str | None = None,
    ) -> list[RetrievedChunk]:
        ...

    @abstractmethod
    def get_problem_context(
        self, content_unit_id: UUID, *, include_solution: bool = False
    ) -> list[RetrievedChunk]:
        ...
