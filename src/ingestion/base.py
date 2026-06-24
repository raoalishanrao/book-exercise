from abc import ABC, abstractmethod
from uuid import UUID

from src.domain.models import BookScope, EmbeddedChunk


class ChunkInjector(ABC):
    """Single responsibility: persist embedded chunks into the vector store."""

    @abstractmethod
    def inject(self, scope: BookScope, chunks: list[EmbeddedChunk]) -> int:
        ...

    @abstractmethod
    def start_job(self, book_id: UUID, source_file: str) -> UUID:
        ...

    @abstractmethod
    def complete_job(self, job_id: UUID, total_chunks: int) -> None:
        ...
