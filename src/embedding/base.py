from abc import ABC, abstractmethod

from src.domain.models import Chunk, EmbeddedChunk


class Embedder(ABC):
    """Single responsibility: produce vector embeddings for chunks."""

    @abstractmethod
    def embed_documents(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        ...

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        ...
