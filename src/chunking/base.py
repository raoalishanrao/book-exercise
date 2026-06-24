from abc import ABC, abstractmethod

from src.domain.models import Chunk, RawPage


class Chunker(ABC):
    """Single responsibility: turn raw book pages into structured chunks."""

    @abstractmethod
    def chunk(self, pages: list[RawPage]) -> list[Chunk]:
        ...
