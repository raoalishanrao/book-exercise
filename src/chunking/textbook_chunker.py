import re
from hashlib import sha256

from src.chunking.base import Chunker
from src.domain.models import Chunk, ChunkType, RawPage

# Patterns tuned for textbook PDFs (adjust per publisher layout)
CHAPTER_RE = re.compile(r"^Chapter\s+(\d+)[:\s]+(.+)$", re.I | re.M)
SECTION_RE = re.compile(r"^(\d+\.\d+)\s+(.+)$", re.M)
EXERCISE_RE = re.compile(
    r"^(?:Exercise|Problem|Q\.?)\s*(\d+(?:\.\d+)?)[:\.\)]\s*(.+)$",
    re.I | re.M,
)
EXAMPLE_RE = re.compile(r"^Example\s+(\d+(?:\.\d+)?)[:\.\)]\s*(.+)$", re.I | re.M)


class TextbookChunker(Chunker):
    """
    Chapter-aware chunking with separate problem/example blocks.
    Theory chunks use overlapping windows; problems stay atomic.
    """

    def __init__(self, theory_window: int = 800, theory_overlap: int = 120):
        self.theory_window = theory_window
        self.theory_overlap = theory_overlap

    def chunk(self, pages: list[RawPage]) -> list[Chunk]:
        full_text = "\n".join(p.text for p in pages)
        page_map = self._build_page_map(pages)
        chunks: list[Chunk] = []

        for block in self._split_blocks(full_text):
            block_type = block["type"]
            text = block["text"].strip()
            if not text:
                continue

            page_start, page_end = self._resolve_pages(text, page_map)

            if block_type == "theory":
                for i, window in enumerate(self._sliding_windows(text)):
                    chunks.append(
                        Chunk(
                            content=window,
                            chunk_type=ChunkType.THEORY,
                            chunk_index=i,
                            chapter_number=block.get("chapter"),
                            section_ref=block.get("section"),
                            page_start=page_start,
                            page_end=page_end,
                        )
                    )
            elif block_type == "example":
                chunks.append(
                    Chunk(
                        content=text,
                        chunk_type=ChunkType.EXAMPLE,
                        problem_number=block.get("ref"),
                        chapter_number=block.get("chapter"),
                        page_start=page_start,
                        page_end=page_end,
                    )
                )
            elif block_type == "exercise":
                chunks.append(
                    Chunk(
                        content=text,
                        chunk_type=ChunkType.PROBLEM_STATEMENT,
                        problem_number=block.get("ref"),
                        chapter_number=block.get("chapter"),
                        page_start=page_start,
                        page_end=page_end,
                        metadata={"has_solution": False},
                    )
                )

        return chunks

    def _split_blocks(self, text: str) -> list[dict]:
        """Split text into theory / example / exercise blocks."""
        blocks: list[dict] = []
        current_chapter: int | None = None
        current_section: str | None = None
        buffer: list[str] = []
        buffer_type = "theory"

        def flush():
            nonlocal buffer, buffer_type
            if buffer:
                blocks.append(
                    {
                        "type": buffer_type,
                        "text": "\n".join(buffer),
                        "chapter": current_chapter,
                        "section": current_section,
                        "ref": None,
                    }
                )
            buffer = []
            buffer_type = "theory"

        for line in text.splitlines():
            ch = CHAPTER_RE.match(line)
            if ch:
                flush()
                current_chapter = int(ch.group(1))
                continue

            sec = SECTION_RE.match(line)
            if sec:
                flush()
                current_section = sec.group(1)
                buffer = [line]
                continue

            ex = EXAMPLE_RE.match(line)
            if ex:
                flush()
                buffer_type = "example"
                buffer = [line]
                blocks.append(
                    {
                        "type": "example",
                        "text": line,
                        "chapter": current_chapter,
                        "section": current_section,
                        "ref": ex.group(1),
                    }
                )
                buffer = []
                buffer_type = "theory"
                continue

            prob = EXERCISE_RE.match(line)
            if prob:
                flush()
                blocks.append(
                    {
                        "type": "exercise",
                        "text": line,
                        "chapter": current_chapter,
                        "section": current_section,
                        "ref": prob.group(1),
                    }
                )
                continue

            buffer.append(line)

        flush()
        return blocks

    def _sliding_windows(self, text: str) -> list[str]:
        words = text.split()
        if len(words) <= self.theory_window:
            return [text]
        windows: list[str] = []
        step = max(1, self.theory_window - self.theory_overlap)
        for start in range(0, len(words), step):
            window = " ".join(words[start : start + self.theory_window])
            if window:
                windows.append(window)
        return windows

    @staticmethod
    def _build_page_map(pages: list[RawPage]) -> dict[str, int]:
        mapping: dict[str, int] = {}
        for page in pages:
            for line in page.text.splitlines():
                snippet = line.strip()[:80]
                if snippet:
                    mapping[snippet] = page.page_number
        return mapping

    @staticmethod
    def _resolve_pages(text: str, page_map: dict[str, int]) -> tuple[int | None, int | None]:
        pages: list[int] = []
        for line in text.splitlines():
            snippet = line.strip()[:80]
            if snippet in page_map:
                pages.append(page_map[snippet])
        if not pages:
            return None, None
        return min(pages), max(pages)

    @staticmethod
    def content_hash(content: str) -> str:
        return sha256(content.encode("utf-8")).hexdigest()
