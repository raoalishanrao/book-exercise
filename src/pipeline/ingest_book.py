"""Orchestrator: wires chunker → embedder → injector (composition, not inheritance)."""

import pickle
import sys
from pathlib import Path
from uuid import UUID

from pypdf import PdfReader

from src.chunking.textbook_chunker import TextbookChunker
from src.config import require_env
from src.domain.models import BookScope, Chunk, EmbeddedChunk, RawPage
from src.embedding.gemini_embedder import GeminiEmbedder
from src.ingestion.supabase_injector import SupabaseInjector

CACHE_DIR = Path(__file__).resolve().parent.parent.parent / ".cache"


def extract_pages(pdf_path: Path) -> list[RawPage]:
    reader = PdfReader(str(pdf_path))
    return [
        RawPage(page_number=i + 1, text=page.extract_text() or "")
        for i, page in enumerate(reader.pages)
    ]


def _cache_path(book_id: UUID) -> Path:
    return CACHE_DIR / f"{book_id}_embedded.pkl"


def _chunks_cache_path(book_id: UUID) -> Path:
    return CACHE_DIR / f"{book_id}_chunks.pkl"


def _load_pickle(path: Path):
    if path.is_file():
        with path.open("rb") as f:
            return pickle.load(f)
    return None


def _save_pickle(path: Path, data) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(data, f)


def _embed_with_resume(
    chunks: list[Chunk], book_id: UUID, embedder: GeminiEmbedder
) -> list[EmbeddedChunk]:
    embedded: list[EmbeddedChunk] = _load_pickle(_cache_path(book_id)) or []
    start = len(embedded)

    if start:
        print(f"Resuming embed from chunk {start + 1}/{len(chunks)}", flush=True)
    if start >= len(chunks):
        return embedded

    batch_size = embedder.batch_size
    for batch_start in range(start, len(chunks), batch_size):
        batch = chunks[batch_start : batch_start + batch_size]
        vectors = embedder._embed_batch([c.content for c in batch])  # noqa: SLF001
        for chunk, vector in zip(batch, vectors):
            embedded.append(
                EmbeddedChunk(
                    **chunk.__dict__,
                    embedding=vector,
                    content_hash=TextbookChunker.content_hash(chunk.content),
                )
            )
        done = min(batch_start + batch_size, len(chunks))
        print(f"  Embedded {done}/{len(chunks)} chunks", flush=True)
        _save_pickle(_cache_path(book_id), embedded)
        if done < len(chunks):
            import time
            from src.embedding.gemini_embedder import BATCH_PAUSE_SEC
            time.sleep(BATCH_PAUSE_SEC)

    return embedded


def ingest_book(
    pdf_path: Path,
    scope: BookScope,
    *,
    chunker: TextbookChunker | None = None,
    embedder: GeminiEmbedder | None = None,
    injector: SupabaseInjector | None = None,
    skip_embed: bool = False,
) -> int:
    chunker = chunker or TextbookChunker()
    embedder = embedder or GeminiEmbedder()
    injector = injector or SupabaseInjector()

    job_id = injector.start_job(scope.book_id, pdf_path.name)

    embedded = _load_pickle(_cache_path(scope.book_id)) if skip_embed else None
    if embedded:
        print(f"Loaded {len(embedded)} embedded chunks from cache", flush=True)
    else:
        chunks = _load_pickle(_chunks_cache_path(scope.book_id))
        if not chunks:
            pages = extract_pages(pdf_path)
            print(f"Extracted {len(pages)} pages", flush=True)
            chunks = chunker.chunk(pages)
            _save_pickle(_chunks_cache_path(scope.book_id), chunks)
            print(f"Created {len(chunks)} chunks", flush=True)
        else:
            print(f"Loaded {len(chunks)} chunks from cache", flush=True)

        embedded = _embed_with_resume(chunks, scope.book_id, embedder)
        print("Embedding cache saved", flush=True)

    print("Injecting into Supabase...", flush=True)
    count = injector.inject(scope, embedded)
    injector.complete_job(job_id, count)
    return count


if __name__ == "__main__":
    book_id = UUID(require_env("BOOK_ID"))
    class_id = UUID(require_env("CLASS_ID"))
    subject_id = UUID(require_env("SUBJECT_ID"))
    pdf_path = Path(require_env("PDF_PATH"))
    skip_embed = "--inject-only" in sys.argv

    scope = BookScope(class_id=class_id, subject_id=subject_id, book_id=book_id)
    total = ingest_book(pdf_path, scope, skip_embed=skip_embed)
    print(f"Ingested {total} chunks")
