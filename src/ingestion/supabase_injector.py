import time
from uuid import UUID

import httpx
from postgrest.exceptions import APIError
from supabase import create_client

from src.config import require_env
from src.domain.models import BookScope, EmbeddedChunk
from src.ingestion.base import ChunkInjector

INJECT_BATCH_SIZE = 5
MAX_INJECT_RETRIES = 5


def _sanitize_text(text: str) -> str:
    """PostgreSQL text fields reject null bytes from PDF extraction."""
    return text.replace("\x00", "")


class SupabaseInjector(ChunkInjector):
    def __init__(self, url: str | None = None, key: str | None = None):
        self.client = create_client(
            url or require_env("SUPABASE_URL"),
            key or require_env("SUPABASE_SERVICE_ROLE_KEY"),
        )

    def start_job(self, book_id: UUID, source_file: str) -> UUID:
        row = (
            self.client.table("ingestion_jobs")
            .insert({"book_id": str(book_id), "source_file": source_file, "status": "pending"})
            .execute()
        )
        return UUID(row.data[0]["id"])

    def complete_job(self, job_id: UUID, total_chunks: int) -> None:
        self.client.table("ingestion_jobs").update(
            {
                "status": "completed",
                "total_chunks": total_chunks,
                "embedded_chunks": total_chunks,
                "completed_at": "now()",
            }
        ).eq("id", str(job_id)).execute()

    def inject(self, scope: BookScope, chunks: list[EmbeddedChunk]) -> int:
        rows = [
            {
                "class_id": str(scope.class_id),
                "subject_id": str(scope.subject_id),
                "book_id": str(scope.book_id),
                "chunk_type": c.chunk_type.value,
                "chunk_index": c.chunk_index,
                "content": _sanitize_text(c.content),
                "content_hash": c.content_hash,
                "problem_number": c.problem_number,
                "topics": c.topics,
                "page_start": c.page_start,
                "page_end": c.page_end,
                "metadata": c.metadata,
                "embedding": c.embedding,
                "embedding_model": "gemini-embedding-001",
            }
            for c in chunks
        ]

        total = len(rows)
        for i in range(0, total, INJECT_BATCH_SIZE):
            batch = rows[i : i + INJECT_BATCH_SIZE]
            self._upsert_with_retry(batch)
            done = min(i + INJECT_BATCH_SIZE, total)
            print(f"  Injected {done}/{total} chunks", flush=True)

        return total

    def _upsert_with_retry(self, batch: list[dict]) -> None:
        for attempt in range(MAX_INJECT_RETRIES):
            try:
                self.client.table("document_chunks").upsert(
                    batch, on_conflict="book_id,content_hash"
                ).execute()
                return
            except (
                httpx.WriteError,
                httpx.ReadError,
                httpx.ConnectError,
                httpx.RemoteProtocolError,
                APIError,
            ) as exc:
                if attempt < MAX_INJECT_RETRIES - 1:
                    wait = 2**attempt
                    print(f"  Inject retry in {wait}s ({exc.__class__.__name__})", flush=True)
                    time.sleep(wait)
                    continue
                raise
