import logging
import time

import httpx
from google import genai
from google.genai.errors import ClientError

from src.chunking.textbook_chunker import TextbookChunker
from src.config import require_env
from src.domain.models import Chunk, EmbeddedChunk
from src.embedding.base import Embedder
from src.utils.google_errors import error_code, retry_delay_seconds

logger = logging.getLogger("tutor.embedding")

EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIM = 768
BATCH_SIZE = 8
BATCH_PAUSE_SEC = 7.0
MAX_RETRIES = 8
RETRYABLE_CODES = {429, 500, 503, 504}

RETRYABLE = (ClientError, httpx.RemoteProtocolError, httpx.ReadError, httpx.WriteError, httpx.ConnectError)


class EmbeddingQuotaError(RuntimeError):
    """Gemini embedding quota exhausted — use text-only search fallback."""


class GeminiEmbedder(Embedder):
    def __init__(self, api_key: str | None = None, batch_size: int = BATCH_SIZE):
        self.client = genai.Client(api_key=api_key or require_env("GEMINI_API_KEY"))
        self.batch_size = batch_size

    def embed_documents(self, chunks: list[Chunk], *, start_index: int = 0) -> list[EmbeddedChunk]:
        results: list[EmbeddedChunk] = []
        total = len(chunks)

        for start in range(start_index, total, self.batch_size):
            batch = chunks[start : start + self.batch_size]
            vectors = self._embed_batch([c.content for c in batch])
            for chunk, vector in zip(batch, vectors):
                results.append(
                    EmbeddedChunk(
                        **chunk.__dict__,
                        embedding=vector,
                        content_hash=TextbookChunker.content_hash(chunk.content),
                    )
                )
            done = min(start + self.batch_size, total)
            print(f"  Embedded {done}/{total} chunks", flush=True)
            if done < total:
                time.sleep(BATCH_PAUSE_SEC)

        return results

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.models.embed_content(
                    model=EMBEDDING_MODEL,
                    contents=texts,
                    config={
                        "task_type": "RETRIEVAL_DOCUMENT",
                        "output_dimensionality": EMBEDDING_DIM,
                    },
                )
                return [list(e.values) for e in response.embeddings]
            except ClientError as exc:
                code = error_code(exc)
                if code in RETRYABLE_CODES and attempt < MAX_RETRIES - 1:
                    wait = retry_delay_seconds(exc, attempt)
                    logger.info("Embed rate limited (code=%s), retry in %ss", code, wait)
                    time.sleep(wait)
                    continue
                raise
            except RETRYABLE as exc:
                if attempt < MAX_RETRIES - 1:
                    wait = min(60, 5 * (2**attempt))
                    logger.warning("Embed network error, retry in %ss: %s", wait, exc)
                    time.sleep(wait)
                    continue
                raise RuntimeError(f"Embedding failed after retries: {exc}") from exc
        raise RuntimeError("Embedding failed after retries")

    def embed_query(self, query: str) -> list[float]:
        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.models.embed_content(
                    model=EMBEDDING_MODEL,
                    contents=query,
                    config={
                        "task_type": "RETRIEVAL_QUERY",
                        "output_dimensionality": EMBEDDING_DIM,
                    },
                )
                return list(response.embeddings[0].values)
            except ClientError as exc:
                code = error_code(exc)
                if code in RETRYABLE_CODES and attempt < MAX_RETRIES - 1:
                    wait = retry_delay_seconds(exc, attempt)
                    logger.info("Query embed retry %s (code=%s) in %ss", attempt + 1, code, wait)
                    time.sleep(wait)
                    continue
                if code == 429:
                    raise EmbeddingQuotaError(
                        "Gemini embedding daily quota exceeded. Using keyword search fallback."
                    ) from exc
                raise
            except RETRYABLE:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(min(30, 3 * (2**attempt)))
                    continue
                raise
        raise RuntimeError("Query embedding failed after retries")
