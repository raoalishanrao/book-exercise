import logging
from uuid import UUID

from src.domain.models import BookScope, ChunkType, RetrievedChunk
from src.embedding.gemini_embedder import EmbeddingQuotaError, GeminiEmbedder
from src.ingestion.supabase_injector import SupabaseInjector
from src.retrieval.base import Retriever

logger = logging.getLogger("tutor.retrieval")


class HybridRetriever(Retriever):
    """Semantic search via match_document_chunks RPC + keyword fallback."""

    def __init__(self, injector: SupabaseInjector | None = None, embedder: GeminiEmbedder | None = None):
        self.injector = injector or SupabaseInjector()
        self.embedder = embedder or GeminiEmbedder()

    def search(
        self,
        query: str,
        scope: BookScope,
        *,
        chunk_types: list[ChunkType] | None = None,
        match_count: int = 12,
        text_query: str | None = None,
    ) -> list[RetrievedChunk]:
        try:
            embedding = self.embedder.embed_query(query)
            return self._search_vector(
                embedding, scope, chunk_types=chunk_types, match_count=match_count, text_query=text_query or query
            )
        except EmbeddingQuotaError as exc:
            logger.warning("%s — falling back to text search", exc)
            return self._search_text_only(
                query, scope, chunk_types=chunk_types, match_count=match_count
            )

    def _search_vector(
        self,
        embedding: list[float],
        scope: BookScope,
        *,
        chunk_types: list[ChunkType] | None,
        match_count: int,
        text_query: str,
    ) -> list[RetrievedChunk]:
        params = {
            "query_embedding": embedding,
            "match_count": match_count,
            "filter_class_id": str(scope.class_id),
            "filter_book_id": str(scope.book_id),
            "filter_subject_id": str(scope.subject_id),
            "text_query": text_query,
        }
        if chunk_types:
            params["filter_chunk_types"] = [t.value for t in chunk_types]

        rows = self.injector.client.rpc("match_document_chunks", params).execute().data or []
        return self._rows_to_chunks(rows)

    def _search_text_only(
        self,
        query: str,
        scope: BookScope,
        *,
        chunk_types: list[ChunkType] | None,
        match_count: int,
    ) -> list[RetrievedChunk]:
        """Keyword search when Gemini embedding quota is exhausted."""
        q = (
            self.injector.client.table("document_chunks")
            .select("id, content, chunk_type, problem_number, metadata")
            .eq("class_id", str(scope.class_id))
            .eq("book_id", str(scope.book_id))
            .eq("subject_id", str(scope.subject_id))
            .text_search("content_tsv", query, options={"type": "plain", "config": "english"})
            .limit(match_count)
        )
        if chunk_types:
            q = q.in_("chunk_type", [t.value for t in chunk_types])

        rows = q.execute().data or []
        return [
            RetrievedChunk(
                id=UUID(r["id"]),
                content=r["content"],
                chunk_type=ChunkType(r["chunk_type"]),
                similarity=0.5,
                problem_number=r.get("problem_number"),
                metadata=r.get("metadata") or {},
            )
            for r in rows
        ]

    @staticmethod
    def _rows_to_chunks(rows: list[dict]) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(
                id=UUID(r["id"]),
                content=r["content"],
                chunk_type=ChunkType(r["chunk_type"]),
                similarity=r["similarity"],
                problem_number=r.get("problem_number"),
                metadata=r.get("metadata") or {},
            )
            for r in rows
        ]

    def get_problem_context(
        self, content_unit_id: UUID, *, include_solution: bool = False
    ) -> list[RetrievedChunk]:
        rows = (
            self.injector.client.rpc(
                "get_problem_context",
                {"p_content_unit_id": str(content_unit_id), "include_solution": include_solution},
            )
            .execute()
            .data
            or []
        )
        return [
            RetrievedChunk(
                id=UUID(r["chunk_id"]),
                content=r["content"],
                chunk_type=ChunkType(r["chunk_type"]),
                similarity=1.0,
            )
            for r in rows
        ]
