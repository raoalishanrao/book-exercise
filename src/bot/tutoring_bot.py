from uuid import UUID

import logging

from src.bot.base import TutoringBot
from src.bot.dual_payload import TutorReply, dual_payload_enabled, parse_dual_payload
from src.bot.llm_client import ChatLLM
from src.bot.prompts import format_response, get_system_prompt
from src.domain.models import BookScope, ChunkType
from src.retrieval.hybrid_retriever import HybridRetriever

logger = logging.getLogger("tutor.bot")

HISTORY_LIMIT = 12


class GeminiTutoringBot(TutoringBot):
    def __init__(self, retriever: HybridRetriever | None = None, llm: ChatLLM | None = None):
        self.retriever = retriever or HybridRetriever()
        self.llm = llm or ChatLLM()
        self.supabase = self.retriever.injector.client

    def reply(
        self,
        session_id: UUID,
        student_message: str,
        scope: BookScope,
        *,
        content_unit_id: UUID | None = None,
    ) -> TutorReply:
        if content_unit_id:
            chunks = self.retriever.get_problem_context(content_unit_id, include_solution=False)
        else:
            chunks = self.retriever.search(
                student_message,
                scope,
                chunk_types=[
                    ChunkType.THEORY,
                    ChunkType.DEFINITION,
                    ChunkType.FORMULA,
                    ChunkType.EXAMPLE,
                    ChunkType.PROBLEM_STATEMENT,
                ],
                match_count=14,
            )

        context_block = "\n\n---\n".join(
            f"[{c.chunk_type.value}] {c.content}" for c in chunks
        )
        chunk_ids = [str(c.id) for c in chunks]
        logger.info("Retrieved %s chunks for session=%s", len(chunk_ids), session_id)

        history_block = self._load_history(session_id)
        user_prompt = (
            f"TEXTBOOK CONTEXT:\n{context_block}\n\n"
            f"CHAT HISTORY:\n{history_block}\n\n"
            f"STUDENT (latest message):\n{student_message}"
        )

        answer, model_used = self.llm.generate(
            get_system_prompt(),
            user_prompt,
            json_mode=dual_payload_enabled(),
        )
        logger.info(
            "LLM raw response model=%s chars=%s\n%s",
            model_used,
            len(answer),
            answer,
        )
        if dual_payload_enabled():
            result = parse_dual_payload(answer)
        else:
            result = TutorReply(
                display_text=format_response(answer),
                speech_text="",
            )
        self._save_messages(
            session_id,
            student_message,
            result,
            chunk_ids,
            model_used,
        )
        return result

    def _load_history(self, session_id: UUID) -> str:
        rows = (
            self.supabase.table("conversation_messages")
            .select("role, content")
            .eq("session_id", str(session_id))
            .order("created_at", desc=True)
            .limit(HISTORY_LIMIT)
            .execute()
        )
        messages = list(reversed(rows.data or []))
        if not messages:
            return "(no prior messages)"
        return "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in messages
        )

    def _save_messages(
        self,
        session_id: UUID,
        student_message: str,
        answer: TutorReply,
        chunk_ids: list[str],
        model_used: str,
    ) -> None:
        sid = str(session_id)
        base = {"session_id": sid, "model": model_used, "metadata": {}}

        self.supabase.table("conversation_messages").insert(
            {
                **base,
                "role": "student",
                "content": student_message,
                "retrieved_chunk_ids": [],
            }
        ).execute()

        assistant_meta: dict = {}
        if answer.speech_ready:
            assistant_meta["speech_text"] = answer.speech_text

        self.supabase.table("conversation_messages").insert(
            {
                **base,
                "role": "assistant",
                "content": answer.display_text,
                "retrieved_chunk_ids": chunk_ids,
                "metadata": assistant_meta,
            }
        ).execute()
