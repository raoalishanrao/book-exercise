from abc import ABC, abstractmethod
from uuid import UUID

from src.bot.dual_payload import TutorReply
from src.domain.models import BookScope


class TutoringBot(ABC):
    """Single responsibility: converse with students using retrieved context."""

    @abstractmethod
    def reply(
        self,
        session_id: UUID,
        student_message: str,
        scope: BookScope,
        *,
        content_unit_id: UUID | None = None,
    ) -> TutorReply:
        ...
