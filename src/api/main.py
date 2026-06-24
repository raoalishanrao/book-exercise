"""FastAPI server for the tutoring bot widget."""

import logging
import traceback
from pathlib import Path
from uuid import UUID

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.bot.tutoring_bot import GeminiTutoringBot
from src.config import require_env
from src.domain.models import BookScope
from src.ingestion.supabase_injector import SupabaseInjector

ROOT = Path(__file__).resolve().parent.parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("tutor.api")

app = FastAPI(title="Book Exercise Tutor API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_bot() -> GeminiTutoringBot:
    return GeminiTutoringBot()


_scope: BookScope | None = None


def get_scope() -> BookScope:
    global _scope
    if _scope is None:
        _scope = BookScope(
            class_id=UUID(require_env("CLASS_ID")),
            subject_id=UUID(require_env("SUBJECT_ID")),
            book_id=UUID(require_env("BOOK_ID")),
        )
    return _scope


class CreateSessionRequest(BaseModel):
    student_name: str | None = None
    title: str | None = "Physics study session"


class CreateSessionResponse(BaseModel):
    session_id: str
    class_label: str
    book_title: str
    subject_name: str


class ChatRequest(BaseModel):
    session_id: str
    message: str = Field(min_length=1, max_length=4000)
    content_unit_id: str | None = None


class ChatResponse(BaseModel):
    reply: str
    session_id: str


class MessageItem(BaseModel):
    role: str
    content: str
    created_at: str


@app.get("/")
def root():
    return FileResponse(ROOT / "widget.html")


@app.get("/widget.html")
def widget():
    return FileResponse(ROOT / "widget.html")


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/config")
def config():
    supabase = SupabaseInjector().client
    scope = get_scope()
    book = (
        supabase.table("books")
        .select("title, class_id, subject_id")
        .eq("id", str(scope.book_id))
        .single()
        .execute()
    )
    cls = (
        supabase.table("academic_classes")
        .select("label, grade")
        .eq("id", book.data["class_id"])
        .single()
        .execute()
    )
    subject = (
        supabase.table("subjects")
        .select("name")
        .eq("id", book.data["subject_id"])
        .single()
        .execute()
    )
    return {
        "class_id": str(scope.class_id),
        "book_id": str(scope.book_id),
        "subject_id": str(scope.subject_id),
        "class_label": cls.data["label"],
        "grade": cls.data["grade"],
        "book_title": book.data["title"],
        "subject_name": subject.data["name"],
    }


@app.post("/api/sessions", response_model=CreateSessionResponse)
def create_session(body: CreateSessionRequest):
    supabase = get_bot().supabase
    scope = get_scope()

    session = (
        supabase.table("conversation_sessions")
        .insert(
            {
                "class_id": str(scope.class_id),
                "book_id": str(scope.book_id),
                "subject_id": str(scope.subject_id),
                "title": body.title,
                "metadata": {"student_name": body.student_name} if body.student_name else {},
            }
        )
        .execute()
    )
    session_id = session.data[0]["id"]

    meta = config()
    return CreateSessionResponse(
        session_id=session_id,
        class_label=meta["class_label"],
        book_title=meta["book_title"],
        subject_name=meta["subject_name"],
    )


@app.post("/api/chat", response_model=ChatResponse)
def chat(body: ChatRequest):
    try:
        session_id = UUID(body.session_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid session_id") from exc

    content_unit_id = None
    if body.content_unit_id:
        try:
            content_unit_id = UUID(body.content_unit_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid content_unit_id") from exc

    try:
        logger.info("Chat session=%s message_len=%s", body.session_id, len(body.message))
        reply = get_bot().reply(
            session_id,
            body.message.strip(),
            get_scope(),
            content_unit_id=content_unit_id,
        )
        logger.info("Chat ok session=%s reply_len=%s", body.session_id, len(reply))
    except Exception as exc:
        logger.error(
            "Chat failed session=%s error=%s\n%s",
            body.session_id,
            exc,
            traceback.format_exc(),
        )
        raise HTTPException(
            status_code=503,
            detail=f"Tutor is temporarily unavailable. Please try again. ({exc})",
        ) from exc
    return ChatResponse(reply=reply, session_id=body.session_id)


@app.get("/api/sessions/{session_id}/messages", response_model=list[MessageItem])
def list_messages(session_id: str):
    rows = (
        get_bot()
        .supabase.table("conversation_messages")
        .select("role, content, created_at")
        .eq("session_id", session_id)
        .order("created_at")
        .execute()
    )
    return [
        MessageItem(role=r["role"], content=r["content"], created_at=r["created_at"])
        for r in (rows.data or [])
    ]
