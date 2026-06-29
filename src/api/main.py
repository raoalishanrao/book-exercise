"""FastAPI server for the tutoring bot widget."""

import logging
import os
import tempfile
import traceback
from pathlib import Path
from uuid import UUID

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from src.bot.tutoring_bot import GeminiTutoringBot
from src.config import optional_env, require_env
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


class SynthesizeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=8000)
    speed: float = Field(default=1.0, ge=0.7, le=1.3)


class TranscribeResponse(BaseModel):
    text: str
    language: str | None = None
    language_probability: float | None = None


def _stt_available() -> bool:
    try:
        from src.voice.stt_service import stt_available

        return stt_available()
    except Exception:
        return False


def _tts_available() -> bool:
    try:
        from huggingface_hub import InferenceClient  # noqa: F401
        from src.voice.tts_service import tts_available

        return tts_available()
    except Exception:
        return False


def _voice_stack_available() -> bool:
    return _stt_available() or _tts_available()


@app.get("/")
def root():
    return FileResponse(ROOT / "widget.html")


@app.get("/widget.html")
def widget():
    return FileResponse(ROOT / "widget.html")


@app.get("/api/health")
def health():
    return {"status": "ok", "voice": _voice_stack_available()}


@app.get("/api/voice/capabilities")
def voice_capabilities():
    stt = _stt_available()
    tts = _tts_available()
    return {
        "available": stt,
        "stt": stt,
        "tts": tts,
        "stt_provider": "groq",
        "stt_language": optional_env("STT_LANGUAGE", "both"),
        "tts_provider": "kokoro" if tts else "disabled",
        "languages": ["en", "ur", "both"],
        "note": (
            "Mic: speak English or Urdu — auto-detected."
            if stt and not tts
            else "STT uses Groq Whisper. TTS uses Hugging Face Inference (Kokoro via fal-ai)."
            if stt and tts
            else "Set GROQ_API_KEY for speech-to-text."
        ),
    }


@app.post("/api/voice/transcribe", response_model=TranscribeResponse)
async def voice_transcribe(
    audio: UploadFile = File(...),
    language: str | None = Form(default=None),
    output: str | None = Form(default=None),
):
    if not _stt_available():
        raise HTTPException(
            status_code=503,
            detail="Speech-to-text is unavailable. Set GROQ_API_KEY in .env.",
        )

    suffix = Path(audio.filename or "recording.webm").suffix or ".webm"
    tmp_path = None
    try:
        content = await audio.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty audio upload")
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        from src.voice.stt_service import STTService

        result = STTService().transcribe(tmp_path, language=language, output=output)
        if not result["text"]:
            raise HTTPException(status_code=422, detail="Could not understand the audio")
        return TranscribeResponse(**result)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Transcribe failed: %s\n%s", exc, traceback.format_exc())
        detail = "Could not process audio — speak 2+ seconds and try again."
        if "too short" in str(exc).lower():
            detail = "Audio too short or invalid — hold mic longer and speak clearly."
        raise HTTPException(status_code=422, detail=detail) from exc
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/api/voice/synthesize")
def voice_synthesize(body: SynthesizeRequest):
    if not _tts_available():
        raise HTTPException(
            status_code=503,
            detail="Kokoro TTS is disabled. Set KOKORO_TTS_ENABLED=true in .env to enable.",
        )
    try:
        from src.voice.tts_service import TTSService

        wav = TTSService().synthesize(body.text, speed=body.speed)
        return Response(content=wav, media_type="audio/wav")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Synthesize failed: %s\n%s", exc, traceback.format_exc())
        raise HTTPException(status_code=503, detail=f"Speech synthesis failed. ({exc})") from exc


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
