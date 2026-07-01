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
    speech_text: str | None = None
    session_id: str


class MessageItem(BaseModel):
    role: str
    content: str
    created_at: str


class SynthesizeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=8000)
    speed: float = Field(default=1.0, ge=0.7, le=1.3)
    speech_ready: bool = False


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
        from src.voice.tts_service import tts_available

        return tts_available()
    except Exception:
        return False


def _tts_provider_label() -> str:
    try:
        from src.voice.tts_service import tts_available, tts_provider

        if not tts_available():
            return "disabled"
        provider = tts_provider()
        if provider == "openai":
            return "openai"
        if provider in {"edge", "microsoft"}:
            return "edge-urdu"
        if provider in {"mms", "urdu", "roman_urdu"}:
            return "mms-roman-urdu"
        if provider == "kokoro":
            return "kokoro"
        return provider
    except Exception:
        return "disabled"


def _voice_stack_available() -> bool:
    return _stt_available() or _tts_available()


@app.on_event("startup")
def _log_voice_stack_on_startup() -> None:
    stt = _stt_available()
    tts = _tts_available()
    label = _tts_provider_label()
    logger.info("Voice stack ready: stt=%s tts=%s tts_provider=%s", stt, tts, label)
    if tts and label == "edge-urdu":
        voice = optional_env("EDGE_TTS_VOICE_UR", "ur-PK-UzmaNeural")
        prep = optional_env("EDGE_TTS_ROMAN_PREP", "auto")
        logger.info("Edge TTS Urdu voice=%s roman_prep=%s", voice, prep)
        if prep == "mavkif":
            try:
                from src.voice.mavkif_transliterate import download_status, is_mavkif_cached, preload_mavkif_background

                if is_mavkif_cached():
                    logger.info(
                        "Mavkif transliteration model is cached locally — used for Roman Urdu TTS prep"
                    )
                else:
                    st = download_status()
                    if st["partial_bytes"] > 0:
                        logger.info(
                            "Mavkif resuming download: %.1f / %.1f MB (%.1f%%)",
                            st["partial_mb"],
                            st["expected_mb"],
                            st["percent"],
                        )
                    else:
                        logger.info("Mavkif download not started — will fetch ~%.0f MB", st["expected_mb"])
                    logger.info("LLM transliteration until download completes (background resume started)")
                    preload_mavkif_background()
            except Exception as exc:
                logger.warning("Mavkif status check failed: %s", exc)
        if prep == "mavkif" and optional_env("MAVKIF_PRELOAD", "false").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            try:
                from src.voice.mavkif_transliterate import preload_mavkif_model

                preload_mavkif_model()
            except Exception as exc:
                logger.error("Mavkif preload failed: %s", exc, exc_info=True)
    elif tts and label == "mms-roman-urdu":
        model = optional_env("URDU_TTS_MODEL", "facebook/mms-tts-urd-script_latin")
        logger.info("Roman Urdu TTS model=%s (loads on first Listen, or set URDU_TTS_PRELOAD=true)", model)
        if optional_env("URDU_TTS_PRELOAD", "false").lower() in {"1", "true", "yes", "on"}:
            try:
                from src.voice.mms_tts_service import preload_mms_model

                preload_mms_model()
            except Exception as exc:
                logger.error("MMS TTS preload failed: %s", exc, exc_info=True)


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
    tts_provider = _tts_provider_label()
    return {
        "available": stt,
        "stt": stt,
        "tts": tts,
        "stt_provider": "groq",
        "stt_language": optional_env("STT_LANGUAGE", "both"),
        "tts_provider": tts_provider,
        "languages": ["en", "ur", "both"],
        "note": (
            "Mic: speak English or Urdu — auto-detected."
            if stt and not tts
            else "STT: Groq Whisper. TTS: Edge Urdu + Mavkif transliteration (LLM fallback)."
            if stt and tts and tts_provider == "edge-urdu"
            else "STT: Groq Whisper. TTS: MMS Roman Urdu (Latin script, open-source)."
            if stt and tts and tts_provider == "mms-roman-urdu"
            else "STT: Groq Whisper. TTS: Kokoro local (Hindi voice, Roman Urdu)."
            if stt and tts and tts_provider == "kokoro"
            else "STT: Groq Whisper. TTS: OpenAI."
            if stt and tts and tts_provider == "openai"
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
            detail="Text-to-speech is unavailable. Install torch+transformers for MMS Urdu, or set HF_TOKEN / OPENAI_API_KEY.",
        )
    logger.info(
        "TTS synthesize provider=%s chars=%s speech_ready=%s",
        _tts_provider_label(),
        len(body.text),
        body.speech_ready,
    )
    try:
        from src.voice.tts_service import TTSService, tts_output_media_type

        wav = TTSService().synthesize(
            body.text,
            speed=body.speed,
            speech_ready=body.speech_ready,
        )
        return Response(
            content=wav,
            media_type=tts_output_media_type(),
            headers={
                "Content-Disposition": "inline",
                "Cache-Control": "no-store",
                "X-TTS-Playback": "1",
            },
        )
    except ValueError as exc:
        logger.warning("TTS synthesize rejected: %s", exc)
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
        result = get_bot().reply(
            session_id,
            body.message.strip(),
            get_scope(),
            content_unit_id=content_unit_id,
        )
        logger.info(
            "Chat ok session=%s reply_len=%s speech_len=%s",
            body.session_id,
            len(result.display_text),
            len(result.speech_text),
        )
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
    return ChatResponse(
        reply=result.display_text,
        speech_text=result.speech_text or None,
        session_id=body.session_id,
    )


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
