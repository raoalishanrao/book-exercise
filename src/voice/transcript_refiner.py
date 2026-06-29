"""Post-process noisy Whisper output into clean Roman Urdu for the tutor chat."""

import logging
import re

from groq import Groq

from src.config import env_list, optional_env, require_env

logger = logging.getLogger("tutor.voice.transcript_refiner")

_ARABIC_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")

_SYSTEM_PROMPT = """You fix speech-to-text for Pakistani 9th-class Physics students talking to a tutor.
The input is a noisy Urdu/English transcript from Whisper. Return ONLY what the student meant to say.

Rules:
- Plain Roman Urdu in Latin letters (ap, meri, madad, parh rahi hon, etc.).
- Keep English science words when natural: physics, force, velocity, Newton, chapter, page, numerical.
- Fix obvious STT mistakes (fiiziyat -> physics, page not pajap, madad not madada).
- No Arabic script. No Devanagari. No quotes, labels, or explanation.
- If the input is unintelligible noise, return exactly: EMPTY"""


def refine_enabled() -> bool:
    return optional_env("STT_REFINE_ENABLED", "true").lower() in {"1", "true", "yes", "on"}


def _refine_models() -> list[str]:
    models: list[str] = []
    primary = optional_env("GROQ_STT_REFINE_MODEL", "llama-3.1-8b-instant")
    if primary:
        models.append(primary)
    for model in env_list("GROQ_STT_REFINE_FALLBACK_MODELS"):
        if model not in models:
            models.append(model)
    chat = optional_env("GROQ_CHAT_MODEL", "")
    if chat and chat not in models:
        models.append(chat)
    return models


def _looks_clean_roman(text: str) -> bool:
    if _ARABIC_RE.search(text) or _DEVANAGARI_RE.search(text):
        return False
    words = re.findall(r"[a-zA-Z']+", text)
    return len(words) >= 2


def refine_transcript(raw: str) -> str:
    """Turn Whisper output into production-ready Roman Urdu."""
    cleaned = (raw or "").strip()
    if not cleaned or not refine_enabled():
        return cleaned

    client = Groq(api_key=require_env("GROQ_API_KEY"))
    user_prompt = f"Whisper transcript:\n{cleaned}"

    for model in _refine_models():
        try:
            logger.info("Refining transcript model=%s raw_len=%s", model, len(cleaned))
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                max_tokens=256,
            )
            text = (response.choices[0].message.content or "").strip()
            if text.upper() == "EMPTY" or not text:
                logger.info("Refiner returned empty for raw=%r", cleaned[:120])
                return ""
            text = text.strip("\"'")
            if _looks_clean_roman(text):
                logger.info("Refined raw=%r -> %r", cleaned[:120], text[:120])
                return text
            logger.warning("Refiner output not clean Roman, trying next model: %r", text[:80])
        except Exception as exc:
            logger.warning("Refine failed model=%s: %s", model, exc)

    return cleaned
