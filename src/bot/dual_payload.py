"""Parse LLM dual-payload JSON: display_text (Roman) + speech_text (Urdu script)."""

import json
import logging
import re
from dataclasses import dataclass

from src.bot.math_format import normalize_physics_math
from src.voice.script_fix import URDU_ARABIC_RE, reply_to_roman_urdu
from src.voice.urdu_phrases import normalize_pakistani_urdu_script

logger = logging.getLogger("tutor.dual_payload")

_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


@dataclass(frozen=True)
class TutorReply:
    display_text: str
    speech_text: str

    @property
    def speech_ready(self) -> bool:
        return bool(self.speech_text.strip())


def dual_payload_enabled() -> bool:
    from src.config import optional_env

    return optional_env("DUAL_PAYLOAD_MODE", "true").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def format_display_text(text: str) -> str:
    """Roman Urdu for chat bubble — strip markdown, keep Latin."""
    if not text:
        return text
    text = normalize_physics_math(text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\1", text)
    text = re.sub(r"^[\t ]*\*[\t ]+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return reply_to_roman_urdu(text.strip())


def format_speech_text(text: str) -> str:
    """Urdu script for TTS — polish only, never Romanize."""
    if not text:
        return text
    text = normalize_physics_math(text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return normalize_pakistani_urdu_script(text.strip())


def _extract_json_object(raw: str) -> dict | None:
    cleaned = _JSON_FENCE_RE.sub("", raw.strip())
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        try:
            data = json.loads(cleaned[start : end + 1])
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            return None
    return None


def parse_dual_payload(raw: str) -> TutorReply:
    """Parse LLM JSON; fallback to single-field Roman reply."""
    data = _extract_json_object(raw)
    if data:
        display = str(data.get("display_text") or "").strip()
        speech = str(data.get("speech_text") or "").strip()
        if display and speech and URDU_ARABIC_RE.search(speech):
            logger.info(
                "Dual payload ok display=%s speech=%s chars",
                len(display),
                len(speech),
            )
            logger.info("LLM display_text:\n%s", display)
            logger.info("LLM speech_text:\n%s", speech)
            return TutorReply(
                display_text=format_display_text(display),
                speech_text=format_speech_text(speech),
            )
        if display and not speech:
            logger.warning("Dual payload missing speech_text — using display only")
            display = format_display_text(display)
            return TutorReply(display_text=display, speech_text="")

    logger.warning("Dual payload parse failed — treating reply as Roman display_text")
    display = format_display_text(raw)
    return TutorReply(display_text=display, speech_text="")


DUAL_PAYLOAD_PROMPT = """
JSON OUTPUT FORMAT (MANDATORY)
You must respond with ONE valid JSON object only. No markdown fences, no text outside JSON.

{
  "display_text": "Roman Urdu (Latin letters) + clear English physics terms. Shown in chat. Warm, casual, readable.",
  "speech_text": "Same message in Urdu Unicode script (Nastaliq) + English physics words in Latin where natural for the board. For TTS only."
}

Rules for display_text:
- Roman Urdu only (Latin letters). Never Urdu Arabic script here.
- English for physics: velocity, force, Newton, acceleration, formula, numerical, etc.
- Feminine teacher voice: sakti hon, karungi, samjhati hon.

Rules for speech_text:
- Pakistani Urdu script (ی ک ہ) for all conversational words.
- Keep physics terms in English Latin: gravity, force, velocity, acceleration, Newton, numerical, formula, km/h, m/s.
- Same meaning as display_text — not a summary, not longer.
- Greeting example: "السلام علیکم، بیٹا!" not "اسلم" or hyphenated forms.
- Use "اس میں" not Roman "us mein".

Examples:

{
  "display_text": "Shabaash beta! Bohot acha sawal hai. Gravity asal mein woh force hai jo har cheez ko zameen ki taraf kheenchti hai.",
  "speech_text": "شاباش بیٹا! بہت اچھا سوال ہے۔ gravity اصل میں وہ force ہے جو ہر چیز کو زمین کی طرف کھینچتی ہے۔"
}

{
  "display_text": "Chalo, ab hum numerical solve karte hain. Humein acceleration maloom karni hai.",
  "speech_text": "چلو، اب ہم numerical solve کرتے ہیں۔ ہمیں acceleration معلوم کرنی ہے۔"
}
"""
