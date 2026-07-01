"""Prosody helpers for Edge Urdu TTS (plain text only — edge-tts builds SSML internally)."""

import re

from src.config import optional_env

_URDU_SENTENCE_END = re.compile(r"(?<=[۔؟!])\s+")
_DISCOURSE_MARKERS = (
    "السلام علیکم",
    "بیٹا",
    "دیکھئے",
    "چلیں",
    "ٹھیک ہے",
    "بالکل",
    "سمجھیں",
    "یاد رکھیں",
    "مثال کے طور پر",
)


def combined_rate(speed: float) -> str:
    """Widget speed + optional calmer baseline (slower = more natural teacher pace)."""
    base_pct = int(optional_env("EDGE_TTS_BASE_RATE_PCT", "-10"))
    widget_pct = int(round((speed - 1.0) * 100))
    total = max(-30, min(10, base_pct + widget_pct))
    return f"{total:+d}%"


def pitch_string() -> str:
    # Slight lift sounds warmer and less flat on neural Urdu voices.
    raw = optional_env("EDGE_TTS_PITCH", "+3Hz")
    if re.match(r"^[+-]\d+Hz$", raw):
        return raw
    return "+3Hz"


def volume_string() -> str:
    raw = optional_env("EDGE_TTS_VOLUME", "-3%")
    if re.match(r"^[+-]\d+%$", raw):
        return raw
    return "-3%"


def polish_spoken_urdu(text: str) -> str:
    """Shape Urdu for natural spoken delivery on Edge Uzma."""
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s*([.!?])\s*", "۔ ", text)
    text = re.sub(r"۔+", "۔", text)
    text = text.replace(",", "،")
    text = re.sub(r"\s*،\s*", "، ", text)
    text = re.sub(r"،\s*،+", "،", text)
    text = re.sub(r":\s*،\s*", ": ", text)
    # Teacher-style markers get a breath pause (comma) when missing.
    for marker in _DISCOURSE_MARKERS:
        text = re.sub(rf"{re.escape(marker)}(?!\s*،)", f"{marker}،", text)
    # Avoid run-on lists — light pause before "اور" mid-sentence.
    text = re.sub(r"(\S)\s+اور\s+", r"\1، اور ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if text and text[-1] not in "۔؟!":
        text += "۔"
    return text


def _sentence_pause_chars() -> int:
    try:
        return max(80, min(220, int(optional_env("EDGE_TTS_SENTENCE_CHARS", "150"))))
    except ValueError:
        return 150


def split_urdu_speech_units(text: str, *, max_chars: int | None = None) -> list[str]:
    """Split into short spoken units so Uzma pauses naturally between phrases."""
    limit = max_chars or _sentence_pause_chars()
    cleaned = polish_spoken_urdu(text)
    if not cleaned:
        return []
    if len(cleaned) <= limit:
        return [cleaned]

    sentences = [s.strip() for s in _URDU_SENTENCE_END.split(cleaned) if s.strip()]
    units: list[str] = []
    for sentence in sentences:
        if len(sentence) <= limit:
            units.append(sentence)
            continue
        clauses = [c.strip() for c in re.split(r"(?<=،)\s+", sentence) if c.strip()]
        buf = ""
        for clause in clauses:
            candidate = f"{buf} {clause}".strip() if buf else clause
            if len(candidate) <= limit:
                buf = candidate
            else:
                if buf:
                    units.append(buf)
                buf = clause
        if buf:
            units.append(buf)
    return units or [cleaned]
