import re

URDU_SCRIPT_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")

_DISPLAY_MATH_RE = re.compile(r"\$\$[^$]+\$\$", re.DOTALL)
_INLINE_MATH_RE = re.compile(r"\$([^$]+)\$")

_ROMAN_URDU_HINTS = re.compile(
    r"\b("
    r"assalam|alaikum|beta|aap|ap|mein|hai|hain|ho|hon|kya|ke|ki|ko|ka|ki|se|aur|"
    r"main|mujhe|bata|sawal|madad|samajh|theek|nahi|jee|bilkul|poori|karungi|"
    r"sakti|chahiye|parhenge|numerical|graph|cheez|pooch|context"
    r")\b",
    re.IGNORECASE,
)

_ENGLISH_TTS_HINTS = re.compile(
    r"\b("
    r"the|you|your|can|help|what|how|why|english|newton|force|physics|chapter|"
    r"speed|velocity|distance|time|formula|step|given|calculation|result|graph|"
    r"numerical|km/h|m/s|meter|second"
    r")\b",
    re.IGNORECASE,
)

_DISPLAY_MATH_RE = re.compile(r"\$\$[^$]+\$\$", re.DOTALL)
_INLINE_MATH_RE = re.compile(r"\$([^$]+)\$")


def clean_for_speech(text: str) -> str:
    if not text:
        return ""
    text = _DISPLAY_MATH_RE.sub(" ", text)
    text = _INLINE_MATH_RE.sub(lambda m: m.group(1).replace("\\,", " "), text)
    text = re.sub(r"[*_#>`]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_script_segments(text: str) -> list[tuple[str, str]]:
    """Split text into (kokoro_lang_code, segment) pairs."""
    return split_kokoro_segments(text, mode="auto")


def split_kokoro_segments(text: str, *, mode: str = "roman_mixed") -> list[tuple[str, str]]:
    """Split text for Kokoro TTS.

    Modes:
      roman_mixed — Roman Urdu → Urdu script + Hindi voice; English terms → EN voice
      roman_hindi — all Latin → Hindi voice (often robotic on Roman Urdu)
      english     — American English voice for all text
      auto        — Arabic Urdu → Hindi, other Latin → English
    """
    if not text:
        return []

    if mode == "english":
        return [("a", text)]

    if mode == "roman_hindi":
        return [("h", text)]

    if mode == "roman_mixed":
        return _split_roman_mixed(text)

    pattern = re.compile(
        r"([\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+|[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+)"
    )
    segments: list[tuple[str, str]] = []
    for match in pattern.finditer(text):
        chunk = match.group(0).strip()
        if not chunk:
            continue
        lang = "h" if URDU_SCRIPT_RE.search(chunk) else "a"
        segments.append((lang, chunk))
    return segments or [("a", text)]


def _is_english_latin_chunk(chunk: str) -> bool:
    if URDU_SCRIPT_RE.search(chunk):
        return False
    en_hits = len(_ENGLISH_TTS_HINTS.findall(chunk))
    ur_hits = len(_ROMAN_URDU_HINTS.findall(chunk))
    if en_hits >= 2 and en_hits > ur_hits:
        return True
    if ur_hits >= 2:
        return False
    # Short tokens like "km/h", "Step 1", numbers with units
    if re.fullmatch(r"[\d\s./+\-=kmsh]+", chunk, re.IGNORECASE):
        return True
    return en_hits > ur_hits and len(chunk.split()) <= 6


def _latin_to_urdu_speech(chunk: str) -> str:
    from src.voice.script_fix import to_urdu_script

    converted = to_urdu_script(chunk)
    if converted and URDU_SCRIPT_RE.search(converted):
        return converted
    return chunk


def _split_roman_mixed(text: str) -> list[tuple[str, str]]:
    pattern = re.compile(
        r"([\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+|[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+)"
    )
    segments: list[tuple[str, str]] = []
    for match in pattern.finditer(text):
        chunk = match.group(0).strip()
        if not chunk:
            continue
        if URDU_SCRIPT_RE.search(chunk):
            segments.append(("h", chunk))
            continue
        if _is_english_latin_chunk(chunk):
            segments.append(("a", chunk))
        else:
            segments.append(("h", _latin_to_urdu_speech(chunk)))
    return segments or [("h", _latin_to_urdu_speech(text))]


def split_bilingual_raw(text: str) -> list[tuple[str, str]]:
    """Word-level Urdu/English split without transliteration."""
    if not text:
        return []
    if URDU_SCRIPT_RE.search(text):
        return split_kokoro_segments(text, mode="auto")

    segments: list[tuple[str, str]] = []
    current_lang: str | None = None
    current_words: list[str] = []

    def flush() -> None:
        nonlocal current_lang, current_words
        if not current_words or current_lang is None:
            return
        segments.append((current_lang, " ".join(current_words)))
        current_words = []

    for token in re.findall(r"\S+", text):
        lang = "a" if _is_english_latin_chunk(token) else "h"
        if current_lang is None:
            current_lang = lang
            current_words = [token]
        elif lang == current_lang:
            current_words.append(token)
        else:
            flush()
            current_lang = lang
            current_words = [token]
    flush()
    return segments or [("h", text)]


def split_bilingual_speech(text: str) -> list[tuple[str, str]]:
    """Word-level Urdu/English split — keeps physics terms in English voice."""
    if not text:
        return []
    segments = split_bilingual_raw(text)
    out: list[tuple[str, str]] = []
    for lang, chunk in segments:
        if lang == "h":
            out.append(("h", _latin_to_urdu_speech(chunk)))
        else:
            out.append(("a", chunk))
    return out
