import logging
import re
import unicodedata

logger = logging.getLogger("tutor.voice.script_fix")

DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
URDU_ARABIC_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")
LATIN_RE = re.compile(r"[A-Za-z]")

_ROMAN_WORD_FIXES = (
    (r"\btuma\b", "tum"),
    (r"\baap\b", "ap"),
    (r"\bmeM\b", "mein"),
    (r"\bmem\b", "mein"),
    (r"\bphijiksa\b", "physics"),
    (r"\bphiziks\b", "physics"),
    (r"\bfijiks\b", "physics"),
    (r"\bfiziks\b", "physics"),
    (r"\bfijikis\b", "physics"),
    (r"\bmadada\b", "madad"),
    (r"\bsakate\b", "sakte"),
    (r"\bsakati\b", "sakti"),
    (r"\bsaktay\b", "sakte"),
    (r"\bsaktai\b", "sakti"),
    (r"\bkarata\b", "karta"),
    (r"\bkarati\b", "karti"),
    (r"\bkarate\b", "karte"),
    (r"\bkara\b", "kar"),
    (r"\bmatalik\b", "mutaliq"),
    (r"\bmatalika\b", "mutaliq"),
    (r"\bmutaliq\b", "mutaliq"),
    (r"\bkyose\b", "kaise"),
    (r"\bkya\b", "kya"),
    (r"\bnahin\b", "nahi"),
    (r"\bhoon\b", "hon"),
    (r"\bhai\b", "hai"),
    (r"\bmerI\b", "meri"),
    (r"\bmeri\b", "meri"),
    (r"\bmere\b", "meri"),
)

_ARABIC_ROMAN = {
    "فزکس": "physics",
    "فزیقس": "physics",
    "مدد": "madad",
    "میں": "mein",
    "میرے": "meri",
    "میری": "meri",
    "آپ": "ap",
    "تم": "tum",
    "کر": "kar",
    "سکتے": "sakte",
    "سکتی": "sakti",
    "ہو": "ho",
    "ہوں": "hon",
    "کیا": "kya",
    "کیسے": "kaise",
    "نہیں": "nahi",
    "متعلق": "mutaliq",
}


def _strip_diacritics(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def _latin_ratio(text: str) -> float:
    letters = len(LATIN_RE.findall(text))
    total = len(re.sub(r"\s+", "", text))
    return letters / total if total else 0.0


def _strip_arabic_diacritics(text: str) -> str:
    return re.sub(r"[\u064B-\u065F\u0670\u0610-\u061A\u06D6-\u06ED]", "", text)


def _devanagari_to_urdu(text: str) -> str:
    from aksharamukha.transliterate import process

    return process("Devanagari", "Urdu", text)


def _roman_to_urdu(text: str) -> str:
    from aksharamukha.transliterate import process

    return process("ITRANS", "Urdu", text.lower())


def to_urdu_script(text: str) -> str:
    """Normalize STT output to plain Urdu (Arabic script)."""
    if not text:
        return text

    cleaned = text.strip()
    try:
        if DEVANAGARI_RE.search(cleaned):
            logger.info("Transcript Devanagari -> Urdu script")
            cleaned = _devanagari_to_urdu(cleaned)
        elif _latin_ratio(cleaned) >= 0.55 and not URDU_ARABIC_RE.search(cleaned):
            logger.info("Transcript Roman -> Urdu script")
            cleaned = _roman_to_urdu(cleaned)
    except Exception as exc:
        logger.warning("Urdu script conversion failed: %s", exc)

    cleaned = _strip_arabic_diacritics(cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def is_mostly_english(text: str) -> bool:
    """True when Whisper output is primarily English (not Urdu script)."""
    cleaned = (text or "").strip()
    if not cleaned:
        return False
    if URDU_ARABIC_RE.search(cleaned) or DEVANAGARI_RE.search(cleaned):
        return False
    if _latin_ratio(cleaned) >= 0.72:
        return True
    english_hits = len(
        re.findall(
            r"\b(the|you|your|can|help|what|how|why|english|newton|force|physics|chapter|page|please|explain|tell|about|know)\b",
            cleaned,
            re.IGNORECASE,
        )
    )
    return english_hits >= 2 and _latin_ratio(cleaned) >= 0.45


def normalize_transcript(text: str, *, output_mode: str = "auto") -> str:
    """Format STT for the text box — keep English, convert Urdu per mode."""
    if not text:
        return text
    cleaned = text.strip()
    mode = (output_mode or "auto").lower()

    if is_mostly_english(cleaned):
        return cleaned

    if mode == "roman":
        roman = to_roman_urdu(cleaned)
        if _latin_ratio(roman) >= 0.45 and len(roman.split()) >= 3:
            return roman
        return to_urdu_script(cleaned)

    if mode == "urdu":
        return to_urdu_script(cleaned)

    if mode == "english":
        if is_mostly_english(cleaned):
            return cleaned
        roman = to_roman_urdu(cleaned)
        return roman if _latin_ratio(roman) >= 0.35 else cleaned

    # auto: English stays English; Urdu script stays Urdu (best fidelity from Whisper)
    if URDU_ARABIC_RE.search(cleaned) or DEVANAGARI_RE.search(cleaned):
        return to_urdu_script(cleaned)

    roman = to_roman_urdu(cleaned)
    if _latin_ratio(roman) >= 0.4:
        return roman
    return cleaned


def _devanagari_to_roman(text: str) -> str:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate

    return transliterate(text, sanscript.DEVANAGARI, sanscript.ITRANS)


def _urdu_script_to_roman(text: str) -> str:
    from aksharamukha.transliterate import process

    devanagari = process("Urdu", "Devanagari", text)
    return _devanagari_to_roman(devanagari)


def _apply_arabic_roman_map(text: str) -> str:
    for arabic, roman in _ARABIC_ROMAN.items():
        text = text.replace(arabic, roman)
    return text


def _scrub_leftover_arabic(text: str) -> str:
    text = text.replace("سکte", "sakte").replace("سکti", "sakti").replace("سکta", "sakta")
    if URDU_ARABIC_RE.search(text):
        text = URDU_ARABIC_RE.sub(" ", text)
        text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_roman_urdu(text: str, *, aggressive: bool = True) -> str:
    text = _apply_arabic_roman_map(_strip_diacritics(text.strip()))
    text = re.sub(r"[^\w\s.,!?;:'\"()\-/$]+", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    lowered = text.lower()
    if aggressive:
        for pattern, replacement in _ROMAN_WORD_FIXES:
            lowered = re.sub(pattern, replacement, lowered, flags=re.IGNORECASE)
    return _scrub_leftover_arabic(lowered)


def to_roman_urdu(text: str) -> str:
    if not text:
        return text

    cleaned = text.strip()
    if (
        _latin_ratio(cleaned) >= 0.55
        and not URDU_ARABIC_RE.search(cleaned)
        and not DEVANAGARI_RE.search(cleaned)
    ):
        return normalize_roman_urdu(cleaned, aggressive=False)

    try:
        if DEVANAGARI_RE.search(cleaned):
            logger.info("Transcript Devanagari -> Roman Urdu")
            cleaned = _devanagari_to_roman(cleaned)
        elif URDU_ARABIC_RE.search(cleaned):
            logger.info("Transcript Urdu script -> Roman Urdu")
            cleaned = _urdu_script_to_roman(cleaned)
    except Exception as exc:
        logger.warning("Roman Urdu conversion failed: %s", exc)

    result = normalize_roman_urdu(cleaned)
    if URDU_ARABIC_RE.search(result):
        result = normalize_roman_urdu(_apply_arabic_roman_map(result))
    return result


_TTS_URDU_FIXES: tuple[tuple[str, str], ...] = (
    (r"بارومीटर|بارومीٹر|بارومीٹر", "بارومیٹر"),
    (r"واٹر\s*بارومیٹر|water\s*barometer", "پانی کا بارومیٹر"),
    (r"فضائیدباؤ", "فضائی دباؤ"),
    (r"قدم-بی-قدم|قدم\s*بی\s*قدم", "قدم بہ قدم"),
    (r"\bک\s+(تصور|معنی|مطلب|طریقہ|استعمال)", r"کا \1"),
    (r"([\u0600-\u06FF])کا([\u0600-\u06FF])", r"\1 کا \2"),
)


def sanitize_urdu_tts_text(text: str) -> str:
    """Fix LLM/Mavkif glitches before Edge TTS (Devanagari leaks, merged words)."""
    from src.voice.urdu_phrases import normalize_pakistani_urdu_script

    if not text:
        return text
    out = normalize_pakistani_urdu_script(text)
    if DEVANAGARI_RE.search(out):
        logger.info("TTS: fixing Devanagari script leaks in Urdu text")
        out = DEVANAGARI_RE.sub(lambda m: _devanagari_to_urdu(m.group(0)), out)
        out = normalize_pakistani_urdu_script(out)
    for pattern, replacement in _TTS_URDU_FIXES:
        out = re.sub(pattern, replacement, out)
    out = normalize_pakistani_urdu_script(out)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def reply_to_roman_urdu(text: str) -> str:
    """Force tutor replies to plain Roman Urdu — never Urdu Arabic or Devanagari."""
    if not text:
        return text
    if not URDU_ARABIC_RE.search(text) and not DEVANAGARI_RE.search(text):
        return text
    lines = []
    for line in text.splitlines():
        if URDU_ARABIC_RE.search(line) or DEVANAGARI_RE.search(line):
            lines.append(to_roman_urdu(line))
        else:
            lines.append(line)
    return "\n".join(lines)
