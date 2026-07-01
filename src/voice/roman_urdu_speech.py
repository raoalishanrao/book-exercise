"""Prepare Roman Urdu tutor text for natural Urdu neural TTS."""

import hashlib
import logging
import re

from src.config import optional_env
from src.voice.edge_ssml import polish_spoken_urdu
from src.voice.physics_speech import (
    apply_physics_glossary,
    contains_physics_english,
    finalize_urdu_speech,
    has_bad_physics_transliteration,
    keep_english_physics_terms,
    protect_english_physics,
    protect_urdu_physics,
    restore_english_physics,
    restore_urdu_physics,
    strip_physics_english_for_ratio,
    use_hybrid_physics_tts,
)
from src.voice.script_fix import URDU_ARABIC_RE, to_urdu_script
from src.voice.text_clean import URDU_SCRIPT_RE, split_bilingual_raw

logger = logging.getLogger("tutor.voice.roman_urdu_speech")

_LATIN_RE = re.compile(r"[A-Za-z]")
_ENG_TOKEN_RE = re.compile(r"ZZENG\d{3}ZZ")
_URDU_TOKEN_RE = re.compile(r"ZZURD\d{3}ZZ")
_CHUNK_CACHE: dict[str, list[str]] = {}
_HYBRID_CACHE: dict[str, list[tuple[str, str]]] = {}
_CACHE_MAX = 64


def _tts_chunk_chars() -> int:
    try:
        return max(120, min(600, int(optional_env("EDGE_TTS_CHUNK_CHARS", "380"))))
    except ValueError:
        return 380


def _prep_chunk_chars() -> int:
    try:
        return max(200, min(900, int(optional_env("EDGE_TTS_PREP_CHUNK_CHARS", "480"))))
    except ValueError:
        return 480

_SYSTEM_PROMPT_ENGLISH_PHYSICS = """You rewrite Pakistani 9th-class PHYSICS tutor messages for VOICE (dual Urdu + English TTS).

Input: Roman Urdu (Latin) + English physics terms.

Output rules:
1. Convert conversational Roman Urdu to Urdu Arabic script (بیٹا، آپ، میں، کر سکتی ہوں).
2. KEEP all physics terms in Latin exactly: velocity, acceleration, Newton, force, inertia,
   kinetic energy, potential energy, mass, friction, gravity, formula, joule, watt, km/h, m/s.
3. Translate other English words to Urdu: specific→مخصوص, topic→موضوع, chapter→باب, example→مثال.
4. Short spoken sentences. Commas ، for pauses. End with ۔
5. Keep tokens like ZZENG000ZZ exactly as-is.

Example:
بیٹا، acceleration وقت کے ساتھ velocity کی تبدیلی ہے۔

Output ONLY the spoken text."""

_SYSTEM_PROMPT_URDU_PHYSICS = """You rewrite Pakistani 9th-class PHYSICS tutor messages for VOICE (Edge TTS Urdu).

Input: Roman Urdu (Latin) + English physics terms.

Output rules:
1. ONLY Pakistani Urdu Arabic script — use ی ک ہ (not Arabic ي ك ة). Never mix MSA Arabic.
2. Greeting: السلام علیکم (never اسلم or اسلم-او-الیکم). Casual: سلام.
3. NEVER leave English-looking Roman in output: us→اس (us mein→اس میں), is→اس, un/in→ان, to→تو, or→یا/یا پھر. Use Urdu script for particles (میں، کو، کا، سے، پر).
4. Feminine polite teacher: بیٹا، آپ، میں، کر سکتی ہوں، سمجھا سکتی ہوں.
5. Short spoken sentences (8–14 words). One idea per sentence. Commas ، for breath pauses. End with ۔
6. Sound like a calm live teacher — دیکھئے، بیٹا، چلیں، ٹھیک ہے، بالکل (with commas).
7. Avoid long lists; break into separate short sentences.

MANDATORY physics vocabulary (use exactly):
- kinetic energy → حرکی توانائی
- potential energy → ممکناتی توانائی
- energy → توانائی
- force → قوت
- velocity / speed → رفتار
- acceleration → تعجیل
- distance → فاصلہ
- displacement → نقلِ و حرکت
- mass → کمیت
- weight → وزن
- gravity → کششِ ثقل
- work → کام
- power → طاقت
- momentum → حرکت کا زور
- inertia → جمود
- friction → اصطحکاک
- pressure → دباؤ
- physics → فزکس
- newton → نیوٹن
- joule → جول
- watt → واٹ
- km/h → کلومیٹر فی گھنٹہ
- m/s → میٹر فی سیکنڈ
- numerical → عددی مسئلہ
- formula → فارمولا
- graph → گراف
- chapter → باب

Common English (translate to Urdu — Uzma cannot read Latin):
specific→مخصوص, topic→موضوع, concept→تصور, question→سوال, example→مثال,
exercise→مشق, help→مدد, exam→امتحان, important→اہم, related→متعلق,
definition→تعریف, practice→مشق, summary→خلاصہ

ZERO Latin letters in final output.
Keep tokens like ZZURD000ZZ exactly as-is — do not translate or remove them.
Output ONLY the spoken Urdu text."""


def _system_prompt() -> str:
    if use_hybrid_physics_tts():
        return _SYSTEM_PROMPT_ENGLISH_PHYSICS
    if keep_english_physics_terms():
        return _SYSTEM_PROMPT_ENGLISH_PHYSICS
    return _SYSTEM_PROMPT_URDU_PHYSICS


def _prep_mode() -> str:
    return optional_env("EDGE_TTS_ROMAN_PREP", "auto").lower()


def _prefer_llm_prep() -> bool:
    """LLM produces cleaner Urdu for Edge Uzma; Mavkif is the fast offline fallback."""
    mode = _prep_mode()
    if mode in {"llm", "gemini", "groq", "quality"}:
        return True
    if mode in {"mavkif", "local", "fast"}:
        return False
    return _use_llm_fallback()


def _has_leaked_tokens(text: str) -> bool:
    return bool(_ENG_TOKEN_RE.search(text) or _URDU_TOKEN_RE.search(text))


_MAVKIF_ARTIFACT_RE = re.compile(
    r"بیت[۔،\s]|اسلم|اسلم-|مےء|ءاسر|آپک\s|جیہون|عليكم|"
    r"[\u0600-\u06FF]-[\u0600-\u06FF][\u0600-\u06FF\-]*"
)


def _has_mavkif_artifacts(text: str) -> bool:
    return bool(_MAVKIF_ARTIFACT_RE.search(text))


def _is_good_hybrid_conversion(result: str, source: str) -> bool:
    """Mixed Urdu + English physics is valid for hybrid TTS."""
    if not result or not _looks_like_urdu_script(result):
        return False
    if _has_leaked_tokens(result):
        return False
    if has_bad_physics_transliteration(result):
        return False
    return len(result.strip()) >= max(2, len(source.strip()) // 10)


def _is_good_conversion(result: str, source: str) -> bool:
    if use_hybrid_physics_tts() or keep_english_physics_terms():
        return _is_good_hybrid_conversion(result, source)
    return _is_good_urdu_conversion(result, source)


def _use_mavkif(text: str, *, original: str | None = None) -> bool:
    mode = _prep_mode()
    if mode in {"0", "false", "no", "off", "none", "llm"}:
        return False
    from src.voice.mavkif_transliterate import is_mavkif_cached, mavkif_available

    if not mavkif_available():
        return False
    if optional_env("MAVKIF_USE_ONLY_IF_CACHED", "true").lower() in {
        "1",
        "true",
        "yes",
        "on",
    } and not is_mavkif_cached():
        logger.info("Mavkif not cached yet — using LLM transliteration for this request")
        return False

    if keep_english_physics_terms() or use_hybrid_physics_tts():
        logger.info("Using Mavkif for Roman Urdu (English physics terms protected)")
        return True

    # Urdu mode: glossary maps physics to Urdu; Mavkif transliterates Roman, keeps Urdu blocks.
    if contains_physics_english(text):
        logger.info("Untranslated physics Latin in prep — using LLM for TTS prep")
        return False
    return True


def _use_llm_fallback() -> bool:
    mode = _prep_mode()
    if mode in {"0", "false", "no", "off", "none"}:
        return False
    return bool(optional_env("GEMINI_API_KEY") or optional_env("GROQ_API_KEY"))


def prepare_roman_tts_text(text: str) -> str:
    """Roman phrase glossary + physics glossary before transliteration."""
    from src.voice.urdu_phrases import apply_roman_phrase_glossary

    return apply_physics_glossary(apply_roman_phrase_glossary(text))


def _cache_key(text: str) -> str:
    if use_hybrid_physics_tts():
        mode = "hybrid"
    elif keep_english_physics_terms():
        mode = "en"
    else:
        mode = "ur"
    return hashlib.sha256(f"v18|{mode}|{text}".encode("utf-8")).hexdigest()


def _looks_like_urdu_script(text: str) -> bool:
    return bool(URDU_ARABIC_RE.search(text))


def _latin_ratio(text: str) -> float:
    letters = len(_LATIN_RE.findall(text))
    total = len(re.sub(r"\s+", "", text))
    return letters / total if total else 0.0


def _latin_ratio_conversational(text: str) -> float:
    stripped = strip_physics_english_for_ratio(text)
    letters = len(_LATIN_RE.findall(stripped))
    total = len(re.sub(r"\s+", "", stripped))
    return letters / total if total else 0.0


def _is_good_urdu_conversion(result: str, source: str) -> bool:
    if not result or not _looks_like_urdu_script(result):
        return False
    if _has_leaked_tokens(result):
        return False
    if keep_english_physics_terms():
        if _latin_ratio_conversational(result) > 0.15:
            return False
    else:
        if _latin_ratio(result) > 0.06:
            return False
        if contains_physics_english(result):
            return False
    if has_bad_physics_transliteration(result):
        return False
    if _has_mavkif_artifacts(result):
        return False
    return len(result.strip()) >= max(2, len(source.strip()) // 8)


def _trust_llm_output(result: str, source: str) -> bool:
    """Prefer LLM output over aksharamukha fallback when it is usable Urdu."""
    if not result or not _looks_like_urdu_script(result) or _has_leaked_tokens(result):
        return False
    if len(result.strip()) < max(2, len(source.strip()) // 10):
        return False
    if has_bad_physics_transliteration(result):
        return False
    if _has_mavkif_artifacts(result):
        return False
    if keep_english_physics_terms():
        return _latin_ratio_conversational(result) < 0.22
    return _latin_ratio(result) < 0.10


def _mavkif_convert_mixed(text: str) -> str:
    from src.voice.mavkif_transliterate import transliterate_roman_urdu_long
    from src.voice.script_fix import normalize_roman_urdu

    if keep_english_physics_terms():
        prepared, placeholders = protect_english_physics(text)
        restore = lambda s: restore_english_physics(s, placeholders)
    else:
        prepared = prepare_roman_tts_text(text)
        placeholders: dict[str, str] = {}
        restore = lambda s: s

    if _ENG_TOKEN_RE.search(prepared) or _URDU_TOKEN_RE.search(prepared):
        segments = split_bilingual_raw(prepared)
    elif not _LATIN_RE.search(prepared):
        return finalize_urdu_speech(restore(prepared.strip()))
    elif not URDU_SCRIPT_RE.search(prepared):
        roman = normalize_roman_urdu(prepared, aggressive=False)
        merged = restore(transliterate_roman_urdu_long(roman))
        return finalize_urdu_speech(merged)
    else:
        segments = _split_mixed_script(prepared)

    parts: list[str] = []
    for lang, chunk in segments:
        if not chunk.strip():
            continue
        if _ENG_TOKEN_RE.search(chunk) or _URDU_TOKEN_RE.search(chunk):
            parts.append(chunk)
        elif lang == "ur" or URDU_SCRIPT_RE.search(chunk):
            parts.append(chunk)
        elif _LATIN_RE.search(chunk):
            roman = normalize_roman_urdu(chunk, aggressive=False)
            parts.append(transliterate_roman_urdu_long(roman))
        else:
            parts.append(chunk)
    merged = restore(" ".join(p for p in parts if p).strip())
    return finalize_urdu_speech(merged)


def _llm_convert(text: str) -> str:
    from src.bot.llm_client import ChatLLM

    if keep_english_physics_terms():
        prepared, placeholders = protect_english_physics(text)
    else:
        prepared = prepare_roman_tts_text(text)
        prepared, placeholders = protect_urdu_physics(prepared)
    converted, model = ChatLLM().generate(_system_prompt(), prepared)
    if keep_english_physics_terms():
        restored = restore_english_physics(converted.strip(), placeholders)
    else:
        restored = restore_urdu_physics(converted.strip(), placeholders)
    logger.info("Roman Urdu TTS prep via LLM %s chars=%s", model, len(restored))
    return finalize_urdu_speech(restored)


def _llm_convert_long(text: str) -> str:
    """LLM prep for long replies — only when Mavkif is unavailable or low quality."""
    if len(text) <= _prep_chunk_chars():
        return _llm_convert(text)
    parts = _split_source_for_conversion(text)
    logger.info("Long TTS LLM fallback: converting %s parts (%s chars)", len(parts), len(text))
    converted = [_llm_convert(part) for part in parts if part.strip()]
    merged = " ".join(c for c in converted if c).strip()
    return finalize_urdu_speech(merged)


def _transliterate_preserving_physics(text: str, placeholders: dict[str, str]) -> str:
    """Roman Urdu → Urdu script without touching protected physics tokens."""
    if not placeholders:
        converted = to_urdu_script(text)
        return converted if _looks_like_urdu_script(converted) else text

    pattern = "|".join(re.escape(token) for token in placeholders)
    parts = re.split(f"({pattern})", text)
    out: list[str] = []
    for part in parts:
        if not part:
            continue
        if part in placeholders:
            out.append(placeholders[part])
            continue
        converted = to_urdu_script(part)
        out.append(converted if _looks_like_urdu_script(converted) else part)
    return " ".join(out)


def _fallback_urdu_text(text: str) -> str:
    logger.warning("TTS prep using aksharamukha fallback — quality may be lower")
    if keep_english_physics_terms():
        prepared, placeholders = protect_english_physics(text)
        if URDU_SCRIPT_RE.search(prepared) and not placeholders:
            return finalize_urdu_speech(prepared.strip())
        body = _transliterate_preserving_physics(prepared, placeholders)
        return finalize_urdu_speech(body)
    prepared = prepare_roman_tts_text(text)
    if URDU_SCRIPT_RE.search(prepared):
        return finalize_urdu_speech(prepared.strip())
    converted = to_urdu_script(prepared)
    result = converted if _looks_like_urdu_script(converted) else prepared
    return finalize_urdu_speech(result)


def _hard_split_text(text: str, max_len: int) -> list[str]:
    """Split long text at spaces without dropping characters."""
    if len(text) <= max_len:
        return [text]
    parts: list[str] = []
    rest = text.strip()
    while rest:
        if len(rest) <= max_len:
            parts.append(rest)
            break
        cut = rest.rfind(" ", 0, max_len + 1)
        if cut < max_len // 4:
            cut = max_len
        piece = rest[:cut].strip()
        if piece:
            parts.append(piece)
        rest = rest[cut:].strip()
    return parts or [text[:max_len]]


def _split_by_sentences(text: str, max_len: int) -> list[str]:
    """Group sentences into chunks up to max_len (never truncate)."""
    if len(text) <= max_len:
        return [text]
    sentences = re.split(r"(?<=[۔.!?؟])\s+|\n+", text)
    chunks: list[str] = []
    buf = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > max_len:
            if buf:
                chunks.append(buf)
                buf = ""
            chunks.extend(_hard_split_text(sentence, max_len))
            continue
        candidate = f"{buf} {sentence}".strip() if buf else sentence
        if len(candidate) <= max_len:
            buf = candidate
        else:
            if buf:
                chunks.append(buf)
            buf = sentence
    if buf:
        chunks.append(buf)
    return chunks or _hard_split_text(text, max_len)


def _split_long_urdu(text: str, max_len: int | None = None) -> list[str]:
    limit = max_len or _tts_chunk_chars()
    return _split_by_sentences(text, limit)


def _split_source_for_conversion(text: str) -> list[str]:
    """Split long Roman tutor replies so LLM transliteration does not truncate."""
    limit = _prep_chunk_chars()
    return _split_by_sentences(text, limit)


def _roman_to_urdu_script(text: str) -> str:
    original = text.strip()
    if not original:
        return ""
    if URDU_SCRIPT_RE.search(original) and _latin_ratio(original) < 0.15:
        return finalize_urdu_speech(original)

    prepared = prepare_roman_tts_text(original)
    urdu_text = ""

    if _prefer_llm_prep() and _use_llm_fallback():
        try:
            convert = _llm_convert_long if len(original) > _prep_chunk_chars() else _llm_convert
            candidate = convert(original)
            if _trust_llm_output(candidate, original) or _is_good_conversion(
                candidate, original
            ):
                urdu_text = candidate
                logger.info("Roman Urdu TTS prep via LLM (quality) chars=%s", len(urdu_text))
            else:
                logger.warning("LLM TTS prep below quality bar, trying Mavkif")
        except Exception as exc:
            logger.warning("LLM TTS prep failed (%s), trying Mavkif", exc)

    if not _is_good_conversion(urdu_text, original) and _use_mavkif(prepared, original=original):
        try:
            candidate = _mavkif_convert_mixed(prepared)
            if _is_good_conversion(candidate, original):
                urdu_text = candidate
                logger.info("Roman Urdu TTS prep via Mavkif chars=%s", len(urdu_text))
            else:
                logger.warning("Mavkif output low quality, trying LLM")
        except Exception as exc:
            logger.warning("Mavkif Roman Urdu prep failed: %s", exc)

    if not _is_good_conversion(urdu_text, original) and _use_llm_fallback() and not _prefer_llm_prep():
        try:
            convert = _llm_convert_long if len(original) > _prep_chunk_chars() else _llm_convert
            candidate = convert(original)
            if _trust_llm_output(candidate, original) or _is_good_conversion(
                candidate, original
            ):
                urdu_text = candidate
            else:
                logger.warning("LLM TTS prep below quality bar, keeping best effort")
                if _looks_like_urdu_script(candidate) and not _has_leaked_tokens(candidate):
                    urdu_text = candidate
        except Exception as exc:
            logger.warning("LLM Roman Urdu TTS prep failed: %s", exc)

    # Only use aksharamukha when LLM/Mavkif produced nothing usable
    if not _looks_like_urdu_script(urdu_text):
        urdu_text = _fallback_urdu_text(original)

    return finalize_urdu_speech(urdu_text.strip())


def prepare_urdu_speech_chunks(text: str) -> list[str]:
    if not text:
        return []

    key = _cache_key(text)
    cached = _CHUNK_CACHE.get(key)
    if cached is not None:
        return cached

    urdu_text = _roman_to_urdu_script(text)
    if not urdu_text:
        raise ValueError("No Urdu speech text after preparation")

    urdu_text = polish_spoken_urdu(urdu_text)
    from src.voice.conversational_speech import latin_word_count, scrub_remaining_latin

    if latin_word_count(urdu_text):
        logger.info("Scrubbing %s leftover Latin words for TTS", latin_word_count(urdu_text))
        urdu_text = scrub_remaining_latin(urdu_text)
    chunks = _split_long_urdu(urdu_text)
    if len(_CHUNK_CACHE) >= _CACHE_MAX:
        _CHUNK_CACHE.pop(next(iter(_CHUNK_CACHE)))
    _CHUNK_CACHE[key] = chunks
    return chunks


def prepare_urdu_speech_text(text: str) -> str:
    return " ".join(prepare_urdu_speech_chunks(text))


def prepare_prepared_urdu_speech_chunks(text: str) -> list[str]:
    """LLM speech_text — Urdu script + English physics; skip Roman→Urdu conversion."""
    if not text:
        return []

    key = f"prepared:{_cache_key(text)}"
    cached = _CHUNK_CACHE.get(key)
    if cached is not None:
        return cached

    from src.voice.script_fix import sanitize_urdu_tts_text
    from src.voice.urdu_phrases import normalize_pakistani_urdu_script

    urdu_text = normalize_pakistani_urdu_script(text.strip())
    urdu_text = sanitize_urdu_tts_text(urdu_text)
    urdu_text = polish_spoken_urdu(urdu_text)
    if not urdu_text:
        raise ValueError("No Urdu speech text after preparation")

    chunks = _split_long_urdu(urdu_text)
    if len(_CHUNK_CACHE) >= _CACHE_MAX:
        _CHUNK_CACHE.pop(next(iter(_CHUNK_CACHE)))
    _CHUNK_CACHE[key] = chunks
    logger.info(
        "Prepared Urdu TTS chunks=%s total_chars=%s preview=%r",
        len(chunks),
        sum(len(c) for c in chunks),
        chunks[0][:80],
    )
    return chunks


def _split_mixed_script(text: str) -> list[tuple[str, str]]:
    """Split Urdu script blocks vs Latin (physics) blocks."""
    pattern = re.compile(
        r"([\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+|[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+)"
    )
    raw: list[tuple[str, str]] = []
    for match in pattern.finditer(text):
        chunk = match.group(0).strip(" ،,\t")
        if not chunk or not re.search(r"\w", chunk, flags=re.UNICODE):
            continue
        if URDU_SCRIPT_RE.search(chunk):
            raw.append(("ur", chunk))
        elif _LATIN_RE.search(chunk):
            raw.append(("en", chunk))
    merged: list[tuple[str, str]] = []
    for lang, segment in raw:
        if merged and merged[-1][0] == lang:
            merged[-1] = (lang, f"{merged[-1][1]} {segment}".strip())
        else:
            merged.append((lang, segment))
    return merged


def prepare_hybrid_speech_segments(text: str) -> list[tuple[str, str]]:
    """Urdu segments for Uzma + English segments for physics terms (dual Edge voices)."""
    if not text:
        return []

    key = _cache_key(text)
    cached = _HYBRID_CACHE.get(key)
    if cached is not None:
        return cached

    mixed = _roman_to_urdu_script(text)
    if not mixed:
        raise ValueError("No speech text after hybrid preparation")

    mixed = polish_spoken_urdu(mixed)
    from src.voice.conversational_speech import latin_word_count, scrub_remaining_latin

    if latin_word_count(mixed):
        mixed = scrub_remaining_latin(mixed)

    segments: list[tuple[str, str]] = []
    for lang, segment in _split_mixed_script(mixed):
        if lang == "ur":
            for chunk in _split_long_urdu(segment):
                segments.append(("ur", chunk))
        else:
            segments.append(("en", segment))

    if not segments:
        raise ValueError("No hybrid speech segments after split")

    if len(_HYBRID_CACHE) >= _CACHE_MAX:
        _HYBRID_CACHE.pop(next(iter(_HYBRID_CACHE)))
    _HYBRID_CACHE[key] = segments
    logger.info(
        "Hybrid TTS segments: %s ur + %s en, preview=%r",
        sum(1 for l, _ in segments if l == "ur"),
        sum(1 for l, _ in segments if l == "en"),
        segments[0][1][:60],
    )
    return segments
