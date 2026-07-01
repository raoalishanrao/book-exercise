"""Convert leftover English words to Urdu script for Edge Uzma TTS."""

import logging
import re

from src.voice.physics_speech import (
    keep_english_physics_terms,
    protect_english_physics,
    restore_english_physics,
    use_hybrid_physics_tts,
)
from src.voice.script_fix import to_urdu_script
from src.voice.text_clean import URDU_SCRIPT_RE

logger = logging.getLogger("tutor.voice.conversational_speech")

# Common English in Pakistani tutor Roman Urdu — longest phrases first.
CONVERSATIONAL_ENGLISH_TO_URDU: tuple[tuple[str, str], ...] = (
    ("for example", "مثال کے طور پر"),
    ("in other words", "دوسرے الفاظ میں"),
    ("step by step", "قدم بہ قدم"),
    ("board exam", "بورڈ امتحان"),
    ("board exams", "بورڈ امتحانات"),
    ("specific", "مخصوص"),
    ("topic", "موضوع"),
    ("topics", "موضوعات"),
    ("chapter", "باب"),
    ("chapters", "ابواب"),
    ("concept", "تصور"),
    ("concepts", "تصورات"),
    ("question", "سوال"),
    ("questions", "سوالات"),
    ("answer", "جواب"),
    ("answers", "جوابات"),
    ("problem", "مسئلہ"),
    ("problems", "مسائل"),
    ("example", "مثال"),
    ("examples", "مثالیں"),
    ("exercise", "مشق"),
    ("exercises", "مشقیں"),
    ("definition", "تعریف"),
    ("definitions", "تعاریف"),
    ("formula", "فارمولا"),
    ("formulas", "فارمولے"),
    ("graph", "گراف"),
    ("graphs", "گراف"),
    ("diagram", "خاکہ"),
    ("diagrams", "خاکے"),
    ("numerical", "عددی مسئلہ"),
    ("numericals", "عددی مسائل"),
    ("related", "متعلق"),
    ("important", "اہم"),
    ("general", "عمومی"),
    ("detail", "تفصیل"),
    ("details", "تفصیلات"),
    ("summary", "خلاصہ"),
    ("introduction", "تعارف"),
    ("conclusion", "نتیجہ"),
    ("practice", "مشق"),
    ("revision", "دہرائی"),
    ("understanding", "سمجھ"),
    ("explain", "سمجھانا"),
    ("explained", "سمجھایا"),
    ("help", "مدد"),
    ("test", "ٹیسٹ"),
    ("exam", "امتحان"),
    ("exams", "امتحانات"),
    ("board", "بورڈ"),
    ("class", "کلاس"),
    ("student", "طالب علم"),
    ("teacher", "استاد"),
    ("simple", "آسان"),
    ("difficult", "مشکل"),
    ("basic", "بنیادی"),
    ("advanced", "اعلیٰ"),
    ("complete", "مکمل"),
    ("correct", "درست"),
    ("wrong", "غلط"),
    ("right", "صحیح"),
    ("first", "پہلا"),
    ("second", "دوسرا"),
    ("third", "تیسرا"),
    ("last", "آخری"),
    ("next", "اگلا"),
    ("previous", "پچھلا"),
    ("section", "حصہ"),
    ("part", "حصہ"),
    ("parts", "حصے"),
    ("point", "نکتہ"),
    ("points", "نکات"),
    ("type", "قسم"),
    ("types", "اقسام"),
    ("method", "طریقہ"),
    ("methods", "طریقے"),
    ("reason", "وجہ"),
    ("reasons", "وجوہات"),
    ("result", "نتیجہ"),
    ("results", "نتائج"),
    ("value", "قدر"),
    ("values", "قدریں"),
    ("number", "عدد"),
    ("numbers", "اعداد"),
    ("data", "اعداد"),
    ("unit", "اکائی"),
    ("units", "اکائیاں"),
    ("law", "قانون"),
    ("laws", "قوانین"),
    ("rule", "اصول"),
    ("rules", "اصول"),
    ("note", "نوٹ"),
    ("notes", "نوٹس"),
    ("please", "براہ کرم"),
    ("sorry", "معذرت"),
    ("okay", "ٹھیک"),
    ("ok", "ٹھیک"),
    ("yes", "جی"),
    ("no", "نہیں"),
    ("the", ""),  # drop filler when stuck between Urdu
    ("a", ""),
    ("an", ""),
    ("and", "اور"),
    ("or", "یا"),
    ("but", "لیکن"),
    ("with", "کے ساتھ"),
    ("without", "کے بغیر"),
    ("about", "کے بارے میں"),
    ("because", "کیونکہ"),
    ("if", "اگر"),
    ("when", "جب"),
    ("what", "کیا"),
    ("why", "کیوں"),
    ("how", "کیسے"),
    ("which", "کون سا"),
    ("where", "کہاں"),
    ("who", "کون"),
    ("this", "یہ"),
    ("that", "وہ"),
    ("these", "یہ"),
    ("those", "وہ"),
    ("here", "یہاں"),
    ("there", "وہاں"),
    ("now", "اب"),
    ("then", "پھر"),
    ("also", "بھی"),
    ("only", "صرف"),
    ("just", "بس"),
    ("very", "بہت"),
    ("more", "مزید"),
    ("less", "کم"),
    ("some", "کچھ"),
    ("any", "کوئی"),
    ("all", "تمام"),
    ("each", "ہر"),
    ("every", "ہر"),
    ("other", "دوسرا"),
    ("same", "ویسا ہی"),
    ("different", "مختلف"),
    ("new", "نیا"),
    ("old", "پرانا"),
    ("good", "اچھا"),
    ("bad", "برا"),
    ("best", "بہترین"),
    ("better", "بہتر"),
    ("main", "اہم"),
    ("key", "اہم"),
    ("clear", "واضح"),
    ("easy", "آسان"),
    ("hard", "مشکل"),
    ("true", "درست"),
    ("false", "غلط"),
    ("mean", "مطلب"),
    ("means", "مطلب"),
    ("called", "کہلاتا"),
    ("known", "مشہور"),
    ("using", "استعمال کرتے ہوئے"),
    ("use", "استعمال"),
    ("used", "استعمال"),
    ("given", "دی گئی"),
    ("find", "تلاش"),
    ("show", "دکھائیں"),
    ("tell", "بتائیں"),
    ("ask", "پوچھیں"),
    ("said", "کہا"),
    ("say", "کہیں"),
    ("think", "سوچیں"),
    ("know", "جانیں"),
    ("learn", "سیکھیں"),
    ("read", "پڑھیں"),
    ("write", "لکھیں"),
    ("start", "شروع"),
    ("end", "ختم"),
    ("begin", "شروع"),
    ("continue", "جاری رکھیں"),
    ("try", "کوشش"),
    ("remember", "یاد رکھیں"),
    ("forget", "بھولیں"),
    ("understand", "سمجھیں"),
    ("mistake", "غلطی"),
    ("mistakes", "غلطیاں"),
    ("solution", "حل"),
    ("solutions", "حل"),
    ("calculation", "حساب"),
    ("calculations", "حساب"),
)

_SORTED_CONVERSATIONAL = sorted(CONVERSATIONAL_ENGLISH_TO_URDU, key=lambda item: -len(item[0]))
_LATIN_WORD_RE = re.compile(r"\b[A-Za-z]+\b")


def scrub_conversational_latin(text: str) -> str:
    """Hybrid mode: convert topic/specific etc. to Urdu; keep physics terms in English."""
    if not text:
        return text

    prepared, placeholders = protect_english_physics(text)
    out = prepared
    for english, urdu in _SORTED_CONVERSATIONAL:
        if not english or not urdu:
            continue
        out = re.sub(rf"(?<!\w){re.escape(english)}(?!\w)", urdu, out, flags=re.IGNORECASE)

    def _replace_word(match: re.Match[str]) -> str:
        word = match.group(0)
        converted = to_urdu_script(word)
        if converted and URDU_SCRIPT_RE.search(converted):
            return converted.strip()
        logger.warning("TTS hybrid: unknown Latin %r — keeping for English voice", word)
        return word

    if _LATIN_WORD_RE.search(out):
        out = _LATIN_WORD_RE.sub(_replace_word, out)
    out = re.sub(r"\s+", " ", out).strip()
    return restore_english_physics(out, placeholders)


def scrub_remaining_latin(text: str) -> str:
    """Replace or transliterate any Latin left in Urdu-only TTS text."""
    if not text:
        return text
    if use_hybrid_physics_tts():
        return scrub_conversational_latin(text)
    if keep_english_physics_terms():
        return text

    from src.voice.urdu_phrases import apply_roman_phrase_glossary

    out = apply_roman_phrase_glossary(text)
    for english, urdu in _SORTED_CONVERSATIONAL:
        if not english:
            continue
        out = re.sub(rf"(?<!\w){re.escape(english)}(?!\w)", urdu, out, flags=re.IGNORECASE)

    # Drop stray articles/prepositions that became empty
    out = re.sub(r"\s+", " ", out).strip()

    def _replace_word(match: re.Match[str]) -> str:
        word = match.group(0)
        from src.voice.urdu_phrases import apply_roman_phrase_glossary

        mapped = apply_roman_phrase_glossary(word)
        if mapped != word and URDU_SCRIPT_RE.search(mapped):
            return mapped.strip()
        converted = to_urdu_script(word)
        if converted and URDU_SCRIPT_RE.search(converted):
            from src.voice.urdu_phrases import normalize_pakistani_urdu_script

            return normalize_pakistani_urdu_script(converted.strip())
        logger.warning("TTS: leftover Latin word %r — Uzma may mispronounce", word)
        return word

    if _LATIN_WORD_RE.search(out):
        out = _LATIN_WORD_RE.sub(_replace_word, out)
        out = re.sub(r"\s+", " ", out).strip()

    return out


def latin_word_count(text: str) -> int:
    return len(_LATIN_WORD_RE.findall(text))
