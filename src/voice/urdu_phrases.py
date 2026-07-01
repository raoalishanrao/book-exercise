"""Fixed Pakistani Urdu phrases — Roman words that look English must become Urdu script for Uzma."""

import logging
import re

logger = logging.getLogger("tutor.voice.urdu_phrases")

URDU_ARABIC_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")
_LATIN_WORD_RE = re.compile(r"\b[A-Za-z]+\b")

# Phrases first (longest match wins).
ROMAN_TO_URDU_PHRASES: tuple[tuple[str, str], ...] = (
    ("assalam u alaikum", "السلام علیکم"),
    ("assalam-o-alaikum", "السلام علیکم"),
    ("assalam o alaikum", "السلام علیکم"),
    ("assalamu alaikum", "السلام علیکم"),
    ("asalam o alaikum", "السلام علیکم"),
    ("asalamu alaikum", "السلام علیکم"),
    ("salam alaikum", "السلام علیکم"),
    ("salam alaykum", "السلام علیکم"),
    ("wa alaikum assalam", "وعلیکم السلام"),
    ("walaikum assalam", "وعلیکم السلام"),
    ("alaikum assalam", "وعلیکم السلام"),
    ("alaikum salam", "وعلیکم السلام"),
    ("is liye", "اس لیے"),
    ("islye", "اس لیے"),
    ("us liye", "اس لیے"),
    ("uss liye", "اس لیے"),
    ("iss liye", "اس لیے"),
    ("ke liye", "کے لیے"),
    ("ki wajah", "کی وجہ"),
    ("ki waja", "کی وجہ"),
    ("ka matlab", "کا مطلب"),
    ("se related", "سے متعلق"),
    ("se mutaliq", "سے متعلق"),
    ("se pehle", "سے پہلے"),
    ("se pahle", "سے پہلے"),
    ("se baad", "سے بعد"),
    ("ke baad", "کے بعد"),
    ("ke pehle", "کے پہلے"),
    ("ke andar", "کے اندر"),
    ("ke bahar", "کے باہر"),
    ("ke sath", "کے ساتھ"),
    ("ke saath", "کے ساتھ"),
    ("ki taraf", "کی طرف"),
    ("ke zariye", "کے ذریعے"),
    ("ke zarye", "کے ذریعے"),
    ("par depend", "پر منحصر"),
    ("jee bilkul", "جی بالکل"),
    ("ji bilkul", "جی بالکل"),
    ("theek hai", "ٹھیک ہے"),
    ("thik hai", "ٹھیک ہے"),
    ("bilkul theek", "بالکل ٹھیک"),
    ("step by step", "قدم بہ قدم"),
    ("misaal ke tor par", "مثال کے طور پر"),
    ("for example", "مثال کے طور پر"),
    ("madad kar sakti hon", "مدد کر سکتی ہوں"),
    ("kar sakti hon", "کر سکتی ہوں"),
    ("samajh aaya", "سمجھ آیا"),
    ("samajh aa gaya", "سمجھ آ گیا"),
    ("yad rakhiye", "یاد رکھئے"),
    ("aap ki", "آپ کی"),
    ("aap ka", "آپ کا"),
    ("aap ke", "آپ کے"),
    ("aap ko", "آپ کو"),
    ("aap ne", "آپ نے"),
    ("tum ne", "تم نے"),
    ("hum ne", "ہم نے"),
    ("main ne", "میں نے"),
    ("mein ne", "میں نے"),
    ("yeh hai", "یہ ہے"),
    ("ye hai", "یہ ہے"),
    ("yeh hain", "یہ ہیں"),
    ("ye hain", "یہ ہیں"),
    ("wo hai", "وہ ہے"),
    ("vo hai", "وہ ہے"),
    ("wo hain", "وہ ہیں"),
    ("vo hain", "وہ ہیں"),
    ("shukriya", "شکریہ"),
    ("shukria", "شکریہ"),
    ("dekhiye", "دیکھئے"),
    ("dekho", "دیکھو"),
    ("assalam", "السلام"),
    ("salam", "سلام"),
    ("alaikum", "علیکم"),
    ("chalen", "چلیں"),
    ("chalein", "چلیں"),
)

# us / is / un + postposition — Uzma reads "us" as English "we" if left in Latin.
_CONTEXT_ROMAN_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\b(u?ss|us|is|iss)\s+(mein|main|men)\b", "اس میں"),
    (r"\b(u?ss|us|is|iss)\s+(ko)\b", "اس کو"),
    (r"\b(u?ss|us|is|iss)\s+(ka)\b", "اس کا"),
    (r"\b(u?ss|us|is|iss)\s+(ki)\b", "اس کی"),
    (r"\b(u?ss|us|is|iss)\s+(ke)\b", "اس کے"),
    (r"\b(u?ss|us|is|iss)\s+(se)\b", "اس سے"),
    (r"\b(u?ss|us|is|iss)\s+(par|pe)\b", "اس پر"),
    (r"\b(u?ss|us|is|iss)\s+(ne)\b", "اس نے"),
    (r"\b(u?ss|us|is|iss)\s+(liye|lye|ly)\b", "اس لیے"),
    (r"\b(u?ss|us|is|iss)\s+(tarah|trah)\b", "اس طرح"),
    (r"\b(u?ss|us|is|iss)\s+(wakt|waqt)\b", "اس وقت"),
    (r"\bin\s+(mein|main|men)\b", "ان میں"),
    (r"\bun\s+(mein|main|men)\b", "ان میں"),
    (r"\bun\s+(ko)\b", "ان کو"),
    (r"\bun\s+(ka)\b", "ان کا"),
    (r"\bun\s+(ki)\b", "ان کی"),
    (r"\bun\s+(ke)\b", "ان کے"),
    (r"\bun\s+(se)\b", "ان سے"),
    (r"\bun\s+(par|pe)\b", "ان پر"),
    (r"\bun\s+(ne)\b", "ان نے"),
    (r"\bun\s+(liye|lye)\b", "ان لیے"),
    (r"\bjis\s+(mein|main|men)\b", "جس میں"),
    (r"\bjis\s+(ko)\b", "جس کو"),
    (r"\bjis\s+(ka)\b", "جس کا"),
    (r"\bjis\s+(ki)\b", "جس کی"),
    (r"\bjis\s+(ke)\b", "جس کے"),
    (r"\bjis\s+(se)\b", "جس سے"),
    (r"\bjis\s+(par|pe)\b", "جس پر"),
    (r"\bjis\s+(ne)\b", "جس نے"),
    (r"\bjis\s+(tarah|trah)\b", "جس طرح"),
    (r"\bkis\s+(ko)\b", "کس کو"),
    (r"\bkis\s+(ka)\b", "کس کا"),
    (r"\bkis\s+(ki)\b", "کس کی"),
    (r"\bkis\s+(ke)\b", "کس کے"),
    (r"\bkis\s+(tarah|trah)\b", "کس طرح"),
    (r"\b(yeh|ye)\s+(ko)\b", "یہ کو"),
    (r"\b(yeh|ye)\s+(ka)\b", "یہ کا"),
    (r"\b(yeh|ye)\s+(ki)\b", "یہ کی"),
    (r"\b(yeh|ye)\s+(ke)\b", "یہ کے"),
    (r"\b(yeh|ye)\s+(se)\b", "یہ سے"),
    (r"\b(yeh|ye)\s+(par|pe)\b", "یہ پر"),
    (r"\b(wo|vo)\s+(ko)\b", "وہ کو"),
    (r"\b(wo|vo)\s+(ka)\b", "وہ کا"),
    (r"\b(wo|vo)\s+(ki)\b", "وہ کی"),
    (r"\b(wo|vo)\s+(ke)\b", "وہ کے"),
    (r"\b(wo|vo)\s+(se)\b", "وہ سے"),
    (r"\b(wo|vo)\s+(par|pe)\b", "وہ پر"),
    (r"\b(wo|vo)\s+(ne)\b", "وہ نے"),
    (r"\bun\s+(mein|main|men)\s+(se)\b", "ان میں سے"),
    (r"\b(is|iss)\s+(ke)\s+(baad)\b", "اس کے بعد"),
    (r"\b(is|iss)\s+(topic|concept)\b", "اس موضوع"),
    (r"\bto\s+ab\b", "تو اب"),
    (r"\btoh\s+ab\b", "تو اب"),
    (r"\bto\s+phir\b", "تو پھر"),
    (r"\btoh\s+phir\b", "تو پھر"),
    (r"\bhum\s+(is|iss)\b", "ہم اس"),
)

_SORTED_CONTEXT_PATTERNS = sorted(_CONTEXT_ROMAN_PATTERNS, key=lambda item: -len(item[0]))

# Safe single-word Roman Urdu (not physics English).
ROMAN_URDU_PARTICLES: dict[str, str] = {
    "beta": "بیٹا",
    "bhai": "بھائی",
    "jee": "جی",
    "ji": "جی",
    "mein": "میں",
    "main": "میں",
    "aap": "آپ",
    "ap": "آپ",
    "hai": "ہے",
    "hain": "ہیں",
    "nahi": "نہیں",
    "nahin": "نہیں",
    "aur": "اور",
    "madad": "مدد",
    "uss": "اس",
    "iss": "اس",
    "jis": "جس",
    "kis": "کس",
    "yeh": "یہ",
    "ye": "یہ",
    "wo": "وہ",
    "vo": "وہ",
    "un": "ان",
    "in": "ان",
    "hum": "ہم",
    "tum": "تم",
    "mujhe": "مجھے",
    "mujh": "مجھ",
    "tujhe": "تجھے",
    "tujh": "تجھ",
    "unka": "ان کا",
    "unki": "ان کی",
    "unke": "ان کے",
    "unko": "ان کو",
    "iska": "اس کا",
    "iski": "اس کی",
    "iske": "اس کے",
    "isko": "اس کو",
    "isne": "اس نے",
    "usne": "اس نے",
    "jiska": "جس کا",
    "jiski": "جس کی",
    "jiske": "جس کے",
    "jisko": "جس کو",
    "jisne": "جس نے",
    "kiska": "کس کا",
    "kiski": "کس کی",
    "kiske": "کس کے",
    "kisko": "کس کو",
    "kuch": "کچھ",
    "sab": "سب",
    "har": "ہر",
    "bhi": "بھی",
    "tak": "تک",
    "se": "سے",
    "par": "پر",
    "pe": "پر",
    "ne": "نے",
    "ko": "کو",
    "ka": "کا",
    "ki": "کی",
    "ke": "کے",
    "ho": "ہو",
    "hon": "ہوں",
    "tha": "تھا",
    "thi": "تھی",
    "the": "تھے",
    "raha": "رہا",
    "rahi": "رہی",
    "rahe": "رہے",
    "gaya": "گیا",
    "gayi": "گئی",
    "gaye": "گئے",
    "karta": "کرتا",
    "karti": "کرتی",
    "karte": "کرتے",
    "kiya": "کیا",
    "kiye": "کیے",
    "karein": "کریں",
    "karo": "کرو",
    "kare": "کرے",
    "karega": "کرے گا",
    "karegi": "کرے گی",
    "sakta": "سکتا",
    "sakti": "سکتی",
    "sakte": "سکتے",
    "sakta": "سکتا",
    "lijiye": "لیجیے",
    "layein": "لائیں",
    "lekin": "لیکن",
    "magar": "مگر",
    "agar": "اگر",
    "phir": "پھر",
    "tab": "تب",
    "ab": "اب",
    "yahan": "یہاں",
    "wahan": "وہاں",
    "jahan": "جہاں",
    "kahan": "کہاں",
    "kab": "کب",
    "jab": "جب",
    "kaise": "کیسے",
    "kese": "کیسے",
    "kyun": "کیوں",
    "kyon": "کیوں",
    "kaun": "کون",
    "jo": "جو",
    "tou": "تو",
    "toh": "تو",
    "warna": "ورنہ",
    "isliye": "اس لیے",
    "matlab": "مطلب",
    "yaani": "یعنی",
    "yani": "یعنی",
    "bohat": "بہت",
    "bahut": "بہت",
    "zyada": "زیادہ",
    "ziyada": "زیادہ",
    "thoda": "تھوڑا",
    "thori": "تھوڑی",
    "kam": "کم",
    "sirf": "صرف",
    "bas": "بس",
    "bilkul": "بالکل",
    "theek": "ٹھیک",
    "thik": "ٹھیک",
    "sahi": "صحیح",
    "galat": "غلط",
    "asaan": "آسان",
    "aasan": "آسان",
    "mushkil": "مشکل",
    "zaroor": "ضرور",
    "zaruri": "ضروری",
    "cheez": "چیز",
    "cheezein": "چیزیں",
    "tarah": "طرح",
    "trah": "طرح",
    "liye": "لیے",
    "lye": "لیے",
    "wajah": "وجہ",
    "waja": "وجہ",
    "liye": "لیے",
    "samajh": "سمجھ",
    "samjho": "سمجھو",
    "samjhein": "سمجھیں",
    "parh": "پڑھ",
    "parho": "پڑھو",
    "parhen": "پڑھیں",
    "likho": "لکھو",
    "dekho": "دیکھو",
    "suniye": "سنیے",
    "suno": "سنو",
    "batao": "بتاؤ",
    "batayein": "بتائیں",
    "poocho": "پوچھو",
    "poochiye": "پوچھیے",
    "wakt": "وقت",
    "waqt": "وقت",
    "taraf": "طرف",
    "andar": "اندر",
    "bahar": "باہر",
    "upar": "اوپر",
    "neeche": "نیچے",
    "niche": "نیچے",
    "beech": "بیچ",
    "darmiyan": "درمیان",
    "sath": "ساتھ",
    "saath": "ساتھ",
    "parhenge": "پڑھیں گے",
    "parhen": "پڑھیں",
    "samjhte": "سمجھتے",
    "samjhein": "سمجھیں",
    "samjho": "سمجھو",
    "lagti": "لگتی",
    "lagta": "لگتا",
    "lagte": "لگتے",
    "hoti": "ہوتی",
    "hota": "ہوتا",
    "hote": "ہوتے",
    "rehti": "رہتی",
    "rehta": "رہتا",
    "rehte": "رہتے",
    "karti": "کرتی",
    "karta": "کرتا",
    "karte": "کرتے",
    "karenge": "کریں گے",
    "important": "اہم",
    "constant": "ثابت",
    "related": "متعلق",
    "leykin": "لیکن",
}

_SORTED_ROMAN_PHRASES = sorted(ROMAN_TO_URDU_PHRASES, key=lambda item: -len(item[0]))
_SORTED_PARTICLES = sorted(ROMAN_URDU_PARTICLES.items(), key=lambda item: -len(item[0]))

# Latin tokens that must stay for physics/board English (never particle-mapped).
_PHYSICS_LATIN_SKIP: frozenset[str] = frozenset(
    {
        "a", "an", "the", "and", "or", "of", "at", "by", "for", "from", "into",
        "with", "without", "between", "about", "over", "under", "after", "before",
        "newton", "force", "energy", "velocity", "acceleration", "mass", "weight",
        "pressure", "gravity", "friction", "momentum", "inertia", "work", "power",
        "joule", "watt", "pascal", "hertz", "kelvin", "meter", "metre", "second",
        "minute", "hour", "graph", "formula", "numerical", "physics", "chapter",
        "example", "topic", "concept", "question", "answer", "definition", "unit",
        "units", "vector", "scalar", "kinetic", "potential", "thermal", "heat",
        "speed", "distance", "displacement", "frequency", "wavelength", "amplitude",
        "equation", "calculate", "substitute", "given", "find", "solution", "step",
        "class", "board", "exam", "test", "diagram", "exercise", "numericals",
        "km", "hr", "kg", "n", "j", "pa", "hz",
    }
)

URDU_SCRIPT_FIXES: tuple[tuple[str, str], ...] = (
    (r"اسلم\s*[-–]\s*او\s*[-–]?\s*الیکم", "السلام علیکم"),
    (r"اسلم\s*[-–]\s*علیکم", "السلام علیکم"),
    (r"اسلم\s+علیکم", "السلام علیکم"),
    (r"السلام\s*[-–]\s*علیکم", "السلام علیکم"),
    (r"و\s+سلام(?!\s*علیکم)", "سلام"),
    (r"\bاسلم\b", "سلام"),
    (r"\bبیت\b", "بیٹا"),
    (r"\bآپک\b", "آپ کو"),
    (r"جیہون|جی\s*ہون", "جی ہوں"),
    (r"مےءاسر|مےئن|مےء", "میں"),
    (r"عليكم", "علیکم"),
    (r"عليك", "علیک"),
    (r"اليوم", "آج"),
    (r"كيا", "کیا"),
    (r"لاكن", "لیکن"),
)

_ARABIC_TO_URDU_CHARS = str.maketrans(
    {
        "\u064a": "\u06cc",
        "\u0643": "\u06a9",
        "\u0623": "\u0627",
        "\u0625": "\u0627",
        "\u0671": "\u0627",
        "\u0649": "\u06cc",
    }
)

_HYPHENATED_URDU_RE = re.compile(r"[\u0600-\u06FF]-[\u0600-\u06FF](?:-[\u0600-\u06FF])+")


def _should_skip_latin_word(word: str) -> bool:
    lower = word.lower()
    if lower in _PHYSICS_LATIN_SKIP:
        return True
    from src.voice.physics_speech import contains_physics_english

    if contains_physics_english(word):
        return True
    return False


def _fix_hyphenated_urdu(text: str) -> str:
    def _replace(match: re.Match[str]) -> str:
        chunk = match.group(0)
        if "اسلم" in chunk and "الیکم" in chunk:
            return "السلام علیکم"
        return chunk.replace("-", " ")

    return _HYPHENATED_URDU_RE.sub(_replace, text)


def _apply_context_patterns(text: str) -> str:
    out = text
    for pattern, replacement in _SORTED_CONTEXT_PATTERNS:
        out = re.sub(pattern, replacement, out, flags=re.IGNORECASE)
    return out


def _apply_particle_words(text: str) -> str:
    def _replace(match: re.Match[str]) -> str:
        word = match.group(0)
        lower = word.lower()
        if _should_skip_latin_word(lower):
            return word
        mapped = ROMAN_URDU_PARTICLES.get(lower)
        if mapped:
            return mapped
        return word

    return _LATIN_WORD_RE.sub(_replace, text)


def apply_roman_phrase_glossary(text: str) -> str:
    """Map Roman Urdu (including English-looking words) to Urdu script for Uzma."""
    if not text:
        return text
    if URDU_ARABIC_RE.search(text) and not _LATIN_WORD_RE.search(text):
        return text

    out = _apply_context_patterns(text)
    for roman, urdu in _SORTED_ROMAN_PHRASES:
        if not roman:
            continue
        out = re.sub(
            rf"(?<![\w\u0600-\u06FF]){re.escape(roman)}(?![\w\u0600-\u06FF])",
            urdu,
            out,
            flags=re.IGNORECASE,
        )
    out = _apply_particle_words(out)
    return out


def normalize_arabic_letters_to_urdu(text: str) -> str:
    if not text:
        return text
    return text.translate(_ARABIC_TO_URDU_CHARS)


def normalize_pakistani_urdu_script(text: str) -> str:
    if not text:
        return text
    out = normalize_arabic_letters_to_urdu(text.strip())
    out = re.sub(r"[\u064B-\u065F\u0670\u0610-\u061A\u06D6-\u06ED]", "", out)
    out = _fix_hyphenated_urdu(out)
    for pattern, replacement in URDU_SCRIPT_FIXES:
        out = re.sub(pattern, replacement, out)
    out = re.sub(r"\s+", " ", out).strip()
    return out
