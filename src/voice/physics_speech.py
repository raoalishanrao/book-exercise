"""Physics term handling for tutor TTS — keep English terms or Urdu script (env)."""

import re

from src.config import optional_env

PHYSICS_ENGLISH_TERMS: tuple[str, ...] = (
    "kinetic energy",
    "potential energy",
    "thermal energy",
    "mechanical energy",
    "heat energy",
    "electrical energy",
    "chemical energy",
    "nuclear energy",
    "solar energy",
    "uniform velocity",
    "uniform speed",
    "initial velocity",
    "final velocity",
    "average velocity",
    "average speed",
    "linear motion",
    "circular motion",
    "simple harmonic motion",
    "equations of motion",
    "equation of motion",
    "newton's first law",
    "newton's second law",
    "newton's third law",
    "newtons first law",
    "newtons second law",
    "newtons third law",
    "law of inertia",
    "centripetal force",
    "gravitational force",
    "frictional force",
    "normal force",
    "net force",
    "kilometer per hour",
    "metre per second",
    "meter per second",
    "angular momentum",
    "linear momentum",
    "si unit",
    "km/h",
    "m/s",
    "acceleration",
    "velocity",
    "displacement",
    "distance",
    "momentum",
    "inertia",
    "friction",
    "gravity",
    "pressure",
    "density",
    "volume",
    "temperature",
    "frequency",
    "wavelength",
    "amplitude",
    "oscillation",
    "vibration",
    "water barometer",
    "barometer",
    "manometer",
    "thermometer",
    "hygrometer",
    "atmospheric pressure",
    "numericals",
    "numerical",
    "formulas",
    "formula",
    "equations",
    "equation",
    "definition",
    "exercise",
    "diagram",
    "chapter",
    "physics",
    "centripetal",
    "centrifugal",
    "magnitude",
    "substitute",
    "calculate",
    "solution",
    "answer",
    "energy",
    "force",
    "power",
    "graph",
    "example",
    "weight",
    "speed",
    "torque",
    "scalar",
    "vector",
    "units",
    "unit",
    "class",
    "given",
    "find",
    "kinetic",
    "potential",
    "newton",
    "joule",
    "pascal",
    "hertz",
    "kelvin",
    "watt",
    "meter",
    "metre",
    "kilogram",
    "second",
    "minute",
    "hour",
    "mass",
    "work",
    "time",
    "step",
    "miss",
)

_SORTED_ENGLISH = sorted(set(PHYSICS_ENGLISH_TERMS), key=len, reverse=True)

PHYSICS_URDU_TERMS: tuple[tuple[str, str], ...] = (
    ("kinetic energy", "حرکی توانائی"),
    ("potential energy", "ممکناتی توانائی"),
    ("thermal energy", "حرارتی توانائی"),
    ("mechanical energy", "میکانیکی توانائی"),
    ("heat energy", "حرارتی توانائی"),
    ("electrical energy", "برقی توانائی"),
    ("chemical energy", "کیمیائی توانائی"),
    ("nuclear energy", "جوہری توانائی"),
    ("solar energy", "شمسی توانائی"),
    ("uniform velocity", "یکساں رفتار"),
    ("uniform speed", "یکساں رفتار"),
    ("initial velocity", "ابتدائی رفتار"),
    ("final velocity", "حتمی رفتار"),
    ("average velocity", "اوسط رفتار"),
    ("average speed", "اوسط رفتار"),
    ("linear motion", "خطی حرکت"),
    ("circular motion", "دوری حرکت"),
    ("simple harmonic motion", "سادہ توافقی حرکت"),
    ("equations of motion", "معادلاتِ حرکت"),
    ("equation of motion", "معادلۂ حرکت"),
    ("newton's first law", "نیوٹن کا پہلا قانون"),
    ("newton's second law", "نیوٹن کا دوسرا قانون"),
    ("newton's third law", "نیوٹن کا تیسرا قانون"),
    ("newtons first law", "نیوٹن کا پہلا قانون"),
    ("newtons second law", "نیوٹن کا دوسرا قانون"),
    ("newtons third law", "نیوٹن کا تیسرا قانون"),
    ("law of inertia", "قانونِ جَمود"),
    ("centripetal force", "مرکز کی طرف قوت"),
    ("gravitational force", "کششِ ثقل کی قوت"),
    ("frictional force", "اصطحکاکی قوت"),
    ("normal force", "عمودی قوت"),
    ("net force", "مجموعی قوت"),
    ("kilometer per hour", "کلومیٹر فی گھنٹہ"),
    ("metre per second", "میٹر فی سیکنڈ"),
    ("meter per second", "میٹر فی سیکنڈ"),
    ("km/h", "کلومیٹر فی گھنٹہ"),
    ("m/s", "میٹر فی سیکنڈ"),
    ("angular momentum", "زاویائی حرکت کا زور"),
    ("linear momentum", "خطی حرکت کا زور"),
    ("acceleration", "تعجیل"),
    ("velocity", "رفتار"),
    ("displacement", "نقلِ و حرکت"),
    ("distance", "فاصلہ"),
    ("momentum", "حرکت کا زور"),
    ("inertia", "جمود"),
    ("friction", "اصطحکاک"),
    ("gravity", "کششِ ثقل"),
    ("pressure", "دباؤ"),
    ("density", "کثافت"),
    ("volume", "حجم"),
    ("temperature", "درجہ حرارت"),
    ("frequency", "تعدد"),
    ("wavelength", "طولِ موج"),
    ("amplitude", "طول"),
    ("oscillation", "ارتعاش"),
    ("vibration", "ارتعاش"),
    ("water barometer", "پانی کا بارومیٹر"),
    ("barometer", "بارومیٹر"),
    ("manometer", "مینومیٹر"),
    ("thermometer", "تھرمامیٹر"),
    ("hygrometer", "نمی ناپ"),
    ("atmospheric pressure", "فضائی دباؤ"),
    ("numerical", "عددی مسئلہ"),
    ("numericals", "عددی مسائل"),
    ("formula", "فارمولا"),
    ("formulas", "فارمولے"),
    ("equation", "معادلہ"),
    ("equations", "معادلات"),
    ("graph", "گراف"),
    ("diagram", "خاکہ"),
    ("chapter", "باب"),
    ("exercise", "مشق"),
    ("example", "مثال"),
    ("definition", "تعریف"),
    ("physics", "فزکس"),
    ("force", "قوت"),
    ("energy", "توانائی"),
    ("power", "طاقت"),
    ("work", "کام"),
    ("mass", "کمیت"),
    ("weight", "وزن"),
    ("speed", "رفتار"),
    ("time", "وقت"),
    ("newton", "نیوٹن"),
    ("joule", "جول"),
    ("watt", "واٹ"),
    ("pascal", "پاسکل"),
    ("hertz", "ہرٹز"),
    ("kelvin", "کیلون"),
    ("meter", "میٹر"),
    ("metre", "میٹر"),
    ("kilogram", "کلوگرام"),
    ("second", "سیکنڈ"),
    ("minute", "منٹ"),
    ("hour", "گھنٹہ"),
    ("class", "کلاس"),
    ("step", "قدم"),
    ("miss", "مس"),
    ("kinetic", "حرکی"),
    ("potential", "ممکناتی"),
)

_SORTED_URDU_GLOSSARY = sorted(PHYSICS_URDU_TERMS, key=lambda item: -len(item[0]))
_URDU_TO_ENGLISH = sorted(
    ((urdu, english) for english, urdu in PHYSICS_URDU_TERMS),
    key=lambda item: -len(item[0]),
)

_ROMAN_PHYSICS_ALIASES: tuple[tuple[str, str], ...] = (
    (r"\bkinatic\s+energy\b", "kinetic energy"),
    (r"\bkinetik\s+energy\b", "kinetic energy"),
    (r"\bkenetic\s+energy\b", "kinetic energy"),
    (r"\bkinetic\s+enrgy\b", "kinetic energy"),
    (r"\bkinetic\s+energi\b", "kinetic energy"),
    (r"\bpotensh?al\s+energy\b", "potential energy"),
    (r"\bpotential\s+enrgy\b", "potential energy"),
    (r"\bvelosity\b", "velocity"),
    (r"\bvelocety\b", "velocity"),
    (r"\baccil?eration\b", "acceleration"),
    (r"\bintertia\b", "inertia"),
    (r"\bnewten\b", "newton"),
    (r"\bnueton\b", "newton"),
    (r"\binertiaa?\b", "inertia"),
    (r"\bfricton\b", "friction"),
    (r"\bgravty\b", "gravity"),
    (r"\bdisplacment\b", "displacement"),
    (r"\bfijiks\b", "physics"),
    (r"\bfiziks\b", "physics"),
    (r"\bphiziks\b", "physics"),
)

_LATIN_RE = re.compile(r"[A-Za-z]")
_URDU_TOKEN_RE = re.compile(r"ZZURD\d{3}ZZ")
_UNIQUE_URDU_TERMS = sorted({urdu for _, urdu in PHYSICS_URDU_TERMS}, key=len, reverse=True)


def physics_terms_mode() -> str:
    return optional_env("EDGE_TTS_PHYSICS_TERMS", "urdu").lower()


def use_hybrid_physics_tts() -> bool:
    """Urdu voice + English voice for physics terms (best for mixed tutor replies)."""
    return physics_terms_mode() == "hybrid"


def keep_english_physics_terms() -> bool:
    return physics_terms_mode() in {"english", "en", "latin", "hybrid"}


def normalize_physics_aliases(text: str) -> str:
    out = text
    for pattern, replacement in _ROMAN_PHYSICS_ALIASES:
        out = re.sub(pattern, replacement, out, flags=re.IGNORECASE)
    return out


def _apply_urdu_glossary(text: str) -> str:
    out = normalize_physics_aliases(text)
    for english, urdu in _SORTED_URDU_GLOSSARY:
        out = re.sub(rf"(?<!\w){re.escape(english)}(?!\w)", urdu, out, flags=re.IGNORECASE)
    return out


def apply_physics_glossary(text: str) -> str:
    """Normalize typos; optionally map physics terms to Urdu script."""
    if keep_english_physics_terms():
        return normalize_physics_aliases(text)
    return _apply_urdu_glossary(text)


def protect_english_physics(text: str) -> tuple[str, dict[str, str]]:
    """Hide English physics terms from LLM/transliterator so they stay Latin."""
    out = normalize_physics_aliases(text)
    mapping: dict[str, str] = {}
    token_idx = 0
    for term in _SORTED_ENGLISH:
        pattern = re.compile(rf"(?<!\w)({re.escape(term)})(?!\w)", re.IGNORECASE)

        def _repl(match: re.Match[str]) -> str:
            nonlocal token_idx
            token = f"ZZENG{token_idx:03d}ZZ"
            while token in mapping:
                token_idx += 1
                token = f"ZZENG{token_idx:03d}ZZ"
            token_idx += 1
            mapping[token] = match.group(1)
            return token

        out = pattern.sub(_repl, out)
    return out, mapping


def restore_english_physics(text: str, mapping: dict[str, str]) -> str:
    out = text
    for token, english in mapping.items():
        out = out.replace(token, english)
    return out


def protect_urdu_physics(text: str) -> tuple[str, dict[str, str]]:
    """Hide embedded Urdu physics terms so LLM does not mistransliterate them."""
    out = text
    mapping: dict[str, str] = {}
    token_idx = 0
    for urdu in _UNIQUE_URDU_TERMS:
        if urdu not in out:
            continue
        token = f"ZZURD{token_idx:03d}ZZ"
        while token in mapping:
            token_idx += 1
            token = f"ZZURD{token_idx:03d}ZZ"
        token_idx += 1
        mapping[token] = urdu
        out = out.replace(urdu, token)
    return out, mapping


def restore_urdu_physics(text: str, mapping: dict[str, str]) -> str:
    out = text
    for token, urdu in mapping.items():
        out = out.replace(token, urdu)
    return out


_ENG_TOKEN_RE = re.compile(r"ZZENG\d{3}ZZ")


def _revert_urdu_physics_to_english(text: str) -> str:
    out = text
    for urdu, english in _URDU_TO_ENGLISH:
        out = out.replace(urdu, english)
    phonetic_fixes = (
        (r"\bکینٹک\b", "kinetic"),
        (r"\bکینیٹک\b", "kinetic"),
        (r"\bانرجی\b", "energy"),
        (r"\bاینرجی\b", "energy"),
        (r"\bپوٹینشل\b", "potential"),
        (r"\bپوٹینشیل\b", "potential"),
        (r"\bفورس\b", "force"),
        (r"\bویلیوسٹی\b", "velocity"),
        (r"\bویلاسٹی\b", "velocity"),
        (r"\bاینرشیا\b", "inertia"),
        (r"\bفریکشن\b", "friction"),
        (r"\bفزیکس\b", "physics"),
        (r"\bنیوٹن\b", "Newton"),
        (r"\bتعجیل\b", "acceleration"),
        (r"\bجمود\b", "inertia"),
        (r"\bرفتار\b", "velocity"),
        (r"\bقوت\b", "force"),
        (r"\bتوانائی\b", "energy"),
    )
    for pattern, replacement in phonetic_fixes:
        out = re.sub(pattern, replacement, out, flags=re.IGNORECASE)
    return out


def contains_physics_english(text: str) -> bool:
    lower = text.lower()
    for term in _SORTED_ENGLISH:
        if re.search(rf"(?<!\w){re.escape(term.lower())}(?!\w)", lower):
            return True
    return False


def strip_physics_english_for_ratio(text: str) -> str:
    out = _ENG_TOKEN_RE.sub("", text)
    for term in _SORTED_ENGLISH:
        out = re.sub(rf"(?<!\w){re.escape(term)}(?!\w)", "", out, flags=re.IGNORECASE)
    return out


def has_bad_physics_transliteration(text: str) -> bool:
    """In English mode: unwanted Urdu physics words. In Urdu mode: phonetic leaks."""
    if keep_english_physics_terms():
        for urdu, _ in _URDU_TO_ENGLISH:
            if urdu in text:
                return True
        return bool(re.search(r"کینٹک|انرجی|ویلیوسٹی|فورس|نیوٹن|تعجیل|جمود", text))
    patterns = (
        r"کینٹک|کینیٹک|کائنیٹک",
        r"\bانرجی\b|\bاینرجی\b",
        r"ویلیوسٹی|ویلاسٹی",
        r"پوٹینش",
        r"\bفورس\b",
    )
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


def finalize_urdu_speech(text: str) -> str:
    """Final cleanup before TTS."""
    out = text.strip()
    if use_hybrid_physics_tts() or keep_english_physics_terms():
        # Only fix obvious phonetic leaks — do not replace normal Urdu words.
        for pattern, replacement in (
            (r"کینٹک\s*(?:توانائی|انرجی|اینرجی)", "kinetic energy"),
            (r"کینیٹک\s*(?:توانائی|انرجی|اینرجی)", "kinetic energy"),
            (r"ممکناتی\s*(?:انرجی|اینرجی)", "potential energy"),
            (r"\bکینٹک\b", "kinetic"),
            (r"\bانرجی\b", "energy"),
            (r"\bاینرجی\b", "energy"),
            (r"\bویلیوسٹی\b", "velocity"),
            (r"\bفورس\b", "force"),
        ):
            out = re.sub(pattern, replacement, out, flags=re.IGNORECASE)
        out = re.sub(r"ZZENG\d{3}ZZ", "", out)
    else:
        out = _apply_urdu_glossary(out)
        for pattern, replacement in (
            (r"کینٹک\s*(?:توانائی|انرجی|اینرجی)", "حرکی توانائی"),
            (r"کینیٹک\s*(?:توانائی|انرجی|اینرجی)", "حرکی توانائی"),
            (r"حرکی\s*(?:انرجی|اینرجی)", "حرکی توانائی"),
            (r"ممکناتی\s*(?:انرجی|اینرجی)", "ممکناتی توانائی"),
            (r"\bکینٹک\b", "حرکی"),
            (r"\bانرجی\b", "توانائی"),
            (r"\bاینرجی\b", "توانائی"),
            (r"\bویلیوسٹی\b", "رفتار"),
            (r"\bفورس\b", "قوت"),
            (r"\bنیوٹن\b", "نیوٹن"),
        ):
            out = re.sub(pattern, replacement, out, flags=re.IGNORECASE)
        out = re.sub(r"ZZURD\d{3}ZZ", "", out)
        from src.voice.conversational_speech import scrub_remaining_latin

        out = scrub_remaining_latin(out)
    from src.voice.script_fix import sanitize_urdu_tts_text

    out = sanitize_urdu_tts_text(out)
    out = re.sub(r"\s+", " ", out).strip()
    return out
