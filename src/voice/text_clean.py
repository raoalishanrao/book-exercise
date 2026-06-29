import re

URDU_SCRIPT_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")

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
    if not text:
        return []
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
