"""Convert LaTeX-style physics notation to plain text for the chat widget."""

import re


def _clean_latex_fragment(fragment: str) -> str:
    s = fragment.strip()
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\frac\{([^}]*)\}\{([^}]*)\}", r"(\1)/(\2)", s)
    s = re.sub(r"\\begin\{[a-zA-Z*]+\}|\\end\{[a-zA-Z*]+\}", "", s)
    s = re.sub(r"\\,", " ", s)
    s = re.sub(r"\\times", " x ", s)
    s = re.sub(r"\\cdot", " · ", s)
    s = re.sub(r"&=", "=", s)
    s = re.sub(r"\\\\", "\n", s)
    s = re.sub(r"\^\{([^}]+)\}", r"^(\1)", s)
    s = re.sub(r"_\{([^}]+)\}", r"_\1", s)
    s = re.sub(r"\\left|\\right", "", s)
    # Orphan \text fragments without braces (model typo)
    s = re.sub(r"\\text\s+", "", s)
    s = re.sub(r"\\(?=[a-zA-Z])", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _space_number_unit(text: str) -> str:
    """30km/h -> 30 km/h, 10m/s -> 10 m/s. Preserves newlines."""
    units = (
        r"km/h|m/s\^?2?|m/s|ms\^-?1|kmh\^-?1|"
        r"km|m|s|kg|N|J|W|Pa|Hz|A|V|C"
    )
    lines = []
    for line in text.split("\n"):
        line = re.sub(rf"(\d)({units})\b", r"\1 \2", line)
        line = re.sub(r"[ \t]+", " ", line).strip()
        lines.append(line)
    return "\n".join(lines)


def normalize_response_structure(text: str) -> str:
    """Insert line breaks before board-style steps when the model omits them."""
    if not text:
        return text

    text = re.sub(
        r"(?<=[^\n])\s*(Step\s+\d+\s*[:\-—])",
        r"\n\n\1",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def normalize_physics_math(text: str) -> str:
    """Strip LaTeX delimiters/commands; keep readable SI units for the widget."""
    if not text:
        return text

    def repl_display(match: re.Match[str]) -> str:
        body = _clean_latex_fragment(match.group(1))
        return f"\n{body}\n" if body else ""

    text = re.sub(r"\$\$([^$]+)\$\$", repl_display, text, flags=re.DOTALL)
    text = re.sub(r"\$([^$\n]+)\$", lambda m: _clean_latex_fragment(m.group(1)), text)
    text = re.sub(r"\\text\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\\s+", " ", text)
    text = _space_number_unit(text)
    text = normalize_response_structure(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
