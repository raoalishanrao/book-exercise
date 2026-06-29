import re

URDU_ARABIC_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")
DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
LATIN_RE = re.compile(r"[A-Za-z]")

# Whisper hallucinations in languages we never expect for this app.
_FOREIGN_SCRIPT_RE = re.compile(
    r"[\u0400-\u04FF"  # Cyrillic
    r"\u0370-\u03FF"  # Greek
    r"\u0590-\u05FF"  # Hebrew
    r"\u0B80-\u0BFF"  # Tamil
    r"\u0980-\u09FF"  # Bengali
    r"]"
)
_ICELANDIC_RE = re.compile(r"[ÞþðÆæ]")
_NORDIC_HALLUCINATION = re.compile(
    r"\b(þeir|stuðið|heim|hérna|svo|að|færstu)\b",
    re.IGNORECASE,
)

_PLAUSIBLE_ENGLISH = re.compile(
    r"\b(the|you|your|can|could|help|what|how|why|when|where|which|who|"
    r"english|newton|force|physics|chapter|page|please|explain|tell|about|know|"
    r"want|motion|law|first|second|third|difference|between|and|is|are|was|"
    r"do|does|did|have|has|had|this|that|with|for|from|mean|means)\b",
    re.IGNORECASE,
)

_PLAUSIBLE_ROMAN = re.compile(
    r"\b("
    r"ap|aap|meri|mein|main|mujhe|mujh|kya|kaise|kar|karna|madad|sakte|sakti|sakta|"
    r"physics|fiziks|samajh|samjh|nahi|nahin|hai|hon|hoon|beta|sir|miss|help|"
    r"what|how|why|explain|chapter|topic|velocity|force|inertia|numerical|"
    r"tum|ki|ke|ko|se|par|theek|assalam|shukriya|please|can|you|my|in|"
    r"urdu|english|problem|question|solve|bata|batao|samjha|samjhao|janana|parh"
    r")\b",
    re.IGNORECASE,
)

_GIBBERISH = re.compile(
    r"\b(amanda|dversa|subscribe|thank you for watching|mbc|nolj|metn law)\b",
    re.IGNORECASE,
)
_ENGLISH_FILLER = re.compile(
    r"^(thank you|thanks|thank you\.|you\.?|okay|ok\.?|hmm\.?)\.?$",
    re.IGNORECASE,
)
_URDU_CUES = re.compile(
    r"آپ|میں|مجھے|میری|میرے|کو|معلوم|فرق|کیا|کیسے|ہے|ہیں|ہو|مدد|کر|سکتے|سکتی|نہیں|براہ|"
    r"فزکس|physics|نمبر|سوال|باب|سبق|صفحہ|page|کہ|اور"
)
_URDU_GARBAGE_TOKENS = re.compile(r"مرین|ری مرین")

# Known prompt phrases that Whisper echoes instead of real speech.
_PROMPT_ECHOES = (
    "میں نویں جماعت کی فزکس پڑھ رہا ہوں",
    "آپ میری کیا مدد کر سکتے ہیں",
    "aap meri kya madad kar sakte hain",
    "main navin jamaat ki physics",
)


def _normalize_for_compare(text: str) -> str:
    text = text.strip().lower()
    text = text.replace("۔", "").replace(".", "").replace("،", ",")
    return re.sub(r"\s+", " ", text)


def is_prompt_echo(text: str, prompt: str | None = None) -> bool:
    """True only when the transcript is essentially the STT prompt, not real speech."""
    norm = _normalize_for_compare(text)
    if not norm:
        return False

    for fragment in _PROMPT_ECHOES:
        frag_norm = _normalize_for_compare(fragment)
        if len(frag_norm) < 8:
            continue
        if norm == frag_norm:
            return True
        # Longer real speech may start with a common phrase — only flag short echoes.
        if frag_norm in norm and len(norm) <= len(frag_norm) * 1.25:
            return True

    if prompt:
        prompt_norm = _normalize_for_compare(prompt)
        if len(prompt_norm) >= 12 and norm == prompt_norm:
            return True
        if len(prompt_norm) >= 12 and prompt_norm in norm and len(norm) <= len(prompt_norm) * 1.25:
            return True
        for part in re.split(r"[.!?؟،]\s*", prompt):
            part_norm = _normalize_for_compare(part)
            if len(part_norm) >= 12 and part_norm == norm:
                return True
    return False


def is_foreign_hallucination(text: str) -> bool:
    """Reject Whisper output in unexpected languages (Icelandic, Cyrillic, etc.)."""
    cleaned = (text or "").strip()
    if not cleaned:
        return False
    if _ICELANDIC_RE.search(cleaned):
        return True
    if _NORDIC_HALLUCINATION.search(cleaned):
        return True
    if _FOREIGN_SCRIPT_RE.search(cleaned):
        return True
    # Latin text with no English or Roman-Urdu cues is likely a wrong-language guess.
    if LATIN_RE.search(cleaned) and not URDU_ARABIC_RE.search(cleaned):
        if not _PLAUSIBLE_ENGLISH.search(cleaned) and not _PLAUSIBLE_ROMAN.search(cleaned):
            words = re.findall(r"[a-zA-Z']+", cleaned)
            if len(words) >= 4:
                return True
    return False


def is_urdu_hallucination(text: str) -> bool:
    """Urdu-script text that is repetitive nonsense from Whisper."""
    cleaned = (text or "").strip()
    if not URDU_ARABIC_RE.search(cleaned):
        return False

    words = cleaned.split()
    if len(words) < 3:
        return False

    if _URDU_GARBAGE_TOKENS.search(cleaned):
        return True

    counts: dict[str, int] = {}
    for word in words:
        counts[word] = counts.get(word, 0) + 1
    if any(n >= 3 for n in counts.values()):
        return True

    # Mixed Urdu + English physics terms — skip bigram checks (e.g. "of motion" twice)
    if _URDU_CUES.search(cleaned):
        return False

    for i in range(len(words) - 1):
        bigram = f"{words[i]} {words[i+1]}"
        if len(bigram) > 7 and cleaned.count(bigram) >= 2:
            return True

    if len(words) >= 6 and not _URDU_CUES.search(cleaned):
        return True

    return False


def is_whisper_hallucination(text: str) -> bool:
    cleaned = (text or "").strip()
    if not cleaned:
        return False
    if _ENGLISH_FILLER.match(cleaned):
        return True
    if is_foreign_hallucination(cleaned):
        return True
    if is_urdu_hallucination(cleaned):
        return True
    return False


def _repetition_penalty(text: str) -> float:
    words = text.split()
    if len(words) < 8:
        return 0.0
    if len(set(words)) / len(words) < 0.38:
        return -50.0
    for size in range(min(7, len(words) // 2), 2, -1):
        for start in range(len(words) - size + 1):
            phrase = " ".join(words[start : start + size])
            if len(phrase) > 14 and text.count(phrase) >= 3:
                return -55.0
    return 0.0


def score_transcript(text: str, *, prompt: str | None = None, whisper_lang: str | None = None) -> float:
    """Higher is better when picking among Whisper attempts."""
    cleaned = (text or "").strip()
    if not cleaned or is_prompt_echo(cleaned, prompt):
        return -100.0
    if is_whisper_hallucination(cleaned):
        return -100.0
    if _GIBBERISH.search(cleaned):
        return -50.0

    score = _repetition_penalty(cleaned)
    if score <= -50:
        return score

    if not is_plausible_transcript(cleaned, prompt=prompt):
        score -= 10.0

    score += min(len(cleaned), 200) * 0.15
    score += len(cleaned.split()) * 3.0
    if _PLAUSIBLE_ROMAN.search(cleaned):
        score += 12.0
    if URDU_ARABIC_RE.search(cleaned):
        score += 10.0
        if whisper_lang == "ur":
            score += 8.0
        if re.search(r"پریدی|کنار(?!\s*ہے)|پیجپ|نالج|میٹن|اپنے مدد", cleaned):
            score -= 15.0
    if DEVANAGARI_RE.search(cleaned):
        score -= 25.0
    latin_words = re.findall(r"[a-zA-Z']+", cleaned)
    if len(latin_words) >= 3 and not URDU_ARABIC_RE.search(cleaned):
        if _PLAUSIBLE_ENGLISH.search(cleaned):
            score += 15.0
            if whisper_lang == "en":
                score += 8.0
    score += lang_match_bonus(cleaned, whisper_lang or "")
    return score


def is_plausible_transcript(text: str, *, prompt: str | None = None) -> bool:
    cleaned = (text or "").strip()
    if len(cleaned) < 4:
        return False
    if is_whisper_hallucination(cleaned):
        return False
    if _GIBBERISH.search(cleaned):
        return False
    if re.search(r"[_]", cleaned):
        return False
    if is_prompt_echo(cleaned, prompt):
        return False

    words = cleaned.split()

    if URDU_ARABIC_RE.search(cleaned):
        if len(words) < 2:
            return False
        if len(re.sub(r"\s+", "", cleaned)) < 10:
            return False
        return True

    if _PLAUSIBLE_ROMAN.search(cleaned):
        return len(words) >= 2 or len(cleaned) >= 12

    latin_words = re.findall(r"[a-zA-Z']+", cleaned)
    if len(latin_words) >= 2:
        return True
    return len(latin_words) == 1 and len(latin_words[0]) >= 6 and _has_vowel_run(latin_words[0])


def _has_vowel_run(word: str) -> bool:
    return bool(re.search(r"[aeiou]{1,}", word.lower()))


def detect_spoken_language(text: str) -> str:
    """Classify transcript content: en | ur | mixed | unknown."""
    cleaned = (text or "").strip()
    if not cleaned:
        return "unknown"

    has_arabic = bool(URDU_ARABIC_RE.search(cleaned))
    has_devanagari = bool(DEVANAGARI_RE.search(cleaned))
    latin_words = re.findall(r"[a-zA-Z']+", cleaned)
    english_hits = len(_PLAUSIBLE_ENGLISH.findall(cleaned))

    if has_arabic or has_devanagari:
        if len(latin_words) >= 2 and english_hits >= 1:
            return "mixed"
        return "ur"

    if english_hits >= 2 and len(latin_words) >= 3:
        return "en"
    if len(latin_words) >= 4 and english_hits >= 1:
        return "en"
    if _PLAUSIBLE_ROMAN.search(cleaned) and english_hits == 0:
        return "ur"
    if len(latin_words) >= 2:
        return "en"
    return "unknown"


def lang_match_bonus(text: str, whisper_lang: str) -> float:
    """Reward transcripts where Whisper language matches detected content."""
    detected = detect_spoken_language(text)
    if whisper_lang == "en":
        if detected == "en":
            return 40.0
        if detected == "mixed":
            return 10.0
        if detected == "ur":
            return -35.0
    if whisper_lang == "ur":
        if detected in {"ur", "mixed"}:
            return 40.0
        if detected == "en":
            return -35.0
    return 0.0


_STRONG_URDU_START = re.compile(
    r"^(کیا\s+(آپ|میں)|آپ\s+(کو|مجھے|سے|کی)|میں\s+|مجھے\s+|براہ)"
)


def pick_bilingual_candidate(
    candidates: list[tuple[float, str, str, str]],
) -> tuple[float, str, str, str] | None:
    """Choose Urdu vs English pass based on what was actually spoken."""
    by_whisper: dict[str, tuple[float, str, str, str]] = {}
    for item in candidates:
        lang = item[2]
        if lang not in by_whisper or item[0] > by_whisper[lang][0]:
            by_whisper[lang] = item

    ur_item = by_whisper.get("ur")
    en_item = by_whisper.get("en")

    if ur_item and en_item:
        ur_text = ur_item[3]
        en_text = en_item[3]
        ur_detected = detect_spoken_language(ur_text)
        en_detected = detect_spoken_language(en_text)
        ur_has_script = bool(URDU_ARABIC_RE.search(ur_text)) or bool(DEVANAGARI_RE.search(ur_text))
        en_is_english = en_detected == "en"
        ur_is_urdu = ur_detected in {"ur", "mixed"}
        ur_score, en_score = ur_item[0], en_item[0]

        # Clear Urdu sentence opening → spoke Urdu (en pass is just a translation)
        if ur_has_script and ur_is_urdu and _STRONG_URDU_START.search(ur_text.strip()):
            return ur_item

        # Clear English from en pass, ur pass is not a strong Urdu sentence
        if en_is_english and not _STRONG_URDU_START.search(ur_text.strip()):
            return en_item

        # Mixed Urdu + English physics terms in ur pass
        if ur_detected == "mixed" and ur_has_script:
            return ur_item

        # Ur pass got English text → user spoke English
        if ur_detected == "en" and en_is_english:
            return en_item

        # Fallback: better language-content match score
        if en_is_english and en_score >= ur_score - 5:
            return en_item
        if ur_is_urdu and ur_has_script:
            return ur_item

    if ur_item:
        return ur_item
    if en_item:
        return en_item
    if candidates:
        return max(candidates, key=lambda item: item[0])
    return None
