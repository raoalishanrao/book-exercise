import logging
from pathlib import Path

from groq import BadRequestError, Groq

from src.config import optional_env, require_env
from src.voice.script_fix import normalize_transcript
from src.voice.transcript_quality import (
    detect_spoken_language,
    is_plausible_transcript,
    is_prompt_echo,
    is_whisper_hallucination,
    pick_bilingual_candidate,
    score_transcript,
)

logger = logging.getLogger("tutor.voice.stt")

_MIN_ACCEPT_SCORE = 12.0


def voice_enabled() -> bool:
    return optional_env("VOICE_ENABLED", "true").lower() in {"1", "true", "yes", "on"}


def stt_available() -> bool:
    return voice_enabled() and bool(optional_env("GROQ_API_KEY"))


class STTService:
    def __init__(self) -> None:
        self.model = optional_env("GROQ_WHISPER_MODEL", "whisper-large-v3")
        self.fallback_model = optional_env("GROQ_WHISPER_FALLBACK_MODEL", "")
        self.output_mode = optional_env("STT_OUTPUT", "auto")
        self.client = Groq(api_key=require_env("GROQ_API_KEY"))

    def _transcribe_raw(
        self,
        audio_bytes: bytes,
        filename: str,
        *,
        language: str,
        model: str,
    ) -> str:
        kwargs: dict = {
            "file": (filename, audio_bytes),
            "model": model,
            "response_format": "json",
            "language": language,
            "temperature": 0,
        }
        try:
            result = self.client.audio.transcriptions.create(**kwargs)
            return (result.text or "").strip()
        except BadRequestError as exc:
            msg = str(exc).lower()
            if "too short" in msg or "invalid" in msg:
                logger.warning("Groq rejected audio file: %s", exc)
                return ""
            raise

    def _language_passes(self, lang_setting: str) -> list[str]:
        if lang_setting == "en":
            return ["en"]
        if lang_setting == "ur":
            return ["ur"]
        return ["ur", "en"]

    def _collect_candidates(
        self,
        audio_bytes: bytes,
        filename: str,
        languages: list[str],
        models: list[str],
    ) -> list[tuple[float, str, str, str]]:
        candidates: list[tuple[float, str, str, str]] = []
        for model in models:
            is_primary = model == self.model
            for lang in languages:
                logger.info(
                    "Groq transcribe model=%s language=%s bytes=%s",
                    model,
                    lang,
                    len(audio_bytes),
                )
                raw = self._transcribe_raw(audio_bytes, filename, language=lang, model=model)
                if not raw or raw in {".", "...", "…"}:
                    continue
                if is_whisper_hallucination(raw):
                    logger.info("Rejected hallucination model=%s lang=%s raw=%r", model, lang, raw[:120])
                    continue
                rating = score_transcript(raw, whisper_lang=lang)
                if is_primary:
                    rating += 8.0
                detected = detect_spoken_language(raw)
                logger.info(
                    "Candidate model=%s lang=%s score=%.1f detected=%s raw=%r",
                    model,
                    lang,
                    rating,
                    detected,
                    raw[:120],
                )
                candidates.append((rating, model, lang, raw))
        return candidates

    def _pick_best(
        self,
        candidates: list[tuple[float, str, str, str]],
        lang_setting: str,
    ) -> tuple[float, str, str, str] | None:
        valid = [c for c in candidates if c[0] >= _MIN_ACCEPT_SCORE]
        if not valid:
            return None

        if lang_setting in {"both", "auto", ""}:
            picked = pick_bilingual_candidate(valid)
            if picked:
                logger.info(
                    "Bilingual pick whisper_lang=%s detected=%s",
                    picked[2],
                    detect_spoken_language(picked[3]),
                )
                return picked

        if lang_setting == "ur":
            ur_only = [c for c in valid if c[2] == "ur"]
            if ur_only:
                return max(ur_only, key=lambda item: item[0])
        if lang_setting == "en":
            en_only = [c for c in valid if c[2] == "en"]
            if en_only:
                return max(en_only, key=lambda item: item[0])

        valid.sort(key=lambda item: item[0], reverse=True)
        return valid[0]

    def _finalize(self, raw: str, whisper_lang: str, output_mode: str | None = None) -> dict:
        empty = {"text": "", "language": whisper_lang, "language_probability": None}
        if not raw or is_prompt_echo(raw) or is_whisper_hallucination(raw):
            return empty

        mode = output_mode or self.output_mode
        text = normalize_transcript(raw, output_mode=mode)
        if not text:
            return empty
        if not is_plausible_transcript(text) and not is_plausible_transcript(raw):
            return empty

        out_lang = "en" if mode == "english" or whisper_lang == "en" else "ur"
        logger.info("Transcript raw=%r out=%r lang=%s mode=%s", raw[:160], text[:160], whisper_lang, mode)
        return {"text": text, "language": out_lang, "language_probability": None}

    def transcribe(
        self,
        audio_path: str | Path,
        *,
        language: str | None = None,
        output: str | None = None,
    ) -> dict:
        path = Path(audio_path)
        lang_setting = optional_env("STT_LANGUAGE", "both")
        if language and language not in {"auto", "", "both"}:
            lang_setting = language
        elif language in {"auto", "", "both"}:
            lang_setting = optional_env("STT_LANGUAGE", "both")

        output_mode = output or self.output_mode
        if lang_setting == "ur":
            output_mode = output or "urdu"
        elif lang_setting == "en":
            output_mode = output or "english"

        audio_bytes = path.read_bytes()
        filename = path.name
        min_bytes = int(optional_env("STT_MIN_AUDIO_BYTES", "6000"))

        if len(audio_bytes) < min_bytes:
            logger.warning("Audio too short bytes=%s min=%s", len(audio_bytes), min_bytes)
            return {"text": "", "language": lang_setting, "language_probability": None}

        languages = self._language_passes(lang_setting)
        empty = {"text": "", "language": lang_setting, "language_probability": None}

        candidates = self._collect_candidates(audio_bytes, filename, languages, [self.model])
        best = self._pick_best(candidates, lang_setting)

        if not best and self.fallback_model and self.fallback_model != self.model:
            logger.info("Primary model had no good transcript — trying %s", self.fallback_model)
            fallback = self._collect_candidates(audio_bytes, filename, languages, [self.fallback_model])
            candidates.extend(fallback)
            best = self._pick_best(fallback, lang_setting)

        if not best:
            logger.warning("No valid transcript from audio")
            return empty

        best_score, best_model, best_lang, best_raw = best
        logger.info("Selected model=%s lang=%s score=%.1f", best_model, best_lang, best_score)
        return self._finalize(best_raw, best_lang, output_mode)
