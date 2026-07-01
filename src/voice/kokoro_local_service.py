"""Local Kokoro TTS — no Hugging Face Inference credits required."""

import io
import logging
import re

import numpy as np
import scipy.io.wavfile

from src.config import optional_env
from src.voice.audio_merge import merge_wav_bytes
from src.voice.text_clean import clean_for_speech, split_kokoro_segments

logger = logging.getLogger("tutor.voice.kokoro_local")

_SAMPLE_RATE = 24000
_pipelines: dict[str, object] = {}


def kokoro_local_available() -> bool:
    try:
        from kokoro import KPipeline  # noqa: F401

        return True
    except ImportError:
        return False


def _voice_for_lang(lang_code: str) -> str:
    if lang_code == "h":
        return optional_env("KOKORO_VOICE_UR", "hf_beta")
    return optional_env("KOKORO_VOICE_EN", "af_bella")


def _effective_speed(requested: float) -> float:
    scale = float(optional_env("KOKORO_SPEED", "0.9"))
    return max(0.7, min(1.15, requested * scale))


def _get_pipeline(lang_code: str):
    from kokoro import KPipeline

    if lang_code not in _pipelines:
        logger.info("Loading local Kokoro pipeline lang=%s", lang_code)
        _pipelines[lang_code] = KPipeline(lang_code=lang_code)
    return _pipelines[lang_code]


def _audio_to_wav(audio: np.ndarray) -> bytes:
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767).astype(np.int16)
    buf = io.BytesIO()
    scipy.io.wavfile.write(buf, _SAMPLE_RATE, pcm)
    return buf.getvalue()


def _split_for_prosody(text: str) -> list[str]:
    """Break long replies at sentence boundaries for more natural pauses."""
    parts = re.split(r"(?<=[.!?۔])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()] or [text]


class KokoroLocalService:
    def synthesize(self, text: str, *, speed: float = 1.0) -> bytes:
        cleaned = clean_for_speech(text)
        if not cleaned:
            raise ValueError("No speakable text after cleaning")

        lang_mode = optional_env("KOKORO_LANG_MODE", "roman_mixed")
        segments = split_kokoro_segments(cleaned, mode=lang_mode)
        audio_parts: list[bytes] = []
        rate = _effective_speed(speed)
        split_pattern = optional_env("KOKORO_SPLIT_PATTERN", r"\n+")

        for lang_code, segment_text in segments:
            voice = _voice_for_lang(lang_code)
            pipeline = _get_pipeline(lang_code)
            for sentence in _split_for_prosody(segment_text):
                logger.info(
                    "Local Kokoro lang=%s voice=%s speed=%.2f chars=%s preview=%r",
                    lang_code,
                    voice,
                    rate,
                    len(sentence),
                    sentence[:60],
                )
                chunks: list[np.ndarray] = []
                for _gs, _ps, audio in pipeline(
                    sentence,
                    voice=voice,
                    speed=rate,
                    split_pattern=split_pattern,
                ):
                    chunks.append(np.asarray(audio, dtype=np.float32))
                if chunks:
                    audio_parts.append(_audio_to_wav(np.concatenate(chunks)))

        if not audio_parts:
            raise ValueError("Local Kokoro produced no audio")

        return merge_wav_bytes(audio_parts, pause_ms=180)
