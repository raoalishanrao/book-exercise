import logging

from huggingface_hub import InferenceClient

from src.config import optional_env, require_env
from src.voice.audio_merge import merge_wav_bytes
from src.voice.text_clean import clean_for_speech, split_script_segments

logger = logging.getLogger("tutor.voice.tts")

_client: InferenceClient | None = None


def kokoro_tts_enabled() -> bool:
    return optional_env("KOKORO_TTS_ENABLED", "false").lower() in {"1", "true", "yes", "on"}


def tts_available() -> bool:
    return (
        voice_enabled()
        and kokoro_tts_enabled()
        and bool(optional_env("HF_TOKEN"))
    )


def voice_enabled() -> bool:
    return optional_env("VOICE_ENABLED", "true").lower() in {"1", "true", "yes", "on"}


def _get_client() -> InferenceClient:
    global _client
    if _client is None:
        _client = InferenceClient(
            provider=optional_env("HF_TTS_PROVIDER", "fal-ai"),
            api_key=require_env("HF_TOKEN"),
        )
    return _client


def _voice_for_lang(lang_code: str) -> str:
    if lang_code == "h":
        return optional_env("KOKORO_VOICE_UR", "hf_alpha")
    return optional_env("KOKORO_VOICE_EN", "af_heart")


class TTSService:
    def synthesize(self, text: str, *, speed: float = 1.0) -> bytes:
        cleaned = clean_for_speech(text)
        if not cleaned:
            raise ValueError("No speakable text after cleaning")

        client = _get_client()
        model = optional_env("HF_TTS_MODEL", "hexgrad/Kokoro-82M")
        segments = split_script_segments(cleaned)
        audio_parts: list[bytes] = []

        for lang_code, segment_text in segments:
            voice = _voice_for_lang(lang_code)
            logger.info("HF TTS model=%s voice=%s chars=%s", model, voice, len(segment_text))
            audio = client.text_to_speech(
                segment_text,
                model=model,
                extra_body={"voice": voice, "speed": speed},
            )
            if not audio:
                continue
            audio_parts.append(audio)

        if not audio_parts:
            raise ValueError("TTS produced no audio")

        return merge_wav_bytes(audio_parts)
