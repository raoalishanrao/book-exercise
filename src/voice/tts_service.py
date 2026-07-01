import logging

from src.config import optional_env, require_env
from src.voice.audio_merge import merge_wav_bytes
from src.voice.text_clean import clean_for_speech, split_kokoro_segments

logger = logging.getLogger("tutor.voice.tts")

_OPENAI_MAX_CHARS = 4096
_hf_client = None


def voice_enabled() -> bool:
    return optional_env("VOICE_ENABLED", "true").lower() in {"1", "true", "yes", "on"}


def tts_provider() -> str:
    return optional_env("TTS_PROVIDER", "edge").lower()


def tts_output_media_type() -> str:
    if tts_provider() in {"edge", "microsoft"}:
        return "audio/mpeg"
    return "audio/wav"


def kokoro_tts_enabled() -> bool:
    return optional_env("KOKORO_TTS_ENABLED", "true").lower() in {"1", "true", "yes", "on"}


def tts_available() -> bool:
    if not voice_enabled():
        return False
    provider = tts_provider()
    if provider == "openai":
        return bool(optional_env("OPENAI_API_KEY"))
    if provider in {"edge", "microsoft"}:
        from src.voice.edge_tts_service import edge_tts_available

        return edge_tts_available()
    if provider == "kokoro":
        backend = optional_env("KOKORO_BACKEND", "local").lower()
        if backend == "local":
            from src.voice.kokoro_local_service import kokoro_local_available

            return kokoro_local_available()
        return kokoro_tts_enabled() and bool(optional_env("HF_TOKEN"))
    if provider in {"mms", "urdu", "roman_urdu"}:
        from src.voice.mms_tts_service import mms_tts_available

        return mms_tts_available()
    return False


def _chunk_for_tts(text: str, max_len: int = _OPENAI_MAX_CHARS) -> list[str]:
    if len(text) <= max_len:
        return [text]

    chunks: list[str] = []
    rest = text
    while rest:
        if len(rest) <= max_len:
            chunks.append(rest.strip())
            break
        cut = rest.rfind(". ", 0, max_len)
        if cut < max_len // 2:
            cut = rest.rfind(" ", 0, max_len)
        if cut < max_len // 2:
            cut = max_len
        chunk = rest[:cut].strip()
        if chunk:
            chunks.append(chunk)
        rest = rest[cut:].strip()
    return chunks


def _get_hf_client():
    global _hf_client
    if _hf_client is None:
        from huggingface_hub import InferenceClient

        # Omit provider= to avoid fal-ai health-check noise; cloud still routes via HF.
        _hf_client = InferenceClient(api_key=require_env("HF_TOKEN"))
    return _hf_client


def _voice_for_lang(lang_code: str) -> str:
    # Kokoro lang h = Hindi (used for Urdu script + Roman Urdu tutor replies)
    if lang_code == "h":
        return optional_env("KOKORO_VOICE_UR", "hf_alpha")
    return optional_env("KOKORO_VOICE_EN", "af_heart")


class OpenAITTSService:
    def __init__(self) -> None:
        from openai import OpenAI

        self.client = OpenAI(api_key=require_env("OPENAI_API_KEY"))
        self.model = optional_env("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
        self.voice = optional_env("OPENAI_TTS_VOICE", "coral")
        self.response_format = optional_env("OPENAI_TTS_FORMAT", "wav")
        self.instructions = optional_env(
            "OPENAI_TTS_INSTRUCTIONS",
            "You are Ustaad Jee, a warm polite female Pakistani physics teacher. "
            "Speak Roman Urdu naturally using Latin letters (beta, aap, main, mein, kar sakti hoon). "
            "Pronounce English physics terms clearly: Physics, force, velocity, km/h. "
            "Sound caring, patient, and encouraging — like a real school teacher in Lahore.",
        )

    def _supports_instructions(self) -> bool:
        return "mini-tts" in self.model

    def _synthesize_chunk(self, text: str, *, speed: float) -> bytes:
        kwargs: dict = {
            "model": self.model,
            "voice": self.voice,
            "input": text,
            "response_format": self.response_format,
            "speed": speed,
        }
        if self.instructions and self._supports_instructions():
            kwargs["instructions"] = self.instructions

        logger.info(
            "OpenAI TTS model=%s voice=%s chars=%s format=%s",
            self.model,
            self.voice,
            len(text),
            self.response_format,
        )
        response = self.client.audio.speech.create(**kwargs)
        audio = response.content
        if not audio:
            raise ValueError("OpenAI TTS returned empty audio")
        return audio

    def synthesize(self, text: str, *, speed: float = 1.0) -> bytes:
        cleaned = clean_for_speech(text)
        if not cleaned:
            raise ValueError("No speakable text after cleaning")

        chunks = _chunk_for_tts(cleaned)
        parts = [self._synthesize_chunk(chunk, speed=speed) for chunk in chunks]

        if self.response_format == "wav":
            return merge_wav_bytes(parts)
        if len(parts) == 1:
            return parts[0]
        raise ValueError("Cannot merge non-wav OpenAI TTS chunks; set OPENAI_TTS_FORMAT=wav")


class KokoroTTSService:
    def synthesize(self, text: str, *, speed: float = 1.0) -> bytes:
        backend = optional_env("KOKORO_BACKEND", "local").lower()
        if backend == "local":
            from src.voice.kokoro_local_service import KokoroLocalService

            return KokoroLocalService().synthesize(text, speed=speed)
        return self._synthesize_cloud(text, speed=speed)

    def _synthesize_cloud(self, text: str, *, speed: float = 1.0) -> bytes:
        cleaned = clean_for_speech(text)
        if not cleaned:
            raise ValueError("No speakable text after cleaning")

        client = _get_hf_client()
        model = optional_env("HF_TTS_MODEL", "hexgrad/Kokoro-82M")
        # fal-ai only supports American English Kokoro voices (not hf_alpha Hindi).
        lang_mode = optional_env("KOKORO_LANG_MODE", "english")
        segments = split_kokoro_segments(cleaned, mode=lang_mode)
        audio_parts: list[bytes] = []

        for lang_code, segment_text in segments:
            voice = _voice_for_lang(lang_code)
            logger.info(
                "Cloud Kokoro model=%s lang=%s voice=%s chars=%s",
                model,
                lang_code,
                voice,
                len(segment_text),
            )
            audio = client.text_to_speech(
                segment_text,
                model=model,
                extra_body={"voice": voice, "speed": speed},
            )
            if audio:
                audio_parts.append(audio)

        if not audio_parts:
            raise ValueError("TTS produced no audio")

        return merge_wav_bytes(audio_parts)


class TTSService:
    def synthesize(
        self,
        text: str,
        *,
        speed: float = 1.0,
        speech_ready: bool = False,
    ) -> bytes:
        provider = tts_provider()
        if provider == "openai":
            return OpenAITTSService().synthesize(text, speed=speed)
        if provider in {"edge", "microsoft"}:
            from src.voice.edge_tts_service import EdgeTTSService

            return EdgeTTSService().synthesize(
                text,
                speed=speed,
                speech_ready=speech_ready,
            )
        if provider in {"mms", "urdu", "roman_urdu"}:
            from src.voice.mms_tts_service import MMSTTSService

            return MMSTTSService().synthesize(text, speed=speed)
        return KokoroTTSService().synthesize(text, speed=speed)
