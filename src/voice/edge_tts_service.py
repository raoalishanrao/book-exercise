"""Microsoft Edge neural TTS — polite Pakistani Urdu female voice (free, no API key)."""

import asyncio
import logging

from src.config import optional_env
from src.voice.audio_merge import merge_mp3_bytes
from src.voice.edge_ssml import (
    combined_rate,
    pitch_string,
    split_urdu_speech_units,
    volume_string,
)
from src.voice.physics_speech import use_hybrid_physics_tts
from src.voice.text_clean import clean_for_speech

logger = logging.getLogger("tutor.voice.edge_tts")

_DEFAULT_VOICE_UR = "ur-PK-UzmaNeural"
_DEFAULT_VOICE_EN = "en-IN-NeerjaNeural"
_DEFAULT_CHUNK_PAUSE_MS = 280
_DEFAULT_HYBRID_PAUSE_MS = 70


def edge_tts_available() -> bool:
    try:
        import edge_tts  # noqa: F401

        return True
    except ImportError:
        return False


def _urdu_voice() -> str:
    return optional_env("EDGE_TTS_VOICE_UR", _DEFAULT_VOICE_UR)


def _english_voice() -> str:
    return optional_env("EDGE_TTS_VOICE_EN", _DEFAULT_VOICE_EN)


def _chunk_pause_ms() -> int:
    try:
        return max(0, int(optional_env("EDGE_TTS_CHUNK_PAUSE_MS", str(_DEFAULT_CHUNK_PAUSE_MS))))
    except ValueError:
        return _DEFAULT_CHUNK_PAUSE_MS


def _hybrid_pause_ms() -> int:
    try:
        return max(0, int(optional_env("EDGE_TTS_HYBRID_PAUSE_MS", str(_DEFAULT_HYBRID_PAUSE_MS))))
    except ValueError:
        return _DEFAULT_HYBRID_PAUSE_MS


async def _synthesize_async(
    text: str,
    *,
    voice: str,
    rate: str,
    pitch: str,
    volume: str,
) -> bytes:
    import edge_tts

    communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch, volume=volume)
    parts: list[bytes] = []
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            parts.append(chunk["data"])
    if not parts:
        raise ValueError("Edge TTS returned no audio")
    return b"".join(parts)


def _synthesize(
    text: str,
    *,
    voice: str,
    rate: str,
    pitch: str,
    volume: str,
) -> bytes:
    return asyncio.run(
        _synthesize_async(text, voice=voice, rate=rate, pitch=pitch, volume=volume)
    )


def _sentence_synthesis_enabled() -> bool:
    return optional_env("EDGE_TTS_SENTENCE_MODE", "true").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _expand_speech_parts(parts: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Split Urdu chunks into short phrases for calmer, human-like pacing."""
    if not _sentence_synthesis_enabled():
        return parts
    expanded: list[tuple[str, str]] = []
    for lang, chunk in parts:
        if lang != "ur" or len(chunk) <= _sentence_pause_limit():
            expanded.append((lang, chunk))
            continue
        for unit in split_urdu_speech_units(chunk):
            if unit.strip():
                expanded.append((lang, unit))
    return expanded or parts


def _sentence_pause_limit() -> int:
    try:
        return max(80, int(optional_env("EDGE_TTS_SENTENCE_CHARS", "150")))
    except ValueError:
        return 150


def _synthesize_parts(
    parts: list[tuple[str, str]],
    *,
    speed: float,
    pause_ms: int,
) -> bytes:
    ur_voice = _urdu_voice()
    en_voice = _english_voice()
    rate_ur = combined_rate(speed)
    rate_en = combined_rate(speed)
    pitch = pitch_string()
    volume = volume_string()

    parts = _expand_speech_parts(parts)
    audio_parts: list[bytes] = []
    for index, (lang, chunk) in enumerate(parts, start=1):
        voice = en_voice if lang == "en" else ur_voice
        rate = rate_en if lang == "en" else rate_ur
        logger.info(
            "Edge TTS %s chunk %s/%s voice=%s chars=%s preview=%r",
            lang,
            index,
            len(parts),
            voice,
            len(chunk),
            chunk[:50],
        )
        audio_parts.append(
            _synthesize(chunk, voice=voice, rate=rate, pitch=pitch, volume=volume)
        )

    if len(audio_parts) == 1:
        return audio_parts[0]
    return merge_mp3_bytes(audio_parts, pause_ms=pause_ms)


class EdgeTTSService:
    def synthesize(
        self,
        text: str,
        *,
        speed: float = 1.0,
        speech_ready: bool = False,
    ) -> bytes:
        cleaned = clean_for_speech(text)
        if not cleaned:
            raise ValueError("No speakable text after cleaning")

        lang_mode = optional_env("EDGE_TTS_LANG_MODE", "roman_mixed")

        if speech_ready:
            from src.voice.roman_urdu_speech import prepare_prepared_urdu_speech_chunks

            speech_chunks = prepare_prepared_urdu_speech_chunks(cleaned)
            if not speech_chunks:
                raise ValueError("No Urdu speech chunks after preparation")

            logger.info(
                "Edge TTS speech-ready chunks=%s total_chars=%s preview=%r",
                len(speech_chunks),
                sum(len(c) for c in speech_chunks),
                speech_chunks[0][:80],
            )
            parts = [("ur", chunk) for chunk in speech_chunks]
            return _synthesize_parts(parts, speed=speed, pause_ms=_chunk_pause_ms())

        if lang_mode == "roman_mixed" and use_hybrid_physics_tts():
            from src.voice.roman_urdu_speech import prepare_hybrid_speech_segments

            segments = prepare_hybrid_speech_segments(cleaned)
            logger.info(
                "Edge hybrid TTS ur=%s en=%s total_chars=%s",
                sum(1 for lang, _ in segments if lang == "ur"),
                sum(1 for lang, _ in segments if lang == "en"),
                sum(len(s) for _, s in segments),
            )
            return _synthesize_parts(
                segments,
                speed=speed,
                pause_ms=_hybrid_pause_ms(),
            )

        if lang_mode == "roman_mixed":
            from src.voice.roman_urdu_speech import prepare_urdu_speech_chunks

            speech_chunks = prepare_urdu_speech_chunks(cleaned)
            if not speech_chunks:
                raise ValueError("No Urdu speech chunks after preparation")

            logger.info(
                "Edge TTS ur-only chunks=%s total_chars=%s preview=%r",
                len(speech_chunks),
                sum(len(c) for c in speech_chunks),
                speech_chunks[0][:80],
            )
            parts = [("ur", chunk) for chunk in speech_chunks]
            return _synthesize_parts(parts, speed=speed, pause_ms=_chunk_pause_ms())

        from src.voice.text_clean import split_kokoro_segments

        voice = _urdu_voice()
        rate = combined_rate(speed)
        pitch = pitch_string()
        volume = volume_string()
        segments = split_kokoro_segments(cleaned, mode=lang_mode)
        chunks = [seg for _, seg in segments if seg.strip()]
        audio_parts = [
            _synthesize(chunk, voice=voice, rate=rate, pitch=pitch, volume=volume)
            for chunk in chunks
        ]
        if len(audio_parts) == 1:
            return audio_parts[0]
        return merge_mp3_bytes(audio_parts, pause_ms=_chunk_pause_ms())
