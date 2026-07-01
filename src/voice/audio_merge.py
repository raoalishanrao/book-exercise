import io
import logging
import subprocess
import wave

logger = logging.getLogger("tutor.voice.audio_merge")


def _read_wav_pcm(part: bytes) -> tuple[tuple[int, int, int], bytes]:
    """Read WAV PCM; ffmpeg pipe WAV may report bogus nframes."""
    with wave.open(io.BytesIO(part), "rb") as reader:
        params = (reader.getnchannels(), reader.getsampwidth(), reader.getframerate())
        frames = bytearray()
        while True:
            block = reader.readframes(4096)
            if not block:
                break
            frames.extend(block)
        return params, bytes(frames)


def merge_wav_bytes(parts: list[bytes], *, pause_ms: int = 120) -> bytes:
    if not parts:
        raise ValueError("No audio to merge")
    if len(parts) == 1:
        return parts[0]

    params: tuple[int, int, int] | None = None
    frames: list[bytes] = []
    for part in parts:
        current_params, pcm = _read_wav_pcm(part)
        if params is None:
            params = current_params
        elif current_params != params:
            raise ValueError("Incompatible audio segments from TTS provider")
        frames.append(pcm)

    assert params is not None
    nchannels, sampwidth, framerate = params
    silence = b"\x00" * int(framerate * pause_ms / 1000) * nchannels * sampwidth

    out = io.BytesIO()
    with wave.open(out, "wb") as writer:
        writer.setnchannels(nchannels)
        writer.setsampwidth(sampwidth)
        writer.setframerate(framerate)
        for index, frame in enumerate(frames):
            writer.writeframes(frame)
            if index < len(frames) - 1:
                writer.writeframes(silence)
    return out.getvalue()


def _ffmpeg_exe() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def _mp3_to_wav(mp3_data: bytes) -> bytes:
    result = subprocess.run(
        [_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-i", "pipe:0", "-f", "wav", "pipe:1"],
        input=mp3_data,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0 or not result.stdout:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace") or "mp3 decode failed")
    return result.stdout


def _wav_to_mp3(wav_data: bytes) -> bytes:
    result = subprocess.run(
        [
            _ffmpeg_exe(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            "pipe:0",
            "-f",
            "mp3",
            "-b:a",
            "128k",
            "pipe:1",
        ],
        input=wav_data,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0 or not result.stdout:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace") or "mp3 encode failed")
    return result.stdout


def merge_mp3_bytes(parts: list[bytes], *, pause_ms: int = 0) -> bytes:
    """Concatenate MP3 segments with optional silence gaps (Edge TTS)."""
    if not parts:
        raise ValueError("No audio to merge")
    if len(parts) == 1:
        return parts[0]
    if pause_ms <= 0:
        return b"".join(parts)

    try:
        wav_parts = [_mp3_to_wav(part) for part in parts]
        merged_wav = merge_wav_bytes(wav_parts, pause_ms=pause_ms)
        return _wav_to_mp3(merged_wav)
    except Exception as exc:
        logger.warning("MP3 pause merge unavailable (%s) — clips joined without gaps", exc)
        return b"".join(parts)
