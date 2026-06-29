import io
import wave


def merge_wav_bytes(parts: list[bytes], *, pause_ms: int = 120) -> bytes:
    if not parts:
        raise ValueError("No audio to merge")
    if len(parts) == 1:
        return parts[0]

    params = None
    frames: list[bytes] = []
    for part in parts:
        with wave.open(io.BytesIO(part), "rb") as reader:
            current = reader.getparams()
            if params is None:
                params = current
            elif current[:4] != params[:4]:
                raise ValueError("Incompatible audio segments from TTS provider")
            frames.append(reader.readframes(reader.getnframes()))

    assert params is not None
    silence = b"\x00" * int(params.framerate * pause_ms / 1000) * params.nchannels * params.sampwidth

    out = io.BytesIO()
    with wave.open(out, "wb") as writer:
        writer.setparams(params)
        for index, frame in enumerate(frames):
            writer.writeframes(frame)
            if index < len(frames) - 1:
                writer.writeframes(silence)
    return out.getvalue()
