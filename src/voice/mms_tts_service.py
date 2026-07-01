"""Local Urdu TTS via Meta MMS VITS — Roman Latin or Arabic script models."""

import io
import logging
import re

import scipy.io.wavfile

from src.config import optional_env
from src.voice.audio_merge import merge_wav_bytes
from src.voice.script_fix import URDU_ARABIC_RE, _latin_ratio, to_urdu_script
from src.voice.text_clean import clean_for_speech

logger = logging.getLogger("tutor.voice.mms_tts")

# Trained for Roman Urdu (Latin letters) — best match for tutor replies.
_DEFAULT_ROMAN_MODEL = "facebook/mms-tts-urd-script_latin"
_DEFAULT_ARABIC_MODEL = "hamza-amin/mms-tts-urd-fine-tuned"
_MAX_CHARS = 220

_model_id: str | None = None
_model = None
_tokenizer = None
_device = None


def mms_tts_available() -> bool:
    try:
        import torch  # noqa: F401
        from transformers import AutoTokenizer, VitsModel  # noqa: F401

        return True
    except ImportError:
        return False


def _resolved_model_id() -> str:
    explicit = optional_env("URDU_TTS_MODEL", "")
    if explicit:
        return explicit
    return _DEFAULT_ROMAN_MODEL


def _is_roman_latin_model(model_id: str) -> bool:
    return "script_latin" in model_id.lower()


def _use_device() -> str:
    import torch

    pref = optional_env("URDU_TTS_DEVICE", "cpu").lower()
    if pref == "cuda" and torch.cuda.is_available():
        return "cuda"
    if pref == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_model():
    global _model_id, _model, _tokenizer, _device
    model_id = _resolved_model_id()
    if _model is not None and _model_id == model_id:
        return _model, _tokenizer, _device

    import torch
    from transformers import AutoTokenizer, VitsModel

    _device = _use_device()
    logger.info("Loading MMS Urdu TTS model=%s device=%s (first request may take 1-2 min)", model_id, _device)
    _tokenizer = AutoTokenizer.from_pretrained(model_id)
    _model = VitsModel.from_pretrained(model_id).to(_device)
    _model.eval()
    _model_id = model_id
    logger.info("MMS Urdu TTS model loaded: %s", model_id)
    return _model, _tokenizer, _device


def preload_mms_model() -> None:
    """Optional startup warmup — downloads/loads weights before first Listen."""
    _load_model()


def _prepare_text(text: str, model_id: str) -> str:
    """Roman Latin model: keep tutor text as-is. Arabic model: optional script convert."""
    if _is_roman_latin_model(model_id):
        return text

    if optional_env("URDU_TTS_ROMAN_TO_SCRIPT", "true").lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return text
    if URDU_ARABIC_RE.search(text):
        return text
    if _latin_ratio(text) >= 0.35:
        converted = to_urdu_script(text)
        if converted and URDU_ARABIC_RE.search(converted):
            logger.info("Roman Urdu -> Urdu script for MMS Arabic model (%s chars)", len(converted))
            return converted
    return text


def _split_sentences(text: str, max_len: int = _MAX_CHARS) -> list[str]:
    if len(text) <= max_len:
        return [text]
    parts = re.split(r"(?<=[۔.!?])\s+", text)
    chunks: list[str] = []
    buf = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        candidate = f"{buf} {part}".strip() if buf else part
        if len(candidate) <= max_len:
            buf = candidate
        else:
            if buf:
                chunks.append(buf)
            buf = part if len(part) <= max_len else part[:max_len]
    if buf:
        chunks.append(buf)
    return chunks or [text[:max_len]]


def _waveform_to_wav(waveform, sample_rate: int) -> bytes:
    import numpy as np

    audio = waveform.squeeze().detach().cpu().numpy()
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767).astype(np.int16)
    buf = io.BytesIO()
    scipy.io.wavfile.write(buf, sample_rate, pcm)
    return buf.getvalue()


class MMSTTSService:
    def synthesize(self, text: str, *, speed: float = 1.0) -> bytes:
        del speed  # MMS VITS has no speed control

        cleaned = clean_for_speech(text)
        if not cleaned:
            raise ValueError("No speakable text after cleaning")

        model_id = _resolved_model_id()
        speech_text = _prepare_text(cleaned, model_id)
        if not speech_text.strip():
            raise ValueError("No text for MMS TTS after preparation")

        import torch

        model, tokenizer, device = _load_model()
        seed = int(optional_env("URDU_TTS_SEED", "42"))
        torch.manual_seed(seed)

        parts: list[bytes] = []
        for segment in _split_sentences(speech_text):
            logger.info(
                "MMS Roman Urdu TTS model=%s chars=%s preview=%r",
                model_id,
                len(segment),
                segment[:80],
            )
            inputs = tokenizer(segment, return_tensors="pt").to(device)
            with torch.no_grad():
                output = model(**inputs).waveform
            parts.append(_waveform_to_wav(output, model.config.sampling_rate))

        return merge_wav_bytes(parts, pause_ms=160)
