"""Roman Urdu -> Urdu script via Mavkif M2M100 (Hugging Face)."""

import logging
import re
import threading
from pathlib import Path

from src.config import optional_env

logger = logging.getLogger("tutor.voice.mavkif")

_MODEL_ID = "Mavkif/m2m100_rup_rur_to_ur"
_TOKENIZER_ID = "Mavkif/m2m100_rup_tokenizer_both"
_MODEL_FILE = "model.safetensors"
# ~1.8 GB for M2M100 0.5B FP32 weights
_MODEL_BYTES_EXPECTED = 1_932_735_288
_MAX_CHARS = 400

_model = None
_tokenizer = None
_device: str | None = None
_download_lock = threading.Lock()

_LATIN_RE = re.compile(r"[A-Za-z]")


def _hub_cache_dir(repo_id: str) -> Path:
    return Path.home() / ".cache" / "huggingface" / "hub" / f"models--{repo_id.replace('/', '--')}"


def _incomplete_bytes(repo_id: str) -> int:
    total = 0
    blobs = _hub_cache_dir(repo_id) / "blobs"
    if blobs.is_dir():
        for path in blobs.glob("*.incomplete"):
            total += path.stat().st_size
    return total


def download_status() -> dict:
    """Report resume progress for the model weights."""
    cached = _is_cached(_MODEL_ID, _MODEL_FILE)
    partial = _incomplete_bytes(_MODEL_ID)
    if cached:
        pct = 100.0
    elif partial > 0:
        pct = min(99.9, 100.0 * partial / _MODEL_BYTES_EXPECTED)
    else:
        pct = 0.0
    return {
        "cached": cached,
        "partial_bytes": partial,
        "partial_mb": round(partial / (1024 * 1024), 1),
        "expected_mb": round(_MODEL_BYTES_EXPECTED / (1024 * 1024), 1),
        "percent": round(pct, 1),
    }


def _log_download_status(phase: str) -> None:
    status = download_status()
    if status["cached"]:
        logger.info("Mavkif %s: already fully cached", phase)
        return
    if status["partial_bytes"] > 0:
        logger.info(
            "Mavkif %s: resuming download %.1f / %.1f MB (%.1f%%)",
            phase,
            status["partial_mb"],
            status["expected_mb"],
            status["percent"],
        )
    else:
        logger.info("Mavkif %s: starting fresh download (~%.0f MB)", phase, status["expected_mb"])


def _is_cached(repo_id: str, filename: str = _MODEL_FILE) -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache

        return try_to_load_from_cache(repo_id, filename) is not None
    except Exception:
        return False


def is_mavkif_cached() -> bool:
    return _is_cached(_MODEL_ID) and _is_cached(_TOKENIZER_ID, "tokenizer_config.json")


def mavkif_available() -> bool:
    try:
        import torch  # noqa: F401
        from transformers import M2M100ForConditionalGeneration, AutoTokenizer  # noqa: F401

        return True
    except ImportError:
        return False


def resume_mavkif_download() -> None:
    """Download or resume partial Hugging Face cache — never deletes .incomplete files."""
    from huggingface_hub import hf_hub_download

    with _download_lock:
        if is_mavkif_cached():
            logger.info("Mavkif already cached — skip download")
            return

        _log_download_status("tokenizer")
        hf_hub_download(_TOKENIZER_ID, "tokenizer_config.json")
        hf_hub_download(_TOKENIZER_ID, "sentencepiece.bpe.model")
        hf_hub_download(_TOKENIZER_ID, "added_tokens.json")

        _log_download_status("model weights")
        # huggingface_hub automatically resumes any existing .incomplete blob
        path = hf_hub_download(_MODEL_ID, _MODEL_FILE)
        logger.info("Mavkif download complete: %s", path)


def _pick_device() -> str:
    import torch

    pref = optional_env("MAVKIF_DEVICE", optional_env("URDU_TTS_DEVICE", "cpu")).lower()
    if pref == "cuda" and torch.cuda.is_available():
        return "cuda"
    if pref == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load():
    global _model, _tokenizer, _device
    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer, _device

    with _download_lock:
        if _model is not None and _tokenizer is not None:
            return _model, _tokenizer, _device

        if not is_mavkif_cached():
            resume_mavkif_download()

        import torch
        from transformers import AutoTokenizer, M2M100ForConditionalGeneration

        _device = _pick_device()
        logger.info("Loading Mavkif transliteration model=%s device=%s", _MODEL_ID, _device)
        _tokenizer = AutoTokenizer.from_pretrained(_TOKENIZER_ID)
        _model = M2M100ForConditionalGeneration.from_pretrained(_MODEL_ID).to(_device)
        _model.eval()
        logger.info("Mavkif transliteration model loaded")
        return _model, _tokenizer, _device


def preload_mavkif_model() -> None:
    _load()


def preload_mavkif_background() -> None:
    if is_mavkif_cached():
        logger.info("Mavkif model already cached — background download skipped")
        return

    def _run() -> None:
        try:
            resume_mavkif_download()
            logger.info("Mavkif background download finished — next Listen will use it")
        except Exception as exc:
            logger.warning("Mavkif background download failed: %s", exc)

    threading.Thread(target=_run, name="mavkif-download", daemon=True).start()


def transliterate_roman_urdu(text: str) -> str:
    segment = text.strip()
    if not segment:
        return ""

    import torch

    model, tokenizer, device = _load()
    tokenizer.src_lang = "roman-ur"
    inputs = tokenizer(segment, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.get_lang_id("ur"),
            max_new_tokens=160,
        )
    return tokenizer.batch_decode(tokens, skip_special_tokens=True)[0].strip()


def _split_for_model(text: str) -> list[str]:
    if len(text) <= _MAX_CHARS:
        return [text]
    parts = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    buf = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        candidate = f"{buf} {part}".strip() if buf else part
        if len(candidate) <= _MAX_CHARS:
            buf = candidate
        else:
            if buf:
                chunks.append(buf)
            buf = part
    if buf:
        chunks.append(buf)
    return chunks or [text[:_MAX_CHARS]]


def transliterate_roman_urdu_long(text: str) -> str:
    pieces = _split_for_model(text)
    out = [transliterate_roman_urdu(p) for p in pieces if p.strip()]
    return " ".join(out).strip()
