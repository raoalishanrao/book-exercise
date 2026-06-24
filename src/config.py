import os
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_FILE = _PROJECT_ROOT / ".env"

# override=True ensures .env wins over empty shell vars
load_dotenv(_ENV_FILE, override=True, encoding="utf-8")


def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(
            f"Missing required environment variable: {name} "
            f"(check {_ENV_FILE})"
        )
    return value


def optional_env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def env_list(name: str) -> list[str]:
    raw = optional_env(name)
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]
