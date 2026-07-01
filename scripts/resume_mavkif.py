"""Resume Mavkif model download (safe — never deletes partial files)."""

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from src.voice.mavkif_transliterate import download_status, is_mavkif_cached, resume_mavkif_download

if __name__ == "__main__":
    st = download_status()
    print(f"Before: cached={st['cached']} partial={st['partial_mb']}MB ({st['percent']}%)")
    if is_mavkif_cached():
        print("Already complete.")
        sys.exit(0)
    resume_mavkif_download()
    st = download_status()
    print(f"After: cached={st['cached']} partial={st['partial_mb']}MB ({st['percent']}%)")
