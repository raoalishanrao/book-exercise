"""Verify .env and external service connectivity before ingest."""

import re
import sys
from pathlib import Path
from uuid import UUID

# Project root on path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import require_env  # noqa: E402

UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.I,
)

REQUIRED = [
    "GEMINI_API_KEY",
    "SUPABASE_URL",
    "SUPABASE_SERVICE_ROLE_KEY",
    "BOOK_ID",
    "CLASS_ID",
    "SUBJECT_ID",
    "PDF_PATH",
]


def check_present() -> list[str]:
    errors: list[str] = []
    for name in REQUIRED:
        try:
            require_env(name)
            print(f"  OK  {name} is set")
        except RuntimeError as e:
            errors.append(str(e))
            print(f"  FAIL {name} is missing")
    return errors


def check_uuids() -> list[str]:
    errors: list[str] = []
    for name in ("BOOK_ID", "CLASS_ID", "SUBJECT_ID"):
        value = require_env(name)
        if not UUID_RE.match(value):
            errors.append(f"{name} is not a valid UUID: {value!r}")
            print(f"  FAIL {name} invalid UUID format")
        else:
            print(f"  OK  {name} valid UUID")
    return errors


def check_pdf() -> list[str]:
    pdf = ROOT / require_env("PDF_PATH")
    if not pdf.is_file():
        print(f"  FAIL PDF not found: {pdf}")
        return [f"PDF not found: {pdf}"]
    size_mb = pdf.stat().st_size / (1024 * 1024)
    print(f"  OK  PDF found ({size_mb:.1f} MB): {pdf.name}")
    return []


def check_supabase() -> list[str]:
    from supabase import create_client

    url = require_env("SUPABASE_URL")
    if not url.startswith("https://") or "supabase.co" not in url:
        print("  FAIL SUPABASE_URL format looks wrong")
        return ["SUPABASE_URL format invalid"]

    client = create_client(url, require_env("SUPABASE_SERVICE_ROLE_KEY"))

    try:
        books = client.table("books").select("id,title").limit(1).execute()
        print(f"  OK  Supabase connected (books table reachable)")
    except Exception as e:
        print(f"  FAIL Supabase connection: {e}")
        return [f"Supabase error: {e}"]

    book_id = require_env("BOOK_ID")
    row = (
        client.table("books")
        .select("id,title,class_id,subject_id")
        .eq("id", book_id)
        .maybe_single()
        .execute()
    )
    if not row.data:
        print(f"  FAIL BOOK_ID not found in books table")
        return [f"BOOK_ID {book_id} not found in Supabase"]

    book = row.data
    class_id = require_env("CLASS_ID")
    subject_id = require_env("SUBJECT_ID")
    if str(book["class_id"]) != class_id:
        print("  FAIL CLASS_ID does not match book record")
        return ["CLASS_ID mismatch with book"]
    if str(book["subject_id"]) != subject_id:
        print("  FAIL SUBJECT_ID does not match book record")
        return ["SUBJECT_ID mismatch with book"]

    print(f"  OK  Book record: {book['title']}")
    return []


def check_gemini() -> list[str]:
    from google import genai

    client = genai.Client(api_key=require_env("GEMINI_API_KEY"))
    try:
        resp = client.models.embed_content(
            model="gemini-embedding-001",
            contents="connectivity test",
            config={"task_type": "RETRIEVAL_QUERY", "output_dimensionality": 768},
        )
        dim = len(resp.embeddings[0].values)
        if dim != 768:
            print(f"  FAIL Gemini embedding dimension {dim}, expected 768")
            return [f"Unexpected embedding dim: {dim}"]
        print(f"  OK  Gemini API connected (embedding dim={dim})")
        return []
    except Exception as e:
        print(f"  FAIL Gemini API: {e}")
        return [f"Gemini error: {e}"]


def main() -> int:
    print("=== Environment verification ===\n")
    all_errors: list[str] = []

    print("[1/5] Required variables")
    all_errors.extend(check_present())
    if all_errors:
        return 1

    print("\n[2/5] UUID format")
    all_errors.extend(check_uuids())

    print("\n[3/5] PDF file")
    all_errors.extend(check_pdf())

    print("\n[4/5] Supabase")
    all_errors.extend(check_supabase())

    print("\n[5/5] Gemini API")
    all_errors.extend(check_gemini())

    print()
    if all_errors:
        print("VERIFICATION FAILED:")
        for err in all_errors:
            print(f"  - {err}")
        return 1

    print("All checks passed. Ready for ingest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
