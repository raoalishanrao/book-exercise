"""Shared helpers for Google GenAI API errors."""


def error_code(exc: Exception) -> int | None:
    code = getattr(exc, "code", None)
    if code is not None:
        return int(code)
    status = getattr(exc, "status_code", None)
    return int(status) if status is not None else None


def retry_delay_seconds(exc: Exception, attempt: int) -> float:
    try:
        response_json = getattr(exc, "response_json", None) or {}
        details = response_json.get("error", {}).get("details", [])
        for item in details:
            if item.get("@type", "").endswith("RetryInfo"):
                delay = item.get("retryDelay", "")
                if isinstance(delay, str) and delay.endswith("s"):
                    return float(delay[:-1]) + 1
    except (AttributeError, ValueError, TypeError):
        pass
    return min(60, 30 * (2**attempt))
