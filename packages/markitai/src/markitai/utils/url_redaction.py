"""Helpers for displaying URLs without leaking embedded credentials."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from markitai.fetch_policy import sensitive_path_segment_indexes

_HTTP_URL_IN_TEXT = re.compile(r"https?://[^\s<>'\"]+")
# `serve` access tokens ride in `?token=` on SSE, file and archive URLs (they
# cannot send headers); Uvicorn's access log prints the bare request target.
_TOKEN_QUERY_PARAM = re.compile(r"([?&]token=)[^&#\s<>'\"]+", re.IGNORECASE)
_REDACTED_PATH_SEGMENT = "[REDACTED]"


def redact_url(url: str) -> str:
    """Return a display-safe URL with credential surfaces removed."""
    try:
        parsed = urlsplit(url)
        hostname = parsed.hostname
        if not parsed.scheme or not hostname:
            return "[URL redacted]"

        host = f"[{hostname}]" if ":" in hostname else hostname
        try:
            port = parsed.port
        except ValueError:
            port = None
        netloc = f"{host}:{port}" if port is not None else host
        path_segments = parsed.path.split("/")
        for index in sensitive_path_segment_indexes(parsed.path):
            path_segments[index] = _REDACTED_PATH_SEGMENT
        safe_path = "/".join(path_segments)
        return urlunsplit((parsed.scheme, netloc, safe_path, "", ""))
    except (TypeError, ValueError):
        return "[URL redacted]"


def redact_urls_in_text(message: str) -> str:
    """Remove credential surfaces from every HTTP(S) URL in text.

    Also masks ``token=`` query values in scheme-less request targets.
    """
    message = _HTTP_URL_IN_TEXT.sub(lambda match: redact_url(match.group(0)), message)
    return _TOKEN_QUERY_PARAM.sub(rf"\1{_REDACTED_PATH_SEGMENT}", message)


def redact_log_record(record: Any) -> None:
    """Loguru patcher: remove URL credential surfaces before any sink."""
    message = record.get("message")
    if isinstance(message, str):
        record["message"] = redact_urls_in_text(message)
