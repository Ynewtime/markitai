"""Helpers for displaying URLs without leaking embedded credentials."""

from __future__ import annotations

import re
from urllib.parse import urlsplit, urlunsplit

_HTTP_URL_IN_TEXT = re.compile(r"https?://[^\s<>'\"]+")


def redact_url(url: str) -> str:
    """Return an origin-and-path URL without userinfo, query, or fragment."""
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
        return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))
    except (TypeError, ValueError):
        return "[URL redacted]"


def redact_urls_in_text(message: str) -> str:
    """Remove credential surfaces from every HTTP(S) URL in text."""
    return _HTTP_URL_IN_TEXT.sub(lambda match: redact_url(match.group(0)), message)
