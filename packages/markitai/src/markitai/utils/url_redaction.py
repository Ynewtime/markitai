"""Helpers for displaying URLs without leaking embedded credentials."""

from __future__ import annotations

import re
from contextlib import suppress
from typing import Any
from urllib.parse import unquote_plus, urlsplit, urlunsplit

from markitai.fetch_policy import (
    is_sensitive_query_key,
    sensitive_path_segment_indexes,
)
from markitai.utils.secret_redaction import redact_secrets

_HTTP_URL_IN_TEXT = re.compile(r"https?://[^\s<>'\"]+")
# `serve` access tokens ride in `?token=` on SSE, file and archive URLs (they
# cannot send headers), and single-use download tickets in `?ticket=`;
# Uvicorn's access log prints the bare request target.
_TOKEN_QUERY_PARAM = re.compile(r"([?&](?:token|ticket)=)[^&#\s<>'\"]+", re.IGNORECASE)
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

    Also masks ``token=`` and ``ticket=`` query values in scheme-less
    request targets.
    """
    message = _HTTP_URL_IN_TEXT.sub(lambda match: redact_url(match.group(0)), message)
    return _TOKEN_QUERY_PARAM.sub(rf"\1{_REDACTED_PATH_SEGMENT}", message)


_MASK = "\u2026"
# markitai-rust's rule for a secret parameter (any name containing one of
# these), on top of the fetch policy's: Azure's `subscription-key`, say
_SECRET_NAME_PARTS = ("token", "key", "secret", "password", "signature", "credential")


def _is_secret_parameter(name: str) -> bool:
    lowered = name.lower()
    return is_sensitive_query_key(name) or any(
        part in lowered for part in _SECRET_NAME_PARTS
    )


def _mask_parameters(text: str) -> str:
    pairs = []
    for pair in text.split("&"):
        name = pair.partition("=")[0]
        secret = _is_secret_parameter(unquote_plus(name))
        pairs.append(f"{name}={_MASK}" if secret else pair)
    return "&".join(pairs)


def mask_url_credentials(source: str) -> str:
    """A configured address as a settings screen may show it.

    The text stays as written, with the userinfo and the value of every secret
    query or fragment parameter replaced by ``…``, so
    ``https://user:pw@host/v1?token=x&v=2`` reads ``https://…@host/v1?token=…&v=2``.
    An ``env:NAME`` reference names a variable and is returned as it is. The
    same text as markitai-rust's ``masked_url``: both serve the same UI.
    """
    if source.lstrip().startswith("env:"):
        return source
    rest, has_fragment, fragment = source.partition("#")
    rest, has_query, query = rest.partition("?")
    scheme_end = rest.find("://")
    scheme, after = (
        (rest[: scheme_end + 3], rest[scheme_end + 3 :])
        if scheme_end >= 0
        else ("", rest)
    )
    authority, slash, path = after.partition("/")
    at = authority.rfind("@")
    if at >= 0:
        authority = _MASK + authority[at:]
    masked = scheme + authority + slash + path
    if has_query:
        masked += "?" + _mask_parameters(query)
    if has_fragment:
        masked += "#" + _mask_parameters(fragment)
    return masked


def url_credential_values(source: str) -> list[str]:
    """The secret parts :func:`mask_url_credentials` hides, as plain text."""
    values: list[str] = []
    rest, _, fragment = source.partition("#")
    rest, _, query = rest.partition("?")
    try:
        parsed = urlsplit(rest)
        for part in (parsed.username, parsed.password):
            if part:
                values += {part, unquote_plus(part)}
    except ValueError:
        pass
    for text in (query, fragment):
        for pair in text.split("&"):
            name, _, value = pair.partition("=")
            if value and _is_secret_parameter(unquote_plus(name)):
                values += {value, unquote_plus(value)}
    return values


def _redact_text(text: str) -> str:
    return redact_secrets(redact_urls_in_text(text))


def _redact_exception_text(exc: BaseException | None) -> None:
    """Clean a logged exception's own text (and its causes') in place.

    A sink prints the exception under its traceback, out of the patcher's
    reach in ``record["message"]``; a crash may carry a key or a URL secret.
    """
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        args = tuple(_redact_text(a) if isinstance(a, str) else a for a in exc.args)
        if args != exc.args:
            exc.args = args
        message = getattr(exc, "message", None)
        if isinstance(message, str) and _redact_text(message) != message:
            with suppress(AttributeError, TypeError):
                exc.message = _redact_text(message)  # type: ignore[attr-defined]
        exc = exc.__cause__ or exc.__context__


def redact_log_record(record: Any) -> None:
    """Loguru patcher: remove credentials and URL secrets before any sink.

    Covers the message and the logged exception's text. Variable values in a
    traceback are kept out by the sinks themselves (``diagnose=False``).
    """
    message = record.get("message")
    if isinstance(message, str):
        record["message"] = _redact_text(message)
    exception = record.get("exception")
    if exception is not None:
        _redact_exception_text(exception.value)
