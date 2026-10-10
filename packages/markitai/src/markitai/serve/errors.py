"""Error bodies of the serve API.

Every non-2xx ``/api/`` answer is ``{"detail", "code", "reason"}``:

* ``detail``: the server's English sentence, or for a settings conflict an
  object carrying its own ``code`` (``stale_revision``, ...);
* ``code``: derived from the HTTP status, coarse and stable;
* ``reason``: the stable machine cause the shared web workbench translates
  (the reason list of the workbench API contract, section 3).

Errors raised as :class:`ApiError` carry their reason; any other
``HTTPException`` gets one derived from its status (or its ``detail.code``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from fastapi import HTTPException
from fastapi.responses import JSONResponse

_CODES = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    409: "conflict",
    413: "payload_too_large",
    422: "invalid_request",
    429: "rate_limited",
    503: "unavailable",
}

# Reasons for errors raised without one: a 404 nobody raised on purpose is
# the router's (no such route), a 401 is the token gate's.
_FALLBACK_REASONS = {
    400: "bad_request",
    401: "token_required",
    403: "forbidden",
    404: "route_not_found",
    405: "method_not_allowed",
    409: "conflict",
    413: "request_too_large",
    422: "invalid_request",
    429: "rate_limited",
    500: "internal_error",
    503: "unavailable",
}


def http_error_code(status_code: int) -> str:
    """Map an HTTP status onto a stable, coarse machine code.

    The ``detail`` string is for humans; ``code`` is the part a client or
    agent branches on when it does not know the ``reason``.
    """
    if status_code in _CODES:
        return _CODES[status_code]
    return "server_error" if status_code >= 500 else "request_failed"


def fallback_reason(status_code: int, detail: Any = None) -> str:
    """Reason for an error raised without one."""
    if isinstance(detail, dict):
        code = detail.get("code")
        if isinstance(code, str) and code:
            return code
    return _FALLBACK_REASONS.get(
        status_code, "internal_error" if status_code >= 500 else "bad_request"
    )


class ApiError(HTTPException):
    """An ``HTTPException`` that names its contract ``reason``."""

    def __init__(
        self,
        status_code: int,
        reason: str,
        detail: Any,
        *,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(status_code=status_code, detail=detail, headers=headers)
        self.reason = reason


def error_body(status_code: int, reason: str, detail: Any) -> dict[str, Any]:
    """The JSON body of one error answer."""
    return {"detail": detail, "code": http_error_code(status_code), "reason": reason}


def error_response(
    status_code: int,
    reason: str,
    detail: Any,
    *,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    """A complete error answer; a 401 always names the Bearer scheme."""
    merged = dict(headers or {})
    if status_code == 401:
        merged.setdefault("WWW-Authenticate", "Bearer")
    return JSONResponse(
        error_body(status_code, reason, detail),
        status_code=status_code,
        headers=merged or None,
    )


def validation_detail(errors: Sequence[Mapping[str, Any]]) -> str:
    """One readable sentence for pydantic validation errors.

    Built from each error's location and message only: the ``input`` pydantic
    records may be a submitted secret (an API key) and never reaches a reply.
    """
    parts: list[str] = []
    for error in errors[:5]:
        location = [
            str(part)
            for part in error.get("loc", ())
            if part not in ("body", "query", "path")
        ]
        message = str(error.get("msg") or "invalid value")
        parts.append(f"{'.'.join(location)}: {message}" if location else message)
    if len(errors) > 5:
        parts.append(f"and {len(errors) - 5} more")
    return "; ".join(parts) or "invalid request"
