"""Mask credentials in text that may reach the terminal, logs or reports.

Provider errors can echo the key a request carried (an HTTP 401 saying
``Incorrect API key provided: sk-...``). Every credential markitai resolves
from its config is registered here, so its exact value is masked wherever it
shows up; credential-shaped tokens (``sk-...``, ``Bearer ...``, JWTs, ...)
are masked too, which covers keys LiteLLM read from the environment itself.
"""

from __future__ import annotations

import os
import re
import threading

REDACTED = "[REDACTED]"

# Shorter values are too likely to occur in ordinary text to replace blindly.
_MIN_SECRET_LENGTH = 8

# Environment variables whose values are credentials: provider keys that
# LiteLLM reads itself when the config names none (OPENAI_API_KEY, ...).
_SECRET_ENV_SUFFIXES = ("_API_KEY", "_TOKEN", "_SECRET", "_SECRET_KEY")

_registered: set[str] = set()
_lock = threading.Lock()

_KEY_PATTERNS = (
    # OpenAI / Anthropic / OpenRouter / DeepSeek (sk-, sk-proj-, sk-ant-, ...),
    # including a partly starred echo (sk-abc*****wxyz).
    re.compile(r"\b(?:sk|rk|pk)[-_][A-Za-z0-9_\-*]{8,}"),
    re.compile(r"\bAIza[0-9A-Za-z_\-]{20,}"),  # Google
    re.compile(r"\b(?:gh[pousr]_|github_pat_)[A-Za-z0-9_]{16,}"),  # GitHub
    re.compile(r"\b(?:xai-|gsk_|hf_|jina_)[A-Za-z0-9_\-]{16,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),  # AWS access key id
    re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}"),
)
_BEARER = re.compile(r"(?i)\b(bearer\s+)[A-Za-z0-9._~+/=\-]{8,}")
# The value may not be an env: reference or a "your_key"-style placeholder:
# a hint saying how to set a key ("export OPENAI_API_KEY=your_key") is not one
_NAMED_CREDENTIAL = re.compile(
    r"(?i)((?:x-)?api[_-]?key|authorization|access[_-]?token|api[_-]?token"
    r"|secret|password)(['\"]?\s*[:=]\s*['\"]?)(?!bearer\s|env:|your[_-])"
    r"([^\s'\",;}&]{6,})"
)


def register_secret(value: str | None) -> None:
    """Mask *value* from now on in everything :func:`redact_secrets` cleans."""
    if not value or len(value) < _MIN_SECRET_LENGTH or value.startswith("env:"):
        return
    with _lock:
        _registered.add(value)


def _known_secrets() -> list[str]:
    with _lock:
        secrets = set(_registered)
    for name, value in os.environ.items():
        if name.upper().endswith(_SECRET_ENV_SUFFIXES) and len(value) >= (
            _MIN_SECRET_LENGTH
        ):
            secrets.add(value)
    # Longest first, so a key that contains another is masked whole.
    return sorted(secrets, key=len, reverse=True)


def redact_secrets(text: str) -> str:
    """Replace configured credentials and credential-shaped tokens."""
    if not text:
        return text
    for secret in _known_secrets():
        if secret in text:
            text = text.replace(secret, REDACTED)
    for pattern in _KEY_PATTERNS:
        text = pattern.sub(REDACTED, text)
    text = _BEARER.sub(rf"\1{REDACTED}", text)
    return _NAMED_CREDENTIAL.sub(rf"\1\2{REDACTED}", text)
