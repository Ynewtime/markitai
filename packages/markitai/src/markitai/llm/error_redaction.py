"""Clean a provider's error text before markitai shows, logs or stores it.

A failed request's exception carries the provider's own wording, which can
echo what the request sent: the API key (an HTTP 401 saying ``Incorrect API
key provided: sk-...``) or the document (an HTTP 400 quoting the prompt).
That wording reaches stderr, log files, ``--json`` and report output, batch
state, ``serve`` jobs and MCP tool errors, so it is cleaned once, in the
exception itself, where the request fails. What stays: the error type, the
HTTP status (``status_code``) and the rest of the provider's message.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Sequence
from contextlib import suppress
from typing import Any

from markitai.utils.secret_redaction import redact_secrets
from markitai.utils.url_redaction import redact_urls_in_text

# A quotation longer than this is replaced: it may be request content.
_QUOTE_CHARS = 40
# Characters of provider wording kept in one message.
_MESSAGE_CHARS = 500
# Letters and digits in a stretch that, also found in the request, marks the
# message as an echo of it. Windows overlap, so any shared stretch of
# ``_ECHO_WINDOW + _ECHO_STEP - 1`` is found.
_ECHO_WINDOW = 16
_ECHO_STEP = 4

WITHHELD = "[provider message withheld: it repeats the request]"

_QUOTES = {"'": "'", '"': '"', "`": "`", "“": "”", "‘": "’"}
_QUOTE_OPEN = re.compile("[" + "".join(_QUOTES) + "]")
# LiteLLM's prefix: "litellm.BadRequestError: OpenAIException - "
_LITELLM_HEAD = re.compile(r"^(?:[\w.]+(?:Error|Exception)(?::| -) )+")
_ESCAPE = re.compile(r"\\(?:[nrt]|u([0-9a-fA-F]{4}))")
_NOT_ALNUM = re.compile(r"[\W_]+")
_IDENTIFIER = re.compile(r"[\w.\-/]{1,60}")
_JSON = json.JSONDecoder()


def request_texts(messages: Any) -> list[str]:
    """The text parts of chat *messages* (images are skipped)."""
    texts: list[str] = []
    if not isinstance(messages, list):
        return texts
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    texts.append(part["text"])
    return texts


def _json_error(value: Any) -> str | None:
    """``type: message`` from a provider's JSON error body, if it has one."""
    if isinstance(value, list) and value:  # Gemini wraps its error in a list
        value = value[0]
    if not isinstance(value, dict):
        return None
    inner = value.get("error")
    if isinstance(inner, str):
        return inner
    if isinstance(inner, dict):
        value = inner
    message = value.get("message") or value.get("detail")
    if not isinstance(message, str):
        return None
    kind = value.get("type") or value.get("code") or value.get("status")
    if isinstance(kind, str) and _IDENTIFIER.fullmatch(kind):
        return f"{kind}: {message}"
    return message


def _unwrap_json_body(text: str) -> str:
    """Replace a raw JSON error body LiteLLM embedded with its message.

    ``DeepseekException - {"error": {"message": "..."}}`` would otherwise
    lose the message to the quotation rule: every JSON string is quoted.
    """
    start = text.find("{")
    for _ in range(8):
        if start < 0:
            break
        try:
            value, end = _JSON.raw_decode(text, start)
        except ValueError:
            start = text.find("{", start + 1)
            continue
        said = _json_error(value)
        if said is not None:
            return text[:start] + said + text[end:]
        start = text.find("{", end)
    return text


def _without_long_quotations(text: str) -> str:
    """Replace quotations longer than ``_QUOTE_CHARS`` with ``…``."""
    out: list[str] = []
    pos = 0
    while (match := _QUOTE_OPEN.search(text, pos)) is not None:
        open_at = match.start()
        quote = text[open_at]
        # An apostrophe inside a word (can't) opens no quotation.
        in_word = quote == "'" and open_at > 0 and text[open_at - 1].isalnum()
        close_at = -1 if in_word else text.find(_QUOTES[quote], open_at + 1)
        # An escaped quote (Python's repr: 'it\'s') does not close it.
        while close_at > 0 and text[close_at - 1] == "\\":
            close_at = text.find(_QUOTES[quote], close_at + 1)
        if close_at < 0:
            out.append(text[pos : open_at + 1])
            pos = open_at + 1
            continue
        inner = text[open_at + 1 : close_at]
        out.append(text[pos : open_at + 1])
        out.append("…" if len(inner) > _QUOTE_CHARS else inner)
        out.append(text[close_at])
        pos = close_at + 1
    out.append(text[pos:])
    return "".join(out)


def _fold(text: str) -> str:
    """Letters and digits only, lowercased, after undoing ``\\n``/``\\uXXXX``."""
    text = _ESCAPE.sub(lambda m: chr(int(m.group(1), 16)) if m.group(1) else " ", text)
    return _NOT_ALNUM.sub("", text).lower()


def _echoes(text: str, request: Sequence[str]) -> bool:
    """Whether *text* repeats a stretch of the request's text."""
    said = _fold(text)
    if len(said) < _ECHO_WINDOW or not request:
        return False
    folded = "\0".join(_fold(part) for part in request)
    starts = [*range(0, len(said) - _ECHO_WINDOW + 1, _ECHO_STEP)]
    starts.append(len(said) - _ECHO_WINDOW)
    return any(said[start : start + _ECHO_WINDOW] in folded for start in starts)


def redact_provider_message(text: str, request: Sequence[str] = ()) -> str:
    """Provider error wording as it may be shown.

    Credentials and URL secrets are masked, long quotations dropped, a
    message that repeats the *request* text withheld (LiteLLM's
    ``ErrorType: ProviderException -`` prefix is kept), and the rest capped.
    """
    if not text:
        return text
    # Only the start is ever shown; the cut bounds the quotation scan.
    text = redact_urls_in_text(redact_secrets(text))[: _MESSAGE_CHARS * 8]
    text = _without_long_quotations(_unwrap_json_body(text))
    if _echoes(text, request):
        head = _LITELLM_HEAD.match(text)
        return f"{head.group(0) if head else ''}{WITHHELD}"
    if len(text) > _MESSAGE_CHARS:
        text = text[:_MESSAGE_CHARS] + "…"
    return text


def _exception_chain(exc: BaseException) -> Iterator[BaseException]:
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def redact_provider_exception(exc: BaseException, messages: Any = None) -> None:
    """Clean a failed request's exception (and its causes) in place.

    In place, so the exception keeps its type and ``status_code`` for retry
    and auth handling, while every later ``str(exc)`` (logs, item errors,
    instructor's retry wrapper) shows the cleaned wording.
    """
    texts = request_texts(messages)
    for current in _exception_chain(exc):
        message = getattr(current, "message", None)
        if isinstance(message, str):
            with suppress(AttributeError, TypeError):
                current.message = redact_provider_message(message, texts)  # type: ignore[attr-defined]
        if current.args and isinstance(current.args[0], str):
            current.args = (
                redact_provider_message(current.args[0], texts),
                *current.args[1:],
            )
