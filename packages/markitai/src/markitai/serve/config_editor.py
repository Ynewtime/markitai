"""The configuration file as text, for the workbench's Edit config page.

Contract v1.13 §8. The page receives the file's own text (its layout and key
order) with each secret string literal replaced by its masked form; what it
sends back is saved exactly as written, except that each masked value kept at
the JSON pointer it was sent at gets the stored literal back in place.

What is a secret follows ``markitai config show``'s redaction rule: any
non-empty string anywhere beneath a key whose words name a secret (api key,
token, authorization, cookie, credential, password, secret, ...), unless it
is an ``env:`` reference, and every value beneath an ``extra_http_headers``
object (Playwright sends those strings as they are, ``env:`` included). An
``api_base`` is shown as the settings' credentials route shows it (user
information and secret query values masked). The standard ``json`` module
validates and decodes; a small scanner locates each string literal.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any

from markitai.utils.text import (
    is_api_base_config_key,
    is_header_map_config_key,
    is_secret_config_key,
)
from markitai.utils.url_redaction import mask_url_credentials

MASK = "…"
_SPACE = " \t\n\r"


class Kind(Enum):
    """What a string at some position of the file is."""

    PLAIN = "plain"
    #: Beneath a key the redaction rule names; an ``env:`` reference is not.
    SECRET = "secret"  # nosec B105 - a kind of value, not a secret
    #: Beneath an ``extra_http_headers`` object: every value.
    HEADER = "header"
    #: An endpoint address (``api_base``, ``*_api_base``).
    ADDRESS = "address"


@dataclass(frozen=True)
class Literal:
    """A string value of a JSON text: its path and its literal's span."""

    path: tuple[str | int, ...]
    start: int
    end: int  # quotes included

    @property
    def pointer(self) -> str:
        """Its JSON pointer (RFC 6901)."""
        return "".join(
            "/" + (step.replace("~", "~0").replace("/", "~1"))
            if isinstance(step, str)
            else f"/{step}"
            for step in self.path
        )

    @property
    def kind(self) -> Kind:
        keys = [step for step in self.path if isinstance(step, str)]
        if any(is_header_map_config_key(key) for key in keys):
            return Kind.HEADER
        if any(is_secret_config_key(key) for key in keys):
            return Kind.SECRET
        if (
            self.path
            and isinstance(self.path[-1], str)
            and is_api_base_config_key(self.path[-1])
        ):
            return Kind.ADDRESS
        return Kind.PLAIN


class _Scanner:
    """A position-aware walk over a JSON text ``json`` has accepted."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.at = 0
        self.path: list[str | int] = []
        self.found: list[Literal] = []

    def _space(self) -> None:
        while self.at < len(self.text) and self.text[self.at] in _SPACE:
            self.at += 1

    def _peek(self) -> str:
        if self.at >= len(self.text):
            raise ValueError("unexpected end")
        return self.text[self.at]

    def _string(self) -> tuple[str, int]:
        """The string literal at the cursor, decoded, and where it starts."""
        if self._peek() != '"':
            raise ValueError("string expected")
        start = self.at
        self.at += 1
        while (char := self._peek()) != '"':
            self.at += 2 if char == "\\" else 1
        self.at += 1
        return json.loads(self.text[start : self.at]), start

    def _expect(self, char: str) -> None:
        self._space()
        if self._peek() != char:
            raise ValueError(f"{char} expected")
        self.at += 1

    def value(self) -> None:
        self._space()
        char = self._peek()
        if char in "{[":
            closing = "}" if char == "{" else "]"
            self.at += 1
            self._space()
            if self._peek() == closing:
                self.at += 1
                return
            index = 0
            while True:
                if char == "{":
                    self._space()
                    key, _ = self._string()
                    self._expect(":")
                    self.path.append(key)
                else:
                    self.path.append(index)
                    index += 1
                self.value()
                self.path.pop()
                self._space()
                following = self._peek()
                self.at += 1
                if following == closing:
                    return
                if following != ",":
                    raise ValueError(", expected")
        elif char == '"':
            _, start = self._string()
            self.found.append(Literal(tuple(self.path), start, self.at))
        else:
            # A number, true, false or null
            start = self.at
            while self.at < len(self.text) and self.text[self.at] not in ",}]" + _SPACE:
                self.at += 1
            if self.at == start:
                raise ValueError("value expected")


def literals(text: str) -> list[Literal] | None:
    """Every string value of *text* (not object keys) in text order.

    None for a text that is not one JSON value.
    """
    scanner = _Scanner(text)
    try:
        scanner.value()
        scanner._space()
    except ValueError:
        return None
    return scanner.found if scanner.at == len(text) else None


def _refuse_constant(name: str) -> Any:
    raise ValueError(f"{name} is not JSON")


def parse(text: str) -> Any:
    """Parse *text* as strict JSON (no NaN or Infinity).

    ``json.JSONDecodeError`` (a ``ValueError``) when it is not JSON.
    """
    try:
        return json.loads(text, parse_constant=_refuse_constant)
    except json.JSONDecodeError:
        raise
    except ValueError as e:
        # A NaN/Infinity constant: report it where the text has it
        name = str(e).split(" ", 1)[0]
        raise json.JSONDecodeError(
            f"{name} is not allowed in JSON", text, max(text.find(name), 0)
        ) from None


def position(error: json.JSONDecodeError) -> str:
    """A JSON parse error as ``line L, column C: message``."""
    return f"line {error.lineno}, column {error.colno}: {error.msg}"


def revision(data: bytes | None) -> str:
    """The file's revision: ``sha256:`` and the hex SHA-256 of its bytes."""
    if data is None:
        return "absent"
    return "sha256:" + hashlib.sha256(data).hexdigest()


def masked_secret(value: str) -> str:
    """A secret as the credentials route shows a key: ``sk…7890``."""
    if len(value) >= 12:
        return f"{value[:2]}{MASK}{value[-4:]}"
    if len(value) >= 4:
        return f"{MASK}{value[-2:]}"
    return MASK


def _masked(value: str, kind: Kind) -> str | None:
    """The masked form of *value*, when it is masked."""
    if kind is Kind.PLAIN or not value:
        return None
    if kind is Kind.SECRET:
        return None if value.startswith("env:") else masked_secret(value)
    if kind is Kind.HEADER:
        return masked_secret(value)
    shown = mask_url_credentials(value)
    return shown if shown != value else None


def _decoded(text: str, literal: Literal) -> str:
    return json.loads(text[literal.start : literal.end])


def _secrets(text: str) -> dict[str, tuple[Literal, str, str]]:
    """Each masked literal of *text*: (literal, stored text, shown value)."""
    found: dict[str, tuple[Literal, str, str]] = {}
    for literal in literals(text) or []:
        shown = _masked(_decoded(text, literal), literal.kind)
        if shown is not None:
            found[literal.pointer] = (
                literal,
                text[literal.start : literal.end],
                shown,
            )
    return found


def _replaced(text: str, replacements: list[tuple[Literal, str]]) -> str:
    """*text* with each literal's span replaced (spans are disjoint)."""
    parts: list[str] = []
    at = 0
    for literal, replacement in sorted(replacements, key=lambda r: r[0].start):
        parts += [text[at : literal.start], replacement]
        at = literal.end
    parts.append(text[at:])
    return "".join(parts)


def mask(text: str) -> tuple[str, int]:
    """The file's text with each secret literal masked, and how many."""
    replacements = [
        (literal, json.dumps(shown, ensure_ascii=False))
        for literal, _, shown in _secrets(text).values()
    ]
    return _replaced(text, replacements), len(replacements)


def view(
    path: str, origin: str, data: bytes | None, *, read_only: bool
) -> dict[str, Any]:
    """The Edit config view of the file *data* (None: there is no file)."""
    text: str | None
    count = 0
    problem: str | None = None
    if data is None:
        text = "{}\n"
    else:
        text = None
        try:
            decoded = data.decode("utf-8")
        except UnicodeDecodeError:
            problem = "the configuration is not UTF-8 text"
        else:
            try:
                value = parse(decoded)
            except json.JSONDecodeError as e:
                problem = position(e)
            else:
                if isinstance(value, dict):
                    text, count = mask(decoded)
                else:
                    problem = "the configuration must be a JSON object"
    return {
        "path": path,
        "origin": origin,
        "exists": data is not None,
        "revision": revision(data),
        "text": text,
        "masked": count,
        "problem": problem,
        "read_only": read_only,
    }


def current_object_text(data: bytes | None) -> str | None:
    """The stored file's text when it is a JSON object, else None."""
    if data is None:
        return None
    try:
        text = data.decode("utf-8")
        return text if isinstance(parse(text), dict) else None
    except (UnicodeDecodeError, ValueError):
        return None


def unmask(text: str, current: str | None) -> str:
    """The client's *text* with each kept mask put back as the stored literal.

    *text* has been accepted as JSON; *current* is the stored file's text (a
    JSON object) or None. A masked value kept at the pointer the GET sent it
    at gets the stored literal back (only that literal's characters change);
    a secret-position string still holding ``…`` anywhere else is refused
    with a ``ValueError`` naming its pointer.
    """
    stored = _secrets(current) if current is not None else {}
    found = literals(text)
    if found is None:
        raise ValueError("the configuration text could not be read")
    replacements: list[tuple[Literal, str]] = []
    for literal in found:
        value = _decoded(text, literal)
        kept = stored.get(literal.pointer)
        if kept is not None and kept[2] == value:
            replacements.append((literal, kept[1]))
            continue
        if literal.kind is not Kind.PLAIN and MASK in value:
            raise ValueError(
                f"{literal.pointer}: a masked value can only be kept where it "
                "was; type the full value or an env: reference"
            )
    return _replaced(text, replacements)


def restart_required(old: dict[str, Any] | None, new: dict[str, Any]) -> bool:
    """Whether anything but ``llm.model_list`` and ``llm.providers`` differs."""

    def rest(value: dict[str, Any]) -> dict[str, Any]:
        value = dict(value)
        llm = value.get("llm")
        if isinstance(llm, dict):
            llm = {
                key: item
                for key, item in llm.items()
                if key not in ("model_list", "providers")
            }
            if llm:
                value["llm"] = llm
            else:
                value.pop("llm")
        return value

    return old is None or rest(old) != rest(new)
