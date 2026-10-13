"""User access tokens (workbench contract v1.13, section 2).

The server's token is the admin's. The admin creates user tokens in the
workbench, each with a label; a user token admits every API route except the
settings and the token routes, and its holder sees only the jobs it created.
Only each token's SHA-256 is kept, in ``~/.markitai/serve/access.json``
(owner-only, replaced atomically; never ``config.json``, which the Edit config
page shows and the CLI reads), so the tokens survive restarts and admin-token
regeneration. The token itself is returned once, when it is created.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
import threading
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from markitai.serve.errors import ApiError
from markitai.utils.clock import now_iso

#: User tokens at most.
MAX_TOKENS = 50
#: A label's characters at most.
MAX_LABEL = 60
#: Every user token starts with this; the rest is 64 lowercase hex digits.
USER_TOKEN_PREFIX = "mku_"  # nosec B105 - a prefix, not a secret

_HEX = re.compile(r"[0-9a-f]+")
_ID = re.compile(r"u_[0-9a-f]{12}")


def _hex_of(value: object, length: int) -> bool:
    return (
        isinstance(value, str) and len(value) == length and bool(_HEX.fullmatch(value))
    )


def digest(token: str) -> str:
    """SHA-256 of *token*, lowercase hexadecimal."""
    return hashlib.sha256(token.encode("utf-8", "surrogatepass")).hexdigest()


def valid_id(value: object) -> bool:
    """Whether *value* has the form of a user-token id (``u_`` + 12 hex)."""
    return isinstance(value, str) and bool(_ID.fullmatch(value))


def looks_like_user_token(value: str) -> bool:
    """Whether *value* has a user token's form, known or not."""
    return value.startswith(USER_TOKEN_PREFIX) and _hex_of(
        value[len(USER_TOKEN_PREFIX) :], 64
    )


def clean_label(text: object) -> str | None:
    """A trimmed label of 1 to 60 characters without control characters."""
    if not isinstance(text, str):
        return None
    text = text.strip()
    if not 1 <= len(text) <= MAX_LABEL:
        return None
    if any(unicodedata.category(char) == "Cc" for char in text):
        return None
    return text


@dataclass(frozen=True)
class _Entry:
    id: str
    label: str
    sha256: str
    hint: str
    created_at: str


def _valid(raw: object) -> _Entry | None:
    fields = ("id", "label", "sha256", "hint", "created_at")
    if not isinstance(raw, dict) or not all(name in raw for name in fields):
        return None
    entry = _Entry(*(raw[name] for name in fields))
    created = entry.created_at
    if not (
        valid_id(entry.id)
        and _hex_of(entry.sha256, 64)
        and clean_label(entry.label) == entry.label
        and _hex_of(entry.hint, 4)
        and isinstance(created, str)
    ):
        return None
    from datetime import datetime

    try:
        datetime.fromisoformat(created)
    except ValueError:
        return None
    return entry


def _read(path: Path) -> list[_Entry]:
    """The saved tokens; ``ValueError`` names what is wrong with the file."""
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return []
    except OSError as e:
        raise ValueError(f"{path}: {type(e).__name__}") from None
    try:
        saved = json.loads(data)
    except (ValueError, UnicodeDecodeError):
        raise ValueError(f"{path}: not JSON") from None
    if not isinstance(saved, dict) or saved.get("version") != 1:
        raise ValueError(f"{path}: unsupported version")
    raw_tokens = saved.get("tokens")
    if not isinstance(raw_tokens, list) or len(raw_tokens) > MAX_TOKENS:
        raise ValueError(f"{path}: invalid token list")
    entries = [_valid(raw) for raw in raw_tokens]
    if any(entry is None for entry in entries):
        raise ValueError(f"{path}: invalid token entry")
    valid = [entry for entry in entries if entry is not None]
    if len({entry.id for entry in valid}) != len(valid):
        raise ValueError(f"{path}: duplicate token id")
    return valid


def _io_failed() -> ApiError:
    return ApiError(500, "settings_io_failed", "Could not save the access tokens")


class AccessTokens:
    """The user tokens: digests on disk, last use in memory."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._write = threading.Lock()
        self._used: dict[str, str] = {}
        try:
            self._tokens = _read(path)
        except ValueError as e:
            # Left as it is until an admin creates or revokes a token
            logger.warning(
                "[Serve] Ignoring the saved access tokens ({}); no user token "
                "works until an admin creates or revokes one.",
                e,
            )
            self._tokens = []

    def authenticate(self, presented: str) -> tuple[str, str] | None:
        """The user token *presented* is, as ``(id, label)``.

        The value is hashed and compared with every stored digest in
        constant time.
        """
        presented_digest = digest(presented).encode()
        found: _Entry | None = None
        with self._lock:
            for entry in self._tokens:
                if (
                    hmac.compare_digest(presented_digest, entry.sha256.encode())
                    and found is None
                ):
                    found = entry
            if found is not None:
                self._used[found.id] = now_iso()
        return (found.id, found.label) if found is not None else None

    def exists(self, token_id: str) -> bool:
        """Whether the user token *token_id* still exists."""
        with self._lock:
            return any(entry.id == token_id for entry in self._tokens)

    def _public(self, entry: _Entry) -> dict[str, Any]:
        return {
            "id": entry.id,
            "label": entry.label,
            "hint": entry.hint,
            "created_at": entry.created_at,
            "last_used_at": self._used.get(entry.id),
        }

    def list(self) -> dict[str, Any]:
        """Every user token, oldest first, without the token itself."""
        with self._lock:
            return {"tokens": [self._public(entry) for entry in self._tokens]}

    def _save(self, tokens: list[_Entry]) -> None:
        from markitai.security import atomic_write_json

        # Staged beside the file, owner-only, then renamed over it
        atomic_write_json(
            self.path,
            {"version": 1, "tokens": [asdict(entry) for entry in tokens]},
            private=True,
        )

    def create(self, label: str) -> dict[str, Any]:
        """Save a new token named *label*; its public form plus ``token``.

        Blocking: it writes the file.
        """
        with self._write:
            with self._lock:
                current = list(self._tokens)
            if len(current) >= MAX_TOKENS:
                raise ApiError(
                    409,
                    "token_limit",
                    f"there are already {MAX_TOKENS} access tokens; revoke one first",
                )
            token = USER_TOKEN_PREFIX + secrets.token_hex(32)
            token_id = "u_" + secrets.token_hex(6)
            while any(entry.id == token_id for entry in current):
                token_id = "u_" + secrets.token_hex(6)
            entry = _Entry(
                id=token_id,
                label=label,
                sha256=digest(token),
                hint=token[-4:],
                created_at=now_iso(),
            )
            try:
                self._save([*current, entry])
            except OSError as e:
                logger.error(
                    "[Serve] Could not save the access tokens: {}", type(e).__name__
                )
                raise _io_failed() from None
            with self._lock:
                self._tokens = [*current, entry]
                return {**self._public(entry), "token": token}

    def revoke(self, token_id: str) -> str:
        """Remove the token *token_id* and return its label.

        Blocking: it writes the file.
        """
        with self._write:
            with self._lock:
                current = list(self._tokens)
            removed = next((entry for entry in current if entry.id == token_id), None)
            if removed is None:
                raise ApiError(404, "token_not_found", "access token not found")
            remaining = [entry for entry in current if entry is not removed]
            try:
                self._save(remaining)
            except OSError as e:
                logger.error(
                    "[Serve] Could not save the access tokens: {}", type(e).__name__
                )
                raise _io_failed() from None
            with self._lock:
                self._tokens = remaining
                self._used.pop(token_id, None)
        return removed.label
