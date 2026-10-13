"""Single-use download tickets.

A browser saves a large file (an archive) by navigating to it, and a
navigation cannot carry the ``Authorization`` header. An authenticated client
therefore first asks ``POST /api/download-tickets`` for a ticket: a random
value that admits exactly one GET of exactly one download path within a
minute. The ticket rides in the URL instead of the access token; once used or
expired it is worthless, so a URL kept in the browser's download list or
history grants nothing. Only the SHA-256 of a ticket is held. A ticket acts
as the caller it was issued to: a user token's ticket sees only that user's
jobs and dies with the token.
"""

from __future__ import annotations

import hashlib
import secrets
import threading
import time
from collections.abc import Callable

#: How long a ticket stays valid after it is issued, in seconds.
LIFETIME_S = 60
#: Outstanding tickets at most; expired ones are dropped before counting.
CAPACITY = 64
MAX_PATH = 4096


def _digest(ticket: str) -> bytes:
    return hashlib.sha256(ticket.encode("utf-8", "surrogatepass")).digest()


def _plain_segment(segment: str) -> bool:
    return bool(segment) and segment not in (".", "..")


def downloadable(path: str) -> bool:
    """Whether *path* is a download route a ticket may name.

    A job's file, a job's archive, an item's original (its bytes or its
    framed page view), an item's archive, or the history archive. Paths are compared as sent, so they must be the exact encoded
    form the browser will request.
    """
    if (
        len(path) > MAX_PATH
        or not path.isascii()
        or any(char in path for char in "?#\\ ")
        or any(ord(char) < 0x20 or ord(char) == 0x7F for char in path)
    ):
        return False
    if path == "/api/history/archive":
        return True
    rest = path.removeprefix("/api/jobs/")
    if rest == path:
        return False
    job_id, _, rest = rest.partition("/")
    if not _plain_segment(job_id):
        return False
    if rest == "archive":
        return True
    if rest.startswith("files/"):
        segments = rest.removeprefix("files/").split("/")
        return all(_plain_segment(segment) for segment in segments)
    # items/{item_id}/archive, items/{item_id}/original/file|view
    parts = rest.split("/")
    if len(parts) < 3 or parts[0] != "items" or not _plain_segment(parts[1]):
        return False
    return parts[2:] in (["archive"], ["original", "file"], ["original", "view"])


class DownloadTickets:
    """In-memory ticket store; only digests are kept."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        # digest -> (path, expiry, the user token id it was issued to)
        self._tickets: dict[bytes, tuple[str, float, str | None]] = {}

    def issue(self, path: str, user: str | None = None) -> str | None:
        """A new ticket for *path*, or None when too many are outstanding.

        *user* is the id of the user token that asked (None: the admin).
        """
        now = self._clock()
        with self._lock:
            self._tickets = {
                digest: entry
                for digest, entry in self._tickets.items()
                if entry[1] > now
            }
            if len(self._tickets) >= CAPACITY:
                return None
            ticket = secrets.token_hex(32)
            self._tickets[_digest(ticket)] = (path, now + LIFETIME_S, user)
        return ticket

    def clear(self) -> None:
        """Revoke every outstanding ticket (the access token changed)."""
        with self._lock:
            self._tickets.clear()

    def revoke_user(self, user: str) -> None:
        """Revoke the outstanding tickets of the user token *user*."""
        with self._lock:
            self._tickets = {
                digest: entry
                for digest, entry in self._tickets.items()
                if entry[2] != user
            }

    def redeem(self, ticket: str, path: str) -> tuple[bool, str | None]:
        """Spend *ticket* on a GET of *path*: (valid, the user it was issued to).

        A ticket presented for another path is spent as well: it was
        disclosed, so it must not stay usable.
        """
        now = self._clock()
        with self._lock:
            entry = self._tickets.pop(_digest(ticket), None)
        if entry is None:
            return False, None
        issued_path, expires, user = entry
        valid = expires > now and secrets.compare_digest(
            issued_path.encode(), path.encode()
        )
        return valid, user
