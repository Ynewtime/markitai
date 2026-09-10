"""Transactional capacity admission shared by the SQLite caches."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Literal

_QUERIES = {
    "cache": (
        "DELETE FROM cache WHERE key = ?",
        "SELECT COALESCE(SUM(size_bytes), 0) FROM cache",
        "SELECT key, size_bytes FROM cache ORDER BY accessed_at ASC, rowid ASC LIMIT 1",
    ),
    "fetch_cache": (
        "DELETE FROM fetch_cache WHERE key = ?",
        "SELECT COALESCE(SUM(size_bytes), 0) FROM fetch_cache",
        "SELECT key, size_bytes FROM fetch_cache ORDER BY accessed_at ASC, rowid ASC LIMIT 1",
    ),
}


@contextmanager
def cache_write(
    connection: sqlite3.Connection,
    table: Literal["cache", "fetch_cache"],
    key: str,
    size: int,
    limit: int,
) -> Iterator[bool]:
    """Reserve capacity for a replacement and commit its insert atomically.

    The caller inserts only when admission succeeds. Oversized values remove
    the stale value for that key without evicting unrelated entries. A failed
    insert rolls back both the replacement and eviction.
    """
    delete, total, oldest = _QUERIES[table]
    with connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(delete, (key,))
        if size > limit:
            yield False
            return
        total_size = connection.execute(total).fetchone()[0]
        while total_size + size > limit:
            entry = connection.execute(oldest).fetchone()
            if entry is None:
                break
            connection.execute(delete, (entry[0],))
            total_size -= entry[1]
        yield True
