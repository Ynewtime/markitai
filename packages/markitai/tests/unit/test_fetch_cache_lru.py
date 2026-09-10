from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from markitai.fetch_cache import FetchCache
from markitai.fetch_types import FetchResult


def _read_accessed_at(db_path: Path, url: str) -> int:
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT accessed_at FROM fetch_cache WHERE url = ?",
            (url,),
        ).fetchone()
        assert row is not None
        return int(row[0])
    finally:
        conn.close()


def test_get_with_validators_refreshes_accessed_at(tmp_path: Path) -> None:
    """Validator-backed cache hits should participate in LRU refresh like normal hits."""
    db_path = tmp_path / "test_cache.db"
    cache = FetchCache(db_path)
    url = "https://example.com/page"

    try:
        result = FetchResult(content="# Test", strategy_used="static", url=url)

        with patch("markitai.fetch_cache.time.time", return_value=1000.0):
            cache.set_with_validators(url, result, etag='"etag-1"', last_modified=None)

        before = _read_accessed_at(db_path, url)

        with patch("markitai.fetch_cache.time.time", return_value=1005.0):
            cached_result, etag, last_modified = cache.get_with_validators(url)

        after = _read_accessed_at(db_path, url)

        assert cached_result is not None
        assert etag == '"etag-1"'
        assert last_modified is None
        assert after > before
    finally:
        cache.close()


@pytest.mark.parametrize("kind", ["llm", "fetch", "validators"])
@pytest.mark.parametrize(
    "replacement", ["WXYZ", "x", "abcdefgh", "too-large-for-cache"]
)
def test_replacement_admits_only_net_size(
    tmp_path: Path, kind: str, replacement: str
) -> None:
    from markitai.llm.cache import SQLiteCache

    cache = (
        SQLiteCache(tmp_path / "cache.db", max_size_bytes=10)
        if kind == "llm"
        else FetchCache(tmp_path / "cache.db", max_size_bytes=10)
    )

    def put(key: str, value: str) -> None:
        if isinstance(cache, SQLiteCache):
            cache.set("prompt", key, value)
        else:
            result = FetchResult(content=value, strategy_used="static", url=key)
            if kind == "validators":
                cache.set_with_validators(key, result, etag="v2")
            else:
                cache.set(key, result)

    def get(key: str) -> str | None:
        if isinstance(cache, SQLiteCache):
            return cache.get("prompt", key)
        result = cache.get(key)
        return result.content if result else None

    try:
        with patch("time.time", return_value=1000):
            put("A", "123456")
        with patch("time.time", return_value=1001):
            put("B", "1234")
        with patch("time.time", return_value=1002):
            put("B", replacement)
            assert get("B") == (replacement if len(replacement) <= 10 else None)
            assert get("A") == (None if 4 < len(replacement) <= 10 else "123456")
    finally:
        cache.close()
