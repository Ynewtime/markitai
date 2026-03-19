from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import patch

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
