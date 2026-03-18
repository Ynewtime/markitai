"""SQLite-based fetch cache and SPA domain learning cache.

Extracted from fetch.py to reduce file size and separate caching concerns
from URL fetching logic.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import time
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from loguru import logger

from markitai.fetch_types import FetchResult
from markitai.security import atomic_write_json


def _make_json_safe(value: Any) -> Any:
    """Convert nested metadata values to JSON-safe types."""
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return _make_json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _make_json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set):
        return [_make_json_safe(item) for item in value]
    return str(value)


class FetchCache:
    """SQLite-based cache for fetch results.

    Caches the fetched content by URL to avoid repeated network requests.
    Uses the same LRU eviction strategy as LLM cache.

    Connection reuse: A single connection is reused for all operations
    within the same FetchCache instance to reduce connection overhead.

    Concurrency: Provides both sync methods (using threading.Lock) and async
    methods (using asyncio.Lock, prefixed with 'a') to avoid blocking the
    event loop during concurrent URL fetching. Async callers should use
    aget/aset/aget_with_validators/aset_with_validators/aupdate_accessed_at.

    **Important**: The sync lock (``threading.Lock``) and async lock
    (``asyncio.Lock``) are independent and do NOT provide mutual exclusion
    between sync and async callers.  Callers MUST NOT mix sync and async
    methods on the same instance concurrently.  In practice this is safe
    because the CLI either runs fully synchronous or fully asynchronous
    within a single event loop.  If the library is ever used in a context
    where both sync and async callers operate concurrently, a unified
    locking strategy (e.g. ``anyio.Lock`` or separate connections per
    access mode) will be required.
    """

    def __init__(self, db_path: Path, max_size_bytes: int = 100 * 1024 * 1024) -> None:
        """Initialize fetch cache.

        Args:
            db_path: Path to SQLite database file
            max_size_bytes: Maximum cache size in bytes (default 100MB)
        """
        import threading

        self._db_path = db_path
        self._max_size_bytes = max_size_bytes
        self._connection: sqlite3.Connection | None = None
        self._lock = threading.Lock()  # Protect database operations (sync callers)
        self._async_lock = asyncio.Lock()  # Protect database operations (async callers)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create a reusable database connection.

        Connection is created on first use and reused for subsequent calls.
        Uses check_same_thread=False to allow cross-thread usage in async context.
        Note: Callers must hold self._lock when calling this method.
        """
        if self._connection is None:
            self._connection = sqlite3.connect(
                str(self._db_path),
                timeout=30.0,
                check_same_thread=False,  # Allow cross-thread usage for async
            )
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=NORMAL")
            self._connection.row_factory = sqlite3.Row
        return self._connection

    def close(self) -> None:
        """Close the database connection.

        Call this during cleanup to release resources.
        """
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def _init_db(self) -> None:
        """Initialize database schema with migration support."""
        with self._lock:
            conn = self._get_connection()
            conn.execute("""
                CREATE TABLE IF NOT EXISTS fetch_cache (
                    key TEXT PRIMARY KEY,
                    url TEXT NOT NULL,
                    content TEXT NOT NULL,
                    strategy_used TEXT NOT NULL,
                    title TEXT,
                    final_url TEXT,
                    metadata TEXT,
                    created_at INTEGER NOT NULL,
                    accessed_at INTEGER NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    etag TEXT,
                    last_modified TEXT,
                    screenshot_path TEXT,
                    static_content TEXT,
                    browser_content TEXT
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_fetch_accessed ON fetch_cache(accessed_at)"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_fetch_url ON fetch_cache(url)")
            conn.commit()

            # Migration: Add etag and last_modified columns if they don't exist
            self._migrate_add_http_validators(conn)
            # Migration: Add multi-source content columns if they don't exist
            self._migrate_add_multi_source_columns(conn)

    def _migrate_add_http_validators(self, conn: sqlite3.Connection) -> None:
        """Add etag and last_modified columns if they don't exist (migration)."""
        cursor = conn.execute("PRAGMA table_info(fetch_cache)")
        columns = {row[1] for row in cursor.fetchall()}

        if "etag" not in columns:
            conn.execute("ALTER TABLE fetch_cache ADD COLUMN etag TEXT")
            logger.debug("[FetchCache] Migration: Added 'etag' column")

        if "last_modified" not in columns:
            conn.execute("ALTER TABLE fetch_cache ADD COLUMN last_modified TEXT")
            logger.debug("[FetchCache] Migration: Added 'last_modified' column")

        conn.commit()

    def _migrate_add_multi_source_columns(self, conn: sqlite3.Connection) -> None:
        """Add screenshot_path, static_content, browser_content columns if missing."""
        cursor = conn.execute("PRAGMA table_info(fetch_cache)")
        columns = {row[1] for row in cursor.fetchall()}

        for col in ("screenshot_path", "static_content", "browser_content"):
            if col not in columns:
                conn.execute(f"ALTER TABLE fetch_cache ADD COLUMN {col} TEXT")
                logger.debug(f"[FetchCache] Migration: Added '{col}' column")

        conn.commit()

    def _compute_hash(self, url: str, strategy: str | None = None) -> str:
        """Compute hash key from URL and optional strategy.

        When strategy is provided, the cache key is scoped to that strategy,
        preventing cache collisions when the same URL is fetched with different
        strategies (e.g., --playwright vs --static).

        Args:
            url: URL to hash
            strategy: Optional strategy name to include in key
        """
        key_input = url if strategy is None else f"{url}\x00{strategy}"
        return hashlib.sha256(key_input.encode()).hexdigest()[:32]

    def _metadata_to_json(self, metadata: dict[str, Any]) -> str:
        """Serialize fetch metadata, coercing unsupported values to JSON-safe types."""
        return json.dumps(_make_json_safe(metadata))

    def _get_unlocked(
        self, url: str, strategy: str | None = None
    ) -> FetchResult | None:
        """Get cached fetch result (no lock). Caller must hold a lock."""
        key = self._compute_hash(url, strategy)
        now = int(time.time())

        conn = self._get_connection()
        row = conn.execute("SELECT * FROM fetch_cache WHERE key = ?", (key,)).fetchone()

        if row:
            # Update accessed_at for LRU tracking
            conn.execute(
                "UPDATE fetch_cache SET accessed_at = ? WHERE key = ?", (now, key)
            )
            conn.commit()

            metadata = json.loads(row["metadata"]) if row["metadata"] else {}
            logger.debug(f"[FetchCache] Cache hit for URL: {url}")

            # Restore screenshot_path as Path if stored
            screenshot_path_str = row["screenshot_path"]
            screenshot_path = Path(screenshot_path_str) if screenshot_path_str else None

            return FetchResult(
                content=row["content"],
                strategy_used=row["strategy_used"],
                title=row["title"],
                url=row["url"],
                final_url=row["final_url"],
                metadata=metadata,
                cache_hit=True,
                screenshot_path=screenshot_path,
                static_content=row["static_content"],
                browser_content=row["browser_content"],
            )

        return None

    def get(self, url: str, strategy: str | None = None) -> FetchResult | None:
        """Get cached fetch result if exists.

        Args:
            url: URL to look up
            strategy: Optional strategy to scope cache lookup

        Returns:
            Cached FetchResult or None if not found
        """
        with self._lock:
            return self._get_unlocked(url, strategy)

    async def aget(self, url: str, strategy: str | None = None) -> FetchResult | None:
        """Async version of get() using asyncio.Lock."""
        async with self._async_lock:
            return self._get_unlocked(url, strategy)

    def _set_unlocked(
        self, url: str, result: FetchResult, strategy: str | None = None
    ) -> None:
        """Cache a fetch result (no lock). Caller must hold a lock."""
        key = self._compute_hash(url, strategy)
        now = int(time.time())
        metadata_json = (
            self._metadata_to_json(result.metadata) if result.metadata else None
        )
        size_bytes = len(result.content.encode("utf-8"))

        conn = self._get_connection()
        # Check current total size
        total_size = conn.execute(
            "SELECT COALESCE(SUM(size_bytes), 0) as total FROM fetch_cache"
        ).fetchone()["total"]

        # Evict LRU entries if needed
        while total_size + size_bytes > self._max_size_bytes:
            oldest = conn.execute(
                "SELECT key, size_bytes FROM fetch_cache ORDER BY accessed_at ASC LIMIT 1"
            ).fetchone()

            if oldest is None:
                break

            conn.execute("DELETE FROM fetch_cache WHERE key = ?", (oldest["key"],))
            total_size -= oldest["size_bytes"]
            logger.debug(f"[FetchCache] Evicted LRU entry: {oldest['key'][:8]}...")

        # Serialize screenshot_path to string for storage
        screenshot_path_str = (
            str(result.screenshot_path) if result.screenshot_path else None
        )

        # Insert or replace
        conn.execute(
            """
            INSERT OR REPLACE INTO fetch_cache
            (key, url, content, strategy_used, title, final_url, metadata,
             created_at, accessed_at, size_bytes,
             screenshot_path, static_content, browser_content)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                key,
                url,
                result.content,
                result.strategy_used,
                result.title,
                result.final_url,
                metadata_json,
                now,
                now,
                size_bytes,
                screenshot_path_str,
                result.static_content,
                result.browser_content,
            ),
        )
        conn.commit()
        logger.debug(f"[FetchCache] Cached URL: {url} ({size_bytes} bytes)")

    def set(self, url: str, result: FetchResult, strategy: str | None = None) -> None:
        """Cache a fetch result.

        Args:
            url: URL that was fetched
            result: FetchResult to cache
            strategy: Optional strategy to scope cache entry
        """
        with self._lock:
            self._set_unlocked(url, result, strategy)

    async def aset(
        self, url: str, result: FetchResult, strategy: str | None = None
    ) -> None:
        """Async version of set() using asyncio.Lock."""
        async with self._async_lock:
            self._set_unlocked(url, result, strategy)

    def stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        with self._lock:
            conn = self._get_connection()
            row = conn.execute(
                """
                SELECT COUNT(*) as count, COALESCE(SUM(size_bytes), 0) as size_bytes
                FROM fetch_cache
                """
            ).fetchone()

        return {
            "count": row["count"],
            "size_bytes": row["size_bytes"],
            "size_mb": round(row["size_bytes"] / (1024 * 1024), 2),
            "max_size_mb": round(self._max_size_bytes / (1024 * 1024), 2),
            "db_path": str(self._db_path),
        }

    def clear(self) -> int:
        """Clear all entries.

        Returns:
            Number of entries deleted
        """
        with self._lock:
            conn = self._get_connection()
            count = conn.execute("SELECT COUNT(*) as cnt FROM fetch_cache").fetchone()[
                "cnt"
            ]
            conn.execute("DELETE FROM fetch_cache")
            conn.commit()
        return count

    def _get_with_validators_unlocked(
        self, url: str, strategy: str | None = None
    ) -> tuple[FetchResult | None, str | None, str | None]:
        """Get cached result with HTTP validators (no lock). Caller must hold a lock."""
        key = self._compute_hash(url, strategy)

        conn = self._get_connection()
        row = conn.execute("SELECT * FROM fetch_cache WHERE key = ?", (key,)).fetchone()

        if row:
            metadata = json.loads(row["metadata"]) if row["metadata"] else {}
            screenshot_path_str = row["screenshot_path"]
            screenshot_path = Path(screenshot_path_str) if screenshot_path_str else None
            result = FetchResult(
                content=row["content"],
                strategy_used=row["strategy_used"],
                title=row["title"],
                url=row["url"],
                final_url=row["final_url"],
                metadata=metadata,
                cache_hit=True,
                screenshot_path=screenshot_path,
                static_content=row["static_content"],
                browser_content=row["browser_content"],
            )
            return result, row["etag"], row["last_modified"]

        return None, None, None

    def get_with_validators(
        self, url: str, strategy: str | None = None
    ) -> tuple[FetchResult | None, str | None, str | None]:
        """Get cached result with HTTP validators for conditional requests.

        Args:
            url: URL to look up
            strategy: Optional strategy to scope cache lookup

        Returns:
            Tuple of (cached_result, etag, last_modified)
            - cached_result: FetchResult if found, None otherwise
            - etag: ETag header from previous fetch (for If-None-Match)
            - last_modified: Last-Modified header from previous fetch (for If-Modified-Since)
        """
        with self._lock:
            return self._get_with_validators_unlocked(url, strategy)

    async def aget_with_validators(
        self, url: str, strategy: str | None = None
    ) -> tuple[FetchResult | None, str | None, str | None]:
        """Async version of get_with_validators() using asyncio.Lock."""
        async with self._async_lock:
            return self._get_with_validators_unlocked(url, strategy)

    def _set_with_validators_unlocked(
        self,
        url: str,
        result: FetchResult,
        etag: str | None = None,
        last_modified: str | None = None,
        strategy: str | None = None,
    ) -> None:
        """Cache a fetch result with HTTP validators (no lock). Caller must hold a lock."""
        key = self._compute_hash(url, strategy)
        now = int(time.time())
        metadata_json = (
            self._metadata_to_json(result.metadata) if result.metadata else None
        )
        size_bytes = len(result.content.encode("utf-8"))

        conn = self._get_connection()
        # Check current total size
        total_size = conn.execute(
            "SELECT COALESCE(SUM(size_bytes), 0) as total FROM fetch_cache"
        ).fetchone()["total"]

        # Evict LRU entries if needed
        while total_size + size_bytes > self._max_size_bytes:
            oldest = conn.execute(
                "SELECT key, size_bytes FROM fetch_cache ORDER BY accessed_at ASC LIMIT 1"
            ).fetchone()

            if oldest is None:
                break

            conn.execute("DELETE FROM fetch_cache WHERE key = ?", (oldest["key"],))
            total_size -= oldest["size_bytes"]
            logger.debug(f"[FetchCache] Evicted LRU entry: {oldest['key'][:8]}...")

        # Serialize screenshot_path to string for storage
        screenshot_path_str = (
            str(result.screenshot_path) if result.screenshot_path else None
        )

        # Insert or replace
        conn.execute(
            """
            INSERT OR REPLACE INTO fetch_cache
            (key, url, content, strategy_used, title, final_url, metadata,
             created_at, accessed_at, size_bytes, etag, last_modified,
             screenshot_path, static_content, browser_content)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                key,
                url,
                result.content,
                result.strategy_used,
                result.title,
                result.final_url,
                metadata_json,
                now,
                now,
                size_bytes,
                etag,
                last_modified,
                screenshot_path_str,
                result.static_content,
                result.browser_content,
            ),
        )
        conn.commit()
        logger.debug(
            f"[FetchCache] Cached URL with validators: {url} "
            f"(etag={etag is not None}, last_modified={last_modified is not None})"
        )

    def set_with_validators(
        self,
        url: str,
        result: FetchResult,
        etag: str | None = None,
        last_modified: str | None = None,
        strategy: str | None = None,
    ) -> None:
        """Cache a fetch result with HTTP validators.

        Args:
            url: URL that was fetched
            result: FetchResult to cache
            etag: ETag response header (for future If-None-Match)
            last_modified: Last-Modified response header (for future If-Modified-Since)
            strategy: Optional strategy to scope cache entry
        """
        with self._lock:
            self._set_with_validators_unlocked(
                url, result, etag, last_modified, strategy
            )

    async def aset_with_validators(
        self,
        url: str,
        result: FetchResult,
        etag: str | None = None,
        last_modified: str | None = None,
        strategy: str | None = None,
    ) -> None:
        """Async version of set_with_validators() using asyncio.Lock."""
        async with self._async_lock:
            self._set_with_validators_unlocked(
                url, result, etag, last_modified, strategy
            )

    def _update_accessed_at_unlocked(
        self, url: str, strategy: str | None = None
    ) -> None:
        """Update accessed_at timestamp (no lock). Caller must hold a lock."""
        key = self._compute_hash(url, strategy)
        now = int(time.time())

        conn = self._get_connection()
        conn.execute("UPDATE fetch_cache SET accessed_at = ? WHERE key = ?", (now, key))
        conn.commit()

    def update_accessed_at(self, url: str, strategy: str | None = None) -> None:
        """Update accessed_at timestamp for cache hit (304 response).

        Args:
            url: URL to update
            strategy: Optional strategy to scope update
        """
        with self._lock:
            self._update_accessed_at_unlocked(url, strategy)

    async def aupdate_accessed_at(self, url: str, strategy: str | None = None) -> None:
        """Async version of update_accessed_at() using asyncio.Lock."""
        async with self._async_lock:
            self._update_accessed_at_unlocked(url, strategy)


class SPADomainCache:
    """JSON-based cache for learned SPA domains.

    When static fetching fails (detected as JS-required), the domain is
    recorded here. On subsequent requests to the same domain, browser
    rendering is used directly, avoiding the wasted static request.

    Storage format:
    {
      "domains": {
        "example.com": {
          "learned_at": "2026-01-27T12:00:00",
          "hits": 3,
          "last_hit": "2026-01-27T15:30:00"
        }
      },
      "version": 1
    }
    """

    # Cache file expires after 30 days of no hits
    EXPIRY_DAYS = 30
    VERSION = 1

    def __init__(self, cache_path: Path | None = None) -> None:
        """Initialize SPA domain cache.

        Args:
            cache_path: Path to cache file. Defaults to ~/.markitai/learned_spa_domains.json
        """
        if cache_path is None:
            cache_path = Path.home() / ".markitai" / "learned_spa_domains.json"
        self._cache_path = cache_path
        self._data: dict[str, Any] = {"domains": {}, "version": self.VERSION}
        self._load()

    def _load(self) -> None:
        """Load cache from disk."""
        if self._cache_path.exists():
            try:
                with open(self._cache_path, encoding="utf-8") as f:
                    data = json.load(f)
                    if data.get("version") == self.VERSION:
                        self._data = data
                    else:
                        logger.debug("SPA domain cache version mismatch, resetting")
            except (json.JSONDecodeError, OSError) as e:
                logger.debug(f"Failed to load SPA domain cache: {e}")

    def _save(self) -> None:
        """Save cache to disk atomically."""
        try:
            # Ensure parent directory exists before saving
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(self._cache_path, self._data)
        except OSError as e:
            logger.warning(f"Failed to save SPA domain cache: {e}")

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        parsed = urlparse(url)
        return parsed.netloc.lower()

    def _is_expired(self, entry: dict[str, Any]) -> bool:
        """Check if a cache entry has expired."""
        from datetime import datetime, timedelta

        last_hit_str = entry.get("last_hit") or entry.get("learned_at", "")
        if not last_hit_str:
            return True
        try:
            last_hit = datetime.fromisoformat(last_hit_str)
            # Handle both naive (old cache) and aware (new cache) timestamps
            if last_hit.tzinfo is None:
                last_hit = last_hit.astimezone()
            return datetime.now().astimezone() - last_hit > timedelta(
                days=self.EXPIRY_DAYS
            )
        except ValueError:
            return True

    def is_known_spa(self, url: str) -> bool:
        """Check if URL's domain is a known SPA that needs browser rendering.

        Args:
            url: URL to check

        Returns:
            True if domain was previously learned to need browser rendering
        """
        domain = self._extract_domain(url)
        if domain in self._data["domains"]:
            entry = self._data["domains"][domain]
            if self._is_expired(entry):
                # Remove expired entry
                del self._data["domains"][domain]
                self._save()
                return False
            return True
        return False

    def record_spa_domain(self, url: str) -> None:
        """Record that a domain needs browser rendering.

        Args:
            url: URL whose domain should be recorded
        """
        from datetime import datetime

        domain = self._extract_domain(url)
        now = datetime.now().astimezone().isoformat()

        if domain in self._data["domains"]:
            # Update existing entry
            self._data["domains"][domain]["hits"] += 1
            self._data["domains"][domain]["last_hit"] = now
        else:
            # New entry
            self._data["domains"][domain] = {
                "learned_at": now,
                "hits": 1,
                "last_hit": now,
            }
            logger.info(f"Learned new SPA domain: {domain}")

        self._save()

    def record_hit(self, url: str) -> None:
        """Record a cache hit (used the cached knowledge).

        Args:
            url: URL that was fetched using cached knowledge
        """
        from datetime import datetime

        domain = self._extract_domain(url)
        if domain in self._data["domains"]:
            self._data["domains"][domain]["hits"] += 1
            self._data["domains"][domain]["last_hit"] = (
                datetime.now().astimezone().isoformat()
            )
            self._save()

    def clear(self) -> int:
        """Clear all learned domains.

        Returns:
            Number of domains cleared
        """
        count = len(self._data["domains"])
        self._data["domains"] = {}
        self._save()
        return count

    def list_domains(self) -> list[dict[str, Any]]:
        """List all learned domains.

        Returns:
            List of domain entries with metadata
        """
        result = []
        for domain, entry in self._data["domains"].items():
            result.append(
                {
                    "domain": domain,
                    "learned_at": entry.get("learned_at"),
                    "hits": entry.get("hits", 0),
                    "last_hit": entry.get("last_hit"),
                    "expired": self._is_expired(entry),
                }
            )
        return sorted(result, key=lambda x: x.get("hits", 0), reverse=True)
