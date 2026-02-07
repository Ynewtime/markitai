"""LLM caching system.

This module provides caching for LLM responses:
- SQLiteCache: Persistent LRU cache with SQLite backend
- PersistentCache: Global cache wrapper with pattern-based skip
- ContentCache: In-memory TTL LRU cache
"""

from __future__ import annotations

import json
import time
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from loguru import logger

from markitai.constants import (
    DEFAULT_CACHE_DB_FILENAME,
    DEFAULT_CACHE_MAXSIZE,
    DEFAULT_CACHE_SIZE_LIMIT,
    DEFAULT_CACHE_TTL_SECONDS,
    DEFAULT_GLOBAL_CACHE_DIR,
)


class SQLiteCache:
    """SQLite-based persistent LRU cache with size limit.

    Thread-safe via SQLite's built-in locking mechanism.
    Uses WAL mode for better concurrent read performance.
    """

    def __init__(
        self,
        db_path: Path,
        max_size_bytes: int = DEFAULT_CACHE_SIZE_LIMIT,
    ) -> None:
        """Initialize SQLite cache.

        Args:
            db_path: Path to the SQLite database file
            max_size_bytes: Maximum total cache size in bytes (default 1GB)
        """
        import hashlib

        self._db_path = Path(db_path)
        self._max_size_bytes = max_size_bytes
        self._hashlib = hashlib

        # Ensure parent directory exists
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_db()

    def _get_connection(self) -> Any:
        """Create a raw database connection.

        Prefer using _connect() context manager which ensures the
        connection is closed after use.
        """
        import sqlite3

        conn = sqlite3.connect(str(self._db_path), timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def _connect(self) -> Iterator[Any]:
        """Context manager that creates a connection and closes it on exit."""
        conn = self._get_connection()
        try:
            yield conn
        finally:
            conn.close()

    def close(self) -> None:
        """Explicit close (no-op — connections are per-call and auto-closed)."""

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    model TEXT DEFAULT '',
                    created_at INTEGER NOT NULL,
                    accessed_at INTEGER NOT NULL,
                    size_bytes INTEGER NOT NULL
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_accessed ON cache(accessed_at)"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_created ON cache(created_at)")
            conn.commit()

    def _compute_hash(self, prompt: str, content: str) -> str:
        """Compute hash key from prompt and content.

        Uses head + tail + length strategy to detect most changes.
        Note: middle-only edits in content >50k chars may not be detected.
        """
        length = len(content)
        head = content[:25000]
        tail = content[-25000:] if length > 25000 else ""
        combined = f"{prompt}|{length}|{head}|{tail}"
        return self._hashlib.sha256(combined.encode()).hexdigest()[:32]

    def get(self, prompt: str, content: str) -> str | None:
        """Get cached value if exists, update accessed_at for LRU.

        Args:
            prompt: Prompt template used
            content: Content being processed

        Returns:
            Cached JSON string or None if not found
        """
        key = self._compute_hash(prompt, content)
        now = int(time.time())

        with self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM cache WHERE key = ?", (key,)
            ).fetchone()

            if row:
                # Update accessed_at for LRU tracking
                conn.execute(
                    "UPDATE cache SET accessed_at = ? WHERE key = ?", (now, key)
                )
                conn.commit()
                return row["value"]

        return None

    def set(self, prompt: str, content: str, value: str, model: str = "") -> None:
        """Set cache value, evict LRU entries if size exceeded.

        Args:
            prompt: Prompt template used
            content: Content being processed
            value: JSON string to cache
            model: Model identifier (for potential invalidation)
        """
        key = self._compute_hash(prompt, content)
        now = int(time.time())
        size_bytes = len(value.encode("utf-8"))

        with self._connect() as conn:
            # Check current total size
            total_size = conn.execute(
                "SELECT COALESCE(SUM(size_bytes), 0) as total FROM cache"
            ).fetchone()["total"]

            # Evict LRU entries if needed
            while total_size + size_bytes > self._max_size_bytes:
                oldest = conn.execute(
                    "SELECT key, size_bytes FROM cache ORDER BY accessed_at ASC LIMIT 1"
                ).fetchone()

                if oldest is None:
                    break

                conn.execute("DELETE FROM cache WHERE key = ?", (oldest["key"],))
                total_size -= oldest["size_bytes"]
                logger.debug(f"[Cache] Evicted LRU entry: {oldest['key'][:8]}...")

            # Insert or replace
            conn.execute(
                """
                INSERT OR REPLACE INTO cache (key, value, model, created_at, accessed_at, size_bytes)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (key, value, model, now, now, size_bytes),
            )
            conn.commit()

    def clear(self) -> int:
        """Clear all entries.

        Returns:
            Number of entries deleted
        """
        with self._connect() as conn:
            count = conn.execute("SELECT COUNT(*) as cnt FROM cache").fetchone()["cnt"]
            conn.execute("DELETE FROM cache")
            conn.commit()
            return count

    def stats(self) -> dict[str, Any]:
        """Return cache statistics.

        Returns:
            Dict with count, size_bytes, size_mb, db_path
        """
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) as count, COALESCE(SUM(size_bytes), 0) as size_bytes
                FROM cache
            """
            ).fetchone()

            return {
                "count": row["count"],
                "size_bytes": row["size_bytes"],
                "size_mb": round(row["size_bytes"] / (1024 * 1024), 2),
                "max_size_mb": round(self._max_size_bytes / (1024 * 1024), 2),
                "db_path": str(self._db_path),
            }

    def stats_verbose(self) -> dict[str, Any]:
        """Return cache statistics with model breakdown in a single query.

        Combines stats() and stats_by_model() into a single table scan.

        Returns:
            Dict with count, size_bytes, size_mb, max_size_mb, db_path, by_model
        """
        with self._connect() as conn:
            # Single query for both overall stats and per-model breakdown
            cursor = conn.execute("""
                SELECT
                    COALESCE(NULLIF(model, ''), 'unknown') as model_name,
                    COUNT(*) as count,
                    COALESCE(SUM(size_bytes), 0) as total_size
                FROM cache
                GROUP BY model_name
                ORDER BY total_size DESC
            """)

            by_model = {}
            total_count = 0
            total_size = 0

            for row in cursor.fetchall():
                model_name = row["model_name"]
                count = row["count"]
                size = row["total_size"]
                by_model[model_name] = {
                    "count": count,
                    "size_bytes": size,
                    "size_mb": round(size / (1024 * 1024), 2),
                }
                total_count += count
                total_size += size

            return {
                "count": total_count,
                "size_bytes": total_size,
                "size_mb": round(total_size / (1024 * 1024), 2),
                "max_size_mb": round(self._max_size_bytes / (1024 * 1024), 2),
                "db_path": str(self._db_path),
                "by_model": by_model,
            }

    def stats_by_model(self) -> dict[str, dict[str, Any]]:
        """Get cache statistics grouped by model.

        Returns:
            Dict mapping model name to {"count": int, "size_bytes": int, "size_mb": float}
        """
        with self._connect() as conn:
            cursor = conn.execute("""
                SELECT
                    COALESCE(NULLIF(model, ''), 'unknown') as model_name,
                    COUNT(*) as count,
                    COALESCE(SUM(size_bytes), 0) as total_size
                FROM cache
                GROUP BY model_name
                ORDER BY total_size DESC
            """)
            result = {}
            for row in cursor.fetchall():
                result[row["model_name"]] = {
                    "count": row["count"],
                    "size_bytes": row["total_size"],
                    "size_mb": round(row["total_size"] / (1024 * 1024), 2),
                }
            return result

    def list_entries(self, limit: int = 20) -> list[dict[str, Any]]:
        """List recent cache entries.

        Args:
            limit: Maximum number of entries to return.

        Returns:
            List of entry dicts with key, model, size_bytes, created_at,
            accessed_at, preview.
        """
        from datetime import UTC, datetime

        with self._connect() as conn:
            cursor = conn.execute(
                """
                SELECT
                    key,
                    model,
                    size_bytes,
                    created_at,
                    accessed_at,
                    substr(value, 1, 200) as value_preview
                FROM cache
                ORDER BY accessed_at DESC
                LIMIT ?
            """,
                (limit,),
            )
            entries = []
            for row in cursor.fetchall():
                entries.append(
                    {
                        "key": row["key"],
                        "model": row["model"] or "unknown",
                        "size_bytes": row["size_bytes"],
                        "created_at": datetime.fromtimestamp(
                            row["created_at"], tz=UTC
                        ).isoformat(),
                        "accessed_at": datetime.fromtimestamp(
                            row["accessed_at"], tz=UTC
                        ).isoformat(),
                        "preview": self._parse_value_preview(row["value_preview"]),
                    }
                )
            return entries

    def _parse_value_preview(self, value: str | None) -> str:
        """Parse cached value to generate a human-readable preview."""
        if not value:
            return ""
        try:
            data = json.loads(value)
            if isinstance(data, dict):
                if "caption" in data:
                    caption = str(data["caption"])[:40]
                    return f"image: {caption}..."
                if "title" in data:
                    title = str(data["title"])[:40]
                    return f"frontmatter: {title}..."
            text = str(data) if not isinstance(data, str) else data
            return f"text: {text[:40]}..."
        except (json.JSONDecodeError, TypeError):
            return f"text: {value[:40]}..."


class PersistentCache:
    """Global persistent cache with pattern-based skip support.

    Provides a single global cache (~/.markitai/cache.db) with:
    - Pattern-based cache skipping for specific files
    - Hit/miss tracking for statistics
    """

    def __init__(
        self,
        global_dir: Path | None = None,
        max_size_bytes: int = DEFAULT_CACHE_SIZE_LIMIT,
        enabled: bool = True,
        skip_read: bool = False,
        no_cache_patterns: list[str] | None = None,
    ) -> None:
        """Initialize global cache.

        Args:
            global_dir: Global cache directory (default ~/.markitai)
            max_size_bytes: Max cache size in bytes
            enabled: Whether caching is enabled (both read and write)
            skip_read: If True, skip reading from cache but still write
            no_cache_patterns: List of glob patterns to skip cache for specific files
        """
        self._enabled = enabled
        self._skip_read = skip_read
        self._no_cache_patterns = no_cache_patterns or []
        self._global_cache: SQLiteCache | None = None
        self._hits = 0
        self._misses = 0

        if not enabled:
            return

        # Initialize global cache
        global_cache_dir = global_dir or Path(DEFAULT_GLOBAL_CACHE_DIR).expanduser()
        global_cache_path = Path(global_cache_dir) / DEFAULT_CACHE_DB_FILENAME
        try:
            self._global_cache = SQLiteCache(global_cache_path, max_size_bytes)
            logger.debug(f"[Cache] Global cache: {global_cache_path}")
        except Exception as e:
            logger.warning(f"[Cache] Failed to init global cache: {e}")

    def close(self) -> None:
        """Close underlying SQLiteCache."""
        if self._global_cache:
            self._global_cache.close()

    def _glob_match(self, path: str, pattern: str) -> bool:
        """Enhanced glob matching for ** patterns."""
        import fnmatch

        if fnmatch.fnmatch(path, pattern):
            return True

        if pattern.startswith("**/"):
            pattern_without_prefix = pattern[3:]
            if fnmatch.fnmatch(path, pattern_without_prefix):
                return True

        if "**/" in pattern and not pattern.startswith("**/"):
            collapsed = pattern.replace("**/", "", 1)
            if fnmatch.fnmatch(path, collapsed):
                return True

        return False

    def _extract_matchable_path(self, context: str) -> str:
        """Extract a matchable file path from various context formats."""
        path = context.replace("\\", "/")

        if ":" in path:
            colon_idx = path.index(":")
            if colon_idx == 1:
                rest = path[2:]
                if ":" in rest:
                    path = path[: 2 + rest.index(":")]
            else:
                path = path[:colon_idx]

        if "/" in path:
            filename = path.rsplit("/", 1)[-1]
        else:
            filename = path

        return filename

    def _should_skip_cache(self, context: str) -> bool:
        """Check if cache should be skipped for the given context."""
        if not context or not self._no_cache_patterns:
            return False

        normalized_context = context.replace("\\", "/")
        filename = self._extract_matchable_path(context)

        for pattern in self._no_cache_patterns:
            normalized_pattern = pattern.replace("\\", "/")

            if self._glob_match(normalized_context, normalized_pattern):
                logger.debug(
                    f"[Cache] Skipping cache for '{context}' (matched pattern: {pattern})"
                )
                return True

            if filename != normalized_context and self._glob_match(
                filename, normalized_pattern
            ):
                logger.debug(
                    f"[Cache] Skipping cache for '{context}' (filename '{filename}' matched pattern: {pattern})"
                )
                return True

        return False

    def get(self, prompt: str, content: str, context: str = "") -> Any | None:
        """Lookup in global cache."""
        if not self._enabled or self._skip_read:
            return None

        if self._should_skip_cache(context):
            return None

        if self._global_cache:
            result = self._global_cache.get(prompt, content)
            if result is not None:
                self._hits += 1
                logger.debug("[Cache] Cache hit")
                return json.loads(result)

        self._misses += 1
        return None

    def set(self, prompt: str, content: str, result: Any, model: str = "") -> None:
        """Write to global cache."""
        if not self._enabled:
            return

        value = json.dumps(result, ensure_ascii=False)

        if self._global_cache:
            try:
                self._global_cache.set(prompt, content, value, model)
            except Exception as e:
                logger.warning(f"[Cache] Failed to write to cache: {e}")

    def clear(self) -> int:
        """Clear all cache entries.

        Returns:
            Number of entries deleted
        """
        if self._global_cache:
            return self._global_cache.clear()
        return 0

    def stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        total_requests = self._hits + self._misses
        hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0.0

        cache_stats = self._global_cache.stats() if self._global_cache else None

        return {
            "cache": cache_stats,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(hit_rate, 2),
        }


class ContentCache:
    """LRU cache with TTL for LLM responses based on content hash.

    Uses OrderedDict for O(1) LRU eviction instead of O(n) min() search.
    """

    def __init__(
        self,
        maxsize: int = DEFAULT_CACHE_MAXSIZE,
        ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS,
    ) -> None:
        """Initialize content cache.

        Args:
            maxsize: Maximum number of entries to cache
            ttl_seconds: Time-to-live in seconds
        """
        self._cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._maxsize = maxsize
        self._ttl = ttl_seconds

    def _compute_hash(self, prompt: str, content: str) -> str:
        """Compute hash key from prompt and content."""
        import hashlib

        combined = f"{prompt}|{content}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    def get(self, prompt: str, content: str) -> Any | None:
        """Get cached result if exists and not expired."""
        key = self._compute_hash(prompt, content)
        if key not in self._cache:
            return None

        result, timestamp = self._cache[key]
        if time.time() - timestamp > self._ttl:
            del self._cache[key]
            return None

        self._cache.move_to_end(key)
        return result

    def set(self, prompt: str, content: str, result: Any) -> None:
        """Cache a result."""
        key = self._compute_hash(prompt, content)

        if key in self._cache:
            self._cache[key] = (result, time.time())
            self._cache.move_to_end(key)
            return

        if len(self._cache) >= self._maxsize:
            self._cache.popitem(last=False)

        self._cache[key] = (result, time.time())

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()

    @property
    def size(self) -> int:
        """Number of cached entries."""
        return len(self._cache)
