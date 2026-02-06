"""URL fetch module for handling static and JS-rendered pages.

This module provides a unified interface for fetching web pages using different
strategies:
- static: Direct HTTP request via markitdown (default, fastest)
- playwright: Headless browser via Playwright Python (recommended for JS-rendered pages)
- jina: Jina Reader API (cloud-based, no local dependencies)
- auto: Auto-detect and fallback (tries static first, then playwright/jina)

Example usage:
    from markitai.fetch import fetch_url, FetchStrategy

    # Auto-detect strategy
    result = await fetch_url("https://example.com", FetchStrategy.AUTO, config.fetch)

    # Force Playwright rendering
    result = await fetch_url("https://x.com/...", FetchStrategy.PLAYWRIGHT, config.fetch)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import sqlite3
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from loguru import logger

from markitai.constants import (
    DEFAULT_JINA_BASE_URL,
    JS_REQUIRED_PATTERNS,
)

if TYPE_CHECKING:
    from markitai.config import FetchConfig, ScreenshotConfig


class FetchStrategy(Enum):
    """URL fetch strategy."""

    AUTO = "auto"
    STATIC = "static"
    PLAYWRIGHT = "playwright"  # Playwright Python (recommended)
    JINA = "jina"


# Reasons that indicate content is completely invalid and should not be used
# When these are detected, we should raise an error instead of using invalid content
CRITICAL_INVALID_REASONS = {
    "javascript_disabled",  # JS-rendered sites (Twitter/X, etc.)
    "javascript_required",  # Sites requiring JS
    "login_required",  # Sites requiring authentication
}


class FetchError(Exception):
    """Base exception for fetch errors."""

    pass


class JinaRateLimitError(FetchError):
    """Raised when Jina Reader API rate limit is exceeded."""

    def __init__(self) -> None:
        super().__init__(
            "Jina Reader rate limit exceeded (free tier: 20 RPM). "
            "Try again later or use --playwright for browser rendering."
        )


class JinaAPIError(FetchError):
    """Raised when Jina Reader API returns an error."""

    def __init__(self, status_code: int, message: str) -> None:
        self.status_code = status_code
        super().__init__(f"Jina Reader API error ({status_code}): {message}")


@dataclass
class FetchResult:
    """Result of a URL fetch operation.

    Supports multi-source content for URL fetching:
    - content: Primary markdown content (best available)
    - static_content: Content from static/jina fetch (pure text)
    - browser_content: Content from browser fetch (rendered page)
    - screenshot_path: Full-page screenshot (visual reference)

    For LLM processing, all three sources can be provided:
    1. static_content - Clean text, reliable but may miss JS content
    2. browser_content - Rendered content, includes JS but may have noise
    3. screenshot - Visual reference for layout/structure
    """

    content: str  # Primary markdown content (best available)
    strategy_used: str  # Actual strategy used (static/browser/jina)
    title: str | None = None  # Page title if available
    url: str = ""  # Original URL
    final_url: str | None = None  # Final URL after redirects
    metadata: dict = field(default_factory=dict)  # Additional metadata
    cache_hit: bool = False  # Whether result was served from cache
    screenshot_path: Path | None = None  # Path to captured screenshot (if any)
    # Multi-source content for enhanced LLM processing
    static_content: str | None = None  # Content from static fetch
    browser_content: str | None = None  # Content from browser fetch


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
                    last_modified TEXT
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_fetch_accessed ON fetch_cache(accessed_at)"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_fetch_url ON fetch_cache(url)")
            conn.commit()

            # Migration: Add etag and last_modified columns if they don't exist
            self._migrate_add_http_validators(conn)

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

    def _compute_hash(self, url: str) -> str:
        """Compute hash key from URL."""
        return hashlib.sha256(url.encode()).hexdigest()[:32]

    def _get_unlocked(self, url: str) -> FetchResult | None:
        """Get cached fetch result (no lock). Caller must hold a lock."""
        key = self._compute_hash(url)
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
            return FetchResult(
                content=row["content"],
                strategy_used=row["strategy_used"],
                title=row["title"],
                url=row["url"],
                final_url=row["final_url"],
                metadata=metadata,
                cache_hit=True,
            )

        return None

    def get(self, url: str) -> FetchResult | None:
        """Get cached fetch result if exists.

        Args:
            url: URL to look up

        Returns:
            Cached FetchResult or None if not found
        """
        with self._lock:
            return self._get_unlocked(url)

    async def aget(self, url: str) -> FetchResult | None:
        """Async version of get() using asyncio.Lock."""
        async with self._async_lock:
            return self._get_unlocked(url)

    def _set_unlocked(self, url: str, result: FetchResult) -> None:
        """Cache a fetch result (no lock). Caller must hold a lock."""
        key = self._compute_hash(url)
        now = int(time.time())
        metadata_json = json.dumps(result.metadata) if result.metadata else None
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

        # Insert or replace
        conn.execute(
            """
            INSERT OR REPLACE INTO fetch_cache
            (key, url, content, strategy_used, title, final_url, metadata, created_at, accessed_at, size_bytes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            ),
        )
        conn.commit()
        logger.debug(f"[FetchCache] Cached URL: {url} ({size_bytes} bytes)")

    def set(self, url: str, result: FetchResult) -> None:
        """Cache a fetch result.

        Args:
            url: URL that was fetched
            result: FetchResult to cache
        """
        with self._lock:
            self._set_unlocked(url, result)

    async def aset(self, url: str, result: FetchResult) -> None:
        """Async version of set() using asyncio.Lock."""
        async with self._async_lock:
            self._set_unlocked(url, result)

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
        self, url: str
    ) -> tuple[FetchResult | None, str | None, str | None]:
        """Get cached result with HTTP validators (no lock). Caller must hold a lock."""
        key = self._compute_hash(url)

        conn = self._get_connection()
        row = conn.execute("SELECT * FROM fetch_cache WHERE key = ?", (key,)).fetchone()

        if row:
            metadata = json.loads(row["metadata"]) if row["metadata"] else {}
            result = FetchResult(
                content=row["content"],
                strategy_used=row["strategy_used"],
                title=row["title"],
                url=row["url"],
                final_url=row["final_url"],
                metadata=metadata,
                cache_hit=True,
            )
            return result, row["etag"], row["last_modified"]

        return None, None, None

    def get_with_validators(
        self, url: str
    ) -> tuple[FetchResult | None, str | None, str | None]:
        """Get cached result with HTTP validators for conditional requests.

        Args:
            url: URL to look up

        Returns:
            Tuple of (cached_result, etag, last_modified)
            - cached_result: FetchResult if found, None otherwise
            - etag: ETag header from previous fetch (for If-None-Match)
            - last_modified: Last-Modified header from previous fetch (for If-Modified-Since)
        """
        with self._lock:
            return self._get_with_validators_unlocked(url)

    async def aget_with_validators(
        self, url: str
    ) -> tuple[FetchResult | None, str | None, str | None]:
        """Async version of get_with_validators() using asyncio.Lock."""
        async with self._async_lock:
            return self._get_with_validators_unlocked(url)

    def _set_with_validators_unlocked(
        self,
        url: str,
        result: FetchResult,
        etag: str | None = None,
        last_modified: str | None = None,
    ) -> None:
        """Cache a fetch result with HTTP validators (no lock). Caller must hold a lock."""
        key = self._compute_hash(url)
        now = int(time.time())
        metadata_json = json.dumps(result.metadata) if result.metadata else None
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

        # Insert or replace
        conn.execute(
            """
            INSERT OR REPLACE INTO fetch_cache
            (key, url, content, strategy_used, title, final_url, metadata, created_at, accessed_at, size_bytes, etag, last_modified)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
    ) -> None:
        """Cache a fetch result with HTTP validators.

        Args:
            url: URL that was fetched
            result: FetchResult to cache
            etag: ETag response header (for future If-None-Match)
            last_modified: Last-Modified response header (for future If-Modified-Since)
        """
        with self._lock:
            self._set_with_validators_unlocked(url, result, etag, last_modified)

    async def aset_with_validators(
        self,
        url: str,
        result: FetchResult,
        etag: str | None = None,
        last_modified: str | None = None,
    ) -> None:
        """Async version of set_with_validators() using asyncio.Lock."""
        async with self._async_lock:
            self._set_with_validators_unlocked(url, result, etag, last_modified)

    def _update_accessed_at_unlocked(self, url: str) -> None:
        """Update accessed_at timestamp (no lock). Caller must hold a lock."""
        key = self._compute_hash(url)
        now = int(time.time())

        conn = self._get_connection()
        conn.execute("UPDATE fetch_cache SET accessed_at = ? WHERE key = ?", (now, key))
        conn.commit()

    def update_accessed_at(self, url: str) -> None:
        """Update accessed_at timestamp for cache hit (304 response).

        Args:
            url: URL to update
        """
        with self._lock:
            self._update_accessed_at_unlocked(url)

    async def aupdate_accessed_at(self, url: str) -> None:
        """Async version of update_accessed_at() using asyncio.Lock."""
        async with self._async_lock:
            self._update_accessed_at_unlocked(url)


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
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
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
        """Save cache to disk."""
        try:
            with open(self._cache_path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2, ensure_ascii=False)
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
            return datetime.now() - last_hit > timedelta(days=self.EXPIRY_DAYS)
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
        now = datetime.now().isoformat()

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
            self._data["domains"][domain]["last_hit"] = datetime.now().isoformat()
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


# Global SPA domain cache instance (initialized lazily)
_spa_domain_cache: SPADomainCache | None = None


def get_spa_domain_cache() -> SPADomainCache:
    """Get or create the global SPA domain cache instance.

    Returns:
        SPADomainCache instance
    """
    global _spa_domain_cache
    if _spa_domain_cache is None:
        _spa_domain_cache = SPADomainCache()
    return _spa_domain_cache


# Global fetch cache instance (initialized lazily)
_fetch_cache: FetchCache | None = None


def get_fetch_cache(
    cache_dir: Path, max_size_bytes: int = 100 * 1024 * 1024
) -> FetchCache:
    """Get or create the global fetch cache instance.

    Args:
        cache_dir: Directory to store cache database
        max_size_bytes: Maximum cache size

    Returns:
        FetchCache instance
    """
    global _fetch_cache
    if _fetch_cache is None:
        db_path = cache_dir / "fetch_cache.db"
        _fetch_cache = FetchCache(db_path, max_size_bytes)
    return _fetch_cache


# Global MarkItDown instance (reused for static fetching)
# Note: MarkItDown's requests.Session is NOT thread-safe. However, since
# fetch_with_static runs in the asyncio event loop (not in a thread pool),
# only one md.convert() call executes at a time, avoiding thread safety issues.
# If fetch_with_static is ever moved to run_in_executor with threads, this
# should be changed to use threading.local() for thread-local instances.
_markitdown_instance: Any = None

# Global httpx.AsyncClient for Jina fetching (reused to avoid connection overhead)
_jina_client: Any = None

# Global Playwright renderer (reused to avoid browser cold starts)
_playwright_renderer: Any = None

# Cached proxy URL (None = not checked, "" = no proxy, "http://..." = proxy URL)
_detected_proxy: str | None = None

# Common proxy ports used by popular proxy software
_COMMON_PROXY_PORTS = [
    7897,  # Clash Verge default
    7890,  # Clash default
    7891,  # Clash mixed
    1082,  # Shadowrocket
    10808,  # V2Ray default
    10809,  # V2Ray HTTP
    1080,  # SOCKS5 common
    8080,  # HTTP proxy common
    8118,  # Privoxy
    9050,  # Tor
]


def _normalize_bypass_list(bypass: str) -> str:
    """Normalize proxy bypass list to Linux no_proxy compatible format.

    Converts Windows/macOS bypass patterns to standard no_proxy format:
    - *.domain.com -> .domain.com (suffix match)
    - *-prefix.domain.com -> .domain.com (extract base domain)
    - <local> -> removed (Windows-specific)
    - 127.* -> 127.0.0.0/8 (CIDR notation)
    - 10.* -> 10.0.0.0/8
    - 172.16-31.* -> 172.16.0.0/12
    - 192.168.* -> 192.168.0.0/16

    Args:
        bypass: Raw bypass list from system config

    Returns:
        Normalized comma-separated bypass list
    """
    if not bypass:
        return ""

    # IP wildcard to CIDR mapping
    ip_cidr_map = {
        "127.*": "127.0.0.0/8",
        "10.*": "10.0.0.0/8",
        "192.168.*": "192.168.0.0/16",
    }
    # Add 172.16-31.* mappings
    for i in range(16, 32):
        ip_cidr_map[f"172.{i}.*"] = "172.16.0.0/12"

    normalized = []
    seen: set[str] = set()

    for item in bypass.split(","):
        item = item.strip()
        if not item:
            continue

        # Skip Windows-specific markers
        if item == "<local>":
            continue

        result = None

        # Pattern: *.domain.com -> .domain.com
        if item.startswith("*."):
            result = item[1:]  # Remove leading *

        # Pattern: *-prefix.domain.com or *suffix.domain.com -> .domain.com
        # Extract the base domain after the first dot
        elif item.startswith("*") and "." in item:
            # Find first dot after the wildcard pattern
            first_dot = item.find(".")
            if first_dot > 0:
                result = item[first_dot:]  # Keep from first dot onwards

        # Handle IP wildcards with exact match
        elif item in ip_cidr_map:
            result = ip_cidr_map[item]

        # Handle partial IP wildcards like 100.64.*, 7.*
        elif item.endswith(".*"):
            # Convert to base IP prefix (best effort compatibility)
            base = item[:-2]  # Remove .*
            result = base

        else:
            result = item

        # Deduplicate
        if result and result not in seen:
            normalized.append(result)
            seen.add(result)

    return ",".join(normalized)


def _get_system_proxy() -> tuple[str, str]:
    """Get system proxy settings from OS configuration.

    Returns:
        Tuple of (proxy_url, bypass_list) where bypass_list is comma-separated hosts
        The bypass list is normalized to Linux no_proxy compatible format.
    """
    import platform
    import subprocess

    system = platform.system()

    if system == "Windows":
        try:
            import winreg  # type: ignore[import-not-found]  # Windows-only module

            with winreg.OpenKey(  # type: ignore[attr-defined]
                winreg.HKEY_CURRENT_USER,  # type: ignore[attr-defined]
                r"Software\Microsoft\Windows\CurrentVersion\Internet Settings",
            ) as key:
                proxy_enable, _ = winreg.QueryValueEx(key, "ProxyEnable")  # type: ignore[attr-defined]
                if proxy_enable:
                    proxy_server, _ = winreg.QueryValueEx(key, "ProxyServer")  # type: ignore[attr-defined]
                    # Handle format: "http=host:port;https=host:port" or "host:port"
                    if "=" in proxy_server:
                        # Parse protocol-specific proxies
                        for part in proxy_server.split(";"):
                            if part.startswith("https=") or part.startswith("http="):
                                proxy_addr = part.split("=", 1)[1]
                                if not proxy_addr.startswith("http"):
                                    proxy_addr = f"http://{proxy_addr}"
                                break
                        else:
                            proxy_addr = ""
                    else:
                        proxy_addr = (
                            f"http://{proxy_server}"
                            if not proxy_server.startswith("http")
                            else proxy_server
                        )

                    # Get bypass list
                    try:
                        bypass, _ = winreg.QueryValueEx(key, "ProxyOverride")  # type: ignore[attr-defined]
                        # Windows uses semicolon, convert to comma
                        bypass = bypass.replace(";", ",") if bypass else ""
                    except FileNotFoundError:
                        bypass = ""

                    if proxy_addr:
                        # Silent - system proxy detection is routine
                        return proxy_addr, bypass  # Return raw, normalize at usage
        except Exception:
            # Silent - registry read failure is not critical
            pass

    elif system == "Darwin":  # macOS
        try:
            # Get network service (usually Wi-Fi or Ethernet)
            result = subprocess.run(
                ["scutil", "--proxy"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                output = result.stdout
                # Parse scutil output
                https_enable = "HTTPSEnable : 1" in output
                http_enable = "HTTPEnable : 1" in output

                proxy_addr = ""
                if https_enable:
                    # Extract HTTPS proxy
                    host = ""
                    for line in output.split("\n"):
                        if "HTTPSProxy :" in line:
                            host = line.split(":")[1].strip()
                        elif "HTTPSPort :" in line:
                            port = line.split(":")[1].strip()
                            if host:
                                proxy_addr = f"http://{host}:{port}"
                            break
                elif http_enable:
                    # Extract HTTP proxy
                    host = ""
                    for line in output.split("\n"):
                        if "HTTPProxy :" in line:
                            host = line.split(":")[1].strip()
                        elif "HTTPPort :" in line:
                            port = line.split(":")[1].strip()
                            if host:
                                proxy_addr = f"http://{host}:{port}"
                            break

                # Get exceptions list
                bypass = ""
                if "ExceptionsList" in output:
                    # Parse exception list from scutil output
                    in_exceptions = False
                    exceptions = []
                    for line in output.split("\n"):
                        if "ExceptionsList" in line:
                            in_exceptions = True
                        elif in_exceptions:
                            if "}" in line:
                                break
                            # Extract host from "0 : localhost" format
                            if ":" in line:
                                host = line.split(":", 1)[1].strip()
                                if host:
                                    exceptions.append(host)
                    bypass = ",".join(exceptions)

                if proxy_addr:
                    # Silent - system proxy detection is routine
                    return proxy_addr, bypass  # Return raw, normalize at usage
        except Exception:
            # Silent - scutil failure is not critical
            pass

    return "", ""


# Cache for system proxy bypass list
_detected_proxy_bypass: str | None = None


def _detect_proxy(force_recheck: bool = False) -> str:
    """Detect proxy settings from environment, system config, or common local ports.

    Detection order:
    1. Environment variables: HTTPS_PROXY, HTTP_PROXY, ALL_PROXY
    2. System proxy settings (Windows registry / macOS scutil)
    3. Probe common proxy ports on localhost

    Args:
        force_recheck: Force re-detection even if cached

    Returns:
        Proxy URL string (e.g., "http://127.0.0.1:7890") or empty string if no proxy
    """
    global _detected_proxy, _detected_proxy_bypass

    if _detected_proxy is not None and not force_recheck:
        return _detected_proxy

    import os
    import socket

    # Check environment variables first (highest priority - user explicit config)
    for var in [
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "ALL_PROXY",
        "https_proxy",
        "http_proxy",
        "all_proxy",
    ]:
        proxy = os.environ.get(var, "").strip()
        if proxy:
            # Silent - proxy from env is routine, no need to log
            _detected_proxy = proxy
            # Also check NO_PROXY env var
            _detected_proxy_bypass = os.environ.get(
                "NO_PROXY", os.environ.get("no_proxy", "")
            )
            return proxy

    # Check system proxy settings (Windows/macOS)
    system_proxy, system_bypass = _get_system_proxy()
    if system_proxy:
        _detected_proxy = system_proxy
        _detected_proxy_bypass = system_bypass
        return system_proxy

    # Probe common proxy ports on localhost (fallback)
    for port in _COMMON_PROXY_PORTS:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.1)  # 100ms timeout
            result = sock.connect_ex(("127.0.0.1", port))
            sock.close()
            if result == 0:
                proxy_url = f"http://127.0.0.1:{port}"
                logger.info(f"[Proxy] Auto-detected local proxy at port {port}")
                _detected_proxy = proxy_url
                _detected_proxy_bypass = ""
                return proxy_url
        except Exception:
            pass

    # Silent - no proxy is common, no need to log
    _detected_proxy = ""
    _detected_proxy_bypass = ""
    return ""


def _get_proxy_bypass() -> str:
    """Get the proxy bypass list (NO_PROXY equivalent).

    Returns:
        Comma-separated list of hosts to bypass proxy
    """
    global _detected_proxy_bypass

    # Ensure proxy detection has run
    if _detected_proxy is None:
        _detect_proxy()

    return _detected_proxy_bypass or ""


def get_proxy_for_url(url: str) -> str:
    """Get proxy URL for a given target URL.

    Only returns proxy for URLs that likely need it (e.g., blocked sites).

    Args:
        url: Target URL to fetch

    Returns:
        Proxy URL or empty string
    """
    # Domains that typically need proxy in China
    proxy_domains = {
        "x.com",
        "twitter.com",
        "facebook.com",
        "instagram.com",
        "youtube.com",
        "google.com",
        "github.com",  # Sometimes slow without proxy
    }

    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        # Check if domain matches any proxy domain
        for pd in proxy_domains:
            if domain == pd or domain.endswith("." + pd):
                return _detect_proxy()
    except Exception:
        pass

    return ""


def _get_markitdown() -> Any:
    """Get or create the shared MarkItDown instance.

    Reusing a single instance avoids repeated initialization overhead.
    """
    global _markitdown_instance
    if _markitdown_instance is None:
        from markitdown import MarkItDown

        _markitdown_instance = MarkItDown()
    return _markitdown_instance


def _get_jina_client(timeout: int = 30, proxy: str = "") -> Any:
    """Get or create the shared httpx.AsyncClient for Jina fetching.

    Reusing a single client instance avoids repeated connection setup overhead.
    The client uses connection pooling for better performance.

    Args:
        timeout: Request timeout in seconds (used on first creation only)
        proxy: Proxy URL (used on first creation only)

    Returns:
        httpx.AsyncClient instance
    """
    global _jina_client
    if _jina_client is None:
        import httpx

        # Use detected proxy if not explicitly provided
        effective_proxy = proxy or _detect_proxy()
        client_kwargs: dict[str, Any] = {
            "timeout": timeout,
            "limits": httpx.Limits(max_connections=10, max_keepalive_connections=5),
        }
        if effective_proxy:
            client_kwargs["proxy"] = effective_proxy
            logger.debug(f"[Jina] Using proxy: {effective_proxy}")

        _jina_client = httpx.AsyncClient(**client_kwargs)
    return _jina_client


async def close_shared_clients() -> None:
    """Close shared client instances.

    Call this during cleanup to release resources.
    """
    global _jina_client, _fetch_cache, _playwright_renderer
    if _jina_client is not None:
        await _jina_client.aclose()
        _jina_client = None
    if _fetch_cache is not None:
        _fetch_cache.close()
        _fetch_cache = None
    if _playwright_renderer is not None:
        await _playwright_renderer.close()
        _playwright_renderer = None


async def _get_playwright_renderer(proxy: str | None = None) -> Any:
    """Get or create the shared PlaywrightRenderer.

    Args:
        proxy: Optional proxy URL

    Returns:
        PlaywrightRenderer instance
    """
    global _playwright_renderer
    if _playwright_renderer is None:
        from markitai.fetch_playwright import PlaywrightRenderer

        _playwright_renderer = PlaywrightRenderer(proxy=proxy)
    return _playwright_renderer


def detect_js_required(content: str) -> bool:
    """Detect if content indicates JavaScript rendering is required.

    Note: This function receives MARKDOWN content (converted by markitdown),
    not raw HTML. Detection strategies must work with Markdown text.

    Uses multiple detection strategies:
    1. Simple string matching for common JS-required messages
    2. Content patterns that survive Markdown conversion
    3. Content length and quality checks

    Args:
        content: Markdown content to check (from markitdown conversion)

    Returns:
        True if content suggests JavaScript is needed
    """
    if not content:
        return True  # Empty content likely means JS-rendered

    content_lower = content.lower()

    # 1. Simple string matching for JS-required messages
    # These text patterns survive Markdown conversion
    for pattern in JS_REQUIRED_PATTERNS:
        if pattern.lower() in content_lower:
            logger.debug(f"JS required: string pattern matched '{pattern}'")
            return True

    # 2. Check for SPA/bot-protection text patterns
    # These patterns are more specific to avoid false positives
    spa_text_patterns = [
        # JS requirement messages (already covered by JS_REQUIRED_PATTERNS, but regex variants)
        r"this (?:page|site|website) requires javascript",
        r"you need (?:to enable )?javascript",
        r"enable javascript to (?:view|continue|access)",
        # Cloudflare/bot protection (only when it's the main content)
        r"^(?:\s*#?\s*)?(?:just a moment|one moment)\.{0,3}\s*$",
        r"checking (?:if the site connection is secure|your browser)",
        r"verifying (?:you are human|your browser)",
        r"ray id:",  # Cloudflare error pages include Ray ID
        # Common SPA loading states (only if very short content)
    ]
    for pattern in spa_text_patterns:
        if re.search(pattern, content_lower, re.MULTILINE):
            logger.debug(f"JS required: SPA text pattern matched '{pattern}'")
            return True

    # 3. Check for very short content (likely a JS-only page)
    # Strip markdown formatting for accurate length check
    text_only = re.sub(r"[#*_\[\]()>`\-|]", "", content)
    text_only = re.sub(r"!\[.*?\]\(.*?\)", "", text_only)  # Remove image refs
    text_only = re.sub(r"\[.*?\]\(.*?\)", "", text_only)  # Remove links
    text_only = " ".join(text_only.split()).strip()

    if len(text_only) < 100:
        logger.debug(f"JS required: content too short ({len(text_only)} chars)")
        return True

    # 4. Check for repetitive/placeholder content (SPA stub pages)
    # Some SPAs return minimal placeholder text
    unique_words = set(text_only.lower().split())
    if len(text_only) < 500 and len(unique_words) < 20:
        logger.debug(
            f"JS required: low content diversity "
            f"({len(unique_words)} unique words in {len(text_only)} chars)"
        )
        return True

    return False


def should_use_browser_for_domain(url: str, fallback_patterns: list[str]) -> bool:
    """Check if URL domain matches fallback patterns that need browser rendering.

    Args:
        url: URL to check
        fallback_patterns: List of domain patterns (e.g., ["twitter.com", "x.com"])

    Returns:
        True if domain matches any pattern
    """
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()

        for pattern in fallback_patterns:
            pattern_lower = pattern.lower()
            # Match exact domain or subdomain
            if domain == pattern_lower or domain.endswith("." + pattern_lower):
                logger.debug(f"Domain {domain} matches fallback pattern {pattern}")
                return True
    except Exception:
        pass

    return False


def _url_to_session_id(url: str) -> str:
    """Generate a stable session ID from URL for potential session reuse.

    Using a hash-based session ID allows browser session caching
    when the same URL is processed multiple times.

    Args:
        url: URL to generate session ID for

    Returns:
        Stable session ID like "markitai-a1b2c3d4"
    """
    url_hash = hashlib.sha256(url.encode()).hexdigest()[:8]
    return f"markitai-{url_hash}"


def _url_to_screenshot_filename(url: str) -> str:
    """Generate a safe filename for URL screenshot.

    Examples:
        https://example.com/path → example.com_path.full.jpg
        https://x.com/user/status/123 → x.com_user_status_123.full.jpg

    Args:
        url: URL to convert

    Returns:
        Safe filename with .full.jpg extension
    """
    try:
        parsed = urlparse(url)
        # Start with domain
        parts = [parsed.netloc] if parsed.netloc else []
        # Add path parts
        if parsed.path and parsed.path != "/":
            path_parts = parsed.path.strip("/").split("/")
            parts.extend(path_parts)

        # If no parts, fall back to hash
        if not parts or not any(parts):
            url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
            return f"screenshot_{url_hash}.full.jpg"

        # Join with underscores
        name = "_".join(p for p in parts if p)

        # Sanitize for filesystem (remove/replace unsafe chars)
        # Windows-unsafe: < > : " / \ | ? *
        # Also remove other problematic chars
        unsafe_chars = r'<>:"/\\|?*\x00-\x1f'
        name = re.sub(f"[{unsafe_chars}]", "_", name)

        # Collapse multiple underscores
        name = re.sub(r"_+", "_", name)

        # Strip leading/trailing underscores
        name = name.strip("_")

        # Limit length (leave room for extension)
        max_length = 200
        if len(name) > max_length:
            name = name[:max_length]

        # Final check for empty name
        if not name:
            url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
            return f"screenshot_{url_hash}.full.jpg"

        return f"{name}.full.jpg"
    except Exception:
        # Fallback: hash the URL
        url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
        return f"screenshot_{url_hash}.full.jpg"


def _compress_screenshot(
    screenshot_path: Path,
    quality: int = 85,
    max_height: int = 10000,
) -> None:
    """Compress a screenshot to JPEG with quality and size limits.

    Args:
        screenshot_path: Path to screenshot file (will be overwritten)
        quality: JPEG quality (1-100)
        max_height: Maximum height in pixels (will resize if exceeded)
    """
    try:
        from PIL import Image

        with Image.open(screenshot_path) as img:
            # Convert to RGB if necessary (for JPEG)
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")

            # Resize if too tall
            width, height = img.size
            if height > max_height:
                ratio = max_height / height
                new_width = int(width * ratio)
                img = img.resize((new_width, max_height), Image.Resampling.LANCZOS)
                logger.debug(
                    f"Resized screenshot from {width}x{height} to {new_width}x{max_height}"
                )

            # Save with compression
            img.save(screenshot_path, "JPEG", quality=quality, optimize=True)
            logger.debug(
                f"Compressed screenshot to quality={quality}: {screenshot_path}"
            )
    except ImportError:
        logger.warning("Pillow not installed, skipping screenshot compression")
    except Exception as e:
        logger.warning(f"Failed to compress screenshot: {e}")


def _html_to_text(html: str) -> str:
    """Extract clean text from HTML content.

    Args:
        html: Raw HTML content

    Returns:
        Extracted text content formatted as markdown
    """
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")

        # Remove script and style elements
        for element in soup(["script", "style", "noscript", "nav", "footer", "header"]):
            element.decompose()

        # Extract text from main content areas
        lines = []

        # Try to find main content area
        main = soup.find("main") or soup.find("article") or soup.find("body")
        if not main:
            return ""

        for element in main.find_all(
            ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "blockquote", "pre", "code"]
        ):
            text = element.get_text(strip=True)
            if not text:
                continue

            tag = element.name
            if tag == "h1":
                lines.append(f"# {text}")
            elif tag == "h2":
                lines.append(f"## {text}")
            elif tag == "h3":
                lines.append(f"### {text}")
            elif tag == "h4":
                lines.append(f"#### {text}")
            elif tag == "h5":
                lines.append(f"##### {text}")
            elif tag == "h6":
                lines.append(f"###### {text}")
            elif tag == "p":
                lines.append(text)
            elif tag == "li":
                lines.append(f"- {text}")
            elif tag == "blockquote":
                lines.append(f"> {text}")
            elif tag == "pre" or tag == "code":
                lines.append(f"```\n{text}\n```")

            lines.append("")

        return "\n".join(lines).strip()

    except ImportError:
        logger.debug("BeautifulSoup not installed, using simple text extraction")
        # Fallback: simple regex-based extraction
        import re

        # Remove tags
        text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
        text = re.sub(r"<[^>]+>", " ", text)
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text
    except Exception as e:
        logger.debug(f"HTML to text extraction failed: {e}")
        return ""


async def fetch_with_static(url: str) -> FetchResult:
    """Fetch URL using markitdown (direct HTTP request).

    Args:
        url: URL to fetch

    Returns:
        FetchResult with markdown content

    Raises:
        FetchError: If fetch fails
    """
    logger.debug(f"Fetching URL with static strategy: {url}")

    try:
        md = _get_markitdown()
        result = md.convert(url)

        if not result.text_content:
            raise FetchError(f"No content extracted from URL: {url}")

        return FetchResult(
            content=result.text_content,
            strategy_used="static",
            title=result.title,
            url=url,
            metadata={"converter": "markitdown"},
        )
    except Exception as e:
        if "No content extracted" in str(e):
            raise
        raise FetchError(f"Failed to fetch URL: {e}")


@dataclass
class ConditionalFetchResult:
    """Result of a conditional fetch operation.

    Attributes:
        result: FetchResult if content was fetched (200 response), None if not modified (304)
        not_modified: True if server returned 304 Not Modified
        etag: ETag response header (for future conditional requests)
        last_modified: Last-Modified response header (for future conditional requests)
    """

    result: FetchResult | None
    not_modified: bool
    etag: str | None = None
    last_modified: str | None = None


async def fetch_with_static_conditional(
    url: str,
    cached_etag: str | None = None,
    cached_last_modified: str | None = None,
) -> ConditionalFetchResult:
    """Fetch URL with HTTP conditional request (single network roundtrip).

    Uses If-None-Match and If-Modified-Since headers for cache validation.
    If the server returns 304 Not Modified, the cached content should be used.

    Args:
        url: URL to fetch
        cached_etag: ETag from previous fetch (sent as If-None-Match)
        cached_last_modified: Last-Modified from previous fetch (sent as If-Modified-Since)

    Returns:
        ConditionalFetchResult with:
        - not_modified=True if 304 response (use cached content)
        - result with new content if 200 response
        - etag/last_modified for future conditional requests
    """
    import tempfile

    import httpx

    logger.debug(
        f"[ConditionalFetch] URL: {url}, etag={cached_etag is not None}, "
        f"last_modified={cached_last_modified is not None}"
    )

    # Build conditional request headers
    headers: dict[str, str] = {}
    if cached_etag:
        headers["If-None-Match"] = cached_etag
    if cached_last_modified:
        headers["If-Modified-Since"] = cached_last_modified

    try:
        # Detect proxy
        proxy_url = _detect_proxy()
        proxy_config = proxy_url if proxy_url else None

        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=30.0,
            proxy=proxy_config,
        ) as client:
            response = await client.get(url, headers=headers)

            # Extract response headers for future conditional requests
            response_etag = response.headers.get("ETag")
            response_last_modified = response.headers.get("Last-Modified")

            # 304 Not Modified - use cached content
            if response.status_code == 304:
                return ConditionalFetchResult(
                    result=None,
                    not_modified=True,
                    etag=response_etag or cached_etag,
                    last_modified=response_last_modified or cached_last_modified,
                )

            # Non-2xx response (except 304)
            if response.status_code >= 400:
                raise FetchError(f"HTTP {response.status_code} fetching URL: {url}")

            # 200 OK (or other 2xx) - process new content
            logger.debug(
                f"[ConditionalFetch] {response.status_code} response, "
                f"content-length={len(response.content)}"
            )

            # Determine file extension from Content-Type or URL
            content_type = response.headers.get("Content-Type", "")
            if "text/html" in content_type:
                suffix = ".html"
            elif "application/pdf" in content_type:
                suffix = ".pdf"
            else:
                # Fallback to URL extension
                from urllib.parse import urlparse

                path = urlparse(url).path
                suffix = Path(path).suffix or ".html"

            # Save response to temp file for markitdown processing
            with tempfile.NamedTemporaryFile(
                suffix=suffix, delete=False, mode="wb"
            ) as f:
                f.write(response.content)
                temp_path = Path(f.name)

            try:
                # Use markitdown to convert
                md = _get_markitdown()
                md_result = md.convert(str(temp_path))

                if not md_result.text_content:
                    raise FetchError(f"No content extracted from URL: {url}")

                fetch_result = FetchResult(
                    content=md_result.text_content,
                    strategy_used="static",
                    title=md_result.title,
                    url=url,
                    final_url=str(response.url),
                    metadata={"converter": "markitdown", "conditional": True},
                )

                return ConditionalFetchResult(
                    result=fetch_result,
                    not_modified=False,
                    etag=response_etag,
                    last_modified=response_last_modified,
                )
            finally:
                # Cleanup temp file
                temp_path.unlink(missing_ok=True)

    except httpx.HTTPError as e:
        raise FetchError(f"HTTP error fetching URL {url}: {e}")
    except Exception as e:
        if isinstance(e, FetchError):
            raise
        raise FetchError(f"Failed to fetch URL with conditional request: {e}")


async def fetch_with_jina(
    url: str,
    api_key: str | None = None,
    timeout: int = 30,
) -> FetchResult:
    """Fetch URL using Jina Reader API with JSON mode.

    Uses JSON mode for reliable structured data extraction (title, content).

    Args:
        url: URL to fetch
        api_key: Optional Jina API key (for higher rate limits)
        timeout: Request timeout in seconds

    Returns:
        FetchResult with markdown content and extracted title

    Raises:
        JinaRateLimitError: If rate limit exceeded
        JinaAPIError: If API returns error
        FetchError: If fetch fails
    """
    import json

    import httpx

    logger.debug(f"Fetching URL with Jina Reader (JSON mode): {url}")

    jina_url = f"{DEFAULT_JINA_BASE_URL}/{url}"
    headers = {
        "Accept": "application/json",  # Use JSON mode for structured response
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        client = _get_jina_client(timeout)
        response = await client.get(jina_url, headers=headers)

        if response.status_code == 429:
            raise JinaRateLimitError()
        elif response.status_code >= 400:
            raise JinaAPIError(response.status_code, response.text[:200])

        # Parse JSON response
        try:
            json_data = response.json()
        except json.JSONDecodeError as e:
            logger.warning(f"Jina API returned invalid JSON: {e}")
            raise FetchError(f"Jina Reader returned invalid JSON response: {e}")

        # Extract data from JSON structure
        # Expected format: {"code": 200, "status": 20000, "data": {...}}
        if not isinstance(json_data, dict):
            raise FetchError("Jina Reader returned unexpected response format")

        data = json_data.get("data")
        if not data or not isinstance(data, dict):
            # Check if the response indicates an error
            error_msg = json_data.get("message") or json_data.get("error")
            if error_msg:
                raise JinaAPIError(
                    json_data.get("code", 500),
                    str(error_msg)[:200],
                )
            raise FetchError("Jina Reader returned empty or invalid data structure")

        # Extract title and content from data
        title = data.get("title")
        content = data.get("content", "")

        if not content or not content.strip():
            raise FetchError(f"No content returned from Jina Reader: {url}")

        # Clean up title if present
        if title:
            title = title.strip()
            if not title:
                title = None

        return FetchResult(
            content=content,
            strategy_used="jina",
            title=title,
            url=url,
            metadata={
                "api": "jina-reader",
                "mode": "json",
                "source_url": data.get("url", url),
            },
        )

    except (JinaRateLimitError, JinaAPIError):
        raise
    except httpx.TimeoutException:
        raise FetchError(f"Jina Reader request timed out after {timeout}s: {url}")
    except FetchError:
        raise
    except Exception as e:
        raise FetchError(f"Jina Reader fetch failed: {e}")


async def fetch_url(
    url: str,
    strategy: FetchStrategy,
    config: FetchConfig,
    explicit_strategy: bool = False,
    cache: FetchCache | None = None,
    skip_read_cache: bool = False,
    *,
    screenshot: bool = False,
    screenshot_dir: Path | None = None,
    screenshot_config: ScreenshotConfig | None = None,
    renderer: Any | None = None,
) -> FetchResult:
    """Fetch URL content using the specified strategy.

    Args:
        url: URL to fetch
        strategy: Fetch strategy to use
        config: Fetch configuration
        explicit_strategy: If True, don't fallback on error (user explicitly chose strategy)
        cache: Optional FetchCache for caching results
        skip_read_cache: If True, skip reading from cache but still write results (--no-cache)
        screenshot: If True, capture full-page screenshot (requires browser strategy)
        screenshot_dir: Directory to save screenshot
        screenshot_config: Screenshot settings (viewport, quality, etc.)
        renderer: Optional shared PlaywrightRenderer

    Returns:
        FetchResult with content and metadata

    Raises:
        FetchError: If fetch fails and no fallback available
        JinaRateLimitError: If --jina used and rate limit exceeded
    """
    # Use provided renderer or get global one if needed
    _renderer = renderer
    if _renderer is None and (
        strategy == FetchStrategy.PLAYWRIGHT
        or (strategy == FetchStrategy.AUTO and screenshot)
        or screenshot
    ):
        # Only initialize global renderer if browser strategy is likely to be used
        proxy = _detect_proxy() if getattr(config, "auto_proxy", True) else None
        _renderer = await _get_playwright_renderer(proxy=proxy)

    # When screenshot is enabled, use multi-source fetching strategy
    # This captures both static content and browser-rendered content
    if screenshot:
        return await _fetch_multi_source(
            url,
            config,
            screenshot_dir=screenshot_dir,
            screenshot_config=screenshot_config,
            cache=cache,
            skip_read_cache=skip_read_cache,
            renderer=_renderer,
        )

    # For static strategy with cache, try HTTP conditional request for efficiency
    # This uses ETag/Last-Modified headers to avoid re-downloading unchanged content
    use_conditional_cache = (
        cache is not None
        and not skip_read_cache
        and strategy in (FetchStrategy.STATIC, FetchStrategy.AUTO)
    )

    if use_conditional_cache:
        # Type narrowing: cache is guaranteed non-None here due to condition above
        assert cache is not None
        # Get cached result with HTTP validators
        (
            cached_result,
            cached_etag,
            cached_last_modified,
        ) = await cache.aget_with_validators(url)

        # If we have validators, try conditional fetch (static strategy only)
        if cached_result is not None and (cached_etag or cached_last_modified):
            # Only use conditional fetch for static strategy
            # AUTO strategy might need browser fallback, so skip conditional optimization
            if strategy == FetchStrategy.STATIC or (
                strategy == FetchStrategy.AUTO
                and not should_use_browser_for_domain(url, config.fallback_patterns)
                and not get_spa_domain_cache().is_known_spa(url)
            ):
                try:
                    cond_result = await fetch_with_static_conditional(
                        url, cached_etag, cached_last_modified
                    )
                    if cond_result.not_modified:
                        # 304 Not Modified - use cached content
                        await cache.aupdate_accessed_at(url)
                        return cached_result
                    elif cond_result.result is not None:
                        # New content received - update cache with new validators
                        await cache.aset_with_validators(
                            url,
                            cond_result.result,
                            cond_result.etag,
                            cond_result.last_modified,
                        )
                        return cond_result.result
                except FetchError:
                    # Conditional fetch failed, fall through to normal flow
                    logger.debug(
                        f"[ConditionalFetch] Failed, falling back to normal fetch: {url}"
                    )

        # No validators but have cached result - use it directly
        elif cached_result is not None:
            return cached_result

    # Traditional cache check for non-conditional strategies
    elif cache is not None and not skip_read_cache:
        cached_result = await cache.aget(url)
        if cached_result is not None:
            return cached_result

    # Screenshot kwargs for browser fetching
    screenshot_kwargs: dict[str, Any] = {"renderer": _renderer}

    # Fetch the content
    result: FetchResult

    # Handle explicit strategy (no fallback)
    if explicit_strategy:
        if strategy == FetchStrategy.PLAYWRIGHT:
            from markitai.fetch_playwright import (
                fetch_with_playwright,
                is_playwright_available,
            )

            if not is_playwright_available():
                raise FetchError(
                    "playwright is not installed. "
                    "Install with: uv add playwright && uv run playwright install chromium "
                    "(Linux: also run 'uv run playwright install-deps chromium')"
                )

            pw_result = await fetch_with_playwright(
                url,
                timeout=config.playwright.timeout,
                wait_for=config.playwright.wait_for,
                extra_wait_ms=config.playwright.extra_wait_ms,
                proxy=_detect_proxy() if getattr(config, "auto_proxy", True) else None,
                screenshot_config=screenshot_config,
                output_dir=screenshot_dir,
                renderer=_renderer,
            )

            result = FetchResult(
                content=pw_result.content,
                strategy_used="playwright",
                title=pw_result.title,
                url=url,
                final_url=pw_result.final_url,
                metadata=pw_result.metadata,
                screenshot_path=pw_result.screenshot_path,
            )
        elif strategy == FetchStrategy.JINA:
            api_key = config.jina.get_resolved_api_key()
            result = await fetch_with_jina(url, api_key, config.jina.timeout)
        elif strategy == FetchStrategy.STATIC:
            # For fresh fetch, use conditional to capture validators
            cond_result = await fetch_with_static_conditional(url)
            if cond_result.result is None:
                raise FetchError(f"No content from conditional fetch: {url}")
            result = cond_result.result
            # Save with validators for future conditional requests
            if cache is not None:
                await cache.aset_with_validators(
                    url, result, cond_result.etag, cond_result.last_modified
                )
                return result
        else:
            # AUTO with explicit=True shouldn't happen, but handle it
            strategy = FetchStrategy.AUTO
            result = await _fetch_with_fallback(
                url, config, start_with_browser=False, **screenshot_kwargs
            )
    elif strategy == FetchStrategy.AUTO:
        # Check if domain needs browser rendering
        # Priority: 1. Configured fallback_patterns, 2. Learned SPA domains
        spa_cache = get_spa_domain_cache()
        use_browser_first = False

        if should_use_browser_for_domain(url, config.fallback_patterns):
            use_browser_first = True
        elif spa_cache.is_known_spa(url):
            spa_cache.record_hit(url)
            use_browser_first = True

        result = await _fetch_with_fallback(
            url, config, start_with_browser=use_browser_first, **screenshot_kwargs
        )
    elif strategy == FetchStrategy.STATIC:
        # For fresh fetch, use conditional to capture validators
        cond_result = await fetch_with_static_conditional(url)
        if cond_result.result is None:
            raise FetchError(f"No content from conditional fetch: {url}")
        result = cond_result.result
        # Save with validators for future conditional requests
        if cache is not None:
            await cache.aset_with_validators(
                url, result, cond_result.etag, cond_result.last_modified
            )
            return result
    elif strategy == FetchStrategy.PLAYWRIGHT:
        from markitai.fetch_playwright import (
            fetch_with_playwright,
            is_playwright_available,
        )

        if not is_playwright_available():
            raise FetchError(
                "playwright is not installed. "
                "Install with: uv add playwright && uv run playwright install chromium "
                "(Linux: also run 'uv run playwright install-deps chromium')"
            )

        pw_result = await fetch_with_playwright(
            url,
            timeout=config.playwright.timeout,
            wait_for=config.playwright.wait_for,
            extra_wait_ms=config.playwright.extra_wait_ms,
            proxy=_detect_proxy() if getattr(config, "auto_proxy", True) else None,
            screenshot_config=screenshot_config,
            output_dir=screenshot_dir,
            renderer=_renderer,
        )

        result = FetchResult(
            content=pw_result.content,
            strategy_used="playwright",
            title=pw_result.title,
            url=url,
            final_url=pw_result.final_url,
            metadata=pw_result.metadata,
            screenshot_path=pw_result.screenshot_path,
        )
    elif strategy == FetchStrategy.JINA:
        api_key = config.jina.get_resolved_api_key()
        result = await fetch_with_jina(url, api_key, config.jina.timeout)
    else:
        raise ValueError(f"Unknown fetch strategy: {strategy}")

    # Cache the result (for non-static strategies that don't use conditional caching)
    if cache is not None:
        await cache.aset(url, result)

    return result


def _is_invalid_content(content: str) -> tuple[bool, str]:
    """Check if fetched content is invalid (JS error page, login prompt, etc.).

    Args:
        content: Fetched content to check

    Returns:
        Tuple of (is_invalid, reason)
    """
    if not content or not content.strip():
        return True, "empty"

    # Check for common invalid content patterns
    invalid_patterns = [
        (r"JavaScript is (not available|disabled)", "javascript_disabled"),
        (r"Please enable JavaScript", "javascript_required"),
        (r"switch to a supported browser", "unsupported_browser"),
        (r"Something went wrong.*let's give it another shot", "error_page"),
        (r"Log in.*Sign up.*to continue", "login_required"),
        (r"You must be logged in", "login_required"),
    ]

    for pattern, reason in invalid_patterns:
        if re.search(pattern, content, re.IGNORECASE | re.DOTALL):
            return True, reason

    # Check content length (after removing markdown links and images)
    clean_content = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", content)  # Remove images
    clean_content = re.sub(r"\[[^\]]*\]\([^)]+\)", "", clean_content)  # Remove links
    clean_content = re.sub(
        r"[#\-*_>\[\]`|]", "", clean_content
    )  # Remove markdown syntax
    clean_content = " ".join(clean_content.split())  # Normalize whitespace

    if len(clean_content) < 100:
        return True, "too_short"

    return False, ""


async def _fetch_multi_source(
    url: str,
    config: FetchConfig,
    screenshot_dir: Path | None = None,
    screenshot_config: ScreenshotConfig | None = None,
    cache: FetchCache | None = None,
    skip_read_cache: bool = False,
    renderer: Any | None = None,
) -> FetchResult:
    """Fetch URL using static-first strategy with browser fallback.

    Strategy:
    1. Fetch both static and browser in parallel
    2. Validate content quality using _is_invalid_content()
    3. If static is valid → use static only (ignore browser content)
    4. Else if browser is valid → use browser only
    5. Else → use browser content with warning (both invalid)

    Screenshot is always included when available.

    Args:
        url: URL to fetch
        config: Fetch configuration
        screenshot_dir: Directory to save screenshot
        screenshot_config: Screenshot settings
        cache: Optional FetchCache for caching results
        skip_read_cache: If True, skip reading from cache
        renderer: Optional shared PlaywrightRenderer

    Returns:
        FetchResult with single-source content (no merging)
    """
    static_content: str | None = None
    browser_result: FetchResult | None = None
    browser_error: str | None = None  # Track browser fetch error
    browser_not_installed: bool = False  # Track if browser is not installed

    # Task 1: Try static fetch (non-blocking)
    async def fetch_static() -> str | None:
        try:
            result = await fetch_with_static(url)
            logger.debug(f"[URL] Static fetch success: {len(result.content)} chars")
            return result.content
        except Exception as e:
            logger.debug(f"[URL] Static fetch failed: {e}")
            return None

    # Task 2: Browser fetch with screenshot (Playwright)
    async def fetch_browser() -> tuple[FetchResult | None, str | None, bool]:
        """Returns (result, error_message, not_installed)"""
        try:
            from markitai.fetch_playwright import (
                fetch_with_playwright,
                is_playwright_available,
            )

            if not is_playwright_available():
                logger.debug("Playwright not available")
                return None, "playwright not installed", True

            logger.debug("Using Playwright for browser fetch")
            pw_result = await fetch_with_playwright(
                url,
                timeout=config.playwright.timeout,
                wait_for=config.playwright.wait_for,
                extra_wait_ms=config.playwright.extra_wait_ms,
                proxy=_detect_proxy() if getattr(config, "auto_proxy", True) else None,
                screenshot_config=screenshot_config,
                output_dir=screenshot_dir,
                renderer=renderer,
            )
            result = FetchResult(
                content=pw_result.content,
                strategy_used="playwright",
                title=pw_result.title,
                url=url,
                final_url=pw_result.final_url,
                metadata=pw_result.metadata,
                screenshot_path=pw_result.screenshot_path,
            )
            logger.debug(f"[URL] Playwright fetch success: {len(result.content)} chars")
            return result, None, False
        except Exception as e:
            error_msg = str(e)
            # Detect browser not installed/available errors
            not_installed = any(
                msg in error_msg.lower()
                for msg in [
                    "executable doesn't exist",
                    "browser is not installed",
                    "failed to launch",
                    "cannot open shared object",
                ]
            )
            logger.debug(f"[URL] Playwright fetch failed: {e}")
            return None, error_msg, not_installed

    # Execute both fetches in parallel
    static_content, browser_fetch_result = await asyncio.gather(
        fetch_static(), fetch_browser()
    )
    browser_result, browser_error, browser_not_installed = browser_fetch_result

    browser_content = browser_result.content if browser_result else None
    screenshot_path = browser_result.screenshot_path if browser_result else None

    # Validate content quality
    static_invalid, static_reason = (
        _is_invalid_content(static_content)
        if static_content
        else (True, "fetch_failed")
    )
    browser_invalid, browser_reason = (
        _is_invalid_content(browser_content)
        if browser_content
        else (True, "fetch_failed")
    )

    if static_invalid:
        logger.debug(f"[URL] Static content invalid: {static_reason}")
    if browser_invalid:
        logger.debug(f"[URL] Browser content invalid: {browser_reason}")

    # Determine which source to use (static-first strategy)
    primary_content = ""
    strategy_used = ""
    warning_message = ""
    final_static_content: str | None = None
    final_browser_content: str | None = None

    if not static_invalid:
        # Static is valid → use static only
        assert static_content is not None
        primary_content = static_content
        final_static_content = static_content
        strategy_used = "static"
    elif not browser_invalid:
        # Static invalid but browser is valid → use browser
        assert browser_content is not None
        primary_content = browser_content
        final_browser_content = browser_content
        strategy_used = "browser"
    elif browser_content:
        # Both invalid, but browser has content → use browser with warning
        primary_content = browser_content
        final_browser_content = browser_content
        strategy_used = "browser"
        warning_message = (
            f"Warning: Content may be incomplete. "
            f"Static: {static_reason}, Browser: {browser_reason}"
        )
        logger.warning(
            f"[URL] Both sources invalid, using browser content with warning: "
            f"static={static_reason}, browser={browser_reason}"
        )
    elif static_content:
        # Both invalid, no browser but has static
        # Check if this is a critical invalid reason (content is completely unusable)
        if static_reason in CRITICAL_INVALID_REASONS:
            # Try Jina as fallback before giving up
            # (screenshot won't be available, but content will be)
            if browser_error or browser_not_installed:
                try:
                    api_key = config.jina.get_resolved_api_key()
                    jina_result = await fetch_with_jina(
                        url, api_key, config.jina.timeout
                    )
                    # Return Jina result (no screenshot available)
                    jina_result.metadata["fallback"] = "jina"
                    jina_result.metadata["browser_error"] = browser_error
                    if cache is not None:
                        await cache.aset(url, jina_result)
                    return jina_result
                except JinaRateLimitError:
                    logger.warning("[URL] Jina fallback failed: rate limit exceeded")
                except Exception as e:
                    logger.warning(f"[URL] Jina fallback failed: {e}")

            # Jina also failed or wasn't tried, generate appropriate error
            browser_timed_out = (
                browser_error is not None and "timed out" in browser_error.lower()
            )

            if browser_timed_out:
                raise FetchError(
                    f"URL requires browser rendering: {url}. "
                    f"Browser fetch timed out. "
                    f"Try: 1) Increase timeout with --fetch-timeout 60000, "
                    f"2) Check network connectivity, "
                    f"3) Use Jina API with --jina-api-key"
                )
            elif browser_error:
                # Browser was attempted but failed with other error
                raise FetchError(
                    f"URL requires browser rendering: {url}. "
                    f"Browser fetch failed: {browser_error}. "
                    f"Install browser deps: sudo apt install libnspr4 libnss3 libatk1.0-0"
                )
            elif browser_not_installed:
                # Browser not installed
                raise FetchError(
                    f"URL requires browser rendering: {url}. "
                    f"Reason: {static_reason}. "
                    f"Please install Playwright: uv add playwright && uv run playwright install chromium "
                    f"(Linux: also run 'uv run playwright install-deps chromium')"
                )
            else:
                # Browser available but returned no content
                raise FetchError(
                    f"URL requires browser rendering: {url}. "
                    f"Browser returned no content. Check if the URL is accessible."
                )

        # Non-critical invalid, use static content with warning
        primary_content = static_content
        final_static_content = static_content
        strategy_used = "static"
        warning_message = f"Warning: Content may be incomplete. Reason: {static_reason}"
        logger.warning(
            f"[URL] Both sources invalid, using static content with warning: {static_reason}"
        )
    else:
        raise FetchError(f"All fetch strategies failed for URL: {url}")

    # Extract title from browser result if available
    title = browser_result.title if browser_result else None
    final_url = browser_result.final_url if browser_result else None

    # If no title from browser, try to extract from primary content
    if not title and primary_content:
        title_match = re.match(r"^#\s+(.+)$", primary_content, re.MULTILINE)
        if title_match:
            title = title_match.group(1)

    metadata: dict[str, Any] = {"single_source": True, "source": strategy_used}
    if warning_message:
        metadata["warning"] = warning_message

    assert primary_content is not None  # Guaranteed by above branches
    result = FetchResult(
        content=primary_content,
        strategy_used=strategy_used,
        title=title,
        url=url,
        final_url=final_url,
        metadata=metadata,
        screenshot_path=screenshot_path,
        static_content=final_static_content,
        browser_content=final_browser_content,
    )

    # Cache the result
    if cache is not None:
        await cache.aset(url, result)

    return result


async def _fetch_with_fallback(
    url: str,
    config: FetchConfig,
    start_with_browser: bool = False,
    renderer: Any | None = None,
    **screenshot_kwargs: Any,
) -> FetchResult:
    """Fetch URL with automatic fallback between strategies.

    Args:
        url: URL to fetch
        config: Fetch configuration
        start_with_browser: If True, try browser first (for known JS domains)
        renderer: Optional shared PlaywrightRenderer
        **screenshot_kwargs: Screenshot options (screenshot, screenshot_dir, screenshot_config)

    Returns:
        FetchResult from first successful strategy
    """
    errors = []

    if start_with_browser:
        # Try browser first for known JS domains
        # Priority: playwright > jina
        strategies = ["playwright", "jina", "static"]
    else:
        # Normal order: static -> playwright -> jina
        strategies = ["static", "playwright", "jina"]

    for strat in strategies:
        try:
            if strat == "static":
                result = await fetch_with_static(url)
                # Check if JS is required
                if detect_js_required(result.content):
                    # Learn this domain for future requests
                    spa_cache = get_spa_domain_cache()
                    spa_cache.record_spa_domain(url)
                    continue
                return result

            elif strat == "playwright":
                from markitai.fetch_playwright import (
                    fetch_with_playwright,
                    is_playwright_available,
                )

                if not is_playwright_available():
                    logger.debug("playwright not available, trying next strategy")
                    continue

                pw_result = await fetch_with_playwright(
                    url,
                    timeout=config.playwright.timeout,
                    wait_for=config.playwright.wait_for,
                    extra_wait_ms=config.playwright.extra_wait_ms,
                    proxy=_detect_proxy()
                    if getattr(config, "auto_proxy", True)
                    else None,
                    screenshot_config=screenshot_kwargs.get("screenshot_config"),
                    output_dir=screenshot_kwargs.get("screenshot_dir"),
                    renderer=renderer,
                )

                return FetchResult(
                    content=pw_result.content,
                    strategy_used="playwright",
                    title=pw_result.title,
                    url=url,
                    final_url=pw_result.final_url,
                    metadata=pw_result.metadata,
                    screenshot_path=pw_result.screenshot_path,
                )

            elif strat == "jina":
                api_key = config.jina.get_resolved_api_key()
                return await fetch_with_jina(url, api_key, config.jina.timeout)

        except JinaRateLimitError as e:
            errors.append(str(e))
            logger.warning(str(e))
            continue
        except FetchError as e:
            errors.append(f"{strat}: {e}")
            logger.debug(f"Strategy {strat} failed: {e}")
            continue
        except Exception as e:
            errors.append(f"{strat}: {e}")
            logger.debug(f"Strategy {strat} failed: {e}")
            continue

    # All strategies failed
    raise FetchError(
        f"All fetch strategies failed for {url}:\n"
        + "\n".join(f"  - {e}" for e in errors)
    )
