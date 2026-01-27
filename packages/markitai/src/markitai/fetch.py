"""URL fetch module for handling static and JS-rendered pages.

This module provides a unified interface for fetching web pages using different
strategies:
- static: Direct HTTP request via markitdown (default, fastest)
- browser: Headless browser via agent-browser (for JS-rendered pages)
- jina: Jina Reader API (cloud-based, no local dependencies)
- auto: Auto-detect and fallback (tries static first, then browser/jina)

Example usage:
    from markitai.fetch import fetch_url, FetchStrategy

    # Auto-detect strategy
    result = await fetch_url("https://example.com", FetchStrategy.AUTO, config.fetch)

    # Force browser rendering
    result = await fetch_url("https://x.com/...", FetchStrategy.BROWSER, config.fetch)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import shutil
import sqlite3
import time
import uuid
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
    BROWSER = "browser"
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


class AgentBrowserNotFoundError(FetchError):
    """Raised when agent-browser is not installed."""

    def __init__(self) -> None:
        super().__init__(
            "agent-browser is not installed. Install with: npm install -g agent-browser && agent-browser install"
        )


class JinaRateLimitError(FetchError):
    """Raised when Jina Reader API rate limit is exceeded."""

    def __init__(self) -> None:
        super().__init__(
            "Jina Reader rate limit exceeded (free tier: 20 RPM). "
            "Try again later or use --agent-browser for browser rendering."
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
    Thread safety is ensured by a lock protecting all database operations.
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
        self._lock = threading.Lock()  # Protect database operations
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
        """Initialize database schema."""
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
                    size_bytes INTEGER NOT NULL
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_fetch_accessed ON fetch_cache(accessed_at)"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_fetch_url ON fetch_cache(url)")
            conn.commit()

    def _compute_hash(self, url: str) -> str:
        """Compute hash key from URL."""
        return hashlib.sha256(url.encode()).hexdigest()[:32]

    def get(self, url: str) -> FetchResult | None:
        """Get cached fetch result if exists.

        Args:
            url: URL to look up

        Returns:
            Cached FetchResult or None if not found
        """
        key = self._compute_hash(url)
        now = int(time.time())

        with self._lock:
            conn = self._get_connection()
            row = conn.execute(
                "SELECT * FROM fetch_cache WHERE key = ?", (key,)
            ).fetchone()

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

    def set(self, url: str, result: FetchResult) -> None:
        """Cache a fetch result.

        Args:
            url: URL that was fetched
            result: FetchResult to cache
        """
        key = self._compute_hash(url)
        now = int(time.time())
        metadata_json = json.dumps(result.metadata) if result.metadata else None
        size_bytes = len(result.content.encode("utf-8"))

        with self._lock:
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
                        logger.debug(
                            f"[Proxy] Found Windows system proxy: {proxy_addr}"
                        )
                        return proxy_addr, bypass  # Return raw, normalize at usage
        except Exception as e:
            logger.debug(f"[Proxy] Failed to read Windows registry: {e}")

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
                    logger.debug(f"[Proxy] Found macOS system proxy: {proxy_addr}")
                    return proxy_addr, bypass  # Return raw, normalize at usage
        except Exception as e:
            logger.debug(f"[Proxy] Failed to read macOS proxy settings: {e}")

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
            logger.debug(f"[Proxy] Found proxy from {var}: {proxy}")
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

    logger.debug("[Proxy] No proxy detected")
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
    global _jina_client, _fetch_cache
    if _jina_client is not None:
        await _jina_client.aclose()
        _jina_client = None
    if _fetch_cache is not None:
        _fetch_cache.close()
        _fetch_cache = None


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


def is_agent_browser_available(command: str = "agent-browser") -> bool:
    """Check if agent-browser CLI is installed and available.

    Args:
        command: Command name or path to check

    Returns:
        True if agent-browser is available
    """
    return shutil.which(command) is not None


# Cache for agent-browser readiness check
_agent_browser_ready_cache: dict[str, tuple[bool, str]] = {}

# Cache for resolved agent-browser executable path (Windows)
_agent_browser_exe_cache: dict[str, str | None] = {}


def _find_windows_agent_browser_exe(cmd_path: str) -> str | None:
    """Find the Windows native executable for agent-browser.

    On Windows, npm/pnpm-installed agent-browser CMD files depend on /bin/sh,
    which doesn't exist in native Windows. However, the package includes
    a native Windows executable (agent-browser-win32-x64.exe) that works directly.

    Args:
        cmd_path: Path to the agent-browser CMD file

    Returns:
        Path to the native Windows exe, or None if not found
    """
    if cmd_path in _agent_browser_exe_cache:
        return _agent_browser_exe_cache[cmd_path]

    try:
        # CMD file is typically at: npm_root/agent-browser.CMD
        # Native exe is at: npm_root/node_modules/agent-browser/bin/agent-browser-win32-x64.exe
        cmd_dir = Path(cmd_path).parent
        exe_path = (
            cmd_dir
            / "node_modules"
            / "agent-browser"
            / "bin"
            / "agent-browser-win32-x64.exe"
        )

        if exe_path.exists():
            result = str(exe_path)
            _agent_browser_exe_cache[cmd_path] = result
            logger.debug(f"Using Windows native agent-browser exe: {result}")
            return result
    except Exception as e:
        logger.debug(f"Failed to find Windows agent-browser exe: {e}")

    _agent_browser_exe_cache[cmd_path] = None
    return None


def verify_agent_browser_ready(
    command: str = "agent-browser", use_cache: bool = True
) -> tuple[bool, str]:
    """Verify that agent-browser is fully ready (command exists + browser installed).

    This performs a more thorough check than is_agent_browser_available() by
    actually running agent-browser to verify it works.

    Args:
        command: Command name or path to check
        use_cache: Whether to use cached result (default True)

    Returns:
        Tuple of (is_ready, message)
        - (True, "agent-browser is ready") if fully functional
        - (False, "error message") if not ready
    """
    import subprocess
    import sys

    # Check cache first
    if use_cache and command in _agent_browser_ready_cache:
        return _agent_browser_ready_cache[command]

    # Step 1: Check if command exists
    cmd_path = shutil.which(command)
    if not cmd_path:
        result = (
            False,
            f"'{command}' command not found. Install with: npm install -g agent-browser",
        )
        _agent_browser_ready_cache[command] = result
        return result

    # Step 2: Windows compatibility check
    # On Windows, CMD files may not work reliably in Python subprocess.
    # Always prefer the native Windows executable when available.
    effective_command = command
    use_shell = False

    if sys.platform == "win32" and cmd_path.lower().endswith((".cmd", ".cmd")):
        # Always try to find native exe first on Windows
        native_exe = _find_windows_agent_browser_exe(cmd_path)
        if native_exe:
            effective_command = native_exe
        else:
            # Check if CMD file requires /bin/sh (pnpm-style)
            try:
                with open(cmd_path, encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                    if "/bin/sh" in content and not Path("/bin/sh").exists():
                        result = (
                            False,
                            "agent-browser was installed via pnpm which requires Git Bash. "
                            "Options: 1) Run markitai from Git Bash terminal, "
                            "2) Reinstall with npm: npm install -g agent-browser && agent-browser install",
                        )
                        _agent_browser_ready_cache[command] = result
                        return result
            except Exception:
                pass  # Unable to read CMD file, continue with regular checks

    # Step 3: Check if agent-browser responds to --help
    # Windows: Hide console window completely
    run_kwargs: dict[str, Any] = {
        "capture_output": True,
        "text": True,
        "timeout": 10,
        "shell": use_shell,
    }
    if sys.platform == "win32":
        run_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        si.wShowWindow = 0  # SW_HIDE
        run_kwargs["startupinfo"] = si

    try:
        proc = subprocess.run(
            [effective_command, "--help"],
            **run_kwargs,
        )
        if proc.returncode != 0:
            result = (False, f"'{command}' command failed: {proc.stderr.strip()}")
            _agent_browser_ready_cache[command] = result
            return result
    except subprocess.TimeoutExpired:
        result = (False, f"'{command}' command timed out")
        _agent_browser_ready_cache[command] = result
        return result
    except FileNotFoundError as e:
        # Windows-specific error message
        if sys.platform == "win32":
            result = (
                False,
                "Cannot execute agent-browser on Windows. "
                "If installed via pnpm, try running from Git Bash terminal "
                "or reinstall with npm: npm install -g agent-browser",
            )
        else:
            result = (False, f"'{command}' command error: {e}")
        _agent_browser_ready_cache[command] = result
        return result
    except Exception as e:
        result = (False, f"'{command}' command error: {e}")
        _agent_browser_ready_cache[command] = result
        return result

    # Step 4: Quick version check instead of browser launch
    # Launching browser is slow and causes popup windows on Windows
    # The --help check above already validates the command works
    run_kwargs["timeout"] = 5
    try:
        proc = subprocess.run(
            [effective_command, "--version"],
            **run_kwargs,
        )
        if proc.returncode == 0:
            version = proc.stdout.strip() if proc.stdout else "unknown"
            logger.debug(f"agent-browser version: {version}")
    except Exception:
        pass  # Version check is optional

    result = (True, "agent-browser is ready")
    _agent_browser_ready_cache[command] = result
    # Also cache the effective command path for later use
    if effective_command != command:
        _agent_browser_exe_cache[command] = effective_command
    return result


def clear_agent_browser_cache() -> None:
    """Clear the agent-browser readiness cache."""
    _agent_browser_ready_cache.clear()


def _get_effective_agent_browser_args(args: list[str]) -> list[str]:
    """Get effective command args with Windows native exe resolution.

    On Windows, automatically resolves to native exe if available.
    Caches the resolved path to avoid repeated lookups.

    Args:
        args: Command arguments (e.g., ["agent-browser", "open", url])

    Returns:
        Effective args with resolved executable path
    """
    import sys

    effective_args = list(args)
    if (
        sys.platform == "win32"
        and args
        and args[0] in ("agent-browser", "agent-browser.CMD")
    ):
        # Check cache for native exe
        cached_exe = _agent_browser_exe_cache.get(args[0])
        if cached_exe:
            effective_args[0] = cached_exe
        else:
            # Try to find native exe
            cmd_path = shutil.which(args[0])
            if cmd_path:
                native_exe = _find_windows_agent_browser_exe(cmd_path)
                if native_exe:
                    effective_args[0] = native_exe
                    _agent_browser_exe_cache[args[0]] = native_exe
    return effective_args


async def _run_agent_browser_command(
    args: list[str], timeout_seconds: float, proxy: str = ""
) -> tuple[bytes, bytes, int]:
    """Run an agent-browser command with cross-platform compatibility.

    On Windows, automatically uses native exe if available.
    On other platforms, uses direct exec.

    Automatically detects and applies proxy settings for Playwright.

    Note: On Windows, uses temp files instead of PIPE to avoid deadlock
    with agent-browser's ANSI colored output.

    Args:
        args: Command arguments (e.g., ["agent-browser", "open", url])
        timeout_seconds: Timeout in seconds
        proxy: Proxy URL (auto-detected if not provided)

    Returns:
        Tuple of (stdout, stderr, returncode)

    Raises:
        asyncio.TimeoutError: If command times out
    """
    import os
    import subprocess
    import sys
    import tempfile

    effective_args = _get_effective_agent_browser_args(args)

    # Build kwargs for subprocess
    kwargs: dict[str, Any] = {}

    # Windows: Use temp files instead of PIPE to avoid deadlock
    # agent-browser's ANSI colored output causes pipe buffer issues on Windows
    # that lead to communicate() hanging indefinitely
    stdout_file = None
    stderr_file = None
    use_temp_files = sys.platform == "win32"

    if use_temp_files:
        stdout_file = tempfile.NamedTemporaryFile(  # noqa: SIM115
            mode="w+b", delete=False, suffix=".stdout"
        )
        stderr_file = tempfile.NamedTemporaryFile(  # noqa: SIM115
            mode="w+b", delete=False, suffix=".stderr"
        )
        kwargs["stdout"] = stdout_file
        kwargs["stderr"] = stderr_file
    else:
        kwargs["stdout"] = asyncio.subprocess.PIPE
        kwargs["stderr"] = asyncio.subprocess.PIPE

    # Windows: Hide console window completely
    # CREATE_NO_WINDOW: Don't create console window for the process
    # DETACHED_PROCESS: Detach from parent's console session
    # STARTUPINFO with SW_HIDE: Additional window hiding
    if sys.platform == "win32":
        kwargs["creationflags"] = (
            subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS
        )
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        si.wShowWindow = 0  # SW_HIDE
        kwargs["startupinfo"] = si

    # Set proxy environment for Playwright/agent-browser
    effective_proxy = proxy or _detect_proxy()
    if effective_proxy:
        # Create a copy of current environment with proxy settings
        env = os.environ.copy()
        # agent-browser uses AGENT_BROWSER_PROXY (preferred) or standard HTTP_PROXY
        env["AGENT_BROWSER_PROXY"] = effective_proxy
        env["HTTPS_PROXY"] = effective_proxy
        env["HTTP_PROXY"] = effective_proxy

        # Set proxy bypass list
        bypass = _get_proxy_bypass()
        if bypass:
            # AGENT_BROWSER_PROXY_BYPASS: Playwright/Chromium supports wildcards
            env["AGENT_BROWSER_PROXY_BYPASS"] = bypass
            # NO_PROXY: Normalize for Linux compatibility (Git Bash, WSL, etc.)
            env["NO_PROXY"] = _normalize_bypass_list(bypass)
            logger.debug(f"[agent-browser] Proxy bypass: {bypass}")

        kwargs["env"] = env
        logger.debug(f"[agent-browser] Using proxy: {effective_proxy}")

    try:
        proc = await asyncio.create_subprocess_exec(*effective_args, **kwargs)

        if use_temp_files:
            # Close file handles so subprocess can write
            assert stdout_file is not None and stderr_file is not None
            stdout_path = stdout_file.name
            stderr_path = stderr_file.name
            stdout_file.close()
            stderr_file.close()

            # Wait for process with timeout
            try:
                await asyncio.wait_for(proc.wait(), timeout=timeout_seconds)
            except TimeoutError:
                proc.kill()
                await proc.wait()
                raise

            # Read output from temp files
            with open(stdout_path, "rb") as f:
                stdout = f.read()
            with open(stderr_path, "rb") as f:
                stderr = f.read()

            return stdout, stderr, proc.returncode or 0
        else:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout_seconds
            )
            return stdout or b"", stderr or b"", proc.returncode or 0

    finally:
        # Cleanup temp files
        if use_temp_files:
            try:
                if stdout_file:
                    os.unlink(stdout_file.name)
            except OSError:
                pass
            try:
                if stderr_file:
                    os.unlink(stderr_file.name)
            except OSError:
                pass


async def _run_agent_browser_batch(
    commands: list[tuple[list[str], float]],
    proxy: str = "",
) -> list[tuple[bytes, bytes, int]]:
    """Run multiple agent-browser commands sequentially but efficiently.

    This reduces per-command overhead by reusing the resolved executable path
    and minimizing repeated path lookups.

    Args:
        commands: List of (args, timeout_seconds) tuples
        proxy: Proxy URL (auto-detected if not provided)

    Returns:
        List of (stdout, stderr, returncode) tuples in same order
    """
    # Detect proxy once for all commands
    effective_proxy = proxy or _detect_proxy()

    results: list[tuple[bytes, bytes, int]] = []
    for args, timeout_seconds in commands:
        try:
            result = await _run_agent_browser_command(
                args, timeout_seconds, proxy=effective_proxy
            )
            results.append(result)
        except TimeoutError:
            results.append((b"", b"Timeout", -1))
        except Exception as e:
            results.append((b"", str(e).encode(), -1))
    return results


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


async def fetch_with_browser(
    url: str,
    command: str = "agent-browser",
    timeout: int = 30000,
    wait_for: str = "domcontentloaded",
    extra_wait_ms: int = 2000,
    session: str | None = None,
    *,
    screenshot: bool = False,
    screenshot_dir: Path | None = None,
    screenshot_config: ScreenshotConfig | None = None,
) -> FetchResult:
    """Fetch URL using agent-browser (headless browser).

    Args:
        url: URL to fetch
        command: agent-browser command name or path
        timeout: Page load timeout in milliseconds
        wait_for: Wait condition (load/domcontentloaded/networkidle)
        extra_wait_ms: Extra wait time after load state (for JS rendering)
        session: Optional session name for isolated browser
        screenshot: If True, capture full-page screenshot
        screenshot_dir: Directory to save screenshot (required if screenshot=True)
        screenshot_config: Screenshot settings (viewport, quality, etc.)

    Returns:
        FetchResult with rendered page content and optional screenshot path

    Raises:
        AgentBrowserNotFoundError: If agent-browser is not installed
        FetchError: If fetch fails
    """
    if not is_agent_browser_available(command):
        raise AgentBrowserNotFoundError()

    logger.debug(f"Fetching URL with browser strategy: {url}")

    # Generate unique session ID to avoid conflicts with concurrent browser fetches
    # Each fetch_with_browser call gets its own isolated browser session
    effective_session = (
        session if session else f"markitai-fetch-{uuid.uuid4().hex[:12]}"
    )

    try:
        # Build command args
        base_args = [command, "--session", effective_session]

        # Detect proxy - must pass via CLI args, not just env vars
        # agent-browser daemon ignores env vars once running
        effective_proxy = _detect_proxy()
        proxy_bypass = _get_proxy_bypass()

        # Step 1: Open URL and wait for page load
        # --proxy and --proxy-bypass must be in 'open' command for daemon to use them
        open_args = [*base_args]
        if effective_proxy:
            open_args.extend(["--proxy", effective_proxy])
            if proxy_bypass:
                open_args.extend(["--proxy-bypass", proxy_bypass])
        open_args.extend(["open", url])
        logger.debug(f"Running: {' '.join(open_args)}")

        stdout, stderr, returncode = await _run_agent_browser_command(
            open_args, timeout / 1000 + 10
        )

        if returncode != 0:
            error_msg = stderr.decode() if stderr else "Unknown error"
            raise FetchError(f"agent-browser open failed: {error_msg}")

        # Step 2: Wait for load state
        wait_args = [*base_args, "wait", "--load", wait_for]
        logger.debug(f"Running: {' '.join(wait_args)}")

        await _run_agent_browser_command(wait_args, timeout / 1000 + 10)

        # Step 2.5: Extra wait for JS rendering (especially for SPAs)
        if extra_wait_ms > 0:
            extra_wait_args = [*base_args, "wait", str(extra_wait_ms)]
            logger.debug(f"Running: {' '.join(extra_wait_args)}")
            await _run_agent_browser_command(extra_wait_args, extra_wait_ms / 1000 + 5)

        # Step 3: Get page content via snapshot (accessibility tree with text)
        # Using snapshot -c (compact) to get clean text structure
        snapshot_args = [*base_args, "snapshot", "-c", "--json"]
        logger.debug(f"Running: {' '.join(snapshot_args)}")

        stdout, stderr, returncode = await _run_agent_browser_command(
            snapshot_args, timeout / 1000 + 10
        )

        if returncode != 0:
            error_msg = stderr.decode() if stderr else "Unknown error"
            raise FetchError(f"agent-browser snapshot failed: {error_msg}")

        # Parse snapshot JSON
        try:
            snapshot_data = json.loads(stdout.decode())
            if snapshot_data.get("success"):
                snapshot_text = snapshot_data.get("data", {}).get("snapshot", "")
            else:
                snapshot_text = stdout.decode()
        except json.JSONDecodeError:
            snapshot_text = stdout.decode()

        # Step 4, 5 & 6: Get page title, final URL and HTML body in parallel
        async def get_title() -> str | None:
            title_args = [*base_args, "get", "title"]
            stdout, _, returncode = await _run_agent_browser_command(title_args, 10)
            if returncode == 0 and stdout:
                return stdout.decode().strip()
            return None

        async def get_final_url() -> str | None:
            url_args = [*base_args, "get", "url"]
            stdout, _, returncode = await _run_agent_browser_command(url_args, 10)
            if returncode == 0 and stdout:
                return stdout.decode().strip()
            return None

        async def get_html_body() -> str | None:
            """Get HTML body content for text extraction."""
            html_args = [*base_args, "get", "html", "body"]
            stdout, _, returncode = await _run_agent_browser_command(html_args, 15)
            if returncode == 0 and stdout:
                return stdout.decode()
            return None

        # Execute title, URL and HTML fetching in parallel
        title, final_url, html_body = await asyncio.gather(
            get_title(), get_final_url(), get_html_body()
        )

        # Convert snapshot to markdown format
        markdown_content = _snapshot_to_markdown(snapshot_text, title, url)

        # Also extract text from HTML as fallback/supplement
        html_text_content: str | None = None
        if html_body:
            html_text_content = _html_to_text(html_body)

        # Use HTML text if snapshot conversion failed or is too short
        if not markdown_content.strip() or len(markdown_content.strip()) < 100:
            if html_text_content and len(html_text_content.strip()) > len(
                markdown_content.strip()
            ):
                logger.debug("Using HTML text extraction as primary content")
                if title:
                    markdown_content = f"# {title}\n\n{html_text_content}"
                else:
                    markdown_content = html_text_content

        if not markdown_content.strip():
            raise FetchError(f"No content extracted from URL via browser: {url}")

        # Step 6: Capture full-page screenshot if requested
        screenshot_path: Path | None = None
        if screenshot and screenshot_dir:
            try:
                screenshot_dir.mkdir(parents=True, exist_ok=True)
                safe_filename = _url_to_screenshot_filename(url)
                screenshot_path = screenshot_dir / safe_filename

                # Check if screenshot already exists (simple cache)
                if not screenshot_path.exists():
                    # Set viewport if configured
                    if screenshot_config:
                        viewport_args = [
                            *base_args,
                            "set",
                            "viewport",
                            str(screenshot_config.viewport_width),
                            str(screenshot_config.viewport_height),
                        ]
                        logger.debug(f"Running: {' '.join(viewport_args)}")
                        await _run_agent_browser_command(viewport_args, 10)

                    # Capture full-page screenshot
                    screenshot_args = [
                        *base_args,
                        "screenshot",
                        "--full",
                        str(screenshot_path),
                    ]
                    logger.debug(f"Running: {' '.join(screenshot_args)}")
                    stdout, stderr, returncode = await _run_agent_browser_command(
                        screenshot_args, 60
                    )

                    if returncode != 0:
                        error_msg = stderr.decode() if stderr else "Unknown error"
                        logger.warning(f"Screenshot capture failed: {error_msg}")
                        screenshot_path = None
                    elif screenshot_path.exists():
                        # Compress screenshot
                        quality = screenshot_config.quality if screenshot_config else 85
                        max_height = (
                            screenshot_config.max_height if screenshot_config else 10000
                        )
                        _compress_screenshot(screenshot_path, quality, max_height)
                        logger.debug(f"Screenshot saved: {screenshot_path}")
                else:
                    logger.debug(f"Screenshot exists, skipping: {screenshot_path}")
            except Exception as e:
                # Screenshot failure should not block the main fetch
                logger.warning(f"Screenshot failed for {url}: {e}")
                screenshot_path = None

        return FetchResult(
            content=markdown_content,
            strategy_used="browser",
            title=title,
            url=url,
            final_url=final_url,
            metadata={"renderer": "agent-browser", "wait_for": wait_for},
            screenshot_path=screenshot_path,
        )

    except TimeoutError:
        raise FetchError(f"Browser fetch timed out after {timeout}ms: {url}")
    except AgentBrowserNotFoundError:
        raise
    except FetchError:
        raise
    except Exception as e:
        raise FetchError(f"Browser fetch failed: {e}")
    finally:
        # Clean up the browser session to avoid resource leaks
        # Only close auto-generated sessions (not user-specified ones)
        if not session:
            try:
                close_args = [command, "--session", effective_session, "close"]
                await _run_agent_browser_command(close_args, 5)
                logger.debug(f"Closed browser session: {effective_session}")
            except Exception as e:
                logger.debug(
                    f"Failed to close browser session {effective_session}: {e}"
                )


def _snapshot_to_markdown(snapshot: str, title: str | None, url: str) -> str:
    """Convert agent-browser snapshot to markdown format.

    The snapshot is an accessibility tree with various formats:
    - heading "Title" [ref=e1] [level=1]
    - paragraph: Text content here
    - link "Link text" [ref=e2]:
        - /url: /path
    - text: Some text

    Args:
        snapshot: Accessibility tree snapshot
        title: Page title
        url: Original URL

    Returns:
        Markdown formatted content
    """
    lines = []

    # Add title as H1 if available
    if title:
        lines.append(f"# {title}")
        lines.append("")

    # Track current link for multi-line link handling
    current_link_text: str | None = None
    current_link_url: str | None = None

    # Parse snapshot and convert to markdown
    for line in snapshot.split("\n"):
        stripped = line.lstrip()

        if not stripped:
            continue

        # Skip structure markers
        if stripped.startswith("- document:") or stripped.startswith("- navigation:"):
            continue
        if stripped.startswith("- main:") or stripped.startswith("- article:"):
            continue
        if stripped.startswith("- contentinfo:") or stripped.startswith("- list:"):
            continue
        if stripped.startswith("- listitem:"):
            continue

        # Remove leading "- " if present
        if stripped.startswith("- "):
            stripped = stripped[2:]

        # Handle URL lines (part of link)
        if stripped.startswith("/url:"):
            current_link_url = stripped[5:].strip()
            if current_link_text:
                lines.append(f"[{current_link_text}]({current_link_url})")
                lines.append("")
                current_link_text = None
                current_link_url = None
            continue

        # Pattern 1: role "content" [attrs] (with or without trailing colon)
        # e.g., heading "Title" [ref=e1] [level=1]
        # e.g., link "Text" [ref=e2]:
        match = re.match(
            r'(\w+)\s+"([^"]*)"(?:\s*\[([^\]]*(?:\]\s*\[[^\]]*)*)\])?:?$', stripped
        )
        if match:
            role, content, attrs_str = match.groups()
            attrs_dict = {}
            if attrs_str:
                # Parse multiple [key=value] attributes
                for attr_match in re.finditer(r"\[?([^=\]]+)=([^\]]+)\]?", attrs_str):
                    k, v = attr_match.groups()
                    attrs_dict[k.strip()] = v.strip()

            # Convert to markdown based on role
            if role == "heading":
                level = int(attrs_dict.get("level", "2"))
                lines.append(f"{'#' * level} {content}")
                lines.append("")
            elif role == "paragraph":
                if content:
                    lines.append(content)
                    lines.append("")
            elif role == "link":
                # Link URL might be on next line
                link_url = attrs_dict.get("url", "")
                if link_url:
                    lines.append(f"[{content}]({link_url})")
                    lines.append("")
                else:
                    # Wait for /url: line
                    current_link_text = content
            elif role == "image":
                alt = content or "image"
                src = attrs_dict.get("url", attrs_dict.get("src", ""))
                if src:
                    lines.append(f"![{alt}]({src})")
                    lines.append("")
            elif role == "listitem":
                lines.append(f"- {content}")
            elif role == "code":
                lines.append(f"`{content}`")
            elif role in ("text", "StaticText"):
                if content:
                    lines.append(content)
            elif role == "button":
                pass  # Skip buttons
            elif role == "textbox":
                pass  # Skip form inputs
            elif role == "switch":
                pass  # Skip toggles
            elif content:
                # Generic fallback - include content
                lines.append(content)
            continue

        # Pattern 2: role: content (no quotes)
        # e.g., paragraph: Text content here
        # e.g., text: Some text
        match2 = re.match(r"(\w+):\s*(.+)$", stripped)
        if match2:
            role, content = match2.groups()
            content = content.strip()

            if role == "paragraph":
                lines.append(content)
                lines.append("")
            elif role == "text":
                # Only add text if it's meaningful (not just punctuation)
                if content and len(content) > 2:
                    lines.append(content)
            elif role == "heading":
                lines.append(f"## {content}")
                lines.append("")
            elif role == "time":
                lines.append(f"*{content}*")
                lines.append("")
            elif role in ("separator",):
                lines.append("---")
                lines.append("")
            continue

        # Pattern 3: Plain text line (not a role definition)
        # Skip structural elements
        if stripped and not stripped.endswith(":"):
            # Check if it looks like content (not a role marker)
            if not re.match(r"^[a-z]+$", stripped):
                pass  # Don't add raw structural lines

    # Clean up: remove consecutive empty lines
    result_lines = []
    prev_empty = False
    for line in lines:
        is_empty = not line.strip()
        if is_empty and prev_empty:
            continue
        result_lines.append(line)
        prev_empty = is_empty

    return "\n".join(result_lines).strip()


async def fetch_with_jina(
    url: str,
    api_key: str | None = None,
    timeout: int = 30,
) -> FetchResult:
    """Fetch URL using Jina Reader API.

    Args:
        url: URL to fetch
        api_key: Optional Jina API key (for higher rate limits)
        timeout: Request timeout in seconds

    Returns:
        FetchResult with markdown content

    Raises:
        JinaRateLimitError: If rate limit exceeded
        JinaAPIError: If API returns error
        FetchError: If fetch fails
    """
    import httpx

    logger.debug(f"Fetching URL with Jina Reader: {url}")

    jina_url = f"{DEFAULT_JINA_BASE_URL}/{url}"
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        client = _get_jina_client(timeout)
        response = await client.get(jina_url, headers=headers)

        if response.status_code == 429:
            raise JinaRateLimitError()
        elif response.status_code >= 400:
            raise JinaAPIError(response.status_code, response.text[:200])

        content = response.text

        if not content.strip():
            raise FetchError(f"No content returned from Jina Reader: {url}")

        # Extract title from first H1 if present
        title = None
        title_match = re.match(r"^#\s+(.+)$", content, re.MULTILINE)
        if title_match:
            title = title_match.group(1)

        return FetchResult(
            content=content,
            strategy_used="jina",
            title=title,
            url=url,
            metadata={"api": "jina-reader"},
        )

    except (JinaRateLimitError, JinaAPIError):
        raise
    except httpx.TimeoutException:
        raise FetchError(f"Jina Reader request timed out after {timeout}s: {url}")
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

    Returns:
        FetchResult with content and metadata

    Raises:
        FetchError: If fetch fails and no fallback available
        AgentBrowserNotFoundError: If --agent-browser used but not installed
        JinaRateLimitError: If --jina used and rate limit exceeded
    """
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
        )

    # Check cache first (unless skip_read_cache is True)
    if cache is not None and not skip_read_cache:
        cached_result = cache.get(url)
        if cached_result is not None:
            logger.info(f"[FetchCache] Using cached content for: {url}")
            return cached_result

    # Screenshot kwargs for browser fetching
    screenshot_kwargs: dict[str, Any] = {}

    # Fetch the content
    result: FetchResult

    # Handle explicit strategy (no fallback)
    if explicit_strategy:
        if strategy == FetchStrategy.BROWSER:
            result = await fetch_with_browser(
                url,
                command=config.agent_browser.command,
                timeout=config.agent_browser.timeout,
                wait_for=config.agent_browser.wait_for,
                extra_wait_ms=config.agent_browser.extra_wait_ms,
                session=config.agent_browser.session,
                **screenshot_kwargs,
            )
        elif strategy == FetchStrategy.JINA:
            api_key = config.jina.get_resolved_api_key()
            result = await fetch_with_jina(url, api_key, config.jina.timeout)
        elif strategy == FetchStrategy.STATIC:
            result = await fetch_with_static(url)
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
            logger.info(f"Domain matches fallback pattern, using browser: {url}")
            use_browser_first = True
        elif spa_cache.is_known_spa(url):
            logger.info(f"Domain is learned SPA, using browser: {url}")
            spa_cache.record_hit(url)
            use_browser_first = True

        result = await _fetch_with_fallback(
            url, config, start_with_browser=use_browser_first, **screenshot_kwargs
        )
    elif strategy == FetchStrategy.STATIC:
        result = await fetch_with_static(url)
    elif strategy == FetchStrategy.BROWSER:
        result = await fetch_with_browser(
            url,
            command=config.agent_browser.command,
            timeout=config.agent_browser.timeout,
            wait_for=config.agent_browser.wait_for,
            extra_wait_ms=config.agent_browser.extra_wait_ms,
            session=config.agent_browser.session,
            **screenshot_kwargs,
        )
    elif strategy == FetchStrategy.JINA:
        api_key = config.jina.get_resolved_api_key()
        result = await fetch_with_jina(url, api_key, config.jina.timeout)
    else:
        raise ValueError(f"Unknown fetch strategy: {strategy}")

    # Cache the result
    if cache is not None:
        cache.set(url, result)

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

    # Task 2: Browser fetch with screenshot
    async def fetch_browser() -> tuple[FetchResult | None, str | None, bool]:
        """Returns (result, error_message, not_installed)"""
        try:
            if not is_agent_browser_available(config.agent_browser.command):
                logger.debug("agent-browser not available")
                return None, None, True

            result = await fetch_with_browser(
                url,
                command=config.agent_browser.command,
                timeout=config.agent_browser.timeout,
                wait_for=config.agent_browser.wait_for,
                extra_wait_ms=config.agent_browser.extra_wait_ms,
                session=config.agent_browser.session,
                screenshot=True,
                screenshot_dir=screenshot_dir,
                screenshot_config=screenshot_config,
            )
            logger.debug(f"[URL] Browser fetch success: {len(result.content)} chars")
            return result, None, False
        except Exception as e:
            logger.debug(f"[URL] Browser fetch failed: {e}")
            return None, str(e), False

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
        logger.info(f"[URL] Using static content (valid, {len(static_content)} chars)")
    elif not browser_invalid:
        # Static invalid but browser is valid → use browser
        assert browser_content is not None
        primary_content = browser_content
        final_browser_content = browser_content
        strategy_used = "browser"
        logger.info(
            f"[URL] Using browser content (static invalid: {static_reason}, "
            f"browser valid, {len(browser_content)} chars)"
        )
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
            # Check browser status to generate accurate error message
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
                    f"Browser fetch failed: {browser_error}"
                )
            elif browser_not_installed:
                # Browser not installed
                raise FetchError(
                    f"URL requires browser rendering: {url}. "
                    f"Reason: {static_reason}. "
                    f"Please install agent-browser: npm install -g agent-browser && agent-browser install"
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
        cache.set(url, result)

    return result


async def _fetch_with_fallback(
    url: str,
    config: FetchConfig,
    start_with_browser: bool = False,
    **screenshot_kwargs: Any,
) -> FetchResult:
    """Fetch URL with automatic fallback between strategies.

    Args:
        url: URL to fetch
        config: Fetch configuration
        start_with_browser: If True, try browser first (for known JS domains)
        **screenshot_kwargs: Screenshot options (screenshot, screenshot_dir, screenshot_config)

    Returns:
        FetchResult from first successful strategy
    """
    errors = []

    if start_with_browser:
        # Try browser first for known JS domains
        strategies = ["browser", "jina", "static"]
    else:
        # Normal order: static -> browser -> jina
        strategies = ["static", "browser", "jina"]

    for strat in strategies:
        try:
            if strat == "static":
                result = await fetch_with_static(url)
                # Check if JS is required
                if detect_js_required(result.content):
                    logger.info(
                        "Static content suggests JS required, trying browser..."
                    )
                    # Learn this domain for future requests
                    spa_cache = get_spa_domain_cache()
                    spa_cache.record_spa_domain(url)
                    continue
                return result

            elif strat == "browser":
                if not is_agent_browser_available(config.agent_browser.command):
                    logger.debug("agent-browser not available, skipping")
                    continue
                return await fetch_with_browser(
                    url,
                    command=config.agent_browser.command,
                    timeout=config.agent_browser.timeout,
                    wait_for=config.agent_browser.wait_for,
                    extra_wait_ms=config.agent_browser.extra_wait_ms,
                    session=config.agent_browser.session,
                    **screenshot_kwargs,
                )

            elif strat == "jina":
                api_key = config.jina.get_resolved_api_key()
                return await fetch_with_jina(url, api_key, config.jina.timeout)

        except AgentBrowserNotFoundError:
            logger.debug("agent-browser not installed, trying next strategy")
            continue
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
