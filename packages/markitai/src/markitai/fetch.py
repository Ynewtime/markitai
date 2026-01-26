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


def _get_markitdown() -> Any:
    """Get or create the shared MarkItDown instance.

    Reusing a single instance avoids repeated initialization overhead.
    """
    global _markitdown_instance
    if _markitdown_instance is None:
        from markitdown import MarkItDown

        _markitdown_instance = MarkItDown()
    return _markitdown_instance


def _get_jina_client(timeout: int = 30) -> Any:
    """Get or create the shared httpx.AsyncClient for Jina fetching.

    Reusing a single client instance avoids repeated connection setup overhead.
    The client uses connection pooling for better performance.

    Args:
        timeout: Request timeout in seconds (used on first creation only)

    Returns:
        httpx.AsyncClient instance
    """
    global _jina_client
    if _jina_client is None:
        import httpx

        _jina_client = httpx.AsyncClient(
            timeout=timeout,
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )
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

    Args:
        content: HTML or Markdown content to check

    Returns:
        True if content suggests JavaScript is needed
    """
    if not content:
        return True  # Empty content likely means JS-rendered

    content_lower = content.lower()
    for pattern in JS_REQUIRED_PATTERNS:
        if pattern.lower() in content_lower:
            logger.debug(f"JS required pattern detected: {pattern}")
            return True

    # Check for very short content (likely a JS-only page)
    # Strip markdown formatting for length check
    text_only = re.sub(r"[#*_\[\]()>`-]", "", content).strip()
    if len(text_only) < 100:
        logger.debug(f"Content too short ({len(text_only)} chars), likely JS-rendered")
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

    # Check cache first
    if use_cache and command in _agent_browser_ready_cache:
        return _agent_browser_ready_cache[command]

    # Step 1: Check if command exists
    if not shutil.which(command):
        result = (
            False,
            f"'{command}' command not found. Install with: npm install -g agent-browser",
        )
        _agent_browser_ready_cache[command] = result
        return result

    # Step 2: Check if agent-browser responds to --help
    try:
        proc = subprocess.run(
            [command, "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if proc.returncode != 0:
            result = (False, f"'{command}' command failed: {proc.stderr.strip()}")
            _agent_browser_ready_cache[command] = result
            return result
    except subprocess.TimeoutExpired:
        result = (False, f"'{command}' command timed out")
        _agent_browser_ready_cache[command] = result
        return result
    except Exception as e:
        result = (False, f"'{command}' command error: {e}")
        _agent_browser_ready_cache[command] = result
        return result

    # Step 3: Try a simple operation to verify browser is installed
    # We use 'agent-browser snapshot' on about:blank which should fail fast if browser not installed
    try:
        proc = subprocess.run(
            [command, "open", "about:blank"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # Check for known error patterns
        if proc.returncode != 0:
            stderr_lower = proc.stderr.lower()
            stderr_orig = proc.stderr.strip()
            # Check for Playwright browser not installed error
            if (
                "executable doesn't exist" in stderr_lower
                or "browsertype.launch" in stderr_lower
            ):
                result = (
                    False,
                    "Playwright browser not installed. Run: agent-browser install "
                    "OR npx playwright install chromium",
                )
                _agent_browser_ready_cache[command] = result
                return result
            # Check for daemon not found error (global install needs AGENT_BROWSER_HOME)
            if "daemon not found" in stderr_lower:
                result = (
                    False,
                    "agent-browser daemon not found. "
                    "Set AGENT_BROWSER_HOME environment variable to the agent-browser package directory. "
                    "For pnpm global install: AGENT_BROWSER_HOME=$(pnpm list -g agent-browser --parseable)/node_modules/agent-browser "
                    "For npm global install: AGENT_BROWSER_HOME=$(npm root -g)/agent-browser",
                )
                _agent_browser_ready_cache[command] = result
                return result
            # Other errors might be transient, still mark as ready
            logger.debug(
                f"agent-browser test returned non-zero but may still work: {stderr_orig}"
            )
    except subprocess.TimeoutExpired:
        # Timeout on about:blank is suspicious but not fatal
        logger.debug("agent-browser test timed out, may still work for real pages")
    except Exception as e:
        logger.debug(f"agent-browser test error (may still work): {e}")

    # Close browser if opened
    try:
        subprocess.run([command, "close"], capture_output=True, timeout=5)
    except Exception:
        pass

    result = (True, "agent-browser is ready")
    _agent_browser_ready_cache[command] = result
    return result


def clear_agent_browser_cache() -> None:
    """Clear the agent-browser readiness cache."""
    _agent_browser_ready_cache.clear()


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

        # Step 1: Open URL and wait for page load
        open_args = [*base_args, "open", url]
        logger.debug(f"Running: {' '.join(open_args)}")

        proc = await asyncio.create_subprocess_exec(
            *open_args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=timeout / 1000 + 10
        )

        if proc.returncode != 0:
            error_msg = stderr.decode() if stderr else "Unknown error"
            raise FetchError(f"agent-browser open failed: {error_msg}")

        # Step 2: Wait for load state
        wait_args = [*base_args, "wait", "--load", wait_for]
        logger.debug(f"Running: {' '.join(wait_args)}")

        proc = await asyncio.create_subprocess_exec(
            *wait_args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await asyncio.wait_for(proc.communicate(), timeout=timeout / 1000 + 10)

        # Step 2.5: Extra wait for JS rendering (especially for SPAs)
        if extra_wait_ms > 0:
            extra_wait_args = [*base_args, "wait", str(extra_wait_ms)]
            logger.debug(f"Running: {' '.join(extra_wait_args)}")
            proc = await asyncio.create_subprocess_exec(
                *extra_wait_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(proc.communicate(), timeout=extra_wait_ms / 1000 + 5)

        # Step 3: Get page content via snapshot (accessibility tree with text)
        # Using snapshot -c (compact) to get clean text structure
        snapshot_args = [*base_args, "snapshot", "-c", "--json"]
        logger.debug(f"Running: {' '.join(snapshot_args)}")

        proc = await asyncio.create_subprocess_exec(
            *snapshot_args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=timeout / 1000 + 10
        )

        if proc.returncode != 0:
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
            proc = await asyncio.create_subprocess_exec(
                *title_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
            if proc.returncode == 0 and stdout:
                return stdout.decode().strip()
            return None

        async def get_final_url() -> str | None:
            url_args = [*base_args, "get", "url"]
            proc = await asyncio.create_subprocess_exec(
                *url_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
            if proc.returncode == 0 and stdout:
                return stdout.decode().strip()
            return None

        async def get_html_body() -> str | None:
            """Get HTML body content for text extraction."""
            html_args = [*base_args, "get", "html", "body"]
            proc = await asyncio.create_subprocess_exec(
                *html_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=15)
            if proc.returncode == 0 and stdout:
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
                        proc = await asyncio.create_subprocess_exec(
                            *viewport_args,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                        )
                        await asyncio.wait_for(proc.communicate(), timeout=10)

                    # Capture full-page screenshot
                    screenshot_args = [
                        *base_args,
                        "screenshot",
                        "--full",
                        str(screenshot_path),
                    ]
                    logger.debug(f"Running: {' '.join(screenshot_args)}")
                    proc = await asyncio.create_subprocess_exec(
                        *screenshot_args,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    stdout, stderr = await asyncio.wait_for(
                        proc.communicate(), timeout=60
                    )

                    if proc.returncode != 0:
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
                proc = await asyncio.create_subprocess_exec(
                    *close_args,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await asyncio.wait_for(proc.communicate(), timeout=5)
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
        if should_use_browser_for_domain(url, config.fallback_patterns):
            logger.info(f"Domain matches fallback pattern, using browser: {url}")
            result = await _fetch_with_fallback(
                url, config, start_with_browser=True, **screenshot_kwargs
            )
        else:
            # Try static first, fallback to browser/jina if JS required
            result = await _fetch_with_fallback(
                url, config, start_with_browser=False, **screenshot_kwargs
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
    async def fetch_browser() -> FetchResult | None:
        try:
            if not is_agent_browser_available(config.agent_browser.command):
                logger.debug("agent-browser not available")
                return None

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
            return result
        except Exception as e:
            logger.debug(f"[URL] Browser fetch failed: {e}")
            return None

    # Execute both fetches in parallel
    static_content, browser_result = await asyncio.gather(
        fetch_static(), fetch_browser()
    )

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
        # Both invalid, no browser but has static → use static with warning
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
