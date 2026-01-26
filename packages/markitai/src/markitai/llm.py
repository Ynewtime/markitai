"""LLM integration module using LiteLLM Router."""

from __future__ import annotations

import asyncio
import base64
import copy
import hashlib
import json
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import instructor
import litellm
from litellm import completion_cost
from litellm.exceptions import (
    APIConnectionError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)
from litellm.integrations.custom_logger import CustomLogger
from litellm.router import Router
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import Choices
from loguru import logger
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from markitai.config import LLMConfig, PromptsConfig
    from markitai.types import LLMUsageByModel, ModelUsageStats


from markitai.constants import (
    DEFAULT_CACHE_DB_FILENAME,
    DEFAULT_CACHE_MAXSIZE,
    DEFAULT_CACHE_SIZE_LIMIT,
    DEFAULT_CACHE_TTL_SECONDS,
    DEFAULT_GLOBAL_CACHE_DIR,
    DEFAULT_INSTRUCTOR_MAX_RETRIES,
    DEFAULT_IO_CONCURRENCY,
    DEFAULT_MAX_CONTENT_CHARS,
    DEFAULT_MAX_IMAGES_PER_BATCH,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MAX_PAGES_PER_BATCH,
    DEFAULT_MAX_RETRIES,
    DEFAULT_PROJECT_CACHE_DIR,
    DEFAULT_RETRY_BASE_DELAY,
    DEFAULT_RETRY_MAX_DELAY,
)
from markitai.prompts import PromptManager
from markitai.utils.mime import get_mime_type, is_llm_supported_image
from markitai.workflow.helpers import detect_language, get_language_name

# Retryable exceptions (kept here as they depend on litellm types)
RETRYABLE_ERRORS = (
    RateLimitError,
    APIConnectionError,
    Timeout,
    ServiceUnavailableError,
)


# Cache for model info to avoid repeated litellm queries
_model_info_cache: dict[str, dict[str, Any]] = {}


def get_model_info_cached(model: str) -> dict[str, Any]:
    """Get model info from litellm with caching.

    Args:
        model: Model identifier (e.g., "deepseek/deepseek-chat", "gemini/gemini-2.5-flash")

    Returns:
        Dict with keys:
            - max_input_tokens: int (context window size)
            - max_output_tokens: int (max output tokens)
            - supports_vision: bool (whether model supports images)
        Returns defaults if litellm info unavailable.
    """
    if model in _model_info_cache:
        return _model_info_cache[model]

    # Defaults
    result = {
        "max_input_tokens": 128000,  # Conservative default
        "max_output_tokens": DEFAULT_MAX_OUTPUT_TOKENS,
        "supports_vision": False,
    }

    try:
        info = litellm.get_model_info(model)
        if info.get("max_input_tokens"):
            result["max_input_tokens"] = info["max_input_tokens"]
        if info.get("max_output_tokens"):
            result["max_output_tokens"] = info["max_output_tokens"]
        supports_vision = info.get("supports_vision")
        if supports_vision is not None:
            result["supports_vision"] = bool(supports_vision)
    except Exception:
        logger.debug(f"[ModelInfo] Could not get info for {model}, using defaults")

    _model_info_cache[model] = result
    return result


def get_model_max_output_tokens(model: str) -> int:
    """Get max_output_tokens for a model using litellm.get_model_info().

    Args:
        model: Model identifier (e.g., "deepseek/deepseek-chat", "gemini/gemini-2.5-flash")

    Returns:
        max_output_tokens value, or DEFAULT_MAX_OUTPUT_TOKENS if unavailable
    """
    return get_model_info_cached(model)["max_output_tokens"]


def _context_display_name(context: str) -> str:
    """Extract display name from context for logging.

    Converts full paths to filenames while preserving suffixes like ':images'.
    Examples:
        'C:/path/to/file.pdf:images' -> 'file.pdf:images'
        'file.pdf' -> 'file.pdf'
        '' -> ''
    """
    if not context:
        return context
    # Split context into path part and suffix (e.g., ':images')
    if (
        ":" in context and context[1:3] != ":\\"
    ):  # Avoid splitting Windows drive letters
        # Find the last colon that's not part of a Windows path
        parts = context.rsplit(":", 1)
        if len(parts) == 2 and not parts[1].startswith("\\"):
            path_part, suffix = parts
            return f"{Path(path_part).name}:{suffix}"
    return Path(context).name


class MarkitaiLLMLogger(CustomLogger):
    """Custom LiteLLM callback logger for capturing additional call details."""

    def __init__(self) -> None:
        self.last_call_details: dict[str, Any] = {}

    def log_success_event(
        self,
        kwargs: dict,
        response_obj: Any,
        start_time: Any,
        end_time: Any,
    ) -> None:
        """Capture details from successful LLM calls."""
        slo = kwargs.get("standard_logging_object", {})
        self.last_call_details = {
            "api_base": slo.get("api_base"),
            "response_time": slo.get("response_time"),
            "cache_hit": kwargs.get("cache_hit", False),
            "model_id": slo.get("model_id"),
        }

    async def async_log_success_event(
        self,
        kwargs: dict,
        response_obj: Any,
        start_time: Any,
        end_time: Any,
    ) -> None:
        """Async version of success event logging."""
        self.log_success_event(kwargs, response_obj, start_time, end_time)

    def log_failure_event(
        self,
        kwargs: dict,
        response_obj: Any,
        start_time: Any,
        end_time: Any,
    ) -> None:
        """Capture details from failed LLM calls."""
        slo = kwargs.get("standard_logging_object", {})
        self.last_call_details = {
            "api_base": slo.get("api_base"),
            "error_code": slo.get("error_code"),
            "error_class": slo.get("error_class"),
        }

    async def async_log_failure_event(
        self,
        kwargs: dict,
        response_obj: Any,
        start_time: Any,
        end_time: Any,
    ) -> None:
        """Async version of failure event logging."""
        self.log_failure_event(kwargs, response_obj, start_time, end_time)


# Global callback instance
_markitai_llm_logger = MarkitaiLLMLogger()


@dataclass
class LLMRuntime:
    """Global LLM runtime with shared concurrency control.

    This allows multiple LLMProcessor instances to share semaphores
    for rate limiting across the entire application.

    Supports separate concurrency limits for:
    - LLM API calls (rate-limited by provider)
    - I/O operations (disk reads, can be higher)

    Usage:
        runtime = LLMRuntime(concurrency=10, io_concurrency=20)
        processor1 = LLMProcessor(config, runtime=runtime)
        processor2 = LLMProcessor(config, runtime=runtime)
        # Both processors share the same semaphores
    """

    concurrency: int
    io_concurrency: int = DEFAULT_IO_CONCURRENCY
    _semaphore: asyncio.Semaphore | None = field(default=None, init=False, repr=False)
    _io_semaphore: asyncio.Semaphore | None = field(
        default=None, init=False, repr=False
    )

    @property
    def semaphore(self) -> asyncio.Semaphore:
        """Get or create the shared LLM concurrency semaphore."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.concurrency)
        return self._semaphore

    @property
    def io_semaphore(self) -> asyncio.Semaphore:
        """Get or create the shared I/O concurrency semaphore."""
        if self._io_semaphore is None:
            self._io_semaphore = asyncio.Semaphore(self.io_concurrency)
        return self._io_semaphore


@dataclass
class LLMResponse:
    """Response from LLM call."""

    content: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


@dataclass
class ImageAnalysis:
    """Result of image analysis.

    Attributes:
        caption: Short alt text for accessibility
        description: Detailed markdown description
        extracted_text: Text extracted from image (OCR)
        llm_usage: LLM usage statistics in format:
            {"<model-name>": {"requests": N, "input_tokens": N,
             "output_tokens": N, "cost_usd": N}}
    """

    caption: str  # Short alt text
    description: str  # Detailed description
    extracted_text: str | None = None  # Text extracted from image
    llm_usage: LLMUsageByModel | None = None  # LLM usage stats


class ImageAnalysisResult(BaseModel):
    """Pydantic model for structured image analysis output."""

    caption: str = Field(description="Short alt text for the image (10-30 characters)")
    description: str = Field(description="Detailed markdown description of the image")
    extracted_text: str | None = Field(
        default=None,
        description="Text extracted from the image, preserving original layout",
    )


class SingleImageResult(BaseModel):
    """Result for a single image in batch analysis."""

    image_index: int = Field(description="Index of the image (1-based)")
    caption: str = Field(description="Short alt text for the image (10-30 characters)")
    description: str = Field(description="Detailed markdown description of the image")
    extracted_text: str | None = Field(
        default=None,
        description="Text extracted from the image, preserving original layout",
    )


class BatchImageAnalysisResult(BaseModel):
    """Result for batch image analysis."""

    images: list[SingleImageResult] = Field(
        description="Analysis results for each image"
    )


class Frontmatter(BaseModel):
    """Pydantic model for document frontmatter."""

    title: str = Field(description="Document title extracted from content")
    description: str = Field(
        description="Brief summary of the document (100 chars max)"
    )
    tags: list[str] = Field(description="Related tags (3-5 items)")


class DocumentProcessResult(BaseModel):
    """Pydantic model for combined cleaner + frontmatter output."""

    cleaned_markdown: str = Field(description="Cleaned and formatted markdown content")
    frontmatter: Frontmatter = Field(description="Document metadata")


class EnhancedDocumentResult(BaseModel):
    """Pydantic model for complete document enhancement output (Vision+LLM combined)."""

    cleaned_markdown: str = Field(description="Enhanced and cleaned markdown content")
    frontmatter: Frontmatter = Field(description="Document metadata")


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
        """Get a new database connection (thread-local)."""
        import sqlite3

        conn = sqlite3.connect(str(self._db_path), timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._get_connection() as conn:
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

        Uses head + tail + length strategy to detect changes anywhere in content:
        - Head: first 25000 chars (catches changes at the beginning)
        - Tail: last 25000 chars (catches changes at the end)
        - Length: total content length (catches changes that alter length)

        This avoids the problem where only using head truncation would miss
        changes at the end of large documents.
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

        with self._get_connection() as conn:
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

        with self._get_connection() as conn:
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
        with self._get_connection() as conn:
            count = conn.execute("SELECT COUNT(*) as cnt FROM cache").fetchone()["cnt"]
            conn.execute("DELETE FROM cache")
            conn.commit()
            return count

    def stats(self) -> dict[str, Any]:
        """Return cache statistics.

        Returns:
            Dict with count, size_bytes, size_mb, db_path
        """
        with self._get_connection() as conn:
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

    def stats_by_model(self) -> dict[str, dict[str, Any]]:
        """Get cache statistics grouped by model.

        Returns:
            Dict mapping model name to {"count": int, "size_bytes": int, "size_mb": float}
        """
        with self._get_connection() as conn:
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

        with self._get_connection() as conn:
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
        """Parse cached value to generate a human-readable preview.

        Args:
            value: The cached value (JSON string or plain text).

        Returns:
            Preview string like "image: Colorful bar chart..." or "text: # Title..."
        """
        import json

        if not value:
            return ""
        try:
            data = json.loads(value)
            if isinstance(data, dict):
                # Image description cache: has "caption" field
                if "caption" in data:
                    caption = str(data["caption"])[:40]
                    return f"image: {caption}..."
                # Frontmatter cache: has "title" field
                if "title" in data:
                    title = str(data["title"])[:40]
                    return f"frontmatter: {title}..."
            # Plain text result
            text = str(data) if not isinstance(data, str) else data
            return f"text: {text[:40]}..."
        except (json.JSONDecodeError, TypeError):
            return f"text: {value[:40]}..."


class PersistentCache:
    """Dual-layer persistent cache: project-level + global-level.

    Lookup order: project cache -> global cache -> None
    Write behavior: write to both caches simultaneously

    Supports "no-cache" mode (skip_read=True) which:
    - Skips reading from cache (always returns None)
    - Still writes to cache (for future use)
    This follows Bun's --no-cache semantics.

    Supports pattern-based cache skip (no_cache_patterns) which:
    - Skips reading from cache for matching files
    - Still writes to cache (for future use)
    - Patterns are glob-style, matched against relative paths
    """

    def __init__(
        self,
        project_dir: Path | None = None,
        global_dir: Path | None = None,
        max_size_bytes: int = DEFAULT_CACHE_SIZE_LIMIT,
        enabled: bool = True,
        skip_read: bool = False,
        no_cache_patterns: list[str] | None = None,
    ) -> None:
        """Initialize dual-layer cache.

        Args:
            project_dir: Project directory (will create .markitai/cache.db)
            global_dir: Global cache directory (default ~/.markitai)
            max_size_bytes: Max size per cache file
            enabled: Whether caching is enabled (both read and write)
            skip_read: If True, skip reading from cache but still write
                       (Bun's --no-cache semantics: force fresh, update cache)
            no_cache_patterns: List of glob patterns to skip cache for specific files.
                              Patterns are matched against relative paths from input_dir.
        """
        self._enabled = enabled
        self._skip_read = skip_read
        self._no_cache_patterns = no_cache_patterns or []
        self._project_cache: SQLiteCache | None = None
        self._global_cache: SQLiteCache | None = None
        self._hits = 0
        self._misses = 0

        if not enabled:
            return

        # Initialize project cache
        if project_dir:
            project_cache_path = (
                Path(project_dir)
                / DEFAULT_PROJECT_CACHE_DIR
                / DEFAULT_CACHE_DB_FILENAME
            )
            try:
                self._project_cache = SQLiteCache(project_cache_path, max_size_bytes)
                logger.debug(f"[Cache] Project cache: {project_cache_path}")
            except Exception as e:
                logger.warning(f"[Cache] Failed to init project cache: {e}")

        # Initialize global cache
        global_cache_dir = global_dir or Path(DEFAULT_GLOBAL_CACHE_DIR).expanduser()
        global_cache_path = Path(global_cache_dir) / DEFAULT_CACHE_DB_FILENAME
        try:
            self._global_cache = SQLiteCache(global_cache_path, max_size_bytes)
            logger.debug(f"[Cache] Global cache: {global_cache_path}")
        except Exception as e:
            logger.warning(f"[Cache] Failed to init global cache: {e}")

    def _glob_match(self, path: str, pattern: str) -> bool:
        """Enhanced glob matching that properly handles ** for zero-or-more directories.

        Standard fnmatch treats ** as matching one-or-more characters, not zero-or-more
        directory levels. This method enhances fnmatch to handle **/ prefix correctly.

        Args:
            path: Normalized path (forward slashes)
            pattern: Glob pattern (forward slashes)

        Returns:
            True if path matches pattern
        """
        import fnmatch

        # Standard fnmatch first
        if fnmatch.fnmatch(path, pattern):
            return True

        # Handle **/ prefix: should match zero or more directories
        # e.g., "**/*.pdf" should match both "file.pdf" and "a/b/file.pdf"
        if pattern.startswith("**/"):
            # Try matching without the **/ prefix (zero directories case)
            pattern_without_prefix = pattern[3:]
            if fnmatch.fnmatch(path, pattern_without_prefix):
                return True

        # Handle **/ in the middle of pattern
        # e.g., "src/**/test.py" should match "src/test.py"
        if "**/" in pattern and not pattern.startswith("**/"):
            # Replace **/ with empty string to test zero-directory case
            collapsed = pattern.replace("**/", "", 1)
            if fnmatch.fnmatch(path, collapsed):
                return True

        return False

    def _extract_matchable_path(self, context: str) -> str:
        """Extract a matchable file path from various context formats.

        Context can come in different formats:
        - Simple filename: "candy.JPG"
        - Relative path: "sub_dir/file.doc"
        - Absolute path: "/home/user/project/sub_dir/file.doc"
        - Path with suffix: "/home/user/project/candy.JPG:images"
        - Windows path: "C:\\Users\\test\\candy.JPG"

        This method extracts just the filename for matching against patterns.

        Args:
            context: Context identifier in any format

        Returns:
            Extracted path suitable for pattern matching
        """
        # Normalize path separators first (Windows -> Unix)
        path = context.replace("\\", "/")

        # Remove common suffixes like ":images", ":clean", ":frontmatter"
        # But be careful with Windows drive letters like "C:"
        if ":" in path:
            # Check if it's a Windows drive letter (single char before colon)
            colon_idx = path.index(":")
            if colon_idx == 1:
                # Windows drive letter, look for next colon
                rest = path[2:]
                if ":" in rest:
                    path = path[: 2 + rest.index(":")]
            else:
                # Regular suffix like ":images"
                path = path[:colon_idx]

        # Extract just the filename from paths
        # This allows patterns like "*.JPG" to match "/full/path/to/candy.JPG"
        if "/" in path:
            filename = path.rsplit("/", 1)[-1]
        else:
            filename = path

        return filename

    def _should_skip_cache(self, context: str) -> bool:
        """Check if cache should be skipped for the given context.

        Args:
            context: Context identifier (can be filename, relative path, or absolute path)

        Returns:
            True if cache should be skipped for this context
        """
        if not context or not self._no_cache_patterns:
            return False

        # Normalize path separators to forward slash for consistent matching
        normalized_context = context.replace("\\", "/")

        # Also extract just the filename for patterns like "*.JPG"
        filename = self._extract_matchable_path(context)

        for pattern in self._no_cache_patterns:
            # Normalize pattern separators
            normalized_pattern = pattern.replace("\\", "/")

            # Try matching against full context path first
            if self._glob_match(normalized_context, normalized_pattern):
                logger.debug(
                    f"[Cache] Skipping cache for '{context}' (matched pattern: {pattern})"
                )
                return True

            # Also try matching against just the filename
            # This handles cases where context is absolute path but pattern is "*.JPG"
            if filename != normalized_context and self._glob_match(
                filename, normalized_pattern
            ):
                logger.debug(
                    f"[Cache] Skipping cache for '{context}' (filename '{filename}' matched pattern: {pattern})"
                )
                return True

        return False

    def get(self, prompt: str, content: str, context: str = "") -> Any | None:
        """Lookup in project cache first, then global cache.

        Args:
            prompt: Prompt template used
            content: Content being processed
            context: Context identifier for pattern matching (e.g., relative file path)

        Returns:
            Cached result (deserialized from JSON) or None
        """
        if not self._enabled or self._skip_read:
            # skip_read: Bun-style --no-cache (force fresh, still write)
            return None

        # Check pattern-based skip
        if self._should_skip_cache(context):
            return None

        # Try project cache first
        if self._project_cache:
            result = self._project_cache.get(prompt, content)
            if result is not None:
                self._hits += 1
                logger.debug("[Cache] Project cache hit")
                return json.loads(result)

        # Fallback to global cache
        if self._global_cache:
            result = self._global_cache.get(prompt, content)
            if result is not None:
                self._hits += 1
                logger.debug("[Cache] Global cache hit")
                return json.loads(result)

        self._misses += 1
        return None

    def set(self, prompt: str, content: str, result: Any, model: str = "") -> None:
        """Write to both caches.

        Args:
            prompt: Prompt template used
            content: Content being processed
            result: Result to cache (will be JSON serialized)
            model: Model identifier
        """
        if not self._enabled:
            return

        value = json.dumps(result, ensure_ascii=False)

        if self._project_cache:
            try:
                self._project_cache.set(prompt, content, value, model)
            except Exception as e:
                logger.warning(f"[Cache] Failed to write to project cache: {e}")

        if self._global_cache:
            try:
                self._global_cache.set(prompt, content, value, model)
            except Exception as e:
                logger.warning(f"[Cache] Failed to write to global cache: {e}")

    def clear(self, scope: str = "project") -> dict[str, int]:
        """Clear cache entries.

        Args:
            scope: "project", "global", or "all"

        Returns:
            Dict with counts of deleted entries
        """
        result = {"project": 0, "global": 0}

        if scope in ("project", "all") and self._project_cache:
            result["project"] = self._project_cache.clear()

        if scope in ("global", "all") and self._global_cache:
            result["global"] = self._global_cache.clear()

        return result

    def stats(self) -> dict[str, Any]:
        """Return cache statistics.

        Returns:
            Dict with project/global stats and hit rate
        """
        total_requests = self._hits + self._misses
        hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0.0

        return {
            "project": self._project_cache.stats() if self._project_cache else None,
            "global": self._global_cache.stats() if self._global_cache else None,
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
        """
        Initialize content cache.

        Args:
            maxsize: Maximum number of entries to cache
            ttl_seconds: Time-to-live in seconds
        """
        from collections import OrderedDict

        self._cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._maxsize = maxsize
        self._ttl = ttl_seconds

    def _compute_hash(self, prompt: str, content: str) -> str:
        """Compute hash key from prompt and content.

        Uses full content for accurate cache keys. For very large content,
        the hash computation is still fast due to incremental SHA256.
        """
        import hashlib

        combined = f"{prompt}|{content}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    def get(self, prompt: str, content: str) -> Any | None:
        """
        Get cached result if exists and not expired.

        On hit, moves the entry to end (most recently used).

        Args:
            prompt: Prompt template used
            content: Content being processed

        Returns:
            Cached result or None if not found/expired
        """
        key = self._compute_hash(prompt, content)
        if key not in self._cache:
            return None

        result, timestamp = self._cache[key]
        if time.time() - timestamp > self._ttl:
            del self._cache[key]
            return None

        # Move to end on access (LRU behavior)
        self._cache.move_to_end(key)
        return result

    def set(self, prompt: str, content: str, result: Any) -> None:
        """
        Cache a result.

        Uses O(1) LRU eviction via OrderedDict.popitem(last=False).

        Args:
            prompt: Prompt template used
            content: Content being processed
            result: Result to cache
        """
        key = self._compute_hash(prompt, content)

        # If key exists, update and move to end
        if key in self._cache:
            self._cache[key] = (result, time.time())
            self._cache.move_to_end(key)
            return

        # Evict oldest entry if cache is full - O(1) operation
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


class LLMProcessor:
    """LLM processor using LiteLLM Router for load balancing."""

    def __init__(
        self,
        config: LLMConfig,
        prompts_config: PromptsConfig | None = None,
        runtime: LLMRuntime | None = None,
        project_dir: Path | None = None,
        no_cache: bool = False,
        no_cache_patterns: list[str] | None = None,
    ) -> None:
        """
        Initialize LLM processor.

        Args:
            config: LLM configuration
            prompts_config: Optional prompts configuration
            runtime: Optional shared runtime for concurrency control.
                     If provided, uses runtime's semaphore instead of creating one.
            project_dir: Optional project directory for project-level cache.
                         If None, only global cache is used.
            no_cache: If True, skip reading from cache but still write results.
                      Follows Bun's --no-cache semantics (force fresh, update cache).
            no_cache_patterns: List of glob patterns to skip cache for specific files.
                              Patterns are matched against relative paths from input_dir.
                              E.g., ["*.pdf", "reports/**", "file.docx"]
        """
        self.config = config
        self._runtime = runtime
        self._router: Router | None = None
        self._vision_router: Router | None = None  # Lazy-initialized vision router
        self._semaphore: asyncio.Semaphore | None = None
        self._prompt_manager = PromptManager(prompts_config)

        # Usage tracking (global across all contexts)
        # Use defaultdict to avoid check-then-create race conditions
        def _make_usage_dict() -> dict[str, Any]:
            return {
                "requests": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0.0,
            }

        self._usage: defaultdict[str, dict[str, Any]] = defaultdict(_make_usage_dict)

        # Per-context usage tracking for batch processing
        self._context_usage: defaultdict[str, defaultdict[str, dict[str, Any]]] = (
            defaultdict(lambda: defaultdict(_make_usage_dict))
        )

        # Call counter for each context (file)
        self._call_counter: defaultdict[str, int] = defaultdict(int)

        # Lock for thread-safe access to usage tracking dicts in concurrent contexts
        # Using threading.Lock instead of asyncio.Lock because:
        # 1. Dict operations are CPU-bound and don't need await
        # 2. Works in both sync and async contexts
        # The lock hold time is minimal (only simple dict updates)
        self._usage_lock = threading.Lock()

        # In-memory content cache for session-level deduplication (fast, no I/O)
        self._cache = ContentCache()
        self._cache_hits = 0
        self._cache_misses = 0

        # Persistent cache for cross-session reuse (SQLite-based)
        # no_cache=True: skip reading but still write (Bun semantics)
        # no_cache_patterns: skip reading for specific files matching patterns
        self._persistent_cache = PersistentCache(
            project_dir=project_dir,
            skip_read=no_cache,
            no_cache_patterns=no_cache_patterns,
        )

        # Image cache for avoiding repeated file reads during document processing
        # Key: file path string, Value: (bytes, base64_encoded_string)
        # Uses OrderedDict for LRU eviction when limits are reached
        from collections import OrderedDict

        self._image_cache: OrderedDict[str, tuple[bytes, str]] = OrderedDict()
        self._image_cache_max_size = 200  # Max number of images to cache
        self._image_cache_max_bytes = 500 * 1024 * 1024  # 500MB max total cache size
        self._image_cache_bytes = 0  # Current total bytes in cache

        # Register LiteLLM callback for additional details
        self._setup_callbacks()

    def _setup_callbacks(self) -> None:
        """Register LiteLLM callbacks for detailed logging."""
        # Add our custom logger to litellm callbacks if not already added
        if _markitai_llm_logger not in (litellm.callbacks or []):
            if litellm.callbacks is None:
                litellm.callbacks = []
            litellm.callbacks.append(_markitai_llm_logger)

    def _get_next_call_index(self, context: str) -> int:
        """Get the next call index for a given context.

        Thread-safe: uses lock for atomic increment.
        """
        with self._usage_lock:
            self._call_counter[context] += 1
            return self._call_counter[context]

    def reset_call_counter(self, context: str = "") -> None:
        """Reset call counter for a context or all contexts.

        Thread-safe: uses lock for safe modification.
        """
        with self._usage_lock:
            if context:
                self._call_counter.pop(context, None)
            else:
                self._call_counter.clear()

    @property
    def router(self) -> Router:
        """Get or create the LiteLLM Router."""
        if self._router is None:
            self._router = self._create_router()
        return self._router

    @property
    def semaphore(self) -> asyncio.Semaphore:
        """Get the LLM concurrency semaphore.

        If a runtime was provided, uses the shared semaphore from runtime.
        Otherwise creates a local semaphore.
        """
        if self._runtime is not None:
            return self._runtime.semaphore
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.config.concurrency)
        return self._semaphore

    @property
    def io_semaphore(self) -> asyncio.Semaphore:
        """Get the I/O concurrency semaphore for file operations.

        Separate from LLM semaphore to allow higher I/O parallelism.
        """
        if self._runtime is not None:
            return self._runtime.io_semaphore
        # Fallback: use higher limit for local I/O operations
        return asyncio.Semaphore(DEFAULT_IO_CONCURRENCY)

    def _create_router(self) -> Router:
        """Create LiteLLM Router from configuration."""
        if not self.config.model_list:
            raise ValueError("No models configured in llm.model_list")

        # Build model list with resolved API keys and max_tokens
        model_list = []
        for model_config in self.config.model_list:
            model_id = model_config.litellm_params.model
            model_entry = {
                "model_name": model_config.model_name,
                "litellm_params": {
                    "model": model_id,
                },
            }

            # Add optional params
            api_key = model_config.litellm_params.get_resolved_api_key()
            if api_key:
                model_entry["litellm_params"]["api_key"] = api_key

            if model_config.litellm_params.api_base:
                model_entry["litellm_params"]["api_base"] = (
                    model_config.litellm_params.api_base
                )

            if model_config.litellm_params.weight != 1:
                model_entry["litellm_params"]["weight"] = (
                    model_config.litellm_params.weight
                )

            # Note: max_tokens is NOT set at Router level
            # It will be calculated dynamically per-request based on input size
            # This avoids context overflow issues with shared context models

            if model_config.model_info:
                model_entry["model_info"] = model_config.model_info.model_dump()

            model_list.append(model_entry)

        # Build router settings
        router_settings = self.config.router_settings.model_dump()

        # Disable internal retries - we handle retries ourselves for better logging
        router_settings["num_retries"] = 0

        # Log router configuration (compact format)
        model_names = [e["litellm_params"]["model"].split("/")[-1] for e in model_list]
        logger.info(
            f"[Router] Creating with strategy={router_settings.get('routing_strategy')}, "
            f"models={len(model_list)}"
        )
        logger.debug(f"[Router] Models: {', '.join(model_names)}")

        return Router(model_list=model_list, **router_settings)

    def _create_router_from_models(
        self, models: list[Any], router_settings: dict[str, Any] | None = None
    ) -> Router:
        """Create a Router from a subset of model configurations.

        Args:
            models: List of ModelConfig objects from self.config.model_list
            router_settings: Optional router settings (uses default if not provided)

        Returns:
            LiteLLM Router instance
        """
        # Build model list with resolved API keys and max_tokens
        model_list = []
        for model_config in models:
            model_id = model_config.litellm_params.model
            model_entry = {
                "model_name": model_config.model_name,
                "litellm_params": {
                    "model": model_id,
                },
            }

            # Add optional params
            api_key = model_config.litellm_params.get_resolved_api_key()
            if api_key:
                model_entry["litellm_params"]["api_key"] = api_key

            if model_config.litellm_params.api_base:
                model_entry["litellm_params"]["api_base"] = (
                    model_config.litellm_params.api_base
                )

            if model_config.litellm_params.weight != 1:
                model_entry["litellm_params"]["weight"] = (
                    model_config.litellm_params.weight
                )

            # Note: max_tokens calculated dynamically per-request

            if model_config.model_info:
                model_entry["model_info"] = model_config.model_info.model_dump()

            model_list.append(model_entry)

        # Use provided settings or default
        settings = router_settings or self.config.router_settings.model_dump()
        settings["num_retries"] = 0  # We handle retries ourselves

        return Router(model_list=model_list, **settings)

    def _is_vision_model(self, model_config: Any) -> bool:
        """Check if a model supports vision.

        Priority:
        1. Config override (model_info.supports_vision) if explicitly set
        2. Auto-detect from litellm.get_model_info()

        Args:
            model_config: Model configuration object

        Returns:
            True if model supports vision
        """
        # Check config override first
        if (
            model_config.model_info
            and model_config.model_info.supports_vision is not None
        ):
            return model_config.model_info.supports_vision

        # Auto-detect from litellm
        model_id = model_config.litellm_params.model
        info = get_model_info_cached(model_id)
        return info.get("supports_vision", False)

    @property
    def vision_router(self) -> Router:
        """Get or create Router with only vision-capable models (lazy).

        Filters models using auto-detection from litellm or config override.
        Falls back to main router if no vision models found.

        Returns:
            LiteLLM Router with vision-capable models only
        """
        if self._vision_router is None:
            vision_models = [
                m for m in self.config.model_list if self._is_vision_model(m)
            ]

            if not vision_models:
                # No dedicated vision models - fall back to main router
                logger.warning(
                    "[Router] No vision-capable models configured, using main router"
                )
                self._vision_router = self.router
            else:
                model_names = [
                    m.litellm_params.model.split("/")[-1] for m in vision_models
                ]
                logger.info(
                    f"[Router] Creating vision router with {len(vision_models)} models"
                )
                logger.debug(f"[Router] Vision models: {', '.join(model_names)}")
                self._vision_router = self._create_router_from_models(vision_models)

        return self._vision_router

    def _message_contains_image(self, messages: list[dict[str, Any]]) -> bool:
        """Detect if messages contain image content.

        Checks for image_url type in message content parts.

        Args:
            messages: List of chat messages

        Returns:
            True if any message contains an image
        """
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        return True
        return False

    async def _call_llm(
        self,
        model: str,
        messages: list[dict[str, Any]],
        context: str = "",
    ) -> LLMResponse:
        """
        Make an LLM call with rate limiting, retry logic, and detailed logging.

        Smart router selection: automatically uses vision_router when messages
        contain images, otherwise uses the main router.

        Args:
            model: Logical model name (e.g., "default")
            messages: Chat messages
            context: Context identifier for logging (e.g., filename)

        Returns:
            LLMResponse with content and usage info
        """
        # Generate call ID for logging
        call_index = self._get_next_call_index(context) if context else 0
        call_id = f"{context}:{call_index}" if context else f"call:{call_index}"

        # Smart router selection based on message content
        requires_vision = self._message_contains_image(messages)
        router = self.vision_router if requires_vision else self.router

        max_retries = self.config.router_settings.num_retries
        return await self._call_llm_with_retry(
            model=model,
            messages=messages,
            call_id=call_id,
            context=context,
            max_retries=max_retries,
            router=router,
        )

    def _calculate_dynamic_max_tokens(
        self, messages: list[Any], model_hint: str | None = None
    ) -> int:
        """Calculate dynamic max_tokens based on input size and model limits.

        Uses conservative estimates to avoid context overflow across all models
        in the router's model list.

        Args:
            messages: Chat messages to estimate input tokens
            model_hint: Optional model name for more accurate limits

        Returns:
            Safe max_tokens value that won't exceed context limits
        """
        import re

        # Estimate input tokens (use gpt-4 tokenizer as reasonable approximation)
        try:
            input_tokens = litellm.token_counter(model="gpt-4", messages=messages)
        except Exception:
            # Fallback: rough estimate based on character count
            total_chars = sum(len(str(m.get("content", ""))) for m in messages)
            input_tokens = total_chars // 4  # ~4 chars per token

        # Detect table-heavy content (tables require more output tokens for formatting)
        content_str = str(messages)
        table_rows = len(re.findall(r"\|[^|]+\|", content_str))
        is_table_heavy = table_rows > 20  # More than 20 table rows

        # Get model limits - use minimum across all configured models for safety
        min_context = float("inf")
        min_output = float("inf")

        for model_config in self.config.model_list:
            model_id = model_config.litellm_params.model
            info = get_model_info_cached(model_id)
            min_context = min(min_context, info["max_input_tokens"])
            min_output = min(min_output, info["max_output_tokens"])

        # Use defaults if no models configured
        if min_context == float("inf"):
            min_context = 128000
        if min_output == float("inf"):
            min_output = DEFAULT_MAX_OUTPUT_TOKENS

        # Calculate available output space
        # Reserve buffer for safety (tokenizer differences, system overhead)
        buffer = max(500, int(input_tokens * 0.1))  # 10% or 500, whichever is larger
        available_context = int(min_context) - input_tokens - buffer

        # For table-heavy content, ensure output has at least 1.5x input tokens
        # since reformatting tables to Markdown often expands token count
        if is_table_heavy:
            min_required_output = int(input_tokens * 1.5)
            available_context = max(available_context, min_required_output)
            logger.debug(
                f"[DynamicTokens] Table-heavy content detected ({table_rows} rows), "
                f"min_required_output={min_required_output}"
            )

        # max_tokens = min(model's max_output, available context space)
        max_tokens = min(int(min_output), available_context)

        # Ensure reasonable minimum (higher for table-heavy content)
        min_floor = 4000 if is_table_heavy else 1000
        max_tokens = max(max_tokens, min_floor)

        logger.debug(
            f"[DynamicTokens] input={input_tokens}, context={int(min_context)}, "
            f"max_output={int(min_output)}, calculated={max_tokens}"
        )

        return max_tokens

    async def _call_llm_with_retry(
        self,
        model: str,
        messages: list[dict[str, Any]],
        call_id: str,
        context: str = "",
        max_retries: int = DEFAULT_MAX_RETRIES,
        router: Router | None = None,
    ) -> LLMResponse:
        """
        Make an LLM call with custom retry logic and detailed logging.

        Args:
            model: Logical model name (e.g., "default")
            messages: Chat messages
            call_id: Unique identifier for this call (for logging)
            context: Context identifier for usage tracking (e.g., filename)
            max_retries: Maximum number of retry attempts
            router: Router to use (defaults to self.router if not provided)

        Returns:
            LLMResponse with content and usage info
        """
        # Use provided router or default to main router
        active_router = router or self.router
        last_exception: Exception | None = None

        # Calculate dynamic max_tokens based on input size
        max_tokens = self._calculate_dynamic_max_tokens(messages)

        for attempt in range(max_retries + 1):
            start_time = time.perf_counter()

            async with self.semaphore:
                try:
                    # Log request start
                    if attempt == 0:
                        logger.debug(f"[LLM:{call_id}] Request to {model}")
                    else:
                        # Log retry attempt
                        error_type = (
                            type(last_exception).__name__
                            if last_exception
                            else "Unknown"
                        )
                        status_code = getattr(last_exception, "status_code", "N/A")
                        logger.warning(
                            f"[LLM:{call_id}] Retry #{attempt}: {error_type} "
                            f"status={status_code}"
                        )

                    response = await active_router.acompletion(
                        model=model,
                        messages=cast(list[AllMessageValues], messages),
                        max_tokens=max_tokens,
                        metadata={"call_id": call_id, "attempt": attempt},
                    )

                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    # litellm returns Choices (not StreamingChoices) for non-streaming
                    choice = cast(Choices, response.choices[0])
                    content = choice.message.content or ""
                    actual_model = response.model or model

                    # Calculate cost
                    try:
                        cost = completion_cost(completion_response=response)
                    except Exception:
                        cost = 0.0

                    # Track usage (usage attr exists at runtime but not in type stubs)
                    usage = getattr(response, "usage", None)
                    input_tokens = usage.prompt_tokens if usage else 0
                    output_tokens = usage.completion_tokens if usage else 0

                    self._track_usage(
                        actual_model, input_tokens, output_tokens, cost, context
                    )

                    # Log result
                    logger.info(
                        f"[LLM:{call_id}] {actual_model} "
                        f"tokens={input_tokens}+{output_tokens} "
                        f"time={elapsed_ms:.0f}ms cost=${cost:.6f}"
                    )

                    # Detect empty response (0 output tokens with substantial input)
                    # This usually indicates a model failure that should be retried
                    if output_tokens == 0 and input_tokens > 100:
                        if attempt < max_retries:
                            logger.warning(
                                f"[LLM:{call_id}] Empty response (0 output tokens), "
                                f"retrying with different model..."
                            )
                            # Treat as retryable error
                            await asyncio.sleep(
                                min(
                                    DEFAULT_RETRY_BASE_DELAY * (2**attempt),
                                    DEFAULT_RETRY_MAX_DELAY,
                                )
                            )
                            continue
                        else:
                            logger.error(
                                f"[LLM:{call_id}] Empty response after {max_retries + 1} "
                                f"attempts, returning empty content"
                            )

                    return LLMResponse(
                        content=content,
                        model=actual_model,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        cost_usd=cost,
                    )

                except RETRYABLE_ERRORS as e:
                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    last_exception = e

                    if attempt == max_retries:
                        # Final failure after all retries
                        error_type = type(e).__name__
                        status_code = getattr(e, "status_code", "N/A")
                        provider = getattr(e, "llm_provider", "N/A")
                        logger.error(
                            f"[LLM:{call_id}] Failed after {max_retries + 1} attempts: "
                            f"{error_type} status={status_code} provider={provider} "
                            f"time={elapsed_ms:.0f}ms"
                        )
                        raise

                    # Calculate exponential backoff delay
                    delay = min(
                        DEFAULT_RETRY_BASE_DELAY * (2**attempt), DEFAULT_RETRY_MAX_DELAY
                    )
                    await asyncio.sleep(delay)

                except Exception as e:
                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    error_type = type(e).__name__
                    status_code = getattr(e, "status_code", "N/A")
                    error_msg = str(e)[:200]  # Truncate long messages
                    logger.error(
                        f"[LLM:{call_id}] Failed: {error_type} "
                        f"status={status_code} msg={error_msg} "
                        f"time={elapsed_ms:.0f}ms"
                    )
                    raise

        # Should not reach here, but just in case
        raise RuntimeError(f"[LLM:{call_id}] Unexpected state in retry loop")

    def _track_usage(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost: float,
        context: str = "",
    ) -> None:
        """Track usage statistics per model (and optionally per context).

        Thread-safe: uses lock to protect concurrent access to usage dicts.

        Args:
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            cost: Cost in USD
            context: Optional context identifier (e.g., filename)
        """
        with self._usage_lock:
            # Track global usage (defaultdict auto-creates entries)
            self._usage[model]["requests"] += 1
            self._usage[model]["input_tokens"] += input_tokens
            self._usage[model]["output_tokens"] += output_tokens
            self._usage[model]["cost_usd"] += cost

            # Track per-context usage if context provided
            if context:
                self._context_usage[context][model]["requests"] += 1
                self._context_usage[context][model]["input_tokens"] += input_tokens
                self._context_usage[context][model]["output_tokens"] += output_tokens
                self._context_usage[context][model]["cost_usd"] += cost

    def get_usage(self) -> dict[str, dict[str, Any]]:
        """Get global usage statistics.

        Thread-safe: uses lock and returns a deep copy.
        """
        import copy

        with self._usage_lock:
            return copy.deepcopy(self._usage)

    def get_total_cost(self) -> float:
        """Get total cost across all models.

        Thread-safe: uses lock for consistent read.
        """
        with self._usage_lock:
            return sum(u["cost_usd"] for u in self._usage.values())

    def get_context_usage(self, context: str) -> dict[str, dict[str, Any]]:
        """Get usage statistics for a specific context.

        Thread-safe: uses lock and returns a deep copy.

        Args:
            context: Context identifier (e.g., filename)

        Returns:
            Usage statistics for that context, or empty dict if not found
        """
        import copy

        with self._usage_lock:
            return copy.deepcopy(self._context_usage.get(context, {}))

    def get_context_cost(self, context: str) -> float:
        """Get total cost for a specific context.

        Thread-safe: uses lock for consistent read.

        Args:
            context: Context identifier (e.g., filename)

        Returns:
            Total cost for that context
        """
        with self._usage_lock:
            context_usage = self._context_usage.get(context, {})
            return sum(u["cost_usd"] for u in context_usage.values())

    def clear_context_usage(self, context: str) -> None:
        """Clear usage tracking for a specific context.

        Thread-safe: uses lock for safe modification.

        Args:
            context: Context identifier to clear
        """
        with self._usage_lock:
            self._context_usage.pop(context, None)
            self._call_counter.pop(context, None)

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dict with memory cache stats, persistent cache stats, and combined hit rate
        """
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0.0
        return {
            "memory": {
                "hits": self._cache_hits,
                "misses": self._cache_misses,
                "hit_rate": round(hit_rate * 100, 2),
                "size": self._cache.size,
            },
            "persistent": self._persistent_cache.stats(),
        }

    def clear_cache(self, scope: str = "memory") -> dict[str, Any]:
        """Clear the content cache and reset statistics.

        Args:
            scope: "memory" (in-memory only), "project", "global", or "all"

        Returns:
            Dict with counts of cleared entries
        """
        result: dict[str, Any] = {"memory": 0, "project": 0, "global": 0}

        if scope in ("memory", "all"):
            result["memory"] = self._cache.size
            self._cache.clear()
            self._cache_hits = 0
            self._cache_misses = 0

        if scope in ("project", "global", "all"):
            persistent_result = self._persistent_cache.clear(scope)
            result["project"] = persistent_result.get("project", 0)
            result["global"] = persistent_result.get("global", 0)

        return result

    def clear_image_cache(self) -> None:
        """Clear the image cache to free memory after document processing."""
        self._image_cache.clear()
        self._image_cache_bytes = 0

    def _get_cached_image(self, image_path: Path) -> tuple[bytes, str]:
        """Get image bytes and base64 encoding, using cache if available.

        Uses LRU eviction when cache limits are reached (both count and bytes).
        Also ensures image is under 5MB limit for LLM API compatibility.

        Args:
            image_path: Path to the image file

        Returns:
            Tuple of (raw bytes, base64 encoded string)
        """
        path_key = str(image_path)

        if path_key in self._image_cache:
            # Move to end for LRU (most recently used)
            self._image_cache.move_to_end(path_key)
            return self._image_cache[path_key]

        # Read and encode image
        image_data = image_path.read_bytes()

        # Check size limit (5MB for Anthropic/LiteLLM safety)
        # Using 4.5MB to be safe
        MAX_IMAGE_SIZE = 4.5 * 1024 * 1024
        if len(image_data) > MAX_IMAGE_SIZE:
            try:
                import io

                from PIL import Image

                with io.BytesIO(image_data) as buffer:
                    img = Image.open(buffer)
                    # Resize logic: iterative downscaling if needed
                    quality = 85
                    max_dim = 2048

                    while True:
                        if max(img.size) > max_dim:
                            img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)

                        out_buffer = io.BytesIO()
                        # Use JPEG for compression efficiency unless transparency is needed
                        fmt = "JPEG"
                        if img.mode in ("RGBA", "LA") or (
                            img.format and img.format.upper() == "PNG"
                        ):
                            # If PNG is too big, convert to JPEG (losing transparency) or resize more
                            # For document analysis, JPEG is usually fine
                            if len(image_data) > 8 * 1024 * 1024:  # If huge, force JPEG
                                img = img.convert("RGB")
                                fmt = "JPEG"
                            else:
                                fmt = "PNG"

                        if fmt == "JPEG" and img.mode != "RGB":
                            img = img.convert("RGB")

                        if fmt == "JPEG":
                            img.save(out_buffer, format=fmt, quality=quality)
                        else:
                            img.save(out_buffer, format=fmt)
                        new_data = out_buffer.getvalue()

                        if len(new_data) <= MAX_IMAGE_SIZE:
                            image_data = new_data
                            logger.debug(
                                f"Resized large image {image_path.name}: {len(new_data) / 1024 / 1024:.2f}MB"
                            )
                            break

                        # If still too big, reduce quality/size
                        if quality > 50 and fmt == "JPEG":
                            quality -= 15
                        else:
                            max_dim = int(max_dim * 0.75)
                            if max_dim < 512:  # Safety floor
                                logger.warning(
                                    f"Could not compress {image_path.name} below 5MB even at 512px"
                                )
                                break

            except Exception as e:
                logger.warning(f"Failed to resize large image {image_path.name}: {e}")

        base64_image = base64.b64encode(image_data).decode()

        # Calculate entry size: raw bytes + base64 string (roughly 1.33x raw size)
        entry_bytes = len(image_data) + len(base64_image)

        # Evict old entries if adding this would exceed limits
        while self._image_cache and (
            len(self._image_cache) >= self._image_cache_max_size
            or self._image_cache_bytes + entry_bytes > self._image_cache_max_bytes
        ):
            # Remove oldest entry (first item in OrderedDict)
            _, oldest_value = self._image_cache.popitem(last=False)
            old_bytes = len(oldest_value[0]) + len(oldest_value[1])
            self._image_cache_bytes -= old_bytes

        # Cache if entry size is reasonable (skip very large single images)
        if entry_bytes < self._image_cache_max_bytes // 2:
            self._image_cache[path_key] = (image_data, base64_image)
            self._image_cache_bytes += entry_bytes

        return image_data, base64_image

    @staticmethod
    def _smart_truncate(text: str, max_chars: int, preserve_end: bool = False) -> str:
        """Truncate text at sentence/paragraph boundary to preserve readability.

        Instead of cutting at arbitrary positions, finds the nearest sentence
        or paragraph ending before the limit.

        Args:
            text: Text to truncate
            max_chars: Maximum character limit
            preserve_end: If True, preserve the end instead of the beginning

        Returns:
            Truncated text at a natural boundary
        """
        if len(text) <= max_chars:
            return text

        if preserve_end:
            # Find a good starting point from the end
            search_start = len(text) - max_chars
            search_text = text[search_start : search_start + 500]

            # Look for paragraph or sentence boundary
            for marker in ["\n\n", "\n", "。", ".", "！", "!", "？", "?"]:
                idx = search_text.find(marker)
                if idx != -1:
                    return text[search_start + idx + len(marker) :]

            return text[-max_chars:]

        # Default: preserve beginning, find a good ending point
        search_text = (
            text[max_chars - 500 : max_chars + 200]
            if max_chars > 500
            else text[: max_chars + 200]
        )
        search_offset = max(0, max_chars - 500)

        # Priority: paragraph > sentence > any break
        for marker in ["\n\n", "。\n", ".\n", "。", ".", "！", "!", "？", "?"]:
            idx = search_text.rfind(marker)
            if idx != -1:
                end_pos = search_offset + idx + len(marker)
                if (
                    end_pos <= max_chars + 100
                ):  # Allow slight overflow for better breaks
                    return text[:end_pos].rstrip()

        # Fall back to simple truncation
        return text[:max_chars]

    @staticmethod
    def extract_protected_content(content: str) -> dict[str, list[str]]:
        """Extract content that must be preserved through LLM processing.

        Extracts:
        - Image links: ![...](...)
        - Slide comments: <!-- Slide X --> or <!-- Slide number: X -->
        - Page number comments: <!-- Page number: X -->
        - Page image comments: <!-- ![Page X](...) --> and <!-- Page images... -->

        Args:
            content: Original markdown content

        Returns:
            Dict with 'images', 'slides', 'page_numbers', 'page_comments' lists
        """
        import re

        protected: dict[str, list[str]] = {
            "images": [],
            "slides": [],
            "page_numbers": [],
            "page_comments": [],
        }

        # Extract image links
        protected["images"] = re.findall(r"!\[[^\]]*\]\([^)]+\)", content)

        # Extract slide comments: <!-- Slide X --> or <!-- Slide number: X -->
        protected["slides"] = re.findall(
            r"<!--\s*Slide\s+(?:number:\s*)?\d+\s*-->", content
        )

        # Extract page number comments: <!-- Page number: X -->
        protected["page_numbers"] = re.findall(
            r"<!--\s*Page number:\s*\d+\s*-->", content
        )

        # Extract page image comments
        # Pattern 1: <!-- Page images for reference -->
        # Pattern 2: <!-- ![Page X](screenshots/...) -->
        page_header_pattern = r"<!--\s*Page images for reference\s*-->"
        page_img_pattern = r"<!--\s*!\[Page\s+\d+\]\([^)]*\)\s*-->"
        protected["page_comments"] = re.findall(
            page_header_pattern, content
        ) + re.findall(page_img_pattern, content)

        return protected

    @staticmethod
    def _protect_content(content: str) -> tuple[str, dict[str, str]]:
        """Replace protected content with placeholders before LLM processing.

        This preserves the position of images, slides, and page comments
        by replacing them with unique placeholders that the LLM is unlikely
        to modify.

        Args:
            content: Original markdown content

        Returns:
            Tuple of (content with placeholders, mapping of placeholder -> original)
        """
        import re

        mapping: dict[str, str] = {}
        result = content

        # Note: Images are NOT protected anymore.
        # The prompt instructs LLM to preserve image positions and only add alt text.
        # Protecting images with placeholders caused issues where LLM would delete
        # the placeholders, and then images would be appended to the end of the file.

        # 1. Protect Page number markers (PDF): <!-- Page number: X -->
        # These must stay at the beginning of each page's content
        page_num_pattern = r"<!--\s*Page number:\s*\d+\s*-->"
        for page_num_idx, match in enumerate(re.finditer(page_num_pattern, result)):
            placeholder = f"__MARKITAI_PAGENUM_{page_num_idx}__"
            mapping[placeholder] = match.group(0)
            result = result.replace(match.group(0), placeholder, 1)

        # 2. Protect Slide number markers (PPTX/PPT): <!-- Slide number: X -->
        # These must stay at the beginning of each slide's content
        slide_num_pattern = r"<!--\s*Slide number:\s*\d+\s*-->"
        for slide_num_idx, match in enumerate(re.finditer(slide_num_pattern, result)):
            placeholder = f"__MARKITAI_SLIDENUM_{slide_num_idx}__"
            mapping[placeholder] = match.group(0)
            result = result.replace(match.group(0), placeholder, 1)

        # 3. Protect page image comments: <!-- ![Page X](...) --> and <!-- Page images... -->
        # Use separate patterns for header and individual page image comments
        page_header_pattern = r"<!--\s*Page images for reference\s*-->"
        page_img_pattern = r"<!--\s*!\[Page\s+\d+\]\([^)]*\)\s*-->"
        page_idx = 0
        for match in re.finditer(page_header_pattern, result):
            placeholder = f"__MARKITAI_PAGE_{page_idx}__"
            mapping[placeholder] = match.group(0)
            result = result.replace(match.group(0), placeholder, 1)
            page_idx += 1
        for match in re.finditer(page_img_pattern, result):
            placeholder = f"__MARKITAI_PAGE_{page_idx}__"
            mapping[placeholder] = match.group(0)
            result = result.replace(match.group(0), placeholder, 1)
            page_idx += 1

        return result, mapping

    @staticmethod
    def _unprotect_content(
        content: str,
        mapping: dict[str, str],
        protected: dict[str, list[str]] | None = None,
    ) -> str:
        """Restore protected content from placeholders after LLM processing.

        Also handles cases where the LLM removed placeholders by appending
        missing content at the end, and detects garbage content replacement.

        Args:
            content: LLM output with placeholders
            mapping: Mapping of placeholder -> original content
            protected: Optional dict of protected content for fallback restoration

        Returns:
            Content with placeholders replaced by original content
        """
        import re

        result = content

        # Remove any slide/page number comments that LLM hallucinated
        # These are NOT from our placeholders and should be removed
        # Pattern: <!-- Slide number: X --> or <!-- Page number: X -->
        hallucinated_slide_pattern = r"<!--\s*Slide\s+number:\s*\d+\s*-->\s*\n?"
        hallucinated_page_pattern = r"<!--\s*Page\s+number:\s*\d+\s*-->\s*\n?"

        # Remove hallucinated markers BEFORE replacing placeholders:
        # - If original had markers (placeholders exist): ALL raw markers are hallucinated
        #   because the real ones are protected as __MARKITAI_SLIDENUM_X__ placeholders
        # - If original had NO markers (placeholders empty): ALL raw markers are hallucinated
        # Either way, we should remove all raw slide/page markers at this point
        result = re.sub(hallucinated_slide_pattern, "", result)
        result = re.sub(hallucinated_page_pattern, "", result)

        # First pass: replace placeholders with original content
        # Ensure page/slide number markers have proper blank lines around them
        for placeholder, original in mapping.items():
            # Check if this is a page or slide number marker
            is_page_slide_marker = "PAGENUM" in placeholder or "SLIDENUM" in placeholder
            if is_page_slide_marker:
                # Find the placeholder and ensure blank lines around it
                # Pattern: optional whitespace/newlines before placeholder
                pattern = rf"(\n*)\s*{re.escape(placeholder)}\s*(\n*)"
                match = re.search(pattern, result)
                if match:
                    # Replace with proper spacing: \n\n before, \n\n after
                    result = re.sub(pattern, f"\n\n{original}\n\n", result, count=1)
                else:
                    result = result.replace(placeholder, original)
            else:
                result = result.replace(placeholder, original)

        # Clean up any residual placeholders that LLM might have duplicated or misplaced
        # Pattern: __MARKITAI_*__ (any of our placeholder formats)
        residual_placeholder_pattern = r"__MARKITAI_[A-Z]+_\d+__\s*\n?"
        residual_count = len(re.findall(residual_placeholder_pattern, result))
        if residual_count > 0:
            logger.debug(
                f"Removing {residual_count} residual placeholders from LLM output"
            )
            result = re.sub(residual_placeholder_pattern, "", result)

        # NOTE: Removed heuristic logic that auto-inserted images into short slide sections.
        # This caused false positives where legitimate short slides like "Agenda", "Thanks",
        # "Q&A" were incorrectly replaced with images. The LLM should preserve slide content
        # as-is, and missing images will be handled by the fallback restoration below.

        # Second pass: if protected content was provided, restore any missing items
        # This handles cases where the LLM removed placeholders entirely
        if protected:
            import re

            # Helper to check if an image is already in result (by filename)
            def image_exists_in_result(img_syntax: str, text: str) -> bool:
                """Check if image already exists in result by filename."""
                match = re.search(r"\]\(([^)]+)\)", img_syntax)
                if match:
                    img_path = match.group(1)
                    img_name = img_path.split("/")[-1]
                    # Check if same filename exists in any image reference
                    return bool(
                        re.search(rf"!\[[^\]]*\]\([^)]*{re.escape(img_name)}\)", text)
                    )
                return False

            # Restore missing images at end (fallback)
            # Only restore if the image filename doesn't already exist
            for img in protected.get("images", []):
                if img not in result and not image_exists_in_result(img, result):
                    match = re.search(r"\]\(([^)]+)\)", img)
                    if match:
                        img_name = match.group(1).split("/")[-1]
                        logger.debug(f"Restoring missing image at end: {img_name}")
                    result = result.rstrip() + "\n\n" + img

            # Restore missing slide comments at heading boundaries
            # Key fix: Match slides to H1/H2 headings more intelligently
            missing_slides = [s for s in protected.get("slides", []) if s not in result]
            if missing_slides:
                slide_info = []
                for slide in missing_slides:
                    # Support both "Slide X" and "Slide number: X" formats
                    match = re.search(r"Slide\s+(?:number:\s*)?(\d+)", slide)
                    if match:
                        slide_info.append((int(match.group(1)), slide))
                slide_info.sort()

                lines = result.split("\n")
                # Find H1 and H2 headings as potential slide boundaries
                heading_positions = [
                    i
                    for i, line in enumerate(lines)
                    if line.startswith("# ") or line.startswith("## ")
                ]

                # Only insert if we have matching heading positions
                # Don't append orphan slide comments to the end
                inserted_count = 0
                for idx, (slide_num, slide) in enumerate(slide_info):
                    if idx < len(heading_positions):
                        insert_pos = heading_positions[idx] + inserted_count * 2
                        lines.insert(insert_pos, slide)
                        lines.insert(insert_pos + 1, "")
                        inserted_count += 1
                        logger.debug(
                            f"Restored slide {slide_num} before heading at line {insert_pos}"
                        )
                    # Don't append orphan slides to the end - they look wrong
                result = "\n".join(lines)

            # Restore missing page number markers
            # Page number markers should be at the beginning of each page's content
            missing_page_nums = [
                p for p in protected.get("page_numbers", []) if p not in result
            ]
            if missing_page_nums:
                # Sort by page number
                page_info = []
                for page_marker in missing_page_nums:
                    match = re.search(r"Page number:\s*(\d+)", page_marker)
                    if match:
                        page_info.append((int(match.group(1)), page_marker))
                page_info.sort()

                # Find major content boundaries (H1/H2 headings) as insertion points
                lines = result.split("\n")
                heading_positions = []
                for i, line in enumerate(lines):
                    if line.startswith("# ") or line.startswith("## "):
                        heading_positions.append(i)

                # Insert missing page markers before headings
                # Only when there are enough heading positions
                inserted_count = 0
                for idx, (page_num, marker) in enumerate(page_info):
                    if idx < len(heading_positions):
                        insert_pos = heading_positions[idx] + inserted_count * 2
                        lines.insert(insert_pos, marker)
                        lines.insert(insert_pos + 1, "")
                        inserted_count += 1
                        logger.debug(
                            f"Restored page number {page_num} before heading at line {insert_pos}"
                        )

                result = "\n".join(lines)

            # Restore missing page comments at end
            # Only restore if not already present (avoid duplicates)
            page_header = "<!-- Page images for reference -->"
            has_page_header = page_header in result

            for comment in protected.get("page_comments", []):
                if comment not in result:
                    # For page header, only add if not present
                    if comment == page_header:
                        if not has_page_header:
                            result = result.rstrip() + "\n\n" + comment
                            has_page_header = True
                    # For individual page image comments, check if already exists
                    else:
                        # Extract page number to check for duplicates
                        page_match = re.search(r"!\[Page\s+(\d+)\]", comment)
                        if page_match:
                            page_num = page_match.group(1)
                            # Check if this page is already referenced (commented or not)
                            page_pattern = rf"!\[Page\s+{page_num}\]"
                            if not re.search(page_pattern, result):
                                result = result.rstrip() + "\n" + comment

        return result

    @staticmethod
    def _fix_malformed_image_refs(text: str) -> str:
        """Fix malformed image references with extra closing parentheses.

        Fixes cases like: ![alt](path.jpg)) -> ![alt](path.jpg)

        This handles a common LLM output error where extra ) are added
        after image references. Uses context-aware parsing to avoid
        breaking legitimate nested structures like:
        - [![alt](img)](link) - clickable image
        - (text: ![alt](img)) - image inside parentheses

        Args:
            text: Content that may contain malformed image refs

        Returns:
            Content with fixed image references
        """
        result = []
        i = 0
        while i < len(text):
            # Check for image reference start: ![
            if text[i : i + 2] == "![":
                # Find the ]( delimiter
                bracket_end = text.find("](", i + 2)
                if bracket_end != -1:
                    # Find the matching ) for the image path
                    # Handle nested parens in path like: ![alt](path(1).jpg)
                    paren_start = bracket_end + 2
                    paren_count = 1
                    j = paren_start
                    while j < len(text) and paren_count > 0:
                        if text[j] == "(":
                            paren_count += 1
                        elif text[j] == ")":
                            paren_count -= 1
                        j += 1

                    # j now points to position after the closing )
                    img_ref = text[i:j]
                    result.append(img_ref)

                    # Count extra ) immediately after the image ref
                    extra_parens = 0
                    while (
                        j + extra_parens < len(text) and text[j + extra_parens] == ")"
                    ):
                        extra_parens += 1

                    if extra_parens > 0:
                        # Check if these ) are legitimate closers for outer parens
                        # by counting unmatched ( in the content before this image
                        prefix = "".join(
                            result[:-1]
                        )  # Exclude the image ref just added
                        open_parens = prefix.count("(") - prefix.count(")")

                        # Only keep ) that match unclosed (
                        keep_parens = min(extra_parens, max(0, open_parens))
                        result.append(")" * keep_parens)
                        i = j + extra_parens
                    else:
                        i = j
                    continue

            result.append(text[i])
            i += 1

        return "".join(result)

    @staticmethod
    def restore_protected_content(result: str, protected: dict[str, list[str]]) -> str:
        """Restore any protected content that was lost during LLM processing.

        Legacy method - use _unprotect_content for new code.

        Args:
            result: LLM output
            protected: Dict of protected content from extract_protected_content

        Returns:
            Result with missing protected content restored
        """
        return LLMProcessor._unprotect_content(result, {}, protected)

    async def clean_markdown(self, content: str, context: str = "") -> str:
        """
        Clean and optimize markdown content.

        Uses placeholder-based protection to preserve images, slides, and
        page comments in their original positions during LLM processing.

        Cache lookup order:
        1. In-memory cache (session-level, fast)
        2. Persistent cache (cross-session, SQLite)
        3. LLM API call

        Args:
            content: Raw markdown content
            context: Context identifier for logging (e.g., filename)

        Returns:
            Cleaned markdown content
        """
        cache_key = "cleaner"

        # 1. Check in-memory cache first (fastest)
        cached = self._cache.get(cache_key, content)
        if cached is not None:
            self._cache_hits += 1
            logger.debug(
                f"[{_context_display_name(context)}] Memory cache hit for clean_markdown"
            )
            return cached

        # 2. Check persistent cache (cross-session)
        cached = self._persistent_cache.get(cache_key, content, context=context)
        if cached is not None:
            self._cache_hits += 1
            logger.debug(
                f"[{_context_display_name(context)}] Persistent cache hit for clean_markdown"
            )
            # Also populate in-memory cache for faster subsequent access
            self._cache.set(cache_key, content, cached)
            return cached

        self._cache_misses += 1

        # 3. Extract and protect content before LLM processing
        protected = self.extract_protected_content(content)
        protected_content, mapping = self._protect_content(content)

        prompt = self._prompt_manager.get_prompt("cleaner", content=protected_content)

        response = await self._call_llm(
            model="default",
            messages=[{"role": "user", "content": prompt}],
            context=context,
        )

        # Restore protected content from placeholders, with fallback for removed items
        result = self._unprotect_content(response.content, mapping, protected)

        # Cache the result in both layers
        self._cache.set(cache_key, content, result)
        self._persistent_cache.set(cache_key, content, result, model="default")

        return result

    async def generate_frontmatter(
        self,
        content: str,
        source: str,
    ) -> str:
        """
        Generate YAML frontmatter for markdown content.

        Cache lookup order:
        1. In-memory cache (session-level, fast)
        2. Persistent cache (cross-session, SQLite)
        3. LLM API call

        Args:
            content: Markdown content
            source: Source file name

        Returns:
            YAML frontmatter string (without --- markers)
        """
        cache_key = f"frontmatter:{source}"

        # 1. Check in-memory cache first (fastest)
        cached = self._cache.get(cache_key, content)
        if cached is not None:
            self._cache_hits += 1
            logger.debug(f"[{source}] Memory cache hit for generate_frontmatter")
            return cached

        # 2. Check persistent cache (cross-session)
        cached = self._persistent_cache.get(cache_key, content, context=source)
        if cached is not None:
            self._cache_hits += 1
            logger.debug(f"[{source}] Persistent cache hit for generate_frontmatter")
            # Also populate in-memory cache for faster subsequent access
            self._cache.set(cache_key, content, cached)
            return cached

        self._cache_misses += 1

        # 3. Detect document language
        language = get_language_name(detect_language(content))

        prompt = self._prompt_manager.get_prompt(
            "frontmatter",
            content=self._smart_truncate(content, 4000),
            source=source,
            language=language,
        )

        response = await self._call_llm(
            model="default",
            messages=[{"role": "user", "content": prompt}],
            context=source,
        )

        result = response.content

        # Cache the result in both layers
        self._cache.set(cache_key, content, result)
        self._persistent_cache.set(cache_key, content, result, model="default")

        return result

    async def analyze_image(
        self, image_path: Path, language: str = "en", context: str = ""
    ) -> ImageAnalysis:
        """
        Analyze an image using vision model.

        Uses Instructor for structured output with fallback mechanisms:
        1. Try Instructor with structured output
        2. Fallback to JSON mode + manual parsing
        3. Fallback to original two-call method

        Args:
            image_path: Path to the image file
            language: Language for output (e.g., "en", "zh")
            context: Context identifier for usage tracking (e.g., source filename)

        Returns:
            ImageAnalysis with caption and description
        """
        # Filter unsupported image formats (SVG, BMP, ICO etc.)
        if not is_llm_supported_image(image_path.suffix):
            logger.debug(
                f"[{image_path.name}] Skipping unsupported format: {image_path.suffix}"
            )
            return ImageAnalysis(
                caption=image_path.stem,
                description=f"Image format {image_path.suffix} not supported for analysis",
            )

        # Get cached image data and base64 encoding
        _, base64_image = self._get_cached_image(image_path)

        # Check persistent cache using image hash + language as key
        # Use SHA256 hash of base64 as image fingerprint to avoid collisions
        # (JPEG files share the same header, so first N chars are identical)
        cache_key = f"image_analysis:{language}"
        image_fingerprint = hashlib.sha256(base64_image.encode()).hexdigest()
        cached = self._persistent_cache.get(
            cache_key, image_fingerprint, context=context
        )
        if cached is not None:
            logger.debug(f"[{image_path.name}] Persistent cache hit for analyze_image")
            # Reconstruct ImageAnalysis from cached dict
            return ImageAnalysis(
                caption=cached.get("caption", ""),
                description=cached.get("description", ""),
                extracted_text=cached.get("extracted_text"),
            )

        # Determine MIME type
        mime_type = get_mime_type(image_path.suffix)

        # Language instruction
        lang_instruction = (
            "Output in English." if language == "en" else "使用中文输出。"
        )

        # Get combined prompt
        prompt = self._prompt_manager.get_prompt("image_analysis")
        prompt = prompt.replace(
            "**输出语言必须与源文档保持一致** - 英文文档用英文，中文文档用中文",
            lang_instruction,
        )

        # Build message with image
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
                    },
                ],
            }
        ]

        # Use "default" model name - smart router will auto-select vision-capable model
        # since the message contains image content
        vision_model = "default"

        # Try structured output methods with fallbacks
        result = await self._analyze_image_with_fallback(
            messages, vision_model, image_path.name, context
        )

        # Store in persistent cache
        cache_value = {
            "caption": result.caption,
            "description": result.description,
            "extracted_text": result.extracted_text,
        }
        self._persistent_cache.set(
            cache_key, image_fingerprint, cache_value, model="vision"
        )

        return result

    async def analyze_images_batch(
        self,
        image_paths: list[Path],
        language: str = "en",
        max_images_per_batch: int = DEFAULT_MAX_IMAGES_PER_BATCH,
        context: str = "",
    ) -> list[ImageAnalysis]:
        """
        Analyze multiple images in batches with parallel execution.

        Batches are processed concurrently using asyncio.gather for better
        throughput. LLM concurrency is controlled by the shared semaphore.

        Args:
            image_paths: List of image paths to analyze
            language: Language for output ("en" or "zh")
            max_images_per_batch: Max images per LLM call (default 10)
            context: Context identifier for usage tracking (e.g., source filename)

        Returns:
            List of ImageAnalysis results in same order as input
        """
        if not image_paths:
            return []

        # Split into batches
        num_batches = (
            len(image_paths) + max_images_per_batch - 1
        ) // max_images_per_batch

        batches: list[tuple[int, list[Path]]] = []
        for batch_num in range(num_batches):
            batch_start = batch_num * max_images_per_batch
            batch_end = min(batch_start + max_images_per_batch, len(image_paths))
            batch_paths = image_paths[batch_start:batch_end]
            batches.append((batch_num, batch_paths))

        # Limit concurrent batches to avoid memory pressure from loading all images
        # at once. The semaphore controls LLM API calls, but images are loaded
        # before acquiring the semaphore. This batch-level limit prevents that.
        max_concurrent_batches = min(self.config.concurrency, num_batches)
        batch_semaphore = asyncio.Semaphore(max_concurrent_batches)

        display_name = _context_display_name(context)
        logger.info(
            f"[{display_name}] Analyzing {len(image_paths)} images in "
            f"{num_batches} batches (max {max_concurrent_batches} concurrent)"
        )

        # Process batches with backpressure and streaming
        async def process_batch(
            batch_num: int, batch_paths: list[Path]
        ) -> tuple[int, list[ImageAnalysis]]:
            """Process a single batch with backpressure control."""
            async with batch_semaphore:
                try:
                    results = await self.analyze_batch(batch_paths, language, context)
                    return (batch_num, results)
                except Exception as e:
                    logger.warning(
                        f"[{display_name}] Batch {batch_num + 1}/{num_batches} failed: {e}"
                    )
                    # Return empty results with placeholder for failed images
                    return (
                        batch_num,
                        [
                            ImageAnalysis(
                                caption=f"Image {i + 1}",
                                description="Analysis failed",
                            )
                            for i in range(len(batch_paths))
                        ],
                    )

        # Launch all batches and process results as they complete
        # Using as_completed allows earlier batches to free resources sooner
        tasks = {
            asyncio.create_task(process_batch(batch_num, paths)): batch_num
            for batch_num, paths in batches
        }

        batch_results: list[tuple[int, list[ImageAnalysis]]] = []
        for coro in asyncio.as_completed(tasks.keys()):
            try:
                result = await coro
                batch_results.append(result)
            except Exception as e:
                # Find which batch failed by checking tasks
                logger.error(f"[{display_name}] Batch processing error: {e}")

        # Sort by batch number and flatten results
        batch_results_sorted = sorted(batch_results, key=lambda x: x[0])
        all_results: list[ImageAnalysis] = []
        for _, results in batch_results_sorted:
            all_results.extend(results)

        return all_results

    async def analyze_batch(
        self,
        image_paths: list[Path],
        language: str,
        context: str = "",
    ) -> list[ImageAnalysis]:
        """Batch image analysis using Instructor.

        Uses the same prompt template as single image analysis for consistency.
        Checks persistent cache first and only calls LLM for uncached images.

        Args:
            image_paths: List of image paths to analyze
            language: Language for output ("en" or "zh")
            context: Context identifier for usage tracking

        Returns:
            List of ImageAnalysis results
        """
        # Filter unsupported formats and track their indices
        unsupported_results: dict[int, ImageAnalysis] = {}
        supported_paths: list[tuple[int, Path]] = []
        for i, image_path in enumerate(image_paths):
            if not is_llm_supported_image(image_path.suffix):
                logger.debug(
                    f"[{image_path.name}] Skipping unsupported format: {image_path.suffix}"
                )
                unsupported_results[i] = ImageAnalysis(
                    caption=image_path.stem,
                    description=f"Image format {image_path.suffix} not supported for analysis",
                )
            else:
                supported_paths.append((i, image_path))

        # If all images are unsupported, return placeholder results
        if not supported_paths:
            return [unsupported_results[i] for i in range(len(image_paths))]

        # Check persistent cache for all images first
        # Use same cache key format as analyze_image for consistency
        cache_key = f"image_analysis:{language}"
        cached_results: dict[int, ImageAnalysis] = {}
        uncached_indices: list[int] = []
        image_fingerprints: dict[int, str] = {}

        for orig_idx, image_path in supported_paths:
            _, base64_image = self._get_cached_image(image_path)
            # Use SHA256 hash to avoid collisions (JPEG files share same header)
            fingerprint = hashlib.sha256(base64_image.encode()).hexdigest()
            image_fingerprints[orig_idx] = fingerprint

            cached = self._persistent_cache.get(cache_key, fingerprint, context=context)
            if cached is not None:
                logger.debug(f"[{image_path.name}] Cache hit in batch analysis")
                cached_results[orig_idx] = ImageAnalysis(
                    caption=cached.get("caption", ""),
                    description=cached.get("description", ""),
                    extracted_text=cached.get("extracted_text"),
                )
            else:
                uncached_indices.append(orig_idx)

        # If all supported images are cached, return merged results
        display_name = _context_display_name(context)
        if not uncached_indices:
            logger.info(
                f"[{display_name}] All {len(supported_paths)} supported images found in cache"
            )
            # Merge unsupported and cached results
            return [
                unsupported_results.get(i) or cached_results[i]
                for i in range(len(image_paths))
            ]

        # Only process uncached images
        uncached_paths = [image_paths[i] for i in uncached_indices]
        logger.debug(
            f"[{display_name}] Cache: {len(cached_results)} hits, "
            f"{len(uncached_indices)} misses"
        )

        # Get base prompt from template (same as single image analysis)
        lang_instruction = (
            "Output in English." if language == "en" else "使用中文输出。"
        )
        base_prompt = self._prompt_manager.get_prompt("image_analysis")
        base_prompt = base_prompt.replace(
            "**输出语言必须与源文档保持一致** - 英文文档用英文，中文文档用中文",
            lang_instruction,
        )

        # Build batch prompt with the same base prompt
        batch_header = (
            f"请依次分析以下 {len(uncached_paths)} 张图片。对每张图片，"
            if language == "zh"
            else f"Analyze the following {len(uncached_paths)} images in order. For each image, "
        )
        prompt = f"{batch_header}{base_prompt}\n\nReturn a JSON object with an 'images' array containing results for each image in order."

        # Build content parts with uncached images only
        content_parts: list[dict] = [{"type": "text", "text": prompt}]

        for i, image_path in enumerate(uncached_paths, 1):
            _, base64_image = self._get_cached_image(image_path)
            mime_type = get_mime_type(image_path.suffix)

            # Unique image label that won't conflict with document content
            content_parts.append(
                {"type": "text", "text": f"\n__MARKITAI_IMG_LABEL_{i}__"}
            )
            content_parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
                }
            )

        try:
            async with self.semaphore:
                # Calculate dynamic max_tokens
                messages = [{"role": "user", "content": content_parts}]
                max_tokens = self._calculate_dynamic_max_tokens(messages)

                client = instructor.from_litellm(
                    self.vision_router.acompletion, mode=instructor.Mode.JSON
                )
                # max_retries allows Instructor to retry with validation error
                # feedback, which helps LLM fix JSON escaping issues
                (
                    response,
                    raw_response,
                ) = await client.chat.completions.create_with_completion(
                    model="default",
                    messages=cast(
                        list[ChatCompletionMessageParam],
                        messages,
                    ),
                    response_model=BatchImageAnalysisResult,
                    max_retries=DEFAULT_INSTRUCTOR_MAX_RETRIES,
                    max_tokens=max_tokens,
                )

                # Check for truncation
                if hasattr(raw_response, "choices") and raw_response.choices:
                    finish_reason = getattr(
                        raw_response.choices[0], "finish_reason", None
                    )
                    if finish_reason == "length":
                        raise ValueError("Output truncated due to max_tokens limit")

                # Track usage
                actual_model = getattr(raw_response, "model", None) or "default"
                input_tokens = 0
                output_tokens = 0
                cost = 0.0
                if hasattr(raw_response, "usage") and raw_response.usage is not None:
                    input_tokens = getattr(raw_response.usage, "prompt_tokens", 0) or 0
                    output_tokens = (
                        getattr(raw_response.usage, "completion_tokens", 0) or 0
                    )
                    try:
                        cost = completion_cost(completion_response=raw_response)
                    except Exception:
                        cost = 0.0
                    self._track_usage(
                        actual_model, input_tokens, output_tokens, cost, context
                    )

                # Calculate per-image usage (divide batch usage by number of images)
                num_images = max(len(response.images), 1)
                per_image_llm_usage: LLMUsageByModel = {
                    actual_model: cast(
                        "ModelUsageStats",
                        {
                            "requests": 1,  # Each image counts as 1 request share
                            "input_tokens": input_tokens // num_images,
                            "output_tokens": output_tokens // num_images,
                            "cost_usd": cost / num_images,
                        },
                    )
                }

                # Convert to ImageAnalysis list and store in cache
                new_results: list[ImageAnalysis] = []
                for idx, img_result in enumerate(response.images):
                    analysis = ImageAnalysis(
                        caption=img_result.caption,
                        description=img_result.description,
                        extracted_text=img_result.extracted_text,
                        llm_usage=per_image_llm_usage,
                    )
                    new_results.append(analysis)

                    # Store in persistent cache using original index
                    if idx < len(uncached_indices):
                        original_idx = uncached_indices[idx]
                        fingerprint = image_fingerprints[original_idx]
                        cache_value = {
                            "caption": analysis.caption,
                            "description": analysis.description,
                            "extracted_text": analysis.extracted_text,
                        }
                        self._persistent_cache.set(
                            cache_key, fingerprint, cache_value, model="vision"
                        )

                # Ensure we have results for all uncached images
                while len(new_results) < len(uncached_paths):
                    new_results.append(
                        ImageAnalysis(
                            caption="Image",
                            description="Image analysis failed",
                            extracted_text=None,
                            llm_usage=per_image_llm_usage,
                        )
                    )

                # Merge unsupported, cached and new results in original order
                final_results: list[ImageAnalysis] = []
                new_result_iter = iter(new_results)
                for i in range(len(image_paths)):
                    if i in unsupported_results:
                        final_results.append(unsupported_results[i])
                    elif i in cached_results:
                        final_results.append(cached_results[i])
                    else:
                        final_results.append(next(new_result_iter))

                return final_results

        except Exception as e:
            logger.warning(
                f"Batch image analysis failed: {e}, falling back to individual analysis"
            )
            # Fallback: analyze each image individually (uses persistent cache)
            # Pass context to maintain accurate per-file usage tracking
            # Note: cached_results may already have some hits from the initial check
            fallback_results: list[ImageAnalysis] = []
            for i, image_path in enumerate(image_paths):
                if i in unsupported_results:
                    # Use unsupported placeholder result
                    fallback_results.append(unsupported_results[i])
                elif i in cached_results:
                    # Use already-cached result
                    fallback_results.append(cached_results[i])
                else:
                    try:
                        # analyze_image will also check/populate cache
                        result = await self.analyze_image(image_path, language, context)
                        fallback_results.append(result)
                    except Exception:
                        fallback_results.append(
                            ImageAnalysis(
                                caption="Image",
                                description="Image analysis failed",
                                extracted_text=None,
                            )
                        )
            return fallback_results

    def _get_actual_model_name(self, logical_name: str) -> str:
        """Get actual model name from router configuration."""
        for model_config in self.config.model_list:
            if model_config.model_name == logical_name:
                return model_config.litellm_params.model
        # Fallback to first model if logical name not found
        if self.config.model_list:
            return self.config.model_list[0].litellm_params.model
        return "gpt-4o-mini"  # Ultimate fallback

    async def _analyze_image_with_fallback(
        self,
        messages: list[dict],
        model: str,
        image_name: str,
        context: str = "",
    ) -> ImageAnalysis:
        """
        Analyze image with multiple fallback strategies.

        Strategy 1: Instructor structured output (most precise)
        Strategy 2: JSON mode + manual parsing
        Strategy 3: Original two-call method (most compatible)

        Args:
            messages: LLM messages with image
            model: Model name to use
            image_name: Image filename for logging
            context: Context identifier for usage tracking
        """
        # Strategy 1: Try Instructor
        try:
            # Deep copy to prevent Instructor from modifying original messages
            result = await self._analyze_with_instructor(
                copy.deepcopy(messages), model, context
            )
            return result
        except Exception as e:
            logger.debug(f"[{image_name}] Instructor failed: {e}, trying JSON mode")

        # Strategy 2: Try JSON mode
        try:
            result = await self._analyze_with_json_mode(
                copy.deepcopy(messages), model, context
            )
            logger.debug(f"[{image_name}] Used JSON mode fallback")
            return result
        except Exception as e:
            logger.debug(
                f"[{image_name}] JSON mode failed: {e}, using two-call fallback"
            )

        # Strategy 3: Original two-call method
        return await self._analyze_with_two_calls(
            copy.deepcopy(messages), model, context=context or image_name
        )

    async def _analyze_with_instructor(
        self,
        messages: list[dict],
        model: str,
        context: str = "",
    ) -> ImageAnalysis:
        """Analyze using Instructor for structured output."""
        async with self.semaphore:
            # Calculate dynamic max_tokens
            max_tokens = self._calculate_dynamic_max_tokens(messages)

            # Create instructor client from vision router for load balancing
            client = instructor.from_litellm(
                self.vision_router.acompletion, mode=instructor.Mode.JSON
            )

            # Use create_with_completion to get both the model and the raw response
            # max_retries allows Instructor to retry with validation error
            # feedback, which helps LLM fix JSON escaping issues
            (
                response,
                raw_response,
            ) = await client.chat.completions.create_with_completion(
                model=model,
                messages=cast(list[ChatCompletionMessageParam], messages),
                response_model=ImageAnalysisResult,
                max_retries=DEFAULT_INSTRUCTOR_MAX_RETRIES,
                max_tokens=max_tokens,
            )

            # Check for truncation
            if hasattr(raw_response, "choices") and raw_response.choices:
                finish_reason = getattr(raw_response.choices[0], "finish_reason", None)
                if finish_reason == "length":
                    raise ValueError("Output truncated due to max_tokens limit")

            # Track usage from raw API response
            # Get actual model from response for accurate tracking
            actual_model = getattr(raw_response, "model", None) or model
            input_tokens = 0
            output_tokens = 0
            cost = 0.0
            if hasattr(raw_response, "usage") and raw_response.usage is not None:
                input_tokens = getattr(raw_response.usage, "prompt_tokens", 0) or 0
                output_tokens = getattr(raw_response.usage, "completion_tokens", 0) or 0
                try:
                    cost = completion_cost(completion_response=raw_response)
                except Exception:
                    cost = 0.0
                self._track_usage(
                    actual_model, input_tokens, output_tokens, cost, context
                )

            # Build llm_usage dict for this analysis
            llm_usage: LLMUsageByModel = {
                actual_model: cast(
                    "ModelUsageStats",
                    {
                        "requests": 1,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "cost_usd": cost,
                    },
                )
            }

            return ImageAnalysis(
                caption=response.caption.strip(),
                description=response.description,
                extracted_text=response.extracted_text,
                llm_usage=llm_usage,
            )

    async def _analyze_with_json_mode(
        self,
        messages: list[dict],
        model: str,
        context: str = "",
    ) -> ImageAnalysis:
        """Analyze using JSON mode with manual parsing."""
        # Add JSON instruction to the prompt
        json_messages = messages.copy()
        json_messages[0] = {
            **messages[0],
            "content": [
                {
                    "type": "text",
                    "text": messages[0]["content"][0]["text"]
                    + "\n\nReturn a JSON object with 'caption' and 'description' fields.",
                },
                messages[0]["content"][1],  # image
            ],
        }

        async with self.semaphore:
            # Calculate dynamic max_tokens for vision request
            max_tokens = self._calculate_dynamic_max_tokens(json_messages)

            # Use vision_router for image analysis (not main router)
            response = await self.vision_router.acompletion(
                model=model,
                messages=cast(list[AllMessageValues], json_messages),
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )

            # litellm returns Choices (not StreamingChoices) for non-streaming
            choice = cast(Choices, response.choices[0])
            content = choice.message.content if choice.message else "{}"
            actual_model = response.model or model

            # Track usage
            usage = getattr(response, "usage", None)
            input_tokens = usage.prompt_tokens if usage else 0
            output_tokens = usage.completion_tokens if usage else 0
            try:
                cost = completion_cost(completion_response=response)
            except Exception:
                cost = 0.0
            self._track_usage(actual_model, input_tokens, output_tokens, cost, context)

            # Parse JSON
            data = json.loads(content or "{}")

            # Build llm_usage dict for this analysis
            llm_usage: LLMUsageByModel = {
                actual_model: cast(
                    "ModelUsageStats",
                    {
                        "requests": 1,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "cost_usd": cost,
                    },
                )
            }

            return ImageAnalysis(
                caption=data.get("caption", "").strip(),
                description=data.get("description", ""),
                extracted_text=data.get("extracted_text"),
                llm_usage=llm_usage,
            )

    async def _analyze_with_two_calls(
        self,
        messages: list[dict],
        model: str,  # noqa: ARG002
        context: str = "",
    ) -> ImageAnalysis:
        """Original two-call method as final fallback."""
        # Extract original prompt and image from messages
        original_content = messages[0]["content"]
        image_content = original_content[1]  # The image part

        # Language instruction (extract from original prompt)
        lang_instruction = "Output in English."
        if "使用中文输出" in original_content[0]["text"]:
            lang_instruction = "使用中文输出。"

        # Generate caption
        caption_prompt = self._prompt_manager.get_prompt("image_caption")
        caption_prompt = caption_prompt.replace(
            "**输出语言必须与源文档保持一致** - 英文文档用英文，中文文档用中文",
            lang_instruction,
        )
        caption_response = await self._call_llm(
            model="default",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": caption_prompt},
                        image_content,
                    ],
                }
            ],
            context=context,
        )

        # Generate description
        desc_prompt = self._prompt_manager.get_prompt("image_description")
        desc_response = await self._call_llm(
            model="default",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": desc_prompt},
                        image_content,
                    ],
                }
            ],
            context=context,
        )

        # Build aggregated llm_usage from both calls
        llm_usage: LLMUsageByModel = {}
        for resp in [caption_response, desc_response]:
            if resp.model not in llm_usage:
                llm_usage[resp.model] = cast(
                    "ModelUsageStats",
                    {
                        "requests": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "cost_usd": 0.0,
                    },
                )
            llm_usage[resp.model]["requests"] += 1
            llm_usage[resp.model]["input_tokens"] += resp.input_tokens
            llm_usage[resp.model]["output_tokens"] += resp.output_tokens
            llm_usage[resp.model]["cost_usd"] += resp.cost_usd

        return ImageAnalysis(
            caption=caption_response.content.strip(),
            description=desc_response.content,
            llm_usage=llm_usage,
        )

    async def extract_page_content(self, image_path: Path, context: str = "") -> str:
        """
        Extract text content from a document page image.

        Used for OCR+LLM mode and PPTX+LLM mode where pages are rendered
        as images and we want to extract structured text content.

        Args:
            image_path: Path to the page image file
            context: Context identifier for logging (e.g., parent document name)

        Returns:
            Extracted markdown content from the page
        """
        # Get cached image data and base64 encoding
        _, base64_image = self._get_cached_image(image_path)

        # Determine MIME type
        mime_type = get_mime_type(image_path.suffix)

        # Get page content extraction prompt
        prompt = self._prompt_manager.get_prompt("page_content")

        # Use image_path.name as context if not provided
        call_context = context or image_path.name

        response = await self._call_llm(
            model="default",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            context=call_context,
        )

        return response.content

    @staticmethod
    def _protect_image_positions(text: str) -> tuple[str, dict[str, str]]:
        """Replace image references with position markers to prevent LLM from moving them.

        Args:
            text: Markdown text with image references

        Returns:
            Tuple of (text with markers, mapping of marker -> original image reference)
        """
        import re

        mapping: dict[str, str] = {}
        result = text

        # Match ALL image references: ![...](...)
        # This includes both local assets and external URLs
        # Excludes screenshots placeholder which has its own protection
        img_pattern = r"!\[[^\]]*\]\([^)]+\)"
        for i, match in enumerate(re.finditer(img_pattern, text)):
            img_ref = match.group(0)
            # Skip screenshot placeholders (handled separately)
            if "screenshots/" in img_ref:
                continue
            marker = f"<!-- IMG_MARKER: {i} -->"
            mapping[marker] = img_ref
            result = result.replace(img_ref, marker, 1)

        return result, mapping

    @staticmethod
    def _restore_image_positions(text: str, mapping: dict[str, str]) -> str:
        """Restore original image references from position markers.

        Args:
            text: Text with position markers
            mapping: Mapping of marker -> original image reference

        Returns:
            Text with original image references restored
        """
        result = text
        for marker, original in mapping.items():
            result = result.replace(marker, original)
        return result

    async def enhance_url_with_vision(
        self,
        content: str,
        screenshot_path: Path,
        context: str = "",
    ) -> tuple[str, str]:
        """
        Enhance URL content using screenshot as visual reference.

        Unlike enhance_document_with_vision, this method:
        - Does NOT use slide/page number protection (URLs don't have these)
        - Generates frontmatter along with cleaned content
        - Uses a simpler content protection strategy

        Args:
            content: URL content (may be multi-source combined)
            screenshot_path: Path to full-page screenshot
            context: Source URL for logging

        Returns:
            Tuple of (cleaned_markdown, frontmatter_yaml)
        """
        import time

        import yaml

        start_time = time.perf_counter()

        # Check persistent cache
        cache_key = f"enhance_url:{context}"
        cache_content = f"{screenshot_path.name}|{content[:1000]}"
        cached = self._persistent_cache.get(cache_key, cache_content, context=context)
        if cached is not None:
            logger.debug(
                f"[{context}] Persistent cache hit for enhance_url_with_vision"
            )
            return cached.get("cleaned_markdown", content), cached.get(
                "frontmatter_yaml", ""
            )

        # Only protect image references, NOT slide/page markers (URLs don't have them)
        protected_text, img_mapping = self._protect_image_positions(content)

        # Get URL-specific prompt (not document_enhance_complete which has slide/page markers)
        prompt = self._prompt_manager.get_prompt(
            "url_enhance",
            source=context,
        )

        # Build content parts
        content_parts: list[dict] = [
            {
                "type": "text",
                "text": f"{prompt}\n\n## URL Content:\n\n{protected_text}",
            },
        ]

        # Add screenshot
        _, base64_image = self._get_cached_image(screenshot_path)
        mime_type = get_mime_type(screenshot_path.suffix)
        content_parts.append({"type": "text", "text": "\n__MARKITAI_SCREENSHOT__"})
        content_parts.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
            }
        )

        async with self.semaphore:
            # Calculate dynamic max_tokens
            messages = [{"role": "user", "content": content_parts}]
            max_tokens = self._calculate_dynamic_max_tokens(messages)

            client = instructor.from_litellm(
                self.vision_router.acompletion, mode=instructor.Mode.JSON
            )
            (
                response,
                raw_response,
            ) = await client.chat.completions.create_with_completion(
                model="default",
                messages=cast(
                    list[ChatCompletionMessageParam],
                    messages,
                ),
                response_model=EnhancedDocumentResult,
                max_retries=DEFAULT_INSTRUCTOR_MAX_RETRIES,
                max_tokens=max_tokens,
            )

            # Track usage and log completion
            actual_model = getattr(raw_response, "model", None) or "default"
            input_tokens = 0
            output_tokens = 0
            cost = 0.0
            elapsed = time.perf_counter() - start_time

            if hasattr(raw_response, "usage") and raw_response.usage is not None:
                input_tokens = getattr(raw_response.usage, "prompt_tokens", 0) or 0
                output_tokens = getattr(raw_response.usage, "completion_tokens", 0) or 0
                try:
                    cost = completion_cost(completion_response=raw_response)
                except Exception:
                    cost = 0.0
                self._track_usage(
                    actual_model,
                    input_tokens,
                    output_tokens,
                    cost,
                    context,
                )

            logger.info(
                f"[LLM:{context}] url_vision_enhance: {actual_model} "
                f"tokens={input_tokens}+{output_tokens} "
                f"time={int(elapsed * 1000)}ms cost=${cost:.6f}"
            )

        # Restore image positions
        cleaned_markdown = self._restore_image_positions(
            response.cleaned_markdown, img_mapping
        )

        # Remove any hallucinated or leaked markers that shouldn't be in URL output
        import re

        # Remove hallucinated slide/page markers (URLs shouldn't have these)
        cleaned_markdown = re.sub(
            r"<!--\s*Slide\s+number:\s*\d+\s*-->\s*\n?", "", cleaned_markdown
        )
        cleaned_markdown = re.sub(
            r"<!--\s*Page\s+number:\s*\d+\s*-->\s*\n?", "", cleaned_markdown
        )
        # Remove source labels that may leak from multi-source content
        cleaned_markdown = re.sub(
            r"<!--\s*Source:\s*[^>]+-->\s*\n?", "", cleaned_markdown
        )
        cleaned_markdown = re.sub(
            r"##\s*(Static Content|Browser Content|Screenshot Reference)\s*\n+",
            "",
            cleaned_markdown,
        )
        # Also remove any residual MARKITAI placeholders
        cleaned_markdown = re.sub(
            r"__MARKITAI_[A-Z_]+_?\d*__\s*\n?", "", cleaned_markdown
        )

        # Fix malformed image refs
        cleaned_markdown = self._fix_malformed_image_refs(cleaned_markdown)

        # Build frontmatter
        frontmatter_dict = {
            "title": response.frontmatter.title,
            "source": context,
            "description": response.frontmatter.description,
            "tags": response.frontmatter.tags,
        }
        frontmatter_yaml = yaml.dump(
            frontmatter_dict, allow_unicode=True, default_flow_style=False
        ).strip()

        # Cache result
        cache_value = {
            "cleaned_markdown": cleaned_markdown,
            "frontmatter_yaml": frontmatter_yaml,
        }
        self._persistent_cache.set(
            cache_key, cache_content, cache_value, model="vision"
        )

        return cleaned_markdown, frontmatter_yaml

    async def enhance_document_with_vision(
        self,
        extracted_text: str,
        page_images: list[Path],
        context: str = "",
    ) -> str:
        """
        Clean document format using extracted text and page images as reference.

        This method only cleans formatting issues (removes residuals, fixes structure).
        It does NOT restructure or rewrite content.

        Uses placeholder-based protection to preserve images, slides, and
        page comments in their original positions during LLM processing.

        Args:
            extracted_text: Text extracted by pymupdf4llm/markitdown
            page_images: List of paths to page/slide images
            context: Context identifier for logging (e.g., document name)

        Returns:
            Cleaned markdown content (same content, cleaner format)
        """
        if not page_images:
            return extracted_text

        # Check persistent cache using page count + text fingerprint as key
        # Create a fingerprint from text + page image names for cache lookup
        page_names = "|".join(p.name for p in page_images[:10])  # First 10 page names
        cache_key = f"enhance_vision:{context}:{len(page_images)}"
        cache_content = f"{page_names}|{extracted_text[:1000]}"
        cached = self._persistent_cache.get(cache_key, cache_content, context=context)
        if cached is not None:
            logger.debug(
                f"[{_context_display_name(context)}] Persistent cache hit for enhance_document_with_vision"
            )
            # Fix malformed image refs even for cached content (handles old cache entries)
            return self._fix_malformed_image_refs(cached)

        # Extract and protect content before LLM processing
        protected = self.extract_protected_content(extracted_text)
        protected_content, mapping = self._protect_content(extracted_text)

        # Build message with text + images
        prompt = self._prompt_manager.get_prompt("document_enhance")

        # Prepare content parts
        content_parts: list[dict] = [
            {
                "type": "text",
                "text": f"{prompt}\n\n## Extracted Text:\n\n{protected_content}",
            },
        ]

        # Add page images (using cache to avoid repeated reads)
        for i, image_path in enumerate(page_images, 1):
            _, base64_image = self._get_cached_image(image_path)
            mime_type = get_mime_type(image_path.suffix)

            # Unique page label that won't conflict with document content
            content_parts.append(
                {
                    "type": "text",
                    "text": f"\n__MARKITAI_PAGE_LABEL_{i}__",
                }
            )
            content_parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
                }
            )

        response = await self._call_llm(
            model="default",
            messages=[{"role": "user", "content": content_parts}],
            context=context,
        )

        # Restore protected content from placeholders, with fallback for removed items
        result = self._unprotect_content(response.content, mapping, protected)

        # Fix malformed image references (e.g., extra closing parentheses)
        result = self._fix_malformed_image_refs(result)

        # Store in persistent cache
        self._persistent_cache.set(cache_key, cache_content, result, model="vision")

        return result

    async def enhance_document_complete(
        self,
        extracted_text: str,
        page_images: list[Path],
        source: str = "",
        max_pages_per_batch: int = DEFAULT_MAX_PAGES_PER_BATCH,
    ) -> tuple[str, str]:
        """
        Complete document enhancement: clean format + generate frontmatter.

        Architecture:
        - Single batch (pages <= max_pages_per_batch): Use Instructor for combined
          cleaning + frontmatter in one LLM call (saves one API call)
        - Multi batch (pages > max_pages_per_batch): Clean in batches, then
          generate frontmatter separately

        Args:
            extracted_text: Text extracted by pymupdf4llm/markitdown
            page_images: List of paths to page/slide images
            source: Source file name
            max_pages_per_batch: Max pages per batch (default 10)

        Returns:
            Tuple of (cleaned_markdown, frontmatter_yaml)
        """
        if not page_images:
            # No images, fall back to regular process_document
            return await self.process_document(extracted_text, source)

        # Single batch: use combined Instructor call (saves one API call)
        if len(page_images) <= max_pages_per_batch:
            logger.info(
                f"[{source}] Processing {len(page_images)} pages with combined call"
            )
            try:
                return await self._enhance_with_frontmatter(
                    extracted_text, page_images, source
                )
            except Exception as e:
                # Log succinct warning instead of full exception trace
                err_msg = str(e)
                if len(err_msg) > 200:
                    err_msg = err_msg[:200] + "..."
                logger.warning(
                    f"[{source}] Combined call failed: {type(e).__name__}: {err_msg}, "
                    "falling back to separate calls"
                )
                # Fallback to separate calls
                cleaned = await self.enhance_document_with_vision(
                    extracted_text, page_images, context=source
                )
                frontmatter = await self.generate_frontmatter(cleaned, source)
                return cleaned, frontmatter

        # Multi batch: clean in batches AND generate frontmatter in parallel
        # Frontmatter can be generated from original text while cleaning proceeds
        logger.info(
            f"[{source}] Processing {len(page_images)} pages in batches of "
            f"{max_pages_per_batch} (parallel frontmatter)"
        )

        # Launch cleaning and frontmatter generation concurrently
        clean_task = asyncio.create_task(
            self._enhance_document_batched_simple(
                extracted_text, page_images, max_pages_per_batch, source
            )
        )
        # Generate frontmatter from beginning of original text (first 5000 chars)
        frontmatter_task = asyncio.create_task(
            self.generate_frontmatter(extracted_text[:5000], source)
        )

        cleaned, frontmatter = await asyncio.gather(clean_task, frontmatter_task)

        return cleaned, frontmatter

    async def _enhance_with_frontmatter(
        self,
        extracted_text: str,
        page_images: list[Path],
        source: str,
    ) -> tuple[str, str]:
        """Enhance document with vision and generate frontmatter in one call.

        Uses Instructor for structured output.

        Args:
            extracted_text: Text to clean
            page_images: Page images for visual reference
            source: Source file name

        Returns:
            Tuple of (cleaned_markdown, frontmatter_yaml)
        """
        import time

        import yaml

        start_time = time.perf_counter()

        # Check persistent cache first
        # Use page count + source + text fingerprint as cache key
        page_names = "|".join(p.name for p in page_images[:10])  # First 10 page names
        cache_key = f"enhance_frontmatter:{source}:{len(page_images)}"
        cache_content = f"{page_names}|{extracted_text[:1000]}"
        cached = self._persistent_cache.get(cache_key, cache_content, context=source)
        if cached is not None:
            logger.debug(
                f"[{source}] Persistent cache hit for _enhance_with_frontmatter"
            )
            # Fix malformed image refs even for cached content (handles old cache entries)
            cleaned = self._fix_malformed_image_refs(cached.get("cleaned_markdown", ""))
            return cleaned, cached.get("frontmatter_yaml", "")

        # Extract protected content for fallback restoration
        protected = self.extract_protected_content(extracted_text)

        # Protect slide comments and images with placeholders before LLM processing
        protected_text, mapping = self._protect_content(extracted_text)

        # Get combined prompt
        prompt = self._prompt_manager.get_prompt(
            "document_enhance_complete",
            source=source,
        )

        # Build content parts
        content_parts: list[dict] = [
            {
                "type": "text",
                "text": f"{prompt}\n\n## Extracted Text:\n\n{protected_text}",
            },
        ]

        # Add page images
        for i, image_path in enumerate(page_images, 1):
            _, base64_image = self._get_cached_image(image_path)
            mime_type = get_mime_type(image_path.suffix)
            # Unique page label that won't conflict with document content
            content_parts.append(
                {"type": "text", "text": f"\n__MARKITAI_PAGE_LABEL_{i}__"}
            )
            content_parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
                }
            )

        async with self.semaphore:
            # Calculate dynamic max_tokens
            messages = [{"role": "user", "content": content_parts}]
            max_tokens = self._calculate_dynamic_max_tokens(messages)

            client = instructor.from_litellm(
                self.vision_router.acompletion, mode=instructor.Mode.JSON
            )
            # max_retries allows Instructor to retry with validation error
            # feedback, which helps LLM fix JSON escaping issues
            (
                response,
                raw_response,
            ) = await client.chat.completions.create_with_completion(
                model="default",
                messages=cast(
                    list[ChatCompletionMessageParam],
                    messages,
                ),
                response_model=EnhancedDocumentResult,
                max_retries=DEFAULT_INSTRUCTOR_MAX_RETRIES,
                max_tokens=max_tokens,
            )

            # Check for truncation
            if hasattr(raw_response, "choices") and raw_response.choices:
                finish_reason = getattr(raw_response.choices[0], "finish_reason", None)
                if finish_reason == "length":
                    raise ValueError("Output truncated due to max_tokens limit")

            # Track usage and log completion
            actual_model = getattr(raw_response, "model", None) or "default"
            input_tokens = 0
            output_tokens = 0
            cost = 0.0
            if hasattr(raw_response, "usage") and raw_response.usage is not None:
                input_tokens = getattr(raw_response.usage, "prompt_tokens", 0) or 0
                output_tokens = getattr(raw_response.usage, "completion_tokens", 0) or 0
                try:
                    cost = completion_cost(completion_response=raw_response)
                except Exception:
                    cost = 0.0
                self._track_usage(
                    actual_model, input_tokens, output_tokens, cost, source
                )

            # Log completion with timing
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            logger.info(
                f"[LLM:{source}] vision_enhance: {actual_model} "
                f"tokens={input_tokens}+{output_tokens} time={elapsed_ms}ms cost=${cost:.6f}"
            )

            # Build frontmatter YAML
            frontmatter_dict = {
                "title": response.frontmatter.title,
                "description": response.frontmatter.description,
                "tags": response.frontmatter.tags,
                "source": source,
            }
            frontmatter_yaml = yaml.dump(
                frontmatter_dict, allow_unicode=True, default_flow_style=False
            ).strip()

            # Restore protected content from placeholders
            # Pass protected dict for fallback restoration if LLM removed placeholders
            cleaned_markdown = self._unprotect_content(
                response.cleaned_markdown, mapping, protected
            )

            # Fix malformed image references (e.g., extra closing parentheses)
            cleaned_markdown = self._fix_malformed_image_refs(cleaned_markdown)

            # Store in persistent cache
            cache_value = {
                "cleaned_markdown": cleaned_markdown,
                "frontmatter_yaml": frontmatter_yaml,
            }
            self._persistent_cache.set(
                cache_key, cache_content, cache_value, model="vision"
            )

            return cleaned_markdown, frontmatter_yaml

    @staticmethod
    def _split_text_by_pages(text: str, num_pages: int) -> list[str]:
        """Split text into chunks corresponding to page ranges.

        Split strategy (in priority order):
        1. Remove trailing page image reference section first
        2. Use <!-- Slide number: N --> markers (PPTX/PPT)
        3. Use <!-- Page number: N --> markers (PDF)
        4. Fallback: split by paragraphs proportionally

        Args:
            text: Full document text
            num_pages: Number of pages/images

        Returns:
            List of text chunks, one per page
        """
        import re

        # Step 1: Remove trailing page image reference section
        # These are screenshot references at the end, not content separators
        ref_marker = "<!-- Page images for reference -->"
        ref_idx = text.find(ref_marker)
        if ref_idx != -1:
            main_content = text[:ref_idx].rstrip()
        else:
            main_content = text

        # Step 2: Try slide markers (PPTX/PPT)
        slide_pattern = r"<!-- Slide number: (\d+) -->"
        slide_markers = list(re.finditer(slide_pattern, main_content))

        if len(slide_markers) >= num_pages:
            # Use slide markers to split - each chunk starts with its slide marker
            chunks = []
            for i in range(num_pages):
                start = slide_markers[i].start()
                if i + 1 < len(slide_markers):
                    end = slide_markers[i + 1].start()
                else:
                    end = len(main_content)
                chunks.append(main_content[start:end].strip())
            return chunks

        # Step 3: Try page markers (PDF)
        page_pattern = r"<!-- Page number: (\d+) -->"
        page_markers = list(re.finditer(page_pattern, main_content))

        if len(page_markers) >= num_pages:
            # Use page markers to split - each chunk starts with its page marker
            chunks = []
            for i in range(num_pages):
                start = page_markers[i].start()
                if i + 1 < len(page_markers):
                    end = page_markers[i + 1].start()
                else:
                    end = len(main_content)
                chunks.append(main_content[start:end].strip())
            return chunks

        # Step 4: Fallback - split by paragraphs proportionally
        paragraphs = main_content.split("\n\n")
        if len(paragraphs) < num_pages:
            # Very short text, just return whole text for each page
            return [main_content] * num_pages

        paragraphs_per_page = len(paragraphs) // num_pages
        chunks = []
        for i in range(num_pages):
            start_idx = i * paragraphs_per_page
            if i == num_pages - 1:
                # Last chunk gets remaining paragraphs
                end_idx = len(paragraphs)
            else:
                end_idx = start_idx + paragraphs_per_page
            chunks.append("\n\n".join(paragraphs[start_idx:end_idx]))

        return chunks

    async def _enhance_document_batched_simple(
        self,
        extracted_text: str,
        page_images: list[Path],
        batch_size: int,
        source: str = "",
    ) -> str:
        """Process long documents in batches - vision cleaning only.

        All batches use the same method for consistent output format.

        Args:
            extracted_text: Full document text
            page_images: All page images
            batch_size: Pages per batch
            source: Source file name

        Returns:
            Merged cleaned content
        """
        num_pages = len(page_images)
        num_batches = (num_pages + batch_size - 1) // batch_size

        # Split text by pages
        page_texts = self._split_text_by_pages(extracted_text, num_pages)

        cleaned_parts = []

        for batch_num in range(num_batches):
            batch_start = batch_num * batch_size
            batch_end = min(batch_start + batch_size, num_pages)

            # Get text and images for this batch
            batch_texts = page_texts[batch_start:batch_end]
            batch_images = page_images[batch_start:batch_end]
            batch_text = "\n\n".join(batch_texts)

            logger.info(
                f"[{source}] Batch {batch_num + 1}/{num_batches}: "
                f"pages {batch_start + 1}-{batch_end}"
            )

            # All batches: clean only (no frontmatter)
            # Use source as context (not batch-specific) so all usage aggregates to same context
            batch_cleaned = await self.enhance_document_with_vision(
                batch_text, batch_images, context=source
            )

            cleaned_parts.append(batch_cleaned)

        # Merge all batches
        return "\n\n".join(cleaned_parts)

    async def process_document(
        self,
        markdown: str,
        source: str,
    ) -> tuple[str, str]:
        """
        Process a document with LLM: clean and generate frontmatter.

        Uses placeholder-based protection to preserve images, slides, and
        page comments in their original positions during LLM processing.

        Uses a combined prompt with Instructor for structured output,
        falling back to parallel separate calls if structured output fails.

        Args:
            markdown: Raw markdown content
            source: Source file name

        Returns:
            Tuple of (cleaned_markdown, frontmatter_yaml)
        """
        # Extract and protect content before LLM processing
        protected = self.extract_protected_content(markdown)
        protected_content, mapping = self._protect_content(markdown)

        # Try combined approach with Instructor first
        try:
            result = await self._process_document_combined(protected_content, source)

            # Restore protected content from placeholders, with fallback
            cleaned = self._unprotect_content(
                result.cleaned_markdown, mapping, protected
            )

            # Convert Frontmatter to YAML string
            import yaml

            frontmatter_dict = {
                "title": result.frontmatter.title,
                "description": result.frontmatter.description,
                "tags": result.frontmatter.tags,
                "source": source,
            }
            frontmatter_yaml = yaml.dump(
                frontmatter_dict, allow_unicode=True, default_flow_style=False
            ).strip()
            logger.debug(f"[{source}] Used combined document processing")
            return cleaned, frontmatter_yaml
        except Exception as e:
            logger.debug(
                f"[{source}] Combined processing failed: {e}, using parallel fallback"
            )

        # Fallback: Run cleaning and frontmatter generation in parallel
        # clean_markdown uses its own protection mechanism
        clean_task = asyncio.create_task(self.clean_markdown(markdown, context=source))
        frontmatter_task = asyncio.create_task(
            self.generate_frontmatter(markdown, source)
        )

        cleaned_result, frontmatter_result = await asyncio.gather(
            clean_task, frontmatter_task, return_exceptions=True
        )

        cleaned: str = (
            markdown if isinstance(cleaned_result, BaseException) else cleaned_result
        )
        if isinstance(cleaned_result, BaseException):
            logger.warning(f"Markdown cleaning failed: {cleaned_result}")

        frontmatter: str = (
            f"title: {source}\nsource: {source}"
            if isinstance(frontmatter_result, BaseException)
            else frontmatter_result
        )
        if isinstance(frontmatter_result, BaseException):
            logger.warning(f"Frontmatter generation failed: {frontmatter_result}")

        return cleaned, frontmatter

    async def _process_document_combined(
        self,
        markdown: str,
        source: str,
    ) -> DocumentProcessResult:
        """
        Process document with combined cleaner + frontmatter using Instructor.

        Cache lookup order:
        1. In-memory cache (session-level, fast)
        2. Persistent cache (cross-session, SQLite)
        3. LLM API call

        Args:
            markdown: Raw markdown content
            source: Source file name

        Returns:
            DocumentProcessResult with cleaned markdown and frontmatter
        """
        cache_key = f"document_process:{source}"

        # Helper to reconstruct DocumentProcessResult from cached dict
        def _from_cache(cached: dict) -> DocumentProcessResult:
            return DocumentProcessResult(
                cleaned_markdown=cached.get("cleaned_markdown", ""),
                frontmatter=Frontmatter(
                    title=cached.get("title", source),
                    description=cached.get("description", ""),
                    tags=cached.get("tags", []),
                ),
            )

        # 1. Check in-memory cache first (fastest)
        cached = self._cache.get(cache_key, markdown)
        if cached is not None:
            self._cache_hits += 1
            logger.debug(f"[{source}] Memory cache hit for _process_document_combined")
            return _from_cache(cached)

        # 2. Check persistent cache (cross-session)
        cached = self._persistent_cache.get(cache_key, markdown, context=source)
        if cached is not None:
            self._cache_hits += 1
            logger.debug(
                f"[{source}] Persistent cache hit for _process_document_combined"
            )
            # Also populate in-memory cache for faster subsequent access
            self._cache.set(cache_key, markdown, cached)
            return _from_cache(cached)

        self._cache_misses += 1

        # Detect document language
        language = get_language_name(detect_language(markdown))

        # Truncate content if needed (with warning)
        original_len = len(markdown)
        truncated_content = self._smart_truncate(markdown, DEFAULT_MAX_CONTENT_CHARS)
        if len(truncated_content) < original_len:
            logger.warning(
                f"[LLM:{source}] Content truncated: {original_len} -> {len(truncated_content)} chars "
                f"(limit: {DEFAULT_MAX_CONTENT_CHARS}). Some content may be lost."
            )

        # Get combined prompt with language
        prompt = self._prompt_manager.get_prompt(
            "document_process",
            content=truncated_content,
            source=source,
            language=language,
        )

        async with self.semaphore:
            start_time = time.perf_counter()

            # Calculate dynamic max_tokens
            messages = cast(
                list[ChatCompletionMessageParam],
                [{"role": "user", "content": prompt}],
            )
            max_tokens = self._calculate_dynamic_max_tokens(messages)

            # Create instructor client from router for load balancing
            client = instructor.from_litellm(
                self.router.acompletion, mode=instructor.Mode.JSON
            )

            # Use create_with_completion to get both the model and the raw response
            # Use logical model name for router load balancing
            # max_retries allows Instructor to retry with validation error
            # feedback, which helps LLM fix JSON escaping issues
            (
                response,
                raw_response,
            ) = await client.chat.completions.create_with_completion(
                model="default",
                messages=messages,
                response_model=DocumentProcessResult,
                max_retries=DEFAULT_INSTRUCTOR_MAX_RETRIES,
                max_tokens=max_tokens,
            )

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            # Check for truncation
            if hasattr(raw_response, "choices") and raw_response.choices:
                finish_reason = getattr(raw_response.choices[0], "finish_reason", None)
                if finish_reason == "length":
                    raise ValueError("Output truncated due to max_tokens limit")

            # Track usage from raw API response
            # Get actual model from response for accurate tracking
            actual_model = getattr(raw_response, "model", None) or "default"
            input_tokens = 0
            output_tokens = 0
            cost = 0.0
            if hasattr(raw_response, "usage") and raw_response.usage is not None:
                input_tokens = getattr(raw_response.usage, "prompt_tokens", 0) or 0
                output_tokens = getattr(raw_response.usage, "completion_tokens", 0) or 0
                try:
                    cost = completion_cost(completion_response=raw_response)
                except Exception:
                    cost = 0.0
                self._track_usage(
                    actual_model, input_tokens, output_tokens, cost, source
                )

            # Log detailed timing for performance analysis
            logger.info(
                f"[LLM:{source}] document_process: {actual_model} "
                f"tokens={input_tokens}+{output_tokens} "
                f"time={elapsed_ms:.0f}ms cost=${cost:.6f}"
            )

            # Store in both cache layers
            cache_value = {
                "cleaned_markdown": response.cleaned_markdown,
                "title": response.frontmatter.title,
                "description": response.frontmatter.description,
                "tags": response.frontmatter.tags,
            }
            self._cache.set(cache_key, markdown, cache_value)
            self._persistent_cache.set(
                cache_key, markdown, cache_value, model="default"
            )

            return response

    def format_llm_output(
        self,
        markdown: str,
        frontmatter: str,
    ) -> str:
        """
        Format final output with frontmatter.

        Args:
            markdown: Cleaned markdown content
            frontmatter: YAML frontmatter (without --- markers)

        Returns:
            Complete markdown with frontmatter
        """
        from datetime import datetime

        import yaml

        from markitai.workflow.helpers import normalize_frontmatter

        frontmatter = self._clean_frontmatter(frontmatter)

        # Parse frontmatter to dict, add timestamp, then normalize
        try:
            frontmatter_dict = yaml.safe_load(frontmatter) or {}
        except yaml.YAMLError:
            frontmatter_dict = {}

        # Add markitai_processed timestamp (use local time)
        timestamp = datetime.now().astimezone().isoformat()
        frontmatter_dict["markitai_processed"] = timestamp

        # Normalize to ensure consistent field order
        frontmatter = normalize_frontmatter(frontmatter_dict)

        # Remove non-commented screenshot references that shouldn't be in content
        # These are page screenshots that should only appear as comments at the end
        # Pattern: ![Page N](screenshots/...) or ![Page N](path/screenshots/...)
        # But NOT: <!-- ![Page N](screenshots/...) --> (already commented)
        markdown = self._remove_uncommented_screenshots(markdown)

        from markitai.utils.text import (
            clean_ppt_headers_footers,
            clean_residual_placeholders,
            fix_broken_markdown_links,
            normalize_markdown_whitespace,
        )

        # Post-processing: fix broken links and clean PPT headers/footers
        # These are fallback cleanups when LLM doesn't fully follow instructions
        markdown = fix_broken_markdown_links(markdown)
        markdown = clean_ppt_headers_footers(markdown)
        markdown = clean_residual_placeholders(markdown)
        markdown = normalize_markdown_whitespace(markdown)
        return f"---\n{frontmatter}\n---\n\n{markdown}"

    @staticmethod
    def _remove_uncommented_screenshots(content: str) -> str:
        """Remove non-commented page screenshot references from content.

        Page screenshots should only appear as HTML comments at the end of the document.
        If LLM accidentally outputs them as regular image references, remove them.

        Also ensures that any screenshot references in the "Page images for reference"
        section are properly commented.

        Args:
            content: Markdown content

        Returns:
            Content with uncommented screenshots removed/fixed
        """
        import re

        # Find the position of "<!-- Page images for reference -->" if it exists
        page_images_header = "<!-- Page images for reference -->"
        header_pos = content.find(page_images_header)

        if header_pos == -1:
            # No page images section, just remove any stray screenshot references
            # IMPORTANT: Only match markitai-generated screenshot patterns to avoid
            # removing user's original screenshots/ references (P0-5 fix).
            # markitai naming format: {filename}.page{NNNN}.{ext} in screenshots/
            # Patterns to remove:
            # 1. ![Page N](screenshots/*.page*.jpg) - markitai standard pattern
            # 2. ![...](screenshots/*.page*.jpg) - LLM-generated variants with same filename
            patterns = [
                # Matches: ![Page N](screenshots/anything.pageNNNN.jpg)
                r"^!\[Page\s+\d+\]\(screenshots/[^)]+\.page\d{4}\.\w+\)\s*$",
                # Matches: ![...](screenshots/anything.pageNNNN.jpg)
                r"^!\[[^\]]*\]\(screenshots/[^)]+\.page\d{4}\.\w+\)\s*$",
            ]
            for pattern in patterns:
                content = re.sub(pattern, "", content, flags=re.MULTILINE)

            # Also remove any page/image labels that LLM may have copied
            # Pattern: ## or ### Page N Image: followed by empty line (legacy format)
            # Pattern: [Page N] or [Image N] on its own line (simple format)
            # Pattern: __MARKITAI_PAGE_LABEL_N__ or __MARKITAI_IMG_LABEL_N__ (unique format)
            content = re.sub(
                r"^#{2,3}\s+Page\s+\d+\s+Image:\s*\n\s*\n",
                "",
                content,
                flags=re.MULTILINE,
            )
            content = re.sub(
                r"^\[(Page|Image)\s+\d+\]\s*\n",
                "",
                content,
                flags=re.MULTILINE,
            )
            content = re.sub(
                r"^__MARKITAI_(PAGE|IMG)_LABEL_\d+__\s*\n",
                "",
                content,
                flags=re.MULTILINE,
            )
            # Remove any leftover slide placeholders (shouldn't exist but cleanup)
            content = re.sub(
                r"^__MARKITAI_SLIDE_\d+__\s*\n",
                "",
                content,
                flags=re.MULTILINE,
            )

            # Clean up any resulting empty lines
            content = re.sub(r"\n{3,}", "\n\n", content)
        else:
            # Split at the page images section
            before = content[:header_pos]
            after = content[header_pos:]

            # Remove screenshot references from BEFORE the page images header
            # IMPORTANT: Only match markitai-generated screenshot patterns (P0-5 fix)
            patterns = [
                # Matches: ![Page N](screenshots/anything.pageNNNN.jpg)
                r"^!\[Page\s+\d+\]\(screenshots/[^)]+\.page\d{4}\.\w+\)\s*$",
                # Matches: ![...](screenshots/anything.pageNNNN.jpg)
                r"^!\[[^\]]*\]\(screenshots/[^)]+\.page\d{4}\.\w+\)\s*$",
            ]
            for pattern in patterns:
                before = re.sub(pattern, "", before, flags=re.MULTILINE)

            # Also remove any page/image labels that LLM may have copied
            before = re.sub(
                r"^#{2,3}\s+Page\s+\d+\s+Image:\s*\n\s*\n",
                "",
                before,
                flags=re.MULTILINE,
            )
            before = re.sub(
                r"^\[(Page|Image)\s+\d+\]\s*\n",
                "",
                before,
                flags=re.MULTILINE,
            )
            before = re.sub(
                r"^__MARKITAI_(PAGE|IMG)_LABEL_\d+__\s*\n",
                "",
                before,
                flags=re.MULTILINE,
            )
            # Remove any leftover slide placeholders (shouldn't exist but cleanup)
            before = re.sub(
                r"^__MARKITAI_SLIDE_\d+__\s*\n",
                "",
                before,
                flags=re.MULTILINE,
            )
            before = re.sub(r"\n{3,}", "\n\n", before)

            # Fix the AFTER section: convert any non-commented page images to comments
            # Match lines with page image references that are not already commented
            # This handles: ![Page N](screenshots/...)
            after_lines = after.split("\n")
            fixed_lines = []
            for line in after_lines:
                stripped = line.strip()
                # Check if it's an uncommented page image reference
                if (
                    stripped.startswith("![Page")
                    and "screenshots/" in stripped
                    and not stripped.startswith("<!--")
                ):
                    fixed_lines.append(f"<!-- {stripped} -->")
                else:
                    fixed_lines.append(line)
            after = "\n".join(fixed_lines)

            content = before + after

        # Clean up screenshot comments section: remove blank lines between comments
        # Pattern: <!-- Page images for reference --> followed by page image comments
        page_section_pattern = (
            r"(<!-- Page images for reference -->)"
            r"((?:\s*<!-- !\[Page \d+\]\([^)]+\) -->)+)"
        )

        def clean_page_section(match: re.Match) -> str:
            header = match.group(1)
            comments_section = match.group(2)
            # Extract individual comments and rejoin without blank lines
            comments = re.findall(r"<!-- !\[Page \d+\]\([^)]+\) -->", comments_section)
            return header + "\n" + "\n".join(comments)

        content = re.sub(page_section_pattern, clean_page_section, content)

        return content

    @staticmethod
    def _clean_frontmatter(frontmatter: str) -> str:
        """
        Clean frontmatter by removing code block markers and --- markers.

        Args:
            frontmatter: Raw frontmatter from LLM

        Returns:
            Clean YAML frontmatter
        """
        import re

        frontmatter = frontmatter.strip()

        # Remove code block markers (```yaml, ```yml, ```)
        # Pattern: ```yaml or ```yml at start, ``` at end
        code_block_pattern = r"^```(?:ya?ml)?\s*\n?(.*?)\n?```$"
        match = re.match(code_block_pattern, frontmatter, re.DOTALL | re.IGNORECASE)
        if match:
            frontmatter = match.group(1).strip()

        # Remove --- markers
        if frontmatter.startswith("---"):
            frontmatter = frontmatter[3:].strip()
        if frontmatter.endswith("---"):
            frontmatter = frontmatter[:-3].strip()

        return frontmatter
