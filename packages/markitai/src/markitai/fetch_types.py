"""Shared types, exceptions, and data classes for the fetch subsystem.

Extracted from fetch.py to break circular dependencies and enable
clean imports from fetch_cache.py and future fetch_strategies.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class FetchStrategy(Enum):
    """URL fetch strategy."""

    AUTO = "auto"
    STATIC = "static"
    DEFUDDLE = "defuddle"  # Defuddle content extraction API
    PLAYWRIGHT = "playwright"  # Playwright Python (recommended)
    CLOUDFLARE = "cloudflare"  # CF Browser Rendering /markdown API
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
