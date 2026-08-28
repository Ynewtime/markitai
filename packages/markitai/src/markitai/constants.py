"""Centralized constants for markitai.

This module contains all hardcoded constants used throughout the codebase.
Grouping them here makes it easier to:
- Find and modify default values
- Understand system limits at a glance
- Maintain consistency across modules
"""

from __future__ import annotations

import re

# =============================================================================
# File Size Limits
# =============================================================================

MAX_STATE_FILE_SIZE = 10 * 1024 * 1024  # 10 MB - batch state file
MAX_IMAGE_SIZE = 100 * 1024 * 1024  # 100 MB - single image
MAX_TOTAL_IMAGES_SIZE = 500 * 1024 * 1024  # 500 MB - all images combined
MAX_DOCUMENT_SIZE = 500 * 1024 * 1024  # 500 MB - input document

# =============================================================================
# LLM Processing
# =============================================================================

# Retry settings
DEFAULT_MAX_RETRIES = 2
DEFAULT_RETRY_BASE_DELAY = 1.0  # seconds
DEFAULT_RETRY_MAX_DELAY = 60.0  # seconds

# Instructor retry settings (for structured JSON output validation)
# When LLM returns malformed JSON, Instructor can retry with validation error
# feedback, allowing the LLM to fix issues like incorrect escaping.
# Common issues: unescaped quotes in CJK text (e.g., Chinese quoted phrases)
# Set to 2: enough for LLM to self-correct common JSON issues
DEFAULT_INSTRUCTOR_MAX_RETRIES = 2

# Token limits
DEFAULT_MAX_OUTPUT_TOKENS = 8192  # Conservative default for most models
DEFAULT_MAX_CONTENT_CHARS = (
    32000  # Max chars for document processing (truncation threshold)
)

# Concurrency
DEFAULT_IO_CONCURRENCY = 20  # I/O operations (file reads, etc.)
DEFAULT_LLM_CONCURRENCY = 10  # LLM API calls (config default)
DEFAULT_BATCH_CONCURRENCY = 10  # Batch file processing (config default)
DEFAULT_URL_CONCURRENCY = 5  # URL fetching (separate from file processing)
DEFAULT_HEAVY_TASK_LIMIT = 0  # Heavy task semaphore limit (0 = auto-detect from RAM)

# Batch sizes
DEFAULT_MAX_IMAGES_PER_BATCH = 10  # Images per LLM vision call
DEFAULT_MAX_PAGES_PER_BATCH = 10  # Pages per LLM call for document processing

# Router settings
DEFAULT_ROUTER_NUM_RETRIES = 2
DEFAULT_ROUTER_TIMEOUT = 120  # seconds
# Per-document LLM request circuit breaker. 50 covers a ~400-page document
# (40 vision batches of 10 pages + frontmatter) plus a healthy retry
# allowance; anything beyond that is far more often a retry storm
# (transport retries x instructor retries x business fallbacks) than a
# legitimate workload. Raise via llm.max_requests_per_document (0 disables).
DEFAULT_MAX_REQUESTS_PER_DOCUMENT = 50
DEFAULT_SQLITE_TIMEOUT = 30.0  # seconds — SQLite connection timeout
DEFAULT_MAX_OUTPUT_TOKENS_HARD_CAP = 128000  # Absolute ceiling for max_tokens
DEFAULT_SUBPROCESS_TIMEOUT = 30  # seconds — LibreOffice/external tool timeout
DEFAULT_LLM_READY_TIMEOUT = 300.0  # seconds — wait for LLM provider readiness
DEFAULT_VISION_MAX_DIMENSION = 2048  # pixels — max dimension for vision input images

# Note: RETRYABLE_ERRORS tuple is defined in markitai.llm.engine as it
# contains actual exception classes from litellm that cannot be imported here

# =============================================================================
# Page markers
# =============================================================================

# Every path that splits a document into pages writes this marker, and every
# later stage — LLM enhancement, page/image alignment, output profiles — finds
# pages by reading it back. Both halves live here because they drifted apart
# once already: screenshot-only mode emitted "<!-- Page 3 -->", which no
# reader matched, so those pages were invisible to all of them.

PAGE_MARKER_RE = re.compile(r"<!--\s*Page number:\s*(\d+)\s*-->")


def page_marker(page_number: int) -> str:
    """The marker introducing page ``page_number``'s content."""
    return f"<!-- Page number: {page_number} -->"


# =============================================================================
# Image Processing
# =============================================================================

DEFAULT_IMAGE_QUALITY = 75  # JPEG quality (1-100)
DEFAULT_RENDER_DPI = 150  # DPI for page screenshots (PDF, PPTX, etc.)
DEFAULT_IMAGE_IO_CONCURRENCY = 8  # Concurrent I/O for image saving (optimized for NVMe)
DEFAULT_IMAGE_MULTIPROCESS_THRESHOLD = (
    10  # Use multiprocess compression when images > this
)
DEFAULT_IMAGE_MAX_WIDTH = 1920
DEFAULT_IMAGE_MAX_HEIGHT = 99999  # Effectively unlimited (thumbnail won't upscale)

# Image filter thresholds
DEFAULT_IMAGE_FILTER_MIN_WIDTH = 50
DEFAULT_IMAGE_FILTER_MIN_HEIGHT = 50
DEFAULT_IMAGE_FILTER_MIN_AREA = 5000

# =============================================================================
# Cache Settings
# =============================================================================

# In-memory cache (legacy, still used for image bytes cache)
DEFAULT_CACHE_MAXSIZE = 100  # Max entries in LLM content cache
DEFAULT_CACHE_TTL_SECONDS = 300  # Cache TTL (5 minutes)

# Persistent SQLite cache
DEFAULT_CACHE_SIZE_LIMIT = 512 * 1024 * 1024  # 512 MB per cache file
DEFAULT_GLOBAL_CACHE_DIR = "~/.markitai"  # Global cache directory
DEFAULT_CACHE_DB_FILENAME = "cache.db"  # SQLite database filename
DEFAULT_CACHE_CONTENT_TRUNCATE = 50000  # Truncate content for hash key (chars)

# =============================================================================
# Batch Processing
# =============================================================================

DEFAULT_STATE_FLUSH_INTERVAL_SECONDS = 10  # Increased to reduce I/O overhead
DEFAULT_SCAN_MAX_DEPTH = 5
DEFAULT_SCAN_MAX_FILES = 10000

# =============================================================================
# Logging
# =============================================================================

DEFAULT_LOG_ROTATION = "10 MB"
DEFAULT_LOG_RETENTION = "7 days"

# =============================================================================
# UI / Display
# =============================================================================

DEFAULT_JSON_INDENT = 2  # JSON output indentation

# =============================================================================
# Paths and Filenames
# =============================================================================

DEFAULT_OUTPUT_DIR = None
DEFAULT_PROMPTS_DIR = "~/.markitai/prompts"
DEFAULT_LOG_DIR = None
CONFIG_FILENAME = "markitai.json"

# =============================================================================
# OCR
# =============================================================================

DEFAULT_OCR_LANG = "en"
DEFAULT_OCR_SAMPLE_PAGES = 3  # Pages to sample for scanned PDF detection

# =============================================================================
# Misc Defaults
# =============================================================================

DEFAULT_MODEL_WEIGHT = 1  # Default model weight in router
DEFAULT_SCREENSHOT_MAX_BYTES = int(
    3.5 * 1024 * 1024
)  # 3.5 MB max (base64 adds ~33%, must stay under 5MB API limit)

# URL Screenshot settings
DEFAULT_SCREENSHOT_VIEWPORT_WIDTH = 1920
DEFAULT_SCREENSHOT_VIEWPORT_HEIGHT = 1080
DEFAULT_SCREENSHOT_QUALITY = 75  # JPEG quality (1-100)
DEFAULT_SCREENSHOT_MAX_HEIGHT = 10000  # Max height for full-page URL screenshots
DEFAULT_SCREENSHOT_TILE_HEIGHT = 2000  # Per-tile height when tiling long screenshots
DEFAULT_ROUTING_STRATEGY = "simple-shuffle"
DEFAULT_IMAGE_FORMAT = "jpeg"
DEFAULT_ON_CONFLICT = "rename"
DEFAULT_LOG_LEVEL = "INFO"

# =============================================================================
# URL Fetch Settings
# =============================================================================

DEFAULT_FETCH_STRATEGY = "auto"  # auto | static | defuddle | playwright | jina
DEFAULT_PLAYWRIGHT_TIMEOUT = 30000  # ms
DEFAULT_PLAYWRIGHT_WAIT_FOR = (
    "domcontentloaded"  # load | domcontentloaded | networkidle
)
DEFAULT_PLAYWRIGHT_EXTRA_WAIT_MS = (
    3000  # Extra wait after load state (for JS rendering)
)

# Playwright auto-scroll settings (inspired by baoyu-skills url-to-markdown)
DEFAULT_PLAYWRIGHT_AUTO_SCROLL_STEPS = 8  # Max scroll iterations
DEFAULT_PLAYWRIGHT_AUTO_SCROLL_DELAY_MS = 600  # Delay between scroll steps
DEFAULT_PLAYWRIGHT_POST_SCROLL_DELAY_MS = 800  # Wait after scrolling completes

# DOM noise selectors to remove before content extraction
# Based on baoyu-skills url-to-markdown's proven selector set
DOM_NOISE_SELECTORS: tuple[str, ...] = (
    "script",
    "style",
    "noscript",
    "iframe",
    "svg",
    "canvas",
    "header nav",
    "footer",
    ".sidebar",
    ".nav",
    ".navigation",
    ".advertisement",
    ".ad",
    ".ads",
    ".cookie-banner",
    ".popup",
    '[role="banner"]',
    '[role="navigation"]',
    '[role="complementary"]',
)

# HTML attributes to remove from DOM (event handlers, inline styles)
DOM_NOISE_ATTRIBUTES: tuple[str, ...] = (
    "style",
    "onclick",
    "onload",
    "onerror",
    "onmouseover",
    "onmouseout",
)

# Site-specific noise selectors injected into Playwright DOM cleanup
# based on the URL domain. These target site chrome that the generic
# selectors cannot match.
_X_COM_NOISE_SELECTORS: tuple[str, ...] = (
    '[data-testid="sidebarColumn"]',
    '[data-testid="DMDrawer"]',
    '[data-testid="sheetDialog"]',
    '[data-testid="bottomBar"]',
    '[data-testid="placementTracking"]',
    '[aria-label="Sign up"]',
    '[aria-label="Footer"]',
)

SITE_NOISE_SELECTORS: dict[str, tuple[str, ...]] = {
    "x.com": _X_COM_NOISE_SELECTORS,
    "twitter.com": _X_COM_NOISE_SELECTORS,
}

DEFAULT_JINA_TIMEOUT = 30  # seconds
DEFAULT_JINA_RPM = 20  # Jina free tier: 20 requests per minute
DEFAULT_JINA_BASE_URL = "https://r.jina.ai"

# Defuddle: free content extraction API (https://defuddle.md)
# Returns clean Markdown with YAML frontmatter (title, author, published, etc.)
# NOTE: Rate limit is undocumented — using conservative default. Adjust if needed.
# The offline path this used to be a TODO for exists: markitai.webextract is a
# port of defuddle (see webextract/PORT_MANIFEST.md) and backs the default
# `auto`/`static` strategies. `-s defuddle` stays as an explicit opt-in to the
# hosted API for pages the local pipeline handles poorly.
DEFAULT_DEFUDDLE_TIMEOUT = 30  # seconds
DEFAULT_DEFUDDLE_RPM = 20  # Conservative default (actual limit undocumented)
DEFAULT_DEFUDDLE_BASE_URL = "https://defuddle.md"

# Domains that typically require JavaScript rendering
DEFAULT_FETCH_FALLBACK_PATTERNS: tuple[str, ...] = (
    "twitter.com",
    "x.com",
    "instagram.com",
    "facebook.com",
    "linkedin.com",
    "threads.net",
)

# Patterns that indicate JavaScript is required (simple string matching)
# These patterns are checked against Markdown content (after conversion)
JS_REQUIRED_PATTERNS: tuple[str, ...] = (
    "JavaScript is disabled",
    "JavaScript is required",
    "Please enable JavaScript",
    "This page requires JavaScript",
    "You need to enable JavaScript",
    "enable javascript",
    "requires javascript",
)

# Fetch strategy categories
# Order matters: this tuple IS the default auto-chain order (local-first —
# static's native webextract now matches remote defuddle quality on the
# ground-truth corpus, and remote strategies are consent-gated)
ALL_FETCH_STRATEGIES: tuple[str, ...] = (
    "static",
    "playwright",
    "defuddle",
    "jina",
    "cloudflare",
)
EXTERNAL_STRATEGIES: tuple[str, ...] = ("defuddle", "jina", "cloudflare")
LOCAL_STRATEGIES: tuple[str, ...] = ("static", "playwright")

# =============================================================================
# Local LLM Provider Settings
# =============================================================================

# Claude Code supported aliases (for validation and dynamic lookup)
# Actual model resolution is done dynamically via LiteLLM's model database
# to automatically pick up new model versions without code changes.
# Source: https://code.claude.com/docs/en/model-config
CLAUDE_CODE_ALIASES: tuple[str, ...] = ("haiku", "sonnet", "opus", "inherit")

# Where each API-key provider's credential is read from. Four copies of this
# lived in the codebase (credential detection, the setup wizard, serve's
# key check, and discovery's card list) and had already drifted — the
# wizard's copy was missing OpenRouter entirely.
#
# Subscription providers (claude-agent, copilot, chatgpt) are absent on
# purpose: they authenticate through their CLI or OAuth, not an env var.
PROVIDER_API_KEY_ENV: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


# The model each provider defaults to when markitai picks one for the user:
# credential auto-detection, the `init` wizard, `serve`'s startup candidates
# and the sample config all read this table. It is deliberately the only
# copy — four hand-written copies existed until 2026-08 and had already
# drifted apart (three still named the previous generation).
#
# Picks follow three rules:
#   - the cheap/fast tier (markitai converts in bulk, and the LLM stage is
#     mostly metadata),
#   - a provider-managed alias over a dated id wherever one exists, so a
#     vendor's next release does not strand this table,
#   - nothing an ordinary account cannot reach: limited-preview deployments
#     (gpt-5.6-luna) may be configured by hand but are never chosen here.
PROVIDER_DEFAULT_MODELS: dict[str, str] = {
    # Subscription / CLI providers (aliases; the CLI resolves the generation)
    "claude-agent": "claude-agent/sonnet",
    "copilot": "copilot/claude-sonnet-4.6",
    "chatgpt": "chatgpt/gpt-5.6",
    # API-key providers
    "anthropic": "anthropic/claude-sonnet-5",
    "openai": "openai/gpt-5.6-luna",
    "gemini": "gemini/gemini-flash-lite-latest",
    "deepseek": "deepseek/deepseek-v4-flash",
    "openrouter": "openrouter/google/gemini-3.1-flash-lite",
}


# Default model info when LiteLLM lookup fails
# Used for local providers (claude-agent/, copilot/) as fallback
# Note: Conservative defaults. Latest models (Opus 4.6, Sonnet 4.6) support
# 1M context (GA, 2026-03-13) and up to 128k output tokens.
LOCAL_PROVIDER_DEFAULT_MODEL_INFO: dict[str, int | bool] = {
    "max_input_tokens": 200000,
    "max_output_tokens": 64000,
    "supports_vision": True,
}

# =============================================================================
# MIME Type Mappings
# =============================================================================

# Extension to MIME type mapping (for encoding images to send to LLM APIs)
EXTENSION_TO_MIME: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
}

# MIME type to extension mapping (for decoding content-type headers)
MIME_TO_EXTENSION: dict[str, str] = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "image/svg+xml": ".svg",
    "image/bmp": ".bmp",
    "image/x-icon": ".ico",
    "image/vnd.microsoft.icon": ".ico",
}

# Supported image extensions for standalone image detection
# HEIC/HEIF/AVIF require the optional pillow-heif dependency (markitai[heif]);
# they are transcoded to PNG at the converter boundary.
IMAGE_EXTENSIONS: tuple[str, ...] = (
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".gif",
    ".bmp",
    ".tiff",
    ".tif",
    ".svg",
    ".heic",
    ".heif",
    ".avif",
)

# Metadata directory namespace — isolates markitai metadata (assets, screenshots,
# reports, states) from user content to prevent collisions with input directories.
MARKITAI_META_DIR = ".markitai"
ASSETS_REL_PATH = f"{MARKITAI_META_DIR}/assets"
SCREENSHOTS_REL_PATH = f"{MARKITAI_META_DIR}/screenshots"
REPORTS_REL_PATH = f"{MARKITAI_META_DIR}/reports"
STATES_REL_PATH = f"{MARKITAI_META_DIR}/states"

# Visible assets directory used by output profiles that relocate images out of
# the hidden metadata namespace (rag/obsidian) so ingestors that skip hidden
# paths (e.g. LlamaIndex SimpleDirectoryReader) still see them.
VISIBLE_ASSETS_REL_PATH = "assets"
