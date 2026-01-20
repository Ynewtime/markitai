"""Centralized constants for markit.

This module contains all hardcoded constants used throughout the codebase.
Grouping them here makes it easier to:
- Find and modify default values
- Understand system limits at a glance
- Maintain consistency across modules
"""

from __future__ import annotations

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

# Token limits
DEFAULT_MAX_OUTPUT_TOKENS = 8192  # Conservative default for most models

# Concurrency
DEFAULT_IO_CONCURRENCY = 20  # I/O operations (file reads, etc.)
DEFAULT_LLM_CONCURRENCY = 10  # LLM API calls (config default)
DEFAULT_BATCH_CONCURRENCY = 10  # Batch file processing (config default)

# Batch sizes
DEFAULT_MAX_IMAGES_PER_BATCH = 10  # Images per LLM vision call
DEFAULT_MAX_PAGES_PER_BATCH = 10  # Pages per LLM call for document processing

# Router settings
DEFAULT_ROUTER_NUM_RETRIES = 2
DEFAULT_ROUTER_TIMEOUT = 120  # seconds

# Note: RETRYABLE_ERRORS tuple is defined in llm.py as it contains
# actual exception classes from litellm that cannot be imported here

# =============================================================================
# Image Processing
# =============================================================================

DEFAULT_IMAGE_QUALITY = 85  # JPEG quality (1-100)
DEFAULT_RENDER_DPI = 150  # DPI for page screenshots (PDF, PPTX, etc.)
DEFAULT_IMAGE_IO_CONCURRENCY = 4  # Concurrent I/O for image saving
DEFAULT_IMAGE_MAX_WIDTH = 1920
DEFAULT_IMAGE_MAX_HEIGHT = 1080

# Image filter thresholds
DEFAULT_IMAGE_FILTER_MIN_WIDTH = 50
DEFAULT_IMAGE_FILTER_MIN_HEIGHT = 50
DEFAULT_IMAGE_FILTER_MIN_AREA = 5000

# =============================================================================
# Cache Settings
# =============================================================================

DEFAULT_CACHE_MAXSIZE = 100  # Max entries in LLM content cache
DEFAULT_CACHE_TTL_SECONDS = 300  # Cache TTL (5 minutes)

# =============================================================================
# Batch Processing
# =============================================================================

DEFAULT_STATE_FLUSH_INTERVAL_SECONDS = 5
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

DEFAULT_LOG_PANEL_MAX_LINES = 8  # Lines shown in verbose mode log panel
DEFAULT_JSON_INDENT = 2  # JSON output indentation

# =============================================================================
# Paths and Filenames
# =============================================================================

DEFAULT_OUTPUT_DIR = "./output"
DEFAULT_PROMPTS_DIR = "~/.markit/prompts"
DEFAULT_LOG_DIR = "~/.markit/logs"
CONFIG_FILENAME = "markit.json"

# =============================================================================
# OCR
# =============================================================================

DEFAULT_OCR_LANG = "zh"
DEFAULT_OCR_SAMPLE_PAGES = 3  # Pages to sample for scanned PDF detection

# =============================================================================
# Misc Defaults
# =============================================================================

DEFAULT_MODEL_WEIGHT = 1  # Default model weight in router
DEFAULT_SCREENSHOT_MAX_BYTES = int(
    3.5 * 1024 * 1024
)  # 3.5 MB max (base64 adds ~33%, must stay under 5MB API limit)
DEFAULT_ROUTING_STRATEGY = "simple-shuffle"
DEFAULT_IMAGE_FORMAT = "jpeg"
DEFAULT_ON_CONFLICT = "rename"
DEFAULT_LOG_LEVEL = "DEBUG"
