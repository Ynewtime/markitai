"""Constants for MarkIt."""

from pathlib import Path

from markit import __version__

# Application constants
APP_NAME = "markit"
APP_VERSION = __version__

# Default paths
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_ASSETS_DIR = "assets"
DEFAULT_LOG_DIR = ".logs"
DEFAULT_STATE_FILE = ".markit-state.json"
DEFAULT_CONFIG_FILE = "markit.yaml"

# Config file locations (in order of priority)
CONFIG_LOCATIONS = [
    Path.cwd() / DEFAULT_CONFIG_FILE,
    Path.home() / ".config" / APP_NAME / "config.yaml",
]

# Supported file extensions
SUPPORTED_EXTENSIONS = {
    # Text
    ".txt": "text",
    # Word documents
    ".docx": "word",
    ".doc": "word_legacy",
    # PowerPoint
    ".pptx": "powerpoint",
    ".ppt": "powerpoint_legacy",
    # Excel
    ".xlsx": "excel",
    ".xls": "excel_legacy",
    # CSV
    ".csv": "csv",
    # PDF
    ".pdf": "pdf",
    # Images
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".gif": "image",
    ".webp": "image",
    ".bmp": "image",
    # HTML
    ".html": "html",
    ".htm": "html",
}

# MarkItDown native supported formats
MARKITDOWN_FORMATS = {
    ".txt",
    ".docx",
    ".pptx",
    ".xlsx",
    ".csv",
    ".pdf",
    ".html",
    ".htm",
    ".png",
    ".jpg",
    ".jpeg",
}

# Legacy formats requiring pre-processing (Office conversion)
LEGACY_FORMATS = {".doc", ".ppt", ".xls"}

# Image formats
IMAGE_FORMATS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}

# Convertible image formats (need to be converted to PNG/JPEG)
CONVERTIBLE_IMAGE_FORMATS = {".emf", ".wmf", ".tiff", ".tif"}

# PDF engines
PDF_ENGINES = ["pymupdf4llm", "pymupdf", "pdfplumber", "markitdown"]

# LLM Providers
LLM_PROVIDERS = ["openai", "anthropic", "gemini", "ollama", "openrouter"]

# Provider API key environment variable names
PROVIDER_API_KEY_ENV_VARS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}

# Default LLM models
DEFAULT_LLM_MODELS = {
    "openai": "gpt-5.2",
    "anthropic": "claude-sonnet-4-5",
    "gemini": "gemini-3-flash-preview",
    "ollama": "llama3.2-vision",
    "openrouter": "google/gemini-3-flash-preview",
}

# Image compression settings
DEFAULT_PNG_OPTIMIZATION_LEVEL = 2  # oxipng: 0-6
DEFAULT_JPEG_QUALITY = 85  # mozjpeg: 0-100
DEFAULT_MAX_IMAGE_DIMENSION = 2048
DEFAULT_SKIP_COMPRESSION_THRESHOLD = 10240  # 10KB

# Image filtering settings (for removing decorative/icon images)
DEFAULT_MIN_IMAGE_DIMENSION = 100  # Minimum width or height in pixels
DEFAULT_MIN_IMAGE_AREA = 40000  # Minimum area in pixels² (200x200)
DEFAULT_MIN_IMAGE_SIZE = 10240  # Minimum file size in bytes (10KB)

# Concurrency defaults
DEFAULT_FILE_WORKERS = 4
DEFAULT_IMAGE_WORKERS = 8
DEFAULT_LLM_WORKERS = 5

# Retry settings
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_MIN_WAIT = 1
DEFAULT_RETRY_MAX_WAIT = 60

# Timeout settings (seconds)
DEFAULT_LLM_TIMEOUT = 120
DEFAULT_CONVERSION_TIMEOUT = 300  # 5 minutes

# Chunk settings
# Optimized for Gemini 3 Flash (1.05M context, 65.5K output)
# 32K tokens per chunk allows ~3 chunks for a 100K token document
# This dramatically reduces LLM calls while staying well within limits
DEFAULT_CHUNK_SIZE = 32000  # tokens (was 4000)
DEFAULT_CHUNK_OVERLAP = 500  # tokens (increased for better context continuity)
