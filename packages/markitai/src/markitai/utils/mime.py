"""MIME type utilities for image handling.

This module provides helper functions for MIME type operations,
using the centralized mappings defined in constants.py.
"""

from __future__ import annotations

import importlib.util

from markitai.constants import EXTENSION_TO_MIME, MIME_TO_EXTENSION

# MIME types supported by vision LLMs (Anthropic Claude, Google Gemini, OpenAI GPT-4V)
# SVG, ICO etc. are NOT supported
LLM_SUPPORTED_MIME_TYPES = frozenset(
    {"image/jpeg", "image/png", "image/gif", "image/webp"}
)

# Cache cairosvg availability check (result never changes during process lifetime)
_HAS_CAIROSVG: bool | None = None


def _has_cairosvg() -> bool:
    """Check if cairosvg is available, caching the result."""
    global _HAS_CAIROSVG  # noqa: PLW0603
    if _HAS_CAIROSVG is None:
        _HAS_CAIROSVG = importlib.util.find_spec("cairosvg") is not None
    return _HAS_CAIROSVG


# Image extensions that LLM APIs don't natively accept but can be converted to PNG
LLM_CONVERTIBLE_EXTENSIONS = frozenset({".bmp", ".tiff", ".tif"})


def normalize_image_extension(fmt: str) -> str:
    """Normalize image format extension for file naming.

    Converts "jpeg" to "jpg" since file extensions conventionally use the
    shorter form. All other formats are returned as-is.
    Input is normalized to lowercase before matching.

    Args:
        fmt: Image format string, e.g. "jpeg", "png", "webp", "JPEG"

    Returns:
        Normalized lowercase extension string, e.g. "jpg", "png", "webp"

    Examples:
        >>> normalize_image_extension("jpeg")
        'jpg'
        >>> normalize_image_extension("JPEG")
        'jpg'
        >>> normalize_image_extension("png")
        'png'
    """
    fmt = fmt.lower()
    if fmt == "jpeg":
        return "jpg"
    return fmt


def get_mime_type(extension: str, default: str = "image/jpeg") -> str:
    """Get MIME type from file extension.

    Args:
        extension: File extension (with or without leading dot), e.g. ".jpg" or "jpg"
        default: Default MIME type if extension is not recognized

    Returns:
        MIME type string, e.g. "image/jpeg"

    Examples:
        >>> get_mime_type(".jpg")
        'image/jpeg'
        >>> get_mime_type("png")
        'image/png'
        >>> get_mime_type(".unknown")
        'image/jpeg'
    """
    # Normalize extension to have leading dot and be lowercase
    ext = extension.lower()
    if not ext.startswith("."):
        ext = f".{ext}"
    return EXTENSION_TO_MIME.get(ext, default)


def get_extension_from_mime(mime_type: str, default: str = ".jpg") -> str:
    """Get file extension from MIME type.

    Args:
        mime_type: MIME type string, e.g. "image/jpeg"
        default: Default extension if MIME type is not recognized

    Returns:
        File extension with leading dot, e.g. ".jpg"

    Examples:
        >>> get_extension_from_mime("image/jpeg")
        '.jpg'
        >>> get_extension_from_mime("image/png")
        '.png'
        >>> get_extension_from_mime("image/unknown")
        '.jpg'
    """
    # Handle content-type with parameters (e.g. "image/jpeg; charset=utf-8")
    clean_mime = mime_type.lower().split(";")[0].strip()
    return MIME_TO_EXTENSION.get(clean_mime, default)


def is_llm_supported_image(extension: str) -> bool:
    """Check if image format is supported by vision LLMs.

    Vision LLMs (Claude, Gemini, GPT-4V) natively support jpeg, png, gif, webp.
    BMP and TIFF are also accepted here because they are auto-converted to PNG
    before being sent to the API (see _get_cached_image in processor.py).
    SVG is supported only when ``cairosvg`` is installed (rasterized to PNG).

    Args:
        extension: File extension (with or without leading dot)

    Returns:
        True if the format is supported (natively or via conversion)

    Examples:
        >>> is_llm_supported_image(".jpg")
        True
        >>> is_llm_supported_image(".bmp")
        True
    """
    ext = extension.lower()
    if not ext.startswith("."):
        ext = f".{ext}"
    if ext in LLM_CONVERTIBLE_EXTENSIONS:
        return True
    # SVG requires cairosvg for rasterization to PNG
    if ext == ".svg":
        return _has_cairosvg()
    mime_type = get_mime_type(extension, default="")
    return mime_type in LLM_SUPPORTED_MIME_TYPES


def get_llm_effective_mime(extension: str) -> str:
    """Get the MIME type that will actually be sent to the LLM API.

    For convertible formats (BMP, TIFF) and SVG (when cairosvg is available),
    returns image/png since they are converted before sending. For natively
    supported formats, returns the normal MIME type.

    Args:
        extension: File extension (with or without leading dot)

    Returns:
        MIME type string for the LLM API
    """
    ext = extension.lower()
    if not ext.startswith("."):
        ext = f".{ext}"
    if ext in LLM_CONVERTIBLE_EXTENSIONS:
        return "image/png"
    # SVG is rasterized to PNG when cairosvg is available
    if ext == ".svg" and _has_cairosvg():
        return "image/png"
    return get_mime_type(extension)
