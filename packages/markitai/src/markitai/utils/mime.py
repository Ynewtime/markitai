"""MIME type utilities for image handling.

This module provides helper functions for MIME type operations,
using the centralized mappings defined in constants.py.
"""

from __future__ import annotations

from markitai.constants import EXTENSION_TO_MIME, MIME_TO_EXTENSION

# MIME types supported by vision LLMs (Anthropic Claude, Google Gemini, OpenAI GPT-4V)
# SVG, BMP, ICO etc. are NOT supported
LLM_SUPPORTED_MIME_TYPES = frozenset(
    {"image/jpeg", "image/png", "image/gif", "image/webp"}
)


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

    Vision LLMs (Claude, Gemini, GPT-4V) only support jpeg, png, gif, webp.
    Formats like SVG, BMP, ICO are NOT supported.

    Args:
        extension: File extension (with or without leading dot)

    Returns:
        True if the format is supported by vision LLMs

    Examples:
        >>> is_llm_supported_image(".jpg")
        True
        >>> is_llm_supported_image(".svg")
        False
    """
    mime_type = get_mime_type(extension, default="")
    return mime_type in LLM_SUPPORTED_MIME_TYPES
