"""Screenshot file helpers shared by the fetch strategies.

Stateless leaf module: filename derivation and JPEG post-processing for
full-page screenshots captured by the browser strategies. Kept separate from
``markitai.fetch`` so ``markitai.fetch_playwright`` can use these helpers
without importing the fetch orchestration module.

The public import path remains ``markitai.fetch``, which re-exports them.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from urllib.parse import urlparse

from loguru import logger


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

        # Quick check: get image info without full decode
        with Image.open(screenshot_path) as img:
            width, height = img.size
            needs_resize = height > max_height
            needs_convert = img.mode in ("RGBA", "P")

        # Skip re-compression if image doesn't need resize or conversion
        # Playwright already saves JPEG with specified quality
        if not needs_resize and not needs_convert:
            logger.debug(
                f"Screenshot within limits ({width}x{height}), skipping re-compression"
            )
            return

        # Only re-process if needed
        with Image.open(screenshot_path) as img:
            if needs_convert:
                img = img.convert("RGB")

            if needs_resize:
                ratio = max_height / height
                new_width = int(width * ratio)
                img = img.resize((new_width, max_height), Image.Resampling.LANCZOS)
                logger.debug(
                    f"Resized screenshot from {width}x{height} to {new_width}x{max_height}"
                )

            img.save(screenshot_path, "JPEG", quality=quality, optimize=True)
            logger.debug(
                f"Compressed screenshot to quality={quality}: {screenshot_path}"
            )
    except ImportError:
        logger.warning("Pillow not installed, skipping screenshot compression")
    except Exception as e:
        logger.warning(f"Failed to compress screenshot: {e}")
