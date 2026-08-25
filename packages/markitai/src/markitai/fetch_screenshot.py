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
    tile_height: int | None = None,
) -> list[Path]:
    """Compress, and tile long screenshots, in place.

    Returns the list of on-disk files: a single file when the screenshot is
    within ``tile_height``, or N vertical tiles (``name.jpg``,
    ``name--1.jpg``, ...) when it is taller. Each tile keeps full width and
    is a VLM-readable height, instead of the old whole-page LANCZOS
    downscale that squished a long page into one unreadable image.

    Tile 0 keeps the original path (so existing single-path consumers stay
    valid); later tiles get a ``--N`` suffix. ``max_height`` remains the
    legacy single-file cap, honored only when ``tile_height`` is 0/None
    (callers and tests that predate tiling).

    Args:
        screenshot_path: Path to screenshot file (may be overwritten)
        quality: JPEG quality (1-100)
        max_height: Legacy single-file height cap (used when tiling off)
        tile_height: Per-tile max height; triggers tiling above this
    """
    try:
        from PIL import Image

        # Quick check: get image info without full decode
        with Image.open(screenshot_path) as img:
            width, height = img.size
            needs_convert = img.mode in ("RGBA", "P")
            effective_tile = tile_height or max_height
            needs_tiling = height > effective_tile

        # Skip re-compression if the image needs neither tiling nor conversion
        if not needs_tiling and not needs_convert:
            logger.debug(
                f"Screenshot within limits ({width}x{height}), skipping re-compression"
            )
            return [screenshot_path]

        with Image.open(screenshot_path) as img:
            if needs_convert:
                img = img.convert("RGB")

            if not needs_tiling:
                # Legacy single-file path: re-compress in place.
                img.save(screenshot_path, "JPEG", quality=quality, optimize=True)
                logger.debug(
                    f"Compressed screenshot to quality={quality}: {screenshot_path}"
                )
                return [screenshot_path]

            # Tile: split the tall image into vertical tiles, each <=
            # effective_tile tall at full width (no downscale — detail kept).
            tiles: list[Path] = []
            n_tiles = (height + effective_tile - 1) // effective_tile
            for i in range(n_tiles):
                top = i * effective_tile
                bottom = min(top + effective_tile, height)
                tile = img.crop((0, top, width, bottom))
                out_path = (
                    screenshot_path
                    if i == 0
                    else screenshot_path.with_name(
                        f"{screenshot_path.stem}--{i}{screenshot_path.suffix}"
                    )
                )
                tile.save(out_path, "JPEG", quality=quality, optimize=True)
                tiles.append(out_path)
            logger.debug(
                f"Tiled screenshot from {width}x{height} into {n_tiles} tiles "
                f"(each <= {effective_tile}px tall)"
            )
            return tiles
    except ImportError:
        logger.warning("Pillow not installed, skipping screenshot compression")
        return [screenshot_path]
    except Exception as e:
        logger.warning(f"Failed to compress screenshot: {e}")
        return [screenshot_path]
