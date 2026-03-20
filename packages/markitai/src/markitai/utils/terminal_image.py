"""Terminal image protocol detection and inline rendering.

Supports Kitty graphics protocol and iTerm2 inline images protocol.
Detection requires stdout to be a TTY — pipes, redirects, and CI
environments always return None to prevent escape sequence leakage.
"""

from __future__ import annotations

import base64
import os
import sys
from enum import Enum
from pathlib import Path

KITTY_CHUNK_SIZE = 4096  # bytes of base64 data per chunk
MAX_INLINE_WIDTH = 800  # max pixel width for terminal inline images


class Protocol(Enum):
    """Supported terminal image protocols."""

    KITTY = "kitty"
    ITERM2 = "iterm2"


# Terminals that support the Kitty graphics protocol (non-exhaustive)
_KITTY_TERM_PROGRAMS = frozenset({"ghostty"})

# iTerm2-compatible TERM_PROGRAM values (non-exhaustive, extensible)
_ITERM2_TERM_PROGRAMS = frozenset({"iTerm.app", "WezTerm", "mintty", "Hyper", "Tabby"})


def detect_protocol() -> Protocol | None:
    """Detect the best supported terminal image protocol.

    Returns None if stdout is not a TTY or no supported protocol is detected.
    When both Kitty and iTerm2 signals are present, Kitty takes priority.
    """
    if not sys.stdout.isatty():
        return None

    # Kitty graphics protocol detection (highest priority)
    if os.environ.get("KITTY_PID"):
        return Protocol.KITTY
    term = os.environ.get("TERM", "")
    if "xterm-kitty" in term or "xterm-ghostty" in term:
        return Protocol.KITTY
    term_program = os.environ.get("TERM_PROGRAM", "")
    if term_program in _KITTY_TERM_PROGRAMS:
        return Protocol.KITTY

    # iTerm2 detection
    if term_program in _ITERM2_TERM_PROGRAMS:
        return Protocol.ITERM2

    return None


def render_inline_image(image_path: Path, protocol: Protocol) -> str:
    """Read an image file and return the terminal escape sequence for inline display.

    Large images are resized to MAX_INLINE_WIDTH to keep memory usage and
    transfer size reasonable for terminal display.

    Args:
        image_path: Path to the image file. Must exist.
        protocol: Which terminal protocol to use.

    Returns:
        String containing the terminal escape sequence.

    Raises:
        FileNotFoundError: If image_path does not exist.
    """
    data = image_path.read_bytes()

    if protocol == Protocol.KITTY:
        return _render_kitty(data)
    elif protocol == Protocol.ITERM2:
        return _render_iterm2(data)
    else:
        raise ValueError(f"Unsupported protocol: {protocol}")


def _prepare_png(data: bytes) -> bytes:
    """Convert image to PNG, resizing if wider than MAX_INLINE_WIDTH.

    Both Kitty (f=100) and iTerm2 protocols work best with PNG.
    Resizing large images keeps memory and transfer size manageable.
    If data is already a small PNG, it passes through without re-encoding.
    """
    import io

    from PIL import Image

    is_png = data[:8] == b"\x89PNG\r\n\x1a\n"

    img = Image.open(io.BytesIO(data))
    needs_resize = img.width > MAX_INLINE_WIDTH

    if is_png and not needs_resize:
        return data  # already a small PNG — pass through

    if needs_resize:
        ratio = MAX_INLINE_WIDTH / img.width
        new_height = int(img.height * ratio)
        img = img.resize((MAX_INLINE_WIDTH, new_height), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _render_kitty(data: bytes) -> str:
    """Render image using Kitty graphics protocol (direct data transmission).

    Uses f=100 (PNG format). Non-PNG images are converted, large images resized.
    """
    png_data = _prepare_png(data)
    encoded = base64.standard_b64encode(png_data).decode("ascii")
    chunks: list[str] = []

    for i in range(0, len(encoded), KITTY_CHUNK_SIZE):
        chunk = encoded[i : i + KITTY_CHUNK_SIZE]
        is_last = i + KITTY_CHUNK_SIZE >= len(encoded)
        m = 0 if is_last else 1

        if i == 0:
            # First chunk: f=100 means PNG format
            chunks.append(f"\033_Ga=T,f=100,t=d,m={m};{chunk}\033\\")
        else:
            chunks.append(f"\033_Gm={m};{chunk}\033\\")

    return "".join(chunks)


def _render_iterm2(data: bytes) -> str:
    """Render image using iTerm2 inline images protocol.

    Converts to PNG for cross-terminal compatibility and resizes if needed.
    """
    png_data = _prepare_png(data)
    encoded = base64.standard_b64encode(png_data).decode("ascii")
    size = len(png_data)
    return f"\033]1337;File=inline=1;size={size}:{encoded}\a"
