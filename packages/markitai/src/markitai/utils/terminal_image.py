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


class Protocol(Enum):
    """Supported terminal image protocols."""

    KITTY = "kitty"
    ITERM2 = "iterm2"


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
    # Kitty, Ghostty, and other terminals that support the Kitty graphics protocol
    if os.environ.get("KITTY_PID"):
        return Protocol.KITTY
    term = os.environ.get("TERM", "")
    if "xterm-kitty" in term or "xterm-ghostty" in term:
        return Protocol.KITTY
    term_program = os.environ.get("TERM_PROGRAM", "")
    if term_program == "ghostty":
        return Protocol.KITTY

    # iTerm2 detection
    if term_program in _ITERM2_TERM_PROGRAMS:
        return Protocol.ITERM2

    return None


def render_inline_image(image_path: Path, protocol: Protocol) -> str:
    """Read an image file and return the terminal escape sequence for inline display.

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


def _to_png_bytes(data: bytes) -> bytes:
    """Convert any image format to PNG bytes.

    Kitty graphics protocol only supports PNG (f=100) or raw pixels (f=32/24).
    Converting to PNG is the simplest approach that handles JPEG, BMP, etc.
    If data is already PNG, it passes through without re-encoding.
    """
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return data  # already PNG
    import io

    from PIL import Image

    img = Image.open(io.BytesIO(data))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _render_kitty(data: bytes) -> str:
    """Render image using Kitty graphics protocol (direct data transmission).

    Uses f=100 (PNG format). Non-PNG images are converted before sending.
    """
    png_data = _to_png_bytes(data)
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
    """Render image using iTerm2 inline images protocol."""
    encoded = base64.standard_b64encode(data).decode("ascii")
    size = len(data)
    return f"\033]1337;File=inline=1;size={size}:{encoded}\a"
