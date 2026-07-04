"""HEIF / HEIC / AVIF input support (optional, via pillow-heif).

Follows the xberg pattern: a cheap 12-byte ``ftyp`` box sniff that is always
available, plus a decode-to-PNG boundary that requires the optional
``pillow-heif`` dependency (``markitai[heif]``). Files are decoded to PNG
once, then flow through the existing image pipeline (OCR, vision analysis,
compression) without further special-casing.
"""

from __future__ import annotations

from pathlib import Path

# Extensions handled by this module (dispatch is extension-based; the ftyp
# sniff below confirms the container before requiring pillow-heif).
HEIF_SUFFIXES = frozenset({".heic", ".heif", ".avif"})

# ISO/IEC 14496-12 / 23008-12 / 23000-22 major brands emitted by HEIF-family
# encoders (mirrors xberg's is_heif_container brand list).
_HEIF_BRANDS = frozenset(
    {
        b"heic",
        b"heix",
        b"heim",
        b"heis",
        b"hevc",
        b"hevm",
        b"hevs",
        b"mif1",
        b"msf1",
        b"avif",
        b"avis",
        b"avcs",
    }
)

_HEIF_INSTALL_HINT = (
    "pillow-heif is required to decode HEIC/HEIF/AVIF images but is not "
    'installed. Install it with: uv tool install "markitai[heif]" '
    '(or: pip install "markitai[heif]")'
)

_opener_registered = False


def is_heif_container(data: bytes) -> bool:
    """Detect a HEIF-family container by sniffing the first 12 bytes.

    Checks for an ISO-BMFF ``ftyp`` box at offset 4..8 with one of the known
    HEIF/AVIF major brands at offset 8..12.
    """
    return (
        len(data) >= 12 and data[4:8] == b"ftyp" and bytes(data[8:12]) in _HEIF_BRANDS
    )


def require_pillow_heif() -> None:
    """Register the pillow-heif Pillow plugin (lazily, once).

    Raises:
        ImportError: If pillow-heif is not installed, with an actionable
            message naming the ``markitai[heif]`` extra.
    """
    global _opener_registered
    if _opener_registered:
        return
    try:
        from pillow_heif import register_heif_opener
    except ImportError as e:
        raise ImportError(_HEIF_INSTALL_HINT) from e
    register_heif_opener()
    _opener_registered = True


def ensure_heif_ready(input_path: Path) -> None:
    """Ensure a HEIF-family file can be decoded, failing early and clearly.

    Sniffs the container first: files that merely carry a ``.heic``-style
    extension but are not actually HEIF (e.g. a renamed JPEG) don't require
    pillow-heif — Pillow dispatches on content.
    """
    with open(input_path, "rb") as f:
        header = f.read(12)
    if is_heif_container(header):
        require_pillow_heif()


def decode_to_png(input_path: Path, dest_path: Path) -> None:
    """Decode a HEIF-family image to PNG once at the boundary.

    Applies EXIF orientation (iPhone photos carry it) so downstream OCR /
    vision / compression see an upright image.
    """
    ensure_heif_ready(input_path)
    from PIL import Image, ImageOps

    with Image.open(input_path) as img:
        upright = ImageOps.exif_transpose(img)
        (upright if upright is not None else img).save(dest_path, format="PNG")
