"""An item's source for the workbench's Compare view.

One descriptor says what the server can show of an item's original (a PDF's
pages, an image, an HTML or text document, or the page screenshots a
conversion saved); page images and a PDF page's positioned text lines feed the
view. The descriptor costs no rendering: page boxes come from the PDF's page
tree or the image header.

Originals are the uploads a job keeps under ``uploads/``. A URL item keeps no
page snapshot here, so it shows its screenshots when the conversion saved any;
a CLI-recorded job keeps no originals at all.
"""

from __future__ import annotations

import io
import mimetypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from markitai.constants import SCREENSHOTS_REL_PATH

if TYPE_CHECKING:
    from collections.abc import Callable

    from markitai.serve.jobs import Job, JobItem

PDF_EXTENSIONS = frozenset({".pdf"})
#: Raster images the Compare view shows as one page.
IMAGE_EXTENSIONS = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".avif",
        ".bmp",
        ".tif",
        ".tiff",
        ".heic",
        ".heif",
    }
)
#: Of those, the ones a browser draws itself (sent unchanged when small enough).
BROWSER_IMAGE_EXTENSIONS = frozenset(
    {".png", ".jpg", ".jpeg", ".gif", ".webp", ".avif", ".bmp"}
)
HTML_EXTENSIONS = frozenset({".html", ".htm", ".xhtml"})
TEXT_EXTENSIONS = frozenset(
    {
        ".md",
        ".markdown",
        ".txt",
        ".csv",
        ".tsv",
        ".json",
        ".xml",
        ".yaml",
        ".yml",
        ".log",
    }
)

#: Page boxes listed in a descriptor at most (``page_count`` is the total).
MAX_LISTED_PAGES = 2000
#: Lines of one page's text layer at most.
MAX_TEXT_LINES = 5000
#: Pixels of one rendered page at most.
MAX_RENDER_PIXELS = 32_000_000
#: An image original larger than this is not decoded for a preview (a JPEG
#: is decoded at a reduced size, other formats in full).
MAX_SOURCE_PIXELS = 100_000_000
MIN_WIDTH, MAX_WIDTH, DEFAULT_WIDTH = 160, 2400, 1200

_MEDIA_TYPES = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".avif": "image/avif",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".heic": "image/heic",
    ".heif": "image/heif",
    ".svg": "image/svg+xml",
    ".html": "text/html",
    ".htm": "text/html",
    ".xhtml": "application/xhtml+xml",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".txt": "text/plain",
    ".csv": "text/csv",
    ".tsv": "text/tab-separated-values",
    ".json": "application/json",
    ".xml": "application/xml",
    ".yaml": "application/yaml",
    ".yml": "application/yaml",
    ".log": "text/plain",
}


class RenderError(Exception):
    """A page that cannot be drawn."""


@dataclass
class Original:
    """What the server can show of one item's source."""

    kind: str  # pdf | image | html | text | screenshots | none
    name: str
    path: Path | None = None  # the stored original, when kept
    media_type: str | None = None
    size: int | None = None
    pages: list[tuple[float, float]] = field(default_factory=list)
    page_count: int = 0
    text_layer: bool = False
    reason: str | None = None
    screenshots: list[Path] = field(default_factory=list)

    def descriptor(self, download: str | None) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "media_type": self.media_type,
            "size": self.size,
            "download": download if self.path is not None else None,
            "pages": [
                {"width": round(width, 2), "height": round(height, 2)}
                for width, height in self.pages[:MAX_LISTED_PAGES]
            ],
            "page_count": self.page_count,
            "text_layer": self.text_layer,
            "reason": self.reason,
        }


def media_type_of(path: Path) -> str:
    suffix = path.suffix.lower()
    return (
        _MEDIA_TYPES.get(suffix)
        or mimetypes.guess_type(path.name)[0]
        or "application/octet-stream"
    )


def kept_upload(job: Job, item: JobItem) -> Path | None:
    """The item's stored upload, never a path outside the job's uploads."""
    if item.kind != "file" or not item.name:
        return None
    uploads = job.uploads_dir.resolve()
    upload = (uploads / item.name).resolve()
    if upload == uploads or not upload.is_relative_to(uploads) or not upload.is_file():
        return None
    return upload


def _screenshot_key(path: Path) -> tuple[str, int, str]:
    """Order a capture's tiles after it: ``x.jpg``, ``x--1.jpg``, ``x--2.jpg``."""
    stem, suffix = path.stem, path.suffix
    head, sep, tail = stem.rpartition("--")
    if sep and tail.isdigit():
        return (f"{head}{suffix}", int(tail), path.name)
    return (path.name, 0, path.name)


def item_screenshots(job: Job, item: JobItem) -> list[Path]:
    """Page screenshots among the item's own files, in page order."""
    from markitai.serve.artifacts import item_asset_files, markdown_pair
    from markitai.serve.jobs import item_base_name

    base_name = item_base_name(item)
    if base_name is None:
        return []
    out_dir = job.out_dir.resolve()
    screenshots_dir = (out_dir / SCREENSHOTS_REL_PATH).resolve()
    pair = markdown_pair(out_dir, base_name)
    owned = [
        path
        for path in item_asset_files(out_dir, base_name, pair)
        if path.resolve().parent == screenshots_dir
        and path.suffix.lower() in BROWSER_IMAGE_EXTENSIONS
    ]
    return sorted(owned, key=_screenshot_key)


def _image_size(path: Path) -> tuple[int, int]:
    """Displayed size of an image: its header size, EXIF rotation applied."""
    from PIL import Image

    if path.suffix.lower() in (".heic", ".heif"):
        from markitai.converter.heif import require_pillow_heif

        require_pillow_heif()
    with Image.open(path) as image:
        width, height = image.size
        try:
            orientation = image.getexif().get(0x0112)
        except Exception:
            orientation = None
    if orientation in (5, 6, 7, 8):
        width, height = height, width
    return width, height


def pdf_pages(path: Path, boxes: bool = True) -> tuple[list[tuple[float, float]], int]:
    """A PDF's page boxes (the first ``MAX_LISTED_PAGES``) and its page count."""
    import pymupdf

    with pymupdf.open(path) as document:
        if document.needs_pass:
            raise RenderError("the PDF is encrypted")
        count = document.page_count
        pages = []
        for index in range(min(count, MAX_LISTED_PAGES) if boxes else 0):
            rect = document[index].rect  # the visible box, rotation applied
            pages.append((float(rect.width), float(rect.height)))
    return pages, count


def describe(
    job: Job,
    item: JobItem,
    *,
    boxes: bool = True,
    pdf_layout: Callable[[Path, bool], tuple[list[tuple[float, float]], int]]
    | None = None,
) -> Original:
    """What the Compare view can show of *item* (no rendering).

    ``boxes=False`` skips the page sizes (a page request needs only the kind
    and the page count, and a long PDF has thousands of boxes).
    """
    upload = kept_upload(job, item)
    if upload is not None:
        suffix = upload.suffix.lower()
        size = upload.stat().st_size
        media_type = media_type_of(upload)
        base = Original(
            kind="none", name=item.name, path=upload, media_type=media_type, size=size
        )
        if suffix in PDF_EXTENSIONS:
            try:
                pages, count = (pdf_layout or pdf_pages)(upload, boxes)
            except Exception:
                base.reason = "render_unavailable"
            else:
                base.kind, base.pages, base.page_count = "pdf", pages, count
                base.text_layer = True
                return base
        elif suffix in IMAGE_EXTENSIONS:
            try:
                width, height = _image_size(upload)
            except Exception:
                base.reason = "render_unavailable"
            else:
                if width * height > MAX_SOURCE_PIXELS:
                    base.reason = "too_large"
                else:
                    base.kind, base.pages, base.page_count = (
                        "image",
                        [(float(width), float(height))],
                        1,
                    )
                    return base
        elif suffix in HTML_EXTENSIONS:
            base.kind = "html"
            return base
        elif suffix in TEXT_EXTENSIONS:
            base.kind = "text"
            return base
        screenshots = item_screenshots(job, item)
        if screenshots:
            return _with_screenshots(base, screenshots, boxes)
        base.reason = base.reason or "unsupported_format"
        return base
    screenshots = item_screenshots(job, item)
    original = Original(kind="none", name=item.name, reason="not_kept")
    if screenshots:
        return _with_screenshots(original, screenshots, boxes)
    return original


def _with_screenshots(
    original: Original, screenshots: list[Path], boxes: bool = True
) -> Original:
    pages: list[tuple[float, float]] = []
    for shot in screenshots[:MAX_LISTED_PAGES] if boxes else []:
        try:
            width, height = _image_size(shot)
        except Exception:
            width, height = 0, 0
        pages.append((float(width), float(height)))
    original.kind = "screenshots"
    original.screenshots = screenshots
    original.pages = pages
    original.page_count = len(screenshots)
    original.reason = None
    original.text_layer = False
    return original


def clamp_width(width: int | None) -> int:
    if width is None:
        return DEFAULT_WIDTH
    return max(MIN_WIDTH, min(MAX_WIDTH, width))


def _fit(width: float, height: float, target: int) -> float:
    """Scale that makes *width* ``target`` wide within the pixel cap."""
    scale = target / width if width > 0 else 1.0
    if width * height * scale * scale > MAX_RENDER_PIXELS:
        scale = (MAX_RENDER_PIXELS / (width * height)) ** 0.5
    return scale


def render_pdf_page(path: Path, page_number: int, width: int) -> bytes:
    """Page *page_number* (1-based) of a PDF as PNG, *width* pixels wide."""
    import pymupdf

    with pymupdf.open(path) as document:
        if not 1 <= page_number <= document.page_count:
            raise IndexError(page_number)
        page = document[page_number - 1]
        rect = page.rect
        if rect.width <= 0 or rect.height <= 0:
            raise RenderError("the page has no area")
        scale = _fit(rect.width, rect.height, width)
        try:
            pixmap = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
            return pixmap.tobytes("png")
        except Exception as e:  # MuPDF raises its own error types
            raise RenderError(str(e)) from e


def render_image(path: Path, width: int) -> tuple[bytes, str]:
    """An image page: the file itself when a browser can show it at this
    width, else a PNG scaled to *width* (EXIF rotation applied).

    Decoded only as large as needed: a JPEG at a reduced scale, other formats
    reduced in steps before the final filter.
    """
    from PIL import Image, ImageOps

    suffix = path.suffix.lower()
    if suffix in (".heic", ".heif"):
        from markitai.converter.heif import require_pillow_heif

        require_pillow_heif()
    try:
        with Image.open(path) as image:
            stored_w, stored_h = image.size
            try:
                orientation = image.getexif().get(0x0112)
            except Exception:
                orientation = None
            turned = orientation in (5, 6, 7, 8)
            shown_w, shown_h = (stored_h, stored_w) if turned else (stored_w, stored_h)
            if (
                suffix in BROWSER_IMAGE_EXTENSIONS
                and shown_w <= width
                and shown_w * shown_h <= MAX_RENDER_PIXELS
            ):
                return path.read_bytes(), media_type_of(path)
            if stored_w * stored_h > MAX_SOURCE_PIXELS and image.format != "JPEG":
                raise RenderError("the image is too large to preview")
            scale = min(1.0, _fit(shown_w, shown_h, width))
            target = (max(1, round(shown_w * scale)), max(1, round(shown_h * scale)))
            box = (target[1], target[0]) if turned else target
            image.draft("RGB", box)  # JPEG only: decode at a reduced scale
            image.thumbnail(box, Image.Resampling.LANCZOS, reducing_gap=2.0)
            frame = ImageOps.exif_transpose(image)
            if frame.mode not in ("RGB", "RGBA", "L", "LA"):
                frame = frame.convert("RGBA" if "A" in frame.getbands() else "RGB")
            buffer = io.BytesIO()
            frame.save(buffer, format="PNG", optimize=False)
            return buffer.getvalue(), "image/png"
    except RenderError:
        raise
    except Exception as e:
        raise RenderError(str(e)) from e


def pdf_page_text(path: Path, page_number: int) -> dict[str, Any]:
    """Positioned text lines of one PDF page, top-left origin, in points."""
    import pymupdf

    with pymupdf.open(path) as document:
        if not 1 <= page_number <= document.page_count:
            raise IndexError(page_number)
        page = document[page_number - 1]
        rect = page.rect
        # Extraction coordinates are in the unrotated page space; the
        # rotation matrix maps them onto the visible (rotated) page.
        rotate = page.rotation_matrix
        lines: list[dict[str, Any]] = []
        try:
            data = page.get_text("dict", sort=True)
        except Exception as e:
            raise RenderError(str(e)) from e
        if not isinstance(data, dict):
            raise RenderError("no text layer")
        for block in data.get("blocks", ()):
            for line in block.get("lines", ()):
                text = "".join(span.get("text", "") for span in line.get("spans", ()))
                if not text.strip():
                    continue
                box = pymupdf.Rect(line["bbox"])
                if page.rotation:
                    box = (box * rotate).normalize()
                lines.append(
                    {
                        "text": text,
                        "bbox": [
                            round(max(0.0, box.x0), 2),
                            round(max(0.0, box.y0), 2),
                            round(min(rect.width, box.x1), 2),
                            round(min(rect.height, box.y1), 2),
                        ],
                    }
                )
                if len(lines) >= MAX_TEXT_LINES:
                    break
            if len(lines) >= MAX_TEXT_LINES:
                break
    return {
        "page": page_number,
        "width": round(float(rect.width), 2),
        "height": round(float(rect.height), 2),
        "lines": lines,
    }


class PdfWorkers:
    """PyMuPDF work for the Compare view, in a few worker processes.

    PyMuPDF holds the GIL while it draws or reads a page, so in a thread a
    page that takes seconds to draw would stall every other request of the
    server. Spawned processes do it instead; a call that runs past *timeout*
    takes its pool down (the processes are killed) and fails with
    :class:`RenderError`, and the next call starts a fresh pool.
    """

    def __init__(self, workers: int = 2, timeout: float = 20.0) -> None:
        import threading

        self.workers = workers
        self.timeout = timeout
        self._lock = threading.Lock()
        self._pool: Any = None

    def _get_pool(self) -> Any:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        with self._lock:
            if self._pool is None:
                self._pool = ProcessPoolExecutor(
                    max_workers=self.workers,
                    mp_context=multiprocessing.get_context("spawn"),
                )
            return self._pool

    def _drop(self, pool: Any, *, kill: bool) -> None:
        with self._lock:
            if self._pool is pool:
                self._pool = None
        if kill:
            for process in list((getattr(pool, "_processes", None) or {}).values()):
                try:
                    process.kill()
                except Exception:  # already gone
                    pass
        pool.shutdown(wait=False, cancel_futures=True)

    def run(self, function: Callable[..., Any], *args: Any) -> Any:
        """Run *function* in a worker and wait for it (call from a thread)."""
        from concurrent.futures import TimeoutError as FutureTimeout
        from concurrent.futures.process import BrokenProcessPool

        pool = self._get_pool()
        try:
            future = pool.submit(function, *args)
        except (BrokenProcessPool, RuntimeError):
            self._drop(pool, kill=True)
            pool = self._get_pool()
            future = pool.submit(function, *args)
        try:
            return future.result(timeout=self.timeout)
        except FutureTimeout:
            self._drop(pool, kill=True)
            raise RenderError(f"gave up after {self.timeout:.0f} s") from None
        except BrokenProcessPool:
            self._drop(pool, kill=False)
            raise RenderError("the page worker stopped") from None

    def close(self) -> None:
        with self._lock:
            pool, self._pool = self._pool, None
        if pool is not None:
            self._drop(pool, kill=True)


class ByteCache:
    """A small LRU of rendered pages, bounded by total size."""

    def __init__(self, max_bytes: int = 64 * 1024 * 1024) -> None:
        import threading
        from collections import OrderedDict

        self.max_bytes = max_bytes
        self._lock = threading.Lock()
        self._entries: OrderedDict[tuple[Any, ...], tuple[bytes, str]] = OrderedDict()
        self._size = 0

    def get(self, key: tuple[Any, ...]) -> tuple[bytes, str] | None:
        with self._lock:
            value = self._entries.get(key)
            if value is not None:
                self._entries.move_to_end(key)
            return value

    def put(self, key: tuple[Any, ...], value: tuple[bytes, str]) -> None:
        if len(value[0]) > self.max_bytes // 4:
            return
        with self._lock:
            old = self._entries.pop(key, None)
            if old is not None:
                self._size -= len(old[0])
            self._entries[key] = value
            self._size += len(value[0])
            while self._size > self.max_bytes and self._entries:
                _, dropped = self._entries.popitem(last=False)
                self._size -= len(dropped[0])
