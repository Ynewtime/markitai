"""An item's source for the workbench's Compare view.

One descriptor says what the server can show of an item's original (a PDF's
pages, an image, an HTML or text document, or the page screenshots a
conversion saved); page images and a PDF page's positioned text lines feed the
view. The descriptor costs no rendering: page boxes come from the PDF's page
tree or the image header.

Originals are the uploads a job keeps under ``uploads/`` and, for a URL item,
the page the server fetched, kept under ``snapshots/<item_id>.html`` (contract
v1.11 section 7) beside the page screenshots the conversion saved. A
CLI-recorded job keeps no originals at all.

A URL item also keeps, in ``snapshots/`` (contract v1.14): how usable the page
and its screenshots were when they were captured (``<item_id>.capture.json``),
where the screenshots show the page's blocks (``<item_id>.blocks.json``) and,
when the conversion saved no screenshots of its own but a browser loaded the
page, compare screenshots (``<item_id>.shots/<n>.jpg``). None of them is a
result file: not listed in a result, not archived, never sent to a model.
"""

from __future__ import annotations

import io
import json
import mimetypes
import os
import re
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from markitai.constants import SCREENSHOTS_REL_PATH

if TYPE_CHECKING:
    from collections.abc import Callable

    from markitai.serve.jobs import Job, JobItem

#: A URL item's fetched page is kept up to this size, and framed up to it
MAX_SNAPSHOT_BYTES = 10 * 1024 * 1024
SNAPSHOTS_DIR = "snapshots"
#: Compare screenshots an item keeps at most
MAX_COMPARE_SHOTS = 20
#: The largest ``capture.json`` read
MAX_CAPTURE_RECORD = 64 * 1024
CAPTURE_STATES = frozenset({"ok", "empty", "challenge"})

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
    #: URL items: ``{"page": …, "screenshot": …}`` (contract v1.14)
    capture: dict[str, Any] | None = None
    #: The screenshots' block boxes, when they belong to ``screenshots``
    blocks: Path | None = None

    def descriptor(
        self,
        download: str | None,
        view: str | None = None,
        blocks: str | None = None,
    ) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "media_type": self.media_type,
            "size": self.size,
            "download": download if self.path is not None else None,
            "view": view if self.kind == "html" and self.path is not None else None,
            "pages": [
                {"width": round(width, 2), "height": round(height, 2)}
                for width, height in self.pages[:MAX_LISTED_PAGES]
            ],
            "page_count": self.page_count,
            "text_layer": self.text_layer,
            "reason": self.reason,
            "capture": self.capture,
            "blocks": blocks if self.blocks is not None else None,
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


def _kept_name(item: JobItem, suffix: str) -> str | None:
    """``<item_id><suffix>`` inside ``snapshots/`` for a URL item whose ID
    is a plain name (an ID read back from disk may be anything)."""
    item_id = item.item_id
    if item.kind != "url" or not item_id.isalnum() or not item_id.isascii():
        return None
    return f"{item_id}{suffix}"


def page_snapshot_path(job: Job, item: JobItem) -> Path | None:
    """Where a URL item's fetched page is kept (None for any other item)."""
    name = _kept_name(item, ".html")
    return None if name is None else job.job_dir / SNAPSHOTS_DIR / name


def capture_record_path(job: Job, item: JobItem) -> Path | None:
    """Where a URL item's capture qualities are kept."""
    name = _kept_name(item, ".capture.json")
    return None if name is None else job.job_dir / SNAPSHOTS_DIR / name


def blocks_path(job: Job, item: JobItem) -> Path | None:
    """Where a URL item's screenshot block boxes are kept."""
    name = _kept_name(item, ".blocks.json")
    return None if name is None else job.job_dir / SNAPSHOTS_DIR / name


def compare_shots_dir(job: Job, item: JobItem) -> Path | None:
    """Where a URL item's compare screenshots are kept."""
    name = _kept_name(item, ".shots")
    return None if name is None else job.job_dir / SNAPSHOTS_DIR / name


def _plain_file(path: Path | None) -> Path | None:
    if path is None or path.is_symlink() or not path.is_file():
        return None
    return path


def kept_snapshot(job: Job, item: JobItem) -> Path | None:
    """The URL item's kept page, when there is one."""
    return _plain_file(page_snapshot_path(job, item))


def kept_blocks(job: Job, item: JobItem) -> Path | None:
    """The URL item's kept block boxes, when there are some."""
    return _plain_file(blocks_path(job, item))


def compare_shots(job: Job, item: JobItem) -> list[Path]:
    """The item's compare screenshots in order (``1.jpg``, ``2.jpg``, … up
    to the first gap)."""
    folder = compare_shots_dir(job, item)
    if folder is None or folder.is_symlink() or not folder.is_dir():
        return []
    shots: list[Path] = []
    for number in range(1, MAX_COMPARE_SHOTS + 1):
        shot = _plain_file(folder / f"{number}.jpg")
        if shot is None:
            break
        shots.append(shot)
    return shots


def kept_capture_files(job: Job, item: JobItem) -> list[Path]:
    """Every file a URL item keeps in ``snapshots/``, for deleting it."""
    found = [
        path
        for path in (
            page_snapshot_path(job, item),
            capture_record_path(job, item),
            blocks_path(job, item),
        )
        if _plain_file(path) is not None
    ]
    folder = compare_shots_dir(job, item)
    if folder is not None and not folder.is_symlink() and folder.is_dir():
        found.extend(path for path in folder.iterdir() if _plain_file(path))
    return [path for path in found if path is not None]


def remove_empty_shots_dir(job: Job, item: JobItem) -> None:
    """Remove the item's compare screenshot folder once it is empty."""
    folder = compare_shots_dir(job, item)
    if folder is not None and not folder.is_symlink():
        try:
            folder.rmdir()
        except OSError:
            pass


def _private_dir(path: Path) -> None:
    path.mkdir(mode=0o700, exist_ok=True)


def _write_private(path: Path, data: bytes) -> bool:
    """Write *data* to *path* readable by the owner only, atomically: a
    reader never sees half a file, and a failure leaves none."""
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    except OSError:
        return False
    finally:
        tmp.unlink(missing_ok=True)
    return True


def keep_page_snapshot(job: Job, item: JobItem, html: str) -> bool:
    """Keep *html* as the URL item's fetched page, replacing an older one.

    UTF-8, at most ``MAX_SNAPSHOT_BYTES`` (a larger page is not kept), and
    written beside its place then moved in, so a reader never sees half a
    page. Returns whether it was kept.
    """
    path = page_snapshot_path(job, item)
    data = html.encode("utf-8", "replace")
    if path is None or len(data) > MAX_SNAPSHOT_BYTES:
        return False
    try:
        _private_dir(path.parent)
    except OSError:
        return False
    return _write_private(path, data)


@dataclass
class KeptCapture:
    """What one conversion left of a URL item's page, to keep with it."""

    html: str | None = None
    page: dict[str, Any] | None = None
    screenshot: dict[str, Any] | None = None
    blocks: bytes | None = None
    compare: list[bytes] = field(default_factory=list)


def capture_to_keep(
    job: Job,
    item: JobItem,
    page_html: str | None,
    sink: Any,
    url: str,
) -> KeptCapture | None:
    """What to keep of a standing result's page load (None: nothing new).

    The page is the HTML the content came from, else a page the browser
    set aside as a verification or blank page (so Compare can say what the
    site showed), but only for an item that keeps no page yet: a retry
    whose browser met a check page keeps the item's earlier page and
    screenshots (contract v1.14 "A retry that loads no page"; Rust keeps
    nothing of such a page). The screenshot view counts only when its
    screenshots are the item's own, or compare screenshots taken because
    it has none.
    """
    from markitai.page_capture import MAX_BLOCKS_BYTES, page_quality

    kept = KeptCapture()
    if page_html is not None:
        if len(page_html.encode("utf-8", "replace")) <= MAX_SNAPSHOT_BYTES:
            kept.html = page_html
            kept.page = page_quality(url, page_html)
    elif (
        sink is not None
        and sink.refused is not None
        and not kept_capture_files(job, item)
    ):
        kept.html, kept.page = sink.refused
    view = sink.view if sink is not None else None
    if view is not None:
        own = item_screenshots(job, item)
        if view.screenshots:
            matched = bool(own) and [p.resolve() for p in own] == [
                Path(p).resolve() for p in view.screenshots
            ]
        else:
            matched = not own and bool(view.compare)
            kept.compare = list(view.compare[:MAX_COMPARE_SHOTS]) if matched else []
        if matched:
            kept.screenshot = view.quality
            if view.blocks is not None and len(view.blocks) <= MAX_BLOCKS_BYTES:
                kept.blocks = view.blocks
    if kept.html is None and kept.screenshot is None:
        return None
    return kept


def keep_capture(job: Job, item: JobItem, kept: KeptCapture) -> None:
    """Keep *kept* as the URL item's page files, replacing all older ones.

    Each file is written on its own (one that cannot be written is only
    missing); a file the new capture does not have goes. Compare
    screenshots are written to a new folder first and moved in whole: a gap
    would misplace the tiles after it.
    """
    import shutil

    html_path = page_snapshot_path(job, item)
    record_path = capture_record_path(job, item)
    boxes_path = blocks_path(job, item)
    shots_path = compare_shots_dir(job, item)
    if None in (html_path, record_path, boxes_path, shots_path):
        return
    assert html_path and record_path and boxes_path and shots_path
    folder = html_path.parent
    try:
        _private_dir(folder)
    except OSError:
        return
    written: set[Path] = set()
    if kept.html is not None and keep_page_snapshot(job, item, kept.html):
        written.add(html_path)
    shots = False
    if kept.compare:
        staging = folder / f".{shots_path.name}.{uuid.uuid4().hex}.tmp"
        try:
            staging.mkdir(mode=0o700)
            for number, data in enumerate(kept.compare, start=1):
                if not _write_private(staging / f"{number}.jpg", data):
                    raise OSError("compare screenshot not written")
            if shots_path.is_symlink() or shots_path.is_file():
                shots_path.unlink()
            elif shots_path.is_dir():
                shutil.rmtree(shots_path)
            os.replace(staging, shots_path)
            shots = True
        except OSError:
            shutil.rmtree(staging, ignore_errors=True)
    # The screenshot verdict and boxes go with screenshots that are kept:
    # the item's own, or the compare ones just written
    shown = kept.screenshot is not None and (shots or not kept.compare)
    screenshot = kept.screenshot if shown else None
    if shown and kept.blocks is not None and _write_private(boxes_path, kept.blocks):
        written.add(boxes_path)
    if kept.page is not None or screenshot is not None:
        record = {"version": 1, "page": kept.page, "screenshot": screenshot}
        if _write_private(record_path, json.dumps(record).encode()):
            written.add(record_path)
    for path in (html_path, record_path, boxes_path):
        if path not in written:
            path.unlink(missing_ok=True)
    if not shots and not shots_path.is_symlink() and shots_path.is_dir():
        shutil.rmtree(shots_path, ignore_errors=True)


def _quality(value: Any) -> dict[str, Any] | None:
    """A recorded ``CaptureQuality``, or None when it is not one."""
    if not isinstance(value, dict) or value.get("state") not in CAPTURE_STATES:
        return None
    length = value.get("text_length")
    title, reason = value.get("title"), value.get("reason")
    if not isinstance(length, int) or isinstance(length, bool) or length < 0:
        return None
    if not (title is None or isinstance(title, str)):
        return None
    if not (reason is None or isinstance(reason, str)):
        return None
    return {
        "state": value["state"],
        "text_length": length,
        "title": title,
        "reason": reason,
    }


def recorded_capture(
    job: Job, item: JobItem
) -> tuple[dict[str, Any] | None, dict[str, Any] | None] | None:
    """``capture.json``'s ``(page, screenshot)``; None without a usable one."""
    path = _plain_file(capture_record_path(job, item))
    if path is None:
        return None
    try:
        with path.open("rb") as fh:
            data = fh.read(MAX_CAPTURE_RECORD + 1)
        record = json.loads(data) if len(data) <= MAX_CAPTURE_RECORD else None
    except (OSError, ValueError):
        return None
    if not isinstance(record, dict) or record.get("version") != 1:
        return None
    page, screenshot = record.get("page"), record.get("screenshot")
    if (page is not None and _quality(page) is None) or (
        screenshot is not None and _quality(screenshot) is None
    ):
        return None
    return _quality(page), _quality(screenshot)


_judged_pages: dict[tuple[str, int, int], dict[str, Any]] = {}
_judged_lock = threading.Lock()


def judged_page(path: Path, url: str) -> dict[str, Any] | None:
    """An older job's kept page judged now (cached while it is unchanged)."""
    from markitai.page_capture import page_quality

    try:
        stat = path.stat()
        key = (str(path), stat.st_mtime_ns, stat.st_size)
        with _judged_lock:
            if key in _judged_pages:
                return _judged_pages[key]
        if stat.st_size > MAX_SNAPSHOT_BYTES:
            return None
        quality = page_quality(url, path.read_bytes().decode("utf-8", "replace"))
    except OSError:
        return None
    with _judged_lock:
        if len(_judged_pages) >= 256:
            _judged_pages.pop(next(iter(_judged_pages)))
        _judged_pages[key] = quality
    return quality


def item_url(item: JobItem) -> str | None:
    """A URL item's address: its source, or for a job read back from disk
    (which keeps no source) its name."""
    if item.kind != "url":
        return None
    if isinstance(item.source, str) and item.source:
        return item.source
    return item.name or None


def capture_of(job: Job, item: JobItem, snapshot: Path | None) -> dict[str, Any]:
    """A URL item's ``capture`` (contract v1.14): as recorded, else its kept
    page judged now (an older job) and no screenshot verdict."""
    recorded = recorded_capture(job, item)
    if recorded is not None:
        page, screenshot = recorded
        return {"page": page, "screenshot": screenshot}
    page = None
    if snapshot is not None:
        page = judged_page(snapshot, item_url(item) or "")
    return {"page": page, "screenshot": None}


def _blocks_fit(path: Path, pages: list[tuple[float, float]]) -> bool:
    """Whether kept block boxes cover the screenshots listed as *pages*:
    their width, and their heights stacked (tiles cut from the top)."""
    try:
        with path.open("rb") as fh:
            head = json.loads(fh.read(MAX_CAPTURE_RECORD * 64 + 1))
    except (OSError, ValueError):
        return False
    if not isinstance(head, dict) or head.get("version") != 1 or not pages:
        return False
    width, height = head.get("width"), head.get("height")
    if not isinstance(width, int | float) or not isinstance(height, int | float):
        return False
    return abs(max(w for w, _ in pages) - width) <= 2 and (
        abs(sum(h for _, h in pages) - height) <= 2 * len(pages)
    )


def kept_original(job: Job, item: JobItem) -> tuple[Path, str] | None:
    """The stored original bytes and the file name they download as.

    An upload keeps its own name; a URL item's page is ``<output base>.html``.
    """
    upload = kept_upload(job, item)
    if upload is not None:
        return upload, item.name
    snapshot = kept_snapshot(job, item)
    if snapshot is None:
        return None
    from markitai.serve.jobs import item_base_name

    return snapshot, f"{item_base_name(item) or 'page'}.html"


#: The policy of a framed HTML original (contract v1.13): the site's own
#: pictures, style sheets, fonts and media load over HTTPS (relative
#: addresses resolve to the site through the inserted ``<base>``); no script
#: runs, no form is sent, links open outside the frame, and only the
#: workbench may frame it. The frame shares the workbench's origin
#: (``allow-same-origin``) so the workbench's own script can read its blocks
#: and link them with the Markdown; that matters only to script, and the
#: page runs none: there is no ``allow-scripts`` and no ``script-src``, so
#: ``default-src 'none'`` refuses every script, inline handler and
#: ``javascript:`` URL; no ``allow-forms`` (and ``form-action 'none'``) and
#: no ``allow-top-navigation``. Never add any of those.
VIEW_POLICY = (
    "sandbox allow-same-origin allow-popups allow-popups-to-escape-sandbox; "
    "default-src 'none'; "
    "img-src https: data:; style-src https: 'unsafe-inline'; "
    "font-src https: data:; media-src https:; "
    "form-action 'none'; frame-ancestors 'self'"
)
_HTML_WHITESPACE = " \t\n\r\x0c"
_ATTRIBUTE_ESCAPES = str.maketrans(
    {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}
)
_START_TAGS = {
    name: re.compile(f"<{name}", re.IGNORECASE | re.ASCII) for name in ("head", "html")
}


def _after_start_tag(page: str, name: str) -> int | None:
    """Where the first ``<name…>`` start tag ends (just after its ``>``).

    ``<name`` matches ASCII case-insensitively and must be followed by
    white space, ``>`` or ``/`` (so ``<header>`` is not ``<head>``).
    """
    pattern = _START_TAGS[name]
    start = 0
    while (found := pattern.search(page, start)) is not None:
        after = found.end()
        if after >= len(page):
            return None
        if page[after] in _HTML_WHITESPACE or page[after] in ">/":
            end = page.find(">", after)
            return None if end < 0 else end + 1
        start = after
    return None


def framed_page(page: str, url: str | None) -> str:
    """The kept page with one insertion for the workbench's sandboxed frame.

    No DNS prefetching, and a base that resolves relative links against the
    page's address (a URL item) and opens them outside the frame. It goes
    after the first ``<head>`` tag, else after the first ``<html>`` tag,
    else at the start after a leading doctype (contract v1.11 section 7).
    """
    base = (
        f'<base href="{url.translate(_ATTRIBUTE_ESCAPES)}" target="_blank">'
        if url is not None
        else '<base target="_blank">'
    )
    insertion = f'<meta http-equiv="x-dns-prefetch-control" content="off">{base}'
    at = _after_start_tag(page, "head")
    if at is None:
        at = _after_start_tag(page, "html")
    if at is None:
        lead = len(page) - len(page.lstrip("\ufeff" + _HTML_WHITESPACE))
        if page[lead : lead + 9].lower() == "<!doctype":
            end = page.find(">", lead)
            at = end + 1 if end >= 0 else lead
        else:
            at = len(page) - len(page.lstrip("\ufeff"))
    return page[:at] + insertion + page[at:]


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
    # A URL item's pages: its own screenshots, else its compare screenshots
    screenshots = item_screenshots(job, item) or compare_shots(job, item)
    snapshot = kept_snapshot(job, item)
    if snapshot is not None:
        # The page as fetched, with its screenshots as its pages
        original = _with_screenshots(
            Original(kind="html", name=item.name), screenshots, boxes
        )
        original.kind, original.path = "html", snapshot
        original.media_type, original.size = "text/html", snapshot.stat().st_size
    else:
        original = Original(kind="none", name=item.name, reason="not_kept")
        if screenshots:
            original = _with_screenshots(original, screenshots, boxes)
    if item.kind == "url" and boxes:
        original.capture = capture_of(job, item, snapshot)
        kept = kept_blocks(job, item) if screenshots else None
        if kept is not None and _blocks_fit(kept, original.pages):
            original.blocks = kept
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


def _init_page_worker() -> None:
    """Ignore Ctrl-C and end with the server.

    A terminal's Ctrl-C reaches every process attached to it; the server
    stops its workers itself (``PdfWorkers.close``), and an idle worker's own
    KeyboardInterrupt would only print a traceback. A server that is killed
    instead (``TerminateProcess`` on Windows, a crash) cannot stop them: they
    end on their own, or they would keep its working directory open.
    """
    import signal

    from markitai.utils.executor import exit_with_parent

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    exit_with_parent()


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
                    initializer=_init_page_worker,
                )
            return self._pool

    def _drop(self, pool: Any, *, kill: bool) -> None:
        with self._lock:
            if self._pool is pool:
                self._pool = None
        processes = list((getattr(pool, "_processes", None) or {}).values())
        # Taken before shutdown(), which drops the pool's reference to it
        manager = getattr(pool, "_executor_manager_thread", None)
        pool.shutdown(wait=False, cancel_futures=True)
        if not kill:
            return
        for process in processes:
            try:
                process.kill()
            except Exception:  # already gone
                pass
        # Wait (bounded) for the workers, then for the pool's manager thread,
        # which holds the queues: their semaphores are released only when
        # the queues are freed, and a server leaving before that got "There
        # appear to be 5 leaked semaphore objects" from the resource tracker
        # on every stop after a Compare (as converter.pdf_parallel does)
        for process in processes:
            try:
                process.join(timeout=1)
            except Exception:
                pass
        if manager is not None:
            manager.join(timeout=2)
        del manager

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
