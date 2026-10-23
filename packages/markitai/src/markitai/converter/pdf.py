"""PDF document converter."""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, NamedTuple, Protocol, TypeVar, cast

from loguru import logger

from markitai.constants import (
    ASSETS_REL_PATH,
    DEFAULT_OCR_LANG,
    DEFAULT_RENDER_DPI,
    PAGE_MARKER_RE,
    SCREENSHOTS_REL_PATH,
    page_marker,
)
from markitai.converter.base import (
    BaseConverter,
    ConvertResult,
    ExtractedImage,
    FileFormat,
    append_screenshot_comments,
    register_converter,
)
from markitai.converter.onnx_threads import import_pymupdf4llm
from markitai.converter.pdf_layout_boxes import (
    PICTURE_TEXT,
    clean_layout_chunks,
    picture_as_text,
)
from markitai.converter.pdf_markdown import repair_pages
from markitai.converter.pdf_ocr import (
    PageShot,
    read_page,
    read_page_task,
    shoot_page,
    shoot_page_task,
)
from markitai.converter.pdf_parallel import (
    map_page_runs,
    run_page_tasks,
    to_markdown_chunks,
)
from markitai.converter.pdf_tables import restore_flattened_tables
from markitai.converter.pdf_text_trust import (
    FAILED,
    NO_TEXT_LAYER,
    TRUSTED,
    GlyphCheck,
    LayerVerdict,
    failed,
    judge_text,
)
from markitai.image import ImageProcessor
from markitai.notices import user_notice
from markitai.ocr import (
    OCR_INSTALL_HINT,
    OCRBackendMissing,
    OCRError,
    OCRLanguageError,
    is_ocr_available,
    ocr_lang_for_script,
    ocr_readable_scripts,
)
from markitai.utils.cjk_radicals import fold_cjk_radicals
from markitai.utils.errors import ConversionError, MissingDependencyError
from markitai.utils.mime import get_mime_type, normalize_image_extension
from markitai.utils.page_chrome import strip_repeated_page_lines
from markitai.utils.paths import (
    create_tracked_temp_dir,
    ensure_assets_dir,
    ensure_screenshots_dir,
)
from markitai.utils.text import extract_asset_image_names, page_ranges
from markitai.vision_consent import ensure_vlm_ocr_disclosed, vlm_ocr_allowed

if TYPE_CHECKING:
    from markitai.config import MarkitaiConfig

# The layout models load as it is imported: on at most four threads each
pymupdf4llm = import_pymupdf4llm()

# --- Scanned/garbled page advisory --------------------------------------------

# A page with less than this much extracted text is "near-zero text".
_SCANNED_MAX_TEXT_CHARS = 50
# Minimum summed image coverage for a near-textless page to look scanned.
_SCANNED_MIN_IMAGE_COVERAGE = 0.5


def _page_image_coverage(page: Any) -> float:
    """Summed image-bbox area over page area, clamped to 1.0.

    Uses ``page.get_images``/``page.get_image_rects`` only (no rendering),
    so overlapping images can inflate the raw sum -- read it as "summed
    image-bbox area", not unique covered area.
    """
    page_area = page.rect.get_area()
    if page_area <= 0:
        return 0.0
    total = 0.0
    for xref in {img[0] for img in page.get_images(full=True)}:
        try:
            rects = page.get_image_rects(xref)
        except Exception:
            continue
        total += sum(rect.get_area() for rect in rects)
    return min(total / page_area, 1.0)


def collect_page_verdicts(
    doc: Any,
    pages: Iterable[int] | None = None,
    *,
    glyph_check: GlyphCheck | None = None,
) -> dict[int, LayerVerdict]:
    """Judge each page's text layer (``converter.pdf_text_trust``).

    A page with less than ``_SCANNED_MAX_TEXT_CHARS`` of text has no text
    layer to speak of: ``scanned`` when pictures cover half of it or more,
    else ``empty``. Any other page gets the text-only checks of
    :func:`~markitai.converter.pdf_text_trust.judge_text`, which render
    nothing, so this is cheap even for large documents; with a
    *glyph_check* (``--ocr``) a page they trust is also compared with what
    its glyphs draw.

    Args:
        doc: An open pymupdf Document
        pages: 0-based pages to check (all when None)
        glyph_check: Compare the text with OCR of a few crops, per font

    Returns:
        1-based page number -> verdict
    """
    verdicts: dict[int, LayerVerdict] = {}
    for i in range(doc.page_count) if pages is None else pages:
        page = doc[i]
        text = page.get_text().strip()
        if len(text) < _SCANNED_MAX_TEXT_CHARS:
            scanned = _page_image_coverage(page) >= _SCANNED_MIN_IMAGE_COVERAGE
            verdicts[i + 1] = LayerVerdict(
                "scanned" if scanned else "empty",
                f"almost no text layer ({len(text)} characters)"
                if text
                else NO_TEXT_LAYER,
            )
        else:
            verdict = judge_text(text)
            if verdict.status == "trusted" and glyph_check is not None:
                why = glyph_check.judge(page)
                if why is not None:
                    verdict = failed(why)
            verdicts[i + 1] = verdict
    return verdicts


def collect_page_advisories(
    doc: Any, pages: Iterable[int] | None = None
) -> tuple[list[int], list[int]]:
    """Compute cheap per-page scan/garbled signals for an open PDF document.

    A page with near-zero extracted text but significant image coverage
    looks scanned; a page whose text layer fails the checks of
    :func:`collect_page_verdicts` (a broken font map, mojibake, reversed
    right-to-left text, consonant soup) is garbled.

    Args:
        doc: An open pymupdf Document
        pages: 0-based pages to check (all when None)

    Returns:
        Tuple of (scanned-looking page numbers, garbled page numbers),
        both 1-based
    """
    return _advisory_pages(collect_page_verdicts(doc, pages))


def _advisory_pages(
    verdicts: dict[int, LayerVerdict],
) -> tuple[list[int], list[int]]:
    """(scanned-looking pages, pages whose text layer failed), in order."""
    pages = sorted(verdicts)
    return (
        [p for p in pages if verdicts[p].status == "scanned"],
        [p for p in pages if verdicts[p].status == "failed"],
    )


def _ocr_reads(verdict: LayerVerdict, readable: frozenset[str]) -> bool:
    """Whether OCR can read a page whose text layer failed: the checks know
    of no writing system for it, or the configured recognizer reads it."""
    return verdict.script is None or verdict.script in readable


def _collect_native_text_pages(
    doc: Any,
    *,
    readable: frozenset[str],
    glyph_check: GlyphCheck | None = None,
) -> tuple[list[int], dict[int, LayerVerdict]]:
    """The pages whose text layer passes the checks, and every verdict.

    Mirrors :func:`collect_page_verdicts`: a page qualifies for native
    extraction when it carries a meaningful text layer that the checks
    trust. Everything else (near-textless / scanned / failed pages) is left
    for OCR, except a failed page in a writing system the recognizer does
    not read (*readable*): OCR would only make it worse, so it keeps its
    text layer.

    Args:
        doc: An open pymupdf Document
        readable: The writing systems the configured OCR language reads
        glyph_check: Also compare each page's text with what its glyphs
            draw (see :func:`collect_page_verdicts`)

    Returns:
        (0-based native pages in order, 1-based page number -> verdict)
    """
    verdicts = collect_page_verdicts(doc, glyph_check=glyph_check)
    native = [
        number - 1
        for number, verdict in sorted(verdicts.items())
        if verdict.status == "trusted"
        or (verdict.status == "failed" and not _ocr_reads(verdict, readable))
    ]
    return native, verdicts


# --- where each page's text came from ------------------------------------------


def _page_source(page: int, source: str, reason: str) -> dict[str, Any]:
    """One entry of ``metadata["page_sources"]``: ``source`` is ``text`` (the
    text layer), ``ocr`` or ``mixed`` (the text layer with OCR'd pictures,
    or a vision model reading the page beside it)."""
    return {"page": page, "source": source, "reason": reason}


def _text_layer_sources(
    pages: Iterable[int], verdicts: dict[int, LayerVerdict]
) -> list[dict[str, Any]]:
    """Every page from its text layer (no OCR), with what the checks said."""
    return [
        _page_source(
            page,
            "text",
            verdicts[page].reason if page in verdicts else "text layer (not checked)",
        )
        for page in pages
    ]


#: Why every page is OCR'd when per-page routing is off
OCR_FORCED = "OCR forced by --ocr (ocr.per_page_routing is off)"
#: Why every page is a vision model's reading of its image and text layer
VLM_READ = (
    "OCR forced by --ocr: the vision model reads the page image beside its text layer"
)


def _ocr_sources(
    total_pages: int,
    *,
    routing: bool,
    native_pages: Iterable[int],
    verdicts: dict[int, LayerVerdict],
    empty_pages: Iterable[int],
    picture_pages: Iterable[int],
    ocr_lang: str,
) -> list[dict[str, Any]]:
    """Where each page's text came from on the ``--ocr`` path (RapidOCR).

    *native_pages*, *empty_pages* (OCR'd, nothing found) and
    *picture_pages* (native pages whose pictures OCR read text from) are
    0-based; *verdicts* are the routing check's, by 1-based page.
    """
    native, empty, pictures = set(native_pages), set(empty_pages), set(picture_pages)
    sources = []
    for index in range(total_pages):
        page = index + 1
        verdict = verdicts.get(page)
        if index in native:
            reason = verdict.reason if verdict else TRUSTED
            if verdict is not None and verdict.status == "failed":
                script = verdict.script or "this script"
                reason += (
                    f"; kept, since OCR with ocr.lang {ocr_lang} does not read {script}"
                )
            if index in pictures:
                sources.append(
                    _page_source(page, "mixed", f"{reason}; text in its pictures OCR'd")
                )
            else:
                sources.append(_page_source(page, "text", reason))
            continue
        if not routing:
            reason = OCR_FORCED
        elif verdict is None:
            reason = "OCR forced by --ocr (the text-layer check could not run)"
        elif verdict.status == "trusted":
            reason = "text layer trusted, but its extraction failed"
        else:
            reason = verdict.reason
        if index in empty:
            reason += "; OCR found no text"
        sources.append(_page_source(page, "ocr", reason))
    return sources


def _log_page_sources(input_path: Path, sources: list[dict[str, Any]]) -> None:
    """Debug-log where the pages' text came from, and why when not trusted."""
    counts: dict[str, int] = {}
    for entry in sources:
        counts[entry["source"]] = counts.get(entry["source"], 0) + 1
        if entry["reason"] != TRUSTED:
            logger.debug(
                "[PDF] {} page {}: {} ({})",
                input_path.name,
                entry["page"],
                entry["source"],
                entry["reason"],
            )
    if counts:
        logger.debug(
            "[PDF] {}: page text from {}",
            input_path.name,
            ", ".join(f"{name} {n}" for name, n in sorted(counts.items())),
        )


# Pages read at once in this process, at most: its engine already spreads
# each page over every core (see PdfConverter._ocr_thread_count).
_MAX_OCR_THREADS = 8

# Pictures on native pages smaller than this (pixels at the render DPI, about
# 1.8 square inches at 150 DPI) are icons, logos or rules: not worth an OCR pass.
_PICTURE_OCR_MIN_PIXELS = 40_000


def _check_openable(input_path: Path) -> None:
    """Refuse a PDF that opens but cannot be converted, with a clear reason.

    A password-protected PDF failed deep in pymupdf4llm with ``TypeError:
    'NoneType' object is not subscriptable``, one without pages with
    ``IndexError: range object index out of range``. A file PyMuPDF cannot
    open at all is left to the extraction, which reports its own error.

    Raises:
        ConversionError: The PDF needs a password, or has no pages.
    """
    import pymupdf

    try:
        doc = pymupdf.open(input_path)
    except Exception:
        return
    try:
        if doc.needs_pass:
            raise ConversionError(
                f"{input_path.name} is password-protected; decrypt it before converting"
            )
        if doc.page_count == 0:
            raise ConversionError(f"{input_path.name} has no pages")
    finally:
        doc.close()


def _document_title(input_path: Path) -> str:
    """The title ``converter.pdf_title`` reads from the PDF, or ``""``."""
    import pymupdf

    from markitai.converter.pdf_title import document_title

    try:
        with pymupdf.open(input_path) as doc:
            return document_title(doc)
    except Exception as e:
        logger.debug("[PDF] Title not read: {}", e)
        return ""


# --- Hidden-text / prompt-injection sanitization -------------------------------

# Spans below this font size (pt) are effectively invisible when rendered.
_HIDDEN_MAX_FONT_SIZE = 2.0
# Fill luminance above this is a white-on-white candidate, when what the
# page shows under the span (sampled from a low-resolution render) is that
# light too: white text on a dark banner is visible.
_HIDDEN_MIN_LUMINANCE = 0.95
# Scale of the page render the background is sampled from (36 dpi).
_BACKGROUND_SAMPLE_SCALE = 0.5
# Invisible text with a picture under this share of its bbox is an OCR layer,
# on a page that looks scanned: a picture covers this share of the page, or
# the page shows no other text.
_OCR_LAYER_MIN_COVER = 0.8
_SCAN_PICTURE_MIN_PAGE_SHARE = 0.6
# Max characters of hidden text quoted in the consolidated warning.
_HIDDEN_EXCERPT_MAX_CHARS = 80
# A picture-text block "remove" left with nothing in it.
_EMPTY_PICTURE_TEXT = re.compile(
    r"(?:<!-- Start of picture text -->|\*\*----- Start of picture text -----\*\*<br>)"
    r"(?:\s|<br>)*"
    r"(?:<!-- End of picture text -->|\*\*----- End of picture text -----\*\*<br>)\n?"
)


def _pages_phrases(pages: Sequence[int]) -> tuple[str, str]:
    """``("1 page", "page 3")`` or ``("4 pages", "pages 1–3, 9")`` for a
    notice: the pages as ranges."""
    numbers = page_ranges(pages)
    if len(set(pages)) == 1:
        return "1 page", f"page {numbers}"
    return f"{len(set(pages))} pages", f"pages {numbers}"


def _color_luminance(color: int) -> float:
    """Rec. 601 luminance of a packed sRGB int color, in [0, 1]."""
    r = (color >> 16) & 0xFF
    g = (color >> 8) & 0xFF
    b = color & 0xFF
    return (0.299 * r + 0.587 * g + 0.114 * b) / 255.0


def _bbox_fully_outside(
    bbox: tuple[float, float, float, float],
    page_rect: tuple[float, float, float, float],
) -> bool:
    """True when ``bbox`` does not intersect ``page_rect`` (the CropBox)."""
    x0, y0, x1, y1 = bbox
    px0, py0, px1, py1 = page_rect
    return x1 <= px0 or x0 >= px1 or y1 <= py0 or y0 >= py1


def _background_sampler(
    page: Any,
) -> Callable[[tuple[float, float, float, float]], float | None]:
    """Mean luminance, in [0, 1], of what *page* shows under a bbox.

    The page is rendered once, in gray at low resolution, and only when a
    span asks for it. None when the render failed or the bbox is off the
    page, so the caller falls back to assuming a white page.
    """
    pix: Any = None

    def sample(bbox: tuple[float, float, float, float]) -> float | None:
        nonlocal pix
        try:
            import pymupdf

            matrix = pymupdf.Matrix(_BACKGROUND_SAMPLE_SCALE, _BACKGROUND_SAMPLE_SCALE)
            if pix is None:
                pix = page.get_pixmap(
                    matrix=matrix, colorspace=pymupdf.csGRAY, alpha=False
                )
            # Span bboxes are unrotated; the render is of the rotated page
            area = (pymupdf.Rect(bbox) * page.rotation_matrix * matrix).irect
            area &= pix.irect
            if area.is_empty:
                return None
            samples = pix.samples
            total = 0
            for y in range(area.y0, area.y1):
                row = (y - pix.y) * pix.stride - pix.x
                total += sum(samples[row + area.x0 : row + area.x1])
            return total / (area.width * area.height * 255)
        except Exception as e:
            logger.debug("[PDF] Background sampling failed: {}", e)
            return None

    return sample


def _page_pictures(page: Any) -> list[tuple[float, float, float, float]]:
    """The bboxes of the pictures drawn on *page* (none when unreadable)."""
    try:
        return [tuple(info["bbox"]) for info in page.get_image_info()]
    except Exception as e:
        logger.debug("[PDF] Pictures of page {} not read: {}", page.number + 1, e)
        return []


def _ocr_layer_pictures(
    page: Any,
    data: dict[str, Any],
    page_rect: tuple[float, float, float, float],
    background: Callable[[tuple[float, float, float, float]], float | None]
    | None = None,
) -> list[tuple[float, float, float, float]]:
    """The pictures invisible text on *page* may be an OCR layer of.

    A searchable scan's page is a picture covering most of the page, or
    pictures with no other text a reader sees: only there is invisible text
    over a picture taken for the scanner's OCR layer. On any other page (a
    text page with a logo or a small decoration) invisible text is hidden
    text, over a picture or not. *data* is the page's ``get_text("dict")``.
    """
    pictures = _page_pictures(page)
    if not pictures:
        return []
    x0, y0, x1, y1 = page_rect
    page_area = (x1 - x0) * (y1 - y0)
    if page_area > 0:
        for px0, py0, px1, py1 in pictures:
            width = min(x1, px1) - max(x0, px0)
            height = min(y1, py1) - max(y0, py0)
            if (
                width > 0
                and height > 0
                and width * height >= _SCAN_PICTURE_MIN_PAGE_SHARE * page_area
            ):
                return pictures
    for block in data.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "")
                if (
                    isinstance(text, str)
                    and text.strip()
                    and not _is_hidden_span(span, page_rect, background)
                ):
                    return []
    return pictures


def _over_picture(
    bbox: Any, pictures: Sequence[tuple[float, float, float, float]]
) -> bool:
    """Whether a picture covers ``_OCR_LAYER_MIN_COVER`` of *bbox*."""
    if not isinstance(bbox, (tuple, list)) or len(bbox) != 4:
        return False
    x0, y0, x1, y1 = bbox
    area = (x1 - x0) * (y1 - y0)
    if area <= 0:
        return False
    for px0, py0, px1, py1 in pictures:
        width = min(x1, px1) - max(x0, px0)
        height = min(y1, py1) - max(y0, py0)
        if width > 0 and height > 0 and width * height >= _OCR_LAYER_MIN_COVER * area:
            return True
    return False


def _is_hidden_span(
    span: dict[str, Any],
    page_rect: tuple[float, float, float, float],
    background: Callable[[tuple[float, float, float, float]], float | None]
    | None = None,
    pictures: Sequence[tuple[float, float, float, float]] = (),
) -> bool:
    """Classify a pymupdf text span as hidden (invisible to a human reader).

    A span is hidden when it is drawn at zero opacity (pymupdf maps text
    render mode 3 to ``alpha == 0``) anywhere but over one of the page's
    *pictures*, its font size is below 2pt, its fill color is near-white
    on a near-white background (sampled with *background*; assumed white
    without one), or its bbox lies fully outside the page CropBox.

    Invisible text over a picture of a scanned page (the *pictures* of
    :func:`_ocr_layer_pictures`) is a searchable scan's OCR layer
    (OCRmyPDF, Acrobat, ABBYY and scanners write one): it says what the
    picture shows, and is the only text such a document has.
    """
    text = span.get("text", "")
    if not isinstance(text, str) or not text.strip():
        return False
    if span.get("alpha", 255) == 0 and not _over_picture(span.get("bbox"), pictures):
        return True
    size = span.get("size")
    if isinstance(size, (int, float)) and size < _HIDDEN_MAX_FONT_SIZE:
        return True
    color = span.get("color")
    bbox = span.get("bbox")
    if isinstance(color, int) and _color_luminance(color) > _HIDDEN_MIN_LUMINANCE:
        under = (
            background(tuple(bbox))
            if background is not None
            and isinstance(bbox, (tuple, list))
            and len(bbox) == 4
            else None
        )
        if under is None or under > _HIDDEN_MIN_LUMINANCE:
            return True
    if (
        isinstance(bbox, (tuple, list))
        and len(bbox) == 4
        and _bbox_fully_outside(tuple(bbox), page_rect)
    ):
        return True
    return False


def collect_hidden_text(
    doc: Any, pages: Iterable[int] | None = None
) -> dict[int, list[str]]:
    """Detect hidden text spans (prompt-injection vector) in an open PDF.

    Scans ``page.get_text("dict")`` spans using a textpage clipped to the
    MediaBox so text placed outside the CropBox is also examined. See
    :func:`_is_hidden_span` for the detection criteria.

    Args:
        doc: An open pymupdf Document
        pages: 0-based pages to scan (all when None)

    Returns:
        Mapping of 1-based page number -> list of hidden span texts
    """
    hidden: dict[int, list[str]] = {}
    for i in range(doc.page_count) if pages is None else pages:
        page = doc[i]
        try:
            page_rect = tuple(page.rect)
            textpage = page.get_textpage(clip=page.mediabox)
            data = page.get_text("dict", textpage=textpage)
        except Exception as e:
            logger.debug("[PDF] Hidden-text scan failed for page {}: {}", i + 1, e)
            continue
        background = _background_sampler(page)
        pictures = _ocr_layer_pictures(page, data, page_rect, background)
        texts: list[str] = []
        for block in data.get("blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    if _is_hidden_span(span, page_rect, background, pictures):
                        texts.append(span["text"].strip())
        if texts:
            hidden[i + 1] = texts
    return hidden


def _scan_visible_text(
    input_path: Path | str, pages: Iterable[int]
) -> dict[int, str] | None:
    """The text a reader sees on each of *pages* (0-based); None on failure.

    The complement of :func:`collect_hidden_text`, keyed by 1-based page:
    the "remove" mode checks a hidden span against it so a hidden copy of
    text that is also visible on the page does not delete the visible one.
    """
    try:
        import pymupdf

        visible: dict[int, str] = {}
        with pymupdf.open(Path(input_path)) as doc:
            for i in pages:
                page = doc[i]
                page_rect = tuple(page.rect)
                data = cast(
                    dict[str, Any],
                    page.get_text(
                        "dict", textpage=page.get_textpage(clip=page.mediabox)
                    ),
                )
                background = _background_sampler(page)
                pictures = _ocr_layer_pictures(page, data, page_rect, background)
                visible[i + 1] = "\n".join(
                    "".join(
                        span.get("text", "")
                        for span in line.get("spans", [])
                        if not _is_hidden_span(span, page_rect, background, pictures)
                    )
                    for block in data.get("blocks", [])
                    for line in block.get("lines", [])
                )
        return visible
    except Exception as e:
        logger.debug("[PDF] Visible-text scan failed for {}: {}", input_path, e)
        return None


def _chunk_text(chunk: Any) -> str:
    """The markdown of one ``page_chunks=True`` result."""
    return chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)


def _chunk_page_number(chunk: Any, index: int) -> int:
    """The page a chunk came from.

    pymupdf4llm reports it in the chunk's metadata; position in the list is
    the fallback for a chunk that carries none.
    """
    if isinstance(chunk, dict):
        number = chunk.get("metadata", {}).get("page_number")
        if isinstance(number, int) and number > 0:
            return number
    return index + 1


_T = TypeVar("_T")
_T_co = TypeVar("_T_co", covariant=True)


def _scan_hidden_text(
    input_path: Path | str, pages: list[int] | None = None
) -> dict[int, list[str]] | None:
    """``collect_hidden_text`` on *input_path*; None when detection failed.

    Module-level so an extraction worker can run it on a run of pages.
    """
    input_path = Path(input_path)
    try:
        import pymupdf

        with pymupdf.open(input_path) as doc:
            return collect_hidden_text(doc, pages)
    except Exception as e:
        logger.debug(
            "[PDF] Hidden-text detection failed for {}: {}", input_path.name, e
        )
        return None


class PageAdvisories(NamedTuple):
    """The scanned/garbled advisory of a document (or a run of its pages).

    Attributes:
        scanned: 1-based pages that look scanned
        garbled: 1-based pages whose text layer failed the checks
        verdicts: 1-based page -> verdict on its text layer
    """

    scanned: list[int]
    garbled: list[int]
    verdicts: dict[int, LayerVerdict]


def _scan_advisories(
    input_path: Path | str, pages: list[int] | None = None
) -> PageAdvisories | None:
    """``collect_page_verdicts`` on *input_path*; None when it failed.

    Module-level so an extraction worker can run it on a run of pages.
    """
    input_path = Path(input_path)
    try:
        import pymupdf

        with pymupdf.open(input_path) as doc:
            verdicts = collect_page_verdicts(doc, pages)
        return PageAdvisories(*_advisory_pages(verdicts), verdicts)
    except Exception as e:
        logger.debug(
            "[PDF] Scan/garbled advisory check failed for {}: {}",
            input_path.name,
            e,
        )
        return None


class _Pending(Protocol[_T_co]):
    """A result still being computed (a Future, or merged page runs)."""

    def result(self) -> _T_co: ...


class _MergedRuns(Generic[_T]):
    """The results of one check run over page runs in the workers, merged.

    Every page is checked: a run the workers do not finish (a worker died,
    a discarded pool cancelled it) is checked in this process instead
    (``PageRun.result``), so hidden text is never let through unscanned.
    A run whose check failed is left out; the check counts as failed (None)
    only when it failed on every run, as the in-process check fails as a
    whole.
    """

    def __init__(
        self,
        runs: Sequence[_Pending[_T | None]],
        merge: Callable[[list[_T]], _T],
    ) -> None:
        self._runs = runs
        self._merge = merge

    def result(self) -> _T | None:
        parts = [part for run in self._runs if (part := run.result()) is not None]
        return self._merge(parts) if parts else None


class _Resolved(Generic[_T]):
    """A result already in hand, where a pending one is expected."""

    def __init__(self, value: _T) -> None:
        self._value = value

    def result(self) -> _T:
        return self._value


def _merge_hidden(parts: list[dict[int, list[str]]]) -> dict[int, list[str]]:
    merged: dict[int, list[str]] = {}
    for part in parts:
        merged.update(part)
    return dict(sorted(merged.items()))


def _merge_advisories(parts: list[PageAdvisories]) -> PageAdvisories:
    verdicts: dict[int, LayerVerdict] = {}
    for part in parts:
        verdicts.update(part.verdicts)
    return PageAdvisories(
        sorted(p for part in parts for p in part.scanned),
        sorted(p for part in parts for p in part.garbled),
        dict(sorted(verdicts.items())),
    )


def _start_prescan(
    input_path: Path, *, hidden_text: bool
) -> tuple[
    _Pending[dict[int, list[str]] | None] | None,
    _Pending[PageAdvisories | None],
]:
    """Start the hidden-text and scanned-page checks alongside the extraction.

    Both read the PDF on their own and need nothing from the extraction, so
    they run while the pages are extracted instead of after it: in the
    extraction workers when the document goes to them (so this process,
    and an event loop it serves, keeps the GIL), else on a background
    thread (onnxruntime releases the GIL while it runs the layout model).
    """
    try:
        import pymupdf

        with pymupdf.open(input_path) as doc:
            page_count = doc.page_count
    except Exception:  # the extraction reports an unreadable file
        page_count = 0
    advisory_runs = map_page_runs(input_path, _scan_advisories, page_count)
    if advisory_runs is not None:
        hidden_runs = (
            map_page_runs(input_path, _scan_hidden_text, page_count)
            if hidden_text
            else None
        )
        return (
            _MergedRuns(hidden_runs, _merge_hidden) if hidden_runs else None,
            _MergedRuns(advisory_runs, _merge_advisories),
        )

    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="markitai-pdf-scan")
    try:
        hidden = executor.submit(_scan_hidden_text, input_path) if hidden_text else None
        advisories = executor.submit(_scan_advisories, input_path)
    finally:
        executor.shutdown(wait=False)
    return hidden, advisories


@register_converter(FileFormat.PDF)
class PdfConverter(BaseConverter):
    """Converter for PDF documents using pymupdf4llm.

    Supports OCR mode for scanned PDFs when --ocr flag is enabled.
    """

    supported_formats = [FileFormat.PDF]

    def __init__(self, config: MarkitaiConfig | None = None) -> None:
        super().__init__(config)
        from markitai.utils.suppress import _suppress_mupdf_logs

        _suppress_mupdf_logs()

    # Absolute cap on internal thread pool workers. PyMuPDF is not thread-safe,
    # so each worker opens its own document copy. These internal pools may run
    # *inside* the shared converter executor from workflow/core.py, so keeping
    # them bounded avoids excessive thread nesting and memory pressure.
    _MAX_INTERNAL_WORKERS = 6

    _IMAGE_REF_RE = re.compile(r"!\[[^\]]*\]\((?:[^)]+/)?([^)]+)\)")
    _TEXT_TOKEN_RE = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)?|[\u4e00-\u9fff]+")

    def _get_worker_count(self, input_path: Path, task_count: int) -> int:
        """Calculate optimal worker count based on file size and system resources.

        Each worker opens its own PDF copy, so memory usage scales with
        workers x file_size. Larger files use fewer workers to limit memory.

        Args:
            input_path: Path to the PDF file (used to check file size).
            task_count: Number of tasks (pages) to process.

        Returns:
            Optimal number of workers (at least 1, at most _MAX_INTERNAL_WORKERS).
        """
        file_size_mb = input_path.stat().st_size / (1024 * 1024)
        cpu_count = os.cpu_count() or 4

        if file_size_mb < 10:
            workers = min(cpu_count // 2 or 2, task_count, self._MAX_INTERNAL_WORKERS)
        elif file_size_mb < 50:
            workers = min(4, task_count)
        else:
            workers = min(2, task_count)

        return max(1, min(workers, self._MAX_INTERNAL_WORKERS))

    @staticmethod
    def _ocr_thread_count(task_count: int) -> int:
        """Threads reading pages in this process (no worker processes).

        One per two cores this process may run on, 8 at most: each page's
        engine run takes its share of the cores (``onnx_threads``), and
        the cores idle only while a page is in Python (layout, cropping).
        Memory may allow fewer.
        """
        from markitai.converter.pdf_parallel import ocr_slots, usable_cores

        threads = min(_MAX_OCR_THREADS, usable_cores() // 2, task_count)
        return max(1, min(threads, ocr_slots(max(1, threads))))

    def _is_text_heavy_picture_text(self, picture_text: str) -> bool:
        """Return True when picture text looks like extracted table/text content."""
        plain_text = picture_text.replace("<br>", "\n")
        lines = [line.strip() for line in plain_text.splitlines() if line.strip()]
        if not lines:
            return False

        total_tokens = 0
        long_lines = 0
        for line in lines:
            tokens = self._TEXT_TOKEN_RE.findall(line)
            total_tokens += len(tokens)
            cjk_chars = len(re.findall(r"[\u4e00-\u9fff]", line))
            if len(tokens) >= 4 or cjk_chars >= 8:
                long_lines += 1

        return total_tokens >= 20 and long_lines >= 3

    def _demote_reference_picture_blocks(
        self,
        page_chunk: dict[str, Any],
        page_num: int,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Remove inline refs for text-heavy picture blocks while keeping picture text."""
        text = page_chunk.get("text", "")
        page_boxes = page_chunk.get("page_boxes")
        if not isinstance(text, str) or not isinstance(page_boxes, list):
            return str(text), []

        rebuilt_parts: list[str] = []
        reference_images: list[dict[str, Any]] = []
        cursor = 0

        for box in page_boxes:
            if not isinstance(box, dict) or box.get("class") != "picture":
                continue
            pos = box.get("pos")
            if (
                not isinstance(pos, (tuple, list))
                or len(pos) != 2
                or not all(isinstance(v, int) for v in pos)
            ):
                continue

            start, end = pos
            if start < cursor or start < 0 or end > len(text) or start >= end:
                continue

            rebuilt_parts.append(text[cursor:start])
            segment = text[start:end]
            match = PICTURE_TEXT.search(segment)
            image_match = self._IMAGE_REF_RE.search(segment)
            as_text = picture_as_text(segment)
            if (
                match is not None
                and image_match is not None
                and as_text is not None
                and self._is_text_heavy_picture_text(match.group(1))
            ):
                image_name = image_match.group(1)
                reference_images.append(
                    {
                        "page": page_num,
                        "name": image_name,
                        "rel_path": f"{ASSETS_REL_PATH}/{image_name}",
                    }
                )
                # The text, readable: pymupdf4llm writes it into an HTML
                # comment under the picture, which no longer stands there
                rebuilt_parts.append(as_text)
            else:
                rebuilt_parts.append(segment)
            cursor = end

        rebuilt_parts.append(text[cursor:])
        return "".join(rebuilt_parts), reference_images

    def convert(
        self, input_path: Path, output_dir: Path | None = None
    ) -> ConvertResult:
        """
        Convert PDF document to Markdown.

        Every path (text layer, OCR, page renders for a vision model)
        reports the same ``title`` (``converter.pdf_title``): the PDF's own
        title when it names the document, else its most prominent text;
        without one the first heading that names it, then the file name,
        are the caller's (``utils.frontmatter.resolve_document_title``).

        Args:
            input_path: Path to the input file
            output_dir: Optional output directory for extracted images

        Returns:
            ConvertResult containing markdown and extracted images

        Raises:
            ConversionError: The PDF needs a password, or has no pages.
        """
        _check_openable(Path(input_path))
        result = self._convert(Path(input_path), output_dir)
        if not result.metadata.get("title"):
            title = _document_title(Path(input_path))
            if title:
                result.metadata["title"] = title
        # A text layer that names radicals for ideographs (Chrome's PDFs:
        # 使⽤, 右下⻆) reads as the ideographs, on every path
        if isinstance(result.markdown, str):
            result.markdown = fold_cjk_radicals(result.markdown)
        for key in ("title", "extracted_text"):  # the LLM paths read these
            value = result.metadata.get(key)
            if isinstance(value, str):
                result.metadata[key] = fold_cjk_radicals(value)
        return result

    def _convert(self, input_path: Path, output_dir: Path | None) -> ConvertResult:
        input_path = Path(input_path)
        images: list[ExtractedImage] = []

        # Check if OCR mode is enabled
        use_ocr = self.config and self.config.ocr.enabled
        use_llm = self.config and self.config.llm.enabled

        if use_ocr:
            if use_llm:
                # --ocr --llm: Render pages as images for LLM Vision analysis.
                # With MARKITAI_NO_VLM_OCR set, never send page images to a
                # remote model — degrade to local OCR or fail with a clear
                # message instead.
                if not vlm_ocr_allowed():
                    return self._degrade_vlm_ocr(input_path, output_dir)
                return self._render_pages_for_llm(input_path, output_dir)
            # --ocr only: Use RapidOCR for text extraction
            return self._convert_with_ocr(input_path, output_dir)

        # Determine image output path
        temp_dir: Path | None = None
        staging_dir: Path | None = None
        try:
            if output_dir:
                # pymupdf4llm never writes into the output dir itself: it
                # names images after a sanitized input name ("a b.pdf" and
                # "a_b.pdf" both become "a_b.pdf-...") and would overwrite the
                # ones an earlier output still references, and it mangles
                # the directory path too (see _new_image_staging_dir). Extract
                # into a private dir, then adopt the images into assets.
                staging_dir = self._new_image_staging_dir()
                image_path = staging_dir
                write_images = True
            else:
                # Use temp directory if no output dir specified
                temp_dir = Path(tempfile.mkdtemp())
                image_path = temp_dir
                write_images = True

            # Get image format from config
            image_format = "png"
            dpi = DEFAULT_RENDER_DPI
            if self.config:
                image_format = normalize_image_extension(self.config.image.format)

            sanitize_mode = self.config.security.pdf_sanitize if self.config else "warn"
            hidden_scan, advisory_scan = _start_prescan(
                input_path, hidden_text=sanitize_mode != "off"
            )

            # Convert using pymupdf4llm with page_chunks=True for page-level splitting
            # This allows proper text-to-screenshot alignment in batched LLM processing
            page_results = to_markdown_chunks(
                input_path,
                extract=pymupdf4llm.to_markdown,
                write_images=write_images,
                image_path=str(image_path),
                image_format=image_format,
                dpi=dpi,
            )

            # Merge page chunks and add page markers for proper splitting
            # Format: <!-- Page number: N --> (consistent with Slide number format)
            # Ensure blank line after marker for proper markdown formatting
            # Page headers/footers the layout model found (when they repeat or
            # are page numbers) out; lines it cropped into pictures back as text
            page_results = clean_layout_chunks(input_path, list(page_results))
            page_numbers: list[int] = []
            page_texts: list[str] = []
            reference_images: list[dict[str, Any]] = []
            for i, chunk in enumerate(page_results):
                page_num = _chunk_page_number(chunk, i)
                page_text = _chunk_text(chunk)
                if isinstance(chunk, dict):
                    page_text, page_references = self._demote_reference_picture_blocks(
                        chunk, page_num
                    )
                    reference_images.extend(page_references)
                page_numbers.append(page_num)
                page_texts.append(page_text)

            # Symbol-font glyphs mapped, headings plain text, body lines the
            # layout model took for headings back in their paragraphs
            page_texts = repair_pages(input_path, page_numbers, page_texts)

            # Strip running headers/footers repeated across page boundaries
            page_texts, stripped_lines = strip_repeated_page_lines(page_texts)
            if stripped_lines:
                logger.debug(
                    "[PDF] Stripped {} repeated header/footer line(s): {}",
                    len(stripped_lines),
                    sorted(stripped_lines),
                )

            # Hidden-text / prompt-injection sanitization (warn or remove)
            page_texts = self._sanitize_hidden_text(
                input_path, page_numbers, page_texts, prescanned=hidden_scan
            )

            # Tables without ruling lines that the extraction flattened into
            # one paragraph. After the sanitization: a paragraph that lost
            # hidden text no longer matches its table's words, so no hidden
            # text is split across cells where "remove" could not find it.
            page_texts = restore_flattened_tables(input_path, page_numbers, page_texts)

            markdown_parts = [
                f"{page_marker(page_num)}\n\n{page_text}"
                for page_num, page_text in zip(page_numbers, page_texts)
            ]

            markdown = "\n\n".join(markdown_parts)

            # Advisory only: warn when pages look scanned or garbled so the
            # user can re-run with --ocr (behavior is never changed here)
            # With an LLM that will be sent the page images, the model may
            # read those pages: the workflow then says it (or not) once it
            # knows, see ``workflow.core.settle_scan_notice``.
            config = self.config
            advisories = advisory_scan.result()
            scan_notice = self._warn_scanned_or_garbled(
                input_path,
                total_pages=len(page_numbers),
                prescanned=_Resolved(advisories),
                defer=bool(
                    config is not None
                    and config.llm.enabled
                    and not config.llm.pure
                    and config.screenshot.enabled
                    and output_dir
                ),
            )

            # Fix image paths in markdown: pymupdf4llm uses absolute/full paths,
            # we need relative paths (assets/xxx.jpg)
            markdown = self._fix_image_paths(markdown, image_path)
            if staging_dir is not None:
                image_path = ensure_assets_dir(cast(Path, output_dir))
                markdown, produced = self._adopt_staged_images(
                    markdown,
                    reference_images,
                    staging_dir,
                    image_path,
                    self.asset_prefix,
                )
            else:
                produced = sorted(p.name for p in image_path.iterdir() if p.is_file())

            # Collect extracted images (only for current file). Resolve the
            # refs the converter wrote into the markdown (plus demoted
            # reference images) — pymupdf4llm sanitizes the source filename
            # when naming assets (spaces → "_", parens → "-"), so a prefix
            # glob on the input name misses them. Only files this conversion
            # wrote are candidates: a prefix glob in the shared assets dir
            # also matched a renamed sibling output's images ("a.pdf" ->
            # "a.pdf.v2-0001-01.jpg"), and recompressed them in place.
            if write_images and produced:
                ref_names = extract_asset_image_names(markdown)
                for ref in reference_images:
                    ref_name = ref.get("name")
                    if (
                        isinstance(ref_name, str)
                        and ref_name
                        and ref_name not in ref_names
                    ):
                        ref_names.append(ref_name)
                own = set(produced)
                img_files = [
                    image_path / name
                    for name in ref_names
                    if name in own and (image_path / name).is_file()
                ]
                if not img_files:
                    # No refs in the markdown: every image this run wrote
                    img_files = [
                        image_path / name
                        for name in produced
                        if (image_path / name).is_file()
                    ]
                image_processor = ImageProcessor(
                    self.config.image if self.config else None
                )
                for idx, img_file in enumerate(img_files):
                    suffix = img_file.suffix.lower().lstrip(".")
                    width = 0
                    height = 0

                    # Optionally compress and overwrite to keep sizes consistent
                    if self.config and self.config.image.compress:
                        format_map = {
                            "jpg": "JPEG",
                            "jpeg": "JPEG",
                            "png": "PNG",
                            "webp": "WEBP",
                        }
                        output_format = format_map.get(suffix, "PNG")
                        try:
                            from PIL import Image

                            with Image.open(img_file) as img:
                                compressed_img, compressed_data = (
                                    image_processor.compress(
                                        img.copy(),
                                        quality=self.config.image.quality,
                                        max_size=(
                                            self.config.image.max_width,
                                            self.config.image.max_height,
                                        ),
                                        output_format=output_format,
                                    )
                                )
                                img_file.write_bytes(compressed_data)
                                width, height = compressed_img.size
                        except Exception as e:
                            logger.debug(
                                "[PDF] Image compression failed for {}: {}",
                                img_file.name,
                                e,
                            )

                    if width == 0 or height == 0:
                        try:
                            from PIL import Image

                            with Image.open(img_file) as img:
                                width, height = img.size
                        except Exception as e:
                            logger.debug(
                                "[PDF] Image dimension extraction failed: {}", e
                            )
                            width, height = 0, 0

                    # Determine MIME type
                    mime_type = get_mime_type(suffix, default="image/png")

                    images.append(
                        ExtractedImage(
                            path=img_file,
                            index=idx + 1,
                            original_name=img_file.name,
                            mime_type=mime_type,
                            width=width,
                            height=height,
                        )
                    )

            metadata: dict[str, Any] = {
                "source": str(input_path),
                "format": "PDF",
                "images": len(images),
                "page_sources": _text_layer_sources(
                    page_numbers, advisories.verdicts if advisories else {}
                ),
            }
            _log_page_sources(input_path, metadata["page_sources"])
            if reference_images and output_dir:
                metadata["reference_images"] = reference_images

            # Render page screenshots if enabled (independent of OCR)
            enable_screenshot = self.config and self.config.screenshot.enabled
            if enable_screenshot and output_dir:
                page_images: list[dict] = []
                screenshots_dir = ensure_screenshots_dir(output_dir)

                screenshot_format = image_format if image_format != "png" else "jpg"
                page_results = self._render_pages_parallel(
                    input_path,
                    screenshots_dir,
                    screenshot_format,
                    dpi=DEFAULT_RENDER_DPI,
                )
                for _extracted_img, page_info in page_results:
                    page_images.append(page_info)

                if page_images:
                    logger.debug(f"Rendered {len(page_images)} page screenshots")

                metadata["page_images"] = page_images
                metadata["pages"] = len(page_images)
                metadata["extracted_text"] = markdown
                if scan_notice is not None:
                    if page_images:
                        metadata["scan_notice"] = list(scan_notice)
                        scan_notice = None
                if not (self.config and self.config.llm.enabled):
                    # Without LLM nothing else points at the renders (the
                    # vision path adds its own references to .llm.md)
                    markdown = append_screenshot_comments(
                        markdown, page_images, PAGE_MARKER_RE, "Page"
                    )

            if scan_notice is not None:
                # Nothing was rendered for the model to read: say it now
                user_notice(*scan_notice)

            # Clean up temporary directory if used
            if temp_dir and temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
                # Clear images whose paths pointed into the now-deleted temp dir,
                # so callers never receive dangling file references.
                images = [img for img in images if img.path and img.path.exists()]
                metadata.pop("reference_images", None)

            return ConvertResult(
                markdown=markdown,
                images=images,
                metadata=metadata,
            )
        finally:
            if temp_dir and temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
            if staging_dir is not None:
                shutil.rmtree(staging_dir, ignore_errors=True)

    _STAGED_IMAGE_TAIL_RE = re.compile(r"-\d+-\d+\.[A-Za-z0-9]+$")

    @staticmethod
    def _new_image_staging_dir() -> Path:
        """A private directory in the system temp dir for pymupdf4llm images.

        pymupdf4llm 1.28 rewrites spaces, ``()`` and ``[]`` anywhere in the
        image path (not just the file name) and then saves to the rewritten
        path, so under an output dir such as iCloud Drive's
        ``~/Library/Mobile Documents`` every image write failed. The system
        temp dir has none of those characters; the images are moved into
        the assets dir afterwards (across devices if need be).
        """
        return Path(tempfile.mkdtemp(prefix="markitai-pdf-images-"))

    @classmethod
    def _adopt_staged_images(
        cls,
        markdown: str,
        reference_images: list[dict[str, Any]],
        staging_dir: Path,
        assets_dir: Path,
        prefix: str | None,
    ) -> tuple[str, list[str]]:
        """Move staged pymupdf4llm images into assets under *prefix*.

        pymupdf4llm names images ``<sanitized input>-<page>-<index>.<ext>``;
        the ``-<page>-<index>.<ext>`` tail is kept and the prefix replaced,
        so every image of this output carries the output's own name and no
        two outputs can land on one file. References in *markdown* and
        *reference_images* are rewritten to the new names (percent-encoded,
        like every other asset reference). Without a prefix the names are
        kept.

        Returns:
            The markdown with rewritten asset references, and the names of
            the images now in *assets_dir*.
        """
        renames = cls._move_staged_images(staging_dir, assets_dir, prefix)
        cls._rename_reference_images(reference_images, renames)
        return cls._rewrite_asset_refs(markdown, renames), list(renames.values())

    @classmethod
    def _move_staged_images(
        cls, staging_dir: Path, assets_dir: Path, prefix: str | None
    ) -> dict[str, str]:
        """Move staged images into *assets_dir* under *prefix*; old -> new names.

        The assets dir is (re)created right here, not trusted to still exist
        from before the extraction: an output-profile migration running
        meanwhile removes it once it is empty.
        """
        renames: dict[str, str] = {}
        staged = sorted(p for p in staging_dir.iterdir() if p.is_file())
        for index, staged_file in enumerate(staged, start=1):
            if prefix:
                match = cls._STAGED_IMAGE_TAIL_RE.search(staged_file.name)
                tail = match.group(0) if match else f"-{index:04d}{staged_file.suffix}"
                new_name = f"{prefix}{tail}"
            else:
                new_name = staged_file.name
            target = assets_dir / new_name
            try:
                assets_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(staged_file, target)
            except FileNotFoundError:
                # Pruned between the mkdir and the move: once more
                assets_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(staged_file, target)
            renames[staged_file.name] = new_name
        return renames

    @staticmethod
    def _rename_reference_images(
        reference_images: list[dict[str, Any]], renames: dict[str, str]
    ) -> None:
        """Point demoted reference images at their adopted names, in place."""
        for ref in reference_images:
            new_name = renames.get(str(ref.get("name")))
            if new_name is not None:
                ref["name"] = new_name
                ref["rel_path"] = f"{ASSETS_REL_PATH}/{new_name}"

    @staticmethod
    def _rewrite_asset_refs(markdown: str, renames: dict[str, str]) -> str:
        """Rewrite ``.markitai/assets/<old>`` references to their adopted names."""
        from urllib.parse import quote, unquote

        if not renames:
            return markdown

        ref_re = re.compile(rf"\]\({re.escape(ASSETS_REL_PATH)}/([^)]+)\)")

        def replace_ref(match: re.Match[str]) -> str:
            old_name = unquote(match.group(1))
            new_name = renames.get(old_name)
            if new_name is None or new_name == old_name:
                return match.group(0)
            return f"]({ASSETS_REL_PATH}/{quote(new_name, safe='/._~-')})"

        return ref_re.sub(replace_ref, markdown)

    def _sanitize_hidden_text(
        self,
        input_path: Path,
        page_numbers: list[int],
        page_texts: list[str],
        *,
        prescanned: _Pending[dict[int, list[str]] | None] | None = None,
    ) -> list[str]:
        """Warn about (or remove) hidden text in the composed page texts.

        Behavior is controlled by ``security.pdf_sanitize``:

        - ``"off"``: no detection, texts returned unchanged.
        - ``"warn"`` (default): one consolidated warning naming the pages
          and a short excerpt of the hidden text; texts unchanged.
        - ``"remove"``: additionally strips hidden span texts from the
          matching page's text. Limitation: pymupdf4llm may reflow or
          style the text (bold markers, hyphenation, line wrapping), so
          only *verbatim* occurrences of a hidden span are removed; text
          outside the CropBox never appears in the output to begin with.
          A hidden span whose text is also visible on its page is kept.

        Detection failures are debug-logged and never break a conversion.

        Args:
            input_path: Path to the PDF file being converted
            page_numbers: 1-based page number for each entry in page_texts
            page_texts: Per-page markdown text, parallel to page_numbers
            prescanned: The detection already started in the background
                (``_start_prescan``); run here when None.

        Returns:
            Page texts, possibly with hidden text removed
        """
        mode = self.config.security.pdf_sanitize if self.config else "warn"
        if mode == "off":
            return page_texts

        hidden = (
            prescanned.result()
            if prescanned is not None
            else _scan_hidden_text(input_path)
        )
        if not hidden:
            return page_texts

        total_spans = sum(len(texts) for texts in hidden.values())
        excerpt = "; ".join(t for texts in hidden.values() for t in texts)
        if len(excerpt) > _HIDDEN_EXCERPT_MAX_CHARS:
            excerpt = excerpt[:_HIDDEN_EXCERPT_MAX_CHARS] + "..."
        # A user notice, not a plain warning: a prompt-injection hint must
        # reach the default console and the batch summary.
        user_notice(
            "[PDF] {}: {} hidden text span(s) detected on page(s) {} "
            "(possible prompt injection; excerpt: {!r}){}",
            input_path.name,
            total_spans,
            page_ranges(hidden),
            excerpt,
            "; removing verbatim matches from output"
            if mode == "remove"
            else "; set security.pdf_sanitize to 'remove' to strip it",
        )

        if mode != "remove":
            return page_texts

        index_by_page = {page_num: i for i, page_num in enumerate(page_numbers)}
        # A page's output mixes its hidden and visible text, so a hidden span
        # that is also part of the visible text ("e", a repeated word) cannot
        # be told apart in it: it is kept rather than deleted page-wide.
        visible = _scan_visible_text(
            input_path,
            [page_num - 1 for page_num in hidden if page_num in index_by_page],
        )
        result = list(page_texts)
        for page_num, texts in hidden.items():
            idx = index_by_page.get(page_num)
            if idx is None:
                continue
            page_visible = (visible or {}).get(page_num, "")
            for text in texts:
                if text in page_visible:
                    logger.debug(
                        "[PDF] {}: hidden text {!r} on page {} is also visible "
                        "there; kept",
                        input_path.name,
                        text,
                        page_num,
                    )
                    continue
                result[idx] = result[idx].replace(text, "")
            # A picture whose text was all hidden keeps no empty text block.
            result[idx] = _EMPTY_PICTURE_TEXT.sub("", result[idx])
        return result

    def _warn_scanned_or_garbled(
        self,
        input_path: Path,
        *,
        total_pages: int,
        prescanned: _Pending[PageAdvisories | None] | None = None,
        defer: bool = False,
    ) -> tuple[Any, ...] | None:
        """Emit one consolidated warning when pages look scanned or garbled:
        ``[PDF] plan.pdf: 4 of 120 pages look scanned/garbled (pages 3–5,
        9); consider re-running with --ocr`` (the serve workbench reads it,
        contract v1.16, and says "This PDF is scanned" at nine pages in ten).

        Advisory only: conversion behavior is unchanged and OCR is never
        auto-enabled. Failures are swallowed (debug-logged) so the check
        can never break a conversion.

        Args:
            input_path: Path to the PDF file being converted
            total_pages: How many pages the document has
            prescanned: The check already started in the background
                (``_start_prescan``); run here when None.
            defer: Return the notice's ``user_notice`` arguments instead of
                logging it (None when there is nothing to say).
        """
        advisories = (
            prescanned.result()
            if prescanned is not None
            else _scan_advisories(input_path)
        )
        if advisories is None:
            return None
        scanned_pages, garbled_pages = advisories[0], advisories[1]
        verdicts: dict[int, LayerVerdict] = advisories[2] if len(advisories) > 2 else {}
        # A failed layer in a writing system the configured recognizer does
        # not read (reversed Hebrew): --ocr would not help, so say what would
        readable = ocr_readable_scripts(self._ocr_lang)
        unreadable = [
            p
            for p in garbled_pages
            if p in verdicts and not _ocr_reads(verdicts[p], readable)
        ]
        if unreadable:
            self._notice_unread_layer(
                input_path, total_pages, unreadable, verdicts, ocr_on=False
            )

        flagged = sorted(set(scanned_pages) | (set(garbled_pages) - set(unreadable)))
        if not flagged:
            return None

        # OCR is an optional extra. Recommending --ocr to someone who does not
        # have the backend costs them a second round trip (run --ocr, hit the
        # ImportError, install, run again), so the install command goes in the
        # very first message instead.
        remedy = (
            "consider re-running with --ocr"
            if is_ocr_available()
            else f"consider re-running with --ocr, which needs the optional "
            f"OCR backend: {OCR_INSTALL_HINT}"
        )
        # A user notice: the default single-file console only shows errors,
        # and this is the one line that says how to get the missing text.
        _count, listed = _pages_phrases(flagged)
        notice = (
            "[PDF] {}: {} of {} pages look scanned/garbled ({}); {}",
            input_path.name,
            len(flagged),
            max(total_pages, flagged[-1]),
            listed,
            remedy,
        )
        if defer:
            return notice
        user_notice(*notice)
        return None

    @property
    def _ocr_lang(self) -> str:
        return self.config.ocr.lang if self.config else DEFAULT_OCR_LANG

    def _notice_unread_layer(
        self,
        input_path: Path,
        total_pages: int,
        pages: list[int],
        verdicts: dict[int, LayerVerdict],
        *,
        ocr_on: bool,
    ) -> None:
        """Say that pages keep a text layer that failed the checks because
        the configured OCR language does not read their writing system:
        ``[PDF] a.pdf: the text layer of 2 of 9 pages reads wrong (pages
        3–4): right-to-left text in reverse order; no OCR language reads
        Hebrew, so it is kept as extracted``."""
        verdict = verdicts[pages[0]]
        script = verdict.script or "this script"
        lang = ocr_lang_for_script(script)
        if ocr_on:
            advice = (
                f"OCR with ocr.lang {self._ocr_lang} does not read {script}, "
                "so the text layer is kept"
            )
            if lang:
                advice += f"; set ocr.lang to {lang} to OCR it"
        elif lang:
            advice = f"consider re-running with --ocr and ocr.lang set to {lang}"
        else:
            advice = f"no OCR language reads {script}, so it is kept as extracted"
        _count, listed = _pages_phrases(pages)
        user_notice(
            "[PDF] {}: the text layer of {} of {} pages reads wrong ({}): {}; {}",
            input_path.name,
            len(pages),
            max(total_pages, pages[-1]),
            listed,
            verdict.reason.removeprefix(FAILED),
            advice,
        )

    def _render_pages_parallel(
        self,
        input_path: Path,
        screenshots_dir: Path,
        image_format: str,
        dpi: int = DEFAULT_RENDER_DPI,
        max_workers: int | None = None,
    ) -> list[tuple[ExtractedImage, dict]]:
        """Render every PDF page and save it as a screenshot.

        PyMuPDF holds the GIL while it renders, so threads in this process
        render one page at a time. When the extraction workers are running
        (they just extracted this document, or serve a batch) each renders
        a page at a time; else threads here do, each opening its own
        document (PyMuPDF is not thread-safe). The files are the same.

        Args:
            input_path: Path to the PDF file
            screenshots_dir: Directory to save screenshots
            image_format: Image format (jpg, png, etc.)
            dpi: Render DPI
            max_workers: Override the thread count. If None, auto-detect
                based on file size.

        Returns:
            List of (ExtractedImage, page_info_dict) tuples, sorted by page number.
        """
        import pymupdf

        screenshots_dir.mkdir(parents=True, exist_ok=True)

        # Get total pages (lightweight - only reads PDF metadata)
        doc = pymupdf.open(input_path)
        total_pages = len(doc)
        doc.close()

        if total_pages == 0:
            return []

        # Named after the resolved output (see BaseConverter.asset_prefix)
        # so a renamed re-run never overwrites an older output's pages.
        shot = PageShot(
            directory=str(screenshots_dir),
            prefix=self.asset_prefix or input_path.name,
            image_format=image_format,
            image=self.config.image if self.config else None,
        )
        saved: dict[int, tuple[tuple[int, int], str]] = {}

        def finished(
            page_num: int, future: Future[tuple[tuple[int, int], str]]
        ) -> None:
            try:
                saved[page_num] = future.result()
            except Exception as e:
                logger.error(f"Failed to render page {page_num + 1}: {e}")
                raise

        pages = list(range(total_pages))
        left = run_page_tasks(
            input_path,
            shoot_page_task,
            pages,
            (dpi, shot),
            finished,
            only_running=True,
        )
        if left is None:
            left = pages
        if left:
            if max_workers is None:
                max_workers = self._get_worker_count(input_path, len(left))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(shoot_page, input_path, i, dpi, shot): i
                    for i in left
                }
                for future in as_completed(futures):
                    finished(futures[future], future)

        results: list[tuple[ExtractedImage, dict]] = []
        for page_num in sorted(saved):
            (width, height), saved_path = saved[page_num]
            # The saved path, which may differ from the requested one when
            # the fallback changes the extension
            actual_path = Path(saved_path)
            actual_name = actual_path.name
            extracted_img = ExtractedImage(
                path=actual_path,
                index=page_num + 1,
                original_name=actual_name,
                mime_type=get_mime_type(
                    actual_path.suffix, default=f"image/{image_format}"
                ),
                width=width,
                height=height,
            )
            page_info = {
                "page": page_num + 1,
                "path": str(actual_path),
                "name": actual_name,
            }
            results.append((extracted_img, page_info))
        return results

    def _fix_image_paths(self, markdown: str, image_path: Path) -> str:
        """Fix image paths to be relative to output directory.

        pymupdf4llm generates paths like: ![](full/path/to/assets/image.jpg)
        We need: ![](assets/image.jpg)

        Uses simple string replacement instead of regex — pymupdf4llm always
        outputs deterministic `](path/filename)` format, and str.replace is
        immune to special characters in filenames (parentheses, $, etc.).

        pymupdf4llm does not echo ``image_path`` verbatim: it may write the
        symlink-resolved form (e.g. macOS ``/tmp`` -> ``/private/tmp``) or a
        path relative to the process CWD. All three spellings are replaced,
        otherwise the refs stay absolute/CWD-relative and every downstream
        consumer (alt text, ref validation, asset discovery) misses them.

        Note: pymupdf4llm always uses forward slashes in markdown, even on Windows.
        We must use as_posix() to ensure consistent path matching.
        """
        import os

        candidates = [image_path.as_posix()]
        resolved = image_path.resolve().as_posix()
        if resolved not in candidates:
            candidates.append(resolved)
        try:
            cwd_relative = Path(os.path.relpath(resolved, Path.cwd())).as_posix()
        except (ValueError, OSError):
            # Different Windows drive, or the cwd no longer exists: there is
            # no relative spelling to rewrite, and that is not an error.
            cwd_relative = None
        if cwd_relative is not None and cwd_relative not in candidates:
            candidates.append(cwd_relative)

        for candidate in candidates:
            markdown = markdown.replace(f"]({candidate}/", f"]({ASSETS_REL_PATH}/")
        return markdown

    def _collect_embedded_images(
        self, assets_dir: Path, input_name: str, markdown: str = ""
    ) -> list[ExtractedImage]:
        """Collect embedded images extracted by pymupdf4llm.

        pymupdf4llm extracts embedded images with names like: filename.pdf-0-0.png
        (page index - image index on that page). The name prefix is a
        sanitized form of the source filename (spaces → "_"), so files are
        resolved from the markdown refs first; matching on the asset name
        prefix is kept as a fallback for markdown without asset refs.

        Args:
            assets_dir: Directory where images were extracted
            input_name: The images' name prefix (the output-derived
                ``asset_prefix``, else the PDF filename)
            markdown: Converted markdown whose asset refs name the files

        Returns:
            List of ExtractedImage for embedded images
        """
        embedded_images: list[ExtractedImage] = []
        # Suffix pattern: ...-{page}-{index}.{ext}
        index_pattern = re.compile(r"-(\d+)-(\d+)\.(png|jpg|jpeg|webp)$", re.IGNORECASE)

        # Nothing was extracted, or an output-profile migration removed the
        # emptied dir meanwhile: no images, not a failed conversion.
        if not assets_dir.is_dir():
            return embedded_images

        image_files = [
            assets_dir / name
            for name in extract_asset_image_names(markdown)
            if (assets_dir / name).is_file()
        ]
        if not image_files:
            legacy_pattern = re.compile(
                rf"^{re.escape(input_name)}-(\d+)-(\d+)\.(png|jpg|jpeg|webp)$"
            )
            try:
                image_files = [
                    f for f in assets_dir.iterdir() if legacy_pattern.match(f.name)
                ]
            except FileNotFoundError:
                return embedded_images

        for image_file in image_files:
            match = index_pattern.search(image_file.name)
            if match:
                page_idx = int(match.group(1))
                img_idx = int(match.group(2))
                ext = match.group(3).lower()

                # Get image dimensions
                try:
                    import pymupdf

                    pix = pymupdf.Pixmap(str(image_file))
                    width, height = pix.width, pix.height
                except Exception:
                    width, height = 0, 0

                embedded_images.append(
                    ExtractedImage(
                        path=image_file,
                        index=page_idx * 100 + img_idx,  # Unique index
                        original_name=image_file.name,
                        mime_type=f"image/{'jpeg' if ext in ('jpg', 'jpeg') else ext}",
                        width=width,
                        height=height,
                    )
                )

        return embedded_images

    def _convert_with_ocr(
        self, input_path: Path, output_dir: Path | None = None
    ) -> ConvertResult:
        """Convert PDF using OCR for scanned documents.

        With per-page routing (the default), pages that carry a healthy
        text layer go through the same pymupdf4llm extraction as the
        standard path -- headings, lists, tables and image references
        intact -- and only scanned/garbled pages are OCR'd. Pictures on
        the native pages are OCR'd too, their text placed under the image
        reference. Running headers/footers are stripped and hidden text
        is sanitized exactly as on the standard path.

        Also renders each page as an image (if enable_screenshot) for reference.

        Args:
            input_path: Path to the PDF file
            output_dir: Optional output directory for extracted images

        Returns:
            ConvertResult containing OCR-extracted markdown with commented page images

        Raises:
            OCRError: OCR failed on one or more pages. The failure is never
                written into the Markdown as if it were page content.
        """
        try:
            import pymupdf
        except ImportError as e:
            raise MissingDependencyError(
                "PyMuPDF is not installed. Install with: uv add pymupdf"
            ) from e

        from markitai.converter.onnx_threads import local_threads
        from markitai.ocr import OCRProcessor

        ocr_config = self.config.ocr if self.config else None
        with pymupdf.open(input_path) as counted:
            page_count = len(counted)
        # Pages read on threads here share the cores (the workers set their own)
        ocr = OCRProcessor(
            ocr_config, threads=local_threads(self._ocr_thread_count(page_count))
        )

        logger.info(f"Converting PDF with OCR: {input_path.name}")

        # Setup screenshots directory for page images
        if output_dir:
            screenshots_dir = ensure_screenshots_dir(output_dir)
        else:
            screenshots_dir = create_tracked_temp_dir()

        # Get image format from config
        image_format = "jpg"
        if self.config:
            image_format = normalize_image_extension(self.config.image.format)

        # Check if screenshot is enabled
        enable_screenshot = self.config and self.config.screenshot.enabled

        page_images: list[dict] = []
        dpi = DEFAULT_RENDER_DPI

        # Per-page OCR routing: pages with a healthy native text layer keep
        # that text; only scanned/garbled pages go through OCR.
        per_page_routing = ocr_config.per_page_routing if ocr_config else True
        native_pages: list[int] = []
        verdicts: dict[int, LayerVerdict] = {}
        readable = ocr_readable_scripts(self._ocr_lang)
        # The text layer is also checked against what its glyphs draw: a few
        # crops per font read by the recognizer (the engine loads on the
        # first crop, so a document without a text layer never loads it here)
        glyph_check = GlyphCheck(ocr.read_lines, readable)
        doc = pymupdf.open(input_path)
        try:
            total_pages = len(doc)
            if per_page_routing:
                try:
                    native_pages, verdicts = _collect_native_text_pages(
                        doc, readable=readable, glyph_check=glyph_check
                    )
                except Exception as e:
                    logger.debug("[PDF] OCR routing check failed: {}", e)
                    native_pages, verdicts = [], {}
        finally:
            doc.close()
        if per_page_routing:
            logger.debug(
                "OCR routing: {} pages native, {} pages OCR ({} line crops read "
                "for the glyph check)",
                len(native_pages),
                total_pages - len(native_pages),
                glyph_check.crops,
            )
            kept = [
                i + 1
                for i in native_pages
                if verdicts.get(i + 1, LayerVerdict("trusted", "")).status == "failed"
            ]
            if kept:
                self._notice_unread_layer(
                    input_path, total_pages, kept, verdicts, ocr_on=True
                )

        # Native pages: the standard path's pymupdf4llm extraction, so they
        # keep their structure and image refs instead of flat get_text().
        temp_assets: Path | None = None
        assets_dir: Path | None = None
        page_texts: dict[int, str] = {}
        reference_images: list[dict[str, Any]] = []
        if native_pages:
            if output_dir:
                # Not created here: extraction stages its images elsewhere
                # and creates the dir when it moves them in (a profile
                # migration running meanwhile prunes an empty one).
                assets_dir = output_dir / ASSETS_REL_PATH
            else:
                temp_assets = Path(tempfile.mkdtemp())
                assets_dir = temp_assets
            try:
                page_texts, reference_images = self._extract_native_pages(
                    input_path, native_pages, assets_dir, image_format
                )
            except Exception as e:
                # Not worth failing over: OCR can still read these pages.
                logger.debug("[PDF] Native extraction failed, OCR'ing all: {}", e)
                page_texts, reference_images = {}, []
            native_pages = sorted(page_texts)
        ocr_pages = [i for i in range(total_pages) if i not in page_texts]

        try:
            # Also the threads for screenshots and pictures on native pages
            max_workers = self._ocr_thread_count(total_pages)
            ocr_texts: dict[int, str] = {}
            failures: dict[int, str] = {}
            shot: PageShot | None = None
            if enable_screenshot:
                screenshots_dir.mkdir(parents=True, exist_ok=True)
                shot = PageShot(
                    directory=str(screenshots_dir),
                    prefix=self.asset_prefix or input_path.name,
                    image_format=image_format,
                    image=self.config.image if self.config else None,
                )
            # Every page when each gets a screenshot, else the ones to read
            targets = list(range(total_pages)) if shot else ocr_pages
            ocr_page_set = set(ocr_pages)
            results: dict[int, dict[str, Any]] = {}

            def finished(page_num: int, future: Future[dict[str, Any]]) -> None:
                try:
                    results[page_num] = future.result()
                    logger.debug(f"OCR processed page {page_num + 1}/{total_pages}")
                except (OCRBackendMissing, OCRLanguageError):
                    raise
                except Exception as e:
                    failures[page_num] = str(e)

            # In the workers when this process has them (one page per core,
            # the engine loaded once per worker), else on threads here
            left = run_page_tasks(
                input_path,
                read_page_task,
                targets,
                (dpi, ocr_config, frozenset(ocr_page_set), shot),
                finished,
            )
            if left is None:
                left = targets
            if left:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(
                            read_page,
                            ocr,
                            input_path,
                            i,
                            dpi,
                            read=i in ocr_page_set,
                            shot=shot,
                        ): i
                        for i in left
                    }
                    for future in as_completed(futures):
                        finished(futures[future], future)

            # In page order. Page renders are screenshots, counted from
            # page_images like on the standard path, never as embedded images.
            for i in sorted(results):
                if results[i]["page_image"] is not None:
                    page_images.append(results[i]["page_image"])
                if results[i]["text"] is not None and i in ocr_page_set:
                    ocr_texts[i] = results[i]["text"]

            if failures:
                pages = sorted(failures)
                raise OCRError(
                    f"OCR failed on {len(pages)} of {total_pages} page(s) of "
                    f"{input_path.name} (pages "
                    f"{page_ranges(p + 1 for p in pages)}): {failures[pages[0]]}"
                )

            empty_pages = [i + 1 for i in ocr_pages if not ocr_texts.get(i)]
            if empty_pages:
                # One notice with the pages as ranges, worded as Rust's
                # (contract v1.17)
                one = len(empty_pages) == 1
                user_notice(
                    "PDF {} {} ({} of {}): OCR found no text; {} blank in the output.",
                    "page" if one else "pages",
                    page_ranges(empty_pages),
                    len(empty_pages),
                    total_pages,
                    "it is" if one else "they are",
                )
            page_texts.update(ocr_texts)

            # Pictures on native pages (a chart or a pasted scan next to real
            # text) are exactly what --ocr is for; read them too.
            pictures_read, picture_pages = 0, set[int]()
            if assets_dir is not None:
                pictures_read, picture_pages = self._ocr_native_page_pictures(
                    ocr, page_texts, native_pages, assets_dir, max_workers, input_path
                )

            ordered = list(range(total_pages))
            texts = [page_texts.get(i, "") for i in ordered]

            # Strip running headers/footers repeated across page boundaries
            texts, stripped_lines = strip_repeated_page_lines(texts)
            if stripped_lines:
                logger.debug(
                    "[PDF] Stripped {} repeated header/footer line(s): {}",
                    len(stripped_lines),
                    sorted(stripped_lines),
                )

            # Hidden-text / prompt-injection sanitization (warn or remove),
            # as on the standard path. Only native text can carry a hidden
            # span; OCR reads what is visible. Called even with no native
            # page, so the warning still names the pages that carry one.
            native_set = set(native_pages)
            native_index = [i for i in ordered if i in native_set]
            sanitized = self._sanitize_hidden_text(
                input_path,
                [i + 1 for i in native_index],
                [texts[i] for i in native_index],
            )
            for i, text in zip(native_index, sanitized):
                texts[i] = text

            page_image_names = {info["page"]: info["name"] for info in page_images}
            markdown_parts = []
            for i, text in zip(ordered, texts):
                part = text
                name = page_image_names.get(i + 1)
                if name:
                    comment = f"<!-- ![Page {i + 1}]({SCREENSHOTS_REL_PATH}/{name}) -->"
                    part = f"{part}\n\n{comment}" if part else comment
                markdown_parts.append(f"{page_marker(i + 1)}\n\n{part}")

            # Mark the page boundaries the loop above already knows. Without
            # them an OCR'd PDF reaches every later stage as one undivided run
            # of text: page/image alignment cannot line up, and output profiles
            # have nothing to rewrite.
            extracted_text = "\n\n".join(markdown_parts)
            embedded_images: list[ExtractedImage] = []
            if assets_dir is not None:
                extracted_text = self._fix_image_paths(extracted_text, assets_dir)
                if output_dir:
                    embedded_images = self._collect_embedded_images(
                        assets_dir,
                        self.asset_prefix or input_path.name,
                        extracted_text,
                    )

            metadata: dict[str, Any] = {
                "source": str(input_path),
                "format": "PDF",
                "ocr_used": bool(ocr_pages) or pictures_read > 0,
                "ocr_path": "rapidocr",
                "pages": total_pages,
                "images": len(embedded_images),
                "extracted_text": extracted_text,
                "page_images": page_images,
                "page_sources": _ocr_sources(
                    total_pages,
                    routing=per_page_routing,
                    native_pages=native_pages,
                    verdicts=verdicts,
                    empty_pages=[page - 1 for page in empty_pages],
                    picture_pages=picture_pages,
                    ocr_lang=self._ocr_lang,
                ),
            }
            _log_page_sources(input_path, metadata["page_sources"])
            if reference_images and output_dir:
                metadata["reference_images"] = reference_images

            return ConvertResult(
                markdown=extracted_text,
                images=embedded_images,
                metadata=metadata,
            )
        finally:
            if temp_assets and temp_assets.exists():
                shutil.rmtree(temp_assets, ignore_errors=True)

    def _extract_native_pages(
        self,
        input_path: Path,
        pages: list[int],
        assets_dir: Path,
        image_format: str,
    ) -> tuple[dict[int, str], list[dict[str, Any]]]:
        """pymupdf4llm Markdown for the given 0-based pages.

        The same extraction the standard (non-OCR) path runs, restricted to
        the pages OCR routing left native.

        Returns:
            Tuple of (0-based page -> page markdown, demoted reference images)
        """
        # Same staging as the standard path: pymupdf4llm would otherwise name
        # images after the sanitized input and overwrite another output's,
        # and cannot write under a path with spaces or brackets at all.
        staging_dir = self._new_image_staging_dir()
        try:
            texts, reference_images = self._extract_native_page_chunks(
                input_path, pages, staging_dir, image_format
            )
            renames = self._move_staged_images(
                staging_dir, assets_dir, self.asset_prefix
            )
            self._rename_reference_images(reference_images, renames)
            # Staged refs point into the staging dir; make them relative
            # first so the rename rewrite (and later stages) can see them.
            texts = {
                page: self._rewrite_asset_refs(
                    self._fix_image_paths(text, staging_dir), renames
                )
                for page, text in texts.items()
            }
            return texts, reference_images
        finally:
            shutil.rmtree(staging_dir, ignore_errors=True)

    def _extract_native_page_chunks(
        self,
        input_path: Path,
        pages: list[int],
        image_dir: Path,
        image_format: str,
    ) -> tuple[dict[int, str], list[dict[str, Any]]]:
        """Run pymupdf4llm on *pages*, writing images into *image_dir*."""
        chunks = cast(
            list[Any],
            to_markdown_chunks(
                input_path,
                extract=pymupdf4llm.to_markdown,
                pages=pages,
                write_images=True,
                image_path=str(image_dir),
                image_format=image_format,
                dpi=DEFAULT_RENDER_DPI,
            ),
        )
        if not isinstance(chunks, list):
            chunks = [chunks]
        texts: dict[int, str] = {}
        reference_images: list[dict[str, Any]] = []
        for index, chunk in enumerate(chunks):
            fallback = pages[index] if index < len(pages) else index
            page_num = _chunk_page_number(chunk, fallback)
            text = _chunk_text(chunk)
            if isinstance(chunk, dict):
                text, refs = self._demote_reference_picture_blocks(chunk, page_num)
                reference_images.extend(refs)
            texts[page_num - 1] = text
        pages_read = sorted(texts)
        repaired = repair_pages(
            input_path,
            [page + 1 for page in pages_read],
            [texts[p] for p in pages_read],
        )
        texts = {
            page: text.strip() for page, text in zip(pages_read, repaired, strict=True)
        }
        return texts, reference_images

    def _ocr_native_page_pictures(
        self,
        ocr: Any,
        page_texts: dict[int, str],
        native_pages: list[int],
        assets_dir: Path,
        max_workers: int,
        input_path: Path,
    ) -> tuple[int, set[int]]:
        """OCR the sizable pictures referenced on native pages, in place.

        Recognized text goes right under the picture's reference. Pictures
        smaller than ``_PICTURE_OCR_MIN_PIXELS`` (icons, logos, rules) are
        skipped. A picture that cannot be read keeps its bare reference
        and raises a notice; the page's own text is unaffected.

        Returns:
            Number of pictures OCR was run on, and the 0-based pages whose
            pictures it read text from
        """
        from urllib.parse import unquote

        from PIL import Image

        targets: list[tuple[int, str, Path]] = []
        for page in native_pages:
            for match in self._IMAGE_REF_RE.finditer(page_texts.get(page, "")):
                # Adopted names are percent-encoded in the refs ("报告 1.pdf"
                # -> "%E6%8A%A5%E5%91%8A%201.pdf-0001-01.png")
                path = assets_dir / unquote(match.group(1))
                if not path.is_file():
                    continue
                try:
                    with Image.open(path) as picture:
                        width, height = picture.size
                except Exception:
                    continue
                if width * height >= _PICTURE_OCR_MIN_PIXELS:
                    targets.append((page, match.group(0), path))
        if not targets:
            return 0, set()

        def read(target: tuple[int, str, Path]) -> str:
            return ocr.recognize(target[2]).text.strip()

        unreadable: list[str] = []
        recognized: dict[tuple[int, str, Path], str] = {}
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(read, t): t for t in targets}
                for future in as_completed(futures):
                    target = futures[future]
                    try:
                        recognized[target] = future.result()
                    except (OCRBackendMissing, OCRLanguageError):
                        raise
                    except Exception as e:
                        logger.debug(
                            "[PDF] Picture OCR failed for {}: {}", target[2], e
                        )
                        unreadable.append(target[2].name)
        except OCRBackendMissing as e:
            # Every page here has a real text layer, so the document converts
            # fine without OCR; only the pictures go unread. Say so instead of
            # failing a PDF that never needed the backend before.
            user_notice("[OCR] Pictures in {} were not read: {}", input_path.name, e)
            return 0, set()

        pages_read: set[int] = set()
        for (page, ref, _path), text in recognized.items():
            if text:
                page_texts[page] = page_texts[page].replace(ref, f"{ref}\n\n{text}", 1)
                pages_read.add(page)
        if unreadable:
            user_notice(
                "[OCR] Could not read {} picture(s) in {}: {}",
                len(unreadable),
                input_path.name,
                ", ".join(sorted(unreadable)),
            )
        return len(targets), pages_read

    def _degrade_vlm_ocr(
        self, input_path: Path, output_dir: Path | None
    ) -> ConvertResult:
        """Fall back to local OCR when MARKITAI_NO_VLM_OCR blocks the VLM path.

        Privacy-preserving degrade: never send page images to a remote model.
        Uses RapidOCR when installed, otherwise fails with an actionable error
        that names both ways out (unset the env var, or install RapidOCR).
        """
        if is_ocr_available():
            logger.warning(
                "[VLM OCR] Disabled by MARKITAI_NO_VLM_OCR; "
                "falling back to local RapidOCR for {}",
                input_path.name,
            )
            if self.config is not None and self.config.screenshot.enabled:
                # The switch covers OCR only. --screenshot --llm is its own
                # explicit request to show the model the rendered pages.
                user_notice(
                    "[VLM OCR] {}: MARKITAI_NO_VLM_OCR keeps OCR local, but "
                    "--screenshot --llm still sends the page screenshots to the "
                    "vision model. Drop --screenshot to keep page images local.",
                    input_path.name,
                )
            return self._convert_with_ocr(input_path, output_dir)
        raise OCRBackendMissing(
            "VLM OCR is disabled by MARKITAI_NO_VLM_OCR=1 and the local "
            "RapidOCR backend is not installed. Either unset "
            "MARKITAI_NO_VLM_OCR to use the vision LLM, or install "
            f"RapidOCR ({OCR_INSTALL_HINT})."
        )

    def _render_pages_for_llm(
        self, input_path: Path, output_dir: Path | None = None
    ) -> ConvertResult:
        """Extract text and render pages for LLM Vision analysis.

        This method:
        1. Extracts text using pymupdf4llm (fast, preserves links/tables)
        2. Renders each page as an image (if screenshot enabled)

        Returns:
            ConvertResult with extracted text and page images
        """
        logger.info(f"Extracting text and rendering pages for LLM: {input_path.name}")

        # Determine output paths
        temp_assets: Path | None = None
        temp_screenshots: Path | None = None
        if output_dir:
            assets_dir = ensure_assets_dir(output_dir)
            screenshots_dir = ensure_screenshots_dir(output_dir)
        else:
            temp_assets = Path(tempfile.mkdtemp())
            temp_screenshots = Path(tempfile.mkdtemp())
            assets_dir = temp_assets
            screenshots_dir = temp_screenshots

        # Get image format from config
        image_format = "jpg"
        if self.config:
            image_format = normalize_image_extension(self.config.image.format)

        # Step 1: Extract text using pymupdf4llm (fast, preserves structure).
        # page_chunks=True so the text carries the same page markers the
        # standard path writes: the vision prompt asks the model to keep each
        # page's content under its own marker and aligned with that page's
        # image, which it cannot do for text that arrives as one long run.
        logger.debug("Extracting text with pymupdf4llm...")
        # Staged like the other paths: pymupdf4llm cannot write under a path
        # with spaces or brackets, and names images after the input.
        staging_dir = self._new_image_staging_dir()
        try:
            page_chunks = cast(
                list[Any],
                to_markdown_chunks(
                    input_path,
                    extract=pymupdf4llm.to_markdown,
                    write_images=True,
                    image_path=str(staging_dir),
                    image_format=image_format,
                    dpi=DEFAULT_RENDER_DPI,
                ),
            )
            # page_chunks=True returns a list; anything else is one whole page.
            # Iterating a bare string here would mark up every character as its
            # own page rather than fail, so the shape is checked, not assumed.
            chunks = page_chunks if isinstance(page_chunks, list) else [page_chunks]
            page_numbers = [
                _chunk_page_number(chunk, index) for index, chunk in enumerate(chunks)
            ]
            # This text goes to the LLM: hidden-text sanitization (warn or
            # remove) as on the other paths.
            page_texts = repair_pages(
                input_path, page_numbers, [_chunk_text(chunk) for chunk in chunks]
            )
            page_texts = self._sanitize_hidden_text(
                input_path, page_numbers, page_texts
            )
            extracted_text = "\n\n".join(
                f"{page_marker(page_num)}\n\n{page_text}"
                for page_num, page_text in zip(page_numbers, page_texts)
            )
            extracted_text = self._fix_image_paths(extracted_text, staging_dir)
            extracted_text, _adopted = self._adopt_staged_images(
                extracted_text, [], staging_dir, assets_dir, self.asset_prefix
            )
        finally:
            shutil.rmtree(staging_dir, ignore_errors=True)

        # Collect embedded images extracted by pymupdf4llm
        embedded_images = self._collect_embedded_images(
            assets_dir, self.asset_prefix or input_path.name, extracted_text
        )

        images: list[ExtractedImage] = list(embedded_images)
        page_images: list[dict] = []

        # OCR+LLM path always renders page images for Vision analysis.
        # This is independent of screenshot.enabled, which controls the
        # "extra screenshot" feature in the standard (non-OCR) convert path.
        if output_dir:
            page_results = self._render_pages_parallel(
                input_path, screenshots_dir, image_format, dpi=DEFAULT_RENDER_DPI
            )
            # Page renders are screenshots, not embedded images: they are
            # counted from page_images, like the standard path does
            page_images.extend(page_info for _image, page_info in page_results)

            if page_images:
                logger.debug(f"Rendered {len(page_images)} page screenshots")

        # Clean up temporary directories if used (no output_dir)
        for temp_dir in (temp_assets, temp_screenshots):
            if temp_dir and temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
        # Remove images pointing into deleted temp dirs
        if temp_assets or temp_screenshots:
            images = [img for img in images if img.path and img.path.exists()]

        # One-time privacy disclosure: the rendered page images are about to
        # be handed to the vision LLM for OCR reading.
        ensure_vlm_ocr_disclosed(
            self.config, page_count=len(page_images) if page_images else None
        )

        # Each page goes to the model as its rendered image and its text
        # layer; without an image (no output dir) only the text is there.
        rendered = {info["page"] for info in page_images}
        page_sources = [
            _page_source(page, "mixed", VLM_READ)
            if page in rendered
            else _page_source(page, "text", "text layer (no page image to read)")
            for page in page_numbers
        ]
        _log_page_sources(input_path, page_sources)
        return ConvertResult(
            markdown=extracted_text,
            images=images,
            metadata={
                "source": str(input_path),
                "format": "PDF",
                "pages": len(page_images) if page_images else 0,
                "ocr_path": "vlm",
                "extracted_text": extracted_text,
                "page_images": page_images,
                "page_sources": page_sources,
            },
        )
