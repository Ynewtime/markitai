"""PDF document converter."""

from __future__ import annotations

import math
import os
import re
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pymupdf4llm
from loguru import logger

from markitai.constants import ASSETS_REL_PATH, DEFAULT_RENDER_DPI, SCREENSHOTS_REL_PATH
from markitai.converter.base import (
    BaseConverter,
    ConvertResult,
    ExtractedImage,
    FileFormat,
    register_converter,
)
from markitai.image import ImageProcessor
from markitai.ocr import is_likely_garbled
from markitai.security import escape_glob_pattern
from markitai.utils.mime import get_mime_type, normalize_image_extension
from markitai.utils.paths import (
    create_tracked_temp_dir,
    ensure_assets_dir,
    ensure_screenshots_dir,
)
from markitai.utils.text import extract_asset_image_names

if TYPE_CHECKING:
    from markitai.config import MarkitaiConfig

# --- Repeated header/footer suppression --------------------------------------

# Documents with fewer pages are exempt: too few samples to call a line
# "running" chrome with confidence.
_HEADER_FOOTER_MIN_DOC_PAGES = 4
# A normalized boundary line must appear on >= this fraction of pages...
_HEADER_FOOTER_MIN_FRACTION = 0.6
# ...and on >= this many pages in absolute terms.
_HEADER_FOOTER_MIN_PAGES = 3
# Number of leading/trailing non-empty lines per page examined as candidates.
_BOUNDARY_LINE_COUNT = 2

_DIGIT_RUN_RE = re.compile(r"\d+")
_WS_RUN_RE = re.compile(r"\s+")


def normalize_boundary_line(line: str) -> str:
    """Normalize a line for cross-page header/footer matching.

    Lowercases, collapses whitespace, and replaces every digit run with
    ``#`` so ``Page 1 of 6`` and ``Page 2 of 6`` collapse to the same key.

    Args:
        line: Raw line text from a page boundary

    Returns:
        Normalized key used to match repeated lines across pages
    """
    collapsed = _WS_RUN_RE.sub(" ", line).strip().lower()
    return _DIGIT_RUN_RE.sub("#", collapsed)


def _boundary_line_indices(lines: list[str]) -> list[int]:
    """Indices of the first/last ``_BOUNDARY_LINE_COUNT`` non-empty lines."""
    non_empty = [i for i, line in enumerate(lines) if line.strip()]
    head = non_empty[:_BOUNDARY_LINE_COUNT]
    tail = non_empty[-_BOUNDARY_LINE_COUNT:]
    return sorted(set(head + tail))


def _is_protected_line(line: str) -> bool:
    """Markdown headings and table rows are never stripped as chrome."""
    stripped = line.lstrip()
    return stripped.startswith(("#", "|"))


def strip_repeated_page_lines(
    page_texts: list[str],
) -> tuple[list[str], set[str]]:
    """Strip running headers/footers repeated across page boundaries.

    For each page the first/last two non-empty lines are candidates. A
    candidate whose normalized form (see :func:`normalize_boundary_line`)
    appears on >=60% of pages and on at least 3 pages is a running
    header/footer and is removed from every page's boundary. Markdown
    headings (``#``) and table rows (``|``) are never removed, and
    documents with fewer than 4 pages are exempt entirely.

    Args:
        page_texts: Per-page markdown text, in page order

    Returns:
        Tuple of (page texts with repeated lines removed, set of
        normalized lines that were stripped)
    """
    if len(page_texts) < _HEADER_FOOTER_MIN_DOC_PAGES:
        return page_texts, set()

    pages_lines = [text.splitlines() for text in page_texts]

    counts: dict[str, int] = {}
    for lines in pages_lines:
        seen: set[str] = set()
        for idx in _boundary_line_indices(lines):
            if _is_protected_line(lines[idx]):
                continue
            norm = normalize_boundary_line(lines[idx])
            if norm:
                seen.add(norm)
        for norm in seen:
            counts[norm] = counts.get(norm, 0) + 1

    threshold = max(
        _HEADER_FOOTER_MIN_PAGES,
        math.ceil(len(page_texts) * _HEADER_FOOTER_MIN_FRACTION),
    )
    repeated = {norm for norm, count in counts.items() if count >= threshold}
    if not repeated:
        return page_texts, set()

    result: list[str] = []
    stripped: set[str] = set()
    for page_index, lines in enumerate(pages_lines):
        drop: set[int] = set()
        for idx in _boundary_line_indices(lines):
            if _is_protected_line(lines[idx]):
                continue
            norm = normalize_boundary_line(lines[idx])
            if norm in repeated:
                drop.add(idx)
                stripped.add(norm)
        if drop:
            result.append(
                "\n".join(line for i, line in enumerate(lines) if i not in drop)
            )
        else:
            result.append(page_texts[page_index])
    return result, stripped


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


def collect_page_advisories(doc: Any) -> tuple[list[int], list[int]]:
    """Compute cheap per-page scan/garbled signals for an open PDF document.

    A page with near-zero extracted text but significant image coverage
    looks scanned; a page whose text fails the vowel-ratio check (see
    :func:`markitai.ocr.is_likely_garbled`) is garbled. No rendering is
    performed, so this is cheap even for large documents.

    Args:
        doc: An open pymupdf Document

    Returns:
        Tuple of (scanned-looking page numbers, garbled page numbers),
        both 1-based
    """
    scanned: list[int] = []
    garbled: list[int] = []
    for i, page in enumerate(doc):
        text = page.get_text().strip()
        if len(text) < _SCANNED_MAX_TEXT_CHARS:
            if _page_image_coverage(page) >= _SCANNED_MIN_IMAGE_COVERAGE:
                scanned.append(i + 1)
        elif is_likely_garbled(text):
            garbled.append(i + 1)
    return scanned, garbled


def _collect_native_text_pages(doc: Any) -> dict[int, str]:
    """Native text for pages that do not look scanned or garbled.

    Mirrors the signals of :func:`collect_page_advisories`: a page
    qualifies for native extraction when it carries a meaningful text
    layer (>= ``_SCANNED_MAX_TEXT_CHARS``) that passes the garble check.
    Everything else (near-textless / scanned / garbled pages) is left
    for OCR.

    Args:
        doc: An open pymupdf Document

    Returns:
        Mapping of 0-based page number -> stripped native page text
    """
    native: dict[int, str] = {}
    for i, page in enumerate(doc):
        text = page.get_text().strip()
        if len(text) >= _SCANNED_MAX_TEXT_CHARS and not is_likely_garbled(text):
            native[i] = text
    return native


# --- Hidden-text / prompt-injection sanitization -------------------------------

# Spans below this font size (pt) are effectively invisible when rendered.
_HIDDEN_MAX_FONT_SIZE = 2.0
# Fill luminance above this is a white-on-white candidate. Background
# sampling is deliberately not performed: the page background is assumed
# to be the default white.
_HIDDEN_MIN_LUMINANCE = 0.95
# Max characters of hidden text quoted in the consolidated warning.
_HIDDEN_EXCERPT_MAX_CHARS = 80


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


def _is_hidden_span(
    span: dict[str, Any], page_rect: tuple[float, float, float, float]
) -> bool:
    """Classify a pymupdf text span as hidden (invisible to a human reader).

    A span is hidden when it is drawn at zero opacity (pymupdf maps text
    render mode 3 to ``alpha == 0``), its font size is below 2pt, its
    fill color is near-white on the assumed-white page background, or
    its bbox lies fully outside the page CropBox.
    """
    text = span.get("text", "")
    if not isinstance(text, str) or not text.strip():
        return False
    if span.get("alpha", 255) == 0:
        return True
    size = span.get("size")
    if isinstance(size, (int, float)) and size < _HIDDEN_MAX_FONT_SIZE:
        return True
    color = span.get("color")
    if isinstance(color, int) and _color_luminance(color) > _HIDDEN_MIN_LUMINANCE:
        return True
    bbox = span.get("bbox")
    if (
        isinstance(bbox, (tuple, list))
        and len(bbox) == 4
        and _bbox_fully_outside(tuple(bbox), page_rect)
    ):
        return True
    return False


def collect_hidden_text(doc: Any) -> dict[int, list[str]]:
    """Detect hidden text spans (prompt-injection vector) in an open PDF.

    Scans ``page.get_text("dict")`` spans using a textpage clipped to the
    MediaBox so text placed outside the CropBox is also examined. See
    :func:`_is_hidden_span` for the detection criteria.

    Args:
        doc: An open pymupdf Document

    Returns:
        Mapping of 1-based page number -> list of hidden span texts
    """
    hidden: dict[int, list[str]] = {}
    for i, page in enumerate(doc):
        try:
            page_rect = tuple(page.rect)
            textpage = page.get_textpage(clip=page.mediabox)
            data = page.get_text("dict", textpage=textpage)
        except Exception as e:
            logger.debug("[PDF] Hidden-text scan failed for page {}: {}", i + 1, e)
            continue
        texts: list[str] = []
        for block in data.get("blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    if _is_hidden_span(span, page_rect):
                        texts.append(span["text"].strip())
        if texts:
            hidden[i + 1] = texts
    return hidden


@register_converter(FileFormat.PDF)
class PdfConverter(BaseConverter):
    """Converter for PDF documents using pymupdf4llm.

    Supports OCR mode for scanned PDFs when --ocr flag is enabled.
    """

    supported_formats = [FileFormat.PDF]

    def __init__(self, config: MarkitaiConfig | None = None) -> None:
        super().__init__(config)

    # Absolute cap on internal thread pool workers. PyMuPDF is not thread-safe,
    # so each worker opens its own document copy. These internal pools may run
    # *inside* the shared converter executor from workflow/core.py, so keeping
    # them bounded avoids excessive thread nesting and memory pressure.
    _MAX_INTERNAL_WORKERS = 6

    _IMAGE_REF_RE = re.compile(r"!\[[^\]]*\]\((?:[^)]+/)?([^)]+)\)")
    _PICTURE_TEXT_RE = re.compile(
        r"\*\*----- Start of picture text -----\*\*<br>\s*(.*?)\s*"
        r"\*\*----- End of picture text -----\*\*<br>",
        re.DOTALL,
    )
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
            match = self._PICTURE_TEXT_RE.search(segment)
            image_match = self._IMAGE_REF_RE.search(segment)
            if (
                match is not None
                and image_match is not None
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
                rebuilt_parts.append(self._IMAGE_REF_RE.sub("", segment, count=1))
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

        Args:
            input_path: Path to the input file
            output_dir: Optional output directory for extracted images

        Returns:
            ConvertResult containing markdown and extracted images
        """
        input_path = Path(input_path)
        images: list[ExtractedImage] = []

        # Check if OCR mode is enabled
        use_ocr = self.config and self.config.ocr.enabled
        use_llm = self.config and self.config.llm.enabled

        if use_ocr:
            if use_llm:
                # --ocr --llm: Render pages as images for LLM Vision analysis
                return self._render_pages_for_llm(input_path, output_dir)
            # --ocr only: Use RapidOCR for text extraction
            return self._convert_with_ocr(input_path, output_dir)

        # Determine image output path
        temp_dir: Path | None = None
        try:
            if output_dir:
                image_path = ensure_assets_dir(output_dir)
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

            # Convert using pymupdf4llm with page_chunks=True for page-level splitting
            # This allows proper text-to-screenshot alignment in batched LLM processing
            page_results = pymupdf4llm.to_markdown(
                str(input_path),
                write_images=write_images,
                image_path=str(image_path),
                image_format=image_format,
                dpi=dpi,
                force_text=True,
                page_chunks=True,  # Return list of page chunks instead of single string
                use_ocr=False,  # Markitai handles OCR separately; suppress Tesseract probing
            )

            # Merge page chunks and add page markers for proper splitting
            # Format: <!-- Page number: N --> (consistent with Slide number format)
            # Ensure blank line after marker for proper markdown formatting
            page_numbers: list[int] = []
            page_texts: list[str] = []
            reference_images: list[dict[str, Any]] = []
            for i, chunk in enumerate(page_results):
                page_num = i + 1
                page_text = (
                    chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
                )
                if isinstance(chunk, dict):
                    chunk_page_num = chunk.get("metadata", {}).get("page_number")
                    if isinstance(chunk_page_num, int) and chunk_page_num > 0:
                        page_num = chunk_page_num
                    page_text, page_references = self._demote_reference_picture_blocks(
                        chunk, page_num
                    )
                    reference_images.extend(page_references)
                page_numbers.append(page_num)
                page_texts.append(page_text)

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
                input_path, page_numbers, page_texts
            )

            markdown_parts = [
                f"<!-- Page number: {page_num} -->\n\n{page_text}"
                for page_num, page_text in zip(page_numbers, page_texts)
            ]

            markdown = "\n\n".join(markdown_parts)

            # Advisory only: warn when pages look scanned or garbled so the
            # user can re-run with --ocr (behavior is never changed here)
            self._warn_scanned_or_garbled(input_path)

            # Fix image paths in markdown: pymupdf4llm uses absolute/full paths,
            # we need relative paths (assets/xxx.jpg)
            markdown = self._fix_image_paths(markdown, image_path)

            # Collect extracted images (only for current file). Resolve the
            # refs the converter wrote into the markdown (plus demoted
            # reference images) — pymupdf4llm sanitizes the source filename
            # when naming assets (spaces → "_", parens → "-"), so a prefix
            # glob on the input name misses them.
            if write_images and image_path.exists():
                ref_names = extract_asset_image_names(markdown)
                for ref in reference_images:
                    ref_name = ref.get("name")
                    if (
                        isinstance(ref_name, str)
                        and ref_name
                        and ref_name not in ref_names
                    ):
                        ref_names.append(ref_name)
                img_files = [
                    image_path / name
                    for name in ref_names
                    if (image_path / name).is_file()
                ]
                if not img_files:
                    # Legacy fallback: prefix glob on the input file name
                    file_prefix = escape_glob_pattern(input_path.name)
                    img_files = sorted(
                        image_path.glob(f"{file_prefix}*.{image_format}")
                    )
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
            }
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

    def _sanitize_hidden_text(
        self,
        input_path: Path,
        page_numbers: list[int],
        page_texts: list[str],
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

        Detection failures are debug-logged and never break a conversion.

        Args:
            input_path: Path to the PDF file being converted
            page_numbers: 1-based page number for each entry in page_texts
            page_texts: Per-page markdown text, parallel to page_numbers

        Returns:
            Page texts, possibly with hidden text removed
        """
        mode = self.config.security.pdf_sanitize if self.config else "warn"
        if mode == "off":
            return page_texts

        try:
            import pymupdf

            doc = pymupdf.open(input_path)
            try:
                hidden = collect_hidden_text(doc)
            finally:
                doc.close()
        except Exception as e:
            logger.debug(
                "[PDF] Hidden-text detection failed for {}: {}", input_path.name, e
            )
            return page_texts

        if not hidden:
            return page_texts

        total_spans = sum(len(texts) for texts in hidden.values())
        excerpt = "; ".join(t for texts in hidden.values() for t in texts)
        if len(excerpt) > _HIDDEN_EXCERPT_MAX_CHARS:
            excerpt = excerpt[:_HIDDEN_EXCERPT_MAX_CHARS] + "..."
        logger.warning(
            "[PDF] {} hidden text span(s) detected on page(s) {} "
            "(possible prompt injection; excerpt: {!r}){}",
            total_spans,
            ", ".join(str(p) for p in sorted(hidden)),
            excerpt,
            "; removing verbatim matches from output"
            if mode == "remove"
            else "; set security.pdf_sanitize to 'remove' to strip it",
        )

        if mode != "remove":
            return page_texts

        index_by_page = {page_num: i for i, page_num in enumerate(page_numbers)}
        result = list(page_texts)
        for page_num, texts in hidden.items():
            idx = index_by_page.get(page_num)
            if idx is None:
                continue
            for text in texts:
                result[idx] = result[idx].replace(text, "")
        return result

    def _warn_scanned_or_garbled(self, input_path: Path) -> None:
        """Emit one consolidated warning when pages look scanned or garbled.

        Advisory only: conversion behavior is unchanged and OCR is never
        auto-enabled. Failures are swallowed (debug-logged) so the check
        can never break a conversion.

        Args:
            input_path: Path to the PDF file being converted
        """
        try:
            import pymupdf

            doc = pymupdf.open(input_path)
            try:
                scanned_pages, garbled_pages = collect_page_advisories(doc)
            finally:
                doc.close()
        except Exception as e:
            logger.debug(
                "[PDF] Scan/garbled advisory check failed for {}: {}",
                input_path.name,
                e,
            )
            return

        flagged = sorted(set(scanned_pages) | set(garbled_pages))
        if not flagged:
            return
        logger.warning(
            "[PDF] {} page(s) look scanned/garbled (pages {}); "
            "consider re-running with --ocr",
            len(flagged),
            ", ".join(str(p) for p in flagged),
        )

    def _render_pages_parallel(
        self,
        input_path: Path,
        screenshots_dir: Path,
        image_format: str,
        dpi: int = DEFAULT_RENDER_DPI,
        max_workers: int | None = None,
    ) -> list[tuple[ExtractedImage, dict]]:
        """Render PDF pages as images in parallel using ThreadPoolExecutor.

        Each thread opens its own PDF document for thread safety (PyMuPDF is not thread-safe).

        Args:
            input_path: Path to the PDF file
            screenshots_dir: Directory to save screenshots
            image_format: Image format (jpg, png, etc.)
            dpi: Render DPI
            max_workers: Override worker count. If None, auto-detect based on file size.

        Returns:
            List of (ExtractedImage, page_info_dict) tuples, sorted by page number.
        """
        import pymupdf

        # Create ImageProcessor once (thread-safe for read-only config access)
        img_processor = ImageProcessor(self.config.image if self.config else None)

        screenshots_dir.mkdir(parents=True, exist_ok=True)

        # Get total pages (lightweight - only reads PDF metadata)
        doc = pymupdf.open(input_path)
        total_pages = len(doc)
        doc.close()

        if total_pages == 0:
            return []

        # Auto-detect worker count if not specified
        if max_workers is None:
            max_workers = self._get_worker_count(input_path, total_pages)

        def _render_single_page(page_num: int) -> tuple[ExtractedImage, dict]:
            """Render a single page (thread-safe).

            Each thread opens its own document copy to ensure thread safety.
            PyMuPDF is not thread-safe when sharing document objects.
            """
            thread_doc = pymupdf.open(input_path)
            try:
                page = thread_doc[page_num]

                # Render page to image
                mat = pymupdf.Matrix(dpi / 72, dpi / 72)
                pix = page.get_pixmap(matrix=mat)

                # Save page image with compression (ensures < 5MB for LLM)
                image_name = f"{input_path.name}.page{page_num + 1:04d}.{image_format}"
                image_path = screenshots_dir / image_name
                final_size, actual_path = img_processor.save_screenshot(
                    pix.samples, pix.width, pix.height, image_path
                )

                # Use the actual path returned by save_screenshot, which may
                # differ from image_path when the fallback changes the extension
                actual_name = actual_path.name
                actual_mime = get_mime_type(
                    actual_path.suffix, default=f"image/{image_format}"
                )

                extracted_img = ExtractedImage(
                    path=actual_path,
                    index=page_num + 1,
                    original_name=actual_name,
                    mime_type=actual_mime,
                    width=final_size[0],
                    height=final_size[1],
                )

                page_info = {
                    "page": page_num + 1,
                    "path": str(actual_path),
                    "name": actual_name,
                }

                return (extracted_img, page_info)
            finally:
                thread_doc.close()

        # Render pages in parallel
        results: list[tuple[int, ExtractedImage, dict]] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_render_single_page, i): i for i in range(total_pages)
            }

            for future in as_completed(futures):
                page_num = futures[future]
                try:
                    extracted_img, page_info = future.result()
                    results.append((page_num, extracted_img, page_info))
                except Exception as e:
                    logger.error(f"Failed to render page {page_num + 1}: {e}")
                    raise

        # Sort by page number to maintain order
        results.sort(key=lambda x: x[0])

        return [(img, info) for _, img, info in results]

    def _fix_image_paths(self, markdown: str, image_path: Path) -> str:
        """Fix image paths to be relative to output directory.

        pymupdf4llm generates paths like: ![](full/path/to/assets/image.jpg)
        We need: ![](assets/image.jpg)

        Uses simple string replacement instead of regex — pymupdf4llm always
        outputs deterministic `](path/filename)` format, and str.replace is
        immune to special characters in filenames (parentheses, $, etc.).

        Note: pymupdf4llm always uses forward slashes in markdown, even on Windows.
        We must use as_posix() to ensure consistent path matching.
        """
        posix_path = image_path.as_posix()
        return markdown.replace(f"]({posix_path}/", f"]({ASSETS_REL_PATH}/")

    def _collect_embedded_images(
        self, assets_dir: Path, input_name: str, markdown: str = ""
    ) -> list[ExtractedImage]:
        """Collect embedded images extracted by pymupdf4llm.

        pymupdf4llm extracts embedded images with names like: filename.pdf-0-0.png
        (page index - image index on that page). The name prefix is a
        sanitized form of the source filename (spaces → "_"), so files are
        resolved from the markdown refs first; matching on the raw input
        name is kept as a fallback for markdown without asset refs.

        Args:
            assets_dir: Directory where images were extracted
            input_name: Original PDF filename
            markdown: Converted markdown whose asset refs name the files

        Returns:
            List of ExtractedImage for embedded images
        """
        embedded_images: list[ExtractedImage] = []
        # Suffix pattern: ...-{page}-{index}.{ext}
        index_pattern = re.compile(r"-(\d+)-(\d+)\.(png|jpg|jpeg|webp)$", re.IGNORECASE)

        image_files = [
            assets_dir / name
            for name in extract_asset_image_names(markdown)
            if (assets_dir / name).is_file()
        ]
        if not image_files:
            legacy_pattern = re.compile(
                rf"^{re.escape(input_name)}-(\d+)-(\d+)\.(png|jpg|jpeg|webp)$"
            )
            image_files = [
                f for f in assets_dir.iterdir() if legacy_pattern.match(f.name)
            ]

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

        Also renders each page as an image (if enable_screenshot) for reference.

        Args:
            input_path: Path to the PDF file
            output_dir: Optional output directory for extracted images

        Returns:
            ConvertResult containing OCR-extracted markdown with commented page images
        """
        try:
            import pymupdf
        except ImportError as e:
            raise ImportError(
                "PyMuPDF is not installed. Install with: uv add pymupdf"
            ) from e

        from markitai.ocr import OCRProcessor

        ocr_config = self.config.ocr if self.config else None
        ocr = OCRProcessor(ocr_config)

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

        images: list[ExtractedImage] = []
        page_images: list[dict] = []
        markdown_parts = []
        dpi = DEFAULT_RENDER_DPI

        # Step 2: Render each page as image (only if screenshot enabled)
        # Use parallel processing for better performance
        # Per-page OCR routing: pages with a healthy native text layer keep
        # that text; only scanned/garbled pages go through OCR.
        per_page_routing = ocr_config.per_page_routing if ocr_config else True
        native_texts: dict[int, str] = {}
        doc = pymupdf.open(input_path)
        total_pages = len(doc)
        if per_page_routing:
            try:
                native_texts = _collect_native_text_pages(doc)
            except Exception as e:
                logger.debug("[PDF] OCR routing check failed: {}", e)
                native_texts = {}
        doc.close()
        if per_page_routing:
            logger.debug(
                "OCR routing: {} pages native, {} pages OCR",
                len(native_texts),
                total_pages - len(native_texts),
            )

        max_workers = self._get_worker_count(input_path, total_pages)

        if enable_screenshot:
            screenshots_dir.mkdir(parents=True, exist_ok=True)

            def process_page_with_screenshot(page_num: int) -> dict:
                """Process a single page: render + OCR (thread-safe)."""
                # Each thread opens its own document (PyMuPDF not thread-safe)
                thread_doc = pymupdf.open(input_path)
                img_processor = ImageProcessor(
                    self.config.image if self.config else None
                )
                try:
                    page = thread_doc[page_num]

                    # Render page to image
                    mat = pymupdf.Matrix(dpi / 72, dpi / 72)
                    pix = page.get_pixmap(matrix=mat)

                    # Save page image with compression
                    image_name = (
                        f"{input_path.name}.page{page_num + 1:04d}.{image_format}"
                    )
                    image_path = screenshots_dir / image_name
                    final_size, actual_path = img_processor.save_screenshot(
                        pix.samples, pix.width, pix.height, image_path
                    )

                    # Use the actual path returned by save_screenshot, which may
                    # differ from image_path when the fallback changes the extension
                    actual_name = actual_path.name
                    actual_mime = get_mime_type(
                        actual_path.suffix, default=f"image/{image_format}"
                    )

                    if page_num in native_texts:
                        # Routed native: page has a healthy text layer, skip OCR
                        text_content = native_texts[page_num]
                    else:
                        # OCR - reuse already rendered pixmap to avoid re-rendering
                        try:
                            result = ocr.recognize_pixmap(
                                pix.samples, pix.width, pix.height, pix.n
                            )
                            text_content = (
                                result.text.strip()
                                if result.text.strip()
                                else "*(No text detected)*"
                            )
                        except Exception as e:
                            logger.warning(f"OCR failed for page {page_num + 1}: {e}")
                            text_content = f"*(OCR failed: {e})*"

                    page_content = f"{text_content}\n\n<!-- ![Page {page_num + 1}]({SCREENSHOTS_REL_PATH}/{actual_name}) -->"

                    return {
                        "page_num": page_num,
                        "image": ExtractedImage(
                            path=actual_path,
                            index=page_num + 1,
                            original_name=actual_name,
                            mime_type=actual_mime,
                            width=final_size[0],
                            height=final_size[1],
                        ),
                        "page_image": {
                            "page": page_num + 1,
                            "path": str(actual_path),
                            "name": actual_name,
                        },
                        "markdown": page_content,
                    }
                finally:
                    thread_doc.close()

            # Process pages in parallel
            results: dict[int, dict] = {}
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(process_page_with_screenshot, i): i
                    for i in range(total_pages)
                }
                for future in as_completed(futures):
                    page_num = futures[future]
                    try:
                        result = future.result()
                        results[page_num] = result
                        logger.debug(f"OCR processed page {page_num + 1}/{total_pages}")
                    except Exception as e:
                        logger.error(f"Failed to process page {page_num + 1}: {e}")
                        results[page_num] = {
                            "page_num": page_num,
                            "image": None,
                            "page_image": None,
                            "markdown": f"*(Page processing failed: {e})*",
                        }

            # Collect results in order
            for i in range(total_pages):
                r = results[i]
                if r["image"]:
                    images.append(r["image"])
                if r["page_image"]:
                    page_images.append(r["page_image"])
                markdown_parts.append(r["markdown"])
        else:

            def process_page_ocr_only(page_num: int) -> dict:
                """Process a single page: OCR only (thread-safe)."""
                if page_num in native_texts:
                    # Routed native: page has a healthy text layer, skip OCR
                    return {"page_num": page_num, "markdown": native_texts[page_num]}
                try:
                    result = ocr.recognize_pdf_page(input_path, page_num, dpi=dpi)
                    text_content = (
                        result.text.strip()
                        if result.text.strip()
                        else "*(No text detected)*"
                    )
                except Exception as e:
                    logger.warning(f"OCR failed for page {page_num + 1}: {e}")
                    text_content = f"*(OCR failed: {e})*"
                return {"page_num": page_num, "markdown": text_content}

            # Process pages in parallel
            results: dict[int, dict] = {}
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(process_page_ocr_only, i): i
                    for i in range(total_pages)
                }
                for future in as_completed(futures):
                    page_num = futures[future]
                    try:
                        result = future.result()
                        results[page_num] = result
                        logger.debug(f"OCR processed page {page_num + 1}/{total_pages}")
                    except Exception as e:
                        logger.error(f"Failed to process page {page_num + 1}: {e}")
                        results[page_num] = {
                            "page_num": page_num,
                            "markdown": f"*(OCR failed: {e})*",
                        }

            # Collect results in order
            for i in range(total_pages):
                markdown_parts.append(results[i]["markdown"])

        extracted_text = f"# {input_path.stem}\n\n" + "\n\n".join(markdown_parts)

        return ConvertResult(
            markdown=extracted_text,
            images=images,
            metadata={
                "source": str(input_path),
                "format": "PDF",
                "ocr_used": True,
                "pages": len(markdown_parts),
                "extracted_text": extracted_text,
                "page_images": page_images,
            },
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

        # Step 1: Extract text using pymupdf4llm (fast, preserves structure)
        logger.debug("Extracting text with pymupdf4llm...")
        extracted_text = cast(
            str,
            pymupdf4llm.to_markdown(
                str(input_path),
                write_images=True,
                image_path=str(assets_dir),
                image_format=image_format,
                dpi=DEFAULT_RENDER_DPI,
                force_text=True,
                use_ocr=False,  # Markitai handles OCR separately; suppress Tesseract probing
            ),
        )
        extracted_text = self._fix_image_paths(extracted_text, assets_dir)

        # Collect embedded images extracted by pymupdf4llm
        embedded_images = self._collect_embedded_images(
            assets_dir, input_path.name, extracted_text
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
            for extracted_img, page_info in page_results:
                images.append(extracted_img)
                page_images.append(page_info)

            if page_images:
                logger.debug(f"Rendered {len(page_images)} page screenshots")

        # Clean up temporary directories if used (no output_dir)
        for temp_dir in (temp_assets, temp_screenshots):
            if temp_dir and temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
        # Remove images pointing into deleted temp dirs
        if temp_assets or temp_screenshots:
            images = [img for img in images if img.path and img.path.exists()]

        return ConvertResult(
            markdown=extracted_text,
            images=images,
            metadata={
                "source": str(input_path),
                "format": "PDF",
                "pages": len(page_images) if page_images else 0,
                "extracted_text": extracted_text,
                "page_images": page_images,
            },
        )
