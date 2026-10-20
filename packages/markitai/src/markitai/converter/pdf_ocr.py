"""Reading the pages of a scanned PDF: render, keep a screenshot, OCR.

One page is one task. In a process markitai owns the tasks run in the PDF
extraction workers (:func:`markitai.converter.pdf_parallel.run_page_tasks`),
each worker running onnxruntime on one thread and keeping its OCR engine for
every later page. A page costs a few CPU seconds, nearly all of it in the
recognizer: onnxruntime spreading one page over every core gains little,
while one page per core gains nearly the core count. Elsewhere (a host
program, a short document) the same task runs on threads in this process.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from markitai.config import ImageConfig, OCRConfig


@dataclass(frozen=True)
class PageShot:
    """Where a page's screenshot goes: ``<directory>/<prefix>.page0001.jpg``."""

    directory: str
    prefix: str
    image_format: str
    image: ImageConfig | None


def read_page(
    ocr: Any,
    path: Path | str,
    page_num: int,
    dpi: int,
    *,
    read: bool,
    shot: PageShot | None,
) -> dict[str, Any]:
    """Render page *page_num* (0-based), save its screenshot, OCR it.

    Args:
        ocr: The :class:`markitai.ocr.OCRProcessor` to read with.
        path: The PDF.
        page_num: The page, 0-based.
        dpi: Render resolution.
        read: OCR the page (False: only its screenshot is wanted).
        shot: Save the render as the page's screenshot.

    Returns:
        ``{"text": str | None, "page_image": {"page", "path", "name"} | None}``
    """
    if shot is None:
        text = ocr.recognize_pdf_page(Path(path), page_num, dpi=dpi).text.strip()
        return {"text": text, "page_image": None}

    pix, _size, saved = _render_and_save(path, page_num, dpi, shot)
    text: str | None = None
    if read:
        # The render the screenshot came from, not a second one
        result = ocr.recognize_pixmap(pix.samples, pix.width, pix.height, pix.n)
        text = result.text.strip()
    # The saved name: the fallback may have changed its extension
    return {
        "text": text,
        "page_image": {"page": page_num + 1, "path": str(saved), "name": saved.name},
    }


def _render_and_save(
    path: Path | str, page_num: int, dpi: int, shot: PageShot
) -> tuple[Any, tuple[int, int], Path]:
    """Render page *page_num* and save it as its screenshot.

    Returns:
        The render, the saved image's size, and where it was saved (the
        fallback may have changed its extension).
    """
    import pymupdf

    from markitai.image import ImageProcessor
    from markitai.ocr import page_render_matrix

    # Each task opens its own document (PyMuPDF is not thread-safe)
    with pymupdf.open(path) as doc:
        page = doc[page_num]
        pix = page.get_pixmap(matrix=page_render_matrix(page, dpi))
    name = f"{shot.prefix}.page{page_num + 1:04d}.{shot.image_format}"
    size, saved = ImageProcessor(shot.image).save_screenshot(
        pix.samples, pix.width, pix.height, Path(shot.directory) / name
    )
    return pix, size, saved


def shoot_page(
    path: Path | str, page_num: int, dpi: int, shot: PageShot
) -> tuple[tuple[int, int], str]:
    """Render page *page_num* (0-based) and save its screenshot, no OCR.

    Returns:
        The saved image's size and its path.
    """
    _pix, size, saved = _render_and_save(path, page_num, dpi, shot)
    return size, str(saved)


def shoot_page_task(
    path: str, page_num: int, _threads: int, dpi: int, shot: PageShot
) -> tuple[tuple[int, int], str]:
    """:func:`shoot_page` as a worker task (see ``run_page_tasks``).

    PyMuPDF holds the GIL while it renders, so threads in one process
    render one page at a time; the workers render one each.
    """
    return shoot_page(path, page_num, dpi, shot)


def read_page_task(
    path: str,
    page_num: int,
    threads: int,
    dpi: int,
    config: OCRConfig | None,
    to_read: frozenset[int],
    shot: PageShot | None,
) -> dict[str, Any]:
    """:func:`read_page` as a worker task, reading the page if in *to_read*.

    The worker keeps its engine (``OCRProcessor``'s process-wide one, run
    on *threads* threads) for every later page.
    """
    from markitai.ocr import OCRProcessor

    return read_page(
        OCRProcessor(config, threads=threads),
        path,
        page_num,
        dpi,
        read=page_num in to_read,
        shot=shot,
    )
