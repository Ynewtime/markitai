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

from markitai.ocr_layout import OcrPage, place_figures

if TYPE_CHECKING:
    from markitai.config import ImageConfig, OCRConfig


@dataclass(frozen=True)
class PageShot:
    """Where a page's screenshot goes: ``<directory>/<prefix>.page0001.jpg``."""

    directory: str
    prefix: str
    image_format: str
    image: ImageConfig | None


@dataclass(frozen=True)
class PageAssets:
    """Where a page's figures go: ``<directory>/<prefix>-0001-01.jpg``.

    ``reference`` is the directory as the Markdown names it
    (``.markitai/assets``).
    """

    directory: str
    reference: str
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
    assets: PageAssets | None = None,
) -> dict[str, Any]:
    """Render page *page_num* (0-based), save its screenshot, OCR it.

    The page is read with its structure (``OCRProcessor.recognize_page``):
    its figures are cut out into *assets* and referenced where they stand
    (without *assets* they are left out, their labels too).

    Args:
        ocr: The :class:`markitai.ocr.OCRProcessor` to read with.
        path: The PDF.
        page_num: The page, 0-based.
        dpi: Render resolution.
        read: OCR the page (False: only its screenshot is wanted).
        shot: Save the render as the page's screenshot.
        assets: Save the page's figures.

    Returns:
        ``{"text": str | None, "page_image": {"page", "path", "name"} | None,
        "layout": dict | None}``. ``text`` keeps heading placeholders
        (``ocr_layout.resolve_heading_levels`` sets the levels over the
        whole document); ``layout`` is the page's recognized lines and
        figure boxes in points from its top-left corner (what the Compare
        view's page text answers for an OCR'd page).
    """
    page_image: dict[str, Any] | None = None
    figures = assets is not None
    result: Any = None
    if shot is None:
        if read:
            result = ocr.recognize_pdf_page(
                Path(path), page_num, dpi=dpi, figures=figures
            )
    else:
        pix, rect = _render(path, page_num, dpi)
        saved = _save_shot(pix, page_num, shot)
        # The saved name: the fallback may have changed its extension
        page_image = {"page": page_num + 1, "path": str(saved), "name": saved.name}
        if read:
            # The render the screenshot came from, not a second one
            result = ocr.recognize_pixmap(
                pix.samples, pix.width, pix.height, pix.n, figures=figures
            )
            result.page_size = rect
    if result is None:
        return {"text": None, "page_image": page_image, "layout": None}
    page = getattr(result, "page", None)
    if not isinstance(page, OcrPage):
        return {"text": result.text.strip(), "page_image": page_image, "layout": None}

    references = [
        _save_figure(result.image, box, page_num, index, assets)
        for index, box in enumerate(page.figures)
    ]
    text = place_figures(page.markdown, references).strip()
    layout: dict[str, Any] | None = None
    size = result.page_size
    if size is not None:
        sx = size[0] / result.image.shape[1]
        sy = size[1] / result.image.shape[0]

        def points(box: Any) -> list[float]:
            return [
                round(max(0.0, box[0] * sx), 2),
                round(max(0.0, box[1] * sy), 2),
                round(min(size[0], box[2] * sx), 2),
                round(min(size[1], box[3] * sy), 2),
            ]

        layout = {
            "page": page_num + 1,
            "width": round(size[0], 2),
            "height": round(size[1], 2),
            "lines": [{"text": t, "bbox": points(b)} for t, b in page.lines],
            "images": [
                {"bbox": points(box)}
                for box, reference in zip(page.figures, references, strict=True)
                if reference
            ],
        }
    return {"text": text, "page_image": page_image, "layout": layout}


def _render(
    path: Path | str, page_num: int, dpi: int
) -> tuple[Any, tuple[float, float]]:
    """Page *page_num* rendered at *dpi* (capped), and its size in points."""
    import pymupdf

    from markitai.ocr import page_render_matrix

    # Each task opens its own document (PyMuPDF is not thread-safe)
    with pymupdf.open(path) as doc:
        page = doc[page_num]
        pix = page.get_pixmap(matrix=page_render_matrix(page, dpi))
        rect = (float(page.rect.width), float(page.rect.height))
    return pix, rect


def _save_shot(pix: Any, page_num: int, shot: PageShot) -> Path:
    from markitai.image import ImageProcessor

    name = f"{shot.prefix}.page{page_num + 1:04d}.{shot.image_format}"
    _size, saved = ImageProcessor(shot.image).save_screenshot(
        pix.samples, pix.width, pix.height, Path(shot.directory) / name
    )
    return saved


def _save_figure(
    image: Any, box: Any, page_num: int, index: int, assets: PageAssets | None
) -> str | None:
    """Cut figure *index* out of the page and save it; its Markdown, or
    None when it could not be saved."""
    if assets is None:
        return None
    import numpy as np
    from loguru import logger

    from markitai.image import ImageProcessor

    x0, y0, x1, y1 = (int(round(v)) for v in box)
    crop = np.ascontiguousarray(image[y0:y1, x0:x1])
    if crop.size == 0:
        return None
    name = f"{assets.prefix}-{page_num + 1:04d}-{index + 1:02d}.{assets.image_format}"
    try:
        Path(assets.directory).mkdir(parents=True, exist_ok=True)
        _size, saved = ImageProcessor(assets.image).save_screenshot(
            crop.tobytes(), crop.shape[1], crop.shape[0], Path(assets.directory) / name
        )
    except Exception as e:  # the page's text stands without the picture
        logger.debug("Figure {} of page {} not saved: {}", index + 1, page_num + 1, e)
        return None
    from urllib.parse import quote

    # Encoded as the native pictures' references are (spaces, brackets);
    # alt as markitai-rust writes its PDF pictures (1.3.0 carries it on)
    reference = f"{assets.reference}/{quote(saved.name, safe='/._~-')}"
    return f"![Image on page {page_num + 1}]({reference})"


def shoot_page(
    path: Path | str, page_num: int, dpi: int, shot: PageShot
) -> tuple[tuple[int, int], str]:
    """Render page *page_num* (0-based) and save its screenshot, no OCR.

    Returns:
        The saved image's size and its path.
    """
    from markitai.image import ImageProcessor

    pix, _rect = _render(path, page_num, dpi)
    name = f"{shot.prefix}.page{page_num + 1:04d}.{shot.image_format}"
    size, saved = ImageProcessor(shot.image).save_screenshot(
        pix.samples, pix.width, pix.height, Path(shot.directory) / name
    )
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
    assets: PageAssets | None = None,
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
        assets=assets,
    )
