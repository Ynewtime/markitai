"""Document structure of OCR'd pages (markitai.ocr_structure, ocr_layout).

The owner's 1.2.1 check: a scan of sample.pdf came out of OCR with its
headings as paragraphs, its table as lines of text, its bullets gone, its
chart's labels read as a line of text and its emphasis lost.

The synthetic page is drawn with PyMuPDF's built-in fonts and rendered at
150 dpi; its line boxes come from the text layer, padded like a detector's.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pymupdf
import pytest

from markitai.ocr_layout import (
    place_figures,
    resolve_heading_levels,
    structure_ocr_page,
)
from markitai.ocr_structure import (
    BOLD,
    ITALIC,
    emphasized,
    page_pixels,
    smooth_emphasis,
)

DPI = 150
SCALE = DPI / 72
BODY = (
    "Morbi tempor porta nunc, sed rutrum lacus feugiat vitae. Integer",
    "vulputate quam at tortor finibus, quis pulvinar nunc aliquet.",
    "Curabitur semper arcu ac ligula semper, nec luctus nisl blandit.",
)


def _text(page: Any, x: float, y: float, text: str, font: str, size: float) -> float:
    page.insert_text((x, y), text, fontname=font, fontsize=size)
    return x + pymupdf.get_text_length(text, fontname=font, fontsize=size)


def _page() -> tuple[np.ndarray, list[str], list[tuple[float, float, float, float]]]:
    """A page with every structure the pixels and boxes should show."""
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    _text(page, 72, 80, "Quarterly report", "hebo", 24)
    y = 120
    for line in BODY:
        _text(page, 72, y, line, "helv", 10.5)
        y += 15
    # One sentence of a paragraph in bold
    x = _text(page, 72, y, "Plain words first, then ", "helv", 10.5)
    x = _text(page, x, y, "strongly marked bold phrase", "hebo", 10.5)
    _text(page, x, y, " and plain words again.", "helv", 10.5)
    _text(page, 72, 200, "Methods and materials", "hebo", 16)
    _text(
        page,
        72,
        225,
        "Vestibulum neque massa, scelerisque sit amet ligula.",
        "helv",
        10.5,
    )
    # Three items with a drawn bullet each (a recognizer drops the dots)
    for index, item in enumerate(("First item of the list", "Second one", "Third")):
        top = 250 + index * 18
        page.draw_circle((84, top - 3.5), 2, color=(0, 0, 0), fill=(0, 0, 0))
        _text(page, 96, top, item, "helv", 10.5)
    # A ruled table, light grey rules
    columns, rows = [72, 220, 330, 500], [320, 342, 364, 386]
    for x in columns:
        page.draw_line((x, rows[0]), (x, rows[-1]), color=(0.7, 0.7, 0.7), width=0.6)
    for y in rows:
        page.draw_line(
            (columns[0], y), (columns[-1], y), color=(0.7, 0.7, 0.7), width=0.6
        )
    cells = [("Name", "Count", "Note"), ("Apples", "12", "Fresh"), ("Pears", "7", "")]
    for r, row in enumerate(cells):
        for c, cell in enumerate(row):
            if cell:
                _text(page, columns[c] + 4, rows[r] + 15, cell, "helv", 10.5)
    # A bar chart with axis labels and a legend
    page.draw_line((140, 470), (140, 580), color=(0.6, 0.6, 0.6), width=0.6)
    page.draw_line((140, 580), (400, 580), color=(0.6, 0.6, 0.6), width=0.6)
    for index, height in enumerate((60, 95, 40, 80)):
        x0 = 160 + index * 60
        page.draw_rect(
            pymupdf.Rect(x0, 580 - height, x0 + 30, 580),
            color=None,
            fill=(0.0, 0.25, 0.55) if index % 2 else (1.0, 0.27, 0.0),
        )
        _text(page, x0 + 6, 594, f"Q{index + 1}", "helv", 9)
    for value, y in (("100", 476), ("50", 530), ("0", 584)):
        _text(page, 118, y, value, "helv", 9)
    _text(page, 378, 520, "Revenue", "helv", 9)
    _text(
        page,
        72,
        640,
        "Etiam vehicula luctus fermentum, in vel metus congue nunc.",
        "helv",
        10.5,
    )
    _text(
        page,
        72,
        655,
        "Pellentesque fringilla mollis efficitur, nullam venenatis.",
        "helv",
        10.5,
    )

    pix = page.get_pixmap(dpi=DPI)
    image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n
    )[:, :, :3]
    texts: list[str] = []
    bounds: list[tuple[float, float, float, float]] = []
    content: Any = page.get_text("dict")
    for block in content["blocks"]:
        for line in block.get("lines", []):
            text = "".join(span["text"] for span in line["spans"]).strip()
            if not text:
                continue
            x0, y0, x1, y1 = (v * SCALE for v in line["bbox"])
            # A detector's box: a few pixels of padding around the ink
            texts.append(text)
            bounds.append((x0 - 3, y0 - 2, x1 + 3, y1 + 2))
    return image.copy(), texts, bounds


@pytest.fixture(scope="module")
def laid_out() -> tuple[str, Any]:
    pytest.importorskip("cv2")
    image, texts, bounds = _page()
    page = structure_ocr_page(texts, bounds, image, figures=True)
    markdown = resolve_heading_levels([page.markdown])[0]
    markdown = place_figures(markdown, ["![](figure-1.jpg)"] * len(page.figures))
    return markdown, page


def test_headings_by_type_size(laid_out: tuple[str, Any]) -> None:
    markdown, _page = laid_out
    headings = [line for line in markdown.splitlines() if line.startswith("#")]
    assert headings == ["# Quarterly report", "## Methods and materials"]


def test_drawn_bullets_make_a_tight_list(laid_out: tuple[str, Any]) -> None:
    markdown, _page = laid_out
    assert "- First item of the list\n- Second one\n- Third" in markdown


def test_a_ruled_grid_is_a_table(laid_out: tuple[str, Any]) -> None:
    markdown, page = laid_out
    assert (
        "| Name | Count | Note |\n| --- | --- | --- |\n"
        "| Apples | 12 | Fresh |\n| Pears | 7 |  |"
    ) in markdown
    # Compare reads the cells where the table is, after the list
    texts = [text for text, _box in page.lines]
    assert (
        texts.index("Third")
        < texts.index("Name")
        < texts.index("Etiam vehicula luctus fermentum, in vel metus congue nunc.")
    )


def test_a_chart_is_a_picture_and_its_labels_are_not_text(
    laid_out: tuple[str, Any],
) -> None:
    markdown, page = laid_out
    assert len(page.figures) == 1
    x0, y0, x1, y1 = page.figures[0]
    # The bars, the axis numbers and the legend, in pixels
    assert x0 < 118 * SCALE and x1 > 400 * SCALE  # the legend beside the bars
    assert y0 < 470 * SCALE and y1 > 594 * SCALE - 10
    for label in ("Q1", "Q4", "100", "Revenue"):
        assert label not in markdown
    figure = markdown.index("![](figure-1.jpg)")
    assert markdown.index("| Pears |") < figure < markdown.index("Etiam vehicula")


def test_a_bold_phrase_is_bold(laid_out: tuple[str, Any]) -> None:
    markdown, _page = laid_out
    assert "then **strongly marked bold phrase** and plain words again." in markdown
    # The body around it is not
    assert "**Morbi" not in markdown


def test_without_pixels_only_the_geometry_speaks() -> None:
    _image, texts, bounds = _page()
    page = structure_ocr_page(texts, bounds, None)
    markdown = resolve_heading_levels([page.markdown])[0]
    assert markdown.startswith("# Quarterly report")
    assert "**" not in markdown and "![" not in markdown


def test_a_dark_page_is_not_analysed() -> None:
    assert page_pixels(np.full((50, 50, 3), 30, dtype=np.uint8)) is None


def test_heading_levels_are_tiers_across_pages() -> None:
    """Sizes within 8 % of a tier's largest share its level, on every page."""
    pages = [
        "\x00H2.350\x00 Title\n\n\x00H1.590\x00 Part one",
        "\x00H1.560\x00 Part two\n\n\x00H1.410\x00 Section",
    ]
    assert resolve_heading_levels(pages) == [
        "# Title\n\n## Part one",
        "## Part two\n\n### Section",
    ]


def test_emphasis_runs_cross_line_breaks_and_take_short_words() -> None:
    words = ["a", "bold", "run", "of", "words", "goes", "on", "here"]
    flags: list[int | None] = [0, BOLD, BOLD, BOLD, BOLD, BOLD, 0, 0]
    smoothed = smooth_emphasis(words, flags)
    assert emphasized(words, smoothed, {4}) == "a **bold run of\nwords goes** on here"
    # A short word alone does not keep its emphasis
    assert smooth_emphasis(["one", "of", "these"], [0, ITALIC, 0]) == [0, 0, 0]


def test_the_scan_of_sample_pdf_reads_like_the_pdf(tmp_path: Path) -> None:
    """Real RapidOCR on a scan of the repository's sample.pdf (its first
    three pages rendered at 150 dpi): the structure its text layer gives."""
    pytest.importorskip("rapidocr")
    from markitai.config import MarkitaiConfig
    from markitai.converter.pdf import PdfConverter

    source = pymupdf.open(Path(__file__).parents[1] / "fixtures" / "sample.pdf")
    scan = pymupdf.open()
    for number in range(3):
        pix = source[number].get_pixmap(dpi=DPI)
        page = scan.new_page(
            width=source[number].rect.width, height=source[number].rect.height
        )
        page.insert_image(page.rect, stream=pix.tobytes("jpeg", jpg_quality=85))
    path = tmp_path / "scan.pdf"
    scan.save(path)
    config = MarkitaiConfig()
    config.ocr.enabled = True
    config.cache.enabled = False

    result = PdfConverter(config).convert(path, tmp_path / "out")
    markdown = result.markdown

    headings = [line for line in markdown.splitlines() if line.startswith("#")]
    assert headings == [
        "# Lorem ipsum",
        "## Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nunc ac faucibus odio.",
        "## Cras fringilla ipsum magna, in fringilla dui commodo a.",
        "## Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
        "### Maecenas mauris lectus, lobortis et purus mattis, blandit dictum tellus.",
        "### In eleifend velit vitae libero sollicitudin euismod.",
    ]
    assert (
        "- **Maecenas non lorem quis tellus placerat varius.**\n- *Nulla facilisi.*"
        in markdown
    )
    assert (
        "| 2 | Cras fringilla ipsum magna, in fringilla dui commodo a. | Ipsum |  |"
        in markdown
    )
    assert "Row 1" not in markdown and "Column 2" not in markdown
    assert "![Image on page 1](.markitai/assets/scan.pdf-0001-01.jpg)" in markdown
    assert (
        tmp_path / "out" / ".markitai" / "assets" / "scan.pdf-0001-01.jpg"
    ).is_file()
    assert "**Vivamus dapibus sodales ex" in markdown
    # Compare's data: each OCR'd page's lines in points
    pages = result.metadata["ocr_pages"]
    assert [p["page"] for p in pages] == [1, 2, 3]
    first = pages[1]["lines"][0]
    assert first["text"].startswith("In non mauris justo.")
    assert 40 < first["bbox"][0] < 70 and first["bbox"][2] < 595


# --- review findings ------------------------------------------------------------


def test_a_cjk_list_item_followed_by_a_measured_latin_line() -> None:
    """A CJK line is one unmeasured word: the glyph's removal must not cut
    the next lines' emphasis out of step (an IndexError failed the page)."""
    from markitai.ocr_layout import _Box, _Page, _render_flow

    page = _Page(body=20)
    first = _Box("• Python 编程 指南", 100, 100, 600, 120)
    more = _Box("with many more details here", 120, 124, 600, 144)
    page.words[id(more)] = [BOLD, BOLD, 0, 0, 0]
    assert _render_flow([first, more], page) == (
        "- Python 编程 指南\n  **with many** more details here"
    )


def test_a_table_beside_text_on_its_top_row_stays_a_table() -> None:
    from markitai.ocr_layout import _Box, _Page, _render_flow

    page = _Page(body=20)
    page.blocks["\x00T0\x00"] = "| A |"
    rows = [
        _Box("As the table shows", 50, 100, 300, 120),
        _Box("\x00T0\x00", 400, 100, 700, 120),
        _Box("And the text goes on.", 50, 300, 300, 320),
    ]
    assert _render_flow(rows, page) == (
        "As the table shows\n\n\x00T0\x00\n\nAnd the text goes on."
    )


def test_words_past_a_tables_outer_rule_stay_in_it() -> None:
    from markitai.ocr_structure import Grid, table_cells

    pixels = page_pixels(np.full((100, 700, 3), 255, dtype=np.uint8))
    assert pixels is not None
    cells = table_cells(
        pixels,
        Grid(rows=[0.0, 50.0], columns=[100.0, 300.0, 500.0]),
        [
            ("alpha beta gamma delta epsilon", (110.0, 10.0, 690.0, 40.0)),
            ("x", (120.0, 10.0, 130.0, 40.0)),
        ],
    )
    assert cells is not None
    words = " ".join(cells[0][0]).split()
    for word in ("alpha", "beta", "gamma", "delta", "epsilon", "x"):
        assert word in words


def test_a_figure_reference_is_encoded(tmp_path: Path) -> None:
    from markitai.converter.pdf_ocr import PageAssets, _save_figure

    image = np.full((60, 80, 3), 200, dtype=np.uint8)
    assets = PageAssets(
        str(tmp_path), ".markitai/assets", "my report (v2).pdf", "jpg", None
    )
    reference = _save_figure(image, (0, 0, 40, 30), 0, 0, assets)
    assert (
        reference
        == "![Image on page 1](.markitai/assets/my%20report%20%28v2%29.pdf-0001-01.jpg)"
    )
    assert (tmp_path / "my report (v2).pdf-0001-01.jpg").is_file()
