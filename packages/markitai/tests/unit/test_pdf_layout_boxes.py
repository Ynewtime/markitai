"""Page chrome and rendered-text pictures in the layout engine's page chunks.

``how-to-cjk-footer.pdf`` is an invented three-page how-to printed by
Chromium: screenshots, CJK step lines with bold words (which Skia writes as
Type3 glyphs, and which the layout model takes for pictures), and a running
footer with the guide's title and the page number.
"""

from __future__ import annotations

import re
from pathlib import Path

import pymupdf
import pytest

from markitai.converter.pdf import PdfConverter
from markitai.converter.pdf_layout_boxes import (
    chrome_core,
    chrome_key,
    clean_layout_chunks,
    is_page_number,
)

FIXTURES = Path(__file__).parent.parent / "fixtures"

PICTURE_TEXT = (
    "![](/tmp/x/doc.pdf-{page:04d}-{n:02d}.png)\n\n\n"
    "<!-- Start of picture text -->\n{text}<br><!-- End of picture text -->\n\n"
)


def _pdf(path: Path, pages: int = 3) -> Path:
    """Pages with a footer title and number; page 2 has a picture."""
    doc = pymupdf.open()
    picture = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 40, 30), False)
    picture.set_rect(picture.irect, (200, 120, 90))
    for number in range(1, pages + 1):
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 100), f"Body text of page {number}.", fontsize=12)
        page.insert_text((72, 400), "Step two: open the panel.", fontsize=12)
        page.insert_text((30, 810), "Field Notes", fontsize=8)
        page.insert_text((560, 810), str(number), fontsize=8)
        if number == 2:
            page.insert_image(pymupdf.Rect(72, 150, 472, 350), pixmap=picture)
    doc.save(path)
    return path


def _chunk(page: int, parts: list[tuple[str, tuple[float, ...], str]]) -> dict:
    text, boxes = "", []
    for cls, bbox, markdown in parts:
        boxes.append(
            {"class": cls, "bbox": bbox, "pos": (len(text), len(text) + len(markdown))}
        )
        text += markdown
    return {"metadata": {"page_number": page}, "text": text, "page_boxes": boxes}


def _footer(page: int) -> list[tuple[str, tuple[float, ...], str]]:
    return [
        ("page-footer", (30, 802, 80, 812), "Field Notes \n\n"),
        ("page-footer", (560, 802, 566, 812), f"{page} \n\n"),
    ]


class TestKeys:
    @pytest.mark.parametrize(
        "text",
        [
            "3",
            "- 3 -",
            "Page 3",
            "page 3 of 9",
            "3 / 9",
            "第 3 页",
            "第3页，共9页",
            "[3]",
        ],
    )
    def test_page_numbers(self, text):
        assert is_page_number(chrome_key(text))

    @pytest.mark.parametrize("text", ["Field Notes", "Chapter 3", "3 apples"])
    def test_not_page_numbers(self, text):
        assert not is_page_number(chrome_key(text))

    def test_a_title_joined_to_its_number_matches_the_title(self):
        assert chrome_core(chrome_key("**Field Notes** 3")) == "field notes"
        assert chrome_core(chrome_key("12 Field Notes")) == "field notes"


class TestCleanLayoutChunks:
    def test_repeated_footer_and_page_numbers_are_dropped(self, tmp_path):
        path = _pdf(tmp_path / "doc.pdf")
        chunks = [
            _chunk(
                n,
                [
                    ("text", (72, 90, 300, 104), f"Body text of page {n}.\n\n"),
                    *_footer(n),
                ],
            )
            for n in (1, 2, 3)
        ]
        cleaned = clean_layout_chunks(path, chunks)
        for n, chunk in enumerate(cleaned, start=1):
            assert chunk["text"] == f"Body text of page {n}.\n\n"
            assert [box["class"] for box in chunk["page_boxes"]] == ["text"]

    def test_a_one_off_header_stays(self, tmp_path):
        path = _pdf(tmp_path / "doc.pdf")
        header = ("page-header", (72, 20, 300, 34), "Draft for review \n\n")
        chunks = [
            _chunk(1, [header, ("text", (72, 90, 300, 104), "Body.\n\n")]),
            _chunk(2, [("text", (72, 90, 300, 104), "More.\n\n")]),
        ]
        cleaned = clean_layout_chunks(path, chunks)
        assert cleaned[0]["text"].startswith("Draft for review")

    def test_chrome_away_from_the_page_edge_stays(self, tmp_path):
        path = _pdf(tmp_path / "doc.pdf")
        mid = ("page-footer", (72, 400, 300, 412), "Field Notes \n\n")
        chunks = [_chunk(n, [mid]) for n in (1, 2)]
        cleaned = clean_layout_chunks(path, chunks)
        assert all("Field Notes" in chunk["text"] for chunk in cleaned)

    def test_a_picture_of_text_becomes_the_text(self, tmp_path):
        path = _pdf(tmp_path / "doc.pdf")
        line = PICTURE_TEXT.format(page=2, n=1, text="Step two: open the panel.")
        chunk = _chunk(2, [("picture", (72, 388, 300, 404), line), *_footer(2)])
        [cleaned] = clean_layout_chunks(path, [chunk])
        # One page: its footer title is not known to repeat, its number goes
        assert cleaned["text"] == "\nStep two: open the panel.\n\nField Notes \n\n"
        assert cleaned["page_boxes"][0]["class"] == "text"

    def test_a_real_picture_keeps_its_image(self, tmp_path):
        path = _pdf(tmp_path / "doc.pdf")
        figure = PICTURE_TEXT.format(page=2, n=0, text="Option 1")
        chunk = _chunk(2, [("picture", (72, 150, 472, 350), figure)])
        [cleaned] = clean_layout_chunks(path, [chunk])
        assert cleaned["text"] == figure
        assert cleaned["page_boxes"][0]["class"] == "picture"

    def test_positions_follow_the_new_text(self, tmp_path):
        path = _pdf(tmp_path / "doc.pdf")
        line = PICTURE_TEXT.format(page=2, n=1, text="Step two: open the panel.")
        parts = [*_footer(2)[:1], ("picture", (72, 388, 300, 404), line)]
        parts.append(("text", (72, 90, 300, 104), "Tail.\n"))
        chunks = [_chunk(2, parts), _chunk(3, _footer(3))]
        [cleaned, _] = clean_layout_chunks(path, chunks)
        for box in cleaned["page_boxes"]:
            start, end = box["pos"]
            assert cleaned["text"][start:end].strip() in {
                "Step two: open the panel.",
                "Tail.",
            }

    def test_chunks_without_boxes_are_left_alone(self, tmp_path):
        chunks = [{"text": "plain", "metadata": {"page_number": 1}}, "raw text"]
        assert clean_layout_chunks(tmp_path / "missing.pdf", chunks) == chunks


class TestFixture:
    def test_chromium_how_to_loses_its_footer_and_doubled_lines(self, tmp_path):
        markdown = (
            PdfConverter()
            .convert(FIXTURES / "how-to-cjk-footer.pdf", tmp_path)
            .markdown
        )
        body = markdown.split("\n", 2)[-1]
        # The running footer and the page numbers are gone
        assert "Desk Lamp Proxy Guide" not in body.replace(
            "# **Desk Lamp Proxy Guide**", ""
        )
        assert not re.search(r"^\s*\d+\s*$", markdown, re.MULTILINE)
        # Each step line is text once, not a picture of it plus the text
        for step in ("2、列表加载后", "3、最后，在控制台"):
            assert markdown.count(step) == 1
        assert "picture text" not in markdown
        # Only the three screenshots are pictures
        assert len(re.findall(r"!\[[^\]]*\]\(", markdown)) == 3
