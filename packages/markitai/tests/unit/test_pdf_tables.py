"""Tables without ruling lines that the PDF extraction flattened.

``fixtures/default-table.pdf`` (from markitai-rust's PDF fixtures: a table
in a browser's default style, no borders, cells a few points apart) came out
of the default conversion as one line, ``Language Year Typing Primary Use
Python 1991 ...``. The other cases are drawn with base-14 fonts, so the word
positions are the same on every platform.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from markitai.converter.pdf_tables import (
    find_unruled_tables,
    replace_flattened_tables,
    restore_flattened_tables,
)

LANGUAGES = [
    ["Language", "Year", "Typing", "Primary Use"],
    ["Python", "1991", "Dynamic", "General purpose"],
    ["Rust", "2015", "Static", "Systems programming"],
    ["TypeScript", "2012", "Static", "Web development"],
]


def _draw_rows(
    rows: list[list[str]], xs: list[float], font: str = "helv"
) -> pymupdf.Document:
    """One page with each row's cells written at the given x positions."""
    doc = pymupdf.open()
    page = doc.new_page()
    for r, row in enumerate(rows):
        for x, text in zip(xs, row):
            page.insert_text((x, 120 + r * 19), text, fontname=font, fontsize=11)
    return doc


def _draw_lines(lines: list[str], font: str = "helv") -> pymupdf.Document:
    doc = pymupdf.open()
    page = doc.new_page()
    for i, line in enumerate(lines):
        page.insert_text((40, 120 + i * 15), line, fontname=font, fontsize=11)
    return doc


def test_the_flattened_fixture_table_comes_back_as_a_table(fixtures_dir: Path) -> None:
    from markitai.converter.pdf import PdfConverter

    markdown = PdfConverter().convert(fixtures_dir / "default-table.pdf").markdown

    assert "|Language|Year|Typing|Primary Use|\n|---|---|---|---|" in markdown
    assert "|TypeScript|2012|Static|Web development|" in markdown
    assert "Language Year Typing" not in markdown
    # the prose around it is untouched
    assert "Each language has its own trade-offs." in markdown


def test_columns_come_from_the_gaps_every_row_shares() -> None:
    doc = _draw_rows(LANGUAGES, [36, 110, 150, 205])

    assert find_unruled_tables(doc[0]) == [LANGUAGES]


@pytest.mark.parametrize(
    "lines",
    [
        # prose: words never leave a gap in the same place on every line
        [
            "Storage costs were reviewed for every team over the quarter, and",
            "the figures below summarize what each tier charged during the",
            "period under review. Each figure is rounded to one decimal place",
            "and the totals include the archive tier for older records.",
        ],
        # a list whose items are short
        [f"- item number {i} with a short note" for i in range(6)],
    ],
    ids=["prose", "list"],
)
def test_text_that_is_not_a_table_is_left_alone(lines: list[str]) -> None:
    assert find_unruled_tables(_draw_lines(lines)[0]) == []


def test_long_cells_are_prose_in_columns_not_a_table() -> None:
    sentence = "a sentence long enough to be prose in a narrow column"
    doc = _draw_rows([[sentence] * 3] * 4, [36, 210, 384])

    assert find_unruled_tables(doc[0]) == []


def test_aligned_text_in_a_monospaced_font_is_left_alone() -> None:
    doc = _draw_rows(LANGUAGES, [36, 110, 150, 205], font="cour")

    assert find_unruled_tables(doc[0]) == []


def test_a_column_of_punctuation_is_grammar_not_a_table() -> None:
    rows = [["OWS", "=", "SP-or-HTAB"], ["RWS", "=", "required"], ["BWS", "=", "bad"]]

    assert find_unruled_tables(_draw_rows(rows, [36, 100, 130])[0]) == []


def test_only_a_paragraph_with_exactly_the_tables_words_is_replaced() -> None:
    flat = " ".join(cell for row in LANGUAGES for cell in row)
    markdown = f"# Title\n\nIntro text.\n\n{flat}\n\nOutro."

    result, count = replace_flattened_tables(markdown, [LANGUAGES])

    assert count == 1
    assert result.startswith("# Title\n\nIntro text.\n\n|Language|Year|")
    assert result.endswith("|TypeScript|2012|Static|Web development|\n\nOutro.")


@pytest.mark.parametrize(
    "paragraph",
    [
        "Language Year Typing Primary Use Python 1991",  # only part of it
        "|Language|Year|Typing|Primary Use|\n|---|---|---|---|",  # already a table
        "**Language** Year Typing Primary Use",  # styled differently
    ],
)
def test_other_paragraphs_are_never_touched(paragraph: str) -> None:
    result, count = replace_flattened_tables(f"Intro.\n\n{paragraph}", [LANGUAGES])

    assert count == 0
    assert result == f"Intro.\n\n{paragraph}"


def test_a_paragraph_with_a_pipe_is_left_alone() -> None:
    """Pipe-delimited plain text stays as it was written."""
    table = [["Op", "Meaning", "Arity"], ["a|b", "or", "2"], ["!a", "not", "1"]]
    flat = " ".join(cell for row in table for cell in row)

    assert replace_flattened_tables(flat, [table]) == (flat, 0)


def test_a_broken_file_leaves_the_text_as_it_was(tmp_path: Path) -> None:
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"%PDF-1.7 not really")

    assert restore_flattened_tables(broken, [1], ["text"]) == ["text"]
