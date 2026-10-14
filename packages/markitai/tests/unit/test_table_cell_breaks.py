"""A line break inside a table cell is written as <br>, the row on one line."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from markitai.converter.base import get_converter
from markitai.webextract.table_cells import (
    CELL_BREAK,
    markdown_cell,
    parser_safe_breaks,
    text_cell,
)

# Every line break str.splitlines knows, plus the Windows pair
BREAKS = ["\n", "\r\n", "\r", "\u2028", "\u2029", "\v", "\x85", "\f"]


def _rows(markdown: str) -> list[str]:
    """The table lines of *markdown*, checked to be whole rows."""
    rows = [line for line in markdown.splitlines() if line.startswith("|")]
    for row in rows:
        assert row.endswith("|"), row
        assert not any(ch in row for ch in "\v\x85\u2028\u2029\r\f"), repr(row)
    return rows


@pytest.mark.parametrize("brk", BREAKS)
def test_text_cell_writes_each_break_as_br(brk):
    assert text_cell(f"Bring gloves{brk}and a trowel") == "Bring gloves<br>and a trowel"


def test_text_cell_collapses_runs_trims_edges_and_escapes():
    assert (
        text_cell("\n Kneel pads \n\n\u2028\v buckets\r\n") == "Kneel pads<br>buckets"
    )
    assert text_cell("Soil | compost") == "Soil \\| compost"
    assert text_cell("write <br> on tags") == "write &lt;br&gt; on tags"
    assert text_cell("write <BR/> on tags") == "write &lt;BR/&gt; on tags"
    assert text_cell("") == ""
    assert text_cell("\n\n") == ""


def test_parser_safe_breaks_writes_dropped_controls_as_line_separators():
    assert (
        parser_safe_breaks("a\vb\fc\x1cd\x1de\x1ef")
        == "a\u2028b\u2028c\u2028d\u2028e\u2028f"
    )
    plain = "<td>a\r\nb\x85c\u2028d\te</td>"
    assert parser_safe_breaks(plain) is plain


def test_markdown_cell_breaks_on_marks_and_separators():
    rendered = f"Bring gloves{CELL_BREAK}{CELL_BREAK} and\u2028a\vtrowel "
    assert markdown_cell(rendered) == "Bring gloves<br>and<br>a<br>trowel"
    # Markdownify's own newlines (a list in a cell) stay spaces
    assert markdown_cell("* rake\n* hoe") == "* rake * hoe"
    # A pipe Markdownify escaped is not escaped twice
    assert markdown_cell("a\\|b | c") == "a\\|b \\| c"


def test_spreadsheet_cells(tmp_path):
    from openpyxl import Workbook

    book = Workbook()
    sheet = book.active
    assert sheet is not None
    sheet.title = "Work plan"
    sheet.append(["Task", "Notes"])
    sheet.append(["Plant bulbs", "Bring gloves\nand a trowel"])
    sheet.append(["Water beds", "Fill cans\u2028at the shed"])
    sheet.append(["Weed paths", "\nKneel pads\n\n\nbuckets\n"])
    sheet.append(["Mulch", "Soil | compost"])
    path = tmp_path / "plan.xlsx"
    book.save(path)
    converter = get_converter(path)
    assert converter is not None

    rows = _rows(converter.convert(path).markdown)

    assert rows[2:] == [
        "| Plant bulbs | Bring gloves<br>and a trowel |",
        "| Water beds | Fill cans<br>at the shed |",
        "| Weed paths | Kneel pads<br>buckets |",
        "| Mulch | Soil \\| compost |",
    ]


def test_csv_cells(tmp_path):
    path = tmp_path / "plan.csv"
    path.write_text(
        'Task,Notes\nPlant bulbs,"Bring gloves\r\nand a trowel"\n'
        "Water beds,Fill cans\u2028at the shed\nWeed paths,Kneel pads\vbuckets\n",
        encoding="utf-8",
        newline="",
    )
    converter = get_converter(path)
    assert converter is not None

    assert _rows(converter.convert(path).markdown) == [
        "| Task | Notes |",
        "| --- | --- |",
        "| Plant bulbs | Bring gloves<br>and a trowel |",
        "| Water beds | Fill cans<br>at the shed |",
        "| Weed paths | Kneel pads<br>buckets |",
    ]


HTML_TABLE = (
    "<table><thead><tr><th>Task</th><th>Notes</th></tr></thead><tbody>"
    "<tr><td>Plant bulbs</td><td>Bring gloves<br>and a trowel</td></tr>"
    "<tr><td>Water beds</td><td><p>Fill cans</p><p>at the shed</p></td></tr>"
    "<tr><td>Weed paths</td><td>Kneel pads<br><br><br>buckets</td></tr>"
    "<tr><td>Rake leaves</td><td>Rake\u2028bag\vtwine</td></tr>"
    "<tr><td>Mulch</td><td>Soil | compost</td></tr>"
    "<tr><td>Label rows</td><td>write &lt;br&gt; on tags</td></tr>"
    "</tbody></table>"
)
HTML_ROWS = [
    "| Task | Notes |",
    "| --- | --- |",
    "| Plant bulbs | Bring gloves<br>and a trowel |",
    "| Water beds | Fill cans<br>at the shed |",
    "| Weed paths | Kneel pads<br>buckets |",
    "| Rake leaves | Rake<br>bag<br>twine |",
    "| Mulch | Soil \\| compost |",
    "| Label rows | write &lt;br&gt; on tags |",
]


def test_html_cells():
    from markitai.webextract.markdownify_compat import CompatibleMarkdownConverter

    assert _rows(CompatibleMarkdownConverter().convert(HTML_TABLE)) == HTML_ROWS


def _html_page(table: str) -> str:
    return (
        "<html><head><title>Garden club work plan</title></head><body>\n"
        "<h1>Garden club work plan</h1>\n<p>Our spring tasks for the community"
        " beds, agreed at the March meeting of the club.</p>\n"
        f"{table}\n<p>Thank you to every volunteer who signed up for a"
        " shift this season.</p>\n</body></html>\n"
    )


# A line break in the HTML source of a cell is whitespace, as a browser shows
# it: one space, never words run together, whatever the file's line endings
SWEEP_ROW = "<tr><td>Sweep</td><td>Sweep\npath\r\nedges</td></tr></tbody>"


@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
def test_html_file_cells(tmp_path, newline):
    """The file is written byte for byte (``newline=newline``); with CRLF
    line endings the cell's own CRLF becomes CR CR LF."""
    path = tmp_path / "plan.html"
    with path.open("w", encoding="utf-8", newline=newline) as handle:
        handle.write(_html_page(HTML_TABLE.replace("</tbody>", SWEEP_ROW)))
    converter = get_converter(path)
    assert converter is not None

    assert _rows(converter.convert(path).markdown) == [
        *HTML_ROWS,
        "| Sweep | Sweep path edges |",
    ]


def test_no_html_parser_sees_a_break_old_libxml2_deletes(tmp_path, monkeypatch):
    """libxml2 before 2.14 (lxml's Windows wheels) deletes a vertical tab, form
    feed or file/group/record separator from HTML text, which ran "bag\vtwine"
    together; every parse gets them as U+2028 instead, so no parser or
    libxml2 version decides the output."""
    import bs4

    import markitai.webextract.dom as dom
    import markitai.webextract.markdownify_compat as compat

    seen: list[str] = []

    class Recording(bs4.BeautifulSoup):
        def __init__(self, markup="", *args, **kwargs):
            if isinstance(markup, str):
                seen.append(markup)
            super().__init__(markup, *args, **kwargs)

    monkeypatch.setattr(dom, "BeautifulSoup", Recording)
    monkeypatch.setattr(compat, "BeautifulSoup", Recording)
    path = tmp_path / "plan.html"
    path.write_bytes(
        _html_page(HTML_TABLE.replace("\u2028", "\f\x1c\x1d\x1e")).encode()
    )
    converter = get_converter(path)
    assert converter is not None

    rows = _rows(converter.convert(path).markdown)
    assert "| Rake leaves | Rake<br>bag<br>twine |" in rows
    assert seen
    assert not any(ch in markup for markup in seen for ch in "\v\f\x1c\x1d\x1e")


def test_markitdown_renderer_cells(tmp_path):
    """MarkItDown's own converter (EPUB, DOCX with equations, fallbacks)."""
    from markitai.converter.markitdown_ext import _convert

    path = tmp_path / "plan.htm"
    path.write_text(f"<html><body>{HTML_TABLE}</body></html>", encoding="utf-8")

    assert _rows(_convert(path).markdown) == HTML_ROWS


def test_cold_table_renderer_matches_markdownify_on_breaks():
    from markitai.converter.html_table import html_table_markdown
    from markitai.webextract.markdownify_compat import CompatibleMarkdownConverter

    source = (
        "<table><thead><tr><th>Task</th></tr></thead><tbody>"
        "<tr><td>Bring gloves<br>and a trowel</td></tr>"
        "<tr><td> Kneel pads <br/><br> buckets\u2028| twine </td></tr>"
        "<tr><td>write &lt;br&gt; on tags</td></tr></tbody></table>"
    )
    actual = html_table_markdown(source)
    assert actual == CompatibleMarkdownConverter().convert(source).strip()
    assert _rows(actual)[2:] == [
        "| Bring gloves<br>and a trowel |",
        "| Kneel pads<br>buckets<br>\\| twine |",
        "| write &lt;br&gt; on tags |",
    ]


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def _docx(path: Path, cells: list[tuple[str, str]]) -> Path:
    """A DOCX of one two-column table; cell XML is given as w:p elements."""

    def cell(xml: str) -> str:
        return f"<w:tc>{xml}</w:tc>"

    rows = "".join(f"<w:tr>{cell(a)}{cell(b)}</w:tr>" for a, b in cells)
    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<w:document xmlns:w="{W_NS}"><w:body>'
        "<w:p><w:r><w:t>Garden club work plan</w:t></w:r></w:p>"
        f"<w:tbl>{rows}</w:tbl></w:body></w:document>"
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas'
            '.openxmlformats.org/package/2006/content-types"><Default Extension'
            '="rels" ContentType="application/vnd.openxmlformats-package.'
            'relationships+xml"/><Default Extension="xml" ContentType="application'
            '/xml"/><Override PartName="/word/document.xml" ContentType="'
            "application/vnd.openxmlformats-officedocument.wordprocessingml."
            'document.main+xml"/></Types>',
        )
        archive.writestr(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://'
            'schemas.openxmlformats.org/package/2006/relationships"><Relationship'
            ' Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/'
            '2006/relationships/officeDocument" Target="word/document.xml"/>'
            "</Relationships>",
        )
        archive.writestr("word/document.xml", document.encode("utf-8"))
    return path


def _p(*runs: str) -> str:
    return "<w:p>" + "".join(runs) + "</w:p>"


def _r(text: str) -> str:
    return f'<w:r><w:t xml:space="preserve">{text}</w:t></w:r>'


DOCX_CELLS = [
    (_p(_r("Task")), _p(_r("Notes"))),
    (_p(_r("Plant bulbs")), _p(_r("Bring gloves")) + _p(_r("and a trowel"))),
    (
        _p(_r("Water beds")),
        _p(_r("Fill cans"), "<w:r><w:br/></w:r>", _r("at the shed")),
    ),
    (_p(_r("Weed paths")), _p(_r("Kneel pads")) + _p() + _p() + _p(_r("buckets"))),
    (_p(_r("Rake leaves")), _p(_r("Rake\u2028bag"))),
    (_p(_r("Mulch")), _p(_r("Soil | compost"))),
]
DOCX_ROWS = [
    "| Plant bulbs | Bring gloves<br>and a trowel |",
    "| Water beds | Fill cans<br>at the shed |",
    "| Weed paths | Kneel pads<br>buckets |",
    "| Rake leaves | Rake<br>bag |",
    "| Mulch | Soil \\| compost |",
]


def test_docx_cells(tmp_path):
    from markitai.converter.docx import convert_docx_without_math

    result = convert_docx_without_math(_docx(tmp_path / "plan.docx", DOCX_CELLS))
    assert result is not None

    assert _rows(result.markdown)[3:] == DOCX_ROWS


def test_docx_cells_through_markitdown(tmp_path):
    from markitai.converter.markitdown_ext import _convert

    path = _docx(tmp_path / "plan.docx", DOCX_CELLS)

    assert _rows(_convert(path).markdown)[3:] == DOCX_ROWS


def test_pptx_cells(tmp_path):
    from pptx import Presentation
    from pptx.util import Inches

    deck = Presentation()
    slide = deck.slides.add_slide(deck.slide_layouts[5])
    table = slide.shapes.add_table(
        4, 2, Inches(1), Inches(2), Inches(6), Inches(2)
    ).table
    # "\n" starts a paragraph, "\v" is a soft line break (<a:br/>)
    values = [
        ("Task", "Notes"),
        ("Plant bulbs", "Bring gloves\nand a trowel"),
        ("Water beds", "Fill cans\vat the shed"),
        ("Rake leaves", "Rake\u2028bag | twine"),
    ]
    for row, (task, notes) in enumerate(values):
        table.cell(row, 0).text = task
        table.cell(row, 1).text = notes
    path = tmp_path / "plan.pptx"
    deck.save(path)
    converter = get_converter(path)
    assert converter is not None

    assert _rows(converter.convert(path).markdown)[2:] == [
        "| Plant bulbs | Bring gloves<br>and a trowel |",
        "| Water beds | Fill cans<br>at the shed |",
        "| Rake leaves | Rake<br>bag \\| twine |",
    ]


def test_opendocument_cell_paragraphs(tmp_path):
    ns = (
        'xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
        'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" '
        'xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0"'
    )

    def cell(*paragraphs: str) -> str:
        body = "".join(f"<text:p>{p}</text:p>" for p in paragraphs)
        return f"<table:table-cell>{body}</table:table-cell>"

    rows = [
        (cell("Task"), cell("Notes")),
        (cell("Plant bulbs"), cell("Bring gloves", "and a trowel")),
        (cell("Water beds"), cell("Fill cans<text:line-break/>at the shed")),
    ]
    table = "".join(f"<table:table-row>{a}{b}</table:table-row>" for a, b in rows)
    content = (
        f'<?xml version="1.0" encoding="UTF-8"?><office:document-content {ns}>'
        f'<office:body><office:text><table:table table:name="Plan">{table}'
        "</table:table></office:text></office:body></office:document-content>"
    )
    path = tmp_path / "plan.odt"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("mimetype", "application/vnd.oasis.opendocument.text")
        archive.writestr("content.xml", content)
    converter = get_converter(path)
    assert converter is not None

    assert _rows(converter.convert(path).markdown)[2:] == [
        "| Plant bulbs | Bring gloves<br>and a trowel |",
        "| Water beds | Fill cans<br>at the shed |",
    ]


# A model's answer (or any other writer) leaves a separator inside a row
MODEL_TABLE = (
    "| Task | Notes |\n"
    "| --- | --- |\n"
    "| Plant bulbs | Bring gloves and a trowel |\n"
    "| Weed paths |  Kneel pads \v\x85 buckets | \n"
    "| Water beds | Fill cans\rat the shed\n"
    "\n"
    "```\n"
    "| kept | as written |\n"
    "```\n"
    "\n"
    "A line outside a table stays as it is.\n"
)


def test_the_shared_cleanup_writes_breaks_left_in_a_row_as_br():
    from markitai.utils.markdown_quality import normalize_markdown

    cleaned = normalize_markdown(MODEL_TABLE)
    assert cleaned.splitlines()[2:5] == [
        "| Plant bulbs | Bring gloves<br>and a trowel |",
        "| Weed paths | Kneel pads<br>buckets |",
        "| Water beds | Fill cans<br>at the shed",
    ]
    assert "| kept | as written |" in cleaned
    assert "A line outside a table stays as it is." in cleaned


def test_an_llm_answer_keeps_its_table_rows_whole():
    """The .llm.md goes through the same cleanup as the base output."""
    from markitai.config import LLMConfig, PromptsConfig
    from markitai.llm import LLMProcessor

    processor = LLMProcessor(LLMConfig(), PromptsConfig())
    written = processor.format_llm_output(MODEL_TABLE, "title: Work plan")
    rows = [line for line in written.splitlines() if line.startswith("| ")]
    assert "| Plant bulbs | Bring gloves<br>and a trowel |" in rows
    assert "| Weed paths | Kneel pads<br>buckets |" in rows
