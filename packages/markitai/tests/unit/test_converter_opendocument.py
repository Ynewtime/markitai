"""Tests for the native OpenDocument (.odt/.ods) converters."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from markitai.converter.opendocument import OdsConverter, OdtConverter
from markitai.utils.errors import ConversionError

FIXTURES = Path(__file__).parent.parent / "fixtures"

_NS = (
    'xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
    'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" '
    'xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0" '
    'xmlns:dc="http://purl.org/dc/elements/1.1/" '
    'xmlns:xlink="http://www.w3.org/1999/xlink" '
    'xmlns:style="urn:oasis:names:tc:opendocument:xmlns:style:1.0" '
    'xmlns:fo="urn:oasis:names:tc:opendocument:xmlns:xsl-fo-compatible:1.0" '
    'xmlns:draw="urn:oasis:names:tc:opendocument:xmlns:drawing:1.0" '
    'xmlns:svg="urn:oasis:names:tc:opendocument:xmlns:svg-compatible:1.0"'
)


def _write_odf(
    path: Path,
    body: str,
    *,
    automatic_styles: str = "",
    styles: str = "",
    members: dict[str, bytes] | None = None,
) -> Path:
    """Wrap a document body in the minimum content.xml an ODF file needs.

    *automatic_styles* go into content.xml, *styles* into a styles.xml
    (its ``office:styles``), and *members* are further files of the zip.
    """
    content = (
        f'<?xml version="1.0" encoding="UTF-8"?>'
        f"<office:document-content {_NS}>"
        f"<office:automatic-styles>{automatic_styles}</office:automatic-styles>"
        f"<office:body>{body}</office:body></office:document-content>"
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("content.xml", content)
        if styles:
            archive.writestr(
                "styles.xml",
                f'<?xml version="1.0" encoding="UTF-8"?>'
                f"<office:document-styles {_NS}><office:styles>{styles}"
                f"</office:styles></office:document-styles>",
            )
        for name, data in (members or {}).items():
            archive.writestr(name, data)
    return path


class TestOdtConverter:
    def test_fixture_headings_and_lists(self) -> None:
        result = OdtConverter().convert(FIXTURES / "sample.odt")

        assert result.markdown.startswith("# These are a few of my favorite things:")
        assert "- Hockey" in result.markdown
        assert result.metadata["title"] == "These are a few of my favorite things:"
        assert result.metadata["format"] == "ODT"

    def test_outline_levels_and_nested_lists(self, tmp_path: Path) -> None:
        path = _write_odf(
            tmp_path / "doc.odt",
            "<office:text>"
            '<text:h text:outline-level="1">Top</text:h>'
            '<text:h text:outline-level="3">Deep</text:h>'
            "<text:p>Prose.</text:p>"
            "<text:list><text:list-item><text:p>outer</text:p>"
            "<text:list><text:list-item><text:p>inner</text:p>"
            "</text:list-item></text:list>"
            "</text:list-item></text:list>"
            "</office:text>",
        )

        markdown = OdtConverter().convert(path).markdown

        assert "# Top" in markdown
        assert "### Deep" in markdown
        assert "Prose." in markdown
        assert "- outer" in markdown
        assert "  - inner" in markdown

    def test_spans_tabs_and_breaks_are_flattened(self, tmp_path: Path) -> None:
        path = _write_odf(
            tmp_path / "spans.odt",
            "<office:text><text:p>a<text:span>b</text:span>"
            '<text:s text:c="3"/>c<text:tab/>d<text:line-break/>e</text:p>'
            "</office:text>",
        )

        assert OdtConverter().convert(path).markdown.strip() == "ab   c\td\ne"

    def test_tables_become_markdown_tables(self, tmp_path: Path) -> None:
        path = _write_odf(
            tmp_path / "table.odt",
            "<office:text><table:table>"
            "<table:table-row><table:table-cell><text:p>h1</text:p></table:table-cell>"
            "<table:table-cell><text:p>h2</text:p></table:table-cell></table:table-row>"
            "<table:table-row><table:table-cell><text:p>a</text:p></table:table-cell>"
            "<table:table-cell><text:p>b</text:p></table:table-cell></table:table-row>"
            "</table:table></office:text>",
        )

        markdown = OdtConverter().convert(path).markdown

        assert "| h1 | h2 |" in markdown
        assert "| --- | --- |" in markdown
        assert "| a | b |" in markdown

    def test_footnotes_links_comments_and_deletions(self, tmp_path: Path) -> None:
        """A footnote's body landed mid-sentence, a comment's author, date
        and text were inlined, a link lost its address, and text a reviewer
        deleted (kept in the tracked-changes record) came back."""
        path = _write_odf(
            tmp_path / "review.odt",
            "<office:text>"
            "<text:tracked-changes><text:changed-region text:id='c1'>"
            "<text:deletion><office:change-info><dc:creator>Bob</dc:creator>"
            "</office:change-info><text:p>DELETED TEXT</text:p></text:deletion>"
            "</text:changed-region></text:tracked-changes>"
            "<text:p>Body<text:note text:note-class='footnote'>"
            "<text:note-citation>1</text:note-citation><text:note-body>"
            "<text:p>The note.</text:p></text:note-body></text:note> and"
            "<office:annotation><dc:creator>Alice</dc:creator>"
            "<text:p>Review this</text:p></office:annotation> more."
            "<text:change text:change-id='c1'/> See "
            "<text:a xlink:href='https://example.com/a b'>the site</text:a>"
            " and <text:a xlink:href='#Intro'>the intro</text:a>.</text:p>"
            "<table:table><table:table-row><table:table-cell><text:p>cell"
            "<text:note text:note-class='endnote'><text:note-citation>i"
            "</text:note-citation><text:note-body><text:p>End note.</text:p>"
            "</text:note-body></text:note></text:p></table:table-cell>"
            "</table:table-row></table:table>"
            "</office:text>",
        )

        markdown = OdtConverter().convert(path).markdown

        assert "Body[^1] and more." in markdown
        assert "[the site](<https://example.com/a b>)" in markdown
        assert "and the intro." in markdown
        assert "| cell[^2] |" in markdown
        assert markdown.rstrip().endswith("[^1]: The note.\n[^2]: End note.")
        for gone in ("DELETED", "Bob", "Alice", "Review this"):
            assert gone not in markdown

    def test_bold_and_italic_spans_and_paragraphs(self, tmp_path: Path) -> None:
        """Every span was flattened: bold and italic text came out plain."""
        path = _write_odf(
            tmp_path / "styled.odt",
            "<office:text>"
            "<text:p>Plain <text:span text:style-name='T1'>bold words</text:span>"
            " and <text:span text:style-name='T2'>italic</text:span>, "
            "<text:span text:style-name='T3'>both</text:span> and "
            "<text:span text:style-name='Strong_20_Emphasis'>strong</text:span>"
            "<text:span text:style-name='T1'> joined</text:span>.</text:p>"
            "<text:p text:style-name='P1'>A whole bold paragraph</text:p>"
            "<text:p text:style-name='P1'>Bold with "
            "<text:span text:style-name='T2'>italic</text:span> inside</text:p>"
            "<text:h text:outline-level='1'>A <text:span text:style-name='T2'>"
            "styled</text:span> heading</text:h>"
            "</office:text>",
            automatic_styles=(
                "<style:style style:name='T1' style:family='text'>"
                "<style:text-properties fo:font-weight='bold'/></style:style>"
                "<style:style style:name='T2' style:family='text'>"
                "<style:text-properties fo:font-style='italic'/></style:style>"
                "<style:style style:name='T3' style:family='text'>"
                "<style:text-properties fo:font-weight='700' fo:font-style='italic'/>"
                "</style:style>"
                "<style:style style:name='P1' style:family='paragraph' "
                "style:parent-style-name='BoldBody'/>"
            ),
            styles=(
                "<style:style style:name='Strong_20_Emphasis' style:family='text'"
                " style:parent-style-name='Base'/>"
                "<style:style style:name='Base' style:family='text'>"
                "<style:text-properties fo:font-weight='bold'/></style:style>"
                "<style:style style:name='BoldBody' style:family='paragraph'>"
                "<style:text-properties fo:font-weight='bold'/></style:style>"
            ),
        )

        lines = OdtConverter().convert(path).markdown.rstrip("\n").split("\n\n")

        assert lines[0] == (
            "Plain **bold words** and *italic*, ***both*** and **strong joined**."
        )
        assert lines[1] == "**A whole bold paragraph**"
        assert lines[2] == "**Bold with *italic* inside**"
        assert lines[3] == "# A *styled* heading"

    def test_ordered_lists_are_numbered(self, tmp_path: Path) -> None:
        """Every list came out as bullets, whatever its list style said."""
        path = _write_odf(
            tmp_path / "lists.odt",
            "<office:text>"
            "<text:list text:style-name='L1'>"
            "<text:list-item><text:p>first</text:p></text:list-item>"
            "<text:list-item><text:p>second</text:p>"
            "<text:list><text:list-item><text:p>nested a</text:p></text:list-item>"
            "<text:list-item><text:p>nested b</text:p></text:list-item>"
            "</text:list></text:list-item>"
            "<text:list-item><text:p>third</text:p>"
            "<text:p>more of third</text:p></text:list-item>"
            "</text:list>"
            "<text:list text:style-name='Bullets'>"
            "<text:list-item><text:p>dot</text:p></text:list-item></text:list>"
            "<text:list text:style-name='L2'>"
            "<text:list-item><text:p>five</text:p></text:list-item>"
            "<text:list-item text:start-value='9'><text:p>nine</text:p>"
            "</text:list-item></text:list>"
            "<text:list text:style-name='L1'><text:list-item>"
            "<text:h text:outline-level='2'>Numbered heading</text:h>"
            "</text:list-item></text:list>"
            "</office:text>",
            automatic_styles=(
                "<text:list-style style:name='L1'>"
                "<text:list-level-style-number text:level='1' style:num-format='1'/>"
                "<text:list-level-style-bullet text:level='2' text:bullet-char='-'/>"
                "</text:list-style>"
                "<text:list-style style:name='L2'>"
                "<text:list-level-style-number text:level='1' style:num-format='1'"
                " text:start-value='5'/></text:list-style>"
            ),
            styles=(
                "<text:list-style style:name='Bullets'>"
                "<text:list-level-style-bullet text:level='1' text:bullet-char='*'/>"
                "</text:list-style>"
            ),
        )

        blocks = OdtConverter().convert(path).markdown.rstrip("\n").split("\n\n")

        assert blocks == [
            "1. first\n2. second\n   - nested a\n   - nested b\n"
            "3. third\n   more of third",
            "- dot",
            "5. five\n9. nine",
            "## Numbered heading",
        ]

    def test_pictures_become_embedded_images(self, tmp_path: Path) -> None:
        """A picture was dropped, with nothing left where it stood."""
        import base64

        png = (FIXTURES / "sample.jpg").read_bytes()
        path = _write_odf(
            tmp_path / "picture.odt",
            "<office:text><text:p>Before "
            "<draw:frame draw:name='img1'><svg:title>A [red] box</svg:title>"
            "<draw:image xlink:href='Pictures/red.jpg' draw:mime-type='image/jpeg'/>"
            "</draw:frame> after</text:p>"
            "<text:p><draw:frame><draw:image xlink:href='Pictures/missing.png'/>"
            "</draw:frame><draw:frame><draw:image "
            "xlink:href='https://example.com/remote.png'/></draw:frame>"
            "<draw:frame><draw:text-box><text:p>Box text</text:p></draw:text-box>"
            "</draw:frame></text:p>"
            "</office:text>",
            members={"Pictures/red.jpg": png},
        )

        result = OdtConverter().convert(path)
        markdown = result.markdown

        encoded = base64.b64encode(png).decode("ascii")
        assert f"Before ![A red box](data:image/jpeg;base64,{encoded}) after" in (
            markdown
        )
        assert "missing.png" not in markdown
        assert "remote.png" not in markdown
        assert "Box text" in markdown

    def test_unreadable_zip_is_a_conversion_error(self, tmp_path: Path) -> None:
        path = tmp_path / "broken.odt"
        path.write_bytes(b"not a zip at all")

        with pytest.raises(ConversionError, match="zip"):
            OdtConverter().convert(path)

    def test_missing_content_xml_is_a_conversion_error(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.odt"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("mimetype", "application/vnd.oasis.opendocument.text")

        with pytest.raises(ConversionError, match="content.xml"):
            OdtConverter().convert(path)

    def test_entity_declaration_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "lol.odt"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(
                "content.xml",
                '<!DOCTYPE lolz [<!ENTITY lol "lol">]><office:document-content/>',
            )

        with pytest.raises(ConversionError, match="DOCTYPE"):
            OdtConverter().convert(path)


class TestOdsConverter:
    def test_a_cell_comment_is_not_cell_text(self, tmp_path: Path) -> None:
        path = _write_odf(
            tmp_path / "comment.ods",
            '<office:spreadsheet><table:table table:name="S">'
            "<table:table-row><table:table-cell>"
            "<office:annotation><dc:creator>Alice</dc:creator>"
            "<text:p>check this</text:p></office:annotation>"
            "<text:p>42</text:p></table:table-cell></table:table-row>"
            "</table:table></office:spreadsheet>",
        )

        markdown = OdsConverter().convert(path).markdown

        assert "42" in markdown
        assert "Alice" not in markdown
        assert "check this" not in markdown

    def test_fixture_sheets_become_sections(self) -> None:
        result = OdsConverter().convert(FIXTURES / "sample.ods")

        assert "## Stanley Cups" in result.markdown
        assert "| Team | Location | Stanley Cups |" in result.markdown
        assert result.metadata["title"] == "Stanley Cups"
        assert result.metadata["format"] == "ODS"

    def test_repeated_columns_are_expanded_and_capped(self, tmp_path: Path) -> None:
        path = _write_odf(
            tmp_path / "repeat.ods",
            '<office:spreadsheet><table:table table:name="S">'
            "<table:table-row>"
            '<table:table-cell table:number-columns-repeated="3">'
            "<text:p>x</text:p></table:table-cell>"
            "<table:table-cell><text:p>y</text:p></table:table-cell>"
            "</table:table-row>"
            "<table:table-row>"
            '<table:table-cell table:number-columns-repeated="99999">'
            "<text:p>z</text:p></table:table-cell>"
            "</table:table-row>"
            "</table:table></office:spreadsheet>",
        )

        lines = OdsConverter().convert(path).markdown.splitlines()

        assert lines[2].startswith("| x | x | x | y |")
        assert lines[4].count("| z ") == 1000  # capped, not 99999

    def test_repeated_empty_cells_and_rows_keep_their_width(
        self, tmp_path: Path
    ) -> None:
        """LibreOffice stores `1, <empty x2>, 4` as one cell repeated twice.

        Collapsing that run to a single blank shifted `4` from column D to
        column C; collapsing a run of blank rows moved later rows up.
        """
        cell = "<table:table-cell><text:p>{}</text:p></table:table-cell>"
        path = _write_odf(
            tmp_path / "gaps.ods",
            '<office:spreadsheet><table:table table:name="S">'
            "<table:table-row>"
            + "".join(cell.format(h) for h in "ABCD")
            + "</table:table-row>"
            "<table:table-row>"
            + cell.format("1")
            + '<table:table-cell table:number-columns-repeated="2"/>'
            + cell.format("4")
            + "</table:table-row>"
            '<table:table-row table:number-rows-repeated="2">'
            "<table:table-cell/></table:table-row>"
            "<table:table-row>" + cell.format("5") + "</table:table-row>"
            "</table:table></office:spreadsheet>",
        )

        lines = OdsConverter().convert(path).markdown.splitlines()

        assert lines[2:] == [
            "| A | B | C | D |",
            "| --- | --- | --- | --- |",
            "| 1 |  |  | 4 |",
            "|  |  |  |  |",
            "|  |  |  |  |",
            "| 5 |  |  |  |",
        ]

    def test_empty_trailing_rows_and_columns_are_trimmed(self, tmp_path: Path) -> None:
        path = _write_odf(
            tmp_path / "trailing.ods",
            '<office:spreadsheet><table:table table:name="S">'
            "<table:table-row><table:table-cell><text:p>a</text:p></table:table-cell>"
            '<table:table-cell table:number-columns-repeated="500"/>'
            "</table:table-row>"
            '<table:table-row table:number-rows-repeated="100000">'
            "<table:table-cell/></table:table-row>"
            "</table:table></office:spreadsheet>",
        )

        markdown = OdsConverter().convert(path).markdown

        assert markdown.splitlines()[2:] == ["| a |", "| --- |"]

    def test_sheet_without_content_is_skipped(self, tmp_path: Path) -> None:
        path = _write_odf(
            tmp_path / "blank.ods",
            '<office:spreadsheet><table:table table:name="Empty">'
            "<table:table-row><table:table-cell/></table:table-row>"
            "</table:table></office:spreadsheet>",
        )

        assert OdsConverter().convert(path).markdown == ""

    def test_unreadable_zip_is_a_conversion_error(self, tmp_path: Path) -> None:
        path = tmp_path / "broken.ods"
        path.write_bytes(b"PK\x03\x04 truncated")

        with pytest.raises(ConversionError, match="zip"):
            OdsConverter().convert(path)
