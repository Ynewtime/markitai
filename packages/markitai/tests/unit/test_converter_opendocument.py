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
    'xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0"'
)


def _write_odf(path: Path, body: str) -> Path:
    """Wrap a document body in the minimum content.xml an ODF file needs."""
    content = (
        f'<?xml version="1.0" encoding="UTF-8"?>'
        f"<office:document-content {_NS}><office:body>{body}"
        f"</office:body></office:document-content>"
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("content.xml", content)
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
