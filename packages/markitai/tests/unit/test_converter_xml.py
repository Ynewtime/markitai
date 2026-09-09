"""Tests for the native XML converter."""

from __future__ import annotations

from pathlib import Path

import pytest

from markitai.converter.xml_doc import XmlConverter
from markitai.utils.errors import ConversionError

FIXTURES = Path(__file__).parent.parent / "fixtures"


class TestXmlConverter:
    def test_fixture_becomes_headings_and_text(self) -> None:
        result = XmlConverter().convert(FIXTURES / "sample.xml")

        assert result.markdown.startswith("# note\n\n## to\n\nTove\n")
        assert "## body" in result.markdown
        assert result.metadata["format"] == "XML"

    def test_small_documents_keep_the_source(self) -> None:
        markdown = XmlConverter().convert(FIXTURES / "sample.xml").markdown

        assert "```xml" in markdown
        assert "<?xml version" in markdown.split("```xml", 1)[1]

    def test_large_documents_drop_the_source_block(self, tmp_path: Path) -> None:
        path = tmp_path / "big.xml"
        filler = "<item>" + "x" * 100 + "</item>"
        path.write_text(f"<root>{filler * 300}</root>", encoding="utf-8")

        assert "```xml" not in XmlConverter().convert(path).markdown

    def test_attributes_render_as_key_value_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "attrs.xml"
        path.write_text('<note id="7" lang="en">hi</note>', encoding="utf-8")

        markdown = XmlConverter().convert(path).markdown

        assert "id: 7" in markdown
        assert "lang: en" in markdown

    def test_namespaces_are_stripped_from_names(self, tmp_path: Path) -> None:
        path = tmp_path / "ns.xml"
        path.write_text(
            '<n:note xmlns:n="urn:x"><n:to>Tove</n:to></n:note>', encoding="utf-8"
        )

        markdown = XmlConverter().convert(path).markdown

        assert markdown.startswith("# note\n\n## to\n\nTove")

    def test_deep_nesting_falls_back_to_a_list(self, tmp_path: Path) -> None:
        path = tmp_path / "deep.xml"
        path.write_text(
            "<a><b><c><d><e><f><g>leaf</g></f></e></d></c></b></a>", encoding="utf-8"
        )

        markdown = XmlConverter().convert(path).markdown

        assert "###### f" in markdown
        assert "- **g**" in markdown

    def test_entity_declaration_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "lol.xml"
        path.write_text(
            '<?xml version="1.0"?>\n'
            '<!DOCTYPE lolz [<!ENTITY lol "lol">]>\n'
            "<lolz>&lol;</lolz>\n",
            encoding="utf-8",
        )

        with pytest.raises(ConversionError, match="DOCTYPE"):
            XmlConverter().convert(path)

    def test_malformed_xml_is_a_conversion_error(self, tmp_path: Path) -> None:
        path = tmp_path / "broken.xml"
        path.write_text("<a><b></a>", encoding="utf-8")

        with pytest.raises(ConversionError, match="malformed XML"):
            XmlConverter().convert(path)
