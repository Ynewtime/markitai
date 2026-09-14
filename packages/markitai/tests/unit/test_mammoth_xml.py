"""Direct XML construction must preserve Mammoth's existing document model."""

import io
import xml.dom.minidom
import zipfile
from concurrent.futures import ThreadPoolExecutor
from xml.parsers.expat import ExpatError

import pytest
from mammoth.docx.xmlparser import parse_xml as reference_parse_xml

XML_CASES = [
    "<root><text>A &amp; B &lt; C &#65; &#x1F680;</text></root>",
    "<root>before<!-- comment -->after<?instruction value?>tail</root>",
    "<root>before<![CDATA[legacy CDATA node]]>after</root>",
    '<root><empty/><text xml:space="preserve">  中文\n body  </text></root>',
    '<root xmlns="urn:known" xmlns:a="urn:known" xmlns:b="urn:other" '
    'plain="x" a:key="first" b:key="second"><a:child/><b:child/></root>',
    '<root xmlns="urn:known"><child xmlns=""><nested xmlns="urn:other"/></child></root>',
    '<!DOCTYPE root [<!ENTITY text "expanded text">]><root>&text;</root>',
    '<!DOCTYPE root [<!ENTITY external SYSTEM "file:///does-not-exist">]>'
    "<root>before&external;after</root>",
    '<!DOCTYPE root [<!ATTLIST root value CDATA "default">]><root/>',
    "<root>" + "text &amp; 中文 " * 10000 + "</root>",
]


class TinyReads(io.BytesIO):
    def read(self, size=-1):
        return super().read(min(size, 7) if size >= 0 else 7)


@pytest.mark.parametrize("xml", XML_CASES, ids=range(len(XML_CASES)))
@pytest.mark.parametrize("encoding", ["utf-8", "utf-16"])
@pytest.mark.parametrize("tiny_reads", [False, True])
def test_xml_model_matches_reference(xml, encoding, tiny_reads):
    from markitai.converter.mammoth_xml import parse_mammoth_xml

    raw = (f'<?xml version="1.0" encoding="{encoding}"?>' + xml).encode(encoding)
    mapping = [("w", "urn:known"), ("x", "urn:other")]
    expected = reference_parse_xml(io.BytesIO(raw), mapping)
    stream = TinyReads(raw) if tiny_reads else io.BytesIO(raw)
    assert parse_mammoth_xml(stream, mapping) == expected


@pytest.mark.parametrize("xml", [b"", b"<root>", b"<a/><b/>", b"<a>&undefined;</a>"])
def test_malformed_xml_keeps_parser_errors(xml):
    from markitai.converter.mammoth_xml import parse_mammoth_xml

    for parser in (reference_parse_xml, parse_mammoth_xml):
        with pytest.raises(ExpatError):
            parser(io.BytesIO(xml))


def test_text_stream_and_filename_inputs(tmp_path):
    from markitai.converter.mammoth_xml import parse_mammoth_xml

    xml = '<root xml:space="preserve">中文 text</root>'
    assert parse_mammoth_xml(io.StringIO(xml)) == reference_parse_xml(io.StringIO(xml))
    path = tmp_path / "document.xml"
    path.write_text(xml)
    assert parse_mammoth_xml(str(path)) == reference_parse_xml(str(path))


def test_docx_conversion_avoids_intermediate_minidom_tree(fixtures_dir, monkeypatch):
    from markitai.converter.docx import convert_docx_without_math

    def unwanted_tree(*args, **kwargs):
        raise AssertionError("DOCX created a second intermediate XML document tree")

    monkeypatch.setattr(xml.dom.minidom, "parse", unwanted_tree)
    result = convert_docx_without_math(fixtures_dir / "sample.docx")
    assert result is not None
    assert "Markitai Snapshot Fixture" in result.markdown
    assert "**bold**" in result.markdown and "*italic*" in result.markdown


def test_installation_preserves_an_explicit_custom_parser(monkeypatch):
    from mammoth.docx import office_xml

    from markitai.converter.mammoth_xml import install_mammoth_xml_parser

    def custom_parser(stream, namespace_mapping=None):
        return reference_parse_xml(stream, namespace_mapping)

    monkeypatch.setattr(office_xml, "parse_xml", custom_parser)
    install_mammoth_xml_parser()
    assert office_xml.parse_xml is custom_parser


def test_every_docx_xml_part_matches_the_reference(fixtures_dir):
    from mammoth.docx.office_xml import _namespaces

    from markitai.converter.mammoth_xml import parse_mammoth_xml

    with zipfile.ZipFile(fixtures_dir / "sample.docx") as archive:
        for member in archive.namelist():
            if member.endswith((".xml", ".rels")):
                raw = archive.read(member)
                assert parse_mammoth_xml(io.BytesIO(raw), _namespaces) == (
                    reference_parse_xml(io.BytesIO(raw), _namespaces)
                ), member


def test_concurrent_parses_do_not_share_namespace_or_text_state():
    from markitai.converter.mammoth_xml import parse_mammoth_xml

    def read(index):
        raw = f'<root xmlns="urn:shared">Document {index}<child/></root>'.encode()
        mapping = [(f"ns{index}", "urn:shared")]
        return (
            parse_mammoth_xml(TinyReads(raw), mapping),
            reference_parse_xml(io.BytesIO(raw), mapping),
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        assert all(actual == expected for actual, expected in pool.map(read, range(40)))
