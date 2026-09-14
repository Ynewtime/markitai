"""Plain DOCX fast path must preserve Mammoth output and defer rich packages."""

from pathlib import Path
from zipfile import ZipFile

import mammoth
import pytest

from markitai.converter import docx_plain as projection
from markitai.converter.docx import convert_docx_without_math

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures/sample.docx"


def native_html(path):
    with ZipFile(path) as archive:
        data = archive.read("word/document.xml")
        return projection.plain_docx_html(archive, projection.ET.fromstring(data), data)


def document(tmp_path, body=None, style_transform=lambda text: text):
    path = tmp_path / "input.docx"
    with ZipFile(FIXTURE) as source, ZipFile(path, "w") as target:
        for name in source.namelist():
            value = source.read(name)
            if name == "word/document.xml" and body is not None:
                value = (
                    '<w:document xmlns:w="'
                    + projection.W[1:-1]
                    + '"><w:body>'
                    + body
                    + "</w:body></w:document>"
                ).encode()
            if name == "word/styles.xml":
                value = style_transform(value.decode()).encode()
            target.writestr(name, value)
    return path


def reference(path):
    with path.open("rb") as stream:
        return mammoth.convert_to_html(stream).value


def test_heading_id_precedes_conflicting_style_name(tmp_path):
    path = document(
        tmp_path,
        '<w:p><w:pPr><w:pStyle w:val="Heading2"/></w:pPr><w:r><w:t>Heading</w:t></w:r></w:p>',
        lambda text: text.replace('styleId="Heading1"', 'styleId="Heading2"'),
    )
    assert native_html(path) == reference(path)


def test_footnote_named_style_precedes_numbering(tmp_path):
    path = document(
        tmp_path,
        style_transform=lambda text: text.replace(
            'w:val="List Paragraph"', 'w:val="footnote text"'
        ),
    )
    assert native_html(path) == reference(path)


def test_cdata_must_defer_to_reference_parser(tmp_path):
    path = document(
        tmp_path,
        "<w:p><w:r><w:t><![CDATA[This reader discards CDATA]]></w:t></w:r></w:p>",
    )
    assert native_html(path) is None


@pytest.mark.parametrize(
    "unsupported",
    [
        "<w:p><w:r><w:drawing/></w:r></w:p>",
        "<w:p><w:hyperlink><w:r><w:t>link</w:t></w:r></w:hyperlink></w:p>",
        '<w:p><w:r><w:rPr><w:vertAlign w:val="superscript"/></w:rPr><w:t>x</w:t></w:r></w:p>',
        '<w:p><w:bookmarkStart w:id="1" w:name="anchor"/></w:p>',
        '<w:tbl><w:tr><w:tc><w:tcPr><w:gridSpan w:val="2"/></w:tcPr></w:tc></w:tr></w:tbl>',
    ],
)
def test_unsupported_content_defers(tmp_path, unsupported):
    assert native_html(document(tmp_path, unsupported)) is None


def test_generated_runs_lists_tables_and_empty_paragraphs(tmp_path):
    import random

    rng = random.Random(20260930)
    texts = ["alpha", "中文", " ", "", " &amp; &lt; &gt; ", "\n", "| * _ [] \\", "é Ω"]
    for _ in range(200):
        blocks = []
        for _ in range(rng.randrange(1, 8)):
            props = rng.choice(
                [
                    "",
                    '<w:pPr><w:pStyle w:val="Heading1"/></w:pPr>',
                    '<w:pPr><w:numPr><w:ilvl w:val="0"/><w:numId w:val="1"/></w:numPr></w:pPr>',
                ]
            )
            runs = []
            for _ in range(rng.randrange(0, 8)):
                run_props = rng.choice(
                    [
                        "",
                        "<w:rPr><w:b/></w:rPr>",
                        "<w:rPr><w:i/></w:rPr>",
                        "<w:rPr><w:b/><w:i/></w:rPr>",
                        '<w:rPr><w:b w:val="false"/></w:rPr>',
                    ]
                )
                runs.append(
                    "<w:r>" + run_props + "<w:t>" + rng.choice(texts) + "</w:t></w:r>"
                )
            paragraph = "<w:p>" + props + "".join(runs) + "</w:p>"
            if rng.random() < 0.2:
                paragraph = (
                    "<w:tbl><w:tr><w:tc>"
                    + paragraph
                    + "</w:tc><w:tc><w:p/></w:tc></w:tr></w:tbl>"
                )
            blocks.append(paragraph)
        path = document(tmp_path, "".join(blocks))
        assert native_html(path) == reference(path)


@pytest.mark.parametrize("style_name", ["HEADING 1", "hEaDiNg 1"])
def test_style_names_match_without_case(tmp_path, style_name):
    path = document(
        tmp_path,
        '<w:p><w:pPr><w:pStyle w:val="Custom"/></w:pPr><w:r><w:t>Heading</w:t></w:r></w:p>',
        lambda text: text.replace('styleId="Heading1"', 'styleId="Custom"').replace(
            'w:val="heading 1"', 'w:val="' + style_name + '"'
        ),
    )
    assert native_html(path) == reference(path)


def test_common_word_writer_and_style_derived_lists():
    path = FIXTURE.with_name("docx_plain_writer.docx")
    assert native_html(path) == reference(path)


def test_mammoth_boolean_spelling_is_preserved(tmp_path):
    path = document(
        tmp_path,
        '<w:p><w:r><w:rPr><w:b w:val="off"/></w:rPr><w:t>text</w:t></w:r></w:p>',
    )
    assert native_html(path) == reference(path)


@pytest.mark.parametrize("num_id", ["0", "1"])
@pytest.mark.parametrize("implicit_level", [False, True])
def test_numbering_zero_id_and_missing_level(tmp_path, num_id, implicit_level):
    path = document(
        tmp_path,
        "<w:p><w:pPr><w:numPr>"
        + ("" if implicit_level else '<w:ilvl w:val="0"/>')
        + '<w:numId w:val="'
        + num_id
        + '"/></w:numPr></w:pPr><w:r><w:t>item</w:t></w:r></w:p>',
    )
    changed = tmp_path / "numbering.docx"
    with ZipFile(path) as source, ZipFile(changed, "w") as target:
        for name in source.namelist():
            value = source.read(name)
            if name == "word/numbering.xml":
                value = value.replace(
                    b'w:numId="1"', ('w:numId="' + num_id + '"').encode()
                )
            target.writestr(name, value)
    assert native_html(changed) == reference(changed)


@pytest.mark.parametrize("implicit_last", [False, True])
def test_explicit_zero_level_wins_over_unindexed_level(tmp_path, implicit_last):
    path = document(
        tmp_path,
        '<w:p><w:pPr><w:numPr><w:numId w:val="1"/></w:numPr></w:pPr><w:r><w:t>item</w:t></w:r></w:p>',
    )
    changed = tmp_path / "numbering.docx"
    levels = [
        '<w:lvl w:ilvl="0"><w:numFmt w:val="bullet"/></w:lvl>',
        '<w:lvl><w:numFmt w:val="decimal"/></w:lvl>',
    ]
    if not implicit_last:
        levels.reverse()
    with ZipFile(path) as source, ZipFile(changed, "w") as target:
        for name in source.namelist():
            value = source.read(name)
            if name == "word/numbering.xml":
                value = (
                    '<w:numbering xmlns:w="'
                    + projection.W[1:-1]
                    + '"><w:abstractNum w:abstractNumId="0">'
                    + "".join(levels)
                    + '</w:abstractNum><w:num w:numId="1"><w:abstractNumId w:val="0"/></w:num></w:numbering>'
                ).encode()
            target.writestr(name, value)
    assert native_html(changed) == reference(changed)


@pytest.mark.parametrize("payload", [b"not a zip", b"<w:document"])
def test_invalid_input_shared_adapter_preserves_reference_deferral(tmp_path, payload):
    path = tmp_path / "invalid.docx"
    if payload.startswith(b"<"):
        with ZipFile(path, "w") as archive:
            archive.writestr("word/document.xml", payload)
    else:
        path.write_bytes(payload)

    assert convert_docx_without_math(path) is None
    assert convert_docx_without_math(path) is None


def test_unsupported_document_does_not_parse_styles(tmp_path, monkeypatch):
    path = document(
        tmp_path,
        '<w:p><w:r><w:rPr><w:vertAlign w:val="superscript"/></w:rPr><w:t>x</w:t></w:r></w:p>',
    )
    parsed = []
    parse = projection.read_xml

    def record(data):
        parsed.append(data)
        return parse(data)

    monkeypatch.setattr(projection, "read_xml", record)
    assert native_html(path) is None
    assert not parsed


@pytest.mark.parametrize("fixture", ["sample.docx", "docx_plain_writer.docx"])
def test_cold_conversion_uses_plain_reader_and_one_document_read(fixture):
    import json
    import os
    import subprocess
    import sys

    path = FIXTURE.with_name(fixture)
    expected = convert_docx_without_math(path)
    assert expected is not None
    code = """
import json, sys, zipfile
from pathlib import Path
from markitai.converter.docx import convert_docx_without_math
reads = []
read = zipfile.ZipFile.read
def counted(self, name, *args, **kwargs):
    reads.append(name)
    return read(self, name, *args, **kwargs)
zipfile.ZipFile.read = counted
result = convert_docx_without_math(Path(sys.argv[1]))
assert result is not None
print(json.dumps({'markdown':result.markdown,'metadata':result.metadata,'images':len(result.images),'mammoth_loaded':'mammoth' in sys.modules,'document_reads':reads.count('word/document.xml')}))
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(path)],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "LITELLM_LOCAL_MODEL_COST_MAP": "True"},
    )
    actual = json.loads(result.stdout)
    assert actual["markdown"] == expected.markdown
    assert actual["metadata"] == {**expected.metadata, "converter": "native-docx"}
    assert actual["images"] == len(expected.images)
    assert actual["mammoth_loaded"] is False
    assert actual["document_reads"] == 1


def test_loaded_mammoth_customizations_are_preserved(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(
        mammoth,
        "convert_to_html",
        lambda *_args, **_kwargs: SimpleNamespace(value="<p>custom reader</p>"),
    )
    result = convert_docx_without_math(FIXTURE)
    assert result is not None
    assert result.markdown == "custom reader"
    assert result.metadata["converter"] == "mammoth"


def replace_part(tmp_path, name, value, source=FIXTURE):
    path = tmp_path / "changed.docx"
    with ZipFile(source) as original, ZipFile(path, "w") as target:
        for existing in original.namelist():
            if existing != name:
                target.writestr(existing, original.read(existing))
        target.writestr(name, value)
    return path


@pytest.mark.parametrize(
    "name,value",
    [
        (
            "[Content_Types].xml",
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default/></Types>',
        ),
        (
            "word/_rels/document.xml.rels",
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Type="a" Target="b"/></Relationships>',
        ),
        (
            "word/numbering.xml",
            '<w:numbering xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:num w:numId="1"/></w:numbering>',
        ),
        (
            "word/styles.xml",
            '<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:style/></w:styles>',
        ),
    ],
)
def test_malformed_auxiliary_parts_defer_to_reference(tmp_path, name, value):
    assert native_html(replace_part(tmp_path, name, value)) is None


def test_style_linked_numbering_defers_before_large_style_parse(tmp_path, monkeypatch):
    numbering = '<w:numbering xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:abstractNum w:abstractNumId="99"><w:numStyleLink w:val="custom"/></w:abstractNum></w:numbering>'
    path = replace_part(
        tmp_path,
        "word/numbering.xml",
        numbering,
        FIXTURE.with_name("docx_plain_writer.docx"),
    )
    parsed = []
    parse = projection.read_xml

    def record(data):
        parsed.append(data)
        return parse(data)

    monkeypatch.setattr(projection, "read_xml", record)
    assert native_html(path) is None
    with ZipFile(path) as archive:
        styles = archive.read("word/styles.xml")
    assert styles not in parsed


@pytest.mark.parametrize(
    "kind,target",
    [
        ("officeDocument", "other/document.xml"),
        ("styles", "other-styles.xml"),
        ("numbering", "other-numbering.xml"),
        ("footnotes", "notes.xml"),
        ("endnotes", "ends.xml"),
        ("comments", "comment-part.xml"),
    ],
)
def test_alternate_part_relationships_defer(tmp_path, kind, target):
    part = "_rels/.rels" if kind == "officeDocument" else "word/_rels/document.xml.rels"
    value = (
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="custom" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/'
        + kind
        + '" Target="'
        + target
        + '"/></Relationships>'
    )
    assert native_html(replace_part(tmp_path, part, value)) is None


def test_empty_numbering_level_is_not_coerced_to_zero(tmp_path):
    path = document(
        tmp_path,
        '<w:p><w:pPr><w:numPr><w:ilvl w:val=""/><w:numId w:val="1"/></w:numPr></w:pPr><w:r><w:t>item</w:t></w:r></w:p>',
    )
    numbering = '<w:numbering xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:abstractNum w:abstractNumId="0"><w:lvl w:ilvl=""><w:numFmt w:val="bullet"/></w:lvl></w:abstractNum><w:num w:numId="1"><w:abstractNumId w:val="0"/></w:num></w:numbering>'
    changed = tmp_path / "empty-level.docx"
    with ZipFile(path) as original, ZipFile(changed, "w") as target:
        for name in original.namelist():
            target.writestr(
                name, numbering if name == "word/numbering.xml" else original.read(name)
            )
    result = native_html(changed)
    assert result is None or result == reference(changed)


@pytest.mark.parametrize(
    "run",
    [
        "<w:r><w:t>\"quotes\" and 'single'</w:t></w:r>",
        '<w:r><w:t>a</w:t><w:tab/><w:t>b</w:t><w:br/><w:t>c</w:t><w:br w:type="page"/></w:r>',
        '<w:r><w:rPr><w:i w:val="0"/><w:b w:val="FALSE"/></w:rPr><w:t>text</w:t></w:r>',
    ],
)
def test_run_literals_and_breaks_match_reference(tmp_path, run):
    path = document(tmp_path, "<w:p>" + run + "</w:p>")
    assert native_html(path) == reference(path)
