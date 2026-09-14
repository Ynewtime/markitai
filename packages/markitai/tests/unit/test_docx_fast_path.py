"""Known DOCX fast path must retain styled text and the OMML fallback."""

import zipfile
from unittest.mock import patch

import pytest

from markitai.converter.office import DocxConverter, OfficeConverter


def _reference_convert(path):
    from mammoth.docx import office_xml, xmlparser

    # A previous conversion may have installed Markitai's adapter. Keep this
    # oracle independent of it, including when the full suite changes order.
    with patch.object(office_xml, "parse_xml", xmlparser.parse_xml):
        return OfficeConverter().convert(path)


@pytest.mark.parametrize("encoding", ["utf-8", "utf-16"])
@pytest.mark.parametrize("math", [False, True])
def test_docx_matches_reference_with_and_without_equations(
    fixtures_dir, tmp_path, encoding, math
):
    path = tmp_path / "document.docx"
    with (
        zipfile.ZipFile(fixtures_dir / "sample.docx") as original,
        zipfile.ZipFile(path, "w") as target,
    ):
        for name in original.namelist():
            data = original.read(name)
            if name == "word/document.xml":
                text = data.decode("utf-8")
                if math:
                    text = text.replace(
                        "</w:body>",
                        '<w:p><m:oMath xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"><m:r><m:t>x</m:t></m:r></m:oMath></w:p></w:body>',
                    )
                text = text.replace('encoding="UTF-8"', f'encoding="{encoding}"')
                data = text.encode(encoding)
            target.writestr(name, data)
    expected = _reference_convert(path)
    actual = DocxConverter().convert(path)
    assert actual.markdown == expected.markdown
    assert actual.images == expected.images
    if math:
        from markitai.converter.docx import convert_docx_without_math

        assert convert_docx_without_math(path) is None
    # The reference preprocessor currently fails to decode UTF-16 OMML.
    # Require parity and explicit fallback there; do not claim it preserves
    # equations that the established path itself loses.
    if math and encoding == "utf-8":
        assert "$x$" in actual.markdown


def test_non_docx_input_defers_to_existing_format_detection(tmp_path):
    from markitai.converter.docx import convert_docx_without_math

    path = tmp_path / "broken.docx"
    path.write_bytes(b"not a zip file")
    assert convert_docx_without_math(path) is None
    assert (
        DocxConverter().convert(path).markdown
        == OfficeConverter().convert(path).markdown
    )


def test_docx_embedded_image_is_preserved(fixtures_dir, tmp_path):
    import base64

    image = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aK1sAAAAASUVORK5CYII="
    )
    path = tmp_path / "image.docx"
    with (
        zipfile.ZipFile(fixtures_dir / "sample.docx") as original,
        zipfile.ZipFile(path, "w") as target,
    ):
        for name in original.namelist():
            data = original.read(name)
            if name == "word/document.xml":
                data = data.replace(
                    b"</w:body>",
                    b'<w:p><w:r><w:pict><v:shape xmlns:v="urn:schemas-microsoft-com:vml"><v:imagedata xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" r:id="rIdImage"/></v:shape></w:pict></w:r></w:p></w:body>',
                )
            elif name == "word/_rels/document.xml.rels":
                data = data.replace(
                    b"</Relationships>",
                    b'<Relationship Id="rIdImage" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/pixel.png"/></Relationships>',
                )
            elif name == "[Content_Types].xml":
                data = data.replace(
                    b"</Types>",
                    b'<Default Extension="png" ContentType="image/png"/></Types>',
                )
            target.writestr(name, data)
        target.writestr("word/media/pixel.png", image)
    expected = _reference_convert(path)
    actual = DocxConverter().convert(path)
    assert "data:image/png;base64," in actual.markdown
    assert actual.markdown == expected.markdown


def test_docx_notes_and_links_match_the_original_xml_parser(fixtures_dir, tmp_path):
    path = tmp_path / "notes.docx"
    word_ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    relationship_ns = (
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    )
    with (
        zipfile.ZipFile(fixtures_dir / "sample.docx") as source,
        zipfile.ZipFile(path, "w") as target,
    ):
        for name in source.namelist():
            data = source.read(name)
            if name == "word/document.xml":
                data = data.replace(
                    b"</w:body>",
                    (
                        "<w:p><w:r><w:t>A note reference</w:t>"
                        '<w:footnoteReference w:id="1"/></w:r>'
                        f'<w:hyperlink xmlns:r="{relationship_ns}" r:id="extraLink">'
                        "<w:r><w:t>Source link</w:t></w:r></w:hyperlink></w:p></w:body>"
                    ).encode(),
                )
            elif name == "word/_rels/document.xml.rels":
                data = data.replace(
                    b"</Relationships>",
                    (
                        f'<Relationship Id="extraLink" Type="{relationship_ns}/hyperlink" '
                        'Target="https://example.com/source" TargetMode="External"/>'
                        f'<Relationship Id="footnotes" Type="{relationship_ns}/footnotes" '
                        'Target="footnotes.xml"/></Relationships>'
                    ).encode(),
                )
            elif name == "[Content_Types].xml":
                data = data.replace(
                    b"</Types>",
                    b'<Override PartName="/word/footnotes.xml" '
                    b'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.footnotes+xml"/></Types>',
                )
            target.writestr(name, data)
        target.writestr(
            "word/footnotes.xml",
            f'<w:footnotes xmlns:w="{word_ns}"><w:footnote w:id="1">'
            "<w:p><w:r><w:rPr><w:i/></w:rPr><w:t>Footnote detail.</w:t></w:r></w:p>"
            "</w:footnote></w:footnotes>",
        )
    expected = _reference_convert(path)
    actual = DocxConverter().convert(path)
    assert actual.metadata["converter"] == "mammoth"
    assert "Footnote detail." in actual.markdown
    assert "https://example.com/source" in actual.markdown
    assert actual.markdown == expected.markdown
