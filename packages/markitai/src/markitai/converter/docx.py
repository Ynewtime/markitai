"""Convert equation-free DOCX directly, retaining Mammoth for rich content."""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

from markitai.converter.base import ConvertResult

_MATH_PARTS = ("word/document.xml", "word/footnotes.xml", "word/endnotes.xml")


def convert_docx_without_math(path: Path) -> ConvertResult | None:
    """Return None when the established OMML preprocessing path is needed.

    Parse XML rather than scanning UTF-8 bytes: Word also accepts UTF-16 and
    arbitrary namespace prefixes. Unrecognized/malformed structures defer to
    the existing converter and its error handling.
    """
    from defusedxml import ElementTree

    html = None
    # Preserve installed readers, XML adapters and runtime Mammoth customization.
    # Cold conversions of plain files do not need to import Mammoth at all.
    allow_plain = "mammoth" not in sys.modules
    try:
        with zipfile.ZipFile(path) as archive:
            document = document_bytes = None
            names = set(archive.namelist())
            for name in _MATH_PARTS:
                if name not in names:
                    continue
                data = archive.read(name)
                root = ElementTree.fromstring(data)
                if any(
                    el.tag.rsplit("}", 1)[-1] in {"oMath", "oMathPara"}
                    for el in root.iter()
                ):
                    return None
                if name == "word/document.xml":
                    document, document_bytes = root, data
            if allow_plain and document is not None and document_bytes is not None:
                from markitai.converter.docx_plain import plain_docx_html

                html = plain_docx_html(archive, document, document_bytes)
    except (zipfile.BadZipFile, ElementTree.ParseError, ValueError):
        return None

    from markitai.converter.structured_text import _normalize
    from markitai.webextract.markdownify_compat import (
        CompatibleMarkdownConverter,
        parse_markdown_html,
    )

    engine = "native-docx"
    if html is None:
        import mammoth

        from markitai.converter.mammoth_xml import install_mammoth_xml_parser

        engine = "mammoth"
        install_mammoth_xml_parser()
        with path.open("rb") as stream:
            html = mammoth.convert_to_html(stream).value
    soup = parse_markdown_html(html, "html.parser")
    for node in soup(["script", "style"]):
        node.extract()
    markdown = (
        CompatibleMarkdownConverter(keep_data_uris=True)
        .convert_soup(soup.body or soup)
        .strip()
    )
    return ConvertResult(
        markdown=_normalize(markdown),
        metadata={"source": str(path), "format": "DOCX", "converter": engine},
    )
