"""Build Mammoth's XML model directly, avoiding an intermediate minidom tree.

Mammoth remains responsible for Word styles, numbering, notes, images and HTML
conversion. Only its XML input adapter changes. State belongs to one parse;
installing the adapter is idempotent and preserves explicitly supplied parsers.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any
from xml.parsers import expat

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import BinaryIO, TextIO


def parse_mammoth_xml(
    fileobj: BinaryIO | TextIO | str,
    namespace_mapping: Sequence[tuple[str, str]] | None = None,
) -> Any:
    """Return the same XmlElement/XmlText model as Mammoth's minidom adapter."""
    from mammoth.docx.xmlparser import XmlElement, XmlText

    # Mammoth's cobble decorator generates these constructors at runtime.
    element_type: Any = XmlElement
    text_type: Any = XmlText
    if isinstance(fileobj, str):
        with open(fileobj, "rb") as stream:
            return parse_mammoth_xml(stream, namespace_mapping)

    prefixes = {uri: prefix for prefix, uri in namespace_mapping or ()}
    stack: list[Any] = []
    roots: list[Any] = []
    text: list[str] = []
    cdata = False

    @lru_cache(maxsize=256)
    def name(expanded: str) -> str:
        if " " not in expanded:
            return expanded
        uri, local = expanded.rsplit(" ", 1)
        prefix = prefixes.get(uri)
        return f"{{{uri}}}{local}" if prefix is None else f"{prefix}:{local}"

    def flush(*_args: str) -> None:
        if text:
            if stack:
                stack[-1].children.append(text_type("".join(text)))
            text.clear()

    def start(tag: str, attributes: dict[str, str]) -> None:
        flush()
        element = element_type(
            name(tag), {name(k): v for k, v in attributes.items()}, []
        )
        if stack:
            stack[-1].children.append(element)
        else:
            roots.append(element)
        stack.append(element)

    def end(_tag: str) -> None:
        flush()
        stack.pop()

    def characters(value: str) -> None:
        if not cdata:
            text.append(value)

    def start_cdata() -> None:
        nonlocal cdata
        flush()
        cdata = True

    def end_cdata() -> None:
        nonlocal cdata
        cdata = False

    parser = expat.ParserCreate(namespace_separator=" ")
    # Match minidom: retain specified attributes only. Namespace declarations
    # are excluded by namespace-aware Expat, as Mammoth's adapter excludes them.
    parser.specified_attributes = True
    parser.buffer_text = True
    parser.StartElementHandler = start
    parser.EndElementHandler = end
    parser.CharacterDataHandler = characters
    # Mammoth drops comment, PI and CDATA nodes but leaves adjacent TEXT_NODEs
    # distinct. Flush at those boundaries, while joining parser buffer chunks.
    parser.CommentHandler = flush
    parser.ProcessingInstructionHandler = flush
    parser.StartCdataSectionHandler = start_cdata
    parser.EndCdataSectionHandler = end_cdata
    # No external entity resolver is installed; external resources stay unread.
    parser.SetParamEntityParsing(expat.XML_PARAM_ENTITY_PARSING_NEVER)
    while chunk := fileobj.read(65536):
        parser.Parse(chunk, False)
    parser.Parse(b"", True)
    return roots[0]


def install_mammoth_xml_parser() -> None:
    """Replace only the default adapter, without changing its reference API."""
    from mammoth.docx import office_xml, xmlparser

    if office_xml.parse_xml is xmlparser.parse_xml:
        office_xml.parse_xml = parse_mammoth_xml
