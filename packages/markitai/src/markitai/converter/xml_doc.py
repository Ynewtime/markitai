"""Native XML converter (Python stdlib ``xml.etree.ElementTree``).

Element names become headings and their text becomes prose, which is what
makes an XML tree readable as Markdown. The stdlib parser cannot be told to
stop expanding entities, so documents carrying a DTD or entity declarations
are refused outright rather than parsed — that is the billion-laughs class of
input, and no sample document needs a DTD to be readable.
"""

from __future__ import annotations

import re
from pathlib import Path
from xml.etree import ElementTree  # noqa: S405 - DOCTYPE/ENTITY refused below

from loguru import logger

from markitai.converter.base import (
    BaseConverter,
    ConvertResult,
    FileFormat,
    conversion_failed,
    register_converter,
)

# Below this, the original document is kept verbatim in a fenced block so no
# structure (attribute order, mixed content, namespaces) is lost.
_MAX_INLINE_SOURCE_BYTES = 20 * 1024

_DECLARATION_RE = re.compile(r"<!\s*(DOCTYPE|ENTITY)\b", re.IGNORECASE)

_MAX_HEADING_LEVEL = 6


def _local(name: str) -> str:
    """Drop the ``{namespace}`` prefix ElementTree puts on qualified names."""
    return name.rsplit("}", 1)[-1] if name.startswith("{") else name


def _render_element(elem: ElementTree.Element, level: int, out: list[str]) -> None:
    """Render one element and its subtree, headings down to ``- `` lists."""
    name = _local(elem.tag)
    if level <= _MAX_HEADING_LEVEL:
        out.append(f"{'#' * level} {name}")
    else:
        indent = "  " * (level - _MAX_HEADING_LEVEL - 1)
        out.append(f"{indent}- **{name}**")

    for key, value in elem.attrib.items():
        out.append(f"{_local(key)}: {value}")

    text = (elem.text or "").strip()
    if text:
        out.append(text)

    for child in elem:
        _render_element(child, level + 1, out)
        tail = (child.tail or "").strip()
        if tail:
            out.append(tail)


@register_converter(FileFormat.XML)
class XmlConverter(BaseConverter):
    """Converter for XML documents."""

    supported_formats = [FileFormat.XML]

    def convert(
        self, input_path: Path, output_dir: Path | None = None
    ) -> ConvertResult:
        input_path = Path(input_path)
        logger.debug("[XmlConverter] Converting: {}", input_path.name)

        metadata: dict = {
            "source": str(input_path),
            "format": "XML",
            "converter": "xml",
        }

        text = input_path.read_text(encoding="utf-8", errors="replace")

        if _DECLARATION_RE.search(text):
            logger.warning("[XmlConverter] {}: DTD refused", input_path.name)
            conversion_failed(
                "XML with a DOCTYPE or ENTITY declaration is refused: the "
                "stdlib parser expands entities, which makes such a document "
                "an entity-expansion (billion laughs) risk",
            )

        try:
            root = ElementTree.fromstring(text)  # noqa: S314 - refused above
        except ElementTree.ParseError as exc:
            logger.warning("[XmlConverter] {}: {}", input_path.name, exc)
            conversion_failed(f"malformed XML: {exc}")

        parts: list[str] = []
        _render_element(root, 1, parts)
        markdown = "\n\n".join(parts)

        source = text.strip()
        if len(source.encode("utf-8")) < _MAX_INLINE_SOURCE_BYTES:
            markdown = f"{markdown}\n\n```xml\n{source}\n```"

        return ConvertResult(markdown=markdown, images=[], metadata=metadata)
