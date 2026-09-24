"""Markitdown-based converters for non-Office formats.

Registers markitdown converters for formats that markitdown supports natively
but markitai didn't previously handle (HTML, CSV, EPUB, MSG, IPYNB).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from markitai.converter._patches import apply_all_patches
from markitai.converter.base import (
    BaseConverter,
    ConvertResult,
    FileFormat,
    register_converter,
)
from markitai.notices import user_notice

if TYPE_CHECKING:
    from markitdown import MarkItDown

_markitdown: MarkItDown | None = None


def _convert_html(input_path: Path) -> ConvertResult:
    """Convert an HTML file using the webextract pipeline.

    Reads the HTML file and runs it through the full webextract noise
    removal + standardization pipeline.  Falls back to plain markitdown
    if webextract is unavailable or produces insufficient output.

    Args:
        input_path: Path to the HTML file.

    Returns:
        ConvertResult with markdown content and metadata.
    """
    input_path = Path(input_path)
    logger.debug("[HtmlConverter] Converting with webextract: {}", input_path.name)

    try:
        from markitai.webextract import (
            extract_web_content,
            is_native_extraction_acceptable,
        )

        html = input_path.read_text(encoding="utf-8", errors="replace")
        source_url = f"file://{input_path.resolve()}"
        extracted = extract_web_content(html, source_url)
        markdown = extracted.markdown

        if is_native_extraction_acceptable(extracted):
            metadata: dict = {
                "source": str(input_path),
                "format": input_path.suffix.lstrip(".").upper(),
                "converter": "webextract",
            }
            if extracted.metadata and extracted.metadata.title:
                metadata["title"] = extracted.metadata.title
            return ConvertResult(
                markdown=markdown,
                images=[],
                metadata=metadata,
            )

        logger.debug(
            "[HtmlConverter] webextract output too short, falling back to markitdown"
        )
    except Exception as exc:
        logger.debug(
            "[HtmlConverter] webextract failed, falling back to markitdown: {}", exc
        )

    return _convert(input_path)


def _convert(input_path: Path) -> ConvertResult:
    """Convert a file to Markdown using markitdown.

    Args:
        input_path: Path to the input file.

    Returns:
        ConvertResult with markdown content and metadata.
    """
    global _markitdown
    if _markitdown is None:
        from markitdown import MarkItDown

        # Only generic file conversion needs the Office compatibility patches.
        # Native HTML completes without loading any of those libraries.
        apply_all_patches()
        _markitdown = MarkItDown()

    input_path = Path(input_path)
    logger.debug("[Markitdown] Converting: {}", input_path.name)

    result = _markitdown.convert(input_path, keep_data_uris=True)

    metadata: dict = {
        "source": str(input_path),
        "format": input_path.suffix.lstrip(".").upper(),
        "converter": "markitdown",
    }
    if result.title:
        metadata["title"] = result.title

    return ConvertResult(
        markdown=result.markdown,
        images=[],
        metadata=metadata,
    )


@register_converter(FileFormat.HTML)
class HtmlConverter(BaseConverter):
    """Converter for HTML files using webextract pipeline + markitdown fallback."""

    supported_formats = [FileFormat.HTML, FileFormat.HTM, FileFormat.XHTML]

    def convert(
        self, input_path: Path, output_dir: Path | None = None
    ) -> ConvertResult:
        return _convert_html(input_path)


# Also register for .htm and .xhtml extensions
@register_converter(FileFormat.HTM)
class HtmConverter(HtmlConverter):
    """Converter for .htm files (delegates to HtmlConverter)."""

    supported_formats = [FileFormat.HTM]


@register_converter(FileFormat.XHTML)
class XhtmlConverter(HtmlConverter):
    """Converter for .xhtml files (delegates to HtmlConverter)."""

    supported_formats = [FileFormat.XHTML]


@register_converter(FileFormat.CSV)
class CsvConverter(BaseConverter):
    """Converter for known CSV input using the native CSV parser."""

    supported_formats = [FileFormat.CSV]

    def convert(
        self, input_path: Path, output_dir: Path | None = None
    ) -> ConvertResult:
        from markitai.converter.structured_text import convert_csv

        return convert_csv(Path(input_path))


@register_converter(FileFormat.EPUB)
class EpubConverter(BaseConverter):
    """Converter for EPUB e-book files using markitdown."""

    supported_formats = [FileFormat.EPUB]

    def convert(
        self, input_path: Path, output_dir: Path | None = None
    ) -> ConvertResult:
        return _convert(input_path)


# MAPI property streams of an Outlook .msg (OLE compound file). markitdown
# only reads the Unicode plain-text body; many messages carry the body only
# as 8-bit text or as HTML.
_MSG_BODY_UNICODE = "__substg1.0_1000001F"
_MSG_BODY_ANSI = "__substg1.0_1000001E"
_MSG_BODY_HTML = "__substg1.0_10130102"
_MSG_BODY_RTF = "__substg1.0_10090102"
_MSG_PROPERTIES = "__properties_version1.0"
# PR_INTERNET_CPID (HTML body charset) and PR_MESSAGE_CODEPAGE (8-bit text).
_PR_INTERNET_CPID = 0x3FDE0003
_PR_MESSAGE_CODEPAGE = 0x3FFD0003
# The top-level message's property stream has a 32-byte header, then fixed
# 16-byte entries: tag (4), flags (4), value (8).
_MSG_PROPERTIES_HEADER = 32
_MSG_PROPERTY_ENTRY = 16
_MSG_CODEPAGE_ALIASES = {
    20127: "ascii",
    28591: "latin-1",
    50220: "iso2022_jp",
    51932: "euc_jp",
    54936: "gb18030",
    65001: "utf-8",
}


def _msg_codepage(properties: bytes, tag: int) -> str | None:
    """Python codec for a code-page property, or None when absent/unknown."""
    import codecs

    for offset in range(
        _MSG_PROPERTIES_HEADER, len(properties) - _MSG_PROPERTY_ENTRY + 1, 16
    ):
        if int.from_bytes(properties[offset : offset + 4], "little") != tag:
            continue
        codepage = int.from_bytes(properties[offset + 8 : offset + 12], "little")
        name = _MSG_CODEPAGE_ALIASES.get(codepage, f"cp{codepage}")
        try:
            return codecs.lookup(name).name
        except LookupError:
            return None
    return None


def _decode_msg_bytes(data: bytes, codec: str | None) -> str:
    """Decode 8-bit body bytes: declared code page, then UTF-8, then cp1252."""
    for candidate in (codec, "utf-8"):
        if candidate is None:
            continue
        try:
            return data.decode(candidate)
        except (UnicodeDecodeError, LookupError):
            continue
    return data.decode("cp1252", errors="replace")


def _msg_fallback_body(input_path: Path) -> tuple[str, str | None]:
    """Recover a .msg body markitdown missed.

    Returns ``(markdown, reason)``: the body (8-bit plain text, else HTML
    converted to Markdown) and ``None``, or ``""`` and a user-facing reason
    when the message holds no body markitai can read.
    """
    import olefile

    from markitai.converter.eml import _html_to_markdown

    with olefile.OleFileIO(str(input_path)) as msg:

        def read(stream: str) -> bytes:
            return msg.openstream(stream).read() if msg.exists(stream) else b""

        if read(_MSG_BODY_UNICODE).decode("utf-16-le", errors="replace").strip():
            # markitdown already rendered it; nothing to recover.
            return "", None
        properties = read(_MSG_PROPERTIES)
        ansi = read(_MSG_BODY_ANSI).rstrip(b"\x00")
        if ansi.strip():
            codec = _msg_codepage(properties, _PR_MESSAGE_CODEPAGE)
            return _decode_msg_bytes(ansi, codec).strip(), None
        html_bytes = read(_MSG_BODY_HTML).rstrip(b"\x00")
        if html_bytes.strip():
            codec = _msg_codepage(properties, _PR_INTERNET_CPID)
            html = _decode_msg_bytes(html_bytes, codec)
            try:
                return _html_to_markdown(
                    html, input_path.resolve().as_uri()
                ).strip(), None
            except Exception as exc:
                logger.warning("[MsgConverter] HTML body conversion failed: {}", exc)
                return html.strip(), None
        if read(_MSG_BODY_RTF):
            return "", "its body is only stored as RTF, which markitai cannot read"
        return "", "it has no plain-text or HTML body"


@register_converter(FileFormat.MSG)
class MsgConverter(BaseConverter):
    """Converter for Outlook MSG email files.

    markitdown renders the headers and the Unicode plain-text body; a body
    stored only as 8-bit text or as HTML is recovered here, and a message
    whose body cannot be read says so instead of converting silently empty.
    """

    supported_formats = [FileFormat.MSG]

    def convert(
        self, input_path: Path, output_dir: Path | None = None
    ) -> ConvertResult:
        input_path = Path(input_path)
        result = _convert(input_path)
        try:
            body, reason = _msg_fallback_body(input_path)
        except Exception as exc:  # a damaged OLE stream must not lose the headers
            logger.debug("[MsgConverter] body fallback failed: {}", exc)
            body, reason = "", None
        if body:
            result.markdown = f"{result.markdown.rstrip()}\n\n{body}"
        elif reason:
            user_notice(
                "[MSG] No readable body in {}: {}; only the headers were converted",
                input_path.name,
                reason,
            )
        return result


@register_converter(FileFormat.IPYNB)
class IpynbConverter(BaseConverter):
    """Converter for Jupyter Notebook files using the native JSON parser."""

    supported_formats = [FileFormat.IPYNB]

    def convert(
        self, input_path: Path, output_dir: Path | None = None
    ) -> ConvertResult:
        from markitai.converter.structured_text import convert_notebook

        return convert_notebook(Path(input_path))


@register_converter(FileFormat.NUMBERS)
class NumbersConverter(BaseConverter):
    """Converter for Apple Numbers spreadsheet files using markitdown."""

    supported_formats = [FileFormat.NUMBERS]

    def convert(
        self, input_path: Path, output_dir: Path | None = None
    ) -> ConvertResult:
        return _convert(input_path)
