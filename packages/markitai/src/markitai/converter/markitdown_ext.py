"""Markitdown-based converters for non-Office formats.

Registers markitdown converters for formats that markitdown supports natively
but markitai didn't previously handle (HTML, CSV, EPUB, MSG, IPYNB).
"""

from __future__ import annotations

import base64
import posixpath
import re
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from markitai.converter._patches import (
    apply_all_patches,
    apply_markitdown_cell_patches,
    import_markitdown,
)
from markitai.converter.base import (
    BaseConverter,
    ConvertResult,
    FileFormat,
    conversion_failed,
    not_a_package,
    register_converter,
)
from markitai.notices import user_notice
from markitai.utils.charset import read_text_file

if TYPE_CHECKING:
    from collections.abc import Callable

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

        html = read_text_file(input_path, html=True)
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
        import_markitdown()
        from markitdown import MarkItDown

        # Only generic file conversion needs the Office compatibility patches.
        # Native HTML completes without loading any of those libraries.
        apply_all_patches()
        apply_markitdown_cell_patches()
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


# A book's own pictures are read from the archive up to these sizes; the
# rest keep their reference (a picture over 32 MiB is no illustration)
_EPUB_IMAGE_MAX_BYTES = 32 * 1024 * 1024
_EPUB_IMAGES_MAX_BYTES = 128 * 1024 * 1024
_EPUB_IMAGE_TYPES = {
    ".png": "png",
    ".jpg": "jpeg",
    ".jpeg": "jpeg",
    ".gif": "gif",
    ".webp": "webp",
}
# Stands in for a picture's data: URI while its chapter is converted, so the
# bytes never go through the HTML parser and markdownify
_EPUB_PICTURE_PREFIX = "markitai-epub-picture:"
_EPUB_PICTURE_RE = re.compile(rf"\]\({_EPUB_PICTURE_PREFIX}(\d+)\)")
_DC_NAMESPACE = "http://purl.org/dc/elements/1.1/"
# Calibre's "no date" and a date written at midnight UTC
_UNDEFINED_DATE = re.compile(r"0\d{3}-")
_MIDNIGHT = re.compile(r"^(\d{4}-\d{2}-\d{2})T00:00(?::00(?:\.0+)?)?(?:Z|[+-]00:?00)?$")
_TAG = re.compile(r"<[^<>]+>")


def _plain(text: str) -> str:
    """OPF text as one plain line (a description may carry escaped HTML)."""
    import html

    return " ".join(html.unescape(_TAG.sub(" ", text)).split())


def _epub_document_meta(texts: Callable[[str], list[str]]) -> dict[str, Any]:
    """The book's own facts for the front matter, from its OPF.

    ``authors`` (every ``dc:creator``, a list), ``language``, ``publisher``,
    ``date`` and ``description`` (plain text), each only when the book
    names it: the keys and order of Rust's EPUB metadata (``office_meta``),
    which Rust writes as a bold block at the top of the body.
    """
    meta: dict[str, Any] = {}
    authors = [_plain(author) for author in texts("creator")]
    if authors := [author for author in authors if author]:
        meta["authors"] = authors
    for name in ("language", "publisher", "date", "description"):
        value = next((v for v in map(_plain, texts(name)) if v), "")
        if name == "date":
            if _UNDEFINED_DATE.match(value):
                continue
            value = _MIDNIGHT.sub(r"\1", value)
        if value:
            meta[name] = value
    return meta


class _EpubPictures:
    """A book's raster pictures, each read from the archive at most once.

    A reference resolves against the folder of the file that makes it
    (``../Images/cover.jpg`` from ``OEBPS/Text/``); one that resolves
    nowhere matches the one picture of that file name. A remote, missing
    or ambiguous reference, SVG, and a picture over 32 MiB (128 MiB per
    book) keep their reference.
    """

    def __init__(self, archive: zipfile.ZipFile) -> None:
        self._archive = archive
        self._images = {
            info.filename: info
            for info in archive.infolist()
            if not info.is_dir()
            and posixpath.splitext(info.filename)[1].lower() in _EPUB_IMAGE_TYPES
        }
        self._budget = _EPUB_IMAGES_MAX_BYTES
        self._tokens: dict[str, str | None] = {}
        self._uris: list[str] = []
        # Pictures some chapter shows (the OPF's cover is added when none does)
        self.shown: set[str] = set()

    def member(self, folder: str, reference: str) -> zipfile.ZipInfo | None:
        """The picture *reference* names from a file in archive *folder*."""
        from urllib.parse import unquote, urlsplit

        parts = urlsplit(reference.strip())
        if parts.scheme or parts.netloc or not parts.path or not self._images:
            return None
        path = unquote(parts.path)
        path = posixpath.normpath(posixpath.join(folder, path)).lstrip("/")
        if path in self._images:
            return self._images[path]
        name = posixpath.basename(path)
        found = [
            info
            for member, info in self._images.items()
            if posixpath.basename(member) == name
        ]
        return found[0] if len(found) == 1 else None

    def token(self, info: zipfile.ZipInfo) -> str | None:
        """A stand-in for the picture's data: URI, None over the limits."""
        if info.filename in self._tokens:
            return self._tokens[info.filename]
        token = None
        limit = min(_EPUB_IMAGE_MAX_BYTES, self._budget)
        if info.file_size <= limit:
            try:
                data = self._archive.read(info)
            except (OSError, zipfile.BadZipFile, RuntimeError):
                data = b""
            if data and len(data) <= limit:
                self._budget -= len(data)
                kind = _EPUB_IMAGE_TYPES[posixpath.splitext(info.filename)[1].lower()]
                encoded = base64.b64encode(data).decode("ascii")
                self._uris.append(f"data:image/{kind};base64,{encoded}")
                token = f"{_EPUB_PICTURE_PREFIX}{len(self._uris)}"
        self._tokens[info.filename] = token
        return token

    def show(self, folder: str, reference: str) -> str | None:
        """The stand-in for a picture a file in *folder* shows, if it has one."""
        info = self.member(folder, reference)
        if info is None:
            return None
        self.shown.add(info.filename)
        return self.token(info)

    def inline(self, markdown: str) -> str:
        """Put each picture's data: URI where its stand-in is."""
        return _EPUB_PICTURE_RE.sub(
            lambda match: f"]({self._uris[int(match.group(1)) - 1]})", markdown
        )


def _epub_chapter_markdown(data: bytes, path: str, pictures: _EpubPictures) -> str:
    """One chapter (XHTML in the archive at *path*) as Markdown.

    Each picture the book holds becomes its data: URI, which the image
    pipeline saves as one of the item's assets and links like a DOCX
    picture. A cover page's picture wrapped in SVG (``<svg><image
    xlink:href="cover.jpg"/></svg>``, which markdownify drops) becomes an
    ``<img>`` first.
    """
    from bs4 import BeautifulSoup

    import_markitdown()
    from markitdown.converters import HtmlConverter

    apply_markitdown_cell_patches()
    soup = BeautifulSoup(data, "html.parser", from_encoding="utf-8")
    folder = posixpath.dirname(path)
    for svg in soup.find_all("svg"):
        # Inside an SVG already replaced. Not svg.decomposed: on a live tag
        # bs4 answers that with a search of the whole subtree
        if svg.__dict__.get("_decomposed"):
            continue
        hrefs = [
            str(image.get("xlink:href") or image.get("href") or "")
            for image in svg.find_all("image")
        ]
        if not any(hrefs):
            continue
        for href in filter(None, hrefs):
            svg.insert_before(soup.new_tag("img", attrs={"src": href, "alt": ""}))
        svg.decompose()
    for img in soup.find_all("img"):
        token = pictures.show(folder, str(img.get("src") or img.get("data-src") or ""))
        if token is None:
            continue
        # Only src and alt, so the link stays the plain ![alt](data:…) the
        # image pipeline replaces (a title would follow the URI)
        alt = str(img.get("alt") or img.get("title") or "")
        img.attrs = {"src": token, "alt": re.sub(r"[\[\]\s]+", " ", alt).strip()}
    return HtmlConverter().convert_string(str(soup)).markdown.strip()


def _convert_epub(input_path: Path) -> ConvertResult:
    """Convert an EPUB the way markitdown does, keeping the book's pictures.

    Each spine document through markitdown's HTML converter, as markitdown
    does; the book's metadata goes to the front matter (``document_meta``,
    see ``_epub_document_meta``), not into the body as markitdown's bold
    ``**Title:** …`` block. Also: a spine entry
    the OPF percent-encodes (``Text/%E7%AC%AC1%E7%AB%A0.xhtml``) is found
    rather than skipped, each picture the book holds is kept (see
    ``_epub_chapter_markdown``), and a cover the OPF names but no page shows
    comes first.
    """
    from urllib.parse import unquote

    from defusedxml import minidom

    input_path = Path(input_path)
    logger.debug("[EpubConverter] Converting: {}", input_path.name)
    with zipfile.ZipFile(input_path) as archive:
        container = minidom.parse(archive.open("META-INF/container.xml"))
        opf_path = (
            container.getElementsByTagNameNS("*", "rootfile")[0]
            .getAttribute("full-path")
            .lstrip("/")
        )
        opf = minidom.parse(archive.open(opf_path))
        opf_folder = posixpath.dirname(opf_path)

        def texts(name: str) -> list[str]:
            return [
                node.firstChild.nodeValue.strip()
                for node in opf.getElementsByTagNameNS(_DC_NAMESPACE, name)
                if node.firstChild is not None and node.firstChild.nodeValue
            ]

        title = next((t for t in map(_plain, texts("title")) if t), None)
        document_meta = _epub_document_meta(texts)

        items = opf.getElementsByTagNameNS("*", "item")
        manifest = {
            item.getAttribute("id"): item.getAttribute("href") for item in items
        }
        names = set(archive.namelist())
        pictures = _EpubPictures(archive)
        chapters: list[str] = []
        for itemref in opf.getElementsByTagNameNS("*", "itemref"):
            href = manifest.get(itemref.getAttribute("idref"))
            if not href:
                continue
            for candidate in (unquote(href), href):
                path = posixpath.normpath(posixpath.join(opf_folder, candidate))
                if path in names:
                    chapters.append(
                        _epub_chapter_markdown(archive.read(path), path, pictures)
                    )
                    break

        # The cover the OPF names (EPUB 2 <meta name="cover">, EPUB 3
        # properties="cover-image"), when no page shows it
        cover_ids = {
            meta.getAttribute("content")
            for meta in opf.getElementsByTagNameNS("*", "meta")
            if meta.getAttribute("name") == "cover"
        }
        for item in items:
            if (
                item.getAttribute("id") not in cover_ids
                and "cover-image" not in item.getAttribute("properties").split()
            ):
                continue
            info = pictures.member(opf_folder, item.getAttribute("href"))
            if info is None:  # a cover page, not its picture
                continue
            if info.filename not in pictures.shown:
                token = pictures.token(info)
                if token is not None:
                    chapters.insert(0, f"![Cover]({token})")
            break

    # markitdown's own clean-up of every result
    markdown = "\n".join(
        line.rstrip() for line in re.split(r"\r?\n", "\n\n".join(chapters))
    )
    markdown = re.sub(r"\n{3,}", "\n\n", markdown)
    metadata: dict = {
        "source": str(input_path),
        "format": "EPUB",
        "converter": "markitdown",
    }
    if title:
        metadata["title"] = title
    if document_meta:
        metadata["document_meta"] = document_meta
    return ConvertResult(
        markdown=pictures.inline(markdown.strip()),
        images=[],
        metadata=metadata,
    )


@register_converter(FileFormat.EPUB)
class EpubConverter(BaseConverter):
    """Converter for EPUB e-book files (markitdown's HTML converter)."""

    supported_formats = [FileFormat.EPUB]

    def convert(
        self, input_path: Path, output_dir: Path | None = None
    ) -> ConvertResult:
        from xml.parsers.expat import ExpatError

        try:
            return _convert_epub(Path(input_path))
        except (
            KeyError,
            IndexError,
            ValueError,
            ExpatError,
            zipfile.BadZipFile,
        ) as exc:
            # No container or OPF, or one that does not parse (defusedxml's
            # refusals are ValueErrors)
            if isinstance(exc, zipfile.BadZipFile):
                reason = not_a_package(input_path, "EPUB")
                if reason:
                    conversion_failed(reason)
            conversion_failed(f"Malformed EPUB: {exc}")


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
    """Apple Numbers spreadsheets, which no local reader can convert.

    markitdown has no Numbers converter: its generic zip handler returned
    the package's metadata members (never the tables) as a successful
    document. The format stays known so ``-b cloudflare`` converts it.
    """

    supported_formats = [FileFormat.NUMBERS]

    def convert(
        self, input_path: Path, output_dir: Path | None = None
    ) -> ConvertResult:
        conversion_failed(
            "Apple Numbers files cannot be converted locally; export the "
            "spreadsheet to .xlsx or .csv, or convert it with -b cloudflare"
        )
