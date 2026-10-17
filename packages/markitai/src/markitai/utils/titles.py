"""What may be a document's ``title``: plain text that names something.

A title comes from a converter's metadata (a PDF's Info title, an EPUB's
``dc:title``), the most prominent text of a PDF's first pages, a heading of
the Markdown, or the file name. Whatever the source, the front matter gets
plain text (:func:`clean_title`: no HTML, no Markdown emphasis, no private-use
glyphs a PDF font maps its bullets to) and only a candidate that names the
document (:func:`meaningful_title`): not an application's default name
(``Untitled``, ``Document1``, ``Microsoft Word - x.docx``, Rust's
``placeholder_title`` rules), not a number (``20240315000127E``, ``2025``),
not a page number. A candidate that fails is skipped for the next source,
down to the file name.

Seen in practice: an insurance policy PDF got ``币值单位：人民币元``
(a small bold caption, the first heading) in its base output and
``<sup>投保人拥有的重要权益</sup>`` with two private-use glyphs in its LLM
output (the first H1, through a second, different extractor).
"""

from __future__ import annotations

import html
import re
import unicodedata

_TAG = re.compile(r"</?[A-Za-z][A-Za-z0-9-]*(?:\s[^<>]*)?/?>")
_BREAK = re.compile(r"<(?:br|/p|/div|/li)\b[^<>]*>", re.IGNORECASE)
# A footnote mark set as superscript: Outlook<sup>1</sup>, Data<sup>*,†</sup>
_MARK = re.compile(r"<sup>\s*[\d*†‡§¶,\s]{1,8}</sup>", re.IGNORECASE)
_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
_FOOTNOTE = re.compile(r"\[\^[^\]]*\]")
_HEADING_MARKS = re.compile(r"^#{1,6}\s+")
_ESCAPE = re.compile(r"\\([\\`*_{}\[\]()#+\-.!|<>~])")
_EMPHASIS = re.compile(r"(\*\*|__|~~)(.+?)\1")
_WRAPPERS = ("**", "__", "~~", "`", "*", "_")
# "Page 3", "第 3 页", "3 / 12", "- 3 -", "p. 7", "Seite 2"
_PAGE_NUMBER = re.compile(
    r"^(?:(?:page|p\.|pg\.?|seite|página|pagina)\s*\d+(?:\s*(?:of|/)\s*\d+)?"
    r"|第\s*\d+\s*[页頁]"
    r"|[-–—]?\s*\d+\s*(?:/\s*\d+)?\s*[-–—]?)$",
    re.IGNORECASE,
)

_EXTENSIONS = frozenset(
    [
        "htm",
        "html",
        "xhtml",
        "mht",
        "mhtml",
        "pdf",
        "doc",
        "docx",
        "docm",
        "dot",
        "dotx",
        "odt",
        "ott",
        "rtf",
        "txt",
        "md",
        "tex",
        "ppt",
        "pptx",
        "pps",
        "ppsx",
        "odp",
        "key",
        "xls",
        "xlsx",
        "ods",
        "csv",
        "pages",
        "numbers",
        "indd",
        "ps",
        "eps",
        "dvi",
        "png",
        "jpg",
        "jpeg",
        "gif",
        "tif",
        "tiff",
        "bmp",
        "svg",
    ]
)
_OFFICE_PREFIXES = (
    "microsoft word - ",
    "microsoft powerpoint - ",
    "microsoft excel - ",
)
_DEFAULT_NAMES = frozenset(
    {
        "untitled",
        "untitled document",
        "untitled presentation",
        "untitled spreadsheet",
        "untitled form",
        "untitled design",
        "untitled drawing",
        "title",
        "document",
        "presentation",
        "new document",
        "word document",
        "microsoft word document",
        "new microsoft word document",
        "powerpoint presentation",
        "microsoft powerpoint presentation",
        "sans titre",
        "unbenannt",
        "sin título",
        "sem título",
        "senza titolo",
        "无标题",
        "未命名",
        "無題",
        "無標題",
    }
)
_NUMBERED_NAMES = frozenset(
    {
        "document",
        "presentation",
        "book",
        "workbook",
        "drawing",
        "publication",
        "doc",
        "文档",
        "演示文稿",
        "工作簿",
    }
)


def _visible(char: str) -> bool:
    """Not a control, format, private-use, surrogate or unassigned code point.

    A PDF font maps its bullets and icons to the private-use area
    (``\\uf043``): text extraction keeps them, a reader sees a symbol.
    """
    return unicodedata.category(char)[0] != "C" and char != "�"


def clean_title(text: str) -> str:
    """*text* as a plain one-line title: no markup, no invisible characters.

    Drops every HTML tag (``<sup>``, ``<mark>``, ``<br>`` …) and decodes
    entities, keeps a link's or an image's text, drops footnote references
    and marks (``[^1]``, ``<sup>1</sup>``),
    heading marks, Markdown emphasis and backslash escapes, and collapses
    whitespace.
    """
    text = "".join(ch if _visible(ch) or ch in "\t\n\r" else " " for ch in text)
    text = " ".join(text.split())
    text = _HEADING_MARKS.sub("", text)
    text = _LINK.sub(r"\1", text)
    text = _FOOTNOTE.sub("", text)
    text = _MARK.sub("", text)
    text = _BREAK.sub(" ", text)
    text = _TAG.sub("", text)
    text = html.unescape(text)
    # Entities can name invisible characters too (&#xf043;)
    text = "".join(ch if _visible(ch) else " " for ch in text)
    text = _EMPHASIS.sub(r"\2", text)
    text = _ESCAPE.sub(r"\1", text)
    text = " ".join(text.split())
    changed = True
    while changed:
        changed = False
        for wrapper in _WRAPPERS:
            if (
                text.startswith(wrapper)
                and text.endswith(wrapper)
                and len(text) > len(wrapper) * 2
            ):
                text = text[len(wrapper) : -len(wrapper)].strip()
                changed = True
    return text


def placeholder_title(title: str) -> bool:
    """A name an application gives a document nobody named.

    Rust's ``placeholder_title`` (``formats/pdf.rs``): an Office print
    driver's prefix (``Microsoft Word - report.doc``), a file name with a
    document or image extension, a lowercase slug (``layout-table``), or a
    default name (``Untitled``, ``Untitled-2``, ``Title``, ``无标题``; numbered
    only: ``Document1``, ``Book1``, ``文档1``). For metadata titles: a heading
    in lower case is a heading.
    """
    lower = title.strip().lower()
    if lower.startswith(_OFFICE_PREFIXES):
        return True
    stem, dot, extension = lower.rpartition(".")
    if dot and stem.strip() and extension in _EXTENSIONS:
        return True
    if _default_name(lower):
        return True
    return any(c.isascii() and c.islower() for c in title) and all(
        (c.isascii() and (c.islower() or c.isdigit())) or c in "-_" for c in title
    )


def _default_name(lower: str) -> bool:
    stem = lower.rstrip("0123456789 -_()")
    numbered = any(c.isdigit() for c in lower[len(stem) :])
    return stem in _DEFAULT_NAMES or (numbered and stem in _NUMBERED_NAMES)


def meaningful_title(title: str, *, metadata: bool = False) -> bool:
    """Whether *title* (already :func:`clean_title`-ed) names the document.

    Two letters at least (``2025``, ``—``, ``1.2.3`` are no names), digits
    at most 70 % of its letters and digits (``20240315000127E`` is an id;
    ``2025年报`` and ``Report 2024`` are names), not a page number, not an
    application's default name (:func:`placeholder_title`; its slug rule
    only for a *metadata* title). A long one is cut where it is written.
    """
    if not title:
        return False
    letters = sum(1 for c in title if c.isalpha())
    digits = sum(1 for c in title if c.isdigit())
    if letters < 2 or digits > 0.7 * (letters + digits):
        return False
    if _PAGE_NUMBER.match(title):
        return False
    lower = title.lower()
    if metadata:
        return not placeholder_title(title)
    return not _default_name(lower) and not lower.startswith(_OFFICE_PREFIXES)


def usable_title(text: str | None, *, metadata: bool = False) -> str:
    """*text* cleaned when it is a meaningful title, else ``""``."""
    if not text:
        return ""
    title = clean_title(text)
    return title if meaningful_title(title, metadata=metadata) else ""


def heading_title(markdown: str) -> str:
    """The first Markdown heading, of any level, that names the document.

    The rule both outputs use (the base ``.md`` and the ``.llm.md``), and
    Rust's (``output.rs``): a line starting with ``# `` or ``##``; a heading
    that is no name (:func:`meaningful_title`) is skipped for the next one.
    Whitespace before the first nonempty line is ignored; code fences are
    not skipped. ``""`` when none names it.
    """
    leading = re.match(r"\s*", markdown)
    position = leading.end() if leading else 0
    while position < len(markdown):
        if markdown[position] != "#":
            boundary = markdown.find("\n#", position)
            if boundary < 0:
                return ""
            position = boundary + 1
        end = markdown.find("\n", position)
        if end < 0:
            end = len(markdown)
        line = markdown[position:end]
        if len(line) > 1 and line[1] in "# ":
            title = usable_title(line.lstrip("#"))
            if title:
                return title
        position = end + 1
    return ""
