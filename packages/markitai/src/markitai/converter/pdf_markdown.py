"""Repairs to the page Markdown pymupdf4llm writes: symbol glyphs and headings.

Seen in practice: a Chinese policy's table of contents in the
workbench listed headings such as ``▯▯<sup>…</sup>`` and whole lines of body
text. Three shapes of pymupdf4llm's output, repaired here for every PDF path
(the base conversion, the native pages of an OCR run, the text the vision
LLM gets; :func:`plain_headings` cleans a model's headings in the
``.llm.md`` too):

1. **Symbol-font glyphs.** A Symbol, Wingdings or Wingdings 2 font maps its
   glyphs to the private-use area (``U+F000`` plus the font's byte): a
   reader sees a box. Each one is mapped by the font that draws it on its
   page (Symbol's Greek and math signs, the common Wingdings bullets, boxes
   and pointing hands); any other private-use code point, a noncharacter
   (iText maps ``.notdef`` to ``U+FFFF``) or a control character is
   dropped. A heading drops dingbats altogether (they decorate it), and so
   does a list item right after its marker (the dingbat was its bullet).
2. **Tags in headings.** A large icon at the start of a line raises the
   line's baseline, so PyMuPDF flags the text after it superscript and
   pymupdf4llm wraps it in ``<sup>``. A heading has no HTML tags (their
   text stays) and no superscript footnote marks (``Outlook<sup>1</sup>``);
   Markdown emphasis and code spans stay, as in Rust's output. In body text
   a ``<sup>`` around two or more Han ideographs is no superscript either (a
   footnote mark never is) and loses its tags.
3. **Lines that are no headings.** The layout model takes bold or underlined
   body lines for section headers. Rust's rules (pdf-inspector
   ``classify.rs``, markitai-rust 29136ec) say what is none: a sentence (it
   ends with or holds ``。！？；``, ends with ``，、``, or runs on past ``，``
   for twelve characters or more), a ``label：value`` field (a label of at
   most twelve characters before a full-width colon, a value after it; a
   section number such as ``第一章`` is no label), more than fifteen words (a
   run of CJK text counts a word for every three characters), a
   table-of-contents entry (leader dots and a page number); and here also a
   CJK sentence leading into a list (sixteen CJK characters or more ending
   with ``：``). Such a line is a paragraph again, with its inline styling.
   A heading that is only a section number (``2.4``) takes the same-level
   heading after it as its title. A heading line that runs to its column's
   edge (30 CJK characters or 60 characters and more) without ending its
   sentence, with the lines after it down to the sentence's short tail (at
   most eight characters, ``利。``), is one paragraph; so is such a tail set
   as a heading after the paragraph it ends.

4. **A heading wrapped onto a second line.** The layout model boxes each
   line of a two-line heading on its own, so ``## Cras fringilla ipsum
   magna, in fringilla dui commodo`` and ``## a.`` came out as two
   headings. Two headings of one level in a row are one when the page's
   text layer shows the second line right below the first (a gap under
   half a line), in the same size (within 10 %) and weight, starting
   before the first line ends, and the first does not end its sentence;
   where the lines cannot be found in the text layer, when the second
   starts with a lower-case letter.

On the round-15 PDF corpus (31 documents and Rust's 8 CJK layout fixtures)
only the documents with these shapes changed.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from loguru import logger

# --- symbol fonts -----------------------------------------------------------

# Adobe Symbol encoding (the byte a Symbol font draws) to Unicode. Bracket
# and radical pieces (0x60, 0xBD-0xBF, 0xE6-0xEF, 0xF3-0xFE) are left out:
# they only draw a part of a larger sign.
_SYMBOL = dict(
    zip(
        range(0x20, 0x7F),
        " !∀#∃%&∋()∗+,−./0123456789:;<=>?"
        "≅ΑΒΧΔΕΦΓΗΙϑΚΛΜΝΟΠΘΡΣΤΥςΩΞΨΖ[∴]⊥_"
        "\0αβχδεφγηιϕκλμνοπθρστυϖωξψζ{|}∼",
        strict=True,
    )
)
_SYMBOL.update(
    zip(
        range(0xA0, 0xBD),
        "€ϒ′≤⁄∞ƒ♣♦♥♠↔←↑→↓°±″≥×∝∂•÷≠≡≈…",
        strict=True,
    )
)
_SYMBOL.update(
    zip(
        range(0xC0, 0xE6),
        "ℵℑℜ℘⊗⊕∅∩∪⊃⊇⊄⊂⊆∈∉∠∇®©™∏√⋅¬∧∨⇔⇐⇑⇒⇓◊〈®©™∑",
        strict=True,
    )
)
_SYMBOL.update({0xF1: "〉", 0xF2: "∫"})
del _SYMBOL[0x60]

# The Wingdings and Wingdings 2 glyphs documents use as bullets, check
# boxes and pointers; others are dropped.
_WINGDINGS = {
    0x20: " ",
    0x45: "☜",
    0x46: "☞",
    0x47: "☝",
    0x48: "☟",
    0x6C: "●",
    0x6E: "■",
    0x6F: "□",
    0x71: "❑",
    0x75: "◆",
    0x76: "❖",
    0xA7: "▪",
    0xA8: "☐",
    0xD8: "➢",
    0xFB: "✗",
    0xFC: "✓",
    0xFD: "☒",
    0xFE: "☑",
}
_WINGDINGS_2 = {
    0x20: " ",
    0x43: "☞",
    0x4F: "✗",
    0x50: "✓",
    0x52: "☑",
    0x54: "☒",
    0xA3: "□",
}
_FONT_TABLES = {
    "symbol": _SYMBOL,
    "wingdings": _WINGDINGS,
    "wingdings2": _WINGDINGS_2,
}
# Longest name first: "wingdings2..." is no "wingdings..."
_FAMILIES = ("wingdings3", "wingdings2", "wingdings", "webdings", "symbol")
_DINGBAT_FAMILIES = frozenset({"wingdings", "wingdings2", "wingdings3", "webdings"})

#: What one private-use character becomes on a page: (text, dingbat?)
Glyph = tuple[str, bool]


def _symbol_family(font: str) -> str | None:
    """``symbol``, ``wingdings``, ``wingdings2`` … for a symbol font's name.

    ``ABCDEF+Wingdings-Regular``, ``SymbolMT``, ``Wingdings2`` (a subset tag,
    a style or a foundry suffix are no part of the family).
    """
    name = re.sub(r"[^a-z0-9]", "", font.rpartition("+")[2].lower())
    return next((family for family in _FAMILIES if name.startswith(family)), None)


def _private(char: str) -> bool:
    code = ord(char)
    return (
        0xE000 <= code <= 0xF8FF
        or 0xF0000 <= code <= 0xFFFFD
        or 0x100000 <= code <= 0x10FFFD
    )


def _invisible(char: str) -> bool:
    """A glyph a reader cannot read: private use, noncharacter, control."""
    if char in "\n\t\r":
        return False
    category = unicodedata.category(char)
    # Cn: unassigned and the noncharacters (U+FFFF, U+FDD0 …); Cs: surrogates
    return category in ("Co", "Cn", "Cs", "Cc")


def _span_glyphs(text: str, font: str, glyphs: dict[str, Glyph]) -> None:
    family = _symbol_family(font)
    for char in text:
        if not _private(char) or char in glyphs:
            continue
        code = ord(char)
        table = _FONT_TABLES.get(family or "", {})
        mapped = table.get(code - 0xF000, "") if 0xF020 <= code <= 0xF0FF else ""
        glyphs[char] = (mapped, family in _DINGBAT_FAMILIES)


def page_glyph_maps(
    input_path: Path, page_numbers: Sequence[int], page_texts: Sequence[str]
) -> list[dict[str, Glyph]]:
    """For each page, what its private-use characters stand for.

    Only pages whose Markdown holds one are read again (most have none).
    A character a font of the page draws in two ways keeps the first.
    """
    maps: list[dict[str, Glyph]] = [{} for _ in page_texts]
    wanted = [
        index
        for index, text in enumerate(page_texts)
        if any(_private(char) for char in text)
    ]
    if not wanted:
        return maps
    import pymupdf

    try:
        with pymupdf.open(input_path) as doc:
            for index in wanted:
                number = page_numbers[index]
                if not 1 <= number <= doc.page_count:
                    continue
                page = doc[number - 1]
                content: dict[str, Any] = page.get_text(  # type: ignore[assignment]
                    "dict", flags=pymupdf.TEXTFLAGS_TEXT
                )
                for block in content.get("blocks", []):
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            _span_glyphs(span["text"], span["font"], maps[index])
    except Exception as e:  # the glyphs are then dropped, not mapped
        logger.debug("[PDF] Symbol fonts not read for {}: {}", input_path.name, e)
    return maps


# --- text rules (Rust classify.rs) -----------------------------------------


def _cjk(char: str) -> bool:
    """Rust pdf-inspector's ``is_cjk_char``."""
    code = ord(char)
    return (
        0x1100 <= code <= 0x11FF
        or 0x3000 <= code <= 0x303F
        or 0x3040 <= code <= 0x30FF
        or 0x3130 <= code <= 0x318F
        or 0x4E00 <= code <= 0x9FFF
        or 0xAC00 <= code <= 0xD7AF
        or 0xF900 <= code <= 0xFAFF
        or 0xFF00 <= code <= 0xFFEF
    )


def _han(char: str) -> bool:
    code = ord(char)
    return (
        0x4E00 <= code <= 0x9FFF
        or 0x3400 <= code <= 0x4DBF
        or 0xF900 <= code <= 0xFAFF
        or 0x20000 <= code <= 0x3134F
    )


def text_words(text: str) -> int:
    """Words set apart by spaces; a CJK run counts one for every three characters."""
    return sum(
        max(1, -(-sum(1 for c in word if _cjk(c)) // 3)) for word in text.split()
    )


_SENTENCE_END = "。！？；"


def reads_as_sentence(text: str) -> bool:
    """A line that reads as a sentence, not a heading (Rust ``reads_as_sentence``).

    It ends with, or holds within, ``。！？；``, ends with ``，`` or ``、`` (the
    sentence runs on), or runs on past ``，`` for twelve characters or more.
    Latin text is not judged: a heading may be a sentence there
    (``Lorem ipsum dolor sit amet.`` in a sample document).
    """
    text = text.strip()
    if not text:
        return False
    if text[-1] in _SENTENCE_END + "，、":
        return True
    if any(
        char in _SENTENCE_END and text[at + 1 :].strip() for at, char in enumerate(text)
    ):
        return True
    return "，" in text and len(text) >= 12


_SECTION_LABEL = re.compile(
    r"^(?:第[\d一二三四五六七八九十百零〇两]+[章节条篇部款编]"
    r"|[一二三四五六七八九十]+|\d+(?:\.\d+)*|(?:part|chapter|section)\b.*)$",
    re.IGNORECASE,
)


def reads_as_field(text: str) -> bool:
    """A ``label：value`` field (Rust ``reads_as_field``), not a section's title.

    A label of at most twelve characters with no sentence inside, a
    full-width colon, a value. A section number is no label
    (``第一章：总则`` is a heading).
    """
    label, colon, value = text.strip().partition("：")
    label = label.strip()
    return (
        bool(colon)
        and bool(label)
        and len(label) <= 12
        and not any(c in label for c in "。，；")
        and bool(value.strip())
        and not _SECTION_LABEL.match(label)
    )


_TOC_ENTRY = re.compile(r"(?:[.…·．]\s*){3,}\s*\d{1,4}(?:[./]\d+)*\s*$")
_SECTION_NUMBER = re.compile(
    r"^(?:\d+(?:\.\d+)*\.?|第[\d一二三四五六七八九十百零〇两]+[章节条篇部款编])$"
)


def _no_heading(plain: str) -> bool:
    return (
        reads_as_sentence(plain)
        or reads_as_field(plain)
        or text_words(plain) > 15
        or bool(_TOC_ENTRY.search(plain))
        # A sentence leading into a list: "…应同时符合下列要求："
        or (plain.endswith("：") and sum(1 for c in plain if _cjk(c)) >= 16)
    )


# --- inline markup ----------------------------------------------------------

_TAG = re.compile(r"</?[A-Za-z][A-Za-z0-9-]*(?:\s[^<>]*)?/?>")
_BREAK = re.compile(r"<(?:br|/p|/div|/li)\b[^<>]*>", re.IGNORECASE)
_SUP = re.compile(r"<sup>(.*?)</sup>", re.IGNORECASE | re.DOTALL)
_EMPHASIS = re.compile(r"(\*\*|__|~~)(.+?)\1")
_UNDERSCORE = re.compile(r"(?<![\w*])_(?=\S)(.+?)(?<=\S)_(?![\w*])")
_STAR = re.compile(r"(?<![\w*])\*(?=[^\s*])(.+?)(?<=[^\s*])\*(?![\w*])")
# A footnote mark: digits (7.5), asterisks, daggers, commas, one letter
_MARK = re.compile(r"^(?:[\d.*†‡§¶,\s]{1,8}|[a-z])$")
_CODE = re.compile(r"(`+)(.+?)\1")
_HEADING = re.compile(r"^(#{1,6})[ \t]+(.*?)(?:[ \t]+#+)?[ \t]*$")
_FENCE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")
# Lines that are no paragraph text: headings, tables, quotes, comments, images, lists
_NOT_PROSE = ("#", "|", ">", "<!--", "![", "- ", "* ", "+ ")
#: A heading ending so is complete: the next heading line is no part of it.
_HEADING_END = ".!?:;" + _SENTENCE_END + "："
#: A heading's second line starts less than this many line heights below
#: its first, in a font size within this share of the first's.
_WRAP_GAP = 0.5
_WRAP_SIZE = 0.1
#: Two headings' plain texts, one right after the other.
Pair = tuple[str, str]
_LIST_MARKER = re.compile(r"^\s*(?:[-*+]|\d+[.)])[ \t]+")


def _strip_emphasis(text: str) -> str:
    for _ in range(3):  # nested: **_text_**
        text = _EMPHASIS.sub(r"\2", text)
        text = _UNDERSCORE.sub(r"\1", text)
        text = _STAR.sub(r"\1", text)
    return text


def _apply_glyphs(text: str, glyphs: dict[str, Glyph], *, heading: bool) -> str:
    """Private-use glyphs mapped (a heading drops dingbats), the rest dropped."""
    if not any(_invisible(char) for char in text):
        return text
    out = []
    for char in text:
        if not _invisible(char):
            out.append(char)
            continue
        mapped, dingbat = glyphs.get(char, ("", False))
        out.append(" " if heading and dingbat else mapped)
    return "".join(out)


def _untagged(text: str) -> str:
    """*text* without HTML tags and superscript footnote marks; code spans kept."""

    def superscript(match: re.Match[str]) -> str:
        inner = _TAG.sub("", match.group(1))
        return "" if _MARK.match(_strip_emphasis(inner)) else inner

    def segment(text: str) -> str:
        return _TAG.sub("", _BREAK.sub(" ", _SUP.sub(superscript, text)))

    parts = []
    position = 0
    for match in _CODE.finditer(text):
        parts.append(segment(text[position : match.start()]))
        parts.append(match.group(0))
        position = match.end()
    parts.append(segment(text[position:]))
    return "".join(parts)


def heading_text(text: str, glyphs: dict[str, Glyph] | None = None) -> str:
    """A heading's inline text with no tags, footnote marks or glyph boxes.

    Dingbats are dropped, a Symbol font's signs mapped; Markdown emphasis
    and code spans stay.
    """
    text = _untagged(_apply_glyphs(text, glyphs or {}, heading=True))
    text = "".join(" " if _invisible(char) else char for char in text)
    return " ".join(text.split())


def _plain(text: str) -> str:
    """Text the heading rules read: :func:`heading_text` without emphasis."""
    return " ".join(_strip_emphasis(text).replace("`", "").split())


def _unwrap_cjk_superscripts(text: str) -> str:
    """``<sup>`` around a run of Han text (two ideographs or more) loses its tags."""

    def unwrap(match: re.Match[str]) -> str:
        han = sum(1 for c in _TAG.sub("", match.group(1)) if _han(c))
        return match.group(1) if han >= 2 else match.group(0)

    return _SUP.sub(unwrap, text) if "<sup>" in text.lower() else text


def _body_line(line: str, glyphs: dict[str, Glyph]) -> str:
    if any(_invisible(char) for char in line):
        marker = _LIST_MARKER.match(line)
        if marker:
            # Glyphs right after a list marker were the item's bullet
            end = marker.end()
            while end < len(line) and (_invisible(line[end]) or line[end] in " \t"):
                end += 1
            if any(_invisible(char) for char in line[marker.end() : end]):
                line = line[: marker.end()] + line[end:]
        line = _apply_glyphs(line, glyphs, heading=False)
        line = re.sub(r"(?<=\S)[ \t]{2,}", " ", line)
        if not line.strip():
            return ""
    return _unwrap_cjk_superscripts(line)


# --- page repair ------------------------------------------------------------


def _join(left: str, right: str) -> str:
    """Two lines of one sentence: no space between CJK characters."""
    left, right = left.rstrip(), right.lstrip()
    last = _TAG.sub("", left).rstrip("*_~ ")[-1:]
    first = _TAG.sub("", right).lstrip("*_~ ")[:1]
    if last and first and _cjk(last) and _cjk(first):
        return left + right
    return f"{left} {right}"


def _wraps(plain: str) -> bool:
    """A line long enough to have run to its column's edge, not ending a sentence."""
    if not plain or plain[-1] in _SENTENCE_END + "：:":
        return False
    return sum(1 for c in plain if _cjk(c)) >= 30 or len(plain) >= 60


def _sentence_tail(plain: str) -> bool:
    """The end of a sentence that ran over from the line before (``利。``)."""
    return _ends_sentence(plain) and len(plain) <= 8


_VERBATIM = -1


class _Line:
    """A line of the page: a heading (level 1-6), a code line or another line."""

    __slots__ = ("level", "raw", "text")

    def __init__(self, level: int, raw: str, text: str = "") -> None:
        self.level = level
        self.raw = raw  # a heading's inline text as written, or the line
        self.text = text  # a heading's plain text (the rules read it)


def _lines(markdown: str, glyphs: dict[str, Glyph]) -> list[_Line]:
    lines: list[_Line] = []
    fence: str | None = None
    for line in markdown.split("\n"):
        opened = _FENCE.match(line)
        if fence is not None or opened:
            if fence is None:
                fence = opened.group(1) if opened else None
            elif opened and opened.group(1)[0] == fence[0]:
                fence = None
            lines.append(_Line(_VERBATIM, line))
            continue
        heading = _HEADING.match(line)
        if heading:
            raw = heading.group(2)
            lines.append(
                _Line(len(heading.group(1)), raw, _plain(heading_text(raw, glyphs)))
            )
        else:
            lines.append(_Line(0, line))
    return lines


def _same_level_run(lines: list[_Line], index: int) -> list[int]:
    """Headings of *lines[index]*'s level right after it (blank lines between)."""
    level = lines[index].level
    run: list[int] = []
    position = index + 1
    while position < len(lines) and len(run) < 4:
        line = lines[position]
        if line.level == 0 and not line.raw.strip():
            position += 1
            continue
        if line.level != level:
            break
        run.append(position)
        position += 1
    return run


def _sentence_run(lines: list[_Line], index: int) -> list[int]:
    """*lines[index]* and the lines after it that end its sentence, or [].

    A heading line that runs to its column's edge without ending its
    sentence, then same-level heading lines that run on, down to the
    sentence's tail; or the tail as a line of body text (``…有权终止服务的权``
    then ``<u>利。</u>``).
    """
    if not _wraps(lines[index].text):
        return []
    run = [index]
    position = index + 1
    while position < len(lines) and len(run) < 5:
        line = lines[position]
        position += 1
        if line.level == 0 and not line.raw.strip():
            continue
        if line.level == 0:
            text = _plain(_untagged(line.raw))
            body = not line.raw.lstrip().startswith(_NOT_PROSE)
            return [*run, position - 1] if body and _sentence_tail(text) else []
        if line.level != lines[index].level:
            return []
        run.append(position - 1)
        if _ends_sentence(line.text):
            return run if _sentence_tail(line.text) else []
        if not _wraps(line.text):
            return []
    return []


def _ends_sentence(text: str) -> bool:
    return bool(text) and text[-1] in _SENTENCE_END


def _end_paragraph(out: list[str]) -> None:
    """A blank line before a new paragraph unless one is there."""
    if out and out[-1].strip():
        out.append("")


def _ends_paragraph(out: list[str], line: _Line) -> bool:
    """Whether *line* is the last line of the paragraph *out* ends with."""
    previous = next((p for p in range(len(out) - 1, -1, -1) if out[p].strip()), None)
    if previous is None or not _sentence_tail(line.text):
        return False
    before = out[previous]
    return (
        _cjk(line.text[0])
        and not before.lstrip().startswith(_NOT_PROSE[:5])
        and _wraps(_plain(_untagged(before)))
    )


def _join_heading(left: str, right: str) -> str:
    """Two heading lines' inline text as one: ``**a** **b**`` is ``**a b**``."""
    left, right = left.rstrip(), right.lstrip()
    for mark in ("***", "**", "*", "__", "_"):
        if left.endswith(mark) and right.startswith(mark):
            return _join(left[: -len(mark)], right[len(mark) :])
    return _join(left, right)


def _continues(first: _Line, second: _Line, wrapped: frozenset[Pair] | None) -> bool:
    """Whether heading *second* is the next line of heading *first*."""
    if not first.text or not second.text or first.text[-1] in _HEADING_END:
        return False
    if wrapped is not None and (first.text, second.text) in wrapped:
        return True
    return second.text[0].islower()


def repair_page_markdown(
    markdown: str,
    glyphs: dict[str, Glyph] | None = None,
    wrapped: frozenset[Pair] | None = None,
) -> str:
    """One page's Markdown with its glyphs mapped and its headings plain.

    Args:
        markdown: The page's Markdown.
        glyphs: What its private-use characters stand for.
        wrapped: Pairs of heading texts the text layer shows as one heading
            wrapped onto a second line (:func:`wrapped_headings`).
    """
    glyphs = glyphs or {}
    lines = _lines(markdown, glyphs)
    out: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.level == _VERBATIM:
            out.append(line.raw)
            index += 1
            continue
        if line.level == 0:
            out.append(_body_line(line.raw, glyphs))
            index += 1
            continue
        following = _same_level_run(lines, index)
        if following and _continues(line, lines[following[0]], wrapped):
            # A heading wrapped onto a second line: one heading
            title = lines[following[0]]
            title.raw = _join_heading(line.raw, title.raw)
            title.text = _join(line.text, title.text)
            index = following[0]
            continue
        if following and _SECTION_NUMBER.match(line.text):
            # "2.4" then "服务范围": one heading
            title = lines[following[0]]
            title.raw = f"{line.raw} {title.raw}"
            title.text = f"{line.text} {title.text}"
            index = following[0]
            continue
        run = _sentence_run(lines, index)
        if run:
            # A long line that does not end its sentence, then the lines that do
            paragraph = _body_line(line.raw, glyphs)
            for position in run[1:]:
                paragraph = _join(paragraph, _body_line(lines[position].raw, glyphs))
            _end_paragraph(out)
            out.append(paragraph)
            index = run[-1] + 1
            continue
        index += 1
        if not any(c.isalnum() for c in line.text):
            continue  # nothing left but glyphs and tags
        if not _no_heading(line.text):
            out.append(f"{'#' * line.level} {heading_text(line.raw, glyphs)}")
        elif _ends_paragraph(out, line):
            # A sentence's last line, set apart as a heading
            while not out[-1].strip():
                out.pop()
            out[-1] = _join(out[-1], _body_line(line.raw, glyphs))
        else:
            _end_paragraph(out)
            out.append(_body_line(line.raw, glyphs))
    return "\n".join(out)


def _heading_pairs(markdown: str) -> list[Pair]:
    """Plain texts of each heading and the same-level heading right after it."""
    lines = _lines(markdown, {})
    return [
        (line.text, lines[following[0]].text)
        for index, line in enumerate(lines)
        if line.level > 0 and (following := _same_level_run(lines, index))
    ]


def _layer_lines(page: Any) -> list[tuple[str, tuple[float, ...], float, bool]]:
    """The page's text lines: (text, bbox, font size, bold), in reading order."""
    import pymupdf

    out: list[tuple[str, tuple[float, ...], float, bool]] = []
    content: dict[str, Any] = page.get_text(  # type: ignore[assignment]
        "dict", flags=pymupdf.TEXTFLAGS_TEXT, sort=True
    )
    for block in content.get("blocks", []):
        for line in block.get("lines", []):
            spans = [s for s in line.get("spans", []) if s["text"].strip()]
            if not spans:
                continue
            text = " ".join("".join(s["text"] for s in spans).split())
            size = max(s["size"] for s in spans)
            bold = all(s["flags"] & 16 or "bold" in s["font"].lower() for s in spans)
            out.append((text, tuple(line["bbox"]), size, bold))
    return out


def _wrapped_pairs(page: Any, pairs: list[Pair]) -> set[Pair]:
    """The *pairs* the page draws as one heading on two lines."""
    lines = _layer_lines(page)
    found: set[Pair] = set()
    for (text, box, size, bold), (text2, box2, size2, bold2) in zip(lines, lines[1:]):
        pair = next((p for p in pairs if p == (_plain(text), _plain(text2))), None)
        if pair is None:
            continue
        height = box[3] - box[1]
        gap = box2[1] - box[3]
        if (
            -height / 2 < gap < height * _WRAP_GAP
            and abs(size2 - size) <= size * _WRAP_SIZE
            and bold == bold2
            and box2[0] < box[2]
        ):
            found.add(pair)
    return found


def wrapped_headings(
    input_path: Path, page_numbers: Sequence[int], page_texts: Sequence[str]
) -> list[frozenset[Pair] | None]:
    """For each page, the heading pairs its text layer shows as one heading.

    Only pages with two headings of a level in a row are read; None for a
    page whose text layer could not be read (the text rule decides there).
    """
    result: list[frozenset[Pair] | None] = [frozenset() for _ in page_texts]
    wanted = {
        i: pairs for i, t in enumerate(page_texts) if (pairs := _heading_pairs(t))
    }
    if not wanted:
        return result
    import pymupdf

    try:
        with pymupdf.open(input_path) as doc:
            for index, pairs in wanted.items():
                number = page_numbers[index]
                result[index] = None
                if 1 <= number <= doc.page_count:
                    result[index] = frozenset(_wrapped_pairs(doc[number - 1], pairs))
    except Exception as e:  # the text rule decides alone
        logger.debug("[PDF] Heading lines not read for {}: {}", input_path.name, e)
        result = [None if i in wanted else r for i, r in enumerate(result)]
    return result


def repair_pages(
    input_path: Path, page_numbers: Sequence[int], page_texts: Sequence[str]
) -> list[str]:
    """:func:`repair_page_markdown` for each page of *input_path*."""
    maps = page_glyph_maps(input_path, page_numbers, page_texts)
    wrapped = wrapped_headings(input_path, page_numbers, page_texts)
    return [
        repair_page_markdown(text, glyphs, pairs)
        for text, glyphs, pairs in zip(page_texts, maps, wrapped, strict=True)
    ]


def plain_headings(markdown: str) -> str:
    """Headings without tags, footnote marks or invisible glyphs.

    For Markdown a model wrote (the ``.llm.md``): only the headings' text
    changes, no heading becomes a paragraph. A heading left with nothing to
    read is dropped. Code blocks are left alone.
    """
    if "#" not in markdown:
        return markdown
    out = []
    for line in _lines(markdown, {}):
        if line.level <= 0:
            out.append(line.raw)
            continue
        if "<" not in line.raw and not any(_invisible(c) for c in line.raw):
            out.append(f"{'#' * line.level} {line.raw}")
            continue
        if any(c.isalnum() for c in line.text):
            out.append(f"{'#' * line.level} {heading_text(line.raw)}")
    return "\n".join(out)
