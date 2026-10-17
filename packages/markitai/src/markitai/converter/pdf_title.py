"""A PDF's title: its own (Info) title, else its most prominent text.

Seen in practice: a policy PDF with no Info title was named after
its first Markdown heading, ``币值单位：人民币元``, a 10 pt bold caption the
layout analysis made a heading, while the page shows ``电子保险单`` at 20 pt
at its top. The title is what a reader sees as the title:

1. The Info dictionary's title when it names the document
   (``utils.titles.meaningful_title``: not ``Untitled``, ``Document1``,
   ``Microsoft Word - x.docx``, a file name or a number).
2. The largest visible text of the first pages, at least 1.2 times the body
   text's size (the size most characters have), on the earliest page that
   has one: a line, with the lines right under it at the same size (a title
   over two lines). Hidden text, superscript marks and a candidate that is
   no name (``2025``, a page number, an id) are passed over for the next
   one. Text repeated on every page is not: a policy prints its title atop
   each page (and a running header's words recur in the body, ``Lorem
   ipsum``), so a large company name in a header can win.
3. Nothing: the caller falls back to the first heading that names the
   document, then to the file name.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from markitai.utils.titles import usable_title

#: Pages read for the most prominent text
TITLE_PAGES = 3
#: How much larger than the body text a title is, at least
_MIN_RATIO = 1.2
# A line of a title is at most this long; more is a paragraph
_MAX_CHARS = 200
_MAX_LINES = 3


def _lines(page: Any) -> list[tuple[float, float, float, str]]:
    """``(size, top, bottom, text)`` of each visible line of *page*.

    A line's size is its largest span's; superscript spans (a footnote
    mark) and spans much smaller than that are left out of its text.
    """
    from markitai.converter.pdf import _background_sampler, _is_hidden_span

    page_rect = tuple(page.rect)
    background = _background_sampler(page)
    lines: list[tuple[float, float, float, str]] = []
    data = page.get_text("dict", flags=0)
    for block in data.get("blocks", []):
        for line in block.get("lines", []):
            spans = [
                span
                for span in line.get("spans", [])
                if str(span.get("text", "")).strip()
                and not int(span.get("flags", 0)) & 1  # superscript
                and not _is_hidden_span(span, page_rect, background)
            ]
            if not spans:
                continue
            size = max(float(span.get("size", 0)) for span in spans)
            text = "".join(
                str(span["text"])
                for span in spans
                if float(span.get("size", 0)) >= 0.7 * size
            )
            box = line.get("bbox") or (0, 0, 0, 0)
            lines.append((round(size, 1), float(box[1]), float(box[3]), text))
    return lines


def _join(parts: list[str]) -> str:
    """Lines of one title, with a space unless both sides are CJK."""
    text = ""
    for part in (p.strip() for p in parts):
        if text and not (_wide(text[-1]) and _wide(part[:1])):
            text += " "
        text += part
    return text


def _wide(char: str) -> bool:
    return bool(char) and ord(char) >= 0x2E80


def prominent_title(doc: Any, pages: int = TITLE_PAGES) -> str:
    """The most prominent text of the first *pages* that names the document.

    ``""`` when no text stands out from the body (a letter, a scan), or none
    of what stands out is a name.
    """
    read = range(min(pages, doc.page_count))
    per_page = [_lines(doc[number]) for number in read]
    weight: dict[float, int] = {}
    for lines in per_page:
        for size, _top, _bottom, text in lines:
            weight[size] = weight.get(size, 0) + len(text.strip())
    if not weight:
        return ""
    body = max(weight, key=lambda size: (weight[size], -size))
    for lines in per_page:
        lines = sorted(lines, key=lambda line: (line[1], line[2]))
        for size in sorted({line[0] for line in lines}, reverse=True):
            if size < body * _MIN_RATIO:
                break
            for index, (line_size, _top, bottom, text) in enumerate(lines):
                if line_size != size:
                    continue
                parts, last = [text], bottom
                for next_size, next_top, next_bottom, next_text in lines[index + 1 :]:
                    if len(parts) == _MAX_LINES:
                        break
                    if next_size != size:
                        if next_top < last:  # beside, on the same line
                            continue
                        break
                    if next_top - last > 0.8 * size:
                        break
                    parts.append(next_text)
                    last = next_bottom
                joined = _join(parts)
                if len(joined) > _MAX_CHARS:
                    continue
                title = usable_title(joined)
                if title:
                    return title
    return ""


def document_title(doc: Any) -> str:
    """The PDF's title by the rules in this module's docstring, or ``""``."""
    try:
        info = (doc.metadata or {}).get("title")
    except Exception:
        info = None
    title = usable_title(info, metadata=True) if isinstance(info, str) else ""
    if title:
        return title
    try:
        return prominent_title(doc)
    except Exception as e:  # a page that does not parse names nothing
        logger.debug("[PDF] Title from the page text failed: {}", e)
        return ""
