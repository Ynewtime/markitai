"""Page chrome and rendered-text pictures in pymupdf4llm's layout chunks.

The layout engine labels every box of a page (``page_boxes``: class, bbox and
the ``pos`` span of its Markdown in the chunk text). Two of its labels need a
second look before the text is used:

1. ``page-header`` / ``page-footer``. pymupdf4llm renders them into the body.
   One is dropped when it is a bare page number ("3", "- 3 -", "Page 3 of 9",
   "第 3 页") or when the same text (digits and a page number at either end
   ignored) is page chrome on at least two pages, and in both cases only when it sits in the top or bottom fifth of
   the page. A one-off header stays.
2. ``picture`` boxes that hold no picture. The model sometimes takes a bold or
   coloured line of text for a figure; pymupdf4llm then crops the page there
   into an image and adds the same words as "picture text". A picture box that
   overlaps no embedded image and is mostly covered by text-layer lines is that
   case: its Markdown becomes the text alone and the crop is not referenced.
   Real figures (an embedded image, or vector art with a few labels) stay.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

from loguru import logger

CHROME_CLASSES = frozenset({"page-header", "page-footer"})
#: A chrome box must lie in this top or bottom fraction of the page.
EDGE_BAND = 0.2
#: Text-layer lines must cover this share of a rendered-text picture box.
TEXT_COVERAGE = 0.5
#: An embedded image overlapping a picture box by this share of the smaller
#: of the two makes it a real picture.
IMAGE_OVERLAP = 0.1

PICTURE_TEXT = re.compile(
    r"(?:<!-- Start of picture text -->|\*\*----- Start of picture text -----\*\*<br>)"
    r"\s*(.*?)\s*"
    r"(?:<!-- End of picture text -->|\*\*----- End of picture text -----\*\*<br>)",
    re.DOTALL,
)
_IMAGE_REF = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MARKUP = re.compile(r"<[^>]+>|[*_`]")
_DIGITS = re.compile(r"\d+")
_SPACE = re.compile(r"\s+")
_EDGE_PUNCT = re.compile(r"^[\s\-–—|·•\[\](){}]+|[\s\-–—|·•\[\](){}]+$")
_NUMBER_EDGE = re.compile(r"^#\s+|\s+#$")
_PAGE_NUMBER = re.compile(
    r"^(?:(?:page|p\.?|pg\.?)\s*#(?:\s*(?:of|/)\s*#)?"
    r"|#(?:\s*(?:/|of)\s*#)?"
    r"|第\s*#\s*[页頁](?:\s*[,，、]?\s*共\s*#\s*[页頁])?"
    r"|#\s*[页頁])$"
)


def chrome_key(markdown: str) -> str:
    """A chrome box's text for matching across pages: digits are ``#``."""
    text = _SPACE.sub(" ", _MARKUP.sub("", markdown)).strip().lower()
    return _EDGE_PUNCT.sub("", _DIGITS.sub("#", text))


def is_page_number(key: str) -> bool:
    return bool(_PAGE_NUMBER.match(key))


def chrome_core(key: str) -> str:
    """The key without a page number at its start or end.

    The layout model sometimes joins a footer's title and its page number
    into one box on one page and keeps them apart on the next.
    """
    core = _NUMBER_EDGE.sub("", key)
    return core or key


def _boxes(chunk: dict[str, Any]) -> list[dict[str, Any]]:
    text = chunk.get("text")
    boxes = chunk.get("page_boxes")
    if not isinstance(text, str) or not isinstance(boxes, list):
        return []
    usable = []
    for box in boxes:
        pos = box.get("pos") if isinstance(box, dict) else None
        if (
            isinstance(pos, (tuple, list))
            and len(pos) == 2
            and all(isinstance(v, int) for v in pos)
            and 0 <= pos[0] < pos[1] <= len(text)
        ):
            usable.append(box)
    usable.sort(key=lambda box: box["pos"][0])
    return usable


def _area(rect: Any) -> float:
    return max(0.0, rect.x1 - rect.x0) * max(0.0, rect.y1 - rect.y0)


def _in_edge_band(bbox: Any, height: float) -> bool:
    return bbox.y1 <= height * EDGE_BAND or bbox.y0 >= height * (1 - EDGE_BAND)


def _is_rendered_text(page: Any, bbox: Any) -> bool:
    import pymupdf

    area = _area(bbox)
    if area <= 0:
        return False
    for info in page.get_image_info():
        image = pymupdf.Rect(info["bbox"])
        overlap = _area(image & bbox)
        if overlap > IMAGE_OVERLAP * min(area, _area(image) or area):
            return False
    covered = 0.0
    data = page.get_text("dict", clip=bbox)
    for block in data.get("blocks", []):
        for line in block.get("lines", []):
            if "".join(span.get("text", "") for span in line.get("spans", [])).strip():
                covered += _area(pymupdf.Rect(line["bbox"]) & bbox)
    return covered >= TEXT_COVERAGE * area


def picture_as_text(segment: str) -> str | None:
    """A picture block's Markdown (its image reference and the "picture
    text" pymupdf4llm read in it) as that text alone, one line per line;
    None when the block has no image reference or no text."""
    match = PICTURE_TEXT.search(segment)
    if match is None or _IMAGE_REF.search(segment) is None:
        return None
    lines = [line.strip() for line in match.group(1).split("<br>")]
    text = "\n".join(line for line in lines if line)
    return f"\n{text}\n\n" if text else None


def clean_layout_chunks(input_path: Path, chunks: list[Any]) -> list[Any]:
    """Drop repeated page chrome and un-picture rendered text, in place.

    Each changed chunk gets new ``text`` and its remaining ``page_boxes``
    get ``pos`` spans that match it; a box turned back into text is
    relabelled ``text``.
    """
    pages = [
        (index, chunk, _boxes(chunk))
        for index, chunk in enumerate(chunks)
        if isinstance(chunk, dict)
    ]
    if not any(boxes for _, _, boxes in pages):
        return chunks

    def segment(chunk: dict[str, Any], box: dict[str, Any]) -> str:
        start, end = box["pos"]
        return chunk["text"][start:end]

    chrome_pages: Counter[str] = Counter()
    for _, chunk, boxes in pages:
        keys = {
            chrome_core(chrome_key(segment(chunk, box)))
            for box in boxes
            if box.get("class") in CHROME_CLASSES
        }
        chrome_pages.update(key for key in keys if key)

    try:
        import pymupdf

        doc = pymupdf.open(input_path)
    except Exception as e:  # the conversion itself opened it; never fail here
        logger.debug("[PDF] Layout box check skipped for {}: {}", input_path, e)
        return chunks

    dropped: Counter[str] = Counter()
    unpictured = 0
    try:
        for index, chunk, boxes in pages:
            number = chunk.get("metadata", {}).get("page_number")
            page_index = (number if isinstance(number, int) else index + 1) - 1
            if not boxes or not 0 <= page_index < doc.page_count:
                continue
            page = doc[page_index]
            height = page.rect.height
            text = chunk["text"]
            parts: list[str] = []
            cursor = 0
            changed = False
            for box in boxes:
                start, end = box["pos"]
                if start < cursor:
                    continue
                piece = text[start:end]
                bbox = pymupdf.Rect(box.get("bbox"))
                replacement: str | None = None
                if box.get("class") in CHROME_CLASSES and _in_edge_band(bbox, height):
                    key = chrome_key(piece)
                    if key and (
                        is_page_number(key) or chrome_pages[chrome_core(key)] >= 2
                    ):
                        replacement = ""
                        dropped[key] += 1
                elif box.get("class") == "picture":
                    as_text = picture_as_text(piece)
                    if as_text is not None and _is_rendered_text(page, bbox):
                        replacement = as_text
                        box["class"] = "text"
                        unpictured += 1
                parts.append(text[cursor:start])
                new_start = sum(map(len, parts))
                if replacement is None:
                    parts.append(piece)
                else:
                    parts.append(replacement)
                    changed = True
                box["pos"] = (new_start, new_start + len(parts[-1]))
                cursor = end
            if changed:
                parts.append(text[cursor:])
                chunk["text"] = "".join(parts)
                chunk["page_boxes"] = [
                    box for box in boxes if box["pos"][0] < box["pos"][1]
                ]
    finally:
        doc.close()
    if dropped or unpictured:
        logger.debug(
            "[PDF] {}: dropped page chrome {}; {} rendered-text picture(s) kept as text",
            input_path.name,
            dict(dropped),
            unpictured,
        )
    return chunks
