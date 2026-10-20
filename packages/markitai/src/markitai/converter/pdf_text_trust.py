"""Whether a PDF page's text layer says what the page shows, and why not.

A digital PDF carries its text, and reading it beats OCR: it is exact, fast
and keeps the structure. But a text layer can be present and still wrong,
and nothing in the extraction notices:

- **A broken font map.** A font whose glyphs map to private-use, replacement
  or control characters (an icon font set as text, a subset font with no
  ``ToUnicode``), or to raw glyph codes written out as ``(cid:12)``.
- **Mojibake.** UTF-8 text that went through Windows-1252 on its way into
  the PDF (``cafÃ©``, ``â€œ``).
- **Right-to-left text in reverse order.** MuPDF already puts the letters of
  a right-to-left line stored in visual order back into reading order, so
  the storage order alone is no fault here; what is checked is the text it
  gives, by the letters that can only end a word (Hebrew final forms, Arabic
  ``ة`` and ``ى``) turning up at the start of words.
- **Latin consonant soup.** A substitution font, the long-standing check in
  :func:`markitai.ocr.is_likely_garbled`.
- **Text that is not what the glyphs draw** (with OCR, :class:`GlyphCheck`).
  A legacy or custom-encoded font stores other codes than the letters it
  draws: Latin codes for Devanagari glyphs, a shifted alphabet, letters for
  CJK glyphs. Only the image tells, so a few short stretches of text are
  cropped from the page and read by the OCR recognizer, then compared with
  the layer by writing system and by characters. A font that fails does so
  on every page, so each font is sampled a few times per document, not each
  page: the cost does not grow with the page count.

The text-only signals cost one ``get_text`` per page and run on every
conversion. The glyph check needs the OCR engine, so it runs with ``--ocr``,
where a page that fails is OCR'd instead of extracted.

The idea of checking a page's text layer against the script its line images
show comes from Indra OCR (datalab-to/indra, ``indra_ocr/textlayer.py`` and
``document.py``); this module is markitai's own implementation of it.
"""

from __future__ import annotations

import bisect
import re
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Any, NamedTuple

from loguru import logger

from markitai.ocr import is_likely_garbled

TRUSTED = "text layer trusted"
NO_TEXT_LAYER = "no text layer"
#: Prefix of every reason a text layer failed for.
FAILED = "text layer failed the script check: "


class LayerVerdict(NamedTuple):
    """What the checks say about one page's text layer.

    Attributes:
        status: ``trusted``; ``empty`` (no or almost no text, no picture
            covering the page); ``scanned`` (the same, over a page-sized
            picture); ``failed`` (text that is there but wrong).
        reason: Why, in a few words: ``text layer trusted``, ``no text
            layer``, ``text layer failed the script check: …``.
        script: The writing system the page shows, when the checks know it
            (a reversed Hebrew layer: ``Hebrew``); None when they do not.
            Whether OCR can read the page depends on it.
    """

    status: str
    reason: str
    script: str | None = None


def failed(detail: str, script: str | None = None) -> LayerVerdict:
    return LayerVerdict("failed", FAILED + detail, script)


# --- writing systems --------------------------------------------------------

# Code point ranges -> script, sorted by start. Han, kana and the CJK marks
# are one script here: one recognizer reads them, and a Japanese line mixes them.
_SCRIPT_RANGES: tuple[tuple[int, int, str], ...] = (
    (0x0370, 0x03FF, "Greek"),
    (0x0400, 0x052F, "Cyrillic"),
    (0x0530, 0x058F, "Armenian"),
    (0x0590, 0x05FF, "Hebrew"),
    (0x0600, 0x06FF, "Arabic"),
    (0x0700, 0x074F, "Syriac"),
    (0x0750, 0x077F, "Arabic"),
    (0x0780, 0x07BF, "Thaana"),
    (0x08A0, 0x08FF, "Arabic"),
    (0x0E00, 0x0E7F, "Thai"),
    (0x0E80, 0x0EFF, "Lao"),
    (0x10A0, 0x10FF, "Georgian"),
    (0x1100, 0x11FF, "Hangul"),
    (0x1E00, 0x1EFF, "Latin"),
    (0x1F00, 0x1FFF, "Greek"),
    (0x2E80, 0x2FDF, "CJK"),
    (0x3040, 0x30FF, "CJK"),
    (0x3130, 0x318F, "Hangul"),
    (0x31F0, 0x31FF, "CJK"),
    (0x3400, 0x4DBF, "CJK"),
    (0x4E00, 0x9FFF, "CJK"),
    (0xAC00, 0xD7AF, "Hangul"),
    (0xF900, 0xFAFF, "CJK"),
    (0xFB1D, 0xFB4F, "Hebrew"),
    (0xFB50, 0xFDFF, "Arabic"),
    (0xFE70, 0xFEFF, "Arabic"),
    (0xFF21, 0xFF3A, "Latin"),
    (0xFF41, 0xFF5A, "Latin"),
    (0xFF66, 0xFF9F, "CJK"),
    (0x20000, 0x3134F, "CJK"),
)
_SCRIPT_STARTS = [start for start, _end, _name in _SCRIPT_RANGES]


@lru_cache(maxsize=8192)
def char_script(char: str) -> str | None:
    """The writing system of a letter; None for anything else (digits,
    signs, math letters such as 𝑥, letters this module has no name for)."""
    if not char.isalpha():
        return None
    code = ord(char)
    if code < 0x2B0:  # ASCII, Latin-1, Latin Extended-A/B, IPA
        return "Latin"
    index = bisect.bisect_right(_SCRIPT_STARTS, code) - 1
    if index >= 0:
        start, end, name = _SCRIPT_RANGES[index]
        if start <= code <= end:
            return name
    if 0x0900 <= code <= 0x1CFF:  # the Indic and Southeast Asian blocks
        words = unicodedata.name(char, "").split()
        return words[0].title() if words else None
    return None


def script_counts(text: str) -> dict[str, int]:
    """Letters of each writing system in *text*."""
    counts: dict[str, int] = {}
    for char in text:
        script = char_script(char)
        if script is not None:
            counts[script] = counts.get(script, 0) + 1
    return counts


def dominant_script(text: str) -> str | None:
    """The writing system most letters of *text* belong to, or None."""
    counts = script_counts(text)
    return max(counts, key=lambda name: counts[name]) if counts else None


# --- text-only signals ------------------------------------------------------

# A broken font map: at least this many unreadable characters, and this
# share of the page's non-blank characters. Bullets and icons drawn from a
# symbol font are a few percent of a page at most.
_BAD_MIN_CHARS = 10
_BAD_MIN_SHARE = 0.2

# (cid:NN) glyph codes: at least this many, covering this share of the text.
_CID_RE = re.compile(r"\(cid:\d+\)")
_CID_MIN = 3
_CID_MIN_SHARE = 0.2

# Mojibake: at least this many words that are UTF-8 read as Windows-1252,
# and this share of the words that hold a non-ASCII character.
_MOJIBAKE_MIN_WORDS = 3
_MOJIBAKE_MIN_SHARE = 0.3

# Right-to-left text in reverse order: the right-to-left letters are at
# least this share of the page's letters, and reversed-order evidence beats
# reading-order evidence by this factor in at least this many words.
_RTL_MIN_SHARE = 0.2
_RTL_MIN_EVIDENCE = 3
_RTL_MARGIN = 2


def _unreadable(char: str) -> bool:
    """A character no font map should produce for text: replacement,
    private-use, unassigned, surrogate or control.

    The Microsoft symbol-font range (U+F020-U+F0FF, Symbol and Wingdings
    bytes) is left out: ``pdf_markdown.repair_pages`` maps those glyphs.
    """
    if char == "�":
        return True
    if 0xF020 <= ord(char) <= 0xF0FF:
        return False
    category = unicodedata.category(char)
    return category in ("Co", "Cn", "Cs") or (category == "Cc" and char not in "\t\r\n")


def _mojibake_words(words: Sequence[str]) -> tuple[int, int, str]:
    """(mojibake words, words holding a non-ASCII character, what the
    mojibake decodes to)."""
    found = 0
    candidates = 0
    decoded: list[str] = []
    for word in words:
        if word.isascii():
            continue
        candidates += 1
        for codec in ("cp1252", "latin-1"):
            try:
                fixed = word.encode(codec).decode("utf-8")
            except UnicodeError:
                continue
            if fixed != word:
                found += 1
                decoded.append(fixed)
            break
    return found, candidates, " ".join(decoded)


_HEBREW_WORD = re.compile(r"[א-ת]{2,}")
_HEBREW_FINALS = frozenset("ךםןףץ")
_HEBREW_NON_FINALS = frozenset("כמנפצ")
_ARABIC_WORD = re.compile(r"[ء-ي]{2,}")
_ARABIC_DIACRITICS = re.compile(r"[ً-ٰٟ]")
# Letters that only ever end an Arabic word: ta marbuta and alef maksura
_ARABIC_FINAL_ONLY = frozenset("ةى")


def _reversed_rtl(text: str, letters: int) -> str | None:
    """``Hebrew`` or ``Arabic`` when the text reads in reverse order."""
    if letters <= 0:
        return None
    normal = backwards = 0
    hebrew = _HEBREW_WORD.findall(text)
    rtl_letters = sum(len(word) for word in hebrew)
    for word in hebrew:
        normal += word[-1] in _HEBREW_FINALS
        backwards += (word[0] in _HEBREW_FINALS) + (word[-1] in _HEBREW_NON_FINALS)
    script = "Hebrew"
    arabic = _ARABIC_WORD.findall(
        _ARABIC_DIACRITICS.sub("", unicodedata.normalize("NFKC", text))
    )
    if sum(len(word) for word in arabic) > rtl_letters:
        script = "Arabic"
        rtl_letters = sum(len(word) for word in arabic)
        normal = backwards = 0
        for word in arabic:
            normal += word[-1] in _ARABIC_FINAL_ONLY
            normal += len(word) >= 4 and word.startswith("ال")
            backwards += word[0] in _ARABIC_FINAL_ONLY
            backwards += len(word) >= 4 and word.endswith("لا")
    if (
        rtl_letters >= _RTL_MIN_SHARE * letters
        and backwards >= _RTL_MIN_EVIDENCE
        and backwards > _RTL_MARGIN * normal
    ):
        return script
    return None


def judge_text(text: str) -> LayerVerdict:
    """The text-only verdict on a page's extracted text (enough of it to
    judge: the caller handles near-empty pages)."""
    chars = [char for char in text if not char.isspace()]
    total = max(len(chars), 1)

    bad = sum(1 for char in chars if _unreadable(char))
    if bad >= _BAD_MIN_CHARS and bad >= _BAD_MIN_SHARE * total:
        return failed(
            f"a broken font map ({bad / total:.0%} of the characters are "
            "private-use, replacement or control codes)"
        )

    cids = _CID_RE.findall(text)
    if len(cids) >= _CID_MIN and sum(map(len, cids)) >= _CID_MIN_SHARE * total:
        return failed(f"raw glyph codes instead of text ({len(cids)} (cid:NN) codes)")

    words = text.split()
    found, candidates, decoded = _mojibake_words(words)
    if found >= _MOJIBAKE_MIN_WORDS and found >= _MOJIBAKE_MIN_SHARE * candidates:
        return failed(
            f"mojibake, UTF-8 read as Windows-1252 ({found} words)",
            dominant_script(decoded),
        )

    letters = sum(1 for char in chars if char.isalpha())
    reversed_script = _reversed_rtl(text, letters)
    if reversed_script is not None:
        return failed("right-to-left text in reverse order", reversed_script)

    if is_likely_garbled(text):
        return failed("Latin letters with almost no vowels (a custom font encoding)")

    return LayerVerdict("trusted", TRUSTED, dominant_script(text))


# --- the glyph check (OCR) --------------------------------------------------

# Characters cut from a span for one sample, and the least letters (and
# share of letters) a sample needs to say anything.
_SAMPLE_CHARS = 20
_SAMPLE_MIN_LETTERS = 6
_SAMPLE_MIN_LETTER_SHARE = 0.6
# Samples per font per document, and crops per page at most.
_SAMPLES_PER_FONT = 2
_MAX_CROPS_PER_PAGE = 4
# The crop is scaled so its height is this many pixels: the recognizer's
# input height, so it is not scaled again.
_CROP_HEIGHT_PX = 48
# Below this similarity (difflib ratio over letters and digits) a sample
# does not say what the glyphs draw. OCR of clean text scores 0.9 and up,
# a shifted or permuted alphabet under 0.3.
_MIN_SIMILARITY = 0.5
# A page fails when fonts whose samples disagree carry this share of its
# letters.
_MAX_BAD_LETTER_SHARE = 0.3

_SUBSET_TAG = re.compile(r"^[A-Z]{6}\+")


def _font_name(name: str) -> str:
    """A font's name without its subset tag (``ABCDEF+Arial`` -> ``Arial``),
    as MuPDF names a span's font."""
    return _SUBSET_TAG.sub("", name or "") or "?"


def _comparable(text: str) -> str:
    """Letters and digits only, case-folded: what OCR and the layer should
    share (OCR reads punctuation and spacing in its own way)."""
    return "".join(
        c for c in unicodedata.normalize("NFKC", text).casefold() if c.isalnum()
    )


def compare_sample(layer: str, seen: str) -> tuple[bool | None, str]:
    """Does the text the layer gives for a crop match what OCR reads in it?

    Returns:
        ``(True, "")`` when it matches; ``(False, why)`` when it does not;
        ``(None, "")`` when OCR read too little to tell.
    """
    seen_letters = sum(1 for char in seen if char.isalpha())
    if seen_letters < 2:
        return None, ""
    layer_script = dominant_script(layer)
    seen_script = dominant_script(seen)
    if layer_script and seen_script and layer_script != seen_script:
        return False, f"the layer reads {layer_script}, the page shows {seen_script}"
    a, b = _comparable(layer), _comparable(seen)
    if (
        a
        and b
        and SequenceMatcher(None, a, b, autojunk=False).ratio() < _MIN_SIMILARITY
    ):
        return False, "the layer's text is not what its glyphs draw"
    return True, ""


def _map_fingerprint(doc: Any, xref: int, encoding: str) -> str:
    """A short hash of a font's ``ToUnicode`` map, or its encoding's name."""
    import hashlib

    try:
        kind, value = doc.xref_get_key(xref, "ToUnicode")
        if kind == "xref":
            stream = doc.xref_stream(int(value.split()[0])) or b""
            return hashlib.blake2b(stream, digest_size=6).hexdigest()
    except Exception as e:  # a damaged font dict: its name alone
        logger.debug("[PDF] ToUnicode of font {} not read: {}", xref, e)
    return encoding or "-"


@dataclass
class _FontTally:
    agree: int = 0
    disagree: int = 0
    why: str = ""
    seen: set[str] = field(default_factory=set)  # sampled windows

    @property
    def settled(self) -> bool:
        return self.agree + self.disagree >= _SAMPLES_PER_FONT

    @property
    def bad(self) -> bool:
        return self.disagree > self.agree


@dataclass
class _Sample:
    font: str
    text: str
    rect: tuple[float, float, float, float]


class GlyphCheck:
    """Compares a document's text layer with what its glyphs draw, per font.

    One instance serves one document, its pages in order. Each page's
    letters are counted per font; a font not yet sampled
    ``_SAMPLES_PER_FONT`` times gets up to that many short stretches of its
    text cropped from the page, rendered at the recognizer's line height,
    read by OCR (``read``, one batch per page) and compared
    (:func:`compare_sample`). A page fails when fonts whose samples mostly
    disagree carry ``_MAX_BAD_LETTER_SHARE`` of its letters.

    Never fatal: when OCR or rendering fails the check stops and every page
    keeps its text-only verdict.

    Args:
        read: Reads line images (RGB arrays) and returns their text, in order.
        readable: The writing systems ``read`` reads (``Latin``, ``CJK`` …):
            text in any other is not sampled.
    """

    def __init__(
        self, read: Callable[[list[Any]], list[str]], readable: frozenset[str]
    ) -> None:
        self._read = read
        self._readable = readable
        self._fonts: dict[str, _FontTally] = {}
        self._maps: dict[int, str] = {}  # font xref -> its map's fingerprint
        self._broken = False
        self.crops = 0  # line images read so far (for the debug log)

    def judge(self, page: Any) -> str | None:
        """Why *page*'s text layer does not match its glyphs, or None."""
        if self._broken or page.rotation:
            return None
        try:
            return self._judge(page)
        except Exception as e:  # never fails a conversion
            logger.debug("[PDF] Glyph check stopped: {}", e)
            self._broken = True
            return None

    def _judge(self, page: Any) -> str | None:
        import pymupdf

        letters: dict[str, int] = {}
        candidates: dict[str, list[_Sample]] = {}
        keys = self._font_keys(page)
        content = page.get_text("rawdict", flags=pymupdf.TEXTFLAGS_TEXT)
        for block in content.get("blocks", []):
            for line in block.get("lines", []):
                horizontal = line.get("wmode", 0) == 0 and abs(line["dir"][1]) < 1e-3
                for span in line.get("spans", []):
                    chars = span.get("chars", [])
                    count = sum(1 for c in chars if c["c"].isalpha())
                    if not count:
                        continue
                    name = str(span.get("font") or "")
                    font = keys.get(name, name)
                    letters[font] = letters.get(font, 0) + count
                    tally = self._fonts.setdefault(font, _FontTally())
                    if horizontal and line["dir"][0] > 0 and not tally.settled:
                        sample = self._sample(font, chars, page.rect)
                        if sample is not None and sample.text not in tally.seen:
                            candidates.setdefault(font, []).append(sample)

        picked: list[_Sample] = []
        for font, found in candidates.items():
            tally = self._fonts[font]
            need = _SAMPLES_PER_FONT - tally.agree - tally.disagree
            # Spread over the page: the first stretches, then the last
            step = max(1, len(found) // max(need, 1))
            for sample in found[::step] + found:
                if need <= 0:
                    break
                if sample.text in tally.seen:  # a running header, a repeat
                    continue
                tally.seen.add(sample.text)
                picked.append(sample)
                need -= 1
        picked = picked[:_MAX_CROPS_PER_PAGE]
        if picked:
            images = [self._render(page, sample.rect) for sample in picked]
            texts = self._read(images)
            self.crops += len(images)
            for sample, seen in zip(picked, texts, strict=False):
                agrees, why = compare_sample(sample.text, seen)
                tally = self._fonts[sample.font]
                if agrees is True:
                    tally.agree += 1
                elif agrees is False:
                    tally.disagree += 1
                    tally.why = why
                logger.debug(
                    "[PDF] Glyph check, font {}: layer {!r}, OCR {!r} -> {}",
                    sample.font,
                    sample.text,
                    seen,
                    "unsure" if agrees is None else ("match" if agrees else why),
                )

        total = sum(letters.values())
        bad = [font for font in letters if self._fonts[font].bad]
        bad_letters = sum(letters[font] for font in bad)
        if total and bad_letters >= _MAX_BAD_LETTER_SHARE * total:
            worst = max(bad, key=lambda font: letters[font])
            return f"{self._fonts[worst].why} (font {worst.split('#')[0]})"
        return None

    def _font_keys(self, page: Any) -> dict[str, str]:
        """Span font name -> the key its samples are counted under.

        MuPDF names a span's font without its subset tag, and two fonts of
        one name can map their glyphs differently (two subsets of a font,
        each with its own ``ToUnicode``). The key adds the map's
        fingerprint: the same map on every page is one font (sampled once
        per document, even when each page has its own copy of the font),
        another map is another font.
        """
        doc = page.parent
        maps: dict[str, set[str]] = {}
        for xref, _ext, _kind, basefont, _ref, encoding in page.get_fonts():
            fingerprint = self._maps.get(xref)
            if fingerprint is None:
                fingerprint = self._maps[xref] = _map_fingerprint(doc, xref, encoding)
            maps.setdefault(_font_name(basefont), set()).add(fingerprint)
        return {
            name: f"{name}#{'+'.join(sorted(found))}" for name, found in maps.items()
        }

    def _sample(
        self, font: str, chars: list[dict[str, Any]], page_rect: Any
    ) -> _Sample | None:
        """The first ``_SAMPLE_CHARS`` characters of a span, when they are
        mostly letters of a script OCR reads and sit on the page."""
        start = 0
        while start < len(chars) and chars[start]["c"].isspace():
            start += 1
        window = chars[start : start + _SAMPLE_CHARS]
        while window and window[-1]["c"].isspace():
            window.pop()
        text = "".join(c["c"] for c in window)
        visible = [c for c in text if not c.isspace()]
        count = sum(1 for c in visible if c.isalpha())
        if count < _SAMPLE_MIN_LETTERS or count < _SAMPLE_MIN_LETTER_SHARE * len(
            visible
        ):
            return None
        if dominant_script(text) not in self._readable:
            return None
        x0 = min(c["bbox"][0] for c in window)
        y0 = min(c["bbox"][1] for c in window)
        x1 = max(c["bbox"][2] for c in window)
        y1 = max(c["bbox"][3] for c in window)
        height = y1 - y0
        if height < 3 or x1 <= x0:
            return None
        pad_x, pad_y = 0.3 * height, 0.15 * height
        rect = (x0 - pad_x, y0 - pad_y, x1 + pad_x, y1 + pad_y)
        if (
            rect[0] < page_rect.x0
            or rect[1] < page_rect.y0
            or rect[2] > page_rect.x1
            or rect[3] > page_rect.y1
        ):
            return None
        return _Sample(font, text, rect)

    @staticmethod
    def _render(page: Any, rect: tuple[float, float, float, float]) -> Any:
        import numpy as np
        import pymupdf

        clip = pymupdf.Rect(rect)
        zoom = min(8.0, max(1.0, _CROP_HEIGHT_PX / clip.height))
        pix = page.get_pixmap(
            matrix=pymupdf.Matrix(zoom, zoom),
            clip=clip,
            colorspace=pymupdf.csRGB,
            alpha=False,
        )
        array = np.frombuffer(pix.samples, dtype=np.uint8)
        return array.reshape(pix.height, pix.stride // 3, 3)[:, : pix.width].copy()
