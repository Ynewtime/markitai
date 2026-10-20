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

The checks cost one ``get_text`` per page and run on every conversion.

The idea of checking a page's text layer against the script its line images
show comes from Indra OCR (datalab-to/indra, ``indra_ocr/textlayer.py`` and
``document.py``); this module is markitai's own implementation of it.
"""

from __future__ import annotations

import bisect
import re
import unicodedata
from collections.abc import Sequence
from functools import lru_cache
from typing import NamedTuple

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
