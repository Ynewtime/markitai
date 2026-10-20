"""Synthetic PDFs whose text layer is right, or present but wrong.

Every document is built here from made-up text, so the text-layer checks
(``converter.pdf_text_trust``) are tested on no real document. A wrong
layer is made the way real ones go wrong: the font draws one thing and its
``ToUnicode`` map says another. Helvetica (a base-14 font every PDF reader
has) draws the glyphs; the map sends its byte codes to private-use code
points, to a shifted alphabet, or to Hebrew and Arabic letters. CJK glyphs
come from PyMuPDF's built-in ``china-s`` font, whose codes are Unicode, so a
map can make them read as Latin letters.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pymupdf

LATIN = (
    "The committee met on Tuesday to review the annual budget and the "
    "plans for the new library."
)
# Distinct lines: the glyph check never samples the same text twice
LATIN_LINES = [
    LATIN,
    "Members asked how the opening hours would change over the summer.",
    "A short report on reading rooms was approved without any changes.",
    "Volunteers will help move the archive boxes during the first week.",
    "Questions about parking should be sent to the facilities office.",
    "The next meeting is planned for the second Monday of the month.",
]
CHINESE = "本报告介绍了城市公共图书馆的建设计划，以及明年预算的主要内容。"
JAPANESE = "この資料は新しい図書館の計画と来年度の予算について説明します。"
# Hebrew and Arabic in reading order (made-up sentences of common words)
HEBREW = "הספר החדש נמצא בבית הספר שלום לכל הילדים והמורים בעולם הגדול"
ARABIC = "الكتاب الجديد في المكتبة على الطاولة الكبيرة مع الطلاب في المدرسة"


def _to_unicode_cmap(codes: dict[int, str], width: int) -> bytes:
    """A ToUnicode CMap sending each code (*width* bytes) to its string."""
    digits = 2 * width
    lines = [
        "/CIDInit /ProcSet findresource begin",
        "12 dict begin",
        "begincmap",
        "/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def",
        "/CMapName /Synthetic def",
        "/CMapType 2 def",
        "1 begincodespacerange",
        f"<{0:0{digits}X}> <{(1 << (8 * width)) - 1:0{digits}X}>",
        "endcodespacerange",
    ]
    items = sorted(codes.items())
    for start in range(0, len(items), 100):  # at most 100 entries per block
        chunk = items[start : start + 100]
        lines.append(f"{len(chunk)} beginbfchar")
        for code, text in chunk:
            lines.append(
                f"<{code:0{digits}X}> <{text.encode('utf-16-be').hex().upper()}>"
            )
        lines.append("endbfchar")
    lines += [
        "endcmap",
        "CMapName currentdict /CMap defineresource pop",
        "end",
        "end",
    ]
    return "\n".join(lines).encode("ascii")


def _attach_to_unicode(doc: pymupdf.Document, font_xref: int, cmap: bytes) -> None:
    xref = doc.get_new_xref()
    doc.update_object(xref, "<< >>")
    doc.update_stream(xref, cmap)
    doc.xref_set_key(font_xref, "ToUnicode", f"{xref} 0 R")


def mapped_helvetica_pdf(
    path: Path,
    pages: Sequence[Sequence[str]],
    mapping: dict[str, str] | Sequence[dict[str, str]],
    *,
    fontsize: float = 12,
) -> Path:
    """Helvetica draws each line's ASCII characters; the text layer reads
    them through *mapping* (characters it leaves out read as themselves).

    One mapping: one font object for every page. A list of mappings, one per
    page: each page its own Helvetica, with its own map.
    """
    doc = pymupdf.open()
    fonts: dict[int, int] = {}  # id of a mapping -> its font xref

    def font_for(page_mapping: dict[str, str]) -> int:
        if id(page_mapping) not in fonts:
            xref = doc.get_new_xref()
            doc.update_object(
                xref,
                "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica "
                "/Encoding /WinAnsiEncoding >>",
            )
            codes = {code: chr(code) for code in range(32, 127)}
            codes.update({ord(char): text for char, text in page_mapping.items()})
            _attach_to_unicode(doc, xref, _to_unicode_cmap(codes, 1))
            fonts[id(page_mapping)] = xref
        return fonts[id(page_mapping)]

    for index, lines in enumerate(pages):
        page_mapping = mapping if isinstance(mapping, dict) else mapping[index]
        font_xref = font_for(page_mapping)
        page = doc.new_page()
        content = []
        y = page.rect.height - 72
        for line in lines:
            escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            content.append(f"BT /F1 {fontsize} Tf 72 {y:.1f} Td ({escaped}) Tj ET")
            y -= fontsize * 1.6
        xref = doc.get_new_xref()
        doc.update_object(xref, "<< >>")
        doc.update_stream(xref, "\n".join(content).encode("latin-1"))
        doc.xref_set_key(page.xref, "Contents", f"{xref} 0 R")
        doc.xref_set_key(
            page.xref, "Resources", f"<< /Font << /F1 {font_xref} 0 R >> >>"
        )
    doc.save(path)
    doc.close()
    return path


def rtl_pdf(path: Path, text: str, *, reversed_layer: bool, lines: int = 6) -> Path:
    """A page of right-to-left *text* (in reading order).

    MuPDF reads the letters of a right-to-left line from right to left. The
    glyphs are drawn left to right in visual order (the text reversed), so
    it gives the text back in reading order; with *reversed_layer* they are
    drawn in reading order instead, so it gives the text reversed, as a
    producer that does not order right-to-left text writes it.
    """
    letters = sorted(set(text) - {" "})
    codes = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    assert len(letters) <= len(codes)
    to_code = {letter: codes[i] for i, letter in enumerate(letters)}
    drawn = text if reversed_layer else text[::-1]
    line = "".join(to_code.get(char, char) for char in drawn)
    mapping = {to_code[letter]: letter for letter in letters}
    return mapped_helvetica_pdf(path, [[line] * lines], mapping)


def shifted_alphabet(shift: int = 7) -> dict[str, str]:
    """Each lowercase consonant reads as the consonant *shift* places on,
    each vowel as the next vowel: the vowel ratio of real text survives, so
    only the glyphs tell that ``bread`` reads ``jvoeq``."""
    vowels = "aeiou"
    consonants = "bcdfghjklmnpqrstvwxyz"
    mapping = {v: vowels[(i + 1) % 5] for i, v in enumerate(vowels)}
    mapping.update(
        {c: consonants[(i + shift) % len(consonants)] for i, c in enumerate(consonants)}
    )
    return mapping


def private_use_alphabet() -> dict[str, str]:
    """Every letter reads as a private-use code point."""
    return {
        char: chr(0xE000 + i)
        for i, char in enumerate("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    }


def cjk_glyphs_latin_layer(path: Path, text: str = CHINESE, *, lines: int = 6) -> Path:
    """CJK glyphs whose text layer reads as Latin letters, as a legacy font
    stores Latin codes for the script it draws."""
    doc = pymupdf.open()
    page = doc.new_page()
    for i in range(lines):
        page.insert_text((72, 90 + 26 * i), text, fontsize=14, fontname="china-s")
    font_xref = page.get_fonts()[0][0]
    letters = "etaoinshrdlucmfwypvbgkqjxz"
    codes = {ord(char): letters[i % 26] for i, char in enumerate(sorted(set(text)))}
    _attach_to_unicode(doc, font_xref, _to_unicode_cmap(codes, 2))
    doc.save(path)
    doc.close()
    return path


def text_pdf(path: Path, pages: Sequence[Sequence[tuple[str, str]]]) -> Path:
    """Correct pages: each line is ``(fontname, text)`` with a built-in
    PyMuPDF font (``helv``, ``cour``, ``china-s``, ``japan``)."""
    doc = pymupdf.open()
    for lines in pages:
        page = doc.new_page()
        y = 80.0
        for fontname, line in lines:
            page.insert_text((60, y), line, fontsize=11, fontname=fontname)
            y += 18
    doc.save(path)
    doc.close()
    return path


#: Correct pages of every kind the checks must leave alone, by name.
NORMAL_PAGES: dict[str, list[tuple[str, str]]] = {
    "english": [("helv", line) for line in LATIN_LINES],
    "chinese": [("china-s", CHINESE)] * 8,
    "japanese": [("japan", JAPANESE)] * 8,
    "mixed_cjk_code": [
        ("china-s", "使用以下命令安装依赖，然后运行测试："),
        ("cour", "uv sync --all-extras && uv run pytest -q"),
        ("cour", "def parse(path: str) -> dict[str, int]:"),
        ("cour", "    return {k: len(v) for k, v in load(path).items()}"),
        ("china-s", "返回值是一个字典，键为文件名，值为行数。"),
        ("helv", "See the README for the full list of options."),
    ],
    "table": [
        ("helv", "Region      Q1      Q2      Q3      Q4     Total"),
        ("helv", "North     1,204   1,388   1,502   1,611    5,705"),
        ("helv", "South       987   1,021   1,150   1,233    4,391"),
        ("helv", "East      2,301   2,250   2,410   2,598    9,559"),
        ("helv", "West        455     502     498     530    1,985"),
        ("helv", "Total     4,947   5,161   5,560   5,972   21,640"),
    ],
    "math": [
        ("helv", "Theorem 2. For every x > 0 and n >= 1 we have"),
        ("china-s", "∑ᵢ αᵢ xᵢ ≤ ∫₀¹ f(x) dx + ε, where ‖x‖₂ ≤ 1 and ∀ε > 0"),
        ("china-s", "f(x) = a₀ + a₁x + a₂x² + … + aₙxⁿ, ∂f/∂x = ∇f · e₁"),
        ("helv", "Proof. Let y = 2x + 3 and z = (y - 1) / 2; then z = x + 1."),
        ("china-s", "P(A ∩ B) = P(A) · P(B | A), Σ λₖ = tr(M), det(M) = Π λₖ"),
        ("helv", "This completes the proof of the inequality stated above."),
    ],
}
