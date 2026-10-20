"""The text-layer trust check (``converter.pdf_text_trust``).

Every PDF here is synthetic (``tests.text_layer_pdfs``): a correct page of
each kind the existing pipelines handle (English, Chinese, Japanese, CJK
mixed with code, a table, math), and layers that are present but wrong (a
private-use font map, ``(cid:NN)`` codes, mojibake, reversed Hebrew and
Arabic, a shifted alphabet, CJK glyphs mapped to Latin letters).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pymupdf
import pytest

from markitai.converter.pdf import PdfConverter, collect_page_verdicts
from markitai.converter.pdf_text_trust import (
    FAILED,
    NO_TEXT_LAYER,
    TRUSTED,
    char_script,
    dominant_script,
    judge_text,
)
from markitai.notices import capture_task_notices
from markitai.ocr import ocr_lang_for_script, ocr_readable_scripts
from tests import text_layer_pdfs as fixtures


def _verdict(path: Path, page: int = 1):
    with pymupdf.open(path) as doc:
        return collect_page_verdicts(doc)[page]


class TestScripts:
    def test_letters_by_writing_system(self) -> None:
        assert char_script("a") == "Latin"
        assert char_script("é") == "Latin"
        assert char_script("中") == "CJK"
        assert char_script("の") == "CJK"
        assert char_script("한") == "Hangul"
        assert char_script("ש") == "Hebrew"
        assert char_script("ب") == "Arabic"
        assert char_script("ж") == "Cyrillic"
        assert char_script("क") == "Devanagari"
        assert char_script("1") is None
        assert char_script("∑") is None

    def test_dominant_script(self) -> None:
        assert dominant_script("使用 pymupdf 提取文本的方法") == "CJK"
        assert dominant_script("Run uv sync 然后") == "Latin"
        assert dominant_script("12 + 3") is None

    def test_what_each_ocr_language_reads(self) -> None:
        assert ocr_readable_scripts("en") == {"Latin", "CJK"}
        assert ocr_readable_scripts("zh") == {"Latin", "CJK"}
        assert ocr_readable_scripts("ar") == {"Latin", "Arabic"}
        assert ocr_readable_scripts("latin") == {"Latin"}
        assert ocr_readable_scripts("no-such-language") == {"Latin", "CJK"}
        assert ocr_lang_for_script("Arabic") == "ar"
        assert ocr_lang_for_script("Hebrew") is None


class TestTextOnlySignals:
    """``judge_text`` on extracted text."""

    def test_private_use_font_map(self) -> None:
        text = "".join(chr(0xE000 + i % 40) for i in range(200))
        verdict = judge_text(text)
        assert verdict.status == "failed"
        assert verdict.reason.startswith(FAILED + "a broken font map")
        assert verdict.script is None  # OCR reads whatever the page shows

    def test_replacement_characters(self) -> None:
        assert judge_text("Report ��� " * 20).status == "failed"

    def test_symbol_font_bullets_are_no_broken_map(self) -> None:
        """Wingdings/Symbol bytes (U+F0xx) are mapped later by repair_pages."""
        text = "\n".join(f" {fixtures.LATIN}" for _ in range(10))
        assert judge_text(text).status == "trusted"

    def test_a_few_icons_on_a_page_are_fine(self) -> None:
        text = f"{fixtures.LATIN}   {fixtures.LATIN} "
        assert judge_text(text).status == "trusted"

    def test_cid_glyph_codes(self) -> None:
        text = "Summary " + "(cid:12)(cid:45)(cid:3) " * 10
        verdict = judge_text(text)
        assert verdict.status == "failed"
        assert "(cid:NN)" in verdict.reason

    def test_a_document_about_cid_codes_is_fine(self) -> None:
        text = f"{fixtures.LATIN} pdfminer writes (cid:12) for an unmapped glyph."
        assert judge_text(text * 3).status == "trusted"

    def test_mojibake(self) -> None:
        text = "The cafÃ© served crÃ¨me brÃ»lÃ©e and a â€œspecialâ€ rÃ©sumÃ©. " * 3
        verdict = judge_text(text)
        assert verdict.status == "failed"
        assert "mojibake" in verdict.reason
        assert verdict.script == "Latin"

    def test_mojibake_of_chinese_reads_as_chinese(self) -> None:
        broken = "中文内容说明".encode().decode("cp1252", errors="replace")
        verdict = judge_text(f"{broken} {broken} {broken} report")
        assert verdict.status == "failed"
        assert verdict.script == "CJK"

    def test_accented_text_is_no_mojibake(self) -> None:
        text = "Le café servait une crème brûlée et un résumé très soigné. " * 3
        assert judge_text(text).status == "trusted"

    @pytest.mark.parametrize(
        "text", [fixtures.HEBREW, fixtures.ARABIC], ids=["hebrew", "arabic"]
    )
    def test_right_to_left_in_reading_order(self, text: str) -> None:
        assert judge_text(text).status == "trusted"

    @pytest.mark.parametrize(
        ("text", "script"),
        [(fixtures.HEBREW, "Hebrew"), (fixtures.ARABIC, "Arabic")],
        ids=["hebrew", "arabic"],
    )
    def test_right_to_left_in_reverse_order(self, text: str, script: str) -> None:
        verdict = judge_text(text[::-1])
        assert verdict.status == "failed"
        assert verdict.reason == FAILED + "right-to-left text in reverse order"
        assert verdict.script == script

    def test_one_reversed_quote_in_an_english_page_is_left_alone(self) -> None:
        text = fixtures.LATIN * 4 + " " + fixtures.HEBREW[:20][::-1]
        assert judge_text(text).status == "trusted"

    def test_consonant_soup(self) -> None:
        verdict = judge_text("bcdfghjklm npqrstvwxz " * 4)
        assert verdict.status == "failed"
        assert "vowels" in verdict.reason


class TestPageVerdicts:
    """``collect_page_verdicts`` on synthetic PDFs."""

    @pytest.mark.parametrize("kind", sorted(fixtures.NORMAL_PAGES))
    def test_correct_pages_are_trusted(self, tmp_path: Path, kind: str) -> None:
        path = fixtures.text_pdf(
            tmp_path / f"{kind}.pdf", [fixtures.NORMAL_PAGES[kind]]
        )
        assert _verdict(path) == (
            "trusted",
            TRUSTED,
            "CJK" if kind in ("chinese", "japanese") else "Latin",
        )

    def test_private_use_font(self, tmp_path: Path) -> None:
        path = fixtures.mapped_helvetica_pdf(
            tmp_path / "pua.pdf",
            [[fixtures.LATIN] * 6],
            fixtures.private_use_alphabet(),
        )
        assert _verdict(path).status == "failed"

    @pytest.mark.parametrize(
        "text", [fixtures.HEBREW, fixtures.ARABIC], ids=["hebrew", "arabic"]
    )
    def test_right_to_left_pages(self, tmp_path: Path, text: str) -> None:
        good = fixtures.rtl_pdf(tmp_path / "good.pdf", text, reversed_layer=False)
        bad = fixtures.rtl_pdf(tmp_path / "bad.pdf", text, reversed_layer=True)
        assert _verdict(good).status == "trusted"
        assert _verdict(bad).status == "failed"

    def test_mojibake_and_cid_pages(self, tmp_path: Path) -> None:
        mojibake = "The cafÃ© served crÃ¨me brÃ»lÃ©e and a â€œspecialâ€ dish."
        cid = "(cid:41)(cid:72)(cid:79)(cid:79)(cid:82) (cid:90)(cid:82)(cid:85)"
        path = fixtures.text_pdf(
            tmp_path / "codes.pdf", [[("helv", mojibake)] * 4, [("helv", cid)] * 4]
        )
        with pymupdf.open(path) as doc:
            verdicts = collect_page_verdicts(doc)
        assert "mojibake" in verdicts[1].reason
        assert "(cid:NN)" in verdicts[2].reason

    def test_pages_without_text(self, tmp_path: Path) -> None:
        doc = pymupdf.open()
        doc.new_page()
        doc.new_page().insert_text((72, 72), "Chapter 1")
        doc.save(tmp_path / "blank.pdf")
        doc.close()
        with pymupdf.open(tmp_path / "blank.pdf") as doc:
            verdicts = collect_page_verdicts(doc)
        assert verdicts[1] == ("empty", NO_TEXT_LAYER, None)
        assert verdicts[2].status == "empty"
        assert verdicts[2].reason == "almost no text layer (9 characters)"


class TestNotices:
    """What the standard (no --ocr) conversion tells the user."""

    @staticmethod
    def _convert(path: Path, lang: str = "en") -> list[str]:
        from markitai.config import MarkitaiConfig

        config = MarkitaiConfig()
        config.ocr.lang = lang
        with (
            patch("markitai.converter.pdf.pymupdf4llm") as extract,
            capture_task_notices() as notices,
        ):
            extract.to_markdown.return_value = [{"text": "text"}]
            PdfConverter(config).convert(path)
        return list(notices)

    def test_a_broken_font_map_asks_for_ocr(self, tmp_path: Path) -> None:
        path = fixtures.mapped_helvetica_pdf(
            tmp_path / "pua.pdf",
            [[fixtures.LATIN] * 6, [fixtures.LATIN] * 6],
            {"z": "z"},  # page 1 correct
        )
        bad = fixtures.mapped_helvetica_pdf(
            tmp_path / "bad.pdf",
            [[fixtures.LATIN] * 6],
            fixtures.private_use_alphabet(),
        )
        assert self._convert(path) == []
        notices = self._convert(bad)
        assert len(notices) == 1
        assert notices[0].startswith(
            "[PDF] bad.pdf: 1 of 1 pages look scanned/garbled (page 1); "
            "consider re-running with --ocr"
        )

    def test_reversed_hebrew_says_ocr_would_not_help(self, tmp_path: Path) -> None:
        path = fixtures.rtl_pdf(
            tmp_path / "he.pdf", fixtures.HEBREW, reversed_layer=True
        )
        assert self._convert(path) == [
            "[PDF] he.pdf: the text layer of 1 of 1 pages reads wrong (page 1): "
            "right-to-left text in reverse order; no OCR language reads Hebrew, "
            "so it is kept as extracted"
        ]

    def test_reversed_arabic_names_the_ocr_language(self, tmp_path: Path) -> None:
        path = fixtures.rtl_pdf(
            tmp_path / "ar.pdf", fixtures.ARABIC, reversed_layer=True
        )
        assert self._convert(path) == [
            "[PDF] ar.pdf: the text layer of 1 of 1 pages reads wrong (page 1): "
            "right-to-left text in reverse order; consider re-running with --ocr "
            "and ocr.lang set to ar"
        ]
        # With ocr.lang ar, --ocr reads it: the usual advice
        notices = self._convert(path, lang="ar")
        assert len(notices) == 1
        assert (
            "look scanned/garbled (page 1); consider re-running with --ocr"
            in (notices[0])
        )

    def test_correct_pages_raise_no_notice(self, tmp_path: Path) -> None:
        path = fixtures.text_pdf(
            tmp_path / "all.pdf", list(fixtures.NORMAL_PAGES.values())
        )
        assert self._convert(path) == []
        good = fixtures.rtl_pdf(
            tmp_path / "he.pdf", fixtures.HEBREW, reversed_layer=False
        )
        assert self._convert(good) == []
