"""PDF headings are plain text, symbol-font glyphs are no boxes.

Seen in practice: the workbench's table of contents for a Chinese
policy listed ``▯▯<sup>…</sup>`` (two Wingdings glyphs, then text PyMuPDF
flagged superscript beside a 26 pt icon) and bold or underlined sentences
the layout model had made headings. The PDFs here are built with invented
text: a symbol font is a non-embedded TrueType font whose ToUnicode CMap
maps each code to ``U+F000`` plus the code, as Word's Symbol and Wingdings
fonts come out of a PDF.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from markitai.config import MarkitaiConfig
from markitai.converter.pdf import PdfConverter
from markitai.converter.pdf_markdown import (
    page_glyph_maps,
    plain_headings,
    reads_as_field,
    reads_as_sentence,
    repair_page_markdown,
    text_words,
)

CJK = "china-s"
BODY = "本服务说明列明服务内容、服务期限及收费方式。会员应仔细阅读各项约定，特别是加粗标示的内容"
HAND = ""  # Wingdings 2 'C': a pointing hand
SPACE = ""
DIAMOND = ""  # Wingdings 'v': a diamond of four
MINUS = ""  # Symbol '-': the minus sign
GLYPHS = {
    HAND: ("☞", True),
    SPACE: (" ", True),
    DIAMOND: ("❖", True),
    MINUS: ("−", False),
}


# --- PDFs ------------------------------------------------------------------


def _symbol_font(doc: pymupdf.Document, name: str, unicode: int = 0xF020) -> int:
    """A non-embedded TrueType font *name*; code c reads as unicode + c - 0x20.

    ``unicode=0xFFFF`` maps every code to U+FFFF (iText's ``.notdef``).
    """
    if unicode == 0xFFFF:
        ranges = "<20> <FF> [" + " ".join(["<FFFF>"] * 224) + "]"
    else:
        ranges = f"<20> <FF> <{unicode:04X}>"
    cmap = (
        "/CIDInit /ProcSet findresource begin 12 dict begin begincmap "
        "/CMapName /Symbolic def 1 begincodespacerange <00> <FF> "
        f"endcodespacerange 1 beginbfrange {ranges} endbfrange endcmap "
        "CMapName currentdict /CMap defineresource pop end end"
    )
    to_unicode = doc.get_new_xref()
    doc.update_object(to_unicode, "<<>>")
    doc.update_stream(to_unicode, cmap.encode())
    descriptor = doc.get_new_xref()
    doc.update_object(
        descriptor,
        f"<< /Type /FontDescriptor /FontName /{name} /Flags 4 "
        "/FontBBox [0 -200 1000 800] /ItalicAngle 0 /Ascent 800 "
        "/Descent -200 /CapHeight 700 /StemV 80 >>",
    )
    font = doc.get_new_xref()
    widths = " ".join(["900"] * 224)
    doc.update_object(
        font,
        f"<< /Type /Font /Subtype /TrueType /BaseFont /{name} /FirstChar 32 "
        f"/LastChar 255 /Widths [{widths}] /FontDescriptor {descriptor} 0 R "
        f"/ToUnicode {to_unicode} 0 R >>",
    )
    return font


class _Page:
    """Text runs written in content order, so runs on one baseline share a line.

    ``insert_text`` gives each call its own content stream and rewrites the
    page's font dictionary, so runs are queued and written last, the CJK
    ones in pymupdf's ``china-s`` font (UTF-16BE codes).
    """

    def __init__(self, doc: pymupdf.Document) -> None:
        self.doc = doc
        self.page = doc.new_page()
        self.runs: list[tuple[str | int, bytes, float, float, float]] = []

    def cjk(self, text: str, x: float, y: float, size: float = 10.4) -> None:
        self.runs.append((CJK, text.encode("utf-16-be"), x, y, size))

    def glyphs(self, font: int, codes: bytes, x: float, y: float, size: float) -> None:
        self.runs.append((font, codes, x, y, size))

    def body(self, y: float, lines: int = 4) -> float:
        for i in range(lines):
            line = BODY[:44]
            self.page.insert_text((60, y + i * 16), line, fontname=CJK, fontsize=10.4)
        return y + lines * 16

    def _add_font(self, tag: str, font: int) -> None:
        """/Resources/Font/<tag>, either dictionary being direct or indirect."""
        doc = self.doc
        holder, key = self.page.xref, "Resources/Font"
        kind, ref = doc.xref_get_key(holder, "Resources")
        if kind == "xref":
            holder, key = int(ref.split()[0]), "Font"
        kind, ref = doc.xref_get_key(holder, key)
        if kind == "xref":
            doc.xref_set_key(int(ref.split()[0]), tag, f"{font} 0 R")
        else:
            doc.xref_set_key(holder, f"{key}/{tag}", f"{font} 0 R")

    def finish(self) -> None:
        doc, page = self.doc, self.page
        ops = []
        for font, data, x, y, size in self.runs:
            tag = font if isinstance(font, str) else f"G{font}"
            if isinstance(font, int):
                self._add_font(tag, font)
            top = page.rect.height - y
            ops.append(f"BT /{tag} {size} Tf 1 0 0 1 {x} {top} Tm <{data.hex()}> Tj ET")
        stream = doc.get_new_xref()
        doc.update_object(stream, "<<>>")
        doc.update_stream(stream, "\n".join(ops).encode())
        contents = " ".join(f"{x} 0 R" for x in [*page.get_contents(), stream])
        doc.xref_set_key(page.xref, "Contents", f"[{contents}]")


def _icon_headings(path: Path) -> Path:
    """Framed sections titled after a 26 pt Wingdings 2 hand, ToC lines after
    Wingdings diamonds: a real page's shape, in invented words."""
    doc = pymupdf.open()
    page = _Page(doc)
    page.page.insert_text((60, 40), "服务说明", fontname=CJK, fontsize=16)
    hand = _symbol_font(doc, "Wingdings2")
    diamond = _symbol_font(doc, "Wingdings-Regular")
    y = 90.0
    sections = {
        "会员享有的主要权利": [
            "会员可在七日内无理由退出服务",
            "服务内容以本说明列明的为准",
            "会员有随时查询服务记录的权利",
        ],
        "会员需要特别留意的事项": [
            "逾期未付费用时服务将暂停",
            "会员有如实提供资料的义务",
        ],
    }
    for title, items in sections.items():
        top = y - 30
        page.glyphs(hand, b"\x43\x20", 50, y, 26)
        page.cjk(title, 100, y - 4, 12)
        y += 34
        for n, item in enumerate(items, 1):
            page.glyphs(diamond, b"\x76", 50, y, 10.4)
            page.cjk(item + "…" * (30 - len(item)) + f"{n}.{n + 1}", 62, y)
            y += 16
        y += 10
        frame = pymupdf.Rect(44, top, 551, y)
        page.page.draw_rect(frame, color=(0, 0, 0), width=0.6, dashes="[2] 0")
        y = page.body(y + 24) + 40
    page.finish()
    doc.save(path)
    return path


def _sentence_headings(path: Path) -> Path:
    """Fields and sentences in an 11 pt bold face over 10.4 pt body text."""
    doc = pymupdf.open()
    page = _Page(doc)
    page.page.insert_text((240, 50), "服务确认书", fontname=CJK, fontsize=20)
    y = 90.0
    for line in ("计价单位：人民币元", "服务编号：A00000000001"):
        box = pymupdf.Rect(60, y, 540, y + 20)
        page.page.insert_htmlbox(box, f"<p style='font-size:11pt'><b>{line}</b></p>")
        y += 24
    y = page.body(y + 10, 5) + 16
    for line in (
        "会员已确认所提供的资料真实完整。",
        "如有隐瞒或资料不实，足以影响服务方决定是否提供服务的，服务方有权终止服务。",
    ):
        box = pymupdf.Rect(60, y, 540, y + 40)
        page.page.insert_htmlbox(box, f"<p style='font-size:11pt'><b>{line}</b></p>")
        y += 40
    y = page.body(y + 10, 5) + 16
    box = pymupdf.Rect(60, y, 540, y + 24)
    page.page.insert_htmlbox(box, "<p style='font-size:12pt'><b>特别说明</b></p>")
    page.body(y + 30, 6)
    doc.save(path)
    return path


def _symbol_minus(path: Path) -> Path:
    """A SymbolMT minus sign inside a CJK sentence."""
    doc = pymupdf.open()
    page = _Page(doc)
    page.page.insert_text((60, 80), "调整条件", fontname=CJK, fontsize=16)
    symbol = _symbol_font(doc, "SymbolMT")
    page.cjk("当年度使用率≥（行业平均使用率", 60, 120)
    page.glyphs(symbol, b"\x2d", 222, 120, 10.4)
    page.cjk(" 10%）时，服务方可调整价格。", 232, 120)
    page.body(150, 8)
    page.finish()
    doc.save(path)
    return path


def _notdef(path: Path) -> Path:
    """A line of glyphs the ToUnicode CMap maps to U+FFFF."""
    doc = pymupdf.open()
    page = _Page(doc)
    page.page.insert_text((60, 80), "服务期间", fontname=CJK, fontsize=16)
    font = _symbol_font(doc, "SimSun", 0xFFFF)
    page.glyphs(font, b"\x21" * 8, 60, 110, 10.4)
    page.body(130, 8)
    page.finish()
    doc.save(path)
    return path


def _convert(path: Path, tmp_path: Path) -> str:
    config = MarkitaiConfig()
    config.cache.enabled = False
    return PdfConverter(config).convert(path, tmp_path / "out").markdown


@pytest.fixture
def unrepaired(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Converts as before the repair: each PDF test first shows its shape."""

    def convert(path: Path) -> str:
        with monkeypatch.context() as patch:
            patch.setattr(
                "markitai.converter.pdf.repair_pages",
                lambda _path, _numbers, texts: list(texts),
            )
            return _convert(path, tmp_path / "unrepaired")

    return convert


def _headings(markdown: str) -> list[str]:
    return [line for line in markdown.splitlines() if line.startswith("#")]


def _private(text: str) -> list[str]:
    return [c for c in text if 0xE000 <= ord(c) <= 0xF8FF or c == "￿"]


# --- the rules -------------------------------------------------------------


class TestRules:
    """Rust pdf-inspector's classify.rs rules, on invented text."""

    def test_words(self) -> None:
        assert text_words("Annual Report 2025") == 3
        # A CJK run counts a word for every three characters
        assert text_words("会员服务说明") == 2
        long_sentence = (
            "本服务方运营状况良好各项指标满足监管要求会员可以登录官方网站"
            "查看最近季度运营报告和服务质量评价等相关信息"
        )
        assert text_words(long_sentence) > 15
        assert text_words("优享会员年度服务（标准版）说明") <= 15

    @pytest.mark.parametrize(
        "text",
        [
            "会员已确认所提供的资料真实完整。",
            "如有隐瞒或资料不实，足以影响服务方决定的，",
            "第一句。第二句",
            "本服务方运营状况良好，会员可以登录",
        ],
    )
    def test_sentences(self, text: str) -> None:
        assert reads_as_sentence(text)

    @pytest.mark.parametrize(
        "text",
        [
            "一、总则，范围",
            "服务质量告知和综合评价",
            "Results.",
            # Latin headings may be sentences (sample.pdf's Lorem ipsum)
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
        ],
    )
    def test_no_sentences(self, text: str) -> None:
        assert not reads_as_sentence(text)

    @pytest.mark.parametrize(
        "text",
        ["会员：示例会员 证件类型：护照", "首期费用合计：100.00", "计价单位：人民币元"],
    )
    def test_fields(self, text: str) -> None:
        assert reads_as_field(text)

    @pytest.mark.parametrize(
        "text",
        [
            "说明：",
            "Chapter 1: Introduction",
            "这是一个很长很长的不像标签的句子开头部分：值",
            # A section number is no label
            "第一章：总则",
            "3.2：服务期限",
        ],
    )
    def test_no_fields(self, text: str) -> None:
        assert not reads_as_field(text)


# --- page Markdown ---------------------------------------------------------


class TestHeadingText:
    def test_icon_glyphs_and_superscript_tags_are_dropped(self) -> None:
        # pymupdf4llm's line for that shape, in invented words
        page = f"#  {HAND}{SPACE}<sup>会员享有的主要权利</sup> \n\n正文。"
        expected = "# 会员享有的主要权利\n\n正文。"
        assert repair_page_markdown(page, GLYPHS) == expected
        # The glyphs go also when the page's fonts were not read
        assert repair_page_markdown(page) == expected

    def test_footnote_marks_go_emphasis_stays(self) -> None:
        assert (
            repair_page_markdown("# **Regional Climate Outlook**<sup>**1**</sup> ")
            == "# **Regional Climate Outlook**"
        )
        assert repair_page_markdown("## Results<sup>a</sup>") == "## Results"
        assert repair_page_markdown("## 20<sup>th</sup> century") == "## 20th century"

    def test_other_tags_keep_their_text(self) -> None:
        assert (
            repair_page_markdown("###### **<mark>办理须知</mark>**")
            == "###### **办理须知**"
        )
        assert repair_page_markdown("## <u>Scope</u>") == "## Scope"

    def test_code_spans_stay(self) -> None:
        assert (
            repair_page_markdown("## The `<sup>` element") == "## The `<sup>` element"
        )

    def test_a_symbol_fonts_sign_is_mapped(self) -> None:
        glyphs = {"": ("α", False)}
        assert repair_page_markdown("## The  helix", glyphs) == "## The α helix"

    def test_a_heading_of_glyphs_only_goes(self) -> None:
        assert repair_page_markdown(f"# {HAND}<sup></sup>\n\nText", GLYPHS) == (
            "\nText"
        )

    def test_headings_inside_code_blocks_are_left(self) -> None:
        page = "```\n# <sup>x</sup> \n```"
        assert repair_page_markdown(page, GLYPHS) == page


class TestNoHeadings:
    def test_sentences_are_paragraphs_with_their_styling(self) -> None:
        page = (
            "###### **会员已确认所提供的资料真实完整。**\n\n"
            "###### **如有隐瞒或资料不实，足以影响服务方决定是否提供服务的，"
            "服务方有权终止服务。**\n\n"
            "###### **<mark>特别说明</mark>**"
        )
        assert repair_page_markdown(page) == (
            "**会员已确认所提供的资料真实完整。**\n\n"
            "**如有隐瞒或资料不实，足以影响服务方决定是否提供服务的，"
            "服务方有权终止服务。**\n\n"
            "###### **特别说明**"
        )

    def test_fields_are_paragraphs(self) -> None:
        page = "###### **计价单位：人民币元**\n\n### 服务编号： **A00000000001**"
        assert repair_page_markdown(page) == (
            "**计价单位：人民币元**\n\n服务编号： **A00000000001**"
        )
        assert repair_page_markdown("## 第一章：总则") == "## 第一章：总则"

    def test_long_lines_are_paragraphs(self) -> None:
        page = (
            "## 优享服务有限公司 请扫描以核对服务说明 优享会员年度服务A 款"
            "互联网专属健康管理服务（标准版）说明"
        )
        assert not repair_page_markdown(page).startswith("#")

    def test_contents_entries_are_paragraphs(self) -> None:
        assert repair_page_markdown("## 服务期限…………………………3.2") == (
            "服务期限…………………………3.2"
        )

    def test_a_sentence_leading_into_a_list(self) -> None:
        page = "###### 本服务方承担费用补偿责任的项目须同时符合下列全部要求："
        assert repair_page_markdown(page) == (
            "本服务方承担费用补偿责任的项目须同时符合下列全部要求："
        )
        assert repair_page_markdown("## 服务内容：") == "## 服务内容："

    def test_latin_headings_stay(self) -> None:
        page = "## **Lorem ipsum dolor sit amet, consectetur adipiscing elit.**"
        assert repair_page_markdown(page) == page


class TestSplitLines:
    # A clause's line, 41 characters, ran to the column's edge
    LINE = (
        "本服务方有权获得并使用会员提交的全部检测样本以及能够对这些样本进行独立复核的权"
    )

    def test_a_section_number_takes_the_title_after_it(self) -> None:
        assert repair_page_markdown("###### 2.4\n\n###### 服务范围\n\n正文") == (
            "###### 2.4 服务范围\n\n正文"
        )

    def test_a_sentence_over_two_heading_lines(self) -> None:
        page = f"###### <u>{self.LINE}</u>\n\n###### <u>利。</u>\n\n下一段。"
        assert repair_page_markdown(page) == (
            f"<u>{self.LINE}</u><u>利。</u>\n\n下一段。"
        )

    def test_a_heading_line_then_its_tail_in_the_body(self) -> None:
        page = f"- 3) 上一项。\n\n###### <u>{self.LINE}</u>\n\n<u>利。</u>\n\n下一段。"
        assert repair_page_markdown(page) == (
            f"- 3) 上一项。\n\n<u>{self.LINE}</u><u>利。</u>\n\n下一段。"
        )

    def test_a_tail_set_as_a_heading_joins_its_paragraph(self) -> None:
        page = f"<u>{self.LINE}</u>\n\n###### <u>利。</u>\n\n下一段。"
        assert repair_page_markdown(page) == (
            f"<u>{self.LINE}</u><u>利。</u>\n\n下一段。"
        )

    def test_a_long_title_keeps_the_sentence_after_it_apart(self) -> None:
        title = "第一部分 优享会员年度专属健康管理与家庭医生签约服务内容及收费标准说明"
        page = f"## {title}\n\n## 本说明适用于全部会员。"
        assert repair_page_markdown(page) == f"## {title}\n\n本说明适用于全部会员。"
        page = f"## {title}\n\n本说明适用于全部会员及其家属，自签约之日起生效。"
        assert repair_page_markdown(page) == page


class TestBodyGlyphs:
    def test_glyphs_are_mapped_by_their_font(self) -> None:
        page = (
            f"- 服务内容…………2.5 {DIAMOND} 退出服务的权利…………3.3\n\n"
            f"使用率≥（平均使用率 {MINUS} 10%）时"
        )
        assert repair_page_markdown(page, GLYPHS) == (
            "- 服务内容…………2.5 ❖ 退出服务的权利…………3.3\n\n使用率≥（平均使用率 − 10%）时"
        )

    def test_a_list_items_dingbat_was_its_bullet(self) -> None:
        page = f"- {HAND}{SPACE}<sup>条款中凡以</sup><sup><u>下划线</u></sup> <u>标示的内容</u>"
        assert repair_page_markdown(page, GLYPHS) == (
            "- 条款中凡以<u>下划线</u> <u>标示的内容</u>"
        )

    def test_unknown_glyphs_and_noncharacters_are_dropped(self) -> None:
        assert repair_page_markdown("AB￿C\x07D") == "ABCD"
        assert repair_page_markdown("前文\n\n￿￿ \n\n后文") == "前文\n\n\n\n后文"

    def test_superscripts_stay_unless_they_are_han_text(self) -> None:
        page = "E = mc<sup>2</sup>，费率<sup>**_7.4_**</sup>，<sup>注</sup>，<sup>旨在帮助</sup>"
        assert repair_page_markdown(page) == (
            "E = mc<sup>2</sup>，费率<sup>**_7.4_**</sup>，<sup>注</sup>，旨在帮助"
        )


class TestLlmOutput:
    def test_a_models_headings_lose_tags_and_glyphs_only(self) -> None:
        markdown = (
            f"# {HAND}<sup>会员享有的主要权利</sup>\n\n"
            "## 会员已确认所提供的资料真实完整。\n\n"
            "正文<sup>1</sup>\n\n"
            "# \n\n"
            "```\n# <sup>x</sup>\n```"
        )
        assert plain_headings(markdown) == (
            "# 会员享有的主要权利\n\n"
            "## 会员已确认所提供的资料真实完整。\n\n"
            "正文<sup>1</sup>\n\n"
            "\n"
            "```\n# <sup>x</sup>\n```"
        )


# --- PDFs, through every path ------------------------------------------------


class TestSymbolFonts:
    def test_glyphs_are_read_by_the_font_that_draws_them(self, tmp_path: Path) -> None:
        path = _icon_headings(tmp_path / "icons.pdf")
        (glyphs,) = page_glyph_maps(path, [1], [HAND + DIAMOND])
        assert glyphs[HAND] == ("☞", True)
        assert glyphs[SPACE] == (" ", True)
        assert glyphs[DIAMOND] == ("❖", True)
        path = _symbol_minus(tmp_path / "minus.pdf")
        assert page_glyph_maps(path, [1], [MINUS]) == [{MINUS: ("−", False)}]

    def test_pages_without_glyphs_are_not_read(self, tmp_path: Path) -> None:
        missing = tmp_path / "missing.pdf"
        assert page_glyph_maps(missing, [1, 2], ["text", "more"]) == [{}, {}]


class TestPdfConversion:
    def test_icon_headings_are_plain(self, tmp_path: Path, unrepaired) -> None:
        path = _icon_headings(tmp_path / "icons.pdf")
        shape = f"# {HAND}{SPACE}<sup>会员享有的主要权利</sup>"
        assert shape in [h.strip() for h in _headings(unrepaired(path))]
        markdown = _convert(path, tmp_path)
        assert "# 会员享有的主要权利" in _headings(markdown)
        assert "# 会员需要特别留意的事项" in _headings(markdown)
        assert "<sup>" not in markdown
        assert not _private(markdown)

    def test_sentences_and_fields_are_no_headings(
        self, tmp_path: Path, unrepaired
    ) -> None:
        path = _sentence_headings(tmp_path / "s.pdf")
        texts = ("服务编号", "会员已确认", "如有隐瞒")
        before = _headings(unrepaired(path))
        assert all(any(text in h for h in before) for text in texts), before
        markdown = _convert(path, tmp_path)
        headings = _headings(markdown)
        assert "## 特别说明" in headings
        for text in texts:
            assert not any(text in heading for heading in headings), headings
            assert text in markdown

    def test_a_symbol_minus_is_kept(self, tmp_path: Path, unrepaired) -> None:
        path = _symbol_minus(tmp_path / "minus.pdf")
        assert f"使用率 {MINUS} 10%" in unrepaired(path)
        markdown = _convert(path, tmp_path)
        assert "（行业平均使用率 − 10%）" in markdown
        assert not _private(markdown)

    def test_notdef_glyphs_read_as_nothing(self, tmp_path: Path, unrepaired) -> None:
        path = _notdef(tmp_path / "notdef.pdf")
        assert "\uffff" * 8 in unrepaired(path)
        markdown = _convert(path, tmp_path)
        assert not _private(markdown)
        assert "# 服务期间" in markdown

    def test_the_vision_paths_text_is_repaired_too(self, tmp_path: Path) -> None:
        config = MarkitaiConfig()
        config.cache.enabled = False
        path = _icon_headings(tmp_path / "icons.pdf")
        result = PdfConverter(config)._render_pages_for_llm(path, tmp_path / "out")
        assert "# 会员享有的主要权利" in _headings(result.markdown)
        assert "<sup>" not in result.markdown
        assert not _private(result.markdown)

    def test_native_pages_of_an_ocr_run_are_repaired_too(self, tmp_path: Path) -> None:
        path = _icon_headings(tmp_path / "icons.pdf")
        assets = tmp_path / "assets"
        assets.mkdir()
        texts, _ = PdfConverter(MarkitaiConfig())._extract_native_pages(
            path, [0], assets, "png"
        )
        assert "# 会员享有的主要权利" in _headings(texts[0])
        assert not _private(texts[0])


# --- headings wrapped onto a second line --------------------------------------


def _two_line_heading(
    path: Path, *, gap: float, second: str, size2: float = 18
) -> Path:
    """An 18 pt bold heading on two lines, ``gap`` points apart, then body text."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text(
        (72, 100), "Quarterly results for the northern", fontname="hebo", fontsize=18
    )
    page.insert_text((60, 100 + 18 + gap), second, fontname="hebo", fontsize=size2)
    page.insert_text(
        (60, 200),
        "Body text of the section follows here.",
        fontname="helv",
        fontsize=10,
    )
    doc.save(path)
    return path


class TestWrappedHeadings:
    """A heading the layout model split at its line break is one heading
    again (the owner's 1.3.0 check: ``## **Cras fringilla ... commodo**``
    then ``## **a.**``)."""

    PAGE = "## **Quarterly results for the northern**\n\n## **Region Office**\n\nBody."

    def test_the_text_layer_shows_one_heading(self, tmp_path: Path) -> None:
        path = _two_line_heading(tmp_path / "h.pdf", gap=2, second="Region Office")
        from markitai.converter.pdf_markdown import wrapped_headings

        (pairs,) = wrapped_headings(path, [1], [self.PAGE])
        assert pairs == {("Quarterly results for the northern", "Region Office")}
        assert repair_page_markdown(self.PAGE, wrapped=pairs).startswith(
            "## **Quarterly results for the northern Region Office**\n\nBody."
        )

    @pytest.mark.parametrize(
        ("gap", "size2"),
        [(30, 18), (2, 12)],
        ids=["a paragraph gap", "another size"],
    )
    def test_two_headings_stay_two(
        self, tmp_path: Path, gap: float, size2: float
    ) -> None:
        path = _two_line_heading(
            tmp_path / "h.pdf", gap=gap, second="Region Office", size2=size2
        )
        from markitai.converter.pdf_markdown import wrapped_headings

        (pairs,) = wrapped_headings(path, [1], [self.PAGE])
        assert pairs == frozenset()
        assert "\n## **Region Office**" in repair_page_markdown(
            self.PAGE, wrapped=pairs
        )

    def test_a_lower_case_line_continues_without_the_text_layer(self) -> None:
        page = (
            "## **Cras fringilla ipsum magna, in fringilla dui commodo**\n\n## **a.**\n"
        )
        assert repair_page_markdown(page).startswith(
            "## **Cras fringilla ipsum magna, in fringilla dui commodo a.**"
        )

    def test_a_finished_heading_is_not_continued(self) -> None:
        page = "## Results.\n\n## methods and data\n"
        assert repair_page_markdown(page).count("## ") == 2

    def test_the_sample_pdf_heading_is_one_line(self, tmp_path: Path) -> None:
        source = Path(__file__).parents[1] / "fixtures" / "sample.pdf"
        markdown = _convert(source, tmp_path)
        headings = _headings(markdown)
        assert (
            "## **Cras fringilla ipsum magna, in fringilla dui commodo a.**" in headings
        )
        assert "## **a.**" not in headings
