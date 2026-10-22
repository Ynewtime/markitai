"""Authored CJK layout fixtures: one small PDF per failure shape.

Each page is set the way a Chinese form printed by iText is: composite
(Type0) fonts named after Microsoft YaHei and SimSun, not embedded, with
ToUnicode CMaps; pinkish section bands; 11 pt fields over 10 pt body text.
The text is an invented community course enrolment form: no document, person
or organisation is described. Run `python3 create-fixtures.py` here to
regenerate the deterministic bytes; `author-fixtures.json` records each
file's size and SHA-256.
"""

from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parent
WIDTH, HEIGHT = 595, 842


class Font:
    """A non-embedded Type0 font: each character gets the next CID, CJK
    characters advance a full em and the rest `half` (a list of widths,
    cycled, for a proportional face). `notdef` maps CID 0 to <ffff> in the
    ToUnicode CMap, as iText writes it."""

    def __init__(self, name, base, flags, half, dw=None, list_cjk=False, notdef=False):
        self.name, self.base, self.flags = name, base, flags
        self.half, self.dw, self.list_cjk, self.notdef = half, dw, list_cjk, notdef
        self.cids = {}

    def cid(self, ch):
        if ch not in self.cids:
            self.cids[ch] = len(self.cids) + 1
        return self.cids[ch]

    def advance(self, ch):
        if ord(ch) >= 0x2E80:
            return 1000
        return self.half[self.cid(ch) % len(self.half)]

    def show(self, text):
        return "<" + "".join(f"{self.cid(ch):04X}" for ch in text) + ">"

    def width(self, text, size):
        return sum(self.advance(ch) for ch in text) * size / 1000

    def objects(self, first):
        """Font, descendant, descriptor and ToUnicode objects numbered from
        `first`."""
        widths = []
        for ch, cid in sorted(self.cids.items(), key=lambda item: item[1]):
            if ord(ch) < 0x2E80 or self.list_cjk:
                widths.append(f"{cid} [{self.advance(ch)}]")
        dw = f"/DW {self.dw} " if self.dw is not None else ""
        entries = [f"<{cid:04X}> <{ord(ch):04X}>" for ch, cid in self.cids.items()]
        if self.notdef:
            entries.insert(0, "<0000> <ffff>")
        sections = "".join(
            f"{len(entries[i:i + 100])} beginbfchar\n" + "\n".join(entries[i:i + 100]) + "\nendbfchar\n"
            for i in range(0, len(entries), 100)
        )
        cmap = (
            "/CIDInit /ProcSet findresource begin 12 dict begin begincmap\n"
            "/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def\n"
            "/CMapName /Adobe-Identity-UCS def /CMapType 2 def\n"
            "1 begincodespacerange <0000> <FFFF> endcodespacerange\n"
            f"{sections}endcmap CMapName currentdict /CMap defineresource pop end end\n"
        ).encode()
        return [
            f"<< /Type /Font /Subtype /Type0 /BaseFont /{self.base} /Encoding /Identity-H "
            f"/DescendantFonts [{first + 1} 0 R] /ToUnicode {first + 3} 0 R >>".encode(),
            f"<< /Type /Font /Subtype /CIDFontType2 /BaseFont /{self.base} "
            f"/CIDSystemInfo << /Registry (Adobe) /Ordering (Identity) /Supplement 0 >> "
            f"{dw}/W [{' '.join(widths)}] /CIDToGIDMap /Identity /FontDescriptor {first + 2} 0 R >>".encode(),
            f"<< /Type /FontDescriptor /FontName /{self.base} /Flags {self.flags} "
            f"/FontBBox [-150 -250 1300 1050] /ItalicAngle 0 /Ascent 1036 /Descent -255 "
            f"/CapHeight 756 /StemV 80 >>".encode(),
            stream(cmap),
        ]


class SymbolFont:
    """A non-embedded TrueType symbol font as Word writes Symbol and Wingdings
    into a PDF: one byte per glyph, a ToUnicode CMap mapping each byte to
    the private-use code U+F000 plus the byte."""

    def __init__(self, name, base):
        self.name, self.base = name, base

    def show(self, codes):
        return "<" + codes.hex().upper() + ">"

    def objects(self, first):
        cmap = (
            "/CIDInit /ProcSet findresource begin 12 dict begin begincmap\n"
            "/CMapName /Symbolic def 1 begincodespacerange <00> <FF> endcodespacerange\n"
            "1 beginbfrange <20> <FF> <F020> endbfrange\n"
            "endcmap CMapName currentdict /CMap defineresource pop end end\n"
        ).encode()
        widths = " ".join(["900"] * 224)
        return [
            f"<< /Type /Font /Subtype /TrueType /BaseFont /{self.base} /FirstChar 32 /LastChar 255 "
            f"/Widths [{widths}] /FontDescriptor {first + 1} 0 R /ToUnicode {first + 2} 0 R >>".encode(),
            f"<< /Type /FontDescriptor /FontName /{self.base} /Flags 4 /FontBBox [0 -200 1000 800] "
            f"/ItalicAngle 0 /Ascent 800 /Descent -200 /CapHeight 700 /StemV 80 >>".encode(),
            stream(cmap),
        ]


def stream(data, extra=""):
    return f"<< /Length {len(data)}{extra} >>\nstream\n".encode() + data + b"\nendstream"


def fonts():
    # Microsoft YaHei as iText describes it: FixedPitch flag set (33) on a
    # proportional face, every glyph listed in /W, /DW 0.
    yahei = Font("F20", "MicrosoftYaHei", 33, [586, 295, 432, 697, 560, 610], dw=0, list_cjk=True)
    bold = Font("F21", "MicrosoftYaHeiUI-Bold", 33, [586, 297, 436, 697, 560, 610], dw=0, list_cjk=True)
    # SimSun as Word exports it: no FixedPitch flag, /W lists only its
    # 500-unit half-width glyphs and /DW 1000 covers the ideographs.
    simsun = Font("F15", "SimSun", 32, [500], dw=1000, notdef=True)
    return yahei, bold, simsun


class Page:
    def __init__(self):
        self.ops = []
        # Symbol fonts the page draws with (see `glyphs`).
        self.symbols = []

    def glyphs(self, font, size, x, y, codes):
        if font not in self.symbols:
            self.symbols.append(font)
        self.ops.append(f"BT /{font.name} {size} Tf 1 0 0 1 {x:.2f} {y:.2f} Tm {font.show(codes)} Tj ET")

    def text(self, font, size, x, y, value, right=None):
        # `right`: justify the line to end there with character spacing.
        spacing = ""
        if right is not None and len(value) > 1:
            spacing = f"{(right - x - font.width(value, size)) / (len(value) - 1):.3f} Tc "
        self.ops.append(f"BT /{font.name} {size} Tf {spacing}1 0 0 1 {x:.2f} {y:.2f} Tm {font.show(value)} Tj ET")
        return font.width(value, size)

    def band(self, y, height=19):
        self.ops.append(f"q 1 0.8588 0.7922 rg 34 {y} 527 {height} re f Q")

    def rule(self, y, x0, x1):
        self.ops.append(f"q 0 G 0.5 w {x0} {y} m {x1} {y} l S Q")

    def lines(self, font, size, x, top, values, leading=12):
        for i, value in enumerate(values):
            self.text(font, size, x, top - i * leading, value)
        return top - len(values) * leading


def build(name, draw, separator=None, form=None):
    yahei, bold, simsun = fonts()
    page = Page()
    draw(page, yahei, bold, simsun)
    content = "\n".join(page.ops).encode()
    if separator is not None:
        content = content.replace(b" ", separator)
    used = [font for font in (yahei, bold, simsun) if font.cids or (font.notdef and form)]
    used += page.symbols
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>", None, None]
    resources = {}
    for font in used:
        resources[font.name] = len(objects) + 1
        objects.extend(font.objects(len(objects) + 1))
    font_dict = " ".join(f"/{key} {value} 0 R" for key, value in resources.items())
    xobject = ""
    if form is not None:
        form_ops = form(simsun).encode()
        form_id = len(objects) + 1
        objects.append(stream(form_ops, f" /Type /XObject /Subtype /Form /BBox [0 0 {WIDTH} {HEIGHT}] /Resources << /Font << {font_dict} >> >>"))
        xobject = f"/XObject << /Xi1 {form_id} 0 R >>"
        content = b"q /Xi1 Do Q\n" + content
    objects[1] = f"<< /Type /Pages /Count 1 /Kids [3 0 R] /MediaBox [0 0 {WIDTH} {HEIGHT}] >>".encode()
    contents_id = len(objects) + 1
    objects.append(stream(content))
    objects[2] = f"<< /Type /Page /Parent 2 0 R /Contents {contents_id} 0 R /Resources << /Font << {font_dict} >> {xobject} >> >>".encode()
    pdf = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for number, obj in enumerate(objects, 1):
        offsets.append(len(pdf))
        pdf.extend(f"{number} 0 obj\n".encode() + obj + b"\nendobj\n")
    xref = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
    for offset in offsets:
        pdf.extend(f"{offset:010} 00000 n \n".encode())
    pdf.extend(f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    (ROOT / name).write_bytes(pdf)
    return {
        "file": name,
        "bytes": len(pdf),
        "sha256": hashlib.sha256(pdf).hexdigest(),
        "provenance": "Authored minimal PDF operators and invented text; no external or personal input.",
    }


BODY = [
    "本报名单由课程简章、报名表、缴费凭证、课程安排和其他书面约定构成，各方应当遵守约定。",
    "学员在课程期间内因故缺课或需要调课的，学习中心按照简章约定安排相应的补课课时。",
    "报名人应当按照简章约定交纳课程费用，未按约定交纳的，学习中心可以暂停相应的课程。",
    "本课程的学习期间为一年，自课程开始之日零时起至结业之日二十四时止，续报另行约定。",
]


def header(page, yahei, bold, simsun, top=790):
    page.text(bold, 20, 461, top, "电子报名单")
    page.band(top - 33, 17)
    page.text(bold, 10, 36, top - 29, "金额单位：人民币元")
    page.text(bold, 10, 418, top - 29, "报名编号：2026000011E")


def body(page, yahei, top, count=4):
    for i in range(count):
        top = page.lines(yahei, 10, 45, top, [BODY[i % len(BODY)]]) - 2
    return top


def nul_separated(page, yahei, bold, simsun):
    header(page, yahei, bold, simsun)
    page.band(725)
    page.text(bold, 14, 45, 730, "报名资料")
    page.text(simsun, 11, 45, 700, "报名人：示例学员")
    page.text(simsun, 11, 319, 700, "报名日期：2026年01月02日")
    body(page, yahei, 670)


def segmented_header_table(page, yahei, bold, simsun):
    header(page, yahei, bold, simsun)
    page.band(725)
    page.text(bold, 14, 45, 730, "课程安排及费用")
    edges = [34, 77, 218, 303, 369, 434, 490, 561]
    for y in (690, 672):
        for x0, x1 in zip(edges, edges[1:]):
            page.rule(y, x0, x1)
    names = ["序号", "课程名称", "累计课时分钟", "课程周期", "上课周数", "上课频次", "材料费"]
    for (x0, x1), name in zip(zip(edges, edges[1:]), names):
        page.text(bold, 11, (x0 + x1) / 2 - bold.width(name, 11) / 2, 677, name)
    rows = [
        ("1", ["社区书法艺术与", "传统绘画研修进阶课程（周", "末）"], "1,440.00", "1年", "1年", "每周", "640.00"),
        ("2", ["社区合唱团基础", "声乐入门课程（晚间班）"], "2,880.00", "1年", "1年", "每周", "320.50"),
    ]
    top = 655
    for number, name_lines, amount, term, years, frequency, premium in rows:
        middle = top - (len(name_lines) - 1) * 13 / 2
        for i, line in enumerate(name_lines):
            page.text(simsun, 11, 147.5 - simsun.width(line, 11) / 2, top - i * 13, line)
        for (x0, x1), value in zip(zip(edges, edges[1:]), [number, None, amount, term, years, frequency, premium]):
            if value:
                page.text(simsun, 11, (x0 + x1) / 2 - simsun.width(value, 11) / 2, middle, value)
        top -= len(name_lines) * 13 + 10
    page.rule(top + 2, 34, 312)
    page.rule(top + 2, 312, 561)
    page.text(bold, 11, 45, top - 14, "首期材料费合计：960.50")
    body(page, yahei, top - 44, 8)


def form_grid(page, yahei, bold, simsun):
    header(page, yahei, bold, simsun)
    page.band(725)
    page.text(bold, 14, 45, 730, "报名资料")
    for y, fields in [
        (700, [(45, "报名人：示例学员"), (177, "证件类型：学生证"), (319, "学员编号：202601020001")]),
        (680, [(45, "报名日期：2026年01月02日"), (319, "开课日期：2026年01月03日上午")]),
    ]:
        for x, value in fields:
            page.text(simsun, 11, x, y, value)
    page.band(650)
    page.text(bold, 14, 45, 655, "学员信息")
    page.text(simsun, 11, 45, 625, "上课学员：示例学员乙")
    page.text(simsun, 11, 319, 625, "与报名人关系：子女")
    page.text(bold, 11, 45, 600, "班级类型：周末班")
    page.text(bold, 11, 45, 584, "家庭报名关联编号：")
    page.text(bold, 11, 45, 571, "20260102000000000000000001")
    body(page, yahei, 540, 8)


def long_sentences(page, yahei, bold, simsun):
    header(page, yahei, bold, simsun)
    top = body(page, yahei, 730)
    page.text(bold, 11, 45, top - 6, "报名人已确认学员符合课程简章所述的报名条件。")
    page.text(bold, 11, 45, top - 22, "如有隐瞒或填写不实，足以影响学习中心正常安排的，学习中心有权取消报名，并对取消报名前发生的任")
    page.text(bold, 11, 45, top - 35, "何课程损失不承担任何责任。")
    page.band(top - 75)
    page.text(bold, 14, 45, top - 70, "学习中心办学资质告知和服务评价")
    page.text(bold, 11, 45, top - 100, "本中心办学资质与服务评价符合要求，您可以登录http://www.example.com/services/")
    page.text(bold, 11, 45, top - 113, "Default.aspx查看最近季度办学资质年检结果和学员服务评价等全部相关信息。")
    body(page, yahei, top - 140, 6)


def fixed_pitch_cjk(page, yahei, bold, simsun):
    header(page, yahei, bold, simsun)
    top = 740
    # Ideographs and the half-width letters and digits of both faces.
    for line in [
        "第1条 报名学员范围：年满18周岁至70周岁、身体健康的个人（ABCDEFGHIJ）。",
        "第2条 学习期间：本课程的学习期间为1年，材料费为640.00元，可以按年交纳。",
        "第3条 试听期：首次报名的试听期为90天，续报的不设试听期（KLMNOPQRST）。",
    ]:
        page.text(simsun, 10.4, 48, top, line)
        page.text(yahei, 10, 48, top - 18, line)
        top -= 40
    body(page, yahei, top)


def notdef_ffff(page, yahei, bold, simsun):
    header(page, yahei, bold, simsun)
    body(page, yahei, 730)


def notdef_form(simsun):
    # The template iText draws on every page: a clip with 32 .notdef glyphs.
    return f"q 54 600 441 36 re W n BT /{simsun.name} 10 Tf 1 0 0 1 56 620 Tm <{'0000' * 32}> Tj ET Q"


def cjk_enumerators(page, yahei, bold, simsun):
    header(page, yahei, bold, simsun)
    page.band(725)
    page.text(bold, 14, 45, 730, "报名须知")
    page.lines(yahei, 10, 45, 700, [
        "1．学员曾经或目前参加过如下2类课程：",
        "1）书法课程：楷书基础（4级及以上）、行书进阶（3课时以上书写/临摹/创作练习）。",
        "2）绘画进阶课程：由素描或水彩基础课程延伸的进阶内容（详见五大类素描延伸课程、七大类水彩延伸课程清",
        "单）。",
        "2．学员有以下上课情况：",
        "1） 学员过去2年内，参加过任何集训或单次连续上课6天及以上。",
        "下述情况，仍可正常报名：",
        "1）楷书基础、行书进阶(1-3级)、素描基础(3课时及以下)。",
        "说明",
        "行书进阶：未曾参加集训或老师未建议跳级的情况下，以近一年内最近一次测评为准。",
    ])


def double_spaced(page, yahei, bold, simsun):
    # A clause set as Word sets it: 10.4 pt at 20 pt leading, a centred
    # article title, the first line indented, a clause's text hanging past
    # its number. Every full line reaches x = 510.
    page.text(simsun, 12, 213, 780, "附加短期课程调整服务条款")
    page.text(simsun, 10.4, 287, 740, "总则")
    lines = [
        (111, 720, "第一条 本附加服务约定（以下简称“本附加约定”）需附加于各类线下课程报名协议、", 530),
        (90, 700, "各类线上课程报名协议（以下简称“主协议”）上方可生效。", None),
        (111, 680, "第二条 本附加约定与主协议相悖之处，以本附加约定为准；本附加约定未尽事宜，以", 530),
        (90, 660, "主协议为准。", None),
        (48, 630, "1.1", None),
        (105, 630, "报名学员范围 首次报名或非连续报名本课程时年满十八周岁至七十周岁的个人可作", 530),
        (105, 610, "为报名学员参加本课程，且报名时需符合本中心当时的报名规定。", None),
    ]
    for x, y, value, right in lines:
        page.text(simsun, 10.4, x, y, value, right)
    for y in range(570, 420, -20):
        page.text(simsun, 10.4, 90, y, BODY[(570 - y) // 20 % len(BODY)], 530)


def short_last_lines(page, yahei, bold, simsun):
    # Paragraphs set without spacing: every line but a paragraph's last runs
    # as far as the longest line of the page, the last stops short after
    # its sentence's `。`. Then a step numbered `2、` standing alone between
    # two paragraphs.
    page.text(bold, 14, 45, 730, "报名流程")
    top = 700
    for paragraph in [
        "本报名单由课程简章、报名表、缴费凭证、课程安排和其他书面约定构成，报名人与学习中心应当共同遵守约定。",
        "学员在课程期间内因故缺课或需要调课的，应当提前告知学习中心，学习中心按照简章约定安排相应的补课课时。",
        "报名人应当按照简章约定按时交纳课程费用，未按约定交纳的，学习中心可以暂停相应的课程并书面告知报名人。",
    ]:
        top = page.lines(yahei, 10, 45, top, [paragraph[:42], paragraph[42:]], 14)
    page.text(yahei, 10, 45, top - 14, "2、登录报名系统后，在课程列表中选择 1 门你要报名的课程")
    body(page, yahei, top - 48)


def symbol_fonts(page, yahei, bold, simsun):
    # Word's icon headings and contents: a 26 pt Wingdings 2 pointing hand
    # before a section's title, entries after Wingdings diamonds, a SymbolMT
    # minus in a sentence, and a note's title in a face of its own a little
    # larger than the body text, right above its paragraph.
    hand = SymbolFont("G1", "Wingdings2")
    diamond = SymbolFont("G2", "ABCDEF+Wingdings-Regular")
    symbol = SymbolFont("G3", "SymbolMT")
    page.text(bold, 16, 45, 790, "课程服务说明")
    page.glyphs(hand, 26, 40, 742, b"\x43")
    page.text(bold, 12, 76, 746, "学员享有的主要权利")
    top = 716
    for n, entry in enumerate(["学员可在七日内申请退课", "课程内容以本说明列明的为准", "学员可随时查询上课记录"], 1):
        page.glyphs(diamond, 10, 45, top, b"\x76")
        page.text(yahei, 10, 57, top, entry + "…" * (24 - len(entry)) + f"{n}.{n + 1}")
        top -= 14
    top = body(page, yahei, top - 16)
    page.text(yahei, 10, 45, top - 6, "当月出勤率≥（班级平均出勤率")
    page.glyphs(symbol, 10, 185, top - 6, b"\x2d")
    page.text(yahei, 10, 192, top - 6, " 10%）时，学习中心可调整课时。")
    page.text(simsun, 10.5, 45, top - 40, "特别说明")
    body(page, yahei, top - 53)


def heading_over_rule(page, yahei, bold, simsun):
    # A heading set right above a table whose top rule is drawn a segment
    # per column, under the heading's text.
    page.text(bold, 14, 45, 746, "课程安排")
    edges = [40, 80, 280, 360, 420, 490]
    for y in (736, 718):
        for x0, x1 in zip(edges, edges[1:]):
            page.rule(y, x0, x1)
    for (x0, x1), name in zip(zip(edges, edges[1:]), ["序号", "课程名称", "课时", "周数", "材料费"]):
        page.text(bold, 10, x0 + 6, 724, name)
    for i, row in enumerate([["1", "社区书法进阶课程", "1,440", "52", "640.00"], ["2", "社区合唱基础课程", "2,880", "52", "320.50"]]):
        for (x0, x1), value in zip(zip(edges, edges[1:]), row):
            page.text(simsun, 10, x0 + 6, 704 - i * 16, value)
    body(page, yahei, 650)


def main():
    records = [
        build("nul-separated.pdf", nul_separated, separator=b" \x00 "),
        build("segmented-header-table.pdf", segmented_header_table),
        build("form-grid.pdf", form_grid),
        build("long-sentences.pdf", long_sentences),
        build("fixed-pitch-cjk.pdf", fixed_pitch_cjk),
        build("notdef-ffff.pdf", notdef_ffff, form=notdef_form),
        build("cjk-enumerators.pdf", cjk_enumerators),
        build("double-spaced.pdf", double_spaced),
        build("short-last-lines.pdf", short_last_lines),
        build("symbol-fonts.pdf", symbol_fonts),
        build("heading-over-rule.pdf", heading_over_rule),
    ]
    (ROOT / "author-fixtures.json").write_text(json.dumps({"fixtures": records}, indent=2) + "\n")
    print(json.dumps(records, indent=2))


main()
