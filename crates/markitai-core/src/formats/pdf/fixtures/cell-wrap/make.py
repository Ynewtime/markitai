"""Invented ruled tables whose cells wrap, as `cell-wrap.pdf` and `cell-hyphens.pdf`.

`cell-wrap.pdf` is set by ReportLab's table layout: page 1 holds a notes
column narrow enough that sentences wrap where the column ends (one only to
put its last short word on a line of its own); page 2 holds cells with line
breaks of their own (two short lines, a label above numbered and bulleted
lines, an empty line, a sentence ending a line short of the edge), a wrapped
centred header and a header broken into two short lines; page 3 holds
Chinese cells set in the standard STSong-Light CID font, wrapped between
characters, and one with a line break.

`cell-hyphens.pdf` lays its cells line by line: each cell's text is wrapped
greedily at the column's inner width, as a word processor wraps it, so a line
ends at a word the column could not hold. Its cells break a word at a
hyphenation point (`extra-` / `ordinary`), a compound after its hyphen
(`long-` / `term`), before a suspended hyphen's conjunction (`short-` /
`and`), and one sets a short line ending in a hyphen above another (a line
break of its own).

Usage: python make.py (needs reportlab).
"""
import os
import re

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen import canvas
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

HERE = os.path.dirname(os.path.abspath(__file__))
pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))

BODY = ParagraphStyle("body", fontName="Helvetica", fontSize=10, leading=12)
CENTRED = ParagraphStyle("centred", parent=BODY, alignment=1)
CJK = ParagraphStyle("cjk", fontName="STSong-Light", fontSize=10, leading=13, wordWrap="CJK")
RULED = TableStyle(
    [
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]
)


def table(rows, widths, styles=None):
    data = [
        [Paragraph(text, (styles or {}).get((r, c), BODY)) for c, text in enumerate(row)]
        for r, row in enumerate(rows)
    ]
    t = Table(data, colWidths=widths)
    t.setStyle(RULED)
    return t


def cell_wrap():
    doc = SimpleDocTemplate(os.path.join(HERE, "cell-wrap.pdf"), pagesize=A4, invariant=1)
    doc.build(
        [
            Paragraph("Harbour duties", BODY),
            Spacer(1, 12),
            table(
                [
                    ["Duty", "Notes", "Crew"],
                    ["Hull", "Tar the hull of the old pilot boat before the rains go.", "3"],
                    ["Lamps", "Check each lamp on the breakwater after dusk and log the faults in the book.", "2"],
                    ["Flags", "Raise the flags", "1"],
                ],
                [60, 230, 40],
            ),
            PageBreak(),
            Paragraph("Workshop notes", BODY),
            Spacer(1, 12),
            table(
                [
                    ["Step", "Hands needed on each of the working days of the week", "Unit<br/>(kg)"],
                    ["1", "Bring gloves<br/>Wear boots", "4"],
                    ["2", "Ingredients:<br/>1. Flour<br/>2. Sugar<br/>3. Butter softened at room temperature for an hour", "2"],
                    ["3", "- first point<br/>- second point that is long enough to wrap in this column", "1"],
                    ["4", "Opening line<br/><br/>After an empty line", "5"],
                    ["5", "Paint the shed doors.<br/>Then oil the hinges.", "3"],
                ],
                [40, 200, 50],
                {(0, 1): CENTRED, (0, 2): CENTRED},
            ),
            PageBreak(),
            Paragraph("Duty roster", BODY),
            Spacer(1, 12),
            table(
                [
                    ["项目", "说明"],
                    ["甲", "码头北侧的缆绳每天中午之前需要全部整理好并逐一登记在值班簿上。"],
                    ["乙", "第一行<br/>第二行"],
                    ["丙", "混合 English 与中文的单元格内容在窄列里也会自然换行显示。"],
                ],
                [40, 150],
                {(r, c): CJK for r in range(4) for c in range(2)},
            ),
        ]
    )


FONT, SIZE, LEAD, PAD = "Helvetica", 10, 12, 4


def width(text):
    return pdfmetrics.stringWidth(text, FONT, SIZE)


def wrap(text, inner):
    """Greedy wrapping at `inner` points: "\\n" breaks a line, "~" is a
    hyphenation point (printed as "-" when a line ends there) and "^" a break
    allowed after a hyphen."""
    lines = []
    for para in text.split("\n"):
        line = ""
        for word in para.split(" "):
            while word:
                glue = " " if line else ""
                plain = re.sub(r"[~^]", "", word)
                if width(line + glue + plain) <= inner:
                    line += glue + plain
                    break
                # The longest head up to a break point that fits, with its
                # hyphen when the point is "~".
                head = None
                for m in re.finditer(r"[~^]", word):
                    candidate = re.sub(r"[~^]", "", word[: m.start()]) + ("-" if m.group() == "~" else "")
                    if width(line + glue + candidate) <= inner:
                        head = (candidate, word[m.end():])
                if head:
                    lines.append(line + glue + head[0])
                    line, word = "", head[1]
                else:
                    lines.append(line)
                    line = ""
                    if width(plain) > inner:
                        raise ValueError(f"{plain!r} does not fit")
        lines.append(line)
    return lines


def cell_hyphens():
    widths = [70, 166, 40]
    rows = [
        ["Case", "Text", "N"],
        ["split", "The committee reviewed the extra~ordinary proposal in detail.", "1"],
        ["compound", "The town council signed a long-^term lease for the park.", "2"],
        ["suspended", "The harbour plan sets out its short- and long-term goals.", "3"],
        ["break", "Flour-\nSugar\nButter", "4"],
    ]
    xs = [50]
    for w in widths:
        xs.append(xs[-1] + w)
    c = canvas.Canvas(os.path.join(HERE, "cell-hyphens.pdf"), pagesize=A4, invariant=1)
    c.setFont(FONT, SIZE)
    c.drawString(xs[0], 800, "Lease terms")
    ys = [780]
    for row in rows:
        cells = [wrap(text, widths[i] - 2 * PAD) for i, text in enumerate(row)]
        y = ys[-1]
        for col, cell in enumerate(cells):
            for i, line in enumerate(cell):
                c.drawString(xs[col] + PAD, y - PAD - SIZE - i * LEAD + 2, line)
        ys.append(y - max(len(cell) for cell in cells) * LEAD - 2 * PAD)
    c.setLineWidth(0.5)
    for x in xs:
        c.line(x, ys[0], x, ys[-1])
    for y in ys:
        c.line(xs[0], y, xs[-1], y)
    c.save()


if __name__ == "__main__":
    cell_wrap()
    cell_hyphens()
