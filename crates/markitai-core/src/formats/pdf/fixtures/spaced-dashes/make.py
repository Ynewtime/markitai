"""Invented two-column newsletter page of spaced and in-word hyphens, as `spaced-dashes.pdf`.

The page is set like `two-column/make.py` (a running header, a title across
both columns, two frames of paragraphs), a layout markitai leaves to the
PDF reader's own text. Its paragraphs set hyphens and dashes between
spaces (`Monday - Friday`, `Inland – Coastal`), in-word hyphens
(`lower-energy`) and words broken at a line end (`conver-` / `sion`). Two
lines above the footer place each word as a run of its own with no space
characters and the hyphens in another face: a word space apart in
`Tide tables - Winter Edition`, touching their words in `lower-cost` and
`long-life`.

Usage: python make.py (needs reportlab).
"""
import os

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.platypus import BaseDocTemplate, Frame, FrameBreak, PageTemplate, Paragraph

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = A4

PARAGRAPHS = [
    "The harbour office is open Monday - Friday, and the ferry runs North - South twice a day. The board "
    "will fit lower-energy lamps and a well-known pump on the pier. These should last longer and cost "
    "less to run, because the old fittings date from the nineteen-sixties.",
    "Finally, the board discussed the summer regatta. Buses run Inland \u2013 Coastal in summer \u2014 "
    "not in winter. The car park behind the chandlery will be opened for the weekend.",
    "The surveyor will check the conver-<br/>sion of the old net shed before any work begins. Members "
    "agreed that the north wall needs new capping stones, and the cost estimate came in lower than "
    "expected.",
]


def runs(canvas, y, parts):
    """Draw `parts` as separate runs: a part after `None` follows a word space; hyphens in Times."""
    x, gap = 50, stringWidth(" ", "Helvetica", 10)
    spaced = False
    for part in parts:
        if part is None:
            spaced = True
            continue
        if spaced:
            x += gap
        font = "Times-Roman" if part == "-" else "Helvetica"
        canvas.setFont(font, 10)
        canvas.drawString(x, y, part)
        x += stringWidth(part, font, 10)
        spaced = False


def chrome(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.drawString(50, H - 30, "Harbour Gazette - Spring Issue")
    canvas.drawRightString(W - 50, 25, f"Page {doc.page}")
    runs(canvas, 80, ["Tide", None, "tables", None, "-", None, "Winter", None, "Edition"])
    runs(canvas, 66, ["Fit", None, "a", None, "lower", "-", "cost", None, "lamp", None, "and", None, "a",
                      None, "long", "-", "life", None, "bulb."])
    canvas.restoreState()


def main():
    ss = getSampleStyleSheet()
    doc = BaseDocTemplate(os.path.join(HERE, "spaced-dashes.pdf"), pagesize=A4, title="Harbour Gazette",
                          author="Invented", creator="reportlab", invariant=1)
    width = (W - 110) / 2
    frames = [
        Frame(50, H - 150, W - 100, 90, id="top"),
        Frame(50, 100, width, H - 260, id="left"),
        Frame(60 + width, 100, width, H - 260, id="right"),
    ]
    doc.addPageTemplates([PageTemplate(id="page", frames=frames, onPage=chrome)])
    story = [Paragraph("Harbour Gazette", ss["Title"]), FrameBreak()]
    for number in range(1, 6):
        story.append(Paragraph(f"Section {number}", ss["Heading2"]))
        story.append(Paragraph(PARAGRAPHS[(number - 1) % 3] + f" Section {number} ends here.", ss["BodyText"]))
    doc.build(story)


if __name__ == "__main__":
    main()
