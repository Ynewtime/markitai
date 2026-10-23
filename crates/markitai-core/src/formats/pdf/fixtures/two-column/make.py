"""Invented two-column newsletter page set by reportlab, as `two-column.pdf`.

A centred title runs across both columns above them, a running header and a
`Page N` footer sit at the page edges, and the body paragraphs fill two
frames. Several lines of the left column end in a full stop short of the
column's edge while the next word could not have fitted on them: wrapped
lines of one paragraph, which must not be split there.

Usage: python make.py (needs reportlab).
"""
import os

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import BaseDocTemplate, Frame, FrameBreak, PageTemplate, Paragraph

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = A4

PARAGRAPHS = [
    "A second item concerned the lights on the pier. Several lamps have failed. The board will replace "
    "them with lower-energy fittings. These should last longer and cost less to run. The electrician will "
    "survey the cabling first, because some of it dates from the nineteen-sixties.",
    "Finally, the board discussed the summer regatta. Entries are already up on last year. Parking will "
    "again be a problem. The car park behind the chandlery will be opened for the weekend, and a shuttle "
    "bus will run from the village green every twenty minutes.",
    "The harbour board met on Thursday to review the winter maintenance plan. Members agreed that the "
    "north wall needs new capping stones. The cost estimate came in lower than expected. Work will start "
    "once the spring tides have passed.",
]


def chrome(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.drawString(50, H - 30, "Harbour Gazette - Spring Issue")
    canvas.drawRightString(W - 50, 25, f"Page {doc.page}")
    canvas.restoreState()


def main():
    ss = getSampleStyleSheet()
    doc = BaseDocTemplate(os.path.join(HERE, "two-column.pdf"), pagesize=A4, title="Harbour Gazette",
                          author="Invented", creator="reportlab", invariant=1)
    width = (W - 110) / 2
    frames = [
        Frame(50, H - 150, W - 100, 90, id="top"),
        Frame(50, 50, width, H - 210, id="left"),
        Frame(60 + width, 50, width, H - 210, id="right"),
    ]
    doc.addPageTemplates([PageTemplate(id="page", frames=frames, onPage=chrome)])
    story = [Paragraph("Harbour Gazette", ss["Title"]), FrameBreak()]
    for number in range(1, 6):
        story.append(Paragraph(f"Section {number}", ss["Heading2"]))
        story.append(Paragraph(PARAGRAPHS[(number - 1) % 3] + f" Section {number} ends here.", ss["BodyText"]))
    doc.build(story)


if __name__ == "__main__":
    main()
