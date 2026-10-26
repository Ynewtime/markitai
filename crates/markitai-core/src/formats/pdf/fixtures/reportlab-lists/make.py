"""Invented one-page trip notes with ReportLab lists.

ReportLab's ListFlowable draws each bullet as a run of its own, larger than
the text (12 pt beside 10 pt) and a little lower, with the item's text
hanging to its right. The reader once read the text as a subscript of the
bullet, wrote `## •Item` headings, and glued the paragraph after the list to
its last item. Real headings (Heading1, Heading2) and a numbered list sit
around the bulleted one.

Usage: python make.py (needs reportlab).
"""
import os

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ss = getSampleStyleSheet()
    body = ss["BodyText"]
    packing = [
        "A waterproof jacket with a hood, because the ridge path is exposed and the forecast changes "
        "quickly in the afternoon.",
        "Two litres of water and a flask of tea.",
        "A paper map of the valley and a compass, in case the phone loses its signal behind the "
        "northern shoulder of the hill.",
    ]
    steps = [
        "Park at the village hall and follow the lane past the mill.",
        "Cross the footbridge and climb the stone steps to the gate.",
        "Turn left along the wall until the cairn comes into view.",
    ]
    story = [
        Paragraph("Ridge Walk", ss["Heading1"]),
        Paragraph("Notes for the walking club's spring outing along the ridge above the valley.", body),
        Paragraph("Packing list", ss["Heading2"]),
        ListFlowable([ListItem(Paragraph(text, body)) for text in packing], bulletType="bullet"),
        Spacer(1, 8),
        Paragraph("Everyone carries their own lunch; the café at the top closes at three.", body),
        Paragraph("Route", ss["Heading2"]),
        ListFlowable([ListItem(Paragraph(text, body)) for text in steps], bulletType="1", bulletFormat="%s."),
        Spacer(1, 8),
        Paragraph("The walk takes about four hours with a stop at the cairn.", body),
    ]
    SimpleDocTemplate(os.path.join(HERE, "reportlab-lists.pdf"), pagesize=A4, title="Ridge Walk",
                      author="Invented", creator="reportlab", invariant=1).build(story)


if __name__ == "__main__":
    main()
