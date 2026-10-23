"""Invented one-page notes with raised and lowered marks, set by two producers.

reportlab.pdf (10 pt) and reportlab-12pt.pdf: reportlab paragraphs whose `<super>`/`<sub>`
runs are set with a text rise (`Ts`) in a smaller size, so a mark sits half
the line's size above it: units (m², cm³), footnote marks after words and
numbers, a letter and an asterisk mark, a subscript, and a mark ending a line.

libreoffice.pdf: the same kind of text written as Markdown, turned into DOCX
by pandoc and printed by LibreOffice (`soffice --convert-to pdf`): footnotes
whose anchors LibreOffice raises in a smaller size, with the footnote texts
at the foot of the page, and pandoc's superscripts and subscripts.

The LibreOffice file is rewritten with pypdf so it carries no build time,
time zone or tool version: its XMP packet is dropped and its information
dictionary holds the title, a generic producer and fixed dates in UTC, as
the reportlab files (`invariant=1`) do.

Usage: python make.py (needs reportlab, pypdf, pandoc and soffice on PATH).
"""
import os, shutil, subprocess, sys, tempfile

from pypdf import PdfReader, PdfWriter
from pypdf.generic import NameObject
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate

HERE = os.path.dirname(os.path.abspath(__file__))


def reportlab_pdf():
    ss = getSampleStyleSheet()
    body10 = ParagraphStyle("b10", parent=ss["BodyText"], fontName="Helvetica", fontSize=10, leading=12)
    body12 = ParagraphStyle("b12", parent=body10, fontSize=12, leading=15, spaceBefore=8)
    story = [
        Paragraph("Tide Notes", ss["Heading1"]),
        Paragraph("The spring tide reached 5.6 m<super>1</super> on the twelfth, the highest reading this year. "
                  "The keeper wrote it down at noon and again at the evening tide, when the water had fallen back.", body10),
        Paragraph("Water covers 70 m<super>2</super> of the car park at high tide, and the new tank holds 450 cm<super>3</super> "
                  "of sand. Harbour dues rose<super>3</super> in April and again<super>4</super> in June after the board met in the hall.", body10),
        Paragraph("The keeper logged CO<sub>2</sub> readings twice a day; see the appendix<super>a</super> and the errata<super>*</super> "
                  "for corrections to the readings taken during the storm in the second week of the month.", body10),
        Paragraph("The last note ends with a mark after its final word, which the reader must keep on it<super>6</super>", body10),
    ]
    SimpleDocTemplate(os.path.join(HERE, "reportlab.pdf"), pagesize=A4, title="Tide Notes",
                      author="Invented", creator="reportlab", invariant=1).build(story)
    story = [
        Paragraph("Ferry Survey", ss["Heading1"]),
        Paragraph("At twelve points the ferry deck measures 210 m<super>2</super> and the hold 95 m<super>3</super>, as the surveyor "
                  "wrote<super>5</super> in the report the board received before the spring sailings began.", body12),
        Paragraph("The ramp takes 3.5 t<super>7</super> per axle and the water tank carries 8 m<super>3</super> for the crossing.", body12),
    ]
    SimpleDocTemplate(os.path.join(HERE, "reportlab-12pt.pdf"), pagesize=A4, title="Ferry Survey",
                      author="Invented", creator="reportlab", invariant=1).build(story)


MARKDOWN = """---
title: Ferry Notes
---

# Ferry Notes

The ferry runs hourly[^1] between the east pier and the island. The deck holds 120 m^2^ of cars.

Harbour dues rose[^2] in April. The tank holds 450 cm^3^ of water and the air has little CO~2~ in it.

The surveyor's note ends the page with a mark after its last word.[^3]

[^1]: Except on public holidays.

[^2]: By five per cent.

[^3]: Written in March.
"""


def scrub(source, target, title):
    """Copy `source` to `target` without its XMP packet, with fixed metadata."""
    writer = PdfWriter(clone_from=PdfReader(source))
    root = writer._root_object
    if NameObject("/Metadata") in root:
        del root[NameObject("/Metadata")]
    writer._info = None
    writer.add_metadata({
        "/Title": title,
        "/Creator": "LibreOffice",
        "/Producer": "LibreOffice",
        "/CreationDate": "D:20000101000000Z",
        "/ModDate": "D:20000101000000Z",
    })
    # The packet stays in the cloned objects until orphans are removed.
    writer.compress_identical_objects(remove_duplicates=False, remove_unreferenced=True)
    with open(target, "wb") as f:
        writer.write(f)


def libreoffice_pdf():
    work = tempfile.mkdtemp()
    try:
        src = os.path.join(work, "libreoffice.md")
        with open(src, "w") as f:
            f.write(MARKDOWN)
        odt = os.path.join(work, "libreoffice.docx")
        subprocess.run(["pandoc", src, "-o", odt], check=True)
        profile = "file://" + os.path.join(work, "profile")
        subprocess.run(["soffice", f"-env:UserInstallation={profile}", "--headless",
                        "--convert-to", "pdf", "--outdir", work, odt], check=True,
                       stdout=subprocess.DEVNULL)
        scrub(os.path.join(work, "libreoffice.pdf"), os.path.join(HERE, "libreoffice.pdf"), "Ferry Notes")
    finally:
        shutil.rmtree(work)


if __name__ == "__main__":
    reportlab_pdf()
    libreoffice_pdf()
