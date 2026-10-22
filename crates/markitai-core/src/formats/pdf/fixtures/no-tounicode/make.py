"""Make greek-subset.pdf: invented Greek text in an embedded TrueType subset
whose ToUnicode map is removed, so the font names neither an encoding nor a
ToUnicode CMap and draws its letters with subset codes 1-31.

python make.py OUTDIR  (ReportLab 5.1 and PyMuPDF 2.0; DejaVu Sans)
"""
import os
import sys

import pymupdf
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate

out = sys.argv[1]
pdfmetrics.registerFont(TTFont("DV", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
title = ParagraphStyle("title", fontName="DV", fontSize=20, leading=26)
body = ParagraphStyle("body", fontName="DV", fontSize=11, leading=15)
source = os.path.join(out, "greek-subset-src.pdf")
SimpleDocTemplate(source, pagesize=A4, invariant=1).build(
    [Paragraph("Δοκιμαστικός τίτλος εγγράφου", title)]
    + [
        Paragraph(
            "Αυτή είναι μια επινοημένη παράγραφος με ελληνικό κείμενο για τον "
            "έλεγχο της μετατροπής. Δεν περιέχει πραγματικά στοιχεία, μόνο "
            "λέξεις για δοκιμή.",
            body,
        )
    ]
    * 2
)
doc = pymupdf.open(source)
for xref in range(1, doc.xref_length()):
    if doc.xref_get_key(xref, "ToUnicode")[0] != "null":
        doc.xref_set_key(xref, "ToUnicode", "null")
doc.save(os.path.join(out, "greek-subset.pdf"), garbage=4, deflate=True, no_new_id=True)
os.remove(source)
