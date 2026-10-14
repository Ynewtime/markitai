"""Invented day plan as ODS: merged banner rows (number-columns-spanned), header-less column, empty cells."""
import sys, zipfile
from xml.sax.saxutils import escape
ROWS = [
    ["Spring Fair 2026 · Day Plan", None, None, None],
    ["Venue: Riverside Hall, east wing", None, None, None],
    ["Contact: volunteer desk, extension 12", None, None, None],
    ["Item", "Time", "Task", "Note"],
    ["Setup crew", "07:30", "Unload tables", None],
    ["Kitchen", "08:00", "Start urns", "Bring spare fuses"],
    ["Stage", None, "Sound check", None],
    ["Afternoon", None, None, None],
    [None, "Time", "Task", "Note"],
    ["Raffle", "14:00", "Draw tickets", "NA"],
    ["Close", 17.5, None, 3],
]
MERGED = {0, 1, 2, 7}
def cell(v):
    if v is None: return "<table:table-cell/>"
    if isinstance(v, (int, float)):
        return f'<table:table-cell office:value-type="float" office:value="{v}"><text:p>{v}</text:p></table:table-cell>'
    return f'<table:table-cell office:value-type="string"><text:p>{escape(v)}</text:p></table:table-cell>'
rows = []
for i, r in enumerate(ROWS):
    if i in MERGED:
        rows.append('<table:table-row><table:table-cell table:number-columns-spanned="4" office:value-type="string"><text:p>'
                    + escape(r[0]) + '</text:p></table:table-cell><table:covered-table-cell table:number-columns-repeated="3"/></table:table-row>')
    else:
        rows.append("<table:table-row>" + "".join(cell(v) for v in r) + "</table:table-row>")
ns = ('xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
      'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" '
      'xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0" office:version="1.3"')
content = (f'<?xml version="1.0" encoding="UTF-8"?><office:document-content {ns}><office:body><office:spreadsheet>'
           '<table:table table:name="Day plan"><table:table-column table:number-columns-repeated="4"/>'
           + "".join(rows) + '</table:table></office:spreadsheet></office:body></office:document-content>')
manifest = ('<?xml version="1.0" encoding="UTF-8"?><manifest:manifest xmlns:manifest="urn:oasis:names:tc:opendocument:xmlns:manifest:1.0" manifest:version="1.3">'
            '<manifest:file-entry manifest:full-path="/" manifest:media-type="application/vnd.oasis.opendocument.spreadsheet"/>'
            '<manifest:file-entry manifest:full-path="content.xml" manifest:media-type="text/xml"/></manifest:manifest>')
with zipfile.ZipFile(sys.argv[1], "w") as z:
    z.writestr(zipfile.ZipInfo("mimetype", (2026, 1, 1, 0, 0, 0)), "application/vnd.oasis.opendocument.spreadsheet", compress_type=zipfile.ZIP_STORED)
    for name, data in (("content.xml", content), ("META-INF/manifest.xml", manifest)):
        z.writestr(zipfile.ZipInfo(name, (2026, 1, 1, 0, 0, 0)), data, compress_type=zipfile.ZIP_DEFLATED)
