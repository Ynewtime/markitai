#!/usr/bin/env python3
"""Write embedded-objects.ppt: a legacy PowerPoint deck whose slides show
embedded OLE objects, built record by record with the standard library only.

Every byte is authored here from the published specifications ([MS-CFB],
[MS-PPT], [MS-XLS], OpenDocument 1.2); no Office application or third-party
library touches the output. The deck has five slides, each a title and one
object shape:

1. "Revenue": a LibreOffice chart object (`package_stream`, an ODF chart
   package), compressed as LibreOffice stores it.
2. "Budget": an Excel worksheet object (BIFF8 `Workbook`), uncompressed, in
   a workbook of two sheets of which one holds data.
3. "Visitors": an MS Graph object, a `Workbook` stream holding only a chart
   substream with its cached series.
4. "Sales": an Excel chart object: a chart sheet (shown) and its data sheet.
5. "Formula": an equation object, which holds no data and adds nothing.

Run: python3 generate.py  (writes embedded-objects.ppt and provenance.json
beside this script; the output is byte-for-byte reproducible).
"""

import hashlib
import io
import json
import os
import struct
import zipfile
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))

# --- [MS-CFB] compound file, version 3, flat (every stream under the root) --

END_OF_CHAIN = 0xFFFFFFFE
FREE = 0xFFFFFFFF
FAT_SECTOR = 0xFFFFFFFD
NO_STREAM = 0xFFFFFFFF


def guid(text):
    """A GUID as stored: the first three fields little-endian."""
    parts = text.split("-")
    return (struct.pack("<IHH", int(parts[0], 16), int(parts[1], 16), int(parts[2], 16))
            + bytes.fromhex(parts[3] + parts[4]))


def compound_file(streams, clsid=bytes(16)):
    """A compound file holding `streams` ((name, bytes) pairs) at its root."""
    names = sorted((n for n, _ in streams), key=lambda n: (len(n), n.upper()))
    data = dict(streams)
    sectors = []
    fat = []

    def chain(payload, size=512):
        if not payload:
            return END_OF_CHAIN
        start = len(sectors)
        count = (len(payload) + size - 1) // size
        for i in range(count):
            sectors.append(payload[i * size:(i + 1) * size].ljust(size, b"\0"))
            fat.append(start + i + 1 if i < count - 1 else END_OF_CHAIN)
        return start

    # Streams under the 4096-byte cutoff live in the mini stream.
    mini = bytearray()
    mini_fat = []
    starts = {}
    for name in names:
        payload = data[name]
        if len(payload) >= 4096:
            continue
        if not payload:
            starts[name] = END_OF_CHAIN
            continue
        first = len(mini) // 64
        count = (len(payload) + 63) // 64
        mini_fat.extend(first + i + 1 if i < count - 1 else END_OF_CHAIN for i in range(count))
        mini += payload.ljust(count * 64, b"\0")
        starts[name] = first
    mini_start = chain(bytes(mini))
    for name in names:
        if len(data[name]) >= 4096:
            starts[name] = chain(data[name])
    mini_fat_bytes = b"".join(struct.pack("<I", v) for v in mini_fat)
    mini_fat_bytes += struct.pack("<I", FREE) * ((-len(mini_fat)) % 128)
    mini_fat_start = chain(mini_fat_bytes) if mini_fat else END_OF_CHAIN

    # The directory: the root, then the streams as a balanced binary tree.
    entries = [None] * (len(names) + 1)

    def tree(lo, hi):
        if lo >= hi:
            return NO_STREAM
        mid = (lo + hi) // 2
        entries[mid + 1] = (names[mid], tree(lo, mid), tree(mid + 1, hi))
        return mid + 1

    root_child = tree(0, len(names))

    def entry(name, kind, left, right, child, start, size, entry_clsid=bytes(16)):
        encoded = (name + "\0").encode("utf-16le")
        return (encoded.ljust(64, b"\0") + struct.pack("<HBB", len(encoded), kind, 1)
                + struct.pack("<III", left, right, child) + entry_clsid
                + bytes(4 + 16) + struct.pack("<IQ", start, size))

    directory = entry("Root Entry", 5, NO_STREAM, NO_STREAM, root_child, mini_start,
                      len(mini), clsid)
    for name, left, right in entries[1:]:
        directory += entry(name, 2, left, right, NO_STREAM, starts[name], len(data[name]))
    while len(directory) % 512:
        directory += ("".encode("utf-16le").ljust(64, b"\0") + struct.pack("<HBB", 0, 0, 0)
                      + struct.pack("<III", NO_STREAM, NO_STREAM, NO_STREAM) + bytes(16 + 4 + 16)
                      + struct.pack("<IQ", 0, 0))
    directory_start = chain(directory)

    # FAT sectors go last; they map themselves too.
    fat_count = 1
    while len(sectors) + fat_count > fat_count * 128:
        fat_count += 1
    fat_start = len(sectors)
    fat.extend([FAT_SECTOR] * fat_count)
    fat += [FREE] * (fat_count * 128 - len(fat))
    fat_bytes = b"".join(struct.pack("<I", v) for v in fat)
    for i in range(fat_count):
        sectors.append(fat_bytes[i * 512:(i + 1) * 512])
    assert fat_count <= 109
    difat = [fat_start + i for i in range(fat_count)] + [FREE] * (109 - fat_count)
    header = (bytes.fromhex("D0CF11E0A1B11AE1") + bytes(16)
              + struct.pack("<HHHHH", 0x003E, 0x0003, 0xFFFE, 9, 6) + bytes(6)
              + struct.pack("<IIIIIIIII", 0, fat_count, directory_start, 0, 4096,
                            mini_fat_start, (len(mini_fat_bytes) // 512) if mini_fat else 0,
                            END_OF_CHAIN, 0)
              + b"".join(struct.pack("<I", v) for v in difat))
    assert len(header) == 512
    return header + b"".join(sectors)


def comp_obj(user_type, prog_id):
    """`\\x01CompObj`: the object's user type and ProgID ([MS-OLEDS] 2.3.8)."""
    def ansi(text):
        encoded = text.encode("ascii") + b"\0"
        return struct.pack("<I", len(encoded)) + encoded
    return (struct.pack("<HHI", 1, 0xFFFE, 0x0A03) + struct.pack("<i", -1) + bytes(16)
            + ansi(user_type) + struct.pack("<I", 0) + ansi(prog_id)
            + struct.pack("<I", 0x71B239F4) + bytes(12))


OLE_STREAM = struct.pack("<IIII", 0x02000001, 0, 0, 0) + bytes(4)

# --- [MS-XLS] BIFF8 ---------------------------------------------------------


def biff(rec_type, body=b""):
    return struct.pack("<HH", rec_type, len(body)) + body


def bof(dt):
    return biff(0x0809, struct.pack("<HHHHII", 0x0600, dt, 0x0DBB, 0x07CC, 0, 0x0006))


EOF = biff(0x000A)


def short_string(text):
    """ShortXLUnicodeString, 8-bit when every character fits."""
    if all(ord(c) < 256 for c in text):
        return struct.pack("<BB", len(text), 0) + text.encode("latin-1")
    return struct.pack("<BB", len(text), 1) + text.encode("utf-16le")


def long_string(text):
    """XLUnicodeString."""
    if all(ord(c) < 256 for c in text):
        return struct.pack("<HB", len(text), 0) + text.encode("latin-1")
    return struct.pack("<HB", len(text), 1) + text.encode("utf-16le")


def xf(ifmt):
    return biff(0x00E0, struct.pack("<HHH", 0, ifmt, 0x0001) + bytes(14))


def window1(shown):
    return biff(0x003D, struct.pack("<HHHHHHHHH", 0, 0, 0x3000, 0x2000, 0x0038, shown, 0, 1,
                                    0x0258))


def number(row, col, ixfe, value):
    return biff(0x0203, struct.pack("<HHHd", row, col, ixfe, value))


def label(row, col, ixfe, text):
    return biff(0x0204, struct.pack("<HHH", row, col, ixfe) + long_string(text))


def labelsst(row, col, ixfe, index):
    return biff(0x00FD, struct.pack("<HHHI", row, col, ixfe, index))


def dimensions(rows, cols):
    return biff(0x0200, struct.pack("<IIHHH", 0, rows, 0, cols, 0))


def boundsheet_records(globals_records, sheets):
    """Globals plus sheet substreams, each BOUNDSHEET pointing at its BOF.
    `sheets`: (name, sheet type, substream bytes)."""
    def assemble(offsets):
        out = bytearray(globals_records)
        for (name, kind, _), offset in zip(sheets, offsets):
            out += biff(0x0085, struct.pack("<IBB", offset, 0, kind) + short_string(name))
        out += EOF
        return out
    stream = assemble([0] * len(sheets))
    offsets = []
    position = len(stream)
    for _, _, substream in sheets:
        offsets.append(position)
        position += len(substream)
    stream = assemble(offsets)
    for _, _, substream in sheets:
        stream += substream
    return bytes(stream)


def chart_substream(title, series, categories, values, ixfe):
    """A chart substream: the CHARTFORMATS block (title label, one SERIESFORMAT
    per series with its literal name) and SERIESDATA (category labels under
    SIIndex 2, values under SIIndex 1, by (point, series))."""
    begin, end = biff(0x1033), biff(0x1034)

    def brai(ident, rt):
        return biff(0x1051, struct.pack("<BBHHH", ident, rt, 0, 0, 0))

    def series_text(text):
        return biff(0x100D, struct.pack("<H", 0) + short_string(text))

    out = bytearray(bof(0x0020))
    out += biff(0x1001, struct.pack("<H", 0))  # Units
    out += biff(0x1002, struct.pack("<iiii", 0, 0, 0x01E00000, 0x01400000))  # Chart
    out += begin
    for name in series:
        out += biff(0x1003, struct.pack("<HHHHHH", 1, 1, len(categories), len(categories), 1, 0))
        out += begin
        out += brai(0, 1) + series_text(name)
        out += brai(1, 2) + brai(2, 2) + brai(3, 1)
        out += biff(0x1045, struct.pack("<H", 0))  # SerToCrt
        out += end
    out += biff(0x1044, struct.pack("<HBB", 0x000A, 0, 0))  # ShtProps
    if title:
        out += biff(0x1025, bytes(32))  # Text
        out += begin
        out += biff(0x104F, bytes(20))  # Pos
        out += brai(0, 1) + series_text(title)
        out += biff(0x1027, struct.pack("<HHH", 1, 0, 0))  # ObjectLink: the chart
        out += end
    out += end
    out += dimensions(len(categories), len(series))
    out += biff(0x1065, struct.pack("<H", 2))
    for point, text in enumerate(categories):
        out += label(point, 0, 0, text)
    out += biff(0x1065, struct.pack("<H", 1))
    for index in range(len(series)):
        for point, value in enumerate(values[index]):
            out += number(point, index, ixfe, value)
    out += biff(0x1065, struct.pack("<H", 3))
    out += EOF
    return bytes(out)


def budget_workbook():
    """Excel.Sheet.8: sheet "Budget" (shown) holds a small table; "Notes" is
    empty. Strings go through the shared string table, as Excel writes them."""
    strings = ["Item", "Q1", "Q2", "Rent", "Travel", "Share"]
    sst = struct.pack("<II", 9, len(strings)) + b"".join(long_string(s) for s in strings)
    globals_records = (bof(0x0005) + biff(0x0042, struct.pack("<H", 1200)) + window1(0)
                       + biff(0x0022, struct.pack("<H", 0))
                       + biff(0x041E, struct.pack("<H", 164) + long_string('"$"#,##0'))
                       + xf(0) + xf(164) + xf(9) + biff(0x00FC, sst))
    sheet = (bof(0x0010) + dimensions(4, 3)
             + labelsst(0, 0, 0, 0) + labelsst(0, 1, 0, 1) + labelsst(0, 2, 0, 2)
             + labelsst(1, 0, 0, 3) + number(1, 1, 1, 1200) + number(1, 2, 1, 1250)
             + labelsst(2, 0, 0, 4) + number(2, 1, 1, 430) + number(2, 2, 1, 515)
             + labelsst(3, 0, 0, 5) + number(3, 1, 2, 0.35) + number(3, 2, 2, 0.4) + EOF)
    notes = bof(0x0010) + dimensions(0, 0) + EOF
    workbook = boundsheet_records(globals_records, [("Budget", 0, sheet), ("Notes", 0, notes)])
    return compound_file([
        ("\x01CompObj", comp_obj("Microsoft Excel Worksheet", "Excel.Sheet.8")),
        ("\x01Ole", OLE_STREAM),
        ("Workbook", workbook),
    ], guid("00020820-0000-0000-C000-000000000046"))


def graph_object():
    """MSGraph.Chart.8: a Workbook stream that is one chart substream, its
    number formats inside it. Values use `#,##0`."""
    chart = chart_substream("Weekly visitors", ["Web", "Store"], ["Mon", "Tue", "Wed"],
                            [[1250, 1430, 980], [310, 275, 402]], ixfe=1)
    # The formats the cached values name sit before the chart records.
    head, rest = chart[:20], chart[20:]
    workbook = head + xf(0) + xf(3) + rest
    return compound_file([
        ("\x01CompObj", comp_obj("Microsoft Graph Chart", "MSGraph.Chart.8")),
        ("\x01Ole", OLE_STREAM),
        ("Workbook", workbook),
    ], guid("00020803-0000-0000-C000-000000000046"))


def excel_chart_object():
    """Excel.Chart.8: chart sheet "Chart1" (shown) over data sheet "Sheet1"."""
    globals_records = (bof(0x0005) + biff(0x0042, struct.pack("<H", 1200)) + window1(0)
                       + xf(0) + xf(0))
    chart = chart_substream(None, ["2025", "2026"], ["Jan", "Feb"], [[12, 15], [14, 19]], 0)
    sheet = (bof(0x0010) + dimensions(3, 3) + label(0, 1, 0, "2025") + label(0, 2, 0, "2026")
             + label(1, 0, 0, "Jan") + number(1, 1, 0, 12) + number(1, 2, 0, 14)
             + label(2, 0, 0, "Feb") + number(2, 1, 0, 15) + number(2, 2, 0, 19)
             + label(4, 0, 0, "Worksheet only") + EOF)
    workbook = boundsheet_records(globals_records, [("Chart1", 2, chart), ("Sheet1", 0, sheet)])
    return compound_file([
        ("\x01CompObj", comp_obj("Microsoft Excel Chart", "Excel.Chart.8")),
        ("\x01Ole", OLE_STREAM),
        ("Workbook", workbook),
    ], guid("00020821-0000-0000-C000-000000000046"))


def equation_object():
    return compound_file([
        ("\x01CompObj", comp_obj("Microsoft Equation 3.0", "Equation.3")),
        ("Equation Native", bytes(28) + b"\x03\x01\x01\x03\x0a\x0a\x01\x08x\x00"),
    ], guid("0002CE02-0000-0000-C000-000000000046"))


# --- OpenDocument chart package, as LibreOffice embeds one ----------------

ODF_CHART = """<?xml version="1.0" encoding="UTF-8"?>
<office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" \
xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" \
xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0" \
xmlns:chart="urn:oasis:names:tc:opendocument:xmlns:chart:1.0" office:version="1.2">\
<office:body><office:chart><chart:chart chart:class="chart:bar">\
<chart:title><text:p>Quarterly revenue</text:p></chart:title>\
<chart:plot-area chart:data-source-has-labels="both"/>\
<table:table table:name="local-table">\
<table:table-header-columns><table:table-column/></table:table-header-columns>\
<table:table-columns><table:table-column table:number-columns-repeated="2"/></table:table-columns>\
<table:table-header-rows><table:table-row><table:table-cell><text:p/></table:table-cell>\
<table:table-cell office:value-type="string"><text:p>North</text:p></table:table-cell>\
<table:table-cell office:value-type="string"><text:p>South</text:p></table:table-cell>\
</table:table-row></table:table-header-rows><table:table-rows>{rows}</table:table-rows>\
</table:table></chart:chart></office:chart></office:body></office:document-content>"""


def odf_chart_object():
    rows = ""
    for quarter, north, south in [("Q1", "4.5", "3.25"), ("Q2", "5.1", "3.9"),
                                  ("Q3", "6", "4.4")]:
        rows += ('<table:table-row><table:table-cell office:value-type="string"><text:p>%s'
                 '</text:p></table:table-cell>' % quarter)
        for value in (north, south):
            rows += ('<table:table-cell office:value-type="float" office:value="%s"><text:p>%s'
                     '</text:p></table:table-cell>' % (value, value))
        rows += '<table:table-cell table:number-columns-repeated="1021"/></table:table-row>'
    rows += '<table:table-row table:number-rows-repeated="1048570"><table:table-cell/></table:table-row>'
    package = io.BytesIO()
    with zipfile.ZipFile(package, "w") as z:
        def add(name, body, method=zipfile.ZIP_DEFLATED):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = method
            z.writestr(info, body)
        add("mimetype", "application/vnd.oasis.opendocument.chart", zipfile.ZIP_STORED)
        add("content.xml", ODF_CHART.format(rows=rows))
        add("META-INF/manifest.xml",
            '<?xml version="1.0" encoding="UTF-8"?><manifest:manifest xmlns:manifest='
            '"urn:oasis:names:tc:opendocument:xmlns:manifest:1.0"><manifest:file-entry '
            'manifest:full-path="/" manifest:media-type="application/vnd.oasis.opendocument.'
            'chart"/><manifest:file-entry manifest:full-path="content.xml" '
            'manifest:media-type="text/xml"/></manifest:manifest>')
    return compound_file([
        ("\x01CompObj", comp_obj("Embedded Object", "LibreOffice.ChartDocument.1")),
        ("\x01Ole", OLE_STREAM),
        ("package_stream", package.getvalue()),
    ], guid("0DD0A57F-CF3B-4FD2-BDA4-9442719B2A73"))


# --- [MS-PPT] record stream -------------------------------------------------


def record(rec_type, body=b"", ver=0, instance=0):
    return struct.pack("<HHI", ver | instance << 4, rec_type, len(body)) + body


def container(rec_type, *children, instance=0):
    return record(rec_type, b"".join(children), 0xF, instance)


def cstring(text, instance):
    return record(0x0FBA, text.encode("utf-16le"), 0, instance)


def text_shape(spid, anchor, text):
    return container(0xF004,
                     record(0xF00A, struct.pack("<II", spid, 0x0A00), 2, 1),
                     record(0xF010, struct.pack("<hhhh", *anchor)),
                     container(0xF011, record(0x0BC3, struct.pack("<iBBH", 0, 0x0D, 0, 0))),
                     container(0xF00D, record(0x0F9F, struct.pack("<I", 0)),
                               record(0x0FA0, text.encode("utf-16le"))))


def object_shape(spid, anchor, object_id):
    return container(0xF004,
                     record(0xF00A, struct.pack("<II", spid, 0x0A10), 2, 75),
                     record(0xF010, struct.pack("<hhhh", *anchor)),
                     container(0xF011, record(0x0BC1, struct.pack("<I", object_id))))


def slide(index, title, object_id):
    base = 1024 * (index + 1)
    group = container(0xF004, record(0xF009, bytes(16), 1),
                      record(0xF00A, struct.pack("<II", base, 0x0005), 2, 0))
    drawing = container(0xF002, record(0xF008, struct.pack("<II", 3, base + 2), 0, index + 1),
                        container(0xF003, group,
                                  text_shape(base + 1, (300, 400, 5360, 900), title),
                                  object_shape(base + 2, (1100, 700, 5000, 3800), object_id)))
    slide_atom = record(0x03EF, struct.pack("<I8sIIHH", 0, bytes(8), 0, 0, 0, 0), 2)
    return container(0x03EE, slide_atom, container(0x040C, drawing))


def deck():
    objects = [  # (exObjId, subType, ProgID, menu name, storage, compressed)
        (1, 0, "LibreOffice.ChartDocument.1", "Object", odf_chart_object(), True),
        (2, 3, "Excel.Sheet.8", "Worksheet", budget_workbook(), False),
        (3, 4, "MSGraph.Chart.8", "Chart", graph_object(), True),
        (4, 14, "Excel.Chart.8", "Chart", excel_chart_object(), True),
        (5, 6, "Equation.3", "Equation", equation_object(), False),
    ]
    titles = ["Revenue", "Budget", "Visitors", "Sales", "Formula"]
    # Persist ids: 1 the document, 2-6 the slides, 7-11 the object storages.
    embeds = b""
    for object_id, sub_type, prog_id, menu, _, _ in objects:
        embeds += container(0x0FCC,
                            record(0x0FCD, struct.pack("<IBBBB", 0, 0, 0, 0, 0)),
                            record(0x0FC3, struct.pack("<IIIIII", 1, 0, object_id, sub_type,
                                                       6 + object_id, 0), 1),
                            cstring(menu, 1), cstring(prog_id, 2), cstring(menu, 3))
    slide_list = b"".join(
        record(0x03F3, struct.pack("<IIiII", 2 + i, 0, 0, 256 + i, 0)) for i in range(len(titles)))
    document_atom = record(0x03E9, struct.pack("<iiiiiiIIHHBBBB", 5760, 4320, 4320, 5760, 1, 2,
                                               0, 0, 1, 0, 0, 0, 0, 0), 1)
    document = container(0x03E8, document_atom,
                         container(0x0409, record(0x040A, struct.pack("<I", len(objects))),
                                   embeds),
                         container(0x0FF0, slide_list, instance=0),
                         record(0x03EA))
    stream = bytearray()
    offsets = {1: 0}
    stream += document
    for index, title in enumerate(titles):
        offsets[2 + index] = len(stream)
        stream += slide(index, title, objects[index][0])
    for object_id, _, _, _, storage, compressed in objects:
        offsets[6 + object_id] = len(stream)
        if compressed:
            body = struct.pack("<I", len(storage)) + zlib.compress(storage, 9)
            stream += record(0x1011, body, 0, 1)
        else:
            stream += record(0x1011, storage, 0, 0)
    ids = sorted(offsets)
    directory = struct.pack("<I", ids[0] | len(ids) << 20)
    directory += b"".join(struct.pack("<I", offsets[i]) for i in ids)
    directory_at = len(stream)
    stream += record(0x1772, directory)
    edit_at = len(stream)
    stream += record(0x0FF5, struct.pack("<IHBBIIIIHH", 256, 0, 0, 3, 0, directory_at, 1,
                                         ids[-1], 1, 0))
    user_name = b"markitai"
    current_user = record(0x0FF6, struct.pack("<IIIHHBBH", 0x14, 0xE391C05F, edit_at,
                                              len(user_name), 0x03F4, 3, 0, 0)
                          + user_name + struct.pack("<I", 8))
    return compound_file([
        ("PowerPoint Document", bytes(stream)),
        ("Current User", current_user),
    ], guid("64818D10-4F9B-11CF-86EA-00AA00B929E8"))


def main():
    output = deck()
    path = os.path.join(HERE, "embedded-objects.ppt")
    with open(path, "wb") as f:
        f.write(output)
    provenance = {
        "provenance": "Original fixture authored for Markitai Rust; project license applies. "
                      "Written record by record by generate.py with the Python standard "
                      "library from [MS-CFB], [MS-PPT], [MS-XLS] and OpenDocument 1.2; no "
                      "Office application produced or opened it.",
        "files": {"embedded-objects.ppt": {"bytes": len(output),
                                           "sha256": hashlib.sha256(output).hexdigest()}},
        "slides": {
            "1 Revenue": "LibreOffice.ChartDocument.1, compressed: title 'Quarterly revenue', "
                         "series North/South over Q1-Q3, trailing repeated empty cells/rows",
            "2 Budget": "Excel.Sheet.8, uncompressed: shown sheet Budget (SST strings, "
                        "currency and percent formats); empty sheet Notes",
            "3 Visitors": "MSGraph.Chart.8, compressed: a Workbook that is one chart "
                          "substream, title 'Weekly visitors', series Web/Store, #,##0",
            "4 Sales": "Excel.Chart.8, compressed: chart sheet Chart1 shown over Sheet1, "
                       "series 2025/2026 over Jan/Feb; Sheet1 also has a cell the chart "
                       "does not plot",
            "5 Formula": "Equation.3, uncompressed: no data, contributes nothing",
        },
    }
    with open(os.path.join(HERE, "provenance.json"), "w") as f:
        json.dump(provenance, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
