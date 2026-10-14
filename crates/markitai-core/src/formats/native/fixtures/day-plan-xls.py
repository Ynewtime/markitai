"""Invented XLS fixture: merged banner rows, header-less first column, empty cells."""
import sys
import xlwt

book = xlwt.Workbook()
s = book.add_sheet("Day plan")
rows = [
    ["Spring Fair 2026 · Day Plan", None, None, None],
    ["Item", "Time", "Task", "Note"],
    ["Setup crew", "07:30", "Unload tables", None],
    ["Stage", None, "Sound check", None],
    ["Afternoon", None, None, None],
    [None, "Time", "Task", "Note"],
    ["Raffle", "14:00", "Draw tickets", "NA"],
    ["Close", 17.5, None, 3],
]
for r, row in enumerate(rows):
    if r in (0, 4):
        s.write_merge(r, r, 0, 3, row[0])
        continue
    for c, value in enumerate(row):
        if value is not None:
            s.write(r, c, value)
book.save(sys.argv[1])
