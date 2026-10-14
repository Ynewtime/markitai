"""Invented spreadsheet fixture: merged banner rows, header-less column, empty cells."""
import sys
from openpyxl import Workbook

book = Workbook()
s = book.active
s.title = "Day plan"
rows = [
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
for r in rows:
    s.append(r)
for r in (1, 2, 3, 8):
    s.merge_cells(start_row=r, start_column=1, end_row=r, end_column=4)
book.save(sys.argv[1])
