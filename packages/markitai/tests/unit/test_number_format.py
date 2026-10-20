"""Spreadsheet number format codes, shown as a spreadsheet shows them."""

from __future__ import annotations

import datetime

import pytest

from markitai.converter.number_format import format_cell

_ACCOUNTING = '_(* #,##0.00_);_(* (#,##0.00);_(* "-"??_);_(@_)'


@pytest.mark.parametrize(
    ("code", "value", "shown"),
    [
        ("0.00%", 0.256, "25.60%"),
        ("0%", 0.5, "50%"),
        ("#,##0.00", 1234.5, "1,234.50"),
        ("#,##0", 0, "0"),
        ("0.00", 0.125, "0.13"),  # half away from zero, not to even
        ("0.00", 5.255, "5.26"),  # at 15 digits, not the binary value
        ("0.00", -0.001, "-0.00"),
        ("0.0#", 2.5, "2.5"),
        ("0.0#", 2.567, "2.57"),
        ("#.##", 0.5, ".5"),
        ("000", 7, "007"),
        ('#,##0,"K"', 1500000, "1,500K"),
        ("0.00E+00", 12345.678, "1.23E+04"),
        ("0.00E+00", 0.000123, "1.23E-04"),
        ("##0.0E+0", 12345.678, "12.3E+3"),
        ('"$"#,##0.00_);[Red]("$"#,##0.00)', -1234.5, "($1,234.50)"),
        ('"$"#,##0.00_);[Red]("$"#,##0.00)', 1234.5, "$1,234.50"),
        (_ACCOUNTING, -1234.5, "(1,234.50)"),
        (_ACCOUNTING, 0, "-"),
        ("[$€-407] #,##0.00", 99.5, "€ 99.50"),
        ('0.00;-0.00;"zero"', 0, "zero"),
        ("[<=9999999]###-####;(###) ###-####", 5551234, "555-1234"),
        ("[<=9999999]###-####;(###) ###-####", 2125551234, "(212) 555-1234"),
        ('General" units"', 12.5, "12.5 units"),
        ("[Blue]0.0", 3, "3.0"),
        ("@", "text", "text"),
        ('"Total: "@', "x", "Total: x"),
        ("m/d/yy h:mm", datetime.datetime(2024, 3, 5, 14, 30, 15), "2024-03-05 14:30"),
        ("yyyy-mm-dd h:mm:ss", datetime.datetime(2024, 3, 5), "2024-03-05"),
        ("d-mmm-yy", datetime.datetime(2024, 3, 5), "2024-03-05"),
        ("mmmm d, yyyy", datetime.datetime(2024, 3, 5), "March 5, 2024"),
        (
            "dddd, mmmm dd, yyyy",
            datetime.datetime(2026, 3, 4),
            "Wednesday, March 04, 2026",
        ),
        ('yyyy"年"m"月"d"日"', datetime.datetime(2024, 3, 5), "2024年3月5日"),
        ("h:mm AM/PM", datetime.time(9, 5, 7), "09:05"),
        ("h:mm:ss", datetime.time(9, 5, 7, 600_000), "09:05:08"),
        ("[h]:mm:ss", datetime.timedelta(hours=27, minutes=3), "27:03:00"),
        ("[mm]:ss", datetime.timedelta(minutes=75, seconds=5), "75:05"),
    ],
)
def test_a_value_shows_through_its_format(code: str, value: object, shown: str) -> None:
    assert format_cell(code, value) == shown


@pytest.mark.parametrize(
    ("code", "value"),
    [
        ("General", 0.1),  # the caller's own General rendering
        ("", 1),
        ("# ?/?", 0.5),  # fractions are not implemented
        ("[DBNum1]0", 3),  # a locale-specific code
        ("0.00", True),
        ("0.00", "text"),  # no text section: the text as it is
        ("[h]:mm", datetime.datetime(2024, 1, 1)),
        ('"unclosed', 1),
    ],
)
def test_outside_the_grammar_is_left_to_general(code: str, value: object) -> None:
    assert format_cell(code, value) is None
