"""Reading order for OCR boxes (markitai.ocr_layout).

The boxes RapidOCR returns are sorted top to bottom. Joined in that order a
two-column page interleaved its columns, vertical Chinese came out
column-reversed, an upside-down scan came out bottom line first, and a table
became one cell per line.
"""

from __future__ import annotations

from markitai.ocr_layout import layout_ocr_text

LINE = 20.0  # line height used by the synthetic pages


def _line(text: str, x0: float, y0: float, width: float) -> tuple[str, tuple]:
    return text, (x0, y0, x0 + width, y0 + LINE)


def _layout(rows: list[tuple[str, tuple]], **kwargs) -> str:
    texts = [text for text, _ in rows]
    bounds = [bound for _, bound in rows]
    return layout_ocr_text(texts, bounds, **kwargs)


def test_single_column_keeps_order_and_breaks_paragraphs_at_gaps() -> None:
    rows = [
        _line("First paragraph line one.", 50, 0, 400),
        _line("First paragraph line two.", 50, 25, 400),
        _line("Second paragraph starts here.", 50, 80, 400),
    ]

    assert _layout(rows) == (
        "First paragraph line one.\n"
        "First paragraph line two.\n"
        "\n"
        "Second paragraph starts here."
    )


def test_two_columns_are_read_column_by_column() -> None:
    rows = []
    for n in range(1, 5):  # recognition order interleaves the columns
        rows.append(_line(f"Left column sentence {n}.", 50, n * 25, 300))
        rows.append(_line(f"Right column sentence {n}.", 450, n * 25, 300))

    text = _layout(rows)

    left = [f"Left column sentence {n}." for n in range(1, 5)]
    right = [f"Right column sentence {n}." for n in range(1, 5)]
    assert text == "\n".join(left) + "\n\n" + "\n".join(right)


def test_a_title_spanning_the_gutter_is_read_before_the_columns() -> None:
    rows = [_line("A title across the whole page width", 50, 0, 700)]
    for n in range(1, 4):
        rows.append(_line(f"Left column sentence {n}.", 50, 30 + n * 25, 300))
        rows.append(_line(f"Right column sentence {n}.", 450, 30 + n * 25, 300))
    rows.append(_line("A footer across the whole page width", 50, 200, 700))

    blocks = _layout(rows).split("\n\n")

    assert blocks[0] == "A title across the whole page width"
    assert blocks[1].startswith("Left column sentence 1.")
    assert blocks[2].startswith("Right column sentence 1.")
    assert blocks[-1] == "A footer across the whole page width"


def test_vertical_chinese_reads_right_to_left() -> None:
    # Columns at x = 260 (first), 230, 200 (last); each box is tall and thin
    rows = [
        ("床前明月光", (200, 0, 220, 100)),
        ("疑是地上霜", (230, 0, 250, 100)),
        ("举头望明月", (260, 0, 280, 100)),
    ]
    rows.reverse()  # arrival order must not matter

    assert _layout(rows) == "举头望明月\n疑是地上霜\n床前明月光"


def test_vertical_paragraphs_break_only_at_wide_gaps() -> None:
    rows = [
        ("第一段第一列", (300, 0, 320, 120)),
        ("第一段第二列", (270, 0, 290, 120)),  # 10 px apart: same paragraph
        ("第二段第一列", (190, 0, 210, 120)),  # 60 px apart: new paragraph
    ]

    assert _layout(rows) == "第一段第一列\n第一段第二列\n\n第二段第一列"


def test_upside_down_page_reads_top_line_first() -> None:
    # In scan coordinates the first line sits at the bottom of the page
    rows = [
        _line("Last line of the page.", 50, 0, 400),
        _line("Middle line of the page.", 50, 25, 400),
        _line("First line of the page.", 50, 50, 400),
    ]

    assert _layout(rows, upside_down=True).splitlines() == [
        "First line of the page.",
        "Middle line of the page.",
        "Last line of the page.",
    ]


def test_rows_of_spaced_cells_become_a_markdown_table() -> None:
    rows = [_line("Quarterly figures", 50, 0, 300)]
    for n, cells in enumerate(
        [("Quarter", "Revenue", "Growth"), ("Q1", "120", "5%"), ("Q2", "150", "25%")]
    ):
        y = 40 + n * 25
        rows += [_line(cell, 50 + i * 200, y, 80) for i, cell in enumerate(cells)]

    assert _layout(rows) == (
        "Quarterly figures\n"
        "\n"
        "| Quarter | Revenue | Growth |\n"
        "| --- | --- | --- |\n"
        "| Q1 | 120 | 5% |\n"
        "| Q2 | 150 | 25% |"
    )


def test_adjacent_cjk_boxes_join_without_a_space() -> None:
    rows = [("第一季度", (0, 0, 80, 20)), ("收入增长", (82, 0, 160, 20))]

    assert _layout(rows) == "第一季度收入增长"


def test_a_visible_gap_between_cjk_boxes_keeps_its_space() -> None:
    rows = [("第一章", (0, 0, 60, 20)), ("日本語のテスト", (80, 0, 220, 20))]

    assert _layout(rows) == "第一章 日本語のテスト"


def test_a_table_is_not_mistaken_for_columns() -> None:
    """Short cells line up in columns too; they must stay row by row."""
    rows = []
    for n in range(4):
        rows += [
            _line(f"a{n}", 50, n * 25, 40),
            _line(f"b{n}", 450, n * 25, 40),
        ]

    lines = _layout(rows).splitlines()

    assert lines[0] == "| a0 | b0 |"
    assert lines[2:] == ["| a1 | b1 |", "| a2 | b2 |", "| a3 | b3 |"]


def test_boxes_without_geometry_are_kept_at_the_end() -> None:
    assert (
        layout_ocr_text(["placed", "loose"], [(0, 0, 50, LINE), None])
        == "placed\n\nloose"
    )


# --- margins, columns, tables and vertical text that are not what they seem ----


def _paragraph_lines(y0: float, count: int, *, prefix: str) -> list[tuple[str, tuple]]:
    return [
        _line(f"{prefix} body line {n} across the page.", 100, y0 + n * 25, 600)
        for n in range(count)
    ]


def test_a_tall_margin_stamp_does_not_swallow_the_body_lines() -> None:
    """arXiv's vertical ``arXiv:… [cs.CL]`` stamp spans most of the page.

    It overlapped every line, so each body line was merged into one line
    with it and the paragraph break between them was lost.
    """
    rows = [
        *_paragraph_lines(100, 3, prefix="First"),
        *_paragraph_lines(220, 3, prefix="Second"),
        ("arXiv:2401.01234v1 [cs.CL] 2 Jan 2024", (20, 80, 45, 700)),
    ]

    text = _layout(rows)

    first = "\n".join(f"First body line {n} across the page." for n in range(3))
    second = "\n".join(f"Second body line {n} across the page." for n in range(3))
    assert text == (f"{first}\n\n{second}\n\narXiv:2401.01234v1 [cs.CL] 2 Jan 2024")


def test_a_tall_box_never_stretches_its_line_over_the_next_ones() -> None:
    """A line's extent is its own height, not the tallest box it collected."""
    rows = [
        _line("Line one of the text.", 100, 0, 400),
        ("|", (520, 0, 530, 70)),  # a tall rule that is not quite a margin
        _line("Line two of the text.", 100, 25, 400),
        _line("Line three of the text.", 100, 50, 400),
    ]

    lines = _layout(rows).splitlines()

    assert "Line two of the text." in lines
    assert "Line three of the text." in lines


def test_a_letter_address_block_is_read_before_the_body() -> None:
    """A right-aligned address at the top is not a second text column."""
    rows = [
        _line("221B Baker Street", 450, 0, 250),
        _line("London NW1 6XE", 450, 25, 250),
        _line("1 January 2026", 450, 50, 250),
        _line("Dear Dr. Watson,", 50, 120, 200),
        _line(
            "Thank you for the notes you sent about the case last week.", 50, 150, 650
        ),
        _line(
            "They were most helpful and I have read every page of them.", 50, 175, 650
        ),
        _line("for the case.", 50, 200, 150),
        _line("I will write again once the matter is settled for good.", 50, 260, 650),
        _line("Kind regards,", 50, 320, 150),
        _line("Sherlock Holmes", 50, 345, 200),
    ]

    text = _layout(rows)

    assert text.startswith("221B Baker Street\nLondon NW1 6XE\n1 January 2026\n\n")
    assert text.index("1 January 2026") < text.index("Dear Dr. Watson,")
    assert text.endswith("Kind regards,\nSherlock Holmes")


def test_a_narrow_centered_title_does_not_hide_the_columns() -> None:
    """A title narrower than the page but across the gutter is a full block.

    It covered the gutter in the x projection, so no gutter was found and
    the two columns were joined line by line.
    """
    rows = [_line("Abstract", 330, 0, 90)]
    for n in range(1, 5):
        rows.append(_line(f"Left column sentence {n}.", 50, 20 + n * 25, 300))
        rows.append(_line(f"Right column sentence {n}.", 400, 20 + n * 25, 300))
    rows.append(_line("1 Introduction", 320, 170, 110))
    for n in range(1, 4):
        rows.append(_line(f"Left body sentence {n}.", 50, 180 + n * 25, 300))
        rows.append(_line(f"Right body sentence {n}.", 400, 180 + n * 25, 300))

    blocks = _layout(rows).split("\n\n")

    assert blocks == [
        "Abstract",
        "\n".join(f"Left column sentence {n}." for n in range(1, 5)),
        "\n".join(f"Right column sentence {n}." for n in range(1, 5)),
        "1 Introduction",
        "\n".join(f"Left body sentence {n}." for n in range(1, 4)),
        "\n".join(f"Right body sentence {n}." for n in range(1, 4)),
    ]


def test_short_lines_and_side_captions_are_not_two_columns() -> None:
    """Short left lines and right-hand captions never sit side by side.

    Their x projections leave a gap, which read them as two columns: every
    caption moved after all the left-hand lines.
    """
    rows = [
        _line("A body line that runs across the whole page.", 50, 0, 650),
        _line("end of the first paragraph.", 50, 25, 220),
        _line("Figure 1. System overview", 480, 60, 220),
        _line("A short heading line", 50, 95, 220),
        _line("another short line here", 50, 120, 220),
        _line("Figure 2. Results so far", 480, 155, 220),
        _line("A closing short line.", 50, 190, 220),
        _line("the last short line.", 50, 215, 220),
        _line("Page 12 of the report", 480, 250, 220),
    ]

    lines = [line for line in _layout(rows).splitlines() if line]

    assert lines == [text for text, _ in rows]


def test_a_real_two_column_page_still_reads_column_by_column() -> None:
    """Columns that start at different heights still read left, then right."""
    rows = [_line(f"Left only sentence {n}.", 50, n * 25, 300) for n in range(3)]
    for n in range(3, 7):
        rows.append(_line(f"Left column sentence {n}.", 50, n * 25, 300))
        rows.append(_line(f"Right column sentence {n}.", 400, n * 25 + 3, 300))

    text = _layout(rows)

    left = [f"Left only sentence {n}." for n in range(3)] + [
        f"Left column sentence {n}." for n in range(3, 7)
    ]
    assert [line for line in text.splitlines() if line] == left + [
        f"Right column sentence {n}." for n in range(3, 7)
    ]


def test_form_labels_and_values_are_not_a_table() -> None:
    """Two label/value rows are text, the label and value one space apart."""
    rows = [
        ("姓名", (50, 0, 90, 20)),
        ("张三", (200, 0, 240, 20)),
        ("日期", (50, 25, 90, 45)),
        ("2026-01-01", (200, 25, 300, 45)),
    ]

    assert _layout(rows) == "姓名 张三\n日期 2026-01-01"


def test_a_hanging_indent_numbered_list_is_not_a_table() -> None:
    rows = []
    for n in range(1, 5):
        rows += [
            _line(f"1.{n}", 50, n * 25, 30),
            _line(f"The text of item {n} that runs on for a while.", 110, n * 25, 400),
        ]

    lines = _layout(rows).splitlines()

    assert "|" not in "".join(lines)
    assert lines[0] == "1.1 The text of item 1 that runs on for a while."


def test_misaligned_two_cell_rows_are_not_a_table() -> None:
    rows = [
        _line("Alpha", 50, 0, 60),
        _line("first value", 200, 0, 120),
        _line("Beta", 50, 25, 60),
        _line("second value", 320, 25, 120),
        _line("Gamma", 50, 50, 60),
        _line("third value", 150, 50, 120),
    ]

    assert "|" not in _layout(rows)


def test_aligned_two_column_rows_are_still_a_table() -> None:
    """Right-aligned amounts line up on their right edge."""
    rows = [
        _line("Item", 50, 0, 60),
        ("Amount", (300, 0, 360, 20)),
        _line("Coffee", 50, 25, 60),
        ("3.50", (320, 25, 360, 45)),
        _line("Lunch", 50, 50, 60),
        ("12.00", (310, 50, 360, 70)),
    ]

    assert _layout(rows) == (
        "| Item | Amount |\n| --- | --- |\n| Coffee | 3.50 |\n| Lunch | 12.00 |"
    )


def test_a_few_upright_short_labels_do_not_make_the_page_vertical() -> None:
    """Short vertical tags outnumber the lines but hold little of the text."""
    rows = [
        _line("The first long horizontal line of body text on the page.", 100, 0, 500),
        _line(
            "The second long horizontal line of body text on the page.", 100, 25, 500
        ),
        ("图一", (20, 100, 30, 130)),
        ("图二", (20, 200, 30, 230)),
        ("图三", (20, 300, 30, 330)),
        ("图四", (20, 400, 30, 430)),
    ]

    text = _layout(rows)

    assert text.startswith(
        "The first long horizontal line of body text on the page.\n"
        "The second long horizontal line of body text on the page."
    )


def test_vertical_latin_text_keeps_its_spaces() -> None:
    """A table scanned rotated by 90° is read as columns of cells."""
    rows = [
        ("Coffee", (300, 0, 320, 60)),
        ("3.50", (300, 80, 320, 120)),
        ("42.00", (300, 140, 320, 190)),
        ("Lunch", (270, 0, 290, 60)),
        ("12", (270, 80, 290, 100)),
        ("7.25", (270, 140, 290, 190)),
    ]

    assert _layout(rows) == "Coffee 3.50 42.00\nLunch 12 7.25"
