"""Spreadsheet formatting parity, with real Excel readers and reference renderer."""

import datetime

import pandas as pd
import pytest

from markitai.converter.office import OfficeConverter, XlsxConverter


@pytest.mark.parametrize(
    "rows",
    [
        [["Name", "Value"], ["Example", 42], ["空值", None]],
        [["dup", "dup", None], [1.25, 1e-9, 1000000000000]],
        [["date", "bool"], [datetime.datetime(2024, 1, 2), True], [None, False]],
        [
            ["Markdown", "HTML"],
            ["*a* _b_ | `c`", "<b>x</b> & <script>"],
            [" a\nb\t c ", "&amp;"],
        ],
        [["NA", "values"], ["NA", "NaN"], ["NULL", "001"]],
        [],
    ],
)
def test_xlsx_matches_reference_for_cell_semantics(tmp_path, rows):
    from openpyxl import Workbook

    path = tmp_path / "cells.xlsx"
    book = Workbook()
    sheet = book.active
    assert sheet is not None
    for row in rows:
        sheet.append(row)
    book.create_sheet("Empty")
    book.save(path)
    assert (
        XlsxConverter().convert(path).markdown
        == OfficeConverter().convert(path).markdown
    )


@pytest.mark.parametrize(
    "data",
    [
        {},
        {"a": [1, 2], "b": ["*x*", "a|b"]},
        {"中文": ["a\nb", "<b>&nbsp;</b>"], "": [None, False]},
    ],
)
def test_dataframe_table_avoids_dom_but_matches_reference(data):
    from unittest.mock import patch

    from markitdown.converters._html_converter import HtmlConverter

    from markitai.converter.xlsx import dataframe_table_markdown

    html = pd.DataFrame(data).to_html(index=False)
    expected = HtmlConverter().convert_string(html).markdown
    with patch("bs4.BeautifulSoup.__init__", side_effect=AssertionError("DOM parse")):
        assert dataframe_table_markdown(html) == expected


def test_mislabeled_docx_retains_generic_detection(fixtures_dir, tmp_path):
    path = tmp_path / "actually-word.xlsx"
    path.write_bytes((fixtures_dir / "sample.docx").read_bytes())
    assert (
        XlsxConverter().convert(path).markdown
        == OfficeConverter().convert(path).markdown
    )


def test_table_cell_escaping_matches_reference_for_generated_strings():
    import random

    from markitdown.converters._html_converter import HtmlConverter

    from markitai.converter.xlsx import dataframe_table_markdown

    rng = random.Random(20260914)
    alphabet = "abc中日é<&>;'\"_*`|\\\r\n\t \u00a0\u2003"
    frame = pd.DataFrame(
        {
            f"column_{i}": ["".join(rng.choices(alphabet, k=40)) for _ in range(50)]
            for i in range(4)
        }
    )
    html = frame.to_html(index=False)
    assert (
        dataframe_table_markdown(html) == HtmlConverter().convert_string(html).markdown
    )


@pytest.mark.parametrize(
    "frame",
    [
        pd.DataFrame(),
        pd.DataFrame(columns=["a", "b"]),
        pd.DataFrame({0: [1, 2], "word": ["中文", "value"]}),
        pd.DataFrame({"n": [1.25, 1e-9, None, 1e20]}),
        pd.DataFrame({"date": [datetime.datetime(2024, 1, 2), None]}),
        pd.DataFrame({"bool": [True, False, None]}),
        pd.DataFrame({" a\nb ": [" a\nb\t c ", "x  y   z", "&nbsp; <b>*x*</b>"]}),
        pd.DataFrame({"long": ["a" * 1000, "é\u00a0中\u2003x"]}),
        pd.DataFrame({"mixed": [1, "foo", None, True, datetime.date(2024, 1, 2)]}),
        pd.DataFrame({"list": [[1, 2], {"a": 1}]}),
    ],
)
def test_direct_dataframe_render_preserves_pandas_formatting(frame):
    from markitai.converter.xlsx import dataframe_markdown, dataframe_table_markdown

    assert dataframe_markdown(frame) == dataframe_table_markdown(
        frame.to_html(index=False)
    )


def test_direct_dataframe_does_not_serialize_or_parse_html(monkeypatch):
    from html.parser import HTMLParser

    from markitai.converter.xlsx import dataframe_markdown, dataframe_table_markdown

    frame = pd.DataFrame({"a": [1, 2], "b": ["a|b", "<x>&amp;"]})
    expected = dataframe_table_markdown(frame.to_html(index=False))

    def forbidden(*args, **kwargs):
        raise AssertionError("HTML serialization or parsing")

    monkeypatch.setattr(pd.DataFrame, "to_html", forbidden)
    monkeypatch.setattr(HTMLParser, "feed", forbidden)
    assert dataframe_markdown(frame) == expected


@pytest.mark.parametrize("precision", [2, 6, 12])
def test_direct_dataframe_respects_pandas_display_options(precision):
    from markitai.converter.xlsx import dataframe_markdown, dataframe_table_markdown

    frame = pd.DataFrame({"n": [1.2345678901, 1e-20], "text": ["x" * 100, "y"]})
    with pd.option_context("display.precision", precision, "display.max_colwidth", 10):
        assert dataframe_markdown(frame) == dataframe_table_markdown(
            frame.to_html(index=False)
        )


def test_direct_dataframe_retains_multiindex_fallback():
    from markitai.converter.xlsx import dataframe_markdown, dataframe_table_markdown

    frame = pd.DataFrame(
        [[1, 2]], columns=pd.MultiIndex.from_tuples([("a", "b"), ("a", "c")])
    )
    assert dataframe_markdown(frame) == dataframe_table_markdown(
        frame.to_html(index=False)
    )


def test_direct_dataframe_generated_text_parity():
    import random

    from markitai.converter.xlsx import dataframe_markdown, dataframe_table_markdown

    rng = random.Random(20260917)
    alphabet = "abc中日é<&>;'\"_*`|\\\r\n\t \u00a0\u2003"
    frame = pd.DataFrame(
        {
            f"column_{i}": ["".join(rng.choices(alphabet, k=40)) for _ in range(100)]
            for i in range(8)
        }
    )
    assert dataframe_markdown(frame) == dataframe_table_markdown(
        frame.to_html(index=False)
    )


def test_direct_dataframe_tracks_installed_formatter_space_semantics(monkeypatch):
    from pandas.io.formats.html import HTMLFormatter

    from markitai.converter.xlsx import dataframe_markdown, dataframe_table_markdown

    original = HTMLFormatter._write_cell

    def without_nbsp(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.elements[-1] = self.elements[-1].replace("&nbsp;", " ")

    # pandas 2.2 emits literal repeated spaces; pandas 3 uses NBSP entities.
    # Both are supported, and the installed formatter remains the oracle.
    monkeypatch.setattr(HTMLFormatter, "_write_cell", without_nbsp)
    frame = pd.DataFrame({"a  b": ["x  y   z"]})
    assert dataframe_markdown(frame) == dataframe_table_markdown(
        frame.to_html(index=False)
    )
