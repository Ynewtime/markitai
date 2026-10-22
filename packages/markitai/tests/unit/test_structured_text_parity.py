"""Output contracts for lightweight structured-text conversion paths."""

import json

import pytest

from markitai.converter.base import get_converter
from markitai.converter.markitdown_ext import _convert


@pytest.mark.parametrize(
    "content,encoding",
    [
        ("Name,Value\nExample,42\n", "utf-8"),
        ("a,b\n1\n2,3,4\n", "utf-8"),
        ("名字,数值\n示例,42\n", "utf-16"),
        ("Name,Value\ncafé,42\n", "cp1252"),
        ("", "utf-8"),
    ],
)
def test_csv_keeps_reference_output(tmp_path, content, encoding):
    path = tmp_path / "table.csv"
    path.write_bytes(content.encode(encoding))
    expected = _convert(path)
    converter = get_converter(path)
    assert converter is not None
    actual = converter.convert(path)
    assert actual.markdown == expected.markdown


def test_csv_utf8_bom_is_not_part_of_the_first_header(tmp_path):
    """MarkItDown keeps the BOM Excel's "CSV UTF-8" writes in the first
    header cell, an invisible character in the table and its column name."""
    path = tmp_path / "table.csv"
    path.write_bytes("Name,Value\r\n例子,42\r\n".encode("utf-8-sig"))
    converter = get_converter(path)
    assert converter is not None
    assert converter.convert(path).markdown == (
        "| Name | Value |\n| --- | --- |\n| 例子 | 42 |"
    )


@pytest.mark.parametrize(
    "content,markdown",
    [
        # MarkItDown writes these cells as they are: the newline splits the
        # row, the "|" adds a column
        (
            'Name,Value\n"Line\nbreak","a,b"\n',
            "| Name | Value |\n| --- | --- |\n| Line<br>break | a,b |",
        ),
        (
            "a|b,c\n d , e \n\n",
            "| a\\|b | c |\n| --- | --- |\n| d | e |\n|  |  |",
        ),
    ],
)
def test_csv_cells_stay_on_their_row(tmp_path, content, markdown):
    path = tmp_path / "table.csv"
    path.write_text(content, encoding="utf-8", newline="")
    converter = get_converter(path)
    assert converter is not None
    assert converter.convert(path).markdown == markdown


@pytest.mark.parametrize(
    "source", [["# Notebook\n", "\n", "Body"], "# Notebook\n\nBody"]
)
@pytest.mark.parametrize("metadata", [{}, {"title": "Metadata title"}])
def test_notebook_preserves_markdown_code_raw_cells_and_title(
    tmp_path, source, metadata
):
    path = tmp_path / "note.ipynb"
    path.write_text(
        json.dumps(
            {
                "nbformat": 4,
                "nbformat_minor": 5,
                "metadata": metadata,
                "cells": [
                    {"cell_type": "markdown", "source": source},
                    {"cell_type": "code", "source": ["print('你好')\n"], "outputs": []},
                    {"cell_type": "raw", "source": "Literal text\n\n\n"},
                ],
            }
        )
    )
    expected = _convert(path)
    converter = get_converter(path)
    assert converter is not None
    actual = converter.convert(path)
    assert actual.markdown == expected.markdown
    assert actual.metadata.get("title") == expected.metadata.get("title")


def test_malformed_notebook_is_reported_as_failure(tmp_path):
    from markitai.utils.errors import ConversionError

    path = tmp_path / "bad.ipynb"
    path.write_text('{"cells": null}')
    converter = get_converter(path)
    assert converter is not None
    with pytest.raises((ConversionError, ValueError)):
        converter.convert(path)


def test_csv_row_longer_than_the_header_keeps_its_cells(tmp_path):
    """A row longer than the header lost its last cells ("4" below); the
    table is now as wide as its longest row, as MarkItDown 0.1.9, the TSV
    reader and markitai-rust write it."""
    path = tmp_path / "ragged.csv"
    path.write_text("a,b\n1\n2,3,4\n", encoding="utf-8")
    converter = get_converter(path)
    assert converter is not None
    assert converter.convert(path).markdown == (
        "| a | b |  |\n| --- | --- | --- |\n| 1 |  |  |\n| 2 | 3 | 4 |"
    )
