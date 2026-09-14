"""Output contracts for lightweight structured-text conversion paths."""

import json

import pytest

from markitai.converter.base import get_converter
from markitai.converter.markitdown_ext import _convert


@pytest.mark.parametrize(
    "content,encoding",
    [
        ("Name,Value\nExample,42\n", "utf-8"),
        ('Name,Value\n"Line\nbreak","a,b"\n', "utf-8"),
        ("a,b\n1\n2,3,4\n", "utf-8"),
        ("名字,数值\n示例,42\n", "utf-16"),
        ("Name,Value\ncafé,42\n", "cp1252"),
        ("Name,Value\r\n例子,42\r\n", "utf-8-sig"),
        ("", "utf-8"),
        ("a|b,c\n d , e \n\n", "utf-8"),
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
