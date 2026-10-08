"""Local text documents decode like fetched bodies (BOM, <meta>, detection)."""

from __future__ import annotations

from pathlib import Path

import pytest

from markitai.converter.base import get_converter

_CHINESE = "这是一个用于测试编码检测的中文段落，包含足够多的汉字。" * 3


def _convert(path: Path) -> str:
    converter = get_converter(path)
    assert converter is not None
    return converter.convert(path).markdown


def test_latin1_text_file_converts_instead_of_raising(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_bytes("Café crème, naïve résumé.\r\nSecond line.\n".encode("latin-1"))

    assert _convert(path) == "Café crème, naïve résumé.\nSecond line.\n"


def test_html_declaring_gbk_is_not_mojibake(tmp_path: Path) -> None:
    path = tmp_path / "page.html"
    html = (
        '<html><head><meta charset="gbk"><title>测试页面</title></head>'
        f"<body><article><h1>标题</h1><p>{_CHINESE}</p></article></body></html>"
    )
    path.write_bytes(html.encode("gbk"))

    markdown = _convert(path)

    assert "汉字" in markdown
    assert "\ufffd" not in markdown


@pytest.mark.parametrize(
    ("name", "template"),
    [
        ("data.xml", "<root><item>{}</item></root>"),
        ("doc.rst", "Title\n=====\n\n{}\n"),
        ("doc.org", "* Title\n\n{}\n"),
        ("doc.tex", "\\begin{{document}}\n{}\n\\end{{document}}\n"),
    ],
)
def test_gbk_markup_is_detected(tmp_path: Path, name: str, template: str) -> None:
    path = tmp_path / name
    path.write_bytes(template.format(_CHINESE).encode("gbk"))

    markdown = _convert(path)

    assert "汉字" in markdown
    assert "\ufffd" not in markdown


def test_csv_without_a_detected_encoding_keeps_its_rows(tmp_path: Path) -> None:
    """The detector finds nothing here; the table used to be the word None."""
    path = tmp_path / "table.csv"
    path.write_bytes(b"name,val\nx,\x81\x00\xff\xfe\x9d\x00\x01\n")

    markdown = _convert(path)

    assert markdown.startswith("| name | val |")
    assert "None" not in markdown
