"""Radical code points in extracted text read as the ideographs they draw."""

from __future__ import annotations

import re
import unicodedata

import pytest

from markitai.utils.cjk_radicals import RADICALS, fold_cjk_radicals


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("使⽤的服务器", "使用的服务器"),  # Kangxi radical, NFKC
        ("右下⻆点击", "右下角点击"),  # supplement form, the table
        ("第⼀步，⻚面", "第一步，页面"),
        ("⺟⻳", "母龟"),  # the two supplement forms NFKC maps
        ("⺁ ⺌", "⺁ ⺌"),  # component shapes stay
        ("plain text", "plain text"),
        ("", ""),
    ],
)
def test_fold(text, expected):
    assert fold_cjk_radicals(text) == expected


def test_every_kangxi_radical_becomes_an_ideograph():
    folded = fold_cjk_radicals("".join(chr(c) for c in range(0x2F00, 0x2FD6)))
    assert all(unicodedata.name(c).startswith("CJK UNIFIED IDEOGRAPH") for c in folded)


def test_the_table_is_the_workbench_table():
    # markitai-rust web/src/lib/compare.ts RADICALS: 28 supplement forms that
    # NFKC leaves alone, each to a unified ideograph
    assert len(RADICALS) == 28
    for radical, ideograph in RADICALS.items():
        assert 0x2E80 <= ord(radical) <= 0x2EFF
        assert unicodedata.normalize("NFKC", radical) == radical
        assert unicodedata.name(ideograph).startswith("CJK UNIFIED IDEOGRAPH")


def test_the_pdf_text_layer_is_folded(tmp_path):
    from pathlib import Path

    from markitai.converter.pdf import PdfConverter

    fixture = Path(__file__).parent.parent / "fixtures" / "how-to-cjk-footer.pdf"
    markdown = PdfConverter().convert(fixture, tmp_path).markdown
    assert "使用的服务器" in markdown and "右下角" in markdown
    assert not re.search("[⺀-⿟]", markdown)
