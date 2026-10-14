"""Cold PPTX table rendering stays lightweight without changing text or adapters."""

import subprocess
import sys
from typing import Any, cast

import pytest


def test_cold_table_renderer_matches_default_without_html_dependencies():
    code = r"""
import html
import random
import sys
from markitai.converter.html_table import html_table_markdown

rng = random.Random(20260933)
alphabet = "abc中日é<&>;'\"_*`|\\\r\n\t \u00a0\u2003"
tables = []
for _ in range(1000):
    values = ["prefix" + "".join(rng.choices(alphabet, k=40)) for _ in range(3)]
    source = "<table><thead><tr><th>col_*</th></tr></thead><tbody>" + "".join(
        "<tr><td>" + html.escape(v) + "</td></tr>" for v in values
    ) + "</tbody></table>"
    actual = html_table_markdown(source)
    tables.append((source, actual))
assert "markdownify" not in sys.modules
assert "bs4" not in sys.modules
from markitai.webextract.markdownify_compat import CompatibleMarkdownConverter
reference = CompatibleMarkdownConverter()
for source, actual in tables:
    assert actual == reference.convert(source).strip(), (source, actual)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("customization", ["defaults", "adapter"])
def test_table_renderer_retains_loaded_customizations(monkeypatch, customization):
    import markdownify

    from markitai.converter.html_table import html_table_markdown
    from markitai.webextract.markdownify_compat import CompatibleMarkdownConverter

    if customization == "defaults":
        defaults = cast(Any, markdownify.MarkdownConverter).DefaultOptions
        monkeypatch.setattr(defaults, "escape_misc", True)
        monkeypatch.setattr(defaults, "escape_asterisks", False)
    else:
        monkeypatch.setattr(
            CompatibleMarkdownConverter,
            "escape",
            lambda _self, text, _tags: text.replace("*", "CUSTOM"),
        )
    source = "<table><thead><tr><th>a*</th></tr></thead><tr><td>&lt;x&gt; _*</td></tr></table>"
    assert (
        html_table_markdown(source)
        == CompatibleMarkdownConverter().convert(source).strip()
    )
