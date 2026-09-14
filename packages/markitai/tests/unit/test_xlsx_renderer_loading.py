"""Cold table rendering stays lightweight without changing text or adapters."""

import subprocess
import sys
from typing import Any, cast

import pytest


@pytest.mark.parametrize("kind", ["html", "plain"])
def test_cold_table_renderer_matches_default_without_html_dependencies(kind):
    code = r"""
import html
import random
import sys
from markitai.converter.xlsx import dataframe_table_markdown, plain_table_markdown

rng = random.Random(20260933)
alphabet = "abc中日é<&>;'\"_*`|\\\r\n\t \u00a0\u2003"
tables = []
for _ in range(1000):
    values = ["prefix" + "".join(rng.choices(alphabet, k=40)) for _ in range(3)]
    if sys.argv[1] == "plain":
        values = [v.translate(str.maketrans("", "", "\n\r\t ")) for v in values]
    source = "<table><thead><tr><th>col_*</th></tr></thead><tbody>" + "".join(
        "<tr><td>" + html.escape(v) + "</td></tr>" for v in values
    ) + "</tbody></table>"
    actual = (dataframe_table_markdown(source) if sys.argv[1] == "html"
              else plain_table_markdown([["col_*"], *[[v] for v in values]]))
    assert actual is not None
    tables.append((source, actual))
assert "markdownify" not in sys.modules
assert "bs4" not in sys.modules
from markitai.webextract.markdownify_compat import CompatibleMarkdownConverter
reference = CompatibleMarkdownConverter()
for source, actual in tables:
    assert actual == reference.convert(source).strip(), (source, actual)
"""
    result = subprocess.run(
        [sys.executable, "-c", code, kind], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("customization", ["defaults", "adapter"])
def test_table_renderer_retains_loaded_customizations(monkeypatch, customization):
    import markdownify

    from markitai.converter.xlsx import dataframe_table_markdown
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
        dataframe_table_markdown(source)
        == CompatibleMarkdownConverter().convert(source).strip()
    )
