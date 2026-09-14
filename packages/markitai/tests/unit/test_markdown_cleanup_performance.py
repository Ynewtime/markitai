"""Keep cleanup semantics while avoiding irrelevant work on ordinary lines."""

import random
import re
from unittest.mock import patch

import pytest

from markitai.utils.text import (
    clean_residual_placeholders,
    fix_malformed_image_refs,
    normalize_markdown_whitespace,
)


def reference_whitespace(content):
    """Former line-by-line implementation used as a differential oracle."""
    content = fix_malformed_image_refs(content)
    lines = [line.rstrip() for line in content.split("\n")]
    result = []
    fence_char = None
    fence_size = 0
    for i, line in enumerate(lines):
        fence = re.match(r"^(`{3,}|~{3,})", line)
        if fence:
            marker = fence.group(1)
            if fence_char is None:
                fence_char, fence_size = marker[0], len(marker)
            elif marker[0] == fence_char and len(marker) >= fence_size:
                fence_char, fence_size = None, 0
        header = bool(re.match(r"^#{1,6}(\s|$)", line))
        slide = bool(re.match(r"^<!--\s*Slide\s+(number:\s*)?\d+\s*-->", line))
        if fence_char is None and (header or slide):
            if result and result[-1] != "":
                result.append("")
            result.append(line)
            if i + 1 < len(lines) and lines[i + 1] != "":
                result.append("")
        else:
            result.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(result)).strip() + "\n"


def test_plain_table_does_not_run_regex_matches_per_line():
    content = "# Table\n" + "| ordinary | data |\n" * 1000
    expected = reference_whitespace(content)
    with patch("markitai.utils.text.re.match", wraps=re.match) as matching:
        assert normalize_markdown_whitespace(content) == expected
    assert matching.call_count <= 3


@pytest.mark.parametrize(
    "function", [clean_residual_placeholders, fix_malformed_image_refs]
)
def test_absent_placeholder_or_image_does_not_run_repair_regexes(function):
    content = "# A complete document\n\nordinary text\n"
    with patch(
        "markitai.utils.text.re.sub", side_effect=AssertionError("Irrelevant repair")
    ):
        assert function(content) == content


def test_generated_fences_headers_images_and_whitespace_match_reference():
    rng = random.Random(20260922)
    lines = [
        "",
        " ",
        "\t",
        "text  ",
        "## Header",
        "#",
        "####### Seven",
        "#tag",
        "#!bin",
        "#\tTab",
        "#\vVertical",
        "#\u2028Unicode",
        "   # Indented",
        "```",
        "````lang",
        "`````",
        "~~~",
        "~~~~",
        "   ```",
        "``` trailing",
        "# inside code",
        "<!-- Slide 1 -->",
        "<!--Slide number: 2 -->",
        "<!-- Slide\t3 -->",
        "<!-- Slide number: 4 -->tail",
        "<!-- Page number: 1 -->",
        "<!-- ordinary -->",
        "![a]![b](x.png)",
        "![a]()",
        "![a](x.png)))",
        "a\r",
        "tail\u00a0\u2003",
    ]
    for _ in range(3000):
        content = "\n".join(rng.choices(lines, k=rng.randrange(1, 60)))
        assert normalize_markdown_whitespace(content) == reference_whitespace(content)


def test_already_bounded_blank_lines_skip_the_collapse_scan():
    content = "# Heading\n\nordinary text\n"
    with patch(
        "markitai.utils.text.re.sub",
        side_effect=AssertionError("No excess blank lines"),
    ):
        assert normalize_markdown_whitespace(content) == content
