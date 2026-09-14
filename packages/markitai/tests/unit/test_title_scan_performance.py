"""Title scanning must preserve heading/fence semantics without splitting a document."""

import random
import re

import pytest

from markitai.utils.frontmatter import (
    _find_first_content_line,
    _find_heading,
    extract_title_from_content,
)


def reference_heading(content: str, level: int) -> str | None:
    """Frozen former scanner, kept as an independent differential oracle."""
    inside = False
    for line in content.split("\n"):
        if re.match(r"^(`{3,}|~{3,})", line):
            inside = not inside
            continue
        if not inside:
            match = re.match(rf"^#{{{level}}}(?!#)\s+(.+)$", line)
            if match:
                return match.group(1).strip()
    return None


@pytest.mark.parametrize("level", [1, 2, 3, 6])
def test_generated_heading_and_fence_boundaries_match_reference(level):
    rng = random.Random(20260919)
    lines = [
        "plain text",
        "",
        "#",
        "##",
        "# ",
        "#  ",
        "##\t",
        "##\t title ",
        "#\r",
        "# \r",
        "## Head\r",
        "#\vform feed",
        "#\u2028unicode",
        "#\u00a0title",
        "### Third",
        "###### Sixth",
        "## # nested",
        "```",
        "````python",
        "~~~",
        "~~~~ x",
        "``` inline ```",
        "~~",
        "``",
        "   ```",
        "   # indented",
        "# ``` literal",
        "###",
        "--> comment",
    ]
    for _ in range(1000):
        content = "\n".join(rng.choices(lines, k=rng.randrange(1, 50)))
        assert _find_heading(content, level) == reference_heading(content, level)


def test_scan_does_not_allocate_all_lines_for_a_long_document():
    class WithoutSplit(str):
        def split(self, *args, **kwargs):
            raise AssertionError("Allocated all document lines")

    content = WithoutSplit("## Sheet\n" + "| ordinary | data |\n" * 10000)
    assert _find_heading(content, 1) is None
    assert _find_heading(content, 2) == "Sheet"


@pytest.mark.parametrize("empty", ["", "\n\t \r", "\u00a0\u2003\u2028"])
def test_whitespace_only_title_retains_fallback(empty):
    assert extract_title_from_content(empty, "fallback") == "fallback"


def test_heading_after_fenced_code_preserves_h1_priority():
    content = "## Earlier\n```\n# Ignored\n~~~\n# Actual\n"
    assert extract_title_from_content(content) == "Actual"


def test_blank_first_heading_retains_existing_fallback_semantics():
    assert extract_title_from_content("#  \n# Later\n## Section") == "Section"


def test_first_line_fallback_does_not_split_the_remaining_document():
    class WithoutSplit(str):
        def split(self, *args, **kwargs):
            raise AssertionError("Allocated all remaining document lines")

    content = WithoutSplit(
        "<!-- comment -->\n![image](a.png)\n---\nUseful title\n" + "data\n" * 10000
    )
    assert _find_first_content_line(content) == "Useful title"


def test_inline_dashes_do_not_trigger_a_full_frontmatter_regex_scan(monkeypatch):
    from markitai.utils import frontmatter

    class NoScan:
        def sub(self, *args, **kwargs):
            raise AssertionError("Scanned a document without a frontmatter opener")

    monkeypatch.setattr(frontmatter, "FRONTMATTER_PATTERN", NoScan())
    content = "## Sheet\n| --- | --- |\n" + "ordinary row\n" * 10000
    assert frontmatter._strip_frontmatter(content) == content


def test_frontmatter_candidate_filter_matches_original_regex():
    from markitai.utils.frontmatter import FRONTMATTER_PATTERN, _strip_frontmatter

    rng = random.Random(20260920)
    lines = [
        "",
        " ",
        "\t",
        "\u00a0",
        "---",
        " --- ",
        "\t---",
        "-----",
        "--- x",
        "---\r",
        "word",
        "prefix ---",
        "| --- | --- |",
        "title: Name",
        "body",
        "# Title",
    ]
    for _ in range(1000):
        content = "\n".join(rng.choices(lines, k=rng.randrange(1, 25)))
        assert _strip_frontmatter(content) == FRONTMATTER_PATTERN.sub(
            "", content, count=1
        )


def test_workflow_heading_title_avoids_splitting_all_lines():
    from markitai.workflow.helpers import _extract_heading_title

    class WithoutSplit(str):
        def split(self, *args, **kwargs):
            raise AssertionError("Allocated all lines for a first heading")

    assert (
        _extract_heading_title(WithoutSplit("## Sheet\n" + "row\n" * 10000)) == "Sheet"
    )


def test_workflow_heading_rules_keep_first_heading_and_leading_whitespace():
    from markitai.workflow.helpers import _extract_heading_title

    rng = random.Random(20260921)
    lines = [
        "",
        "\t",
        "  # Leading",
        "#tag",
        "##No space",
        "###",
        "## **Bold**",
        "word",
        "```",
        "# Inside",
        "## First",
        "# Later",
        "#\tTab",
        "###\r",
    ]
    for _ in range(1000):
        text = "\n".join(rng.choices(lines, k=rng.randrange(1, 30)))
        expected = ""
        for line in text.strip().split("\n"):
            if line.startswith("# ") or (
                len(line) > 1 and line[0] == "#" and line[1] in "# "
            ):
                expected = line.lstrip("#").strip().replace("**", "").strip()
                if expected:
                    break
        assert _extract_heading_title(text) == expected
