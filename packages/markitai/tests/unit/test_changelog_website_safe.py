"""Guard: CHANGELOG.md is rendered by VitePress/Vue on the website.

Vue's template compiler treats a bare ``<word>`` outside a code span as
an unclosed HTML element and fails the whole website build ("Element is
missing end tag"). This has broken the deploy twice; keep angle-bracket
tokens inside backticks.
"""

from __future__ import annotations

import re
from pathlib import Path

CHANGELOG = Path(__file__).resolve().parents[4] / "CHANGELOG.md"

# ``<`` followed by a letter starts a tag for Vue; digits (e.g. "<4-page")
# are safe. Only flag simple word-like tags — the exact shape Vue parses.
_TAG_RE = re.compile(r"<[a-zA-Z][a-zA-Z0-9-]*>")


def _strip_code_spans(line: str) -> str:
    # Remove fenced-marker runs first so odd backtick counts (```lang)
    # don't desync the inline-span pairing on the same line.
    line = line.replace("```", "\x00")
    return re.sub(r"`[^`]*`", "", line)


def test_changelog_has_no_bare_angle_bracket_tags() -> None:
    assert CHANGELOG.exists(), CHANGELOG
    offenders: list[str] = []
    in_fence = False
    for lineno, line in enumerate(
        CHANGELOG.read_text(encoding="utf-8").splitlines(), 1
    ):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for match in _TAG_RE.finditer(_strip_code_spans(line)):
            offenders.append(f"line {lineno}: {match.group(0)}")
    assert not offenders, (
        "Bare angle-bracket tokens break the VitePress/Vue website build; "
        "wrap them in backticks: " + "; ".join(offenders)
    )
