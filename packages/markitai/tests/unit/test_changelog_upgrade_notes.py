"""Guard: one-time effects of upgrading to 1.3.0 are in both changelogs.

Two 1.3.0 fixes change what an existing install sees on its first run:
the LLM cache key changed (review #12), so existing entries miss once, and
``markitai serve`` history now ages by when a job finished (review #27), so
history older than seven days is deleted on the first start. Users read
these in the release notes; they must not get lost in later edits.

``main`` carries no unreleased changelog section (the release commit adds
it), so the guard applies once a ``## [1.3.0]`` section is there.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]

_NOTES = {
    "CHANGELOG.md": ("miss once after upgrading", "deleted on the first start"),
    "CHANGELOG.zh.md": ("升级后首次运行", "首次启动时"),
}


def _release_section(text: str, version: str) -> str:
    start = text.index(f"## [{version}]")
    end = text.find("\n## [", start + 1)
    return text[start : end if end != -1 else None]


@pytest.mark.parametrize("name", sorted(_NOTES))
def test_1_3_0_lists_the_upgrade_effects(name: str) -> None:
    text = (REPO_ROOT / name).read_text(encoding="utf-8")
    if "## [1.3.0]" not in text:
        pytest.skip(f"{name} has no [1.3.0] section until the release commit")
    section = _release_section(text, "1.3.0")
    for phrase in _NOTES[name]:
        assert phrase in section, f"{name} [1.3.0] is missing: {phrase!r}"
