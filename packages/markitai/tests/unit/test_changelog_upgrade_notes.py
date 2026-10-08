"""Guard: one-time effects of upgrading to 1.2.1 are in both changelogs.

Two 1.2.1 fixes change what an existing install sees on its first run:
the LLM cache key changed (review #12), so existing entries miss once, and
``markitai serve`` history now ages by when a job finished (review #27), so
history older than seven days is deleted on the first start. Users read
these in the release notes; they must not get lost in later edits.
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
def test_1_2_1_lists_the_upgrade_effects(name: str) -> None:
    section = _release_section((REPO_ROOT / name).read_text(encoding="utf-8"), "1.2.1")
    for phrase in _NOTES[name]:
        assert phrase in section, f"{name} [1.2.1] is missing: {phrase!r}"
