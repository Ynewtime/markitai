"""scripts/check_changelog.py: main carries released CHANGELOG sections only.

Website Check runs the guard on every push and pull request, the Release
workflow runs ``--release`` on the tagged commit. These tests use invented
changelogs; the git calls are replaced so they run without tags.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

_SCRIPT = Path(__file__).resolve().parents[4] / "scripts" / "check_changelog.py"

_RELEASED = """# Changelog

## [2.1.0] - 2026-02-01

- Fixed: a released entry

```text
## [9.0.0] in a code block is not a heading
```

## [2.0.0] - 2026-01-01

- Added: the first entry
"""


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_check_changelog", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def cc() -> ModuleType:
    return _load()


def _repo(tmp_path: Path, en: str, zh: str | None = None) -> Path:
    (tmp_path / "CHANGELOG.md").write_text(en, encoding="utf-8")
    (tmp_path / "CHANGELOG.zh.md").write_text(
        zh if zh is not None else en, encoding="utf-8"
    )
    return tmp_path


def _tags(
    cc: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tags: list[str],
    remote: tuple[str, ...] = (),
) -> None:
    monkeypatch.setattr(cc, "local_tags", lambda: [cc.parse_version(t) for t in tags])
    monkeypatch.setattr(cc, "remote_has_tag", lambda tag: tag in remote)


def test_released_sections_and_code_blocks_pass(cc, monkeypatch, tmp_path):
    _tags(cc, monkeypatch, ["2.1.0", "2.0.0"])
    assert cc.check_unreleased(_repo(tmp_path, _RELEASED)) == []


def test_an_untagged_section_fails_with_where_the_notes_go(cc, monkeypatch, tmp_path):
    _tags(cc, monkeypatch, ["2.1.0"])
    text = _RELEASED.replace(
        "## [2.1.0]", "## [2.2.0]\n\n- Added: not out yet\n\n## [2.1.0]"
    )
    errors = cc.check_unreleased(_repo(tmp_path, text, _RELEASED))
    assert len(errors) == 1
    assert "CHANGELOG.md has a section for version 2.2.0" in errors[0]
    assert ".local/changelog-unreleased.md" in errors[0]
    assert "git push --atomic origin main vX.Y.Z" in errors[0]


def test_an_unreleased_heading_fails(cc, monkeypatch, tmp_path):
    _tags(cc, monkeypatch, ["2.1.0"])
    text = _RELEASED.replace("## [2.1.0]", "## [Unreleased]\n\n## [2.1.0]")
    errors = cc.check_unreleased(_repo(tmp_path, _RELEASED, text))
    assert (
        len(errors) == 1
        and "CHANGELOG.zh.md" in errors[0]
        and "'Unreleased'" in errors[0]
    )


def test_a_tag_only_origin_has_yet_counts(cc, monkeypatch, tmp_path):
    _tags(cc, monkeypatch, ["2.0.0"], remote=("v2.1.0",))
    assert cc.check_unreleased(_repo(tmp_path, _RELEASED)) == []


def test_no_tags_fails(cc, monkeypatch, tmp_path):
    _tags(cc, monkeypatch, [])
    assert "no vX.Y.Z tag" in cc.check_unreleased(_repo(tmp_path, _RELEASED))[0]


def test_release_needs_the_section_in_both_files(cc, tmp_path):
    root = _repo(tmp_path, _RELEASED, _RELEASED.replace("## [2.1.0] - 2026-02-01", ""))
    assert cc.check_release("v2.0.0", root) == []
    errors = cc.check_release("v2.1.0", root)
    assert len(errors) == 1 and "CHANGELOG.zh.md at v2.1.0" in errors[0]
    # A heading inside a code block is not the release section.
    assert len(cc.check_release("v9.0.0", root)) == 2
    assert cc.check_release("main", root) == ["'main' is not a vX.Y.Z tag"]
