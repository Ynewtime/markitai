from __future__ import annotations

import json
from pathlib import Path

import pytest

from markitai.config import ConfigManager


def test_save_preserves_symlink_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Saving config through a symlink should update the target, not replace the link."""
    monkeypatch.chdir(tmp_path)

    manager = ConfigManager()
    manager.load()
    manager.set("llm.enabled", True)

    target = tmp_path / "real_config.json"
    target.write_text(json.dumps({"llm": {"enabled": False}}) + "\n", encoding="utf-8")

    link = tmp_path / "config.json"
    try:
        link.symlink_to(target.name)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symlinks unavailable in this environment: {exc}")

    manager.save(link)

    assert link.is_symlink()
    assert link.resolve() == target.resolve()
    saved = json.loads(target.read_text(encoding="utf-8"))
    assert saved["llm"]["enabled"] is True
