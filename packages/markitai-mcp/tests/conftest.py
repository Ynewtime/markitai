"""Shared fixtures: isolate every test from the developer's markitai setup."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolated_markitai_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Pin config resolution to a minimal file and clear the job table.

    Without this, ``aconvert(config=None)`` loads the developer's real
    ``~/.markitai/config.json`` — which may enable LLM or define models — and
    the ``MODEL`` env var would defeat the "no model configured" tests. The
    cache directory is redirected so nothing touches the real user cache.
    """
    config_path = tmp_path / "markitai-config.json"
    config_path.write_text(
        json.dumps({"cache": {"global_dir": str(tmp_path / "cache")}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("MARKITAI_CONFIG", str(config_path))
    monkeypatch.delenv("MODEL", raising=False)

    from markitai_mcp import server

    server._JOBS.clear()


@pytest.fixture
def sample_md(tmp_path: Path) -> Path:
    """A small real document that converts in milliseconds."""
    path = tmp_path / "sample.md"
    path.write_text("# Hello\n\nSome body text.\n", encoding="utf-8")
    return path
