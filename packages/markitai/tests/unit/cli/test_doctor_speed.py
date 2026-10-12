"""How long `markitai doctor` and `markitai --version` take (owner, 2026-10-12).

On runner-win (Windows 11 on Arm, x64 Python under emulation) the first
`markitai doctor` after an install took minutes and a warm one 5.9 s,
`--version` 1.9 s. Doctor imported all of LiteLLM (about 2,400 modules) to
ask whether a configured model reads images, launched Chromium on every run
and ran the Claude Code sign-in check after the others; `--version` loaded
the whole CLI to print one line.
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from markitai import __version__
from tests.terminal_env import plain_env

doctor_mod = importlib.import_module("markitai.cli.commands.doctor")


def _imported_modules(stderr: str) -> set[str]:
    """The module names an ``-X importtime`` run reported."""
    return {
        line.rsplit("|", 1)[1].strip()
        for line in stderr.splitlines()
        if line.startswith("import time:") and "|" in line
    }


def _run_markitai(args: list[str], home: Path) -> subprocess.CompletedProcess[str]:
    env = plain_env(HOME=str(home), USERPROFILE=str(home))
    for name in ("DEEPSEEK_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY"):
        env.pop(name, None)
    return subprocess.run(
        [sys.executable, "-X", "importtime", "-m", "markitai", *args],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        cwd=home,
        timeout=120,
    )


def test_version_does_not_load_the_cli(tmp_path: Path) -> None:
    result = _run_markitai(["--version"], tmp_path)
    assert result.returncode == 0
    assert result.stdout == f"markitai {__version__}\n"
    modules = _imported_modules(result.stderr)
    assert not modules & {"click", "rich", "rich_click", "pydantic", "loguru"}
    assert "markitai.cli" not in modules


def test_doctor_with_api_models_does_not_import_litellm(tmp_path: Path) -> None:
    """The owner's four models: three API models and Claude Code."""
    config = tmp_path / ".markitai" / "config.json"
    config.parent.mkdir()
    models = [
        "claude-agent/sonnet",
        "deepseek/deepseek-flash",
        "gemini/gemini-flash-lite-latest",
        "openai/gpt-6-luna",
    ]
    config.write_text(
        json.dumps(
            {
                "llm": {
                    "model_list": [
                        {"model_name": "default", "litellm_params": {"model": m}}
                        for m in models
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    result = _run_markitai(["doctor", "--json"], tmp_path)
    report = json.loads(result.stdout)
    assert report["vision-model"]["status"] == "ok"
    assert "litellm" not in _imported_modules(result.stderr)


@pytest.mark.parametrize(
    "model",
    [
        "deepseek/deepseek-flash",
        "deepseek/deepseek-chat",
        "gemini/gemini-flash-lite-latest",
        "gemini/gemini-2.5-flash",
        "openai/gpt-6-luna",
        "openai/gpt-6.1-sol",
        "openai/gpt-4o-2024-08-06",
        "anthropic/claude-haiku-5-5",
        "openrouter/~anthropic/claude-haiku-latest",
        "openrouter/openai/gpt-4o",
        "mistral/pixtral-large-latest",
        "vertex_ai/gemini-2.5-pro",
        "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
        "groq/llama-4-scout",
        "azure/my-deployment",
        "ollama/llama3",
        "gpt-4o",
    ],
)
def test_the_vision_answer_is_litellms(model: str) -> None:
    """model_catalog reads LiteLLM's table without importing it: same answer.

    Checked over every row of LiteLLM 1.104.2's table on 2026-10-12 (6,978
    names, each as ``provider/id`` and as the bare key): equal except for
    LiteLLM's internal pseudo-prefixes (``bedrock_converse/``,
    ``vertex_ai-language-models/``), which no configuration uses.
    """
    from markitai.llm import get_model_info_cached
    from markitai.model_catalog import register_missing_models, supports_vision

    register_missing_models()
    assert supports_vision(model) == bool(
        get_model_info_cached(model).get("supports_vision")
    )


class TestChromiumLaunchIsRemembered:
    """A launch costs a Python, Playwright's driver and Chromium each run."""

    @pytest.fixture
    def chromium(self, tmp_path: Path) -> Path:
        marker = tmp_path / "ms-playwright" / "chromium-1" / "INSTALLATION_COMPLETE"
        marker.parent.mkdir(parents=True)
        marker.write_text("", encoding="utf-8")
        return marker

    @pytest.fixture
    def launches(
        self, tmp_path: Path, chromium: Path, monkeypatch: pytest.MonkeyPatch
    ) -> list[list[str]]:
        monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
        monkeypatch.setattr(
            "markitai.fetch_playwright.installed_chromium", lambda: chromium
        )
        monkeypatch.setattr(
            "importlib.metadata.version",
            lambda name: "1.63.0" if name == "playwright" else "0",
        )
        calls: list[list[str]] = []

        def fake_run(args: list[str], **_kwargs: Any) -> Any:
            calls.append(args)
            return subprocess.CompletedProcess(args, 0, "", "")

        monkeypatch.setattr(doctor_mod.subprocess, "run", fake_run)
        return calls

    def test_a_second_doctor_does_not_launch_again(
        self, launches: list[list[str]]
    ) -> None:
        assert doctor_mod._smoke_test_playwright_browser() == (True, "")
        assert doctor_mod._smoke_test_playwright_browser() == (True, "")
        assert len(launches) == 1

    def test_fix_always_launches(self, launches: list[list[str]]) -> None:
        doctor_mod._smoke_test_playwright_browser()
        doctor_mod._smoke_test_playwright_browser(remembered=False)
        assert len(launches) == 2

    def test_another_chromium_launches_again(
        self, launches: list[list[str]], chromium: Path
    ) -> None:
        doctor_mod._smoke_test_playwright_browser()
        later = chromium.stat().st_mtime_ns + 10_000_000_000
        os.utime(chromium, ns=(later, later))
        doctor_mod._smoke_test_playwright_browser()
        assert len(launches) == 2

    def test_a_week_old_launch_launches_again(
        self, launches: list[list[str]], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        doctor_mod._smoke_test_playwright_browser()
        week = doctor_mod._LAUNCH_RECORD_DAYS * 86400
        real_time = time.time
        monkeypatch.setattr(doctor_mod.time, "time", lambda: real_time() + week + 1)
        doctor_mod._smoke_test_playwright_browser()
        assert len(launches) == 2

    def test_a_failed_launch_is_not_remembered(
        self, launches: list[list[str]], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def failing_run(args: list[str], **_kwargs: Any) -> Any:
            launches.append(args)
            return subprocess.CompletedProcess(args, 1, "", "error: missing libnss3")

        monkeypatch.setattr(doctor_mod.subprocess, "run", failing_run)
        assert doctor_mod._smoke_test_playwright_browser()[0] is False
        assert doctor_mod._smoke_test_playwright_browser()[0] is False
        assert len(launches) == 2


def test_checks_run_at_the_same_time() -> None:
    """Two checks that wait for each other finish only when run together."""
    barrier = threading.Barrier(2, timeout=10)

    def waiting(name: str) -> Any:
        def check() -> dict[str, Any]:
            barrier.wait()
            return {"name": name, "status": "ok", "message": "", "install_hint": ""}

        return check

    results = doctor_mod._run_checks(
        {
            "a": ("Claude Code sign-in", waiting("a")),
            "b": ("Playwright", waiting("b")),
        },
        show_progress=False,
    )
    assert set(results) == {"a", "b"}


def test_timings_flag_reports_each_check_on_stderr(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MARKITAI_TIMINGS", raising=False)
    launcher = importlib.import_module("markitai.launcher")
    monkeypatch.setattr(launcher, "_steps", [])
    with patch.object(doctor_mod, "_doctor_impl"):
        doctor_mod.doctor.callback(  # type: ignore[misc]
            as_json=False, fix=False, suggest=False, timings=True
        )
    assert os.environ["MARKITAI_TIMINGS"] == "1"
    doctor_mod._run_checks(
        {"anydoc": ("anydoc", doctor_mod._check_anydoc)}, show_progress=False
    )
    launcher.report()
    err = capsys.readouterr().err
    assert "doctor: anydoc" in err
    assert "total" in err
