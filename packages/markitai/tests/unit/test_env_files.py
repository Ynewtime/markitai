"""A folder's .env and markitai.json cannot redirect programs or keys."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from markitai.config import KEYLESS_PLACEHOLDER, ConfigManager, seal_project_models
from markitai.env_files import load_env_files, merge_env_files, project_may_not_set


@pytest.mark.parametrize(
    "name",
    [
        "COPILOT_CLI_PATH",
        "claude_cli_path",
        "MARKITAI_BROWSER_EXECUTABLE",
        "PLAYWRIGHT_BROWSERS_PATH",
        "PATH",
        "LD_PRELOAD",
        "DYLD_INSERT_LIBRARIES",
        "AWS_CONFIG_FILE",
        "AWS_SHARED_CREDENTIALS_FILE",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "OPENAI_API_BASE",
        "OPENAI_BASE_URL",
        "https_proxy",
        "NO_PROXY",
        "HOME",
        "CLAUDE_CONFIG_DIR",
        "MARKITAI_CONFIG",
        "NODE_OPTIONS",
        "PYTHONPATH",
        "SSL_CERT_FILE",
        "REQUESTS_CA_BUNDLE",
    ],
)
def test_a_folders_env_may_not_set(name: str) -> None:
    assert project_may_not_set(name)


@pytest.mark.parametrize(
    "name", ["OPENAI_API_KEY", "DEEPSEEK_API_KEY", "MODEL", "MARKITAI_LANG"]
)
def test_a_folders_env_still_sets_keys_and_options(name: str) -> None:
    assert not project_may_not_set(name)


def test_values_already_set_win_and_refused_names_are_listed() -> None:
    added, ignored = merge_env_files(
        {"MODEL": "process"},
        [
            ("OPENAI_API_KEY", "from-folder"),
            ("COPILOT_CLI_PATH", "./fake"),
            ("HTTPS_PROXY", "http://capture.example:8080"),
            ("MODEL", "folder"),
        ],
        trusted_here=False,
        own=[("COPILOT_CLI_PATH", "/own/copilot"), ("DEEPSEEK_API_KEY", "own")],
    )
    assert added == {
        "OPENAI_API_KEY": "from-folder",
        "COPILOT_CLI_PATH": "/own/copilot",
        "DEEPSEEK_API_KEY": "own",
    }
    assert ignored == ["COPILOT_CLI_PATH", "HTTPS_PROXY"]


def test_loading_a_folders_env_warns_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = tmp_path / "download"
    folder.mkdir()
    (folder / ".env").write_text(
        "COPILOT_CLI_PATH=./fake\nOPENAI_API_BASE=https://capture.example/v1\n"
        "SOME_PROVIDER_API_KEY=k1\n",
        encoding="utf-8",
    )
    environ: dict[str, str] = {}
    monkeypatch.setattr(os, "environ", environ)
    home = tmp_path / "home"
    assert load_env_files(cwd=folder, home=home) == [
        "COPILOT_CLI_PATH",
        "OPENAI_API_BASE",
    ]
    load_env_files(cwd=folder, home=home)
    assert environ == {"SOME_PROVIDER_API_KEY": "k1"}
    warnings = [
        line
        for line in capsys.readouterr().err.splitlines()
        if line.startswith("Warning: Ignored ")
    ]
    assert len(warnings) == 1
    assert warnings[0].startswith(
        "Warning: Ignored COPILOT_CLI_PATH, OPENAI_API_BASE from ./.env"
    )


def _project(tmp_path: Path) -> dict[str, object]:
    return {
        "llm": {
            "model_list": [
                {
                    "model_name": "a",
                    "litellm_params": {
                        "model": "openai/gpt-x",
                        "api_base": "https://capture.example/v1",
                    },
                },
                {
                    "model_name": "b",
                    "litellm_params": {
                        "model": "openai/gpt-x",
                        "api_base": "https://capture.example/v1",
                        "api_key": "env:OPENAI_API_KEY",
                    },
                },
                {
                    "model_name": "c",
                    "litellm_params": {
                        "model": "deepseek/deepseek-chat",
                        "api_base": "https://api.deepseek.com",
                        "api_key": "env:DEEPSEEK_API_KEY",
                    },
                },
                {"model_name": "d", "litellm_params": {"model": "openai/gpt-x"}},
                {
                    "model_name": "e",
                    "litellm_params": {
                        "model": "vertex_ai/gemini-2.5-pro",
                        "api_base": "https://capture.example",
                    },
                },
                {
                    "model_name": "f",
                    "litellm_params": {
                        "model": "openai/gpt-x",
                        "api_base": "https://capture.example/v1",
                        "api_key": "sk-written-here",
                    },
                },
            ]
        }
    }


def test_a_project_config_keeps_environment_keys_at_documented_endpoints_only(
    tmp_path: Path,
) -> None:
    data = _project(tmp_path)
    notes = seal_project_models(data)
    keys = {
        entry["model_name"]: entry["litellm_params"].get("api_key")  # type: ignore[index]
        for entry in data["llm"]["model_list"]  # type: ignore[index]
    }
    assert keys == {
        "a": KEYLESS_PLACEHOLDER,
        "b": KEYLESS_PLACEHOLDER,
        "c": "env:DEEPSEEK_API_KEY",
        "d": None,
        "f": "sk-written-here",
    }
    assert len(notes) == 2
    assert "env:OPENAI_API_KEY" in notes[0]
    assert "cloud login" in notes[1]


def test_only_a_config_found_in_the_folder_is_sealed_and_never_saved_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = tmp_path / "download"
    folder.mkdir()
    written = json.dumps(_project(tmp_path))
    (folder / "markitai.json").write_text(written, encoding="utf-8")
    monkeypatch.chdir(folder)
    monkeypatch.delenv("MARKITAI_CONFIG", raising=False)
    found = ConfigManager()
    cfg = found.load()
    assert found.found_in_folder
    assert [m.litellm_params.api_key for m in cfg.llm.model_list][:2] == [
        KEYLESS_PLACEHOLDER,
        KEYLESS_PLACEHOLDER,
    ]
    assert "vertex_ai/gemini-2.5-pro" not in [
        m.litellm_params.model for m in cfg.llm.model_list
    ]
    found.set("llm.enabled", True)
    found.save()
    saved = (folder / "markitai.json").read_text(encoding="utf-8")
    assert KEYLESS_PLACEHOLDER not in saved
    # Named with -c, the same file keeps its meaning.
    named = ConfigManager()
    cfg = named.load(config_path=folder / "markitai.json")
    assert not named.found_in_folder
    assert cfg.llm.model_list[0].litellm_params.api_key is None
    assert cfg.llm.model_list[1].litellm_params.api_key == "env:OPENAI_API_KEY"
