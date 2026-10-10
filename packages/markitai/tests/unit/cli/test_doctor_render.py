"""How `markitai doctor` reads: wrapping, hints, summary, keys (1.2.1 review)."""

from __future__ import annotations

import importlib
import io
import json
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

import pytest
from rich.console import Console

from markitai.config import LiteLLMParams, MarkitaiConfig, ModelConfig
from markitai.utils import term

# The module, not the click command that markitai.cli.commands exports
# under the same name
doctor_mod = importlib.import_module("markitai.cli.commands.doctor")

UBUNTU = 'NAME="Ubuntu"\nID=ubuntu\nID_LIKE=debian\n'
FEDORA = 'NAME="Fedora Linux"\nID=fedora\n'
UNKNOWN = "ID=plan9\n"

_ALL_KEYS: tuple[str, ...] = (
    "DEEPSEEK_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "OPENROUTER_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
)


def _row(name: str, status: str, message: str, **extra: Any) -> dict[str, Any]:
    return {
        "name": name,
        "description": name,
        "status": status,
        "message": message,
        "install_hint": "",
        **extra,
    }


def _config(*models: str) -> MarkitaiConfig:
    cfg = MarkitaiConfig()
    cfg.llm.model_list = [
        ModelConfig(model_name="default", litellm_params=LiteLLMParams(model=m))
        for m in models
    ]
    return cfg


@contextmanager
def _doctor_env(
    cfg: MarkitaiConfig,
    *,
    env: dict[str, str] | None = None,
    libreoffice_missing: bool = True,
) -> Iterator[None]:
    """Doctor with fixed capability rows, the owner's machine in 1.2.1."""
    playwright = _row(
        "Playwright",
        "warning",
        "installed, but its Chromium browser is not downloaded",
        **doctor_mod._install_fields("playwright"),
    )
    if libreoffice_missing:
        with patch("markitai.utils.office._linux_distro_ids", return_value={"ubuntu"}):
            libreoffice = _row(
                "LibreOffice",
                "missing",
                "not found (needed to render PPTX slides as images)",
                **doctor_mod._install_fields("libreoffice"),
            )
    else:
        libreoffice = _row("LibreOffice", "ok", "installed")
    with (
        patch.object(doctor_mod, "ConfigManager") as manager,
        patch.object(doctor_mod, "_check_playwright", return_value=playwright),
        patch.object(doctor_mod, "_check_libreoffice", return_value=libreoffice),
        patch.object(
            doctor_mod,
            "_check_rapidocr",
            return_value=_row("RapidOCR", "ok", "v3.10.0, reads ... text"),
        ),
        patch.object(
            doctor_mod, "_check_anydoc", return_value=_row("anydoc", "ok", "v1")
        ),
        patch.object(
            doctor_mod, "_check_serve", return_value=_row("Serve", "ok", "installed")
        ),
        patch.object(
            doctor_mod,
            "_check_claude_auth",
            return_value=_row("Claude Agent Auth", "ok", "signed in"),
        ),
        patch("markitai.fetch_playwright.clear_browser_cache"),
        patch(
            "markitai.llm.get_model_info_cached", return_value={"supports_vision": True}
        ),
        patch.object(doctor_mod.shutil, "which", return_value="/usr/bin/claude"),
        patch("importlib.util.find_spec", return_value=object()),
        patch.dict("os.environ"),
    ):
        for name in _ALL_KEYS:
            os.environ.pop(name, None)
        os.environ.update(env or {})
        manager.return_value.load.return_value = cfg
        manager.return_value.config_path = "/home/u/.markitai/config.json"
        yield


def _run(width: int | None) -> tuple[str, int]:
    """Run doctor on a terminal of ``width`` columns, or piped when None."""
    buf = io.StringIO()
    if width is None:
        console: Console = term._SoftWrapConsole(file=buf, width=80)
    else:
        console = term._SoftWrapConsole(
            file=buf, force_terminal=True, width=width, color_system=None
        )
    code = 0
    with (
        patch.object(term, "_console", console),
        patch.object(doctor_mod, "console", console),
    ):
        try:
            doctor_mod._doctor_impl(as_json=False)
        except SystemExit as e:
            code = int(e.code or 0)
    return buf.getvalue(), code


OWNER_MODELS = (
    "claude-agent/haiku",
    "deepseek/deepseek-flash",
    "gemini/gemini-flash-lite-latest",
    "openai/gpt-6-luna",
)
OWNER_KEYS: dict[str, str] = dict.fromkeys(
    _ALL_KEYS[:4] + ("OPENROUTER_API_KEY",), "sk-fake"
)


class TestWrapping:
    """Long lines wrap under their text on a terminal, never when piped."""

    @pytest.mark.parametrize("width", [80, 40])
    def test_terminal_lines_fit_and_hang_under_the_text(self, width: int) -> None:
        with _doctor_env(_config(*OWNER_MODELS), env=OWNER_KEYS):
            out, code = _run(width)

        assert code == 0
        lines = out.splitlines()
        commands = {
            line
            for line in lines
            if "playwright install" in line or line.strip().startswith("sudo ")
        }
        for line in lines:
            if line in commands or "/" in line.strip().split(" ")[-1]:
                continue  # commands and long paths stay whole
            assert len(line) <= width, line
        # Every wrapped continuation starts under the text of its item
        # (column 4), never back at column 0
        top_level = {"◆ System Check", "Optional Capabilities", "LLM", "Authentication"}
        for line in lines:
            if not line or line in top_level or line.startswith("To "):
                continue
            if line.startswith(("✓ Ready", "✗ Not ready")):
                continue
            assert line.startswith("  "), repr(line)
        vlm = next(i for i, line in enumerate(lines) if line.startswith("  ✓ VLM OCR:"))
        assert lines[vlm + 1].startswith("    ") and lines[vlm + 1][4] != " "

    def test_piped_output_is_one_line_per_item(self) -> None:
        with _doctor_env(_config(*OWNER_MODELS), env=OWNER_KEYS):
            out, _ = _run(None)

        vlm = [line for line in out.splitlines() if "VLM OCR" in line]
        assert len(vlm) == 1
        assert vlm[0].endswith("(MARKITAI_NO_VLM_OCR=1 keeps RapidOCR)")
        summary = out.splitlines()[-1]
        assert summary == "✓ Ready. Not available (optional): Playwright, LibreOffice"


class TestHints:
    """Every hint is a command that works on this machine, one per line."""

    def test_libreoffice_on_ubuntu_is_one_apt_command(self) -> None:
        cmds = doctor_mod.get_install_commands(
            "libreoffice", "linux", os_release=UBUNTU
        )
        assert cmds == [("", "sudo apt install libreoffice")]

    def test_libreoffice_on_fedora_is_dnf(self) -> None:
        cmds = doctor_mod.get_install_commands(
            "libreoffice", "linux", os_release=FEDORA
        )
        assert cmds == [("", "sudo dnf install libreoffice")]

    def test_unknown_distro_lists_labelled_alternatives(self) -> None:
        cmds = doctor_mod.get_install_commands(
            "libreoffice", "linux", os_release=UNKNOWN
        )
        assert len(cmds) > 1
        assert all(label and "\n" not in command for label, command in cmds)
        assert "\n" not in doctor_mod.get_install_hint(
            "libreoffice", "linux", os_release=UNKNOWN
        )

    def test_playwright_uses_markitais_own_interpreter(self) -> None:
        """A bare `python` is not markitai's tool environment."""
        (label, command), *rest = doctor_mod.get_install_commands("playwright", "linux")
        assert not rest and not label
        assert command.startswith(doctor_mod._markitai_python() + " -m playwright")
        assert "--with-deps chromium" in command
        mac = doctor_mod.get_install_commands("playwright", "darwin")[0][1]
        assert mac.endswith("-m playwright install chromium")

    def test_windows_path_with_spaces_uses_the_call_operator(self) -> None:
        with (
            patch.object(sys, "platform", "win32"),
            patch.object(
                sys, "executable", r"C:\Users\A B\uv\tools\markitai\python.exe"
            ),
        ):
            python = doctor_mod._markitai_python()
        assert python == r'& "C:\Users\A B\uv\tools\markitai\python.exe"'

    def test_hints_render_commands_on_their_own_lines(self) -> None:
        with _doctor_env(_config(*OWNER_MODELS), env=OWNER_KEYS):
            out, _ = _run(80)

        lines = out.splitlines()
        i = lines.index("  • LibreOffice:")
        assert lines[i + 1] == "    sudo apt install libreoffice"
        j = lines.index("  • Playwright:")
        assert lines[j + 1].startswith("    ") and "playwright install" in lines[j + 1]
        assert "# Fedora" not in out and "dnf" not in out

    def test_multiline_prose_hint_keeps_its_indentation(self) -> None:
        buf = io.StringIO()
        console = term._SoftWrapConsole(file=buf, width=80)
        with patch.object(term, "_console", console):
            doctor_mod._render_hint(
                _row(
                    "Claude Agent Auth",
                    "error",
                    "x",
                    install_hint="Run 'claude login'.\nIf the CLI is missing:\n  curl x | bash",
                )
            )
        assert buf.getvalue().splitlines() == [
            "  • Claude Agent Auth: Run 'claude login'.",
            "    If the CLI is missing:",
            "      curl x | bash",
        ]


class TestSummary:
    """Ready or not, said plainly, with a mark that matches."""

    def test_ready_with_optional_gaps_is_a_success(self) -> None:
        with _doctor_env(_config(*OWNER_MODELS), env=OWNER_KEYS):
            out, code = _run(80)
        assert code == 0
        assert out.rstrip().endswith(
            "✓ Ready. Not available (optional): Playwright, LibreOffice"
        )
        assert "required/configured" not in out and "Core check" not in out

    def test_missing_provider_key_is_not_ready(self) -> None:
        """deepseek/ with no DEEPSEEK_API_KEY: the owner's new terminal."""
        with _doctor_env(_config("deepseek/deepseek-flash"), env={}):
            out, code = _run(None)
        assert code == 1
        assert "✗ LLM API: not set: DEEPSEEK_API_KEY" in out
        assert "export DEEPSEEK_API_KEY=" in out
        assert out.splitlines()[-1] == "✗ Not ready: fix LLM API first (see above)"

    def test_gemini_accepts_google_api_key(self) -> None:
        with _doctor_env(
            _config("gemini/gemini-flash-lite-latest"),
            env={"GOOGLE_API_KEY": "fake"},
        ):
            _out, code = _run(None)
        assert code == 0

    def test_all_good(self) -> None:
        with (
            _doctor_env(
                _config("deepseek/deepseek-flash"),
                env={"DEEPSEEK_API_KEY": "fake"},
                libreoffice_missing=False,
            ),
            patch.object(
                doctor_mod,
                "_check_playwright",
                return_value=_row("Playwright", "ok", "Chromium installed"),
            ),
        ):
            out, code = _run(None)
        assert code == 0
        assert out.splitlines()[-1] == "✓ Ready. Everything is available."


class TestUnusedKeys:
    """Keys that are set but used by no model are named, with why."""

    def test_owner_setup_names_anthropic_and_openrouter(self) -> None:
        with _doctor_env(_config(*OWNER_MODELS), env=OWNER_KEYS):
            out, _ = _run(None)
        assert (
            "  • Not used: ANTHROPIC_API_KEY, OPENROUTER_API_KEY "
            "(set, but no model in llm.model_list uses them)"
        ) in out.splitlines()
        assert (
            "LLM API: 4 active model(s): Claude Code, OpenAI, Gemini, DeepSeek" in out
        )

    def test_json_lists_them(self) -> None:
        buf = io.StringIO()
        with (
            _doctor_env(_config(*OWNER_MODELS), env=OWNER_KEYS),
            patch.object(doctor_mod.click, "echo", lambda s: buf.write(s)),
        ):
            doctor_mod._doctor_impl(as_json=True)

        llm = json.loads(buf.getvalue())["llm-api"]
        assert llm["unused_api_keys"] == ["ANTHROPIC_API_KEY", "OPENROUTER_API_KEY"]
