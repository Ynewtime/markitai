"""How `markitai doctor` reads: wrapping, hints, summary, keys (1.2.1 review)."""

from __future__ import annotations

import importlib
import io
import json
import os
import sys
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from typing import Any
from unittest.mock import patch

import pytest
from rich.console import Console

from markitai.config import LiteLLMParams, MarkitaiConfig, ModelConfig
from markitai.utils import term
from markitai.utils.guidance import markitai_python_command

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
    platform: str = "linux",
) -> Iterator[None]:
    """Doctor with fixed capability rows, the owner's machine in 1.2.1.

    That machine runs Ubuntu: the install and key hints are Linux's whatever
    OS runs the tests (macOS printed brew, Windows winget and $env:).
    """
    with (
        patch.object(sys, "platform", platform),
        _doctor_rows(cfg, env=env, libreoffice_missing=libreoffice_missing),
    ):
        yield


@contextmanager
def _doctor_rows(
    cfg: MarkitaiConfig,
    *,
    env: dict[str, str] | None,
    libreoffice_missing: bool,
) -> Iterator[None]:
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
        patch("markitai.model_catalog.supports_vision", return_value=True),
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
        assert command.startswith(markitai_python_command() + " -m playwright")
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
            python = markitai_python_command()
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

    @pytest.mark.parametrize(
        ("platform", "libreoffice", "playwright", "key"),
        [
            (
                "linux",
                "sudo apt install libreoffice",
                "-m playwright install --with-deps chromium",
                "export DEEPSEEK_API_KEY=your_key",
            ),
            (
                "darwin",
                "brew install --cask libreoffice",
                "-m playwright install chromium",
                "export DEEPSEEK_API_KEY=your_key",
            ),
            (
                "win32",
                "winget install TheDocumentFoundation.LibreOffice",
                "-m playwright install chromium",
                '$env:DEEPSEEK_API_KEY="your_key"',
            ),
        ],
    )
    def test_each_platform_gets_its_own_commands(
        self, platform: str, libreoffice: str, playwright: str, key: str
    ) -> None:
        keys = {k: v for k, v in OWNER_KEYS.items() if k != "DEEPSEEK_API_KEY"}
        with _doctor_env(
            _config(*OWNER_MODELS),
            env={**keys, "PSModulePath": r"C:\Modules"},
            platform=platform,
        ):
            out, _ = _run(None)

        lines = out.splitlines()
        i = lines.index("  • LibreOffice:")
        assert lines[i + 1] == "    " + libreoffice
        j = lines.index("  • Playwright:")
        assert lines[j + 1].endswith(playwright)
        assert "    " + key in lines

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

    def test_some_keys_missing_leaves_those_models_out(self) -> None:
        """The run uses the models that have keys, so doctor is not red."""
        with _doctor_env(_config(*OWNER_MODELS), env={}):
            out, code = _run(None)
        assert code == 0
        assert (
            "  ! LLM API: not set: DEEPSEEK_API_KEY, GEMINI_API_KEY, "
            "OPENAI_API_KEY; 3 models left out, 1 still used"
        ) in out.splitlines()
        assert "    export GEMINI_API_KEY=your_key" in out.splitlines()
        assert out.splitlines()[-1].endswith("Playwright, LibreOffice, 3 LLM models")

    def test_a_vision_model_left_out_for_its_key_is_not_a_vision_model(self) -> None:
        """Gemini is the only model that reads images and its key is unset:
        the run leaves it out, so Vision Model and VLM OCR must not tick."""
        with (
            _doctor_env(
                _config("deepseek/deepseek-chat", "gemini/gemini-2.5-flash"),
                env={"DEEPSEEK_API_KEY": "fake"},
            ),
            patch(
                "markitai.model_catalog.supports_vision",
                side_effect=lambda model: model.startswith("gemini/"),
            ),
        ):
            out, code = _run(None)
        assert code == 0
        lines = out.splitlines()
        assert (
            "  ! LLM API: not set: GEMINI_API_KEY; 1 model left out, 1 still used"
            in lines
        )
        assert not any(line.startswith("  ✓ Vision Model") for line in lines)
        assert not any(line.startswith("  ✓ VLM OCR") for line in lines)
        assert (
            "  ! Vision Model: the vision models are left out until "
            "GEMINI_API_KEY is set (see LLM API)"
        ) in lines

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


class TestWindowsDoctor:
    """runner-win, right after setup.ps1 (item 16)."""

    def test_slow_first_launch_gets_one_retry(self) -> None:
        """Defender scanning a fresh chrome.exe made the first launch time out."""
        import subprocess

        ok = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
        calls: list[float] = []

        def run(*args: Any, timeout: float, **kwargs: Any) -> Any:
            calls.append(timeout)
            if len(calls) == 1:
                raise subprocess.TimeoutExpired(cmd="python", timeout=timeout)
            return ok

        with patch.object(doctor_mod.subprocess, "run", run):
            assert doctor_mod._smoke_test_playwright_browser() == (True, "")
        assert calls == list(doctor_mod._SMOKE_TIMEOUTS_S)

    def test_a_launch_that_stays_slow_is_slow_not_unusable(self) -> None:
        import subprocess

        def run(*args: Any, timeout: float, **kwargs: Any) -> Any:
            raise subprocess.TimeoutExpired(cmd="python", timeout=timeout)

        with (
            patch.object(doctor_mod.subprocess, "run", run),
            patch(
                "markitai.fetch_playwright.is_playwright_available", return_value=True
            ),
            patch(
                "markitai.fetch_playwright.is_playwright_browser_installed",
                return_value=True,
            ),
        ):
            row = doctor_mod._check_playwright()

        assert row["status"] == "warning"
        assert "slow to start" in row["message"]
        assert "run doctor again" in row["message"]
        assert "unusable" not in row["message"]
        assert row["install_commands"] == ["markitai doctor"]

    def test_text_and_json_say_the_same(self) -> None:
        slow = {
            "name": "Playwright",
            "description": "Browser automation for dynamic URLs",
            "status": "warning",
            **doctor_mod._playwright_runtime_problem(doctor_mod.SMOKE_SLOW),
        }
        buf = io.StringIO()
        with (
            _doctor_env(_config(*OWNER_MODELS), env=OWNER_KEYS),
            patch.object(doctor_mod, "_check_playwright", return_value=slow),
            patch.object(doctor_mod.click, "echo", lambda s: buf.write(s)),
        ):
            doctor_mod._doctor_impl(as_json=True)
            out, _ = _run(None)

        message = json.loads(buf.getvalue())["playwright"]["message"]
        assert f"  ! Playwright: {message}" in out.splitlines()
        assert "  • Playwright:" in out.splitlines()
        assert "    markitai doctor" in out.splitlines()
        assert "Playwright (slow to start)" in out.splitlines()[-1]
        assert out.splitlines()[-1].startswith("✓ Ready.")

    def test_the_same_hint_is_printed_once(self) -> None:
        """Vision Model and VLM OCR both asked for a vision-capable model."""
        vision_hint = "Use vision-capable models like gemini-*, gpt-6*, claude-*"
        with (
            _doctor_env(_config("deepseek/deepseek-flash"), env=OWNER_KEYS),
            patch("markitai.model_catalog.supports_vision", return_value=False),
        ):
            out, _ = _run(None)
        assert out.count(vision_hint) == 1
        assert "  • Vision Model, VLM OCR: " + vision_hint in out.splitlines() or (
            "  • Vision Model: " + vision_hint in out.splitlines()
        )

    def test_hint_uses_the_tool_python_never_a_bare_python(self) -> None:
        """`python` on runner-win is only the Microsoft Store alias."""
        exe = r"C:\Users\runner\AppData\Roaming\uv\tools\markitai\Scripts\python.exe"
        with (
            patch.object(sys, "platform", "win32"),
            patch.object(sys, "executable", exe),
        ):
            (_, command), *_ = doctor_mod.get_install_commands("playwright")
        assert command == f"{exe} -m playwright install chromium"

    def test_wraps_one_column_short_on_windows(self) -> None:
        """conhost wraps a full-width line itself: an extra blank row."""
        buf = io.StringIO()
        console = Console(file=buf, force_terminal=True, width=40, color_system=None)
        with patch.object(term.sys, "platform", "win32"):
            term.print_marked("word " * 30, mark=term.MARK_WARNING, console=console)
        lines = buf.getvalue().splitlines()
        assert all(len(line) <= 39 for line in lines)
        assert not any(line.endswith(" ") for line in lines)

    @pytest.mark.parametrize("width", [80, 40])
    def test_no_trailing_spaces(self, width: int) -> None:
        with _doctor_env(_config(*OWNER_MODELS), env=OWNER_KEYS):
            out, _ = _run(width)
        assert not [line for line in out.splitlines() if line.endswith(" ")]


class TestMissingExtras:
    """A plain install (setup.ps1 without a terminal) lacks four extras.

    Each hint reinstalls markitai with everything it already has plus one
    extra, so four of them meant four reinstalls; one command does it.
    """

    @pytest.fixture
    def rows(self, tmp_path, monkeypatch) -> dict[str, dict[str, Any]]:
        from importlib.metadata import PackageNotFoundError

        (tmp_path / "uv-receipt.toml").write_text(
            '[tool]\nrequirements = [{ name = "markitai", directory = "/src/m" }]\n',
            encoding="utf-8",
        )
        monkeypatch.setattr(sys, "prefix", str(tmp_path))
        with (
            patch.object(sys, "platform", "linux"),
            patch("markitai.serve.is_serve_available", return_value=False),
            patch("markitai.ocr.is_ocr_available", return_value=False),
            patch(
                "markitai.fetch_playwright.is_playwright_available", return_value=False
            ),
            patch("importlib.metadata.version", side_effect=PackageNotFoundError("x")),
        ):
            return {
                "_check_playwright": doctor_mod._check_playwright(),
                "_check_rapidocr": doctor_mod._check_rapidocr(MarkitaiConfig()),
                "_check_anydoc": doctor_mod._check_anydoc(),
                "_check_serve": doctor_mod._check_serve(),
            }

    @contextmanager
    def _checks(self, rows: dict[str, dict[str, Any]]) -> Iterator[None]:
        with _doctor_env(_config(*OWNER_MODELS), env=OWNER_KEYS), ExitStack() as stack:
            for name, row in rows.items():
                stack.enter_context(patch.object(doctor_mod, name, return_value=row))
            yield

    def _doctor(self, rows: dict[str, dict[str, Any]]) -> str:
        with self._checks(rows):
            out, _ = _run(None)
        return out

    def test_one_command_adds_them_all(self, rows) -> None:
        out = self._doctor(rows)
        lines = out.splitlines()
        hint = lines.index("  • Playwright, RapidOCR, anydoc, Serve:")
        assert lines[hint + 1] == (
            '    uv tool install "markitai[browser,legacy,ocr,serve] @ /src/m" --force'
        )
        assert out.count("uv tool install") == 1

    def test_one_missing_extra_keeps_its_own_line(self, rows) -> None:
        out = self._doctor({"_check_serve": rows["_check_serve"]})
        lines = out.splitlines()
        hint = lines.index("  • Serve:")
        assert lines[hint + 1] == (
            '    uv tool install "markitai[serve] @ /src/m" --force'
        )

    def test_json_keeps_one_hint_per_check(self, rows) -> None:
        buf = io.StringIO()
        with self._checks(rows), patch.object(doctor_mod.click, "echo", buf.write):
            doctor_mod._doctor_impl(as_json=True)
        report = json.loads(buf.getvalue())
        assert report["serve"]["install_extra"] == "serve"
        assert report["anydoc"]["install_hint"] == (
            'uv tool install "markitai[legacy] @ /src/m" --force'
        )
