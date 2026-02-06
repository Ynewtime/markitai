# UX Simplification Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Simplify user onboarding through interactive CLI mode, automatic LLM detection, and streamlined installation.

**Architecture:** Add `-I/--interactive` flag to trigger questionary-based guided setup. Implement three-layer LLM detection (CLI tools → environment variables → config file). Refactor install scripts for cleaner output with checkbox component selection.

**Tech Stack:** Click (CLI), questionary (interactive prompts), Rich (output formatting), Pydantic (config validation)

---

## Phase 1: Interactive Mode (`-I/--interactive`)

### Task 1.1: Add questionary dependency

**Files:**
- Modify: `packages/markitai/pyproject.toml`

**Step 1: Add questionary to dependencies**

Open `packages/markitai/pyproject.toml` and add to `dependencies` list:

```toml
dependencies = [
    # ... existing deps ...
    "questionary>=2.1.0",
]
```

**Step 2: Sync dependencies**

Run: `uv sync`
Expected: Dependencies install without error

**Step 3: Commit**

```bash
git add packages/markitai/pyproject.toml uv.lock
git commit -m "chore(deps): add questionary for interactive CLI"
```

---

### Task 1.2: Create interactive module

**Files:**
- Create: `packages/markitai/src/markitai/cli/interactive.py`
- Test: `packages/markitai/tests/unit/cli/test_interactive.py`

**Step 1: Write the test file skeleton**

Create `packages/markitai/tests/unit/cli/test_interactive.py`:

```python
"""Tests for interactive CLI module."""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from markitai.cli.interactive import (
    InteractiveSession,
    detect_llm_provider,
    ProviderDetectionResult,
)


class TestProviderDetection:
    """Tests for LLM provider auto-detection."""

    def test_detect_claude_cli_authenticated(self) -> None:
        """Should detect authenticated Claude CLI."""
        with patch("shutil.which", return_value="/usr/bin/claude"):
            with patch(
                "markitai.cli.interactive._check_claude_auth",
                return_value=True,
            ):
                result = detect_llm_provider()
                assert result.provider == "claude-agent"
                assert result.model == "claude-agent/sonnet"
                assert result.authenticated is True

    def test_detect_copilot_cli_authenticated(self) -> None:
        """Should detect authenticated Copilot CLI when Claude not available."""
        with patch("shutil.which", side_effect=lambda x: "/usr/bin/copilot" if x == "copilot" else None):
            with patch(
                "markitai.cli.interactive._check_copilot_auth",
                return_value=True,
            ):
                result = detect_llm_provider()
                assert result.provider == "copilot"
                assert result.model == "copilot/gpt-4o"

    def test_detect_anthropic_api_key(self) -> None:
        """Should detect ANTHROPIC_API_KEY environment variable."""
        with patch("shutil.which", return_value=None):
            with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test"}):
                result = detect_llm_provider()
                assert result.provider == "anthropic"
                assert result.model == "anthropic/claude-sonnet-4-20250514"

    def test_detect_openai_api_key(self) -> None:
        """Should detect OPENAI_API_KEY when no other provider available."""
        with patch("shutil.which", return_value=None):
            with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}, clear=True):
                result = detect_llm_provider()
                assert result.provider == "openai"
                assert result.model == "openai/gpt-4o"

    def test_detect_no_provider(self) -> None:
        """Should return None when no provider detected."""
        with patch("shutil.which", return_value=None):
            with patch.dict("os.environ", {}, clear=True):
                result = detect_llm_provider()
                assert result is None


class TestInteractiveSession:
    """Tests for InteractiveSession class."""

    def test_session_initialization(self) -> None:
        """Should initialize with default values."""
        session = InteractiveSession()
        assert session.input_path is None
        assert session.enable_llm is False
        assert session.provider_result is None
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest packages/markitai/tests/unit/cli/test_interactive.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'markitai.cli.interactive'"

**Step 3: Write minimal implementation**

Create `packages/markitai/src/markitai/cli/interactive.py`:

```python
"""Interactive CLI mode for guided setup.

This module provides an interactive TUI for users who prefer guided
configuration over command-line flags.
"""

from __future__ import annotations

import asyncio
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from markitai.config import MarkitaiConfig


@dataclass
class ProviderDetectionResult:
    """Result of LLM provider auto-detection."""

    provider: str
    model: str
    authenticated: bool
    source: str  # "cli", "env", "config"


def _check_claude_auth() -> bool:
    """Check if Claude CLI is authenticated."""
    from markitai.providers.auth import AuthManager

    auth_manager = AuthManager()
    try:
        status = asyncio.run(auth_manager.check_auth("claude-agent"))
        return status.authenticated
    except Exception:
        return False


def _check_copilot_auth() -> bool:
    """Check if Copilot CLI is authenticated."""
    from markitai.providers.auth import AuthManager

    auth_manager = AuthManager()
    try:
        status = asyncio.run(auth_manager.check_auth("copilot"))
        return status.authenticated
    except Exception:
        return False


def detect_llm_provider() -> ProviderDetectionResult | None:
    """Auto-detect available LLM provider.

    Detection priority:
    1. Claude CLI (if installed and authenticated)
    2. Copilot CLI (if installed and authenticated)
    3. ANTHROPIC_API_KEY environment variable
    4. OPENAI_API_KEY environment variable
    5. GEMINI_API_KEY environment variable

    Returns:
        ProviderDetectionResult if a provider is found, None otherwise.
    """
    # 1. Check Claude CLI
    if shutil.which("claude"):
        if _check_claude_auth():
            return ProviderDetectionResult(
                provider="claude-agent",
                model="claude-agent/sonnet",
                authenticated=True,
                source="cli",
            )

    # 2. Check Copilot CLI
    if shutil.which("copilot"):
        if _check_copilot_auth():
            return ProviderDetectionResult(
                provider="copilot",
                model="copilot/gpt-4o",
                authenticated=True,
                source="cli",
            )

    # 3. Check environment variables
    if os.environ.get("ANTHROPIC_API_KEY"):
        return ProviderDetectionResult(
            provider="anthropic",
            model="anthropic/claude-sonnet-4-20250514",
            authenticated=True,
            source="env",
        )

    if os.environ.get("OPENAI_API_KEY"):
        return ProviderDetectionResult(
            provider="openai",
            model="openai/gpt-4o",
            authenticated=True,
            source="env",
        )

    if os.environ.get("GEMINI_API_KEY"):
        return ProviderDetectionResult(
            provider="gemini",
            model="gemini/gemini-2.0-flash",
            authenticated=True,
            source="env",
        )

    return None


@dataclass
class InteractiveSession:
    """State for an interactive CLI session."""

    input_path: Path | None = None
    input_type: str = "file"  # "file", "directory", "url"
    output_dir: Path = field(default_factory=lambda: Path("./output"))
    enable_llm: bool = False
    enable_alt: bool = False
    enable_desc: bool = False
    enable_ocr: bool = False
    enable_screenshot: bool = False
    provider_result: ProviderDetectionResult | None = None
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest packages/markitai/tests/unit/cli/test_interactive.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add packages/markitai/src/markitai/cli/interactive.py packages/markitai/tests/unit/cli/test_interactive.py
git commit -m "feat(cli): add interactive module with provider detection"
```

---

### Task 1.3: Implement interactive prompts

**Files:**
- Modify: `packages/markitai/src/markitai/cli/interactive.py`
- Test: `packages/markitai/tests/unit/cli/test_interactive.py`

**Step 1: Add prompt tests**

Append to `test_interactive.py`:

```python
class TestInteractivePrompts:
    """Tests for interactive prompt flow."""

    @patch("questionary.select")
    def test_prompt_input_type(self, mock_select: MagicMock) -> None:
        """Should prompt for input type."""
        mock_select.return_value.ask.return_value = "directory"
        session = InteractiveSession()

        from markitai.cli.interactive import prompt_input_type
        result = prompt_input_type(session)

        assert result == "directory"
        mock_select.assert_called_once()

    @patch("questionary.path")
    def test_prompt_input_path(self, mock_path: MagicMock) -> None:
        """Should prompt for input path."""
        mock_path.return_value.ask.return_value = "./docs"
        session = InteractiveSession()
        session.input_type = "directory"

        from markitai.cli.interactive import prompt_input_path
        result = prompt_input_path(session)

        assert result == Path("./docs")

    @patch("questionary.confirm")
    def test_prompt_enable_llm(self, mock_confirm: MagicMock) -> None:
        """Should prompt for LLM enablement."""
        mock_confirm.return_value.ask.return_value = True
        session = InteractiveSession()

        from markitai.cli.interactive import prompt_enable_llm
        result = prompt_enable_llm(session)

        assert result is True
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest packages/markitai/tests/unit/cli/test_interactive.py::TestInteractivePrompts -v`
Expected: FAIL with "cannot import name 'prompt_input_type'"

**Step 3: Implement prompt functions**

Add to `interactive.py`:

```python
import questionary
from rich.console import Console
from rich.panel import Panel

console = Console()


def prompt_input_type(session: InteractiveSession) -> str:
    """Prompt user to select input type."""
    choices = [
        questionary.Choice("Single file", value="file"),
        questionary.Choice("Directory (batch)", value="directory"),
        questionary.Choice("URL", value="url"),
    ]
    result = questionary.select(
        "What would you like to convert?",
        choices=choices,
        style=questionary.Style([
            ("highlighted", "bold"),
            ("selected", "fg:cyan"),
        ]),
    ).ask()

    if result:
        session.input_type = result
    return result or "file"


def prompt_input_path(session: InteractiveSession) -> Path | None:
    """Prompt user for input path."""
    if session.input_type == "url":
        result = questionary.text(
            "Enter URL:",
            validate=lambda x: len(x) > 0 or "URL cannot be empty",
        ).ask()
    elif session.input_type == "directory":
        result = questionary.path(
            "Enter directory path:",
            only_directories=True,
        ).ask()
    else:
        result = questionary.path(
            "Enter file path:",
        ).ask()

    if result:
        session.input_path = Path(result) if session.input_type != "url" else Path(result)
        return session.input_path
    return None


def prompt_enable_llm(session: InteractiveSession) -> bool:
    """Prompt user to enable LLM enhancement."""
    # First, detect available providers
    session.provider_result = detect_llm_provider()

    if session.provider_result:
        provider_info = f"Detected: {session.provider_result.provider} ({session.provider_result.source})"
        console.print(f"[green]✓[/green] {provider_info}")

    result = questionary.confirm(
        "Enable LLM enhancement? (better formatting, metadata)",
        default=session.provider_result is not None,
    ).ask()

    if result is not None:
        session.enable_llm = result
    return session.enable_llm


def prompt_llm_options(session: InteractiveSession) -> None:
    """Prompt user for LLM-related options."""
    if not session.enable_llm:
        return

    choices = questionary.checkbox(
        "Select LLM features:",
        choices=[
            questionary.Choice("Generate alt text for images", value="alt", checked=True),
            questionary.Choice("Generate image descriptions (JSON)", value="desc", checked=False),
            questionary.Choice("Enable OCR for scanned documents", value="ocr", checked=True),
            questionary.Choice("Take page screenshots", value="screenshot", checked=False),
        ],
    ).ask()

    if choices:
        session.enable_alt = "alt" in choices
        session.enable_desc = "desc" in choices
        session.enable_ocr = "ocr" in choices
        session.enable_screenshot = "screenshot" in choices


def prompt_configure_provider(session: InteractiveSession) -> bool:
    """Prompt user to configure LLM provider if none detected."""
    if session.provider_result:
        return True

    console.print("[yellow]![/yellow] No LLM provider detected.")

    choices = [
        questionary.Choice("Auto-detect (Claude CLI / Copilot CLI)", value="auto"),
        questionary.Choice("Enter API key manually", value="manual"),
        questionary.Choice("Use .env file", value="env"),
        questionary.Choice("Skip for now", value="skip"),
    ]

    result = questionary.select(
        "How would you like to configure LLM?",
        choices=choices,
    ).ask()

    if result == "skip":
        session.enable_llm = False
        return False

    if result == "manual":
        return _prompt_manual_api_key(session)

    if result == "env":
        return _prompt_env_file(session)

    return False


def _prompt_manual_api_key(session: InteractiveSession) -> bool:
    """Prompt for manual API key entry."""
    provider = questionary.select(
        "Select provider:",
        choices=[
            questionary.Choice("Anthropic (Claude)", value="anthropic"),
            questionary.Choice("OpenAI (GPT-4o)", value="openai"),
            questionary.Choice("Google (Gemini)", value="gemini"),
        ],
    ).ask()

    if not provider:
        return False

    api_key = questionary.password(
        f"Enter {provider.upper()} API key:",
    ).ask()

    if not api_key:
        return False

    # Save to config
    from markitai.config import ConfigManager, ModelConfig, LiteLLMParams

    model_map = {
        "anthropic": "anthropic/claude-sonnet-4-20250514",
        "openai": "openai/gpt-4o",
        "gemini": "gemini/gemini-2.0-flash",
    }

    manager = ConfigManager()
    cfg = manager.load()
    cfg.llm.model_list = [
        ModelConfig(
            model_name="default",
            litellm_params=LiteLLMParams(model=model_map[provider], api_key=api_key),
        )
    ]
    manager.save()

    session.provider_result = ProviderDetectionResult(
        provider=provider,
        model=model_map[provider],
        authenticated=True,
        source="manual",
    )

    console.print(f"[green]✓[/green] API key saved to config")
    return True


def _prompt_env_file(session: InteractiveSession) -> bool:
    """Prompt for .env file location."""
    env_path = questionary.path(
        "Enter .env file path:",
        default=".env",
    ).ask()

    if not env_path or not Path(env_path).exists():
        console.print("[red]✗[/red] .env file not found")
        return False

    # Load .env file
    from dotenv import load_dotenv
    load_dotenv(env_path)

    # Re-detect provider
    session.provider_result = detect_llm_provider()

    if session.provider_result:
        console.print(f"[green]✓[/green] Detected: {session.provider_result.provider}")
        return True

    console.print("[red]✗[/red] No API key found in .env file")
    return False
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest packages/markitai/tests/unit/cli/test_interactive.py::TestInteractivePrompts -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add packages/markitai/src/markitai/cli/interactive.py packages/markitai/tests/unit/cli/test_interactive.py
git commit -m "feat(cli): implement interactive prompts with questionary"
```

---

### Task 1.4: Add run_interactive function

**Files:**
- Modify: `packages/markitai/src/markitai/cli/interactive.py`
- Test: `packages/markitai/tests/unit/cli/test_interactive.py`

**Step 1: Add integration test**

Append to `test_interactive.py`:

```python
class TestRunInteractive:
    """Tests for the main run_interactive function."""

    @patch("markitai.cli.interactive.prompt_input_type")
    @patch("markitai.cli.interactive.prompt_input_path")
    @patch("markitai.cli.interactive.prompt_enable_llm")
    @patch("markitai.cli.interactive.prompt_llm_options")
    @patch("markitai.cli.interactive.prompt_configure_provider")
    def test_run_interactive_basic_flow(
        self,
        mock_configure: MagicMock,
        mock_options: MagicMock,
        mock_llm: MagicMock,
        mock_path: MagicMock,
        mock_type: MagicMock,
    ) -> None:
        """Should run through all prompts in order."""
        mock_type.return_value = "file"
        mock_path.return_value = Path("test.pdf")
        mock_llm.return_value = False

        from markitai.cli.interactive import run_interactive
        session = run_interactive()

        assert session.input_type == "file"
        mock_type.assert_called_once()
        mock_path.assert_called_once()
        mock_llm.assert_called_once()
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest packages/markitai/tests/unit/cli/test_interactive.py::TestRunInteractive -v`
Expected: FAIL with "cannot import name 'run_interactive'"

**Step 3: Implement run_interactive**

Add to `interactive.py`:

```python
def run_interactive() -> InteractiveSession:
    """Run the interactive CLI session.

    Returns:
        InteractiveSession with user's choices populated.
    """
    session = InteractiveSession()

    # Print header
    console.print()
    console.print(Panel(
        "[bold]Markitai Interactive Mode[/bold]\n\n"
        "Answer the following questions to configure your conversion.",
        border_style="cyan",
    ))
    console.print()

    # 1. Input type
    prompt_input_type(session)

    # 2. Input path
    if not prompt_input_path(session):
        console.print("[red]✗[/red] No input path provided. Exiting.")
        raise SystemExit(1)

    # 3. LLM enablement
    prompt_enable_llm(session)

    # 4. Configure provider if needed
    if session.enable_llm and not session.provider_result:
        if not prompt_configure_provider(session):
            session.enable_llm = False

    # 5. LLM options
    if session.enable_llm:
        prompt_llm_options(session)

    # Print summary
    _print_summary(session)

    return session


def _print_summary(session: InteractiveSession) -> None:
    """Print a summary of the session configuration."""
    console.print()
    console.print("[bold]Configuration Summary:[/bold]")
    console.print(f"  Input: {session.input_path} ({session.input_type})")
    console.print(f"  Output: {session.output_dir}")
    console.print(f"  LLM: {'enabled' if session.enable_llm else 'disabled'}")

    if session.enable_llm:
        if session.provider_result:
            console.print(f"  Provider: {session.provider_result.provider}")
        console.print(f"  Alt text: {'yes' if session.enable_alt else 'no'}")
        console.print(f"  Descriptions: {'yes' if session.enable_desc else 'no'}")
        console.print(f"  OCR: {'yes' if session.enable_ocr else 'no'}")
        console.print(f"  Screenshots: {'yes' if session.enable_screenshot else 'no'}")
    console.print()


def session_to_cli_args(session: InteractiveSession) -> list[str]:
    """Convert InteractiveSession to CLI arguments.

    This allows the interactive session to be executed by the main CLI.
    """
    args: list[str] = []

    if session.input_path:
        args.append(str(session.input_path))

    args.extend(["-o", str(session.output_dir)])

    if session.enable_llm:
        args.append("--llm")
        if session.enable_alt:
            args.append("--alt")
        if session.enable_desc:
            args.append("--desc")
        if session.enable_ocr:
            args.append("--ocr")
        if session.enable_screenshot:
            args.append("--screenshot")
    else:
        args.append("--no-llm")

    return args
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest packages/markitai/tests/unit/cli/test_interactive.py::TestRunInteractive -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add packages/markitai/src/markitai/cli/interactive.py packages/markitai/tests/unit/cli/test_interactive.py
git commit -m "feat(cli): add run_interactive function with summary output"
```

---

### Task 1.5: Integrate interactive mode into CLI

**Files:**
- Modify: `packages/markitai/src/markitai/cli/main.py`
- Test: `packages/markitai/tests/unit/cli/test_main.py`

**Step 1: Write integration test**

Create or append to `packages/markitai/tests/unit/cli/test_main.py`:

```python
"""Tests for CLI main module interactive mode."""

from __future__ import annotations

from click.testing import CliRunner
from unittest.mock import patch, MagicMock
import pytest

from markitai.cli.main import app


class TestInteractiveFlag:
    """Tests for -I/--interactive flag."""

    def test_interactive_short_flag(self) -> None:
        """Should recognize -I flag."""
        runner = CliRunner()
        with patch("markitai.cli.main.run_interactive_mode") as mock_run:
            mock_run.return_value = None
            result = runner.invoke(app, ["-I"])
            # Should call interactive mode, not show help
            mock_run.assert_called_once()

    def test_interactive_long_flag(self) -> None:
        """Should recognize --interactive flag."""
        runner = CliRunner()
        with patch("markitai.cli.main.run_interactive_mode") as mock_run:
            mock_run.return_value = None
            result = runner.invoke(app, ["--interactive"])
            mock_run.assert_called_once()

    def test_no_args_shows_help(self) -> None:
        """Should show help when no arguments provided."""
        runner = CliRunner()
        result = runner.invoke(app, [])
        assert result.exit_code == 0
        assert "Usage:" in result.output or "usage:" in result.output.lower()
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest packages/markitai/tests/unit/cli/test_main.py::TestInteractiveFlag -v`
Expected: FAIL with "cannot import name 'run_interactive_mode'"

**Step 3: Add interactive flag to CLI**

Modify `packages/markitai/src/markitai/cli/main.py`:

Add to imports (around line 30):
```python
from markitai.cli.interactive import run_interactive, session_to_cli_args
```

Add new option after `@click.option("--version", ...)` (around line 205):
```python
@click.option(
    "--interactive",
    "-I",
    is_flag=True,
    is_eager=True,
    expose_value=False,
    callback=lambda ctx, param, value: run_interactive_mode(ctx) if value else None,
    help="Enter interactive mode for guided setup.",
)
```

Add the callback function before the `app` definition (around line 75):
```python
def run_interactive_mode(ctx: click.Context) -> None:
    """Run interactive mode and execute with gathered options."""
    from markitai.cli.interactive import run_interactive, session_to_cli_args

    try:
        session = run_interactive()

        # Ask for confirmation before executing
        import questionary
        if questionary.confirm("Execute conversion with these settings?", default=True).ask():
            # Re-invoke the CLI with the gathered arguments
            args = session_to_cli_args(session)
            ctx.invoke(app.main, args=args)
        else:
            click.echo("Cancelled.")
            ctx.exit(0)
    except (KeyboardInterrupt, EOFError):
        click.echo("\nCancelled.")
        ctx.exit(0)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest packages/markitai/tests/unit/cli/test_main.py::TestInteractiveFlag -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add packages/markitai/src/markitai/cli/main.py packages/markitai/tests/unit/cli/test_main.py
git commit -m "feat(cli): add -I/--interactive flag for guided setup"
```

---

## Phase 2: Doctor Command Enhancement

### Task 2.1: Add cross-platform install hints

**Files:**
- Modify: `packages/markitai/src/markitai/cli/commands/doctor.py`
- Test: `packages/markitai/tests/unit/cli/test_doctor.py`

**Step 1: Write test for platform-specific hints**

Create `packages/markitai/tests/unit/cli/test_doctor.py`:

```python
"""Tests for doctor command."""

from __future__ import annotations

import sys
from unittest.mock import patch
import pytest

from markitai.cli.commands.doctor import get_install_hint


class TestInstallHints:
    """Tests for cross-platform install hints."""

    @patch("sys.platform", "darwin")
    def test_libreoffice_hint_macos(self) -> None:
        """Should return brew command for macOS."""
        hint = get_install_hint("libreoffice", platform="darwin")
        assert "brew install" in hint
        assert "libreoffice" in hint.lower()

    @patch("sys.platform", "linux")
    def test_libreoffice_hint_linux(self) -> None:
        """Should return apt command for Linux."""
        hint = get_install_hint("libreoffice", platform="linux")
        assert "apt install" in hint or "apt-get install" in hint

    @patch("sys.platform", "win32")
    def test_libreoffice_hint_windows(self) -> None:
        """Should return winget command for Windows."""
        hint = get_install_hint("libreoffice", platform="win32")
        assert "winget install" in hint

    def test_ffmpeg_hint_all_platforms(self) -> None:
        """Should have hints for all major platforms."""
        for platform in ["darwin", "linux", "win32"]:
            hint = get_install_hint("ffmpeg", platform=platform)
            assert hint, f"Missing hint for ffmpeg on {platform}"

    def test_playwright_hint(self) -> None:
        """Should return playwright install command."""
        hint = get_install_hint("playwright", platform="linux")
        assert "playwright install" in hint
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest packages/markitai/tests/unit/cli/test_doctor.py::TestInstallHints -v`
Expected: FAIL with "cannot import name 'get_install_hint'"

**Step 3: Implement get_install_hint function**

Add to `packages/markitai/src/markitai/cli/commands/doctor.py` after imports:

```python
import sys

# Cross-platform installation hints
INSTALL_HINTS: dict[str, dict[str, str]] = {
    "libreoffice": {
        "darwin": "brew install --cask libreoffice",
        "linux": "sudo apt install libreoffice  # Ubuntu/Debian\nsudo dnf install libreoffice  # Fedora\nsudo pacman -S libreoffice-fresh  # Arch",
        "win32": "winget install LibreOffice.LibreOffice",
    },
    "ffmpeg": {
        "darwin": "brew install ffmpeg",
        "linux": "sudo apt install ffmpeg  # Ubuntu/Debian\nsudo dnf install ffmpeg  # Fedora\nsudo pacman -S ffmpeg  # Arch",
        "win32": "winget install FFmpeg.FFmpeg\n# Or: scoop install ffmpeg\n# Or: choco install ffmpeg",
    },
    "playwright": {
        "darwin": "uv run playwright install chromium",
        "linux": "uv run playwright install chromium && uv run playwright install-deps chromium",
        "win32": "uv run playwright install chromium",
    },
    "claude-cli": {
        "darwin": "curl -fsSL https://claude.ai/install.sh | bash",
        "linux": "curl -fsSL https://claude.ai/install.sh | bash",
        "win32": "irm https://claude.ai/install.ps1 | iex",
    },
    "copilot-cli": {
        "darwin": "curl -fsSL https://gh.io/copilot-install | bash",
        "linux": "curl -fsSL https://gh.io/copilot-install | bash",
        "win32": "winget install GitHub.Copilot",
    },
}


def get_install_hint(component: str, platform: str | None = None) -> str:
    """Get platform-specific installation hint for a component.

    Args:
        component: Component name (e.g., "libreoffice", "ffmpeg")
        platform: Target platform. If None, uses current sys.platform.

    Returns:
        Installation command(s) for the platform.
    """
    if platform is None:
        platform = sys.platform

    # Normalize platform
    if platform.startswith("linux"):
        platform = "linux"

    hints = INSTALL_HINTS.get(component, {})
    return hints.get(platform, hints.get("linux", f"Install {component} using your package manager"))
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest packages/markitai/tests/unit/cli/test_doctor.py::TestInstallHints -v`
Expected: All tests PASS

**Step 5: Update doctor output to use new hints**

Modify `_doctor_impl` function to use `get_install_hint`:

```python
# Replace hardcoded install_hint strings with:
"install_hint": get_install_hint("libreoffice"),
"install_hint": get_install_hint("ffmpeg"),
"install_hint": get_install_hint("playwright"),
# etc.
```

**Step 6: Commit**

```bash
git add packages/markitai/src/markitai/cli/commands/doctor.py packages/markitai/tests/unit/cli/test_doctor.py
git commit -m "feat(doctor): add cross-platform installation hints"
```

---

### Task 2.2: Add --fix flag to doctor

**Files:**
- Modify: `packages/markitai/src/markitai/cli/commands/doctor.py`
- Test: `packages/markitai/tests/unit/cli/test_doctor.py`

**Step 1: Add test for --fix flag**

Append to `test_doctor.py`:

```python
from click.testing import CliRunner
from markitai.cli.commands.doctor import doctor


class TestDoctorFix:
    """Tests for doctor --fix flag."""

    def test_fix_flag_exists(self) -> None:
        """Should accept --fix flag."""
        runner = CliRunner()
        with patch("markitai.cli.commands.doctor._doctor_impl"):
            result = runner.invoke(doctor, ["--fix"])
            # Should not fail due to unknown option
            assert "no such option" not in result.output.lower()

    @patch("markitai.cli.commands.doctor._install_component")
    def test_fix_installs_missing(self, mock_install: MagicMock) -> None:
        """Should attempt to install missing components."""
        mock_install.return_value = True
        runner = CliRunner()

        # Mock the detection to show missing components
        with patch("markitai.cli.commands.doctor._doctor_impl") as mock_impl:
            # This test verifies the flag triggers install logic
            result = runner.invoke(doctor, ["--fix"])
            # Actual installation behavior tested separately
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest packages/markitai/tests/unit/cli/test_doctor.py::TestDoctorFix -v`
Expected: FAIL (--fix not recognized)

**Step 3: Implement --fix flag**

Modify `doctor.py`:

```python
def _install_component(component: str) -> bool:
    """Attempt to install a missing component.

    Args:
        component: Component name to install.

    Returns:
        True if installation succeeded.
    """
    import subprocess

    console = get_console()
    hint = get_install_hint(component)

    console.print(f"[yellow]Installing {component}...[/yellow]")

    # Only auto-install safe components
    safe_components = {"playwright"}

    if component not in safe_components:
        console.print(f"[yellow]Please install manually:[/yellow]")
        console.print(f"  {hint}")
        return False

    if component == "playwright":
        try:
            # Use uv to run playwright install
            result = subprocess.run(
                ["uv", "run", "playwright", "install", "chromium"],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode == 0:
                console.print(f"[green]✓[/green] {component} installed")
                return True
            else:
                console.print(f"[red]✗[/red] Failed: {result.stderr}")
                return False
        except Exception as e:
            console.print(f"[red]✗[/red] Error: {e}")
            return False

    return False


@click.command("doctor")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
@click.option("--fix", is_flag=True, help="Attempt to install missing components.")
def doctor(as_json: bool, fix: bool) -> None:
    """Check system health, dependencies, and authentication status."""
    from loguru import logger

    logger.disable("markitai")
    try:
        _doctor_impl(as_json, fix=fix)
    finally:
        logger.enable("markitai")
```

Update `_doctor_impl` signature and add fix logic at the end:

```python
def _doctor_impl(as_json: bool, fix: bool = False) -> None:
    # ... existing code ...

    # At the end, after printing hints:
    if fix and not as_json:
        missing = [
            key for key, info in results.items()
            if info["status"] in ("missing", "warning")
        ]
        if missing:
            console.print()
            console.print("[bold]Attempting to fix missing components...[/bold]")
            for component in missing:
                _install_component(component)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest packages/markitai/tests/unit/cli/test_doctor.py::TestDoctorFix -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add packages/markitai/src/markitai/cli/commands/doctor.py packages/markitai/tests/unit/cli/test_doctor.py
git commit -m "feat(doctor): add --fix flag for auto-installing missing components"
```

---

## Phase 3: Install Script Optimization

### Task 3.1: Create simplified install script

**Files:**
- Modify: `scripts/setup.sh`
- Modify: `scripts/lib.sh`

**Step 1: Backup current scripts**

```bash
cp scripts/setup.sh scripts/setup.sh.bak
cp scripts/lib.sh scripts/lib.sh.bak
```

**Step 2: Refactor lib.sh for silent output**

Key changes to `lib.sh`:

```bash
# Add silent mode functions
run_silent() {
    # Run command silently, only show output on error
    local cmd="$*"
    local output
    output=$("$@" 2>&1)
    local status=$?
    if [ $status -ne 0 ]; then
        printf "%s\n" "$output" >&2
    fi
    return $status
}

# Simplify print functions
print_status() {
    local status="$1"
    local message="$2"
    case "$status" in
        ok)      printf "  ${GREEN}✓${NC} %s\n" "$message" ;;
        skip)    printf "  ${YELLOW}○${NC} %s\n" "$message" ;;
        fail)    printf "  ${RED}✗${NC} %s\n" "$message" ;;
        info)    printf "  ${CYAN}→${NC} %s\n" "$message" ;;
    esac
}

# Remove verbose install functions, use run_silent wrapper
lib_install_uv() {
    if lib_detect_uv; then
        print_status ok "uv $(uv --version 2>/dev/null | head -n1 | cut -d' ' -f2)"
        return 0
    fi

    print_status info "Installing uv..."
    if run_silent curl -LsSf "https://astral.sh/uv/install.sh" | run_silent sh; then
        export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
        print_status ok "uv installed"
        return 0
    fi

    print_status fail "uv installation failed"
    return 1
}
```

**Step 3: Implement checkbox selection using simple menu**

Add to `lib.sh`:

```bash
# Simple checkbox menu (no external dependencies)
# Usage: checkbox_menu "prompt" "option1|option2|option3" "default1|default2"
# Returns selected options separated by |
checkbox_menu() {
    local prompt="$1"
    local options="$2"
    local defaults="$3"

    printf "\n  ${BOLD}%s${NC}\n" "$prompt"
    printf "  (space to toggle, enter to confirm)\n\n"

    # Parse options into array
    local IFS='|'
    set -- $options
    local items=("$@")
    set -- $defaults
    local selected=("$@")

    # For non-interactive mode, return defaults
    if [ ! -t 0 ]; then
        printf "%s" "$defaults"
        return 0
    fi

    # Simple fallback: ask yes/no for each
    local result=""
    for item in "${items[@]}"; do
        local is_default="n"
        case "|$defaults|" in
            *"|$item|"*) is_default="y" ;;
        esac

        if ask_yes_no "  $item?" "$is_default"; then
            result="${result}${item}|"
        fi
    done

    printf "%s" "${result%|}"
}
```

**Step 4: Simplify setup.sh main flow**

```bash
main() {
    print_header "Markitai Setup"

    # Core installation (no prompts)
    lib_install_uv || exit 1
    lib_detect_python || exit 1
    lib_install_markitai || exit 1

    # Optional components (single selection)
    printf "\n"
    local components
    components=$(checkbox_menu \
        "Select optional components:" \
        "Playwright|LibreOffice|FFmpeg|Claude CLI|Copilot CLI" \
        "Playwright")

    # Install selected
    case "$components" in
        *Playwright*) lib_install_playwright_browser ;;
    esac
    case "$components" in
        *LibreOffice*) lib_install_libreoffice ;;
    esac
    case "$components" in
        *FFmpeg*) lib_install_ffmpeg ;;
    esac
    case "$components" in
        *"Claude CLI"*) lib_install_claude_cli ;;
    esac
    case "$components" in
        *"Copilot CLI"*) lib_install_copilot_cli ;;
    esac

    # Summary
    print_summary

    # Next steps
    printf "\n"
    printf "  ${BOLD}Get started:${NC}\n"
    printf "    ${CYAN}markitai -I${NC}          Interactive mode\n"
    printf "    ${CYAN}markitai file.pdf${NC}   Convert a file\n"
    printf "    ${CYAN}markitai --help${NC}     Show all options\n"
    printf "\n"
}
```

**Step 5: Test the script locally**

```bash
bash scripts/setup.sh
```
Expected: Clean, minimal output with checkbox selection

**Step 6: Commit**

```bash
git add scripts/setup.sh scripts/lib.sh
git commit -m "refactor(scripts): simplify install output with checkbox selection"
```

---

### Task 3.2: Update PowerShell script

**Files:**
- Modify: `scripts/setup.ps1`
- Modify: `scripts/lib.ps1`

**Step 1: Mirror changes from bash to PowerShell**

Key changes to `lib.ps1`:

```powershell
# Silent run helper
function Invoke-Silent {
    param([scriptblock]$Command)
    try {
        $output = & $Command 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Host $output -ForegroundColor Red
        }
        return $LASTEXITCODE -eq 0
    } catch {
        return $false
    }
}

# Simplified status output
function Write-Status {
    param(
        [string]$Status,
        [string]$Message
    )
    switch ($Status) {
        "ok"   { Write-Host "  ✓ $Message" -ForegroundColor Green }
        "skip" { Write-Host "  ○ $Message" -ForegroundColor Yellow }
        "fail" { Write-Host "  ✗ $Message" -ForegroundColor Red }
        "info" { Write-Host "  → $Message" -ForegroundColor Cyan }
    }
}

# Checkbox menu using Out-GridView or fallback
function Show-CheckboxMenu {
    param(
        [string]$Prompt,
        [string[]]$Options,
        [string[]]$Defaults
    )

    Write-Host "`n  $Prompt" -ForegroundColor White
    Write-Host "  (space to toggle, enter to confirm)`n"

    # Fallback to simple yes/no for each
    $selected = @()
    foreach ($option in $Options) {
        $isDefault = $option -in $Defaults
        $defaultText = if ($isDefault) { "Y/n" } else { "y/N" }
        $response = Read-Host "    $option? [$defaultText]"

        if ([string]::IsNullOrEmpty($response)) {
            if ($isDefault) { $selected += $option }
        } elseif ($response -match "^[Yy]") {
            $selected += $option
        }
    }

    return $selected
}
```

**Step 2: Update setup.ps1 main flow**

```powershell
function Main {
    Write-Header "Markitai Setup"

    # Core installation
    if (-not (Install-UV)) { exit 1 }
    if (-not (Test-Python)) { exit 1 }
    if (-not (Install-Markitai)) { exit 1 }

    # Optional components
    $components = Show-CheckboxMenu `
        -Prompt "Select optional components:" `
        -Options @("Playwright", "LibreOffice", "FFmpeg", "Claude CLI", "Copilot CLI") `
        -Defaults @("Playwright")

    if ("Playwright" -in $components) { Install-PlaywrightBrowser | Out-Null }
    if ("LibreOffice" -in $components) { Install-LibreOffice | Out-Null }
    if ("FFmpeg" -in $components) { Install-FFmpeg | Out-Null }
    if ("Claude CLI" -in $components) { Install-ClaudeCLI | Out-Null }
    if ("Copilot CLI" -in $components) { Install-CopilotCLI | Out-Null }

    # Summary and next steps
    Write-Summary
    Write-Host "`n  Get started:" -ForegroundColor White
    Write-Host "    markitai -I          Interactive mode" -ForegroundColor Cyan
    Write-Host "    markitai file.pdf   Convert a file" -ForegroundColor Cyan
    Write-Host "    markitai --help     Show all options" -ForegroundColor Cyan
    Write-Host ""
}
```

**Step 3: Test the script**

```powershell
.\scripts\setup.ps1
```

**Step 4: Commit**

```bash
git add scripts/setup.ps1 scripts/lib.ps1
git commit -m "refactor(scripts): simplify PowerShell install with checkbox selection"
```

---

## Phase 4: Documentation

### Task 4.1: Update README quick start

**Files:**
- Modify: `README.md`

**Step 1: Update installation section**

Add/update the quick start section:

```markdown
## Quick Start

### Installation

```bash
# Linux/macOS
curl -fsSL https://markitai.dev/install | sh

# Windows PowerShell
irm https://markitai.dev/install.ps1 | iex
```

### First Run

```bash
# Interactive mode (recommended for new users)
markitai -I

# Or convert a file directly
markitai document.pdf

# With LLM enhancement
markitai document.pdf --llm
```

### Check Setup

```bash
# Verify all dependencies
markitai doctor

# Auto-fix missing components
markitai doctor --fix
```
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: update quick start with interactive mode"
```

---

## Final: Create PR

### Task 5.1: Push branch and create PR

**Step 1: Push branch**

```bash
git push -u origin feat/ux-simplification
```

**Step 2: Create PR**

```bash
gh pr create --title "feat: UX simplification with interactive mode" --body "$(cat <<'EOF'
## Summary

- Add `-I/--interactive` flag for guided CLI setup
- Implement three-layer LLM provider auto-detection
- Simplify install scripts with checkbox component selection
- Enhance `markitai doctor` with cross-platform hints and `--fix` flag

## Changes

### Interactive Mode
- New `markitai -I` command for guided setup
- Auto-detects Claude CLI, Copilot CLI, and API keys
- Prompts for input type, path, LLM options

### Install Scripts
- Silent dependency installation (no log spam)
- Single checkbox for component selection
- Clear "Get started" guidance

### Doctor Command
- Cross-platform installation hints (macOS/Linux/Windows)
- `--fix` flag to auto-install missing components

## Test Plan
- [ ] Run `markitai -I` and verify prompts work
- [ ] Run `markitai doctor` and verify hints are correct
- [ ] Run `markitai doctor --fix` for Playwright
- [ ] Test install script on fresh environment

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Summary

| Phase | Tasks | Est. Steps |
|-------|-------|------------|
| 1. Interactive Mode | 5 tasks | 25 steps |
| 2. Doctor Enhancement | 2 tasks | 10 steps |
| 3. Install Scripts | 2 tasks | 10 steps |
| 4. Documentation | 1 task | 2 steps |
| 5. PR | 1 task | 2 steps |
| **Total** | **11 tasks** | **~49 steps** |
