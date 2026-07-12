"""Regression guard for scripts/setup.sh set -e safety.

Root cause of the bug this guards against: setup.sh runs under `set -eu`,
and its optional-component installers return non-zero to signal a
declined/failed OPTIONAL step (skip → `return 2`, fail → `return 1`).
When such a call is unguarded, `set -e` treats the non-zero return as
fatal and exits the whole script — so declining LibreOffice would abort
setup before the summary/next-steps/outro ever printed.

The fix guards every non-fatal orchestration call with `|| true`. This
test fails if any of them regresses back to a bare call.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_SETUP_SH = Path(__file__).resolve().parents[4] / "scripts" / "setup.sh"
_SETUP_PS1 = Path(__file__).resolve().parents[4] / "scripts" / "setup.ps1"

# Non-fatal orchestration calls: declining/failing these must NOT abort setup.
# (install_uv / install_markitai are intentionally fatal and use their own
# `|| { ...; exit 1; }` handler, so they are not in this list.)
_NON_FATAL_CALLS = (
    "install_optional_playwright",
    "install_optional_libreoffice",
    "install_optional_ffmpeg",
    "install_optional_claude_cli",
    "install_optional_copilot_cli",
    "finalize_markitai_extras",
    "install_precommit",
)

_OPTIONAL_INSTALLERS = (
    ("install_optional_playwright", "Install-OptionalPlaywright"),
    ("install_optional_libreoffice", "Install-OptionalLibreOffice"),
    ("install_optional_ffmpeg", "Install-OptionalFFmpeg"),
    ("install_optional_claude_cli", "Install-OptionalClaudeCLI"),
    ("install_optional_copilot_cli", "Install-OptionalCopilotCLI"),
)


def _shell_function(name: str) -> str:
    """Return one top-level POSIX shell function body."""
    text = _SETUP_SH.read_text(encoding="utf-8")
    function = re.search(
        rf"^{re.escape(name)}\(\) \{{\n(?P<body>.*?)^\}}",
        text,
        re.MULTILINE | re.DOTALL,
    )
    assert function is not None, f"{name}() not found"
    return function.group("body")


def _powershell_function(name: str) -> str:
    """Return one top-level PowerShell function body."""
    text = _SETUP_PS1.read_text(encoding="utf-8")
    function = re.search(
        rf"^function {re.escape(name)}(?: \{{|\s*\([^\n]*\)\s*\{{)\n(?P<body>.*?)^\}}",
        text,
        re.MULTILINE | re.DOTALL,
    )
    assert function is not None, f"{name} not found"
    return function.group("body")


@pytest.mark.skipif(not _SETUP_SH.exists(), reason="scripts/setup.sh not present")
def test_setup_sh_runs_under_set_e() -> None:
    """The guards only matter because the script is `set -e`."""
    assert re.search(
        r"^set -eu?\b", _SETUP_SH.read_text(encoding="utf-8"), re.MULTILINE
    ), (
        "setup.sh is expected to run under set -e; if that changed, revisit "
        "whether the || true guards are still needed."
    )


@pytest.mark.skipif(not _SETUP_SH.exists(), reason="scripts/setup.sh not present")
@pytest.mark.parametrize("func", _NON_FATAL_CALLS)
def test_non_fatal_calls_are_guarded(func: str) -> None:
    """Every orchestration call to a non-fatal installer must be guarded.

    A bare `    install_optional_libreoffice` line under set -e aborts the
    script when the function returns non-zero (declined/failed).
    """
    text = _SETUP_SH.read_text(encoding="utf-8")
    # A call site is the function name as a whole indented statement. The
    # definition line ends with `()`, so `\b(?!\()` excludes it.
    bare = re.compile(rf"^[ \t]+{re.escape(func)}[ \t]*$", re.MULTILINE)
    guarded = re.compile(rf"^[ \t]+{re.escape(func)}[ \t]*\|\|", re.MULTILINE)

    bare_calls = bare.findall(text)
    # There must be at least one call site (guarded), and zero bare ones.
    assert guarded.search(text), f"expected a guarded call to {func}()"
    assert not bare_calls, (
        f"unguarded call to {func}() found: under set -e this aborts setup "
        f"when the user declines or it fails. Append ` || true`."
    )


def test_setup_sh_preserves_existing_config() -> None:
    """A repeat shell install must not run the mutating init command."""
    body = _shell_function("init_config")
    guard = body.find('[ -f "$HOME/.markitai/config.json" ]')
    init = body.find("markitai init --yes")
    assert 0 <= guard < init, "existing-config guard must run before markitai init"


def test_setup_ps1_preserves_existing_config() -> None:
    """A repeat PowerShell install must not run the mutating init command."""
    body = _powershell_function("Initialize-Config")
    guard = body.find('Test-Path "$HOME/.markitai/config.json"')
    init = body.find("markitai init --yes")
    assert 0 <= guard < init, "existing-config guard must run before markitai init"


def test_setup_scripts_only_select_supported_python_versions() -> None:
    """Installers must not select a Python version excluded by package metadata."""
    shell_text = _SETUP_SH.read_text(encoding="utf-8")
    powershell_text = _SETUP_PS1.read_text(encoding="utf-8")

    assert "uv python find '>=3.11,<3.14'" in shell_text
    assert 'uv python find ">=3.11,<3.14"' in powershell_text
    assert ">=3.11,<3.15" not in shell_text
    assert ">=3.11,<3.15" not in powershell_text


def test_setup_sh_honors_version_pin_for_existing_tool() -> None:
    """A pinned repeat install must bypass the receipt-based generic upgrade."""
    body = _shell_function("install_markitai")
    unpinned_guard = body.find(
        '[ "$MARKITAI_SOURCE" != "local" ] && [ -z "$MARKITAI_VERSION" ]'
    )
    generic_upgrade = body.find("uv tool upgrade markitai")
    exact_reinstall = body.find(
        'uv tool install "$_mi_pkg" --python "$PYTHON_CMD" --force'
    )
    version_check = body.find('"$_mi_version" != "$MARKITAI_VERSION"')

    assert 0 <= unpinned_guard < generic_upgrade < exact_reinstall < version_check
    spec = _shell_function("markitai_pkg_spec")
    assert "==$MARKITAI_VERSION" in spec


def test_setup_ps1_honors_version_pin_for_existing_tool() -> None:
    """PowerShell must also bypass a generic upgrade for an explicit pin."""
    body = _powershell_function("Install-Markitai")
    unpinned_guard = body.find(
        '$script:MARKITAI_SOURCE -ne "local" -and -not $script:MarkitaiVersion'
    )
    generic_upgrade = body.find("uv tool upgrade markitai")
    exact_reinstall = body.find("uv tool install $pkg --python $pythonArg --force")
    version_check = body.find("$version -ne $script:MarkitaiVersion")

    assert 0 <= unpinned_guard < generic_upgrade < exact_reinstall < version_check
    spec = _powershell_function("Get-MarkitaiPkgSpec")
    assert "==$($script:MarkitaiVersion)" in spec


@pytest.mark.parametrize("shell_name,powershell_name", _OPTIONAL_INSTALLERS)
def test_optional_installers_use_noninteractive_guard(
    shell_name: str, powershell_name: str
) -> None:
    """Every optional installer must route confirmation through the TTY guard."""
    assert "clack_confirm_optional" in _shell_function(shell_name)
    assert "Confirm-OptionalInstall" in _powershell_function(powershell_name)


def test_setup_sh_noninteractive_optionals_require_explicit_flag() -> None:
    """The shell helper must default off and only accept the opt-in flag."""
    confirm = _shell_function("clack_confirm_optional")
    requested = _shell_function("optional_install_requested")
    assert "if ! has_interactive_tty; then" in confirm
    assert "optional_install_requested" in confirm
    assert "MARKITAI_INSTALL_OPTIONAL:-" in requested
    assert "1|true|TRUE|yes|YES|on|ON" in requested


def test_setup_ps1_noninteractive_optionals_require_explicit_flag() -> None:
    """The PowerShell helper must have the same non-interactive opt-in contract."""
    confirm = _powershell_function("Confirm-OptionalInstall")
    requested = _powershell_function("Test-OptionalInstallRequested")
    assert "Test-InteractiveInput" in confirm
    assert "Test-OptionalInstallRequested" in confirm
    assert "MARKITAI_INSTALL_OPTIONAL" in requested
    assert "1|true|yes|on" in requested


def test_noninteractive_core_install_does_not_seed_browser_extra() -> None:
    """Fresh headless installs start from the core package unless opted in."""
    shell_text = _SETUP_SH.read_text(encoding="utf-8")
    ps_text = _SETUP_PS1.read_text(encoding="utf-8")
    assert 'MARKITAI_EXTRAS=""' in shell_text
    assert "optional_install_requested" in shell_text
    assert '$script:MARKITAI_EXTRAS = ""' in ps_text
    assert "Test-OptionalInstallRequested" in ps_text
