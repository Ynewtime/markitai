"""Regression guards for the cross-platform setup scripts.

One root cause guarded here: setup.sh runs under `set -eu`,
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
import tomllib
from pathlib import Path

import pytest

_SETUP_SH = Path(__file__).resolve().parents[4] / "scripts" / "setup.sh"
_SETUP_PS1 = Path(__file__).resolve().parents[4] / "scripts" / "setup.ps1"
_MARKITAI_PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"

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


def test_user_install_selects_serve_before_installing_markitai() -> None:
    """Serve must join the first combined requirement, never replace it later."""
    shell_flow = _shell_function("run_user_setup")
    shell_load = shell_flow.find("load_existing_markitai_extras")
    shell_select = shell_flow.find("select_markitai_serve")
    shell_install = shell_flow.find("install_markitai ||")
    shell_track = shell_flow.find("track_markitai_serve")
    assert 0 <= shell_load < shell_select < shell_install < shell_track

    ps_flow = _powershell_function("Run-UserSetup")
    ps_load = ps_flow.find("Import-MarkitaiReceiptExtras")
    ps_select = ps_flow.find("Select-MarkitaiServe")
    ps_install = ps_flow.find("Install-Markitai))")
    ps_track = ps_flow.find("Track-MarkitaiServe")
    assert 0 <= ps_load < ps_select < ps_install < ps_track

    shell_select_body = _shell_function("select_markitai_serve")
    assert "clack_confirm_optional" in shell_select_body
    assert 'install_markitai_extra "serve"' in shell_select_body

    ps_select_body = _powershell_function("Select-MarkitaiServe")
    assert "Confirm-OptionalInstall" in ps_select_body
    assert 'Install-MarkitaiExtra -ExtraName "serve"' in ps_select_body

    shell_text = _SETUP_SH.read_text(encoding="utf-8")
    ps_text = _SETUP_PS1.read_text(encoding="utf-8")
    assert "confirm_serve)" in shell_text and "serve)" in shell_text
    assert '"confirm_serve"' in ps_text and '"serve"' in ps_text


def test_combined_extra_install_bypasses_receipt_upgrade_when_extras_changed() -> None:
    """A newly selected extra must be applied by the combined package spec."""
    shell_body = _shell_function("install_markitai")
    changed_guard = shell_body.find("! markitai_extras_need_update")
    generic_upgrade = shell_body.find("uv tool upgrade markitai")
    combined_install = shell_body.find(
        'uv tool install "$_mi_pkg" --python "$PYTHON_CMD" --force'
    )
    assert 0 <= changed_guard < generic_upgrade < combined_install

    ps_body = _powershell_function("Install-Markitai")
    changed_guard = ps_body.find("-not (Test-MarkitaiExtrasNeedUpdate)")
    generic_upgrade = ps_body.find("uv tool upgrade markitai")
    combined_install = ps_body.find("uv tool install $pkg --python $pythonArg --force")
    assert 0 <= changed_guard < generic_upgrade < combined_install


def test_extra_accumulators_treat_all_as_a_superset() -> None:
    """`all` must remain canonical instead of becoming `all,serve` or `,all`."""
    shell_body = _shell_function("install_markitai_extra")
    assert '[ "$_extra_name" = "all" ]' in shell_body
    assert 'MARKITAI_EXTRAS="all"' in shell_body
    assert 'if [ -z "$MARKITAI_EXTRAS" ]; then' in shell_body
    assert "markitai_extra_enabled" in shell_body

    ps_body = _powershell_function("Install-MarkitaiExtra")
    assert '$ExtraName -eq "all"' in ps_body
    assert '$script:MARKITAI_EXTRAS = "all"' in ps_body
    assert "[string]::IsNullOrEmpty($script:MARKITAI_EXTRAS)" in ps_body
    assert "Test-MarkitaiExtraEnabled" in ps_body


def test_all_extra_and_its_fallback_include_serve() -> None:
    """The public `all` contract and installer fallback both include Web UI."""
    metadata = tomllib.loads(_MARKITAI_PYPROJECT.read_text(encoding="utf-8"))
    extras = metadata["project"]["optional-dependencies"]

    def package_name(requirement: str) -> str:
        return re.split(r"[<>=!~ ]", requirement, maxsplit=1)[0].lower()

    serve_packages = {package_name(requirement) for requirement in extras["serve"]}
    all_packages = {package_name(requirement) for requirement in extras["all"]}
    assert serve_packages <= all_packages

    shell_text = _SETUP_SH.read_text(encoding="utf-8")
    ps_text = _SETUP_PS1.read_text(encoding="utf-8")
    assert (
        'MARKITAI_ALL_FALLBACK_EXTRAS="browser,extra-fetch,kreuzberg,svg,heif,serve"'
        in shell_text
    )
    assert (
        '$script:MARKITAI_ALL_FALLBACK_EXTRAS = "browser,extra-fetch,kreuzberg,svg,heif,serve"'
        in ps_text
    )
