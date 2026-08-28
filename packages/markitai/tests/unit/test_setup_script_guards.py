"""Regression guards for the cross-platform setup scripts.

One root cause guarded here: setup.sh runs under `set -eu`,
and its optional-component installers return non-zero to signal a
declined/failed OPTIONAL step (skip → `return 2`, fail → `return 1`).
When such a call is unguarded, `set -e` treats the non-zero return as
fatal and exits the whole script — so declining LibreOffice would abort
setup before the summary/next-steps/outro ever printed.

The fix guards every non-fatal orchestration call with `|| true`. This
test fails if any of them regresses back to a bare call.

The module also verifies failure-output containment: noisy third-party ANSI
stderr is bounded, unexpected exits close the tree once, and PowerShell native
commands do not leak verbose red ErrorRecords.

Two later contracts live here too:

* The installer must not offer capabilities the product does not have. FFmpeg
  was offered on both platforms although no audio/video extension is
  registered, so its whole surface is asserted gone.
* The China-mirror question must be driven by evidence, not by the absence of
  a proxy variable. `probe_default_index` / `Test-DefaultIndexReachable` issue
  a real, time-boxed HTTP request; only a failing probe may raise the prompt.
"""

from __future__ import annotations

import os
import re
import shutil
import socket
import subprocess
import threading
import tomllib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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
    "install_optional_claude_cli",
    "install_optional_copilot_cli",
    "finalize_markitai_extras",
    "install_precommit",
)

_OPTIONAL_INSTALLERS = (
    ("install_optional_playwright", "Install-OptionalPlaywright"),
    ("install_optional_libreoffice", "Install-OptionalLibreOffice"),
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


def test_local_provider_runtime_extras_are_applied_before_auto_config() -> None:
    """Auto-detected CLI models must already be runnable in the tool environment."""
    metadata = tomllib.loads(_MARKITAI_PYPROJECT.read_text(encoding="utf-8"))
    extras = metadata["project"]["optional-dependencies"]
    assert any(req.startswith("claude-agent-sdk") for req in extras["claude-agent"])
    assert any(req.startswith("github-copilot-sdk") for req in extras["copilot"])

    shell_flow = _shell_function("run_user_setup")
    assert shell_flow.find("finalize_markitai_extras") < shell_flow.find("init_config")
    ps_flow = _powershell_function("Run-UserSetup")
    assert ps_flow.find("Finalize-MarkitaiExtras") < ps_flow.find("Initialize-Config")


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

    shell_completion = _shell_function("print_user_completion")
    assert 'markitai_extra_enabled "serve"' in shell_completion
    assert "markitai serve" in shell_completion
    ps_completion = _powershell_function("Print-UserCompletion")
    assert 'Test-MarkitaiExtraEnabled -ExtraName "serve"' in ps_completion
    assert "markitai serve" in ps_completion
    assert "uv run markitai serve" in _shell_function("print_dev_completion")
    assert "uv run markitai serve" in _powershell_function("Print-DevCompletion")


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


# Extras the fallback deliberately omits: both need an SDK the installer
# cannot assume, and a failed install of one would take the whole set down.
_SDK_DEPENDENT_EXTRAS = frozenset({"claude-agent", "copilot"})


def _fallback_extras(text: str) -> set[str]:
    match = re.search(r'MARKITAI_ALL_FALLBACK_EXTRAS\s*=\s*"([^"]*)"', text)
    assert match is not None, "the installer no longer defines a fallback extras set"
    return {name for name in match.group(1).split(",") if name}


def test_all_extra_and_its_fallback_include_serve() -> None:
    """The public `all` contract and installer fallback both include Web UI."""
    metadata = tomllib.loads(_MARKITAI_PYPROJECT.read_text(encoding="utf-8"))
    extras = metadata["project"]["optional-dependencies"]

    def package_name(requirement: str) -> str:
        return re.split(r"[<>=!~ ]", requirement, maxsplit=1)[0].lower()

    serve_packages = {package_name(requirement) for requirement in extras["serve"]}
    all_packages = {package_name(requirement) for requirement in extras["all"]}
    assert serve_packages <= all_packages
    assert "serve" in _fallback_extras(_SETUP_SH.read_text(encoding="utf-8"))
    assert "serve" in _fallback_extras(_SETUP_PS1.read_text(encoding="utf-8"))


def test_the_fallback_offers_everything_all_would_have() -> None:
    """The fallback is `all` minus the two SDK-dependent extras.

    It used to be checked by asserting the literal string, so when `legacy`
    and `mcp` were added to `all` the fallback silently stopped offering
    them: a user whose `markitai[all]` install failed lost legacy Office
    conversion and the MCP server without being told. Computing the
    expected set from pyproject means the next extra cannot slip through.
    """
    metadata = tomllib.loads(_MARKITAI_PYPROJECT.read_text(encoding="utf-8"))
    declared = set(metadata["project"]["optional-dependencies"]) - {"all"}
    expected = declared - _SDK_DEPENDENT_EXTRAS

    for path in (_SETUP_SH, _SETUP_PS1):
        actual = _fallback_extras(path.read_text(encoding="utf-8"))
        assert actual == expected, (
            f"{path.name} fallback extras drifted from pyproject: "
            f"missing {sorted(expected - actual)}, "
            f"unknown {sorted(actual - expected)}"
        )


def test_setup_sh_bounds_and_sanitizes_failed_command_output(tmp_path: Path) -> None:
    """A noisy ANSI-colored failure must stay inside and close the tree."""
    text = _SETUP_SH.read_text(encoding="utf-8")
    visual_start = text.index("# Color Definitions")
    visual_end = text.index("# Installation Status Tracking")
    visual_runtime = text[visual_start:visual_end]

    script = tmp_path / "failure-guard.sh"
    script.write_text(
        "#!/bin/sh\n"
        "set -eu\n"
        "i18n() {\n"
        '  case "$1" in\n'
        '    error_unexpected) echo "Unexpected error" ;;\n'
        '    info_error_log) echo "Full error log" ;;\n'
        '    *) echo "Setup failed" ;;\n'
        "  esac\n"
        "}\n"
        f"{visual_runtime}\n"
        "trap 'setup_on_exit' 0\n"
        "noisy_failure() {\n"
        "  i=1\n"
        '  while [ "$i" -le 20 ]; do\n'
        '    printf "\\033[31mdiagnostic line %s\\033[0m\\n" "$i" >&2\n'
        "    i=$((i+1))\n"
        "  done\n"
        "  return 7\n"
        "}\n"
        'clack_intro "Failure test"\n'
        'clack_spinner "fragile step..." noisy_failure\n'
        'clack_outro "must not print"\n',
        encoding="utf-8",
    )

    full_log = tmp_path / "full-setup.log"
    result = subprocess.run(
        ["sh", str(script)],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "MARKITAI_SETUP_LOG": str(full_log)},
    )
    clean = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", result.stdout)
    diagnostic_lines = [
        line for line in clean.splitlines() if "diagnostic line" in line
    ]

    assert result.returncode == 7
    assert result.stderr == ""
    assert len(diagnostic_lines) == 6
    assert diagnostic_lines[0].endswith("diagnostic line 15")
    assert diagnostic_lines[-1].endswith("diagnostic line 20")
    assert clean.count("Unexpected error") == 1
    assert clean.count("Setup failed") == 1
    assert str(full_log) in clean
    assert "must not print" not in clean

    persisted = full_log.read_text(encoding="utf-8")
    assert persisted.count("diagnostic line") == 20
    assert "diagnostic line 1" in persisted
    assert "diagnostic line 20" in persisted
    assert "\x1b[31m" not in persisted


def test_setup_sh_expected_and_unexpected_failures_close_once() -> None:
    """The EXIT guard must not duplicate an already-rendered cancellation."""
    text = _SETUP_SH.read_text(encoding="utf-8")
    assert "trap 'setup_on_exit' 0" in text
    assert "CLACK_PENDING_DETAIL=" in _shell_function("clack_spinner")
    assert "clack_flush_detail" in _shell_function("clack_error")
    assert "CLACK_SESSION_CLOSED=true" in _shell_function("clack_cancel")
    assert "CLACK_SESSION_CLOSED=true" in _shell_function("clack_outro")
    assert "CLACK_SESSION_CLOSED=true" in _shell_function("clack_outro_warn")
    for flow in ("run_user_setup", "run_dev_setup"):
        assert "clack_outro_warn" in _shell_function(flow)


def test_setup_ps1_has_bounded_native_and_global_error_guards() -> None:
    """PowerShell native stderr and unexpected exceptions must not leak raw UI."""
    text = _SETUP_PS1.read_text(encoding="utf-8")
    assert re.search(
        r"try \{\s*Main\s*\} catch \{\s*Complete-UnexpectedFailure",
        text,
        re.DOTALL,
    )
    assert "} finally {" in text

    unexpected = _powershell_function("Complete-UnexpectedFailure")
    assert "Clack-Detail" in unexpected
    assert "Clack-Cancel" in unexpected
    assert "Write-Error" not in unexpected

    detail = _powershell_function("Clack-Detail")
    assert "Write-SetupDiagnostic" in detail
    assert "Show-SetupErrorLog" in _powershell_function("Clack-Cancel")

    native_guard = _powershell_function("Invoke-NativeQuietly")
    assert '$ErrorActionPreference = "Continue"' in native_guard
    assert "$output = @(& $Command 2>&1)" in native_guard

    installer_guard = _powershell_function("Invoke-PowerShellInstallerQuietly")
    assert "-File $tempPath" in installer_guard
    assert "Invoke-NativeQuietly" in installer_guard
    assert "Invoke-Expression" not in text

    for function in (
        "Install-OptionalLibreOffice",
        "Install-OptionalClaudeCLI",
        "Install-OptionalCopilotCLI",
    ):
        assert "Invoke-NativeQuietly" in _powershell_function(function)

    for flow in ("Run-UserSetup", "Run-DevSetup"):
        body = _powershell_function(flow)
        assert "Invoke-OptionalStep" in body
        assert "Clack-OutroWarning" in body

    assert "extras update $(i18n 'failed'): $uvErr" not in text


# ============================================================
# Dead capability: FFmpeg
# ============================================================


@pytest.mark.parametrize("script", (_SETUP_SH, _SETUP_PS1))
def test_setup_scripts_no_longer_mention_ffmpeg(script: Path) -> None:
    """No audio/video extension is registered, so nothing may install FFmpeg.

    converter/base.py's EXTENSION_MAP has no audio or video entry: an FFmpeg
    install can never change what the tool converts. Offering it advertised a
    capability that does not exist and slowed every guided install down.
    """
    lines = [
        f"{number}: {line}"
        for number, line in enumerate(
            script.read_text(encoding="utf-8").splitlines(), start=1
        )
        if "ffmpeg" in line.lower()
    ]
    assert not lines, f"{script.name} still references FFmpeg:\n" + "\n".join(lines)


def test_setup_scripts_dropped_the_ffmpeg_i18n_strings() -> None:
    """Both languages of every FFmpeg string must be gone, not just English."""
    for script in (_SETUP_SH, _SETUP_PS1):
        text = script.read_text(encoding="utf-8")
        for stale in (
            "confirm_ffmpeg",
            "info_ffmpeg_purpose",
            "audio/video processing",
            "音视频处理",
            "处理音频和视频文件",
        ):
            assert stale not in text, f"{script.name} still contains {stale!r}"


# ============================================================
# OCR became an opt-in extra
# ============================================================


def test_setup_scripts_offer_the_ocr_extra() -> None:
    """OCR left the core install, so the installer has to let users pick it.

    `pip install markitai` no longer ships rapidocr. Without a selection step
    the guided installer would silently decide for the user in either
    direction.
    """
    shell_body = _shell_function("select_markitai_ocr")
    assert "clack_confirm_optional" in shell_body
    assert 'install_markitai_extra "ocr"' in shell_body
    assert 'markitai_extra_enabled "ocr"' in shell_body

    ps_body = _powershell_function("Select-MarkitaiOcr")
    assert "Confirm-OptionalInstall" in ps_body
    assert 'Install-MarkitaiExtra -ExtraName "ocr"' in ps_body
    assert 'Test-MarkitaiExtraEnabled -ExtraName "ocr"' in ps_body


def test_ocr_selection_joins_the_first_combined_install() -> None:
    """Selecting OCR must widen the initial spec, not trigger a reinstall."""
    shell_flow = _shell_function("run_user_setup")
    load = shell_flow.find("load_existing_markitai_extras")
    select = shell_flow.find("select_markitai_ocr")
    install = shell_flow.find("install_markitai ||")
    track = shell_flow.find("track_markitai_ocr")
    assert 0 <= load < select < install < track

    ps_flow = _powershell_function("Run-UserSetup")
    ps_load = ps_flow.find("Import-MarkitaiReceiptExtras")
    ps_select = ps_flow.find("Select-MarkitaiOcr")
    ps_install = ps_flow.find("Install-Markitai))")
    ps_track = ps_flow.find("Track-MarkitaiOcr")
    assert 0 <= ps_load < ps_select < ps_install < ps_track


def test_declined_ocr_is_not_re_added_by_suggest_extras() -> None:
    """A declined extra must survive the doctor --suggest-extras merge.

    `markitai doctor --suggest-extras` always lists `ocr`. Merging it blindly
    would reinstall what the user just declined, making the prompt a lie.
    """
    shell_finalize = _shell_function("finalize_markitai_extras")
    assert "markitai_extra_declined" in shell_finalize
    shell_select = _shell_function("select_markitai_ocr")
    assert "decline_markitai_extra" in shell_select

    ps_finalize = _powershell_function("Finalize-MarkitaiExtras")
    assert "Test-MarkitaiExtraDeclined" in ps_finalize
    ps_select = _powershell_function("Select-MarkitaiOcr")
    assert "Deny-MarkitaiExtra" in ps_select


def test_ocr_bilingual_strings_exist_in_both_scripts() -> None:
    """Every new prompt needs an English and a Chinese spelling."""
    for script in (_SETUP_SH, _SETUP_PS1):
        text = script.read_text(encoding="utf-8")
        assert text.count("confirm_ocr") >= 3, f"{script.name}: zh + en + call site"
        assert "扫描件" in text, f"{script.name}: missing Chinese OCR copy"
        assert "scanned" in text, f"{script.name}: missing English OCR copy"


# ============================================================
# China mirrors: evidence, not geography
# ============================================================


def test_mirror_prompt_no_longer_warns_every_proxyless_user() -> None:
    """The old "no proxy detected" warning fired for the whole planet."""
    for script in (_SETUP_SH, _SETUP_PS1):
        text = script.read_text(encoding="utf-8")
        for stale in (
            "mirror_no_proxy",
            "No proxy detected",
            "未检测到代理",
        ):
            assert stale not in text, f"{script.name} still contains {stale!r}"


def test_mirrors_are_offered_only_after_a_failed_index_probe() -> None:
    """Order matters: measure first, ask second, and never ask without a TTY."""
    shell = _shell_function("configure_mirrors")
    assert "MARKITAI_USE_MIRROR" in shell
    assert "has_interactive_tty" in shell
    probe = shell.find("probe_default_index")
    prompt = shell.find("clack_confirm ")
    assert 0 <= probe < prompt, "the probe must gate the prompt"

    ps = _powershell_function("Configure-Mirrors")
    assert "MARKITAI_USE_MIRROR" in ps
    assert "Test-InteractiveInput" in ps
    ps_probe = ps.find("Test-DefaultIndexReachable")
    ps_prompt = ps.find("Clack-Confirm ")
    assert 0 <= ps_probe < ps_prompt, "the probe must gate the prompt"


def test_index_probe_is_application_layer_not_a_port_check() -> None:
    """A TUN-mode proxy answers every TCP port, so ports prove nothing.

    The probe has to perform a real HTTP request and validate the response —
    status *and* a non-empty body, because an intermittently empty body still
    returns 200.
    """
    shell_probe = _shell_function("probe_default_index")
    assert "http_code" in shell_probe
    assert "size_download" in shell_probe
    for banned in ("nc -z", "/dev/tcp", "telnet", "--head", "-I "):
        assert banned not in shell_probe, f"port/HEAD probe leaked in: {banned!r}"

    ps_probe = _powershell_function("Test-DefaultIndexReachable")
    assert "Invoke-WebRequest" in ps_probe
    assert "StatusCode" in ps_probe
    for banned in ("Test-NetConnection", "TcpClient", "Test-Connection"):
        assert banned not in ps_probe, f"port probe leaked in: {banned!r}"


class _ProbeHandler(BaseHTTPRequestHandler):
    """Serve the three responses the probe has to tell apart."""

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path.startswith("/ok"):
            body = b"<a href='markitai-0.23.0.whl'>markitai</a>"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith("/empty"):
            # 200 with nothing in it: the exact failure a status-only check
            # would report as healthy.
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()
        elif self.path.startswith("/slow"):
            self.server.slow_gate.wait(timeout=10)  # type: ignore[attr-defined]
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"ok")
        else:
            self.send_error(503)

    def log_message(self, *args: object) -> None:
        """Keep the test output clean."""


@pytest.fixture
def probe_server():
    """A loopback HTTP server plus a gate that releases the /slow request."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ProbeHandler)
    server.slow_gate = threading.Event()  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.slow_gate.set()  # type: ignore[attr-defined]
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _run_shell_probe(tmp_path: Path, url: str, timeout: str = "3") -> int:
    """Run setup.sh's probe_default_index() standalone against ``url``."""
    body = _shell_function("probe_default_index")
    script = tmp_path / "probe.sh"
    script.write_text(
        "#!/bin/sh\nset -eu\n"
        f"probe_default_index() {{\n{body}}}\n"
        "if probe_default_index; then exit 0; else exit 1; fi\n",
        encoding="utf-8",
    )
    env = {
        key: value
        for key, value in os.environ.items()
        if key.lower() not in {"http_proxy", "https_proxy", "all_proxy"}
    }
    env["no_proxy"] = "127.0.0.1,localhost"
    env["NO_PROXY"] = "127.0.0.1,localhost"
    env["MARKITAI_INDEX_PROBE_URL"] = url
    env["MARKITAI_INDEX_PROBE_TIMEOUT"] = timeout
    return subprocess.run(
        ["sh", str(script)], check=False, capture_output=True, text=True, env=env
    ).returncode


@pytest.mark.skipif(shutil.which("curl") is None, reason="probe needs curl")
def test_shell_probe_accepts_a_healthy_index(tmp_path: Path, probe_server: str) -> None:
    """A 200 with a body means the default index is usable: stay silent."""
    assert _run_shell_probe(tmp_path, f"{probe_server}/ok") == 0


@pytest.mark.skipif(shutil.which("curl") is None, reason="probe needs curl")
def test_shell_probe_rejects_an_empty_200(tmp_path: Path, probe_server: str) -> None:
    """HTTP 200 is not success: an empty body must count as unreachable."""
    assert _run_shell_probe(tmp_path, f"{probe_server}/empty") == 1


@pytest.mark.skipif(shutil.which("curl") is None, reason="probe needs curl")
def test_shell_probe_rejects_an_error_status(tmp_path: Path, probe_server: str) -> None:
    """A 5xx index is unreachable for installation purposes."""
    assert _run_shell_probe(tmp_path, f"{probe_server}/boom") == 1


@pytest.mark.skipif(shutil.which("curl") is None, reason="probe needs curl")
def test_shell_probe_times_out_on_a_slow_index(
    tmp_path: Path, probe_server: str
) -> None:
    """A hanging index must not hang setup: slowness is the whole point."""
    assert _run_shell_probe(tmp_path, f"{probe_server}/slow", timeout="1") == 1


@pytest.mark.skipif(shutil.which("curl") is None, reason="probe needs curl")
def test_shell_probe_rejects_a_dead_port(tmp_path: Path) -> None:
    """Nothing listening at all is the plainest unreachable case."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        dead_port = sock.getsockname()[1]
    assert _run_shell_probe(tmp_path, f"http://127.0.0.1:{dead_port}/ok") == 1


_POWERSHELL = shutil.which("pwsh") or shutil.which("powershell")


def _run_powershell_probe(tmp_path: Path, url: str, timeout: str = "3") -> int:
    """Run setup.ps1's Test-DefaultIndexReachable standalone against ``url``."""
    assert _POWERSHELL is not None
    body = _powershell_function("Test-DefaultIndexReachable")
    script = tmp_path / "probe.ps1"
    script.write_text(
        f"function Test-DefaultIndexReachable {{\n{body}}}\n"
        "if (Test-DefaultIndexReachable) { exit 0 } else { exit 1 }\n",
        encoding="utf-8",
    )
    env = {
        key: value
        for key, value in os.environ.items()
        if key.lower() not in {"http_proxy", "https_proxy", "all_proxy"}
    }
    env["no_proxy"] = "127.0.0.1,localhost"
    env["NO_PROXY"] = "127.0.0.1,localhost"
    env["MARKITAI_INDEX_PROBE_URL"] = url
    env["MARKITAI_INDEX_PROBE_TIMEOUT"] = timeout
    return subprocess.run(
        [_POWERSHELL, "-NoLogo", "-NoProfile", "-File", str(script)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    ).returncode


@pytest.mark.skipif(_POWERSHELL is None, reason="no PowerShell available")
@pytest.mark.parametrize(
    "path,expected", (("ok", 0), ("empty", 1), ("boom", 1), ("slow", 1))
)
def test_powershell_probe_matches_the_shell_probe(
    tmp_path: Path, probe_server: str, path: str, expected: int
) -> None:
    """Both installers must agree on what "reachable" means."""
    timeout = "1" if path == "slow" else "3"
    actual = _run_powershell_probe(tmp_path, f"{probe_server}/{path}", timeout)
    assert actual == expected
