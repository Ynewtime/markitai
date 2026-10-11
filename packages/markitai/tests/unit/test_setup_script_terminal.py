"""Setup scripts: encoding, terminal style, setup mode and elevation warnings.

Four contracts are guarded here.

* ``setup.ps1`` stays ASCII. Windows PowerShell 5.1 reads a script without a
  BOM in the ANSI code page and ``irm`` decodes the download as ISO-8859-1, so
  the literal Chinese strings it used to carry became mojibake: 48 parse errors
  under ``powershell -File`` (Windows-1252), garbled prompts under
  ``irm | iex``. The Chinese table is now ``\\uXXXX`` escapes decoded at run
  time, and must match setup.sh's strings.
* Both scripts follow one terminal style: colour means one thing (gray guide,
  bold or default structure, cyan accent, green success, yellow warning, red
  error), marks carry the meaning without colour, a guide row before each
  section and none inside it, plain one-line-per-message output off a
  terminal, ASCII marks where Unicode cannot be shown, hanging-indent wrapping
  on a terminal.
* A checkout root no longer switches to a development setup silently: it asks
  (default: install markitai), and ``MARKITAI_SETUP_MODE`` chooses without a
  terminal.
* Running as root or administrator is explained (markitai installs per user)
  and does not block a run that has no terminal to ask.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[4]
_SETUP_SH = _REPO / "scripts" / "setup.sh"
_SETUP_PS1 = _REPO / "scripts" / "setup.ps1"
_CI_WORKFLOW = _REPO / ".github" / "workflows" / "ci.yml"
_PWSH = shutil.which("pwsh") or shutil.which("powershell")
_WINDOWS_POWERSHELL = shutil.which("powershell") if sys.platform == "win32" else None

_needs_pwsh = pytest.mark.skipif(_PWSH is None, reason="no PowerShell available")
# setup.sh runs under a POSIX sh: on Windows only with Git for Windows' bin
# folder (or another sh) on PATH. Without one its tests are skipped, never
# failed; the PowerShell tests run either way.
_SH = shutil.which("sh")
_NO_SH = "no POSIX sh on PATH (on Windows, Git for Windows' bin folder)"
_needs_pty = pytest.mark.skipif(
    sys.platform == "win32", reason="pseudo-terminals are POSIX only"
)

# The scheme's SGR parameters: reset, bold, red, green, yellow, cyan, gray.
_ALLOWED_SGR = {"0", "1", "31", "32", "33", "36", "90"}
_SGR = re.compile(r"\x1b\[([0-9;]*)m")
_ESCAPE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_SECTION_END = "# ============================================================\n"


# ============================================================
# Helpers
# ============================================================


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _decode_escapes(text: str) -> str:
    return re.sub(r"\\u([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), text)


def _shell_function(name: str) -> str:
    """Return one top-level POSIX shell function, definition included."""
    match = re.search(
        rf"^{re.escape(name)}\(\) \{{\n.*?^\}}\n", _text(_SETUP_SH), re.M | re.S
    )
    assert match is not None, f"{name}() not found"
    return match.group(0)


def _powershell_function(name: str) -> str:
    """Return one top-level PowerShell function, definition included."""
    match = re.search(
        rf"^function {re.escape(name)} \{{\n.*?^\}}\n", _text(_SETUP_PS1), re.M | re.S
    )
    assert match is not None, f"{name} not found"
    return match.group(0)


def _sh_runtime() -> str:
    """setup.sh's real i18n and output layer, without its entry point."""
    text = _text(_SETUP_SH)
    start = text.index("detect_lang() {")
    end = text.index(_SECTION_END + "# Installation Status Tracking")
    return text[start:end]


def _ps_runtime() -> str:
    """setup.ps1 up to the installers: capabilities, i18n and output layer."""
    text = _text(_SETUP_PS1)
    return text[: text.index(_SECTION_END + "# Installation Status Tracking")]


def _env(**overrides: str) -> dict[str, str]:
    """A clean environment: no colour or locale settings leak in from CI."""
    drop = {
        "NO_COLOR",
        "FORCE_COLOR",
        "COLUMNS",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "LC_MESSAGES",
        "TERM",
        "CI",
        "WT_SESSION",
        "TERM_PROGRAM",
        "MARKITAI_SETUP_MODE",
        "MARKITAI_SETUP_SOURCE",
        "MARKITAI_VERSION",
        "MARKITAI_INSTALL_OPTIONAL",
        "UV_COMPILE_BYTECODE",
    }
    env = {key: value for key, value in os.environ.items() if key not in drop}
    env.update({"LANG": "en_US.UTF-8", "TERM": "xterm-256color"})
    env.update(overrides)
    return env


def _run_sh(tmp_path: Path, body: str, **env: str) -> subprocess.CompletedProcess[str]:
    script = tmp_path / "harness.sh"
    script.write_text(
        f"#!/bin/sh\nset -eu\n{_sh_runtime()}\n{body}\n", encoding="utf-8"
    )
    if _SH is None:
        pytest.skip(_NO_SH)
    return subprocess.run(
        [_SH, str(script)],
        check=False,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        env=_env(**env),
        timeout=60,
    )


def _run_ps(
    tmp_path: Path, body: str, *, runtime: bool = True, **env: str
) -> subprocess.CompletedProcess[str]:
    assert _PWSH is not None
    script = tmp_path / "harness.ps1"
    prefix = _ps_runtime() if runtime else ""
    script.write_text(f"{prefix}\n{body}\n", encoding="utf-8")
    return subprocess.run(
        [
            _PWSH,
            "-NoLogo",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script),
        ],
        check=False,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        env=_env(**env),
        timeout=120,
    )


def _display_width(text: str) -> int:
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def _i18n_table(script: Path, lang: str) -> dict[str, str]:
    """Parse one language of a script's i18n table into key -> text."""
    text = _text(script)
    if script.suffix == ".sh":
        body = text[text.index("i18n() {") :]
        zh_end = body.index("    else\n        # English (default)")
        part = body[:zh_end] if lang == "zh" else body[zh_end : body.index("\n}\n")]
        pairs = re.findall(r'^\s+([a-z_]+)\)\s+echo "(.*)" ;;$', part, re.M)
        return dict(pairs)
    body = text[text.index("function i18n {") :]
    en_start = body.index("    # English (default)")
    part = body[:en_start] if lang == "zh" else body[en_start : body.index("\n}\n")]
    pairs = re.findall(r"^\s+\"([a-z_]+)\"\s+\{ (?:return )?'(.*)' \}$", part, re.M)
    return {key: _decode_escapes(value.replace("''", "'")) for key, value in pairs}


def _run_in_pty(
    argv: list[str], env: dict[str, str], answer: bytes, prompt: bytes = b"[y/N]"
) -> str:
    """Run argv with a pseudo-terminal as stdio, answer the first ``prompt``."""
    import pty
    import select

    master, slave = pty.openpty()
    process = subprocess.Popen(
        argv, stdin=slave, stdout=slave, stderr=slave, env=env, start_new_session=True
    )
    os.close(slave)
    output = b""
    answered = False
    deadline = time.monotonic() + 30
    try:
        while time.monotonic() < deadline:
            ready, _, _ = select.select([master], [], [], 0.1)
            if not ready:
                if process.poll() is not None:
                    break
                continue
            try:
                chunk = os.read(master, 4096)
            except OSError:
                break
            if not chunk:
                break
            output += chunk
            if not answered and prompt in output:
                os.write(master, answer + b"\r")
                answered = True
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=10)
        os.close(master)
    return output.decode("utf-8", "replace").replace("\r\n", "\n")


# ============================================================
# Encoding: setup.ps1 must survive Windows PowerShell 5.1
# ============================================================


def _powershell_sources() -> list[Path]:
    skip = {".git", ".venv", "node_modules", "dist", "build"}
    return sorted(
        path
        for pattern in ("*.ps1", "*.psm1", "*.psd1")
        for path in _REPO.rglob(pattern)
        if not skip.intersection(path.relative_to(_REPO).parts)
    )


def test_every_powershell_script_is_ascii_without_bom() -> None:
    """Any non-ASCII byte turns into mojibake in Windows PowerShell 5.1.

    It reads a BOM-less script in the ANSI code page (Windows-1252, GBK, ...),
    and `irm` decodes a download served without a charset as ISO-8859-1.
    UTF-8 punctuation became smart quotes there, which end strings early.
    """
    sources = _powershell_sources()
    assert _SETUP_PS1 in sources
    problems = []
    for path in sources:
        data = path.read_bytes()
        if data.startswith(b"\xef\xbb\xbf"):
            problems.append(f"{path.name}: starts with a UTF-8 BOM")
        for number, line in enumerate(data.split(b"\n"), start=1):
            if not line.isascii():
                problems.append(
                    f"{path.name}:{number}: {line.decode('utf-8', 'replace')}"
                )
    assert not problems, "PowerShell sources must be ASCII:\n" + "\n".join(problems)


@_needs_pwsh
@pytest.mark.parametrize("codec", ["cp1252", "gbk", "cp437"])
def test_setup_ps1_parses_when_read_in_a_legacy_code_page(
    tmp_path: Path, codec: str
) -> None:
    """Emulate `powershell -File` on 5.1: decode in the ANSI/OEM code page, parse."""
    decoded = _SETUP_PS1.read_bytes().decode(codec, errors="replace")
    copy = tmp_path / "decoded.ps1"
    copy.write_text(decoded, encoding="utf-8-sig")
    result = _run_ps(
        tmp_path,
        "$tokens = $null; $errors = $null\n"
        f"$text = [IO.File]::ReadAllText('{copy}')\n"
        "$null = [System.Management.Automation.Language.Parser]::ParseInput("
        "$text, [ref]$tokens, [ref]$errors)\n"
        "$errors | ForEach-Object { $_.Message }\n"
        '"errors=$($errors.Count)"',
        runtime=False,
    )
    assert result.stdout.strip().endswith("errors=0"), result.stdout + result.stderr


@pytest.mark.parametrize(
    "engine",
    [
        pytest.param(_PWSH, id="pwsh", marks=_needs_pwsh),
        pytest.param(
            _WINDOWS_POWERSHELL,
            id="windows-powershell-5.1",
            marks=pytest.mark.skipif(
                _WINDOWS_POWERSHELL is None, reason="Windows PowerShell only on Windows"
            ),
        ),
    ],
)
def test_powershell_loads_setup_ps1_like_file_mode(engine: str | None) -> None:
    """`Get-Command <file>` reads and parses the script exactly as -File does."""
    assert engine is not None
    result = subprocess.run(
        [
            engine,
            "-NoLogo",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            f"$info = Get-Command -Name '{_SETUP_PS1}'; "
            "$null = $info.ScriptBlock; 'parsed'",
        ],
        check=False,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "parsed" in result.stdout


@pytest.mark.parametrize("lang", ["zh", "en"])
def test_setup_ps1_strings_match_setup_sh(lang: str) -> None:
    """Both installers say the same thing; the escaped zh table decodes to it."""
    shell = _i18n_table(_SETUP_SH, lang)
    powershell = _i18n_table(_SETUP_PS1, lang)
    shared = set(shell) & set(powershell)
    assert len(shared) > 80
    mismatched = {
        key: (shell[key], powershell[key])
        for key in sorted(shared)
        if shell[key] != powershell[key]
    }
    assert not mismatched, mismatched
    # Every key either script uses must exist in both of its languages.
    assert set(_i18n_table(_SETUP_PS1, "zh")) == set(_i18n_table(_SETUP_PS1, "en"))
    assert set(_i18n_table(_SETUP_SH, "zh")) == set(_i18n_table(_SETUP_SH, "en"))


@_needs_pwsh
def test_setup_ps1_decodes_chinese_at_run_time(tmp_path: Path) -> None:
    """The escapes become the real characters in PowerShell itself."""
    expected = _i18n_table(_SETUP_SH, "zh")
    keys = ("welcome", "confirm_ocr", "info_chatgpt_route", "mkai_conflict")
    body = '$script:LANG_CODE = "zh"\n' + "".join(f"i18n '{key}'\n" for key in keys)
    result = _run_ps(tmp_path, body)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[-len(keys) :] == [expected[key] for key in keys]


@_needs_pwsh
def test_setup_ps1_picks_chinese_from_the_culture(tmp_path: Path) -> None:
    result = _run_ps(
        tmp_path,
        '"lang=$($script:LANG_CODE)"\nClack-Intro (i18n "welcome")',
        LANG="zh_CN.UTF-8",
        LC_ALL="zh_CN.UTF-8",
    )
    if "lang=zh" not in result.stdout:
        pytest.skip("this PowerShell does not report a zh culture from the locale")
    assert "Markitai \u5b89\u88c5" in result.stdout


# ============================================================
# Terminal style
# ============================================================

_SH_DEMO = """
clack_intro "Markitai Setup"
clack_info "Mode: install markitai"
clack_section "Core components"
clack_success "uv 0.9.8"
clack_warn "a warning"
clack_error "an error"
clack_skip "Playwright browser skipped"
clack_step "Installing markitai..."
clack_hint "Convert a file" "markitai file.pdf"
clack_confirm "Install OCR support?" "n" || true
clack_section "Summary"
clack_success "markitai"
clack_outro "Markitai is ready"
"""

_EXPECTED_UNICODE = [
    "",
    "\u256d\u2500 Markitai Setup",
    "\u2502  \u2022 Mode: install markitai",
    "\u2502",
    "\u251c\u2500 Core components",
    "\u2502  \u2713 uv 0.9.8",
    "\u2502  ! a warning",
    "\u2502  \u2717 an error",
    "\u2502  \u25cb Playwright browser skipped",
    "\u2502  \u2192 Installing markitai...",
    "\u2502  \u2022 Convert a file:",
    "\u2502    markitai file.pdf",
    "\u2502  ? Install OCR support? [y/N] n",
    "\u2502",
    "\u251c\u2500 Summary",
    "\u2502  \u2713 markitai",
    "\u2502",
    "\u2570\u2500 Markitai is ready",
    "",
    "",
]

_EXPECTED_ASCII = [
    "",
    "+- Markitai Setup",
    "|  - Mode: install markitai",
    "|",
    "+- Core components",
    "|  + uv 0.9.8",
    "|  ! a warning",
    "|  x an error",
    "|  o Playwright browser skipped",
    "|  > Installing markitai...",
    "|  - Convert a file:",
    "|    markitai file.pdf",
    "|  ? Install OCR support? [y/N] n",
    "|",
    "+- Summary",
    "|  + markitai",
    "|",
    "+- Markitai is ready",
    "",
    "",
]


def test_setup_sh_piped_output_is_plain_and_follows_the_layout(tmp_path: Path) -> None:
    """Off a terminal: no escapes, one line per item, the documented rhythm.

    One guide row before each section header, none between a header and its
    first item, commands two columns further in, one blank line at the end.
    """
    result = _run_sh(tmp_path, _SH_DEMO)
    assert result.returncode == 0, result.stderr
    assert "\x1b" not in result.stdout
    assert result.stdout.split("\n") == _EXPECTED_UNICODE


@pytest.mark.parametrize(
    "env",
    [{"LANG": "C"}, {"LANG": "en_US.ISO-8859-1"}, {"TERM": "dumb"}],
    ids=["LANG=C", "latin1-locale", "TERM=dumb"],
)
def test_setup_sh_falls_back_to_ascii_marks(
    tmp_path: Path, env: dict[str, str]
) -> None:
    result = _run_sh(tmp_path, _SH_DEMO, **env)
    assert result.returncode == 0, result.stderr
    assert result.stdout.split("\n") == _EXPECTED_ASCII


def test_setup_sh_colours_mean_one_thing(tmp_path: Path) -> None:
    """With colour forced: only the scheme's codes, each mark in its own role."""
    result = _run_sh(tmp_path, _SH_DEMO, FORCE_COLOR="1")
    out = result.stdout
    assert result.returncode == 0, result.stderr
    used = {code for params in _SGR.findall(out) for code in params.split(";")}
    assert used <= _ALLOWED_SGR, f"colours outside the scheme: {used - _ALLOWED_SGR}"
    gray, bold, reset = "\x1b[90m", "\x1b[1m", "\x1b[0m"
    # Structure is never coloured: gray branch, bold title.
    assert f"{gray}\u251c\u2500{reset} {bold}Core components{reset}" in out
    assert f"\x1b[32m\u2713{reset} uv 0.9.8" in out
    assert f"\x1b[33m!{reset} a warning" in out
    assert f"\x1b[31m\u2717{reset} an error" in out
    assert f"{gray}\u25cb{reset} {gray}Playwright browser skipped" in out
    assert f"\x1b[36m?{reset} Install OCR support?" in out
    assert "\x1b[36mmarkitai file.pdf" in out
    assert f"\x1b[32mMarkitai is ready{reset}" in out


@pytest.mark.parametrize(
    "env", [{"NO_COLOR": "1", "FORCE_COLOR": "1"}, {"TERM": "dumb", "FORCE_COLOR": "1"}]
)
def test_setup_sh_no_color_wins(tmp_path: Path, env: dict[str, str]) -> None:
    result = _run_sh(tmp_path, _SH_DEMO, **env)
    assert result.returncode == 0, result.stderr
    assert "\x1b" not in result.stdout


def test_setup_sh_source_has_no_colour_outside_the_scheme() -> None:
    """No magenta, no bold-as-bright yellow, no bright or dim codes."""
    codes = set(re.findall(r"\\033\[([0-9;]*)m", _text(_SETUP_SH)))
    params = {code for group in codes for code in group.split(";")}
    assert params <= _ALLOWED_SGR, params - _ALLOWED_SGR
    assert "MAGENTA" not in _text(_SETUP_SH)


def test_setup_ps1_colours_mean_one_thing() -> None:
    """One Write-Host -ForegroundColor (in Write-Styled), colour per role.

    The roles use the console slots that match ANSI 90/36/32/33/31; the
    bright slots appear only for the classic blue PowerShell window, whose
    shortcut repaints DarkYellow near-white.
    """
    text = _text(_SETUP_PS1)
    assert text.count("-ForegroundColor") == 1
    assert "-ForegroundColor" in _powershell_function("Write-Styled")
    roles = re.findall(r'^\$script:(C_[A-Z]+) = "(\w+)"$', text, re.M)
    assert roles[:6] == [
        ("C_GUIDE", "DarkGray"),
        ("C_DIM", "DarkGray"),
        ("C_ACCENT", "DarkCyan"),
        ("C_OK", "DarkGreen"),
        ("C_WARN", "DarkYellow"),
        ("C_ERR", "DarkRed"),
    ]
    classic = text[text.index("if (Test-ClassicPowerShellPalette) {") :]
    classic = classic[: classic.index("\n}\n")]
    bright = re.findall(r'^\s+\$script:(C_[A-Z]+) = "(\w+)"$', classic, re.M)
    assert bright == [
        ("C_ACCENT", "Cyan"),
        ("C_OK", "Green"),
        ("C_WARN", "Yellow"),
        ("C_ERR", "Red"),
    ]
    assert not re.search(r"\bMagenta\b", text.replace("DarkMagenta", ""))


_PS_DEMO = """
Clack-Intro "Markitai Setup"
Clack-Info "Mode: install markitai"
Clack-Section "Core components"
Clack-Success "uv 0.9.8"
Clack-Warn "a warning"
Clack-Error "an error"
Clack-Skip "Playwright browser skipped"
Clack-Step "Installing markitai..."
Clack-Hint "Convert a file" "markitai file.pdf"
$null = Clack-Confirm "Install OCR support?" "n"
Clack-Section "Summary"
Clack-Success "markitai"
Clack-Outro "Markitai is ready"
"""


@_needs_pwsh
def test_setup_ps1_redirected_output_matches_setup_sh(tmp_path: Path) -> None:
    """Same rows, marks and rhythm as setup.sh, plain when redirected."""
    result = _run_ps(tmp_path, _PS_DEMO)
    assert result.returncode == 0, result.stderr
    assert "\x1b" not in result.stdout
    assert result.stdout.replace("\r\n", "\n").split("\n") == _EXPECTED_UNICODE


@_needs_pwsh
def test_setup_ps1_falls_back_to_ascii_marks(tmp_path: Path) -> None:
    result = _run_ps(tmp_path, _PS_DEMO, TERM="dumb")
    assert result.returncode == 0, result.stderr
    assert result.stdout.replace("\r\n", "\n").split("\n") == _EXPECTED_ASCII


_LONG_EN = (
    "markitai installs per user and never needs root. As root, it can end up "
    "in root's home or leave files your own account cannot update."
)
_LONG_ZH = (
    "markitai \u6309\u7528\u6237\u5b89\u88c5\uff0c\u4ece\u4e0d\u9700\u8981 root "
    "\u6743\u9650\u3002\u4ee5 root \u8fd0\u884c\u65f6\uff0c\u5b83\u53ef\u80fd"
    "\u88c5\u8fdb root \u7684\u4e3b\u76ee\u5f55\uff0c\u6216\u7559\u4e0b\u4f60"
    "\u81ea\u5df1\u7684\u8d26\u6237\u65e0\u6cd5\u66f4\u65b0\u7684\u6587\u4ef6"
)
_LONG_COMMAND = "uv tool install 'markitai[browser,serve,ocr]' --python /opt/python/3.13/bin/python3"


def _assert_wrapped(lines: list[str], text: str, lead: str, width: int) -> None:
    """Rows of one wrapped message: hanging indent, within width, nothing lost."""
    assert lines[0].startswith(lead)
    for line in lines[1:]:
        assert line.startswith("\u2502    "), line
        assert not line[5:].startswith(" ")
    for line in lines:
        assert _display_width(line) <= width, (line, _display_width(line))
    joined = " ".join(line[5:] for line in lines)
    assert joined.replace(" ", "") == text.replace(" ", "")


def _message_rows(output: str, first: str) -> list[str]:
    lines = output.replace("\r\n", "\n").split("\n")
    start = next(i for i, line in enumerate(lines) if line.startswith(first))
    rows = [lines[start]]
    for line in lines[start + 1 :]:
        if not line.startswith("\u2502    ") or line.startswith("\u2502    uv tool"):
            break
        rows.append(line)
    return rows


def test_setup_sh_wraps_on_a_terminal_with_a_hanging_indent(tmp_path: Path) -> None:
    """At 40 columns: word wrap, CJK counted double, commands never split."""
    body = (
        "SETUP_WRAP_WIDTH=40\n"
        f'clack_info "{_LONG_EN}"\n'
        f'clack_warn "{_LONG_ZH}"\n'
        f'clack_cmd "{_LONG_COMMAND}"\n'
    )
    result = _run_sh(tmp_path, body)
    assert result.returncode == 0, result.stderr
    _assert_wrapped(
        _message_rows(result.stdout, "\u2502  \u2022 markitai"),
        _LONG_EN,
        "\u2502  \u2022 ",
        40,
    )
    _assert_wrapped(
        _message_rows(result.stdout, "\u2502  ! markitai"), _LONG_ZH, "\u2502  ! ", 40
    )
    assert f"\u2502    {_LONG_COMMAND}" in result.stdout.split("\n")


@_needs_pwsh
def test_setup_ps1_wraps_on_a_console_with_a_hanging_indent(tmp_path: Path) -> None:
    """The console width minus one: the legacy host wraps a full line itself."""
    body = (
        "$script:WrapWidth = 40\n"
        f"Clack-Info '{_LONG_EN.replace(chr(39), chr(39) * 2)}'\n"
        f"Clack-Warn (ConvertFrom-CodePointEscape '{_LONG_ZH.encode('ascii', 'backslashreplace').decode()}')\n"
        f"Clack-Command '{_LONG_COMMAND.replace(chr(39), chr(39) * 2)}'\n"
    )
    result = _run_ps(tmp_path, body)
    assert result.returncode == 0, result.stderr
    out = result.stdout
    _assert_wrapped(
        _message_rows(out, "\u2502  \u2022 markitai"), _LONG_EN, "\u2502  \u2022 ", 39
    )
    _assert_wrapped(
        _message_rows(out, "\u2502  ! markitai"), _LONG_ZH, "\u2502  ! ", 39
    )
    assert f"\u2502    {_LONG_COMMAND}" in out.replace("\r\n", "\n").split("\n")


def test_piped_output_never_wraps(tmp_path: Path) -> None:
    """The 1.2.1 rule: off a terminal every message stays on one line."""
    result = _run_sh(tmp_path, f'clack_info "{_LONG_EN} {_LONG_EN}"', COLUMNS="40")
    assert result.stdout.strip("\n").count("\n") == 0


# ============================================================
# Setup mode: a checkout root asks, defaulting to install
# ============================================================


def _sh_mode_body(in_checkout: bool, has_root: bool) -> str:
    return (
        f"is_dev_mode() {{ return {0 if in_checkout else 1}; }}\n"
        f"get_project_root() {{ {'echo /repo' if has_root else 'return 1'}; }}\n"
        'SETUP_MODE="install"\nCHECKOUT_ANNOUNCED=false\n'
        f"{_shell_function('choose_setup_mode')}"
        'clack_intro "Markitai Setup"\n'
        "choose_setup_mode </dev/null\n"
        'printf "mode=%s\\n" "$SETUP_MODE"\n'
    )


def test_setup_sh_checkout_root_without_terminal_installs(tmp_path: Path) -> None:
    result = _run_sh(tmp_path, _sh_mode_body(in_checkout=True, has_root=True))
    assert result.returncode == 0, result.stderr
    assert result.stdout.rstrip().endswith("mode=install")
    assert "This folder is a markitai source checkout" in result.stdout
    assert "MARKITAI_SETUP_MODE=dev" in result.stdout


@pytest.mark.parametrize("value", ["dev", "DEV", "development"])
def test_setup_sh_mode_env_selects_dev(tmp_path: Path, value: str) -> None:
    body = _sh_mode_body(in_checkout=False, has_root=True)
    result = _run_sh(tmp_path, body, MARKITAI_SETUP_MODE=value)
    assert result.stdout.rstrip().endswith("mode=dev")
    assert "Mode: development environment" in result.stdout


def test_setup_sh_mode_env_install_skips_the_question(tmp_path: Path) -> None:
    body = _sh_mode_body(in_checkout=True, has_root=True)
    result = _run_sh(tmp_path, body, MARKITAI_SETUP_MODE="install")
    assert result.stdout.rstrip().endswith("mode=install")
    assert "source checkout" not in result.stdout


def test_setup_sh_dev_mode_needs_a_checkout(tmp_path: Path) -> None:
    body = _sh_mode_body(in_checkout=False, has_root=False)
    result = _run_sh(tmp_path, body, MARKITAI_SETUP_MODE="dev")
    assert result.returncode == 1
    assert "needs a markitai checkout" in result.stdout
    assert "Setup failed" in result.stdout


def test_setup_sh_unknown_mode_warns_and_installs(tmp_path: Path) -> None:
    body = _sh_mode_body(in_checkout=False, has_root=True)
    result = _run_sh(tmp_path, body, MARKITAI_SETUP_MODE="devel")
    assert result.stdout.rstrip().endswith("mode=install")
    assert "! Ignoring MARKITAI_SETUP_MODE" in result.stdout


@_needs_pty
@pytest.mark.parametrize(("answer", "mode"), [(b"", "install"), (b"y", "dev")])
def test_setup_sh_checkout_root_asks_on_a_terminal(
    tmp_path: Path, answer: bytes, mode: str
) -> None:
    """Enter keeps the install; only an explicit yes sets up development."""
    body = _sh_mode_body(in_checkout=True, has_root=True).replace(
        "choose_setup_mode </dev/null", "choose_setup_mode"
    )
    script = tmp_path / "harness.sh"
    script.write_text(
        f"#!/bin/sh\nset -eu\n{_sh_runtime()}\n{body}\n", encoding="utf-8"
    )
    if _SH is None:
        pytest.skip(_NO_SH)
    output = _run_in_pty([_SH, str(script)], _env(NO_COLOR="1"), answer)
    assert "? Set up a development environment here instead of installing" in output
    assert f"mode={mode}" in output


def test_setup_sh_main_no_longer_switches_modes_silently() -> None:
    main = _shell_function("main")
    assert "if is_dev_mode" not in main
    assert main.index("clack_intro") < main.index("warn_if_root")
    assert main.index("warn_if_root") < main.index("choose_setup_mode")
    assert 'if [ "$SETUP_MODE" = "dev" ]' in main


def _ps_mode_body(in_checkout: bool, has_root: bool) -> str:
    root = "'/repo'" if has_root else "$null"
    return (
        f"function Test-DevMode {{ return ${'true' if in_checkout else 'false'} }}\n"
        f"function Get-ProjectRoot {{ return {root} }}\n"
        "function Test-InteractiveInput { return $false }\n"
        '$script:SETUP_MODE = "install"\n$script:CHECKOUT_ANNOUNCED = $false\n'
        f"{_powershell_function('Select-SetupMode')}"
        'Clack-Intro "Markitai Setup"\n'
        "Select-SetupMode\n"
        '"mode=$($script:SETUP_MODE)"\n'
    )


@_needs_pwsh
def test_setup_ps1_checkout_root_without_terminal_installs(tmp_path: Path) -> None:
    result = _run_ps(tmp_path, _ps_mode_body(in_checkout=True, has_root=True))
    assert result.returncode == 0, result.stderr
    assert result.stdout.rstrip().endswith("mode=install")
    assert "MARKITAI_SETUP_MODE=dev" in result.stdout


@_needs_pwsh
def test_setup_ps1_mode_env_selects_dev(tmp_path: Path) -> None:
    body = _ps_mode_body(in_checkout=False, has_root=True)
    result = _run_ps(tmp_path, body, MARKITAI_SETUP_MODE="Dev")
    assert result.stdout.rstrip().endswith("mode=dev")


@_needs_pwsh
def test_setup_ps1_dev_mode_needs_a_checkout(tmp_path: Path) -> None:
    body = _ps_mode_body(in_checkout=False, has_root=False)
    result = _run_ps(tmp_path, body, MARKITAI_SETUP_MODE="dev")
    assert result.returncode == 1
    assert "needs a markitai checkout" in result.stdout


def test_setup_ps1_main_no_longer_switches_modes_silently() -> None:
    main = _powershell_function("Main")
    assert "Test-DevMode" not in main
    assert main.index("Clack-Intro") < main.index("Test-AdminWarning")
    assert main.index("Test-AdminWarning") < main.index("Select-SetupMode")
    assert '$script:SETUP_MODE -eq "dev"' in main


# ============================================================
# Install source: in a checkout, install from it unless told otherwise
# ============================================================


def _sh_source_body(in_checkout: bool) -> str:
    detect = "MARKITAI_REPO_ROOT=/repo; return 0" if in_checkout else "return 1"
    return (
        f"detect_markitai_repo() {{ {detect}; }}\n"
        'MARKITAI_SOURCE="pypi"\nMARKITAI_LOCAL_PATH=""\nMARKITAI_REPO_ROOT=""\n'
        "CHECKOUT_ANNOUNCED=false\n"
        f"{_shell_function('choose_install_source')}"
        'clack_intro "Markitai Setup"\n'
        "choose_install_source </dev/null\n"
        'printf "source=%s path=%s\\n" "$MARKITAI_SOURCE" "$MARKITAI_LOCAL_PATH"\n'
    )


def test_setup_sh_checkout_without_terminal_installs_the_checkout(
    tmp_path: Path,
) -> None:
    result = _run_sh(tmp_path, _sh_source_body(in_checkout=True))
    assert result.returncode == 0, result.stderr
    assert result.stdout.rstrip().endswith("source=local path=/repo/packages/markitai")
    assert "MARKITAI_SETUP_SOURCE=pypi" in result.stdout


@pytest.mark.parametrize("value", ["pypi", "PyPI", "release"])
def test_setup_sh_source_env_selects_pypi(tmp_path: Path, value: str) -> None:
    result = _run_sh(
        tmp_path, _sh_source_body(in_checkout=True), MARKITAI_SETUP_SOURCE=value
    )
    assert result.stdout.rstrip().endswith("source=pypi path=")
    assert "Install source: PyPI" in result.stdout


def test_setup_sh_pinned_version_installs_that_release(tmp_path: Path) -> None:
    result = _run_sh(
        tmp_path, _sh_source_body(in_checkout=True), MARKITAI_VERSION="1.2.1"
    )
    assert result.stdout.rstrip().endswith("source=pypi path=")
    assert "MARKITAI_VERSION is set" in result.stdout


def test_setup_sh_local_source_needs_a_checkout(tmp_path: Path) -> None:
    result = _run_sh(
        tmp_path, _sh_source_body(in_checkout=False), MARKITAI_SETUP_SOURCE="local"
    )
    assert result.returncode == 1
    assert "needs this script to run from a markitai checkout" in result.stdout


def test_setup_sh_unknown_source_warns_and_keeps_the_default(tmp_path: Path) -> None:
    result = _run_sh(
        tmp_path, _sh_source_body(in_checkout=True), MARKITAI_SETUP_SOURCE="git"
    )
    assert "! Ignoring MARKITAI_SETUP_SOURCE" in result.stdout
    assert result.stdout.rstrip().endswith("source=local path=/repo/packages/markitai")


@_needs_pty
@pytest.mark.parametrize(("answer", "source"), [(b"", "local"), (b"n", "pypi")])
def test_setup_sh_source_question_defaults_to_the_checkout(
    tmp_path: Path, answer: bytes, source: str
) -> None:
    """Enter installs this checkout; only an explicit no picks PyPI."""
    body = _sh_source_body(in_checkout=True).replace(
        "choose_install_source </dev/null", "choose_install_source"
    )
    script = tmp_path / "harness.sh"
    script.write_text(
        f"#!/bin/sh\nset -eu\n{_sh_runtime()}\n{body}\n", encoding="utf-8"
    )
    if _SH is None:
        pytest.skip(_NO_SH)
    output = _run_in_pty([_SH, str(script)], _env(NO_COLOR="1"), answer, b"[Y/n]")
    assert "? Install markitai from this checkout? (n: the PyPI release instead)" in (
        output
    )
    assert f"source={source}" in output


def _ps_source_body(in_checkout: bool) -> str:
    root = "'/repo'" if in_checkout else "$null"
    return (
        f"function Get-MarkitaiRepoRoot {{ return {root} }}\n"
        "function Test-InteractiveInput { return $false }\n"
        '$script:MARKITAI_SOURCE = "pypi"\n$script:MarkitaiLocalPath = $null\n'
        "$script:MarkitaiVersion = $env:MARKITAI_VERSION\n"
        "$script:CHECKOUT_ANNOUNCED = $false\n"
        f"{_powershell_function('Select-InstallSource')}"
        'Clack-Intro "Markitai Setup"\n'
        "Select-InstallSource\n"
        '"source=$($script:MARKITAI_SOURCE) path=$($script:MarkitaiLocalPath)"\n'
    )


@_needs_pwsh
def test_setup_ps1_checkout_without_terminal_installs_the_checkout(
    tmp_path: Path,
) -> None:
    result = _run_ps(tmp_path, _ps_source_body(in_checkout=True))
    assert result.returncode == 0, result.stderr
    assert "source=local path=" in result.stdout
    assert "packages" in result.stdout.rstrip().splitlines()[-1]
    assert "MARKITAI_SETUP_SOURCE=pypi" in result.stdout


@_needs_pwsh
def test_setup_ps1_source_env_and_pin_select_pypi(tmp_path: Path) -> None:
    body = _ps_source_body(in_checkout=True)
    by_env = _run_ps(tmp_path, body, MARKITAI_SETUP_SOURCE="PyPI")
    assert by_env.stdout.rstrip().endswith("source=pypi path=")
    pinned = _run_ps(tmp_path, body, MARKITAI_VERSION="1.2.1")
    assert pinned.stdout.rstrip().endswith("source=pypi path=")
    assert "MARKITAI_VERSION is set" in pinned.stdout


@_needs_pwsh
def test_setup_ps1_local_source_needs_a_checkout(tmp_path: Path) -> None:
    body = _ps_source_body(in_checkout=False)
    result = _run_ps(tmp_path, body, MARKITAI_SETUP_SOURCE="local")
    assert result.returncode == 1
    assert "needs this script to run from a markitai checkout" in result.stdout


def test_both_scripts_default_the_source_question_to_the_checkout() -> None:
    sh = _shell_function("choose_install_source")
    ps = _powershell_function("Select-InstallSource")
    assert 'clack_confirm "$(i18n confirm_local_install)" "y"' in sh
    assert 'Clack-Confirm (i18n "confirm_local_install") "y"' in ps


# ============================================================
# Root / administrator: explained, never a dead end without a terminal
# ============================================================


def _sh_root_body(interactive: bool) -> str:
    return (
        "id() { echo 0; }\n"
        f"has_interactive_tty() {{ return {0 if interactive else 1}; }}\n"
        "clack_confirm() { return 1; }\n"
        f"{_shell_function('warn_if_root')}"
        'clack_intro "Markitai Setup"\n'
        "warn_if_root\n"
        "echo continued\n"
    )


def test_setup_sh_root_warning_explains_and_continues_headless(tmp_path: Path) -> None:
    result = _run_sh(tmp_path, _sh_root_body(interactive=False))
    assert result.returncode == 0, result.stderr
    assert "! Running as root" in result.stdout
    assert "installs per user and never needs root" in result.stdout
    assert "No terminal to ask; continuing as root" in result.stdout
    assert result.stdout.rstrip().endswith("continued")


def test_setup_sh_root_prompt_defaults_to_stop(tmp_path: Path) -> None:
    result = _run_sh(tmp_path, _sh_root_body(interactive=True))
    assert result.returncode == 1
    assert "Setup cancelled" in result.stdout
    assert "continued" not in result.stdout


def _ps_admin_body(interactive: bool) -> str:
    return (
        "function Test-IsAdministrator { return $true }\n"
        f"function Test-InteractiveInput {{ return ${'true' if interactive else 'false'} }}\n"
        "function Clack-Confirm { return $false }\n"
        f"{_powershell_function('Test-AdminWarning')}"
        'Clack-Intro "Markitai Setup"\n'
        "Test-AdminWarning\n"
        "'continued'\n"
    )


@_needs_pwsh
def test_setup_ps1_admin_warning_explains_and_continues_headless(
    tmp_path: Path,
) -> None:
    result = _run_ps(tmp_path, _ps_admin_body(interactive=False))
    assert result.returncode == 0, result.stderr
    assert "! Running as administrator" in result.stdout
    assert "installs per user and never needs administrator rights" in result.stdout
    assert "continuing as administrator" in result.stdout
    assert result.stdout.rstrip().endswith("continued")


@_needs_pwsh
def test_setup_ps1_admin_prompt_defaults_to_stop(tmp_path: Path) -> None:
    result = _run_ps(tmp_path, _ps_admin_body(interactive=True))
    assert result.returncode == 1
    assert "Setup cancelled" in result.stdout


def test_wsl_question_no_longer_asks_about_administrator() -> None:
    body = _powershell_function("Test-WSLWarning")
    assert "confirm_continue_as_admin" not in body
    assert "confirm_continue" in body


# ============================================================
# Third-party CLIs: asked first, off without a terminal
# ============================================================


@pytest.mark.parametrize(
    "installer", ["install_optional_claude_cli", "install_optional_copilot_cli"]
)
def test_setup_sh_third_party_clis_default_off_headless(
    tmp_path: Path, installer: str
) -> None:
    marker = tmp_path / "installed"
    body = (
        "has_interactive_tty() { return 1; }\n"
        f'run_remote_shell_installer() {{ echo remote >> "{marker}"; return 0; }}\n'
        f'clack_run_quiet() {{ echo quiet >> "{marker}"; return 0; }}\n'
        "install_markitai_extra() { :; }\ntrack_install() { :; }\n"
        f"{_shell_function(installer)}"
        f"if {installer}; then rc=0; else rc=$?; fi\n"
        'echo "rc=$rc"\n'
    )
    result = _run_sh(tmp_path, body, PATH="/usr/bin:/bin")
    assert result.stdout.rstrip().endswith("rc=2"), result.stdout + result.stderr
    assert "skipped" in result.stdout
    assert not marker.exists()


@_needs_pwsh
@pytest.mark.parametrize(
    "installer", ["Install-OptionalClaudeCLI", "Install-OptionalCopilotCLI"]
)
def test_setup_ps1_third_party_clis_default_off_headless(
    tmp_path: Path, installer: str
) -> None:
    marker = tmp_path / "installed"
    body = (
        "function Get-Command { [CmdletBinding()] param([string]$Name) return $null }\n"
        "function Test-InteractiveInput { return $false }\n"
        f"function Invoke-PowerShellInstallerQuietly {{ 'x' | Add-Content '{marker}' }}\n"
        f"function Invoke-NativeQuietly {{ 'x' | Add-Content '{marker}' }}\n"
        "function Install-MarkitaiExtra {}\nfunction Track-Install {}\n"
        f"{_powershell_function(installer)}"
        f'"result=$({installer})"\n'
    )
    result = _run_ps(tmp_path, body)
    assert result.stdout.rstrip().endswith("result=False"), (
        result.stdout + result.stderr
    )
    assert "skipped" in result.stdout
    assert not marker.exists()


# ============================================================
# The summary of a headless install
# ============================================================

# (terminal, MARKITAI_INSTALL_OPTIONAL, components skipped, hint expected)
_HEADLESS_SUMMARY_CASES = [
    pytest.param(False, "", True, True, id="headless"),
    pytest.param(False, "1", True, False, id="opted-in"),
    pytest.param(True, "", True, False, id="terminal"),
    pytest.param(False, "", False, False, id="nothing-skipped"),
]
_HEADLESS_HINT = (
    "No terminal, so optional components were skipped; to install them too, "
    "set this and run setup again:"
)


@pytest.mark.parametrize(
    ("terminal", "opted_in", "skipped", "shown"), _HEADLESS_SUMMARY_CASES
)
def test_setup_sh_headless_summary_says_how_to_get_everything(
    tmp_path: Path, terminal: bool, opted_in: str, skipped: bool, shown: bool
) -> None:
    """Without a terminal only the core goes in; say how to get the rest."""
    body = (
        f"has_interactive_tty() {{ return {0 if terminal else 1}; }}\n"
        'INSTALLED_COMPONENTS="uv|python|markitai|"\n'
        f'SKIPPED_COMPONENTS="{"serve|ocr|playwright|" if skipped else ""}"\n'
        'FAILED_COMPONENTS=""\n'
        f"{_shell_function('print_summary_rows')}{_shell_function('print_summary')}"
        "print_summary --no-header\n"
    )
    result = _run_sh(tmp_path, body, MARKITAI_INSTALL_OPTIONAL=opted_in)
    assert result.returncode == 0, result.stderr
    lines = [line.rstrip() for line in result.stdout.splitlines()]
    if not shown:
        assert "MARKITAI_INSTALL_OPTIONAL" not in result.stdout
        return
    hint = next(i for i, line in enumerate(lines) if _HEADLESS_HINT in line)
    assert lines[hint + 1].endswith("export MARKITAI_INSTALL_OPTIONAL=1")


@_needs_pwsh
@pytest.mark.parametrize(
    ("terminal", "opted_in", "skipped", "shown"), _HEADLESS_SUMMARY_CASES
)
def test_setup_ps1_headless_summary_says_how_to_get_everything(
    tmp_path: Path, terminal: bool, opted_in: str, skipped: bool, shown: bool
) -> None:
    components = "@('serve', 'ocr', 'playwright')" if skipped else "@()"
    body = (
        f"function Test-InteractiveInput {{ return ${str(terminal).lower()} }}\n"
        "$script:INSTALLED_COMPONENTS = @('uv', 'python', 'markitai')\n"
        f"$script:SKIPPED_COMPONENTS = {components}\n"
        "$script:FAILED_COMPONENTS = @()\n"
        f"{_powershell_function('Print-Summary')}"
        "Print-Summary -NoHeader\n"
    )
    result = _run_ps(tmp_path, body, MARKITAI_INSTALL_OPTIONAL=opted_in)
    assert result.returncode == 0, result.stderr
    lines = [line.rstrip() for line in result.stdout.splitlines()]
    if not shown:
        assert "MARKITAI_INSTALL_OPTIONAL" not in result.stdout
        return
    hint = next(i for i, line in enumerate(lines) if _HEADLESS_HINT in line)
    assert lines[hint + 1].endswith('$env:MARKITAI_INSTALL_OPTIONAL = "1"')


# ============================================================
# uv's tool command folder
# ============================================================


def test_setup_sh_finds_markitai_where_uv_put_it(tmp_path: Path) -> None:
    """UV_TOOL_BIN_DIR off PATH used to lose the version, init and extras.

    The script assumed ~/.local/bin, so with another tool bin dir markitai
    was not found after the install: `doctor --suggest-extras` returned
    nothing (no legacy, mcp, svg, heif extras), `init` never ran, and the
    "Getting started" commands did not work in the user's shell.
    """
    bin_dir = tmp_path / "tool-bin"
    body = (
        f'uv() {{ echo "{bin_dir}"; }}\n'
        f"{_shell_function('use_uv_tool_bin_dir')}"
        f"{_shell_function('warn_if_tool_bin_off_path')}"
        'SETUP_ORIGINAL_PATH="$PATH"\n'
        "use_uv_tool_bin_dir\n"
        'case ":$PATH:" in *":$UV_TOOL_BIN:"*) echo on-path ;; esac\n'
        "warn_if_tool_bin_off_path\n"
    )
    result = _run_sh(tmp_path, body)
    assert result.returncode == 0, result.stderr
    assert "on-path" in result.stdout
    assert (
        f"PATH does not include markitai's command folder: {bin_dir}" in result.stdout
    )
    assert "\u2502    uv tool update-shell" in result.stdout
    for name in ("install_markitai",):
        body = _shell_function(name)
        assert "$HOME/.local/bin" not in body
        assert "use_uv_tool_bin_dir" in body
    assert "warn_if_tool_bin_off_path" in _shell_function("print_user_completion")


# ============================================================
# Windows on Arm: markitai on an x64 Python
# ============================================================


def _run_install_python(
    tmp_path: Path, *, on_arm: bool, available: list[str]
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    """Install-Python against a stub uv that finds *available* requests.

    The stub is a real program on PATH, so "not found" reaches PowerShell as
    native stderr: Windows PowerShell 5.1 turns that into a terminating error
    under setup.ps1's "Stop" unless the call guards against it.
    """
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    state = tmp_path / "uv-state.json"
    state.write_text(json.dumps({"available": available, "calls": []}), "utf-8")
    script = stub_dir / "uv_stub.py"
    script.write_text(
        "import json, sys\n"
        f"state_path = {str(state)!r}\n"
        "state = json.load(open(state_path, encoding='utf-8'))\n"
        "args = sys.argv[1:]\n"
        "state['calls'].append(' '.join(args))\n"
        "code = 0\n"
        "if args[:2] == ['python', 'install']:\n"
        "    state['available'].append(args[2])\n"
        "elif args[:2] == ['python', 'find'] and args[2] in state['available']:\n"
        "    print(sys.executable)\n"
        "else:\n"
        "    print('error: No interpreter found', file=sys.stderr)\n"
        "    code = 2\n"
        "json.dump(state, open(state_path, 'w', encoding='utf-8'))\n"
        "sys.exit(code)\n",
        encoding="utf-8",
    )
    # A PowerShell script, not a uv.cmd: cmd.exe read the unquoted
    # ">=3.11,<3.14" request as redirections. PowerShell hands it over
    # whole, as it does to the real uv.exe.
    (stub_dir / "uv.ps1").write_text(
        f"& '{sys.executable}' '{script}' @args\nexit $LASTEXITCODE\n",
        encoding="utf-8",
    )
    body = (
        f"function Test-WindowsOnArm {{ return ${str(on_arm).lower()} }}\n"
        "function Test-CommandExists { return $true }\n"
        f"{_powershell_function('Invoke-NativeQuietly')}"
        f"{_powershell_function('Find-UvPython')}"
        f"{_powershell_function('Select-Python')}"
        f"{_powershell_function('Install-Python')}"
        '"result=$(Install-Python) python=$($script:PYTHON_CMD)"\n'
    )
    result = _run_ps(
        tmp_path, body, PATH=f"{stub_dir}{os.pathsep}{os.environ.get('PATH', '')}"
    )
    calls = json.loads(state.read_text("utf-8"))["calls"]
    return result, calls


_X64 = "cpython-{}-windows-x86_64"


@_needs_pwsh
def test_setup_ps1_uses_an_x64_python_on_windows_on_arm(tmp_path: Path) -> None:
    """A native ARM64 Python cannot install markitai (pymupdf-layout has no
    ARM64 build); the x64 one uv finds is used, and the run says why."""
    result, calls = _run_install_python(
        tmp_path, on_arm=True, available=[_X64.format("3.12"), ">=3.11,<3.14"]
    )
    assert result.returncode == 0, result.stderr
    assert f"result=True python={sys.executable}" in result.stdout
    assert calls == [
        f"python find {_X64.format('3.13')}",
        f"python find {_X64.format('3.12')}",
    ]
    assert "Windows on Arm: markitai runs on an x64 Python" in result.stdout
    assert "(x64)" in result.stdout


@_needs_pwsh
def test_setup_ps1_installs_an_x64_python_on_a_fresh_arm_machine(
    tmp_path: Path,
) -> None:
    """Nothing found (uv says so on stderr, which must not end setup): uv
    installs CPython 3.13 x64, not the native ARM64 build."""
    result, calls = _run_install_python(tmp_path, on_arm=True, available=[])
    assert result.returncode == 0, result.stderr
    assert f"result=True python={sys.executable}" in result.stdout
    assert calls == [
        f"python find {_X64.format('3.13')}",
        f"python find {_X64.format('3.12')}",
        f"python find {_X64.format('3.11')}",
        f"python install {_X64.format('3.13')}",
        f"python find {_X64.format('3.13')}",
    ]


@_needs_pwsh
def test_setup_ps1_keeps_any_supported_python_elsewhere(tmp_path: Path) -> None:
    result, calls = _run_install_python(
        tmp_path, on_arm=False, available=[">=3.11,<3.14"]
    )
    assert result.returncode == 0, result.stderr
    assert f"result=True python={sys.executable}" in result.stdout
    assert calls == ["python find >=3.11,<3.14"]
    assert "Windows on Arm" not in result.stdout


@_needs_pwsh
def test_setup_ps1_installs_python_when_none_is_found(tmp_path: Path) -> None:
    result, calls = _run_install_python(tmp_path, on_arm=False, available=[])
    assert result.returncode == 0, result.stderr
    assert calls == [
        "python find >=3.11,<3.14",
        "python install 3.13",
        "python find 3.13",
    ]


@_needs_pwsh
def test_setup_ps1_reads_the_machine_architecture(tmp_path: Path) -> None:
    """The registry's machine value wins over the process's own (an emulated
    x64 PowerShell sees AMD64); the variable is the fallback."""
    function = _powershell_function("Test-WindowsOnArm")
    body = (
        "$script:OnWindows = $true\n"
        "function Get-ItemProperty { [CmdletBinding()] param([string]$Path) "
        "return [pscustomobject]@{ PROCESSOR_ARCHITECTURE = $env:TEST_MACHINE_ARCH } }\n"
        f"{function}"
        '"arm=$(Test-WindowsOnArm)"\n'
    )
    arm = _run_ps(
        tmp_path, body, TEST_MACHINE_ARCH="ARM64", PROCESSOR_ARCHITECTURE="AMD64"
    )
    assert arm.stdout.rstrip().endswith("arm=True"), arm.stdout + arm.stderr
    x64 = _run_ps(tmp_path, body, TEST_MACHINE_ARCH="AMD64")
    assert x64.stdout.rstrip().endswith("arm=False"), x64.stdout + x64.stderr
    off = _run_ps(
        tmp_path, body.replace("$true", "$false", 1), TEST_MACHINE_ARCH="ARM64"
    )
    assert off.stdout.rstrip().endswith("arm=False"), off.stdout + off.stderr


def test_setup_ps1_finds_markitai_where_uv_put_it() -> None:
    assert "Add-UvToolBinToPath" in _powershell_function("Install-Markitai")
    assert "Show-UvToolBinPathWarning" in _powershell_function("Print-UserCompletion")
    assert "uv tool dir --bin" in _powershell_function("Add-UvToolBinToPath")


def test_ci_runs_the_guards_when_only_a_setup_script_changes() -> None:
    """Python CI used to ignore scripts/setup.*, so these tests never ran on them."""
    import yaml

    workflow = yaml.safe_load(_text(_CI_WORKFLOW))
    triggers = workflow.get("on", workflow.get(True))  # YAML 1.1 reads `on` as True
    for event in ("push", "pull_request"):
        paths = triggers[event]["paths"]
        assert "scripts/setup.sh" in paths, event
        assert "scripts/setup.ps1" in paths, event


@_needs_pwsh
@pytest.mark.parametrize(
    ("redirected", "code_page", "modern", "vt", "term", "unicode"),
    [
        # Windows pipes and files: UTF-8 once the script switched the console
        # encoding, whatever the shell's own code page was
        pytest.param("$true", 65001, "$false", "$false", "", True, id="file-utf8"),
        pytest.param("$true", 437, "$true", "$true", "", False, id="file-not-utf8"),
        # Consoles: escape sequences on (Windows 10 1511+), or a modern terminal
        pytest.param("$false", 437, "$false", "$true", "", True, id="console-vt"),
        pytest.param("$false", 437, "$true", "$false", "", True, id="windows-terminal"),
        pytest.param("$false", 437, "$false", "$false", "", False, id="legacy-console"),
        pytest.param("$true", 65001, "$true", "$true", "dumb", False, id="term-dumb"),
    ],
)
def test_setup_ps1_marks_follow_markitais_rule(
    tmp_path: Path,
    redirected: str,
    code_page: int,
    modern: str,
    vt: str,
    term: str,
    unicode: bool,
) -> None:
    """Marks mean the same as in markitai (Python and Rust): ASCII only for
    TERM=dumb, a console without escape sequences, or output that is not
    UTF-8. A redirect used to get ASCII unless the shell's own code page was
    65001, although the bytes were UTF-8."""
    body = (
        "$script:OnWindows = $true\n"
        f"$script:ModernTerminal = {modern}\n"
        f"function Test-OutputRedirected {{ return {redirected} }}\n"
        f"function Get-OutputCodePage {{ return {code_page} }}\n"
        f"function Test-VirtualTerminalConsole {{ return {vt} }}\n"
        f"{_powershell_function('Test-UnicodeMarks')}"
        '"unicode=$(Test-UnicodeMarks)"\n'
    )
    result = _run_ps(tmp_path, body, runtime=False, TERM=term)
    assert result.stdout.rstrip().endswith(f"unicode={unicode}"), (
        result.stdout + result.stderr
    )


@_needs_pwsh
@pytest.mark.parametrize(("preset", "expected"), [(None, "1"), ("0", "0")])
def test_setup_ps1_compiles_bytecode_unless_told_otherwise(
    tmp_path: Path, preset: str | None, expected: str
) -> None:
    env = {} if preset is None else {"UV_COMPILE_BYTECODE": preset}
    result = _run_ps(tmp_path, '"compile=$env:UV_COMPILE_BYTECODE"', **env)
    assert result.stdout.rstrip().endswith(f"compile={expected}"), (
        result.stdout + result.stderr
    )
