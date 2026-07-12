"""MS Office automation fallback for macOS (AppleScript via osascript).

Used only when LibreOffice is not installed. Mirrors the Windows COM
fallback in converter/legacy.py architecturally: shell out to the OS
automation layer (osascript here, PowerShell there), so no Python
dependency is added.

Sandbox note: Office apps are sandboxed; making them read/write arbitrary
paths (e.g. /tmp) triggers a per-folder "Grant File Access" dialog. All
file I/O therefore goes through a staging directory inside Office's own
group container (~/Library/Group Containers/UBF8T346G9.Office/), which
the apps can access without prompting. The staging root is fixed so that
any one-off grant the user does make persists across runs.

Automation (TCC) note: the first Apple Event sent to each Office app
triggers a one-time "wants to control" consent dialog for the host app.
A denial surfaces as osascript error -1743, mapped to an actionable
message below.

Cold-start note (verified live 2026-07-12, Office 16.110): on an app's
first scripted launch after an Office update, the app answers Apple
Events normally but silently drops ``open`` requests that carry labeled
parameters, while sitting hidden at its startup gallery with an idle
main thread. A parameterless ``open`` goes through instantly and heals
the instance for the rest of its lifetime. Word's script therefore
retries with a plain ``open`` when the existence poll stalls; PowerPoint
fails that state differently (its ``open`` raises -9074 outright), so
its script has no in-poll fallback and the error message carries the
recovery steps instead. Quitting and retrying does NOT clear the state —
opening the app manually once does.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from functools import cache
from pathlib import Path

from loguru import logger

# Timeouts aligned with the LibreOffice paths they substitute for
# (legacy.py: 120s per file; office.py PDF export: 600s).
LEGACY_TIMEOUT = 120
PDF_TIMEOUT = 600

_OFFICE_GROUP_CONTAINER = (
    Path.home() / "Library" / "Group Containers" / "UBF8T346G9.Office"
)
_STAGING_ROOT = _OFFICE_GROUP_CONTAINER / "markitai"
_USER_ID = getattr(os, "getuid", lambda: 0)()
_FALLBACK_STAGING_ROOT = Path(tempfile.gettempdir()) / f"markitai-office-{_USER_ID}"
_STALE_STAGING_AGE_SECONDS = 24 * 60 * 60

# osascript error code for a denied/unanswered TCC automation prompt
_ERR_NOT_AUTHORIZED = "-1743"

# Script result marking that the mid-poll plain-open fallback ran
# (see _wrap_office_script); surfaced on stdout for logging only.
_FALLBACK_MARKER = "markitai:fallback-open-used"

APP_BY_SUFFIX: dict[str, str] = {
    ".doc": "Microsoft Word",
    ".ppt": "Microsoft PowerPoint",
}

# Office apps are single-instance and expose process-global automation
# settings. Serialize per app within this process; _office_app_lock adds the
# cross-process half before any setting is changed or document is opened.
_APP_LOCKS: dict[str, threading.Lock] = {
    "Microsoft Word": threading.Lock(),
    "Microsoft PowerPoint": threading.Lock(),
}
_APP_LOCK_NAMES: dict[str, str] = {
    "Microsoft Word": "word.lock",
    "Microsoft PowerPoint": "powerpoint.lock",
}


_APP_SEARCH_BASES: tuple[Path, ...] = (
    Path("/Applications"),
    Path.home() / "Applications",
)


@cache
def find_ms_office_app(app_name: str) -> bool:
    """Check whether an MS Office application is installed (macOS only).

    Filesystem check only -- never launches the app (unlike the Windows
    COM probe in utils/office.py).
    """
    if platform.system() != "Darwin":
        return False
    for base in _APP_SEARCH_BASES:
        if (base / f"{app_name}.app").exists():
            logger.debug(f"{app_name} found at: {base / f'{app_name}.app'}")
            return True
    logger.debug(f"{app_name} not found")
    return False


def legacy_app_available(suffix: str) -> bool:
    """Check whether the Office app handling *suffix* is installed."""
    app = APP_BY_SUFFIX.get(suffix.lower())
    return find_ms_office_app(app) if app else False


def powerpoint_available() -> bool:
    """Check whether Microsoft PowerPoint is installed (for PDF export)."""
    return find_ms_office_app("Microsoft PowerPoint")


def _as_quote(path: Path) -> str:
    """Escape a path for embedding in an AppleScript string literal."""
    return str(path).replace("\\", "\\\\").replace('"', '\\"')


# AppleScript container class for "the document we opened", per app.
_CONTAINER_BY_APP: dict[str, str] = {
    "Microsoft Word": "document",
    "Microsoft PowerPoint": "presentation",
}


def _restore_security_lines(indent: str = "") -> list[str]:
    """Restore automation security without ever leaving ForceDisable behind.

    A cold-launching app (observed with PowerPoint) can answer the initial
    ``automation security`` read with missing value; restoring that literal
    raises and — worse — silently strands the app in ForceDisable. Fall back
    to msoAutomationSecurityByUI, the factory default, in that case.
    """
    return [
        f"{indent}if previousAutomationSecurity is not missing value then",
        f"{indent}    set automation security to previousAutomationSecurity",
        f"{indent}else",
        f"{indent}    set automation security to msoAutomationSecurityByUI",
        f"{indent}end if",
    ]


def _close_by_name_lines(app: str, in_name: str, out_name: str) -> list[str]:
    """Best-effort close of the staged document under either of its names.

    Save-as renames the open document to the output file name (verified live
    for Word and PowerPoint), so the staged document may be known by its
    input or its output name at close time. Both closes are individually
    try-wrapped: whichever name no longer resolves is a silent no-op. The
    staged names are unique per conversion (uuid hex), so no user document
    can ever match.
    """
    container = _CONTAINER_BY_APP[app]
    lines = []
    for name in (out_name, in_name):
        lines.extend(
            [
                "try",
                f'    close {container} "{name}" saving no',
                "end try",
            ]
        )
    return lines


def _wrap_office_script(
    app: str,
    *,
    open_lines: list[str],
    action_lines: list[str],
    in_name: str,
    out_name: str,
    extra_setting: tuple[str, str] | None = None,
    fallback_open_line: str | None = None,
) -> str:
    """Wrap one conversion in safe Office automation state handling.

    ``automation security`` is an application-global setting, so it is
    restored immediately after the staged file opens and again on every error
    path.

    The opened document is never taken from ``open``'s return value: Word's
    and PowerPoint's AppleScript ``open`` returns nothing (verified live), so
    ``set openedItem to open ...`` leaves the variable undefined and every
    later reference dies with -2753 — with the real error masked by the
    handler's own reference to the same variable. Instead the document is
    bound by its staged file name (unique uuid hex per conversion, so user
    documents are never referenced), and cleanup closes by name with no
    variable dependency, so the original error always propagates.

    ``fallback_open_line`` is a parameterless ``open`` retried mid-poll when
    the document has not registered after 5s: an app's first scripted launch
    after an Office update silently drops parametered opens but accepts plain
    ones (see the module cold-start note). The fallback trades the primary
    open's side-effect guards for getting the document open at all — the
    staged copy stays safe (chmod 0400), but it may land in the app's recent
    files. When the fallback ran, the script returns a marker string so the
    caller can log the recovery.
    """
    container = _CONTAINER_BY_APP[app]
    lines = [
        f'tell application "{app}"',
        "    set previousAutomationSecurity to automation security",
        "    set usedFallbackOpen to false",
    ]
    if extra_setting is not None:
        setting, _disabled_value = extra_setting
        lines.append(f"    set previousExtraSetting to {setting}")

    lines.extend(
        [
            "    try",
            "        set automation security to msoAutomationSecurityForceDisable",
        ]
    )
    if extra_setting is not None:
        setting, disabled_value = extra_setting
        lines.append(f"        set {setting} to {disabled_value}")
    lines.extend(f"        {line}" for line in open_lines)
    # Bind by name, not by open's (absent) return value. The no-result
    # open is asynchronous in practice: poll until the document registers
    # (~30s ceiling) so the bind cannot grab missing value mid-load.
    lines.extend(
        [
            "        set waitCount to 0",
            f'        repeat until (exists {container} "{in_name}") or waitCount ≥ 150',
            "            delay 0.2",
            "            set waitCount to waitCount + 1",
        ]
    )
    if fallback_open_line is not None:
        # 25 * 0.2s = 5s of silence; healthy opens register in well under
        # 2s even across a cold app launch (measured).
        lines.extend(
            [
                "            if waitCount = 25 then",
                "                set usedFallbackOpen to true",
                "                try",
                f"                    {fallback_open_line}",
                "                end try",
                "            end if",
            ]
        )
    lines.extend(
        [
            "        end repeat",
            # Surface the stall explicitly: without this the bind grabs
            # missing value and dies later at save with an inscrutable -1708.
            f'        if not (exists {container} "{in_name}") then',
            f'            error "{app} accepted the open request but no '
            f"{container} appeared. This is typical of the first scripted "
            f"launch after an Office update: open {app} manually once, let "
            f'it finish first-run setup, then retry." number 6001',
            "        end if",
            f'        set openedItem to {container} "{in_name}"',
        ]
    )
    lines.extend(f"        {line}" for line in _restore_security_lines())
    if extra_setting is not None:
        setting, _disabled_value = extra_setting
        lines.append(f"        set {setting} to previousExtraSetting")
    lines.extend(f"        {line}" for line in action_lines)
    lines.extend(
        f"        {line}" for line in _close_by_name_lines(app, in_name, out_name)
    )
    if fallback_open_line is not None:
        lines.append(f'        if usedFallbackOpen then return "{_FALLBACK_MARKER}"')
    lines.extend(
        [
            "    on error errorMessage number errorNumber",
            "        try",
            *(f"        {line}" for line in _restore_security_lines(indent="    ")),
            "        end try",
        ]
    )
    if extra_setting is not None:
        setting, _disabled_value = extra_setting
        lines.extend(
            [
                "        try",
                f"            set {setting} to previousExtraSetting",
                "        end try",
            ]
        )
    lines.extend(
        f"        {line}" for line in _close_by_name_lines(app, in_name, out_name)
    )
    lines.extend(
        [
            "        error errorMessage number errorNumber",
            "    end try",
            "end tell",
        ]
    )
    return "\n".join(lines)


# Save-as enums verified against each app's sdef dictionary; they map 1:1
# to the Windows COM format codes used in converter/legacy.py:
#   Word  "format document default"        == wdFormatDocumentDefault (16)
#   PPT   "save as Open XML presentation"  == ppSaveAsOpenXMLPresentation (24)
# Word takes a text path ("POSIX file ... as string" -> HFS text);
# PowerPoint takes the file object directly (verified live).
def _build_legacy_script(app: str, staged_in: Path, staged_out: Path) -> str:
    inp = _as_quote(staged_in)
    out = _as_quote(staged_out)
    if app == "Microsoft Word":
        return _wrap_office_script(
            app,
            open_lines=[
                f'open (POSIX file "{inp}") read only true add to recent files false'
            ],
            action_lines=[
                f'set outPath to (POSIX file "{out}") as string',
                "save as openedItem file name outPath "
                "file format format document default",
            ],
            in_name=staged_in.name,
            out_name=staged_out.name,
            # Word otherwise updates embedded OLE links while opening.
            extra_setting=("update links at open of settings", "false"),
            # Word's post-update first launch drops the parametered open
            # above; a plain open penetrates that state (verified live).
            fallback_open_line=f'open (POSIX file "{inp}")',
        )
    if app == "Microsoft PowerPoint":
        return _wrap_office_script(
            app,
            open_lines=[f'open (POSIX file "{inp}")'],
            action_lines=[
                f'save openedItem in (POSIX file "{out}") '
                "as save as Open XML presentation"
            ],
            in_name=staged_in.name,
            out_name=staged_out.name,
        )
    raise ValueError(f"Unsupported Office app: {app}")


def _build_pdf_script(staged_in: Path, staged_out: Path) -> str:
    inp = _as_quote(staged_in)
    out = _as_quote(staged_out)
    return _wrap_office_script(
        "Microsoft PowerPoint",
        open_lines=[f'open (POSIX file "{inp}")'],
        action_lines=[f'save openedItem in (POSIX file "{out}") as save as PDF'],
        in_name=staged_in.name,
        out_name=staged_out.name,
    )


def _run_applescript(script: str, *, timeout: int, app: str) -> None:
    """Run an AppleScript via osascript, mapping failures to clear errors."""
    wrapped = f"with timeout of {timeout} seconds\n{script}\nend timeout"
    try:
        result = subprocess.run(
            ["osascript", "-e", wrapped],
            capture_output=True,
            text=True,
            timeout=timeout + 30,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"{app} automation timed out after {timeout}s. A macOS "
            "permission dialog may be waiting on screen, or the document "
            "opened a modal prompt (e.g. a macro warning)."
        ) from None

    if result.returncode != 0:
        stderr = result.stderr.strip()
        if _ERR_NOT_AUTHORIZED in stderr:
            raise RuntimeError(
                f"Not authorized to automate {app}. Allow your terminal to "
                f"control {app} in System Settings > Privacy & Security > "
                "Automation, then retry."
            )
        if "-9074" in stderr:
            # PowerPoint's catch-all open refusal; observed on the first
            # scripted launch after an Office update (see the module
            # cold-start note). Quitting and retrying does not clear it.
            raise RuntimeError(
                f"{app} refused to open the document (error -9074). This is "
                f"typical of the first scripted launch after an Office "
                f"update: open {app} manually once, let it finish first-run "
                "setup, then retry."
            )
        raise RuntimeError(f"{app} automation failed: {stderr[:300]}")

    if _FALLBACK_MARKER in result.stdout:
        logger.info(
            f"[OfficeMac] {app} dropped the parametered open request "
            "(post-update first-launch state); recovered via the plain "
            "open fallback."
        )


def _staging_base() -> Path:
    """Return the private base directory shared by staging and app locks."""
    if _OFFICE_GROUP_CONTAINER.is_dir():
        return _STAGING_ROOT
    return _FALLBACK_STAGING_ROOT


def _ensure_private_dir(path: Path) -> None:
    if path.is_symlink():
        raise RuntimeError(f"Refusing symlinked Office staging directory: {path}")
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.stat().st_uid != _USER_ID:
        raise RuntimeError(f"Office staging directory is not user-owned: {path}")
    path.chmod(0o700)


@contextmanager
def _office_app_lock(app: str, *, root: Path | None = None) -> Iterator[None]:
    """Serialize Office automation across threads and Markitai processes."""
    import fcntl

    lock_name = _APP_LOCK_NAMES[app]
    with _APP_LOCKS[app]:
        lock_root = (root or _staging_base()) / ".locks"
        _ensure_private_dir(lock_root)
        flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(lock_root / lock_name, flags, 0o600)
        locked = False
        try:
            os.fchmod(fd, 0o600)
            fcntl.flock(fd, fcntl.LOCK_EX)
            locked = True
            yield
        finally:
            if locked:
                fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)


def _is_owned_staging_dir(path: Path) -> bool:
    """Return whether *path* is one of our UUID staging directories."""
    try:
        return (
            not path.is_symlink()
            and path.is_dir()
            and len(path.name) == 32
            and uuid.UUID(hex=path.name).hex == path.name
            and path.stat().st_uid == _USER_ID
        )
    except (OSError, ValueError):
        return False


def _purge_stale_staging_dirs(root: Path) -> None:
    """Remove abandoned private work directories without touching live ones."""
    cutoff = time.time() - _STALE_STAGING_AGE_SECONDS
    try:
        children = list(root.iterdir())
    except OSError as exc:
        logger.warning(f"[OfficeMac] Could not inspect stale staging files: {exc}")
        return

    for child in children:
        if not _is_owned_staging_dir(child):
            continue
        try:
            if child.stat().st_mtime >= cutoff:
                continue
            shutil.rmtree(child)
        except OSError as exc:
            logger.warning(
                f"[OfficeMac] Could not remove stale staging directory {child}: {exc}"
            )


def _cleanup_staging_dir(work: Path) -> None:
    try:
        shutil.rmtree(work)
    except OSError as exc:
        logger.warning(
            f"[OfficeMac] Could not remove staging directory {work}; "
            f"the source copy may remain on disk: {exc}"
        )


def _make_staging_dir() -> Path:
    """Create a per-conversion staging dir Office can access without prompts.

    Falls back to a regular temp dir when the Office group container is
    missing (unusual); Office may then show a one-time Grant File Access
    dialog for that folder.
    """
    if not _OFFICE_GROUP_CONTAINER.is_dir():
        logger.debug(
            "[OfficeMac] Office group container missing; using temp staging "
            "(a Grant File Access dialog may appear)"
        )
    root = _staging_base()
    _ensure_private_dir(root)
    _purge_stale_staging_dirs(root)
    work = root / uuid.uuid4().hex
    work.mkdir(mode=0o700)
    return work


def _convert_via_staging(
    input_path: Path,
    output_file: Path,
    *,
    app: str,
    build_script,
    timeout: int,
) -> Path:
    """Copy input into staging, run the script, move the product out."""
    work = _make_staging_dir()
    try:
        staged_in = work / f"{work.name}{input_path.suffix.lower()}"
        # Do not retain source metadata, flags, or broad permissions in the
        # persistent Office container. A read-only staged source also prevents
        # PowerPoint (whose AppleScript open command has no read-only option)
        # from modifying the original conversion input.
        shutil.copyfile(input_path, staged_in)
        staged_in.chmod(0o400)
        staged_out = work / f"{work.name}{output_file.suffix}"

        script = build_script(staged_in, staged_out)
        with _office_app_lock(app, root=work.parent):
            _run_applescript(script, timeout=timeout, app=app)

        if not staged_out.exists():
            raise RuntimeError(
                f"{app} did not produce {output_file.suffix} output "
                f"for {input_path.name}"
            )
        output_file.parent.mkdir(parents=True, exist_ok=True)
        staged_out.chmod(0o600)
        shutil.move(str(staged_out), str(output_file))
        return output_file
    finally:
        _cleanup_staging_dir(work)


def convert_legacy(input_path: Path, target_format: str, output_dir: Path) -> Path:
    """Convert a legacy Office file (.doc/.ppt) to its modern format.

    Args:
        input_path: Path to the legacy file.
        target_format: Target extension without dot (docx, pptx).
        output_dir: Directory for the converted file.

    Returns:
        Path to the converted file.

    Raises:
        RuntimeError: If the app is unavailable or conversion fails.
    """
    suffix = input_path.suffix.lower()
    app = APP_BY_SUFFIX.get(suffix)
    if not app or not find_ms_office_app(app):
        raise RuntimeError(f"No Microsoft Office app available for {suffix} files.")

    return _convert_via_staging(
        input_path,
        output_dir / f"{input_path.stem}.{target_format}",
        app=app,
        build_script=lambda i, o: _build_legacy_script(app, i, o),
        timeout=LEGACY_TIMEOUT,
    )


def pptx_to_pdf(input_path: Path, output_dir: Path) -> Path:
    """Export a PPTX to PDF via PowerPoint (for slide rendering).

    Returns the path to ``<output_dir>/<stem>.pdf``.

    Raises:
        RuntimeError: If PowerPoint is unavailable or the export fails.
    """
    if not powerpoint_available():
        raise RuntimeError("Microsoft PowerPoint not found.")

    return _convert_via_staging(
        input_path,
        output_dir / f"{input_path.stem}.pdf",
        app="Microsoft PowerPoint",
        build_script=_build_pdf_script,
        timeout=PDF_TIMEOUT,
    )
