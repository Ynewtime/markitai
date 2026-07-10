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

APP_BY_SUFFIX: dict[str, str] = {
    ".doc": "Microsoft Word",
    ".ppt": "Microsoft PowerPoint",
    ".xls": "Microsoft Excel",
}

# Office apps are single-instance and expose process-global automation
# settings. Serialize per app within this process; _office_app_lock adds the
# cross-process half before any setting is changed or document is opened.
_APP_LOCKS: dict[str, threading.Lock] = {
    "Microsoft Word": threading.Lock(),
    "Microsoft PowerPoint": threading.Lock(),
    "Microsoft Excel": threading.Lock(),
}
_APP_LOCK_NAMES: dict[str, str] = {
    "Microsoft Word": "word.lock",
    "Microsoft PowerPoint": "powerpoint.lock",
    "Microsoft Excel": "excel.lock",
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


def _wrap_office_script(
    app: str,
    *,
    open_lines: list[str],
    action_lines: list[str],
    extra_setting: tuple[str, str] | None = None,
) -> str:
    """Wrap one conversion in safe Office automation state handling.

    ``automation security`` is an application-global setting, so it is
    restored immediately after the staged file opens and again on every error
    path. The exact object returned by ``open`` is retained throughout; user
    documents and whichever window happens to be active are never referenced.
    """
    lines = [
        f'tell application "{app}"',
        "    set openedItem to missing value",
        "    set previousAutomationSecurity to automation security",
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
    lines.append("        set automation security to previousAutomationSecurity")
    if extra_setting is not None:
        setting, _disabled_value = extra_setting
        lines.append(f"        set {setting} to previousExtraSetting")
    lines.extend(f"        {line}" for line in action_lines)
    lines.extend(
        [
            "        close openedItem saving no",
            "    on error errorMessage number errorNumber",
            "        try",
            "            set automation security to previousAutomationSecurity",
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
        [
            "        if openedItem is not missing value then",
            "            try",
            "                close openedItem saving no",
            "            end try",
            "        end if",
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
#   Excel "Excel XML file format"          == xlOpenXMLWorkbook (51)
# Word/Excel take a text path ("POSIX file ... as string" -> HFS text);
# PowerPoint takes the file object directly (verified live).
def _build_legacy_script(app: str, staged_in: Path, staged_out: Path) -> str:
    inp = _as_quote(staged_in)
    out = _as_quote(staged_out)
    if app == "Microsoft Word":
        return _wrap_office_script(
            app,
            open_lines=[
                f'set openedItem to open (POSIX file "{inp}") '
                "read only true add to recent files false"
            ],
            action_lines=[
                f'set outPath to (POSIX file "{out}") as string',
                "save as openedItem file name outPath "
                "file format format document default",
            ],
            # Word otherwise updates embedded OLE links while opening.
            extra_setting=("update links at open of settings", "false"),
        )
    if app == "Microsoft PowerPoint":
        return _wrap_office_script(
            app,
            open_lines=[f'set openedItem to open (POSIX file "{inp}")'],
            action_lines=[
                f'save openedItem in (POSIX file "{out}") '
                "as save as Open XML presentation"
            ],
        )
    if app == "Microsoft Excel":
        return _wrap_office_script(
            app,
            open_lines=[
                f'set inPath to (POSIX file "{inp}") as string',
                "set openedItem to open workbook workbook file name inPath "
                "update links do not update links read only true "
                "ignore read only recommended true add to mru false",
            ],
            action_lines=[
                f'set outPath to (POSIX file "{out}") as string',
                "save workbook as openedItem filename outPath "
                "file format Excel XML file format",
            ],
        )
    raise ValueError(f"Unsupported Office app: {app}")


def _build_pdf_script(staged_in: Path, staged_out: Path) -> str:
    inp = _as_quote(staged_in)
    out = _as_quote(staged_out)
    return _wrap_office_script(
        "Microsoft PowerPoint",
        open_lines=[f'set openedItem to open (POSIX file "{inp}")'],
        action_lines=[f'save openedItem in (POSIX file "{out}") as save as PDF'],
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
        raise RuntimeError(f"{app} automation failed: {stderr[:300]}")


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
        staged_out = work / f"output{output_file.suffix}"

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
    """Convert a legacy Office file (.doc/.ppt/.xls) to its modern format.

    Args:
        input_path: Path to the legacy file.
        target_format: Target extension without dot (docx, pptx, xlsx).
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
