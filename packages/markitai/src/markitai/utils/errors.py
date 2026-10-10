"""Shared error taxonomy for messages that are already written for users.

Most failures reach the user through
:func:`markitai.utils.text.format_error_message`, which prefixes the
exception class name (``ValueError: ...``) because an unexpected exception's
type is the single most useful diagnostic hint we have.

Some failures are not unexpected: a missing optional backend, an oversized
input. Their messages are hand-written, name the problem and the fix, and
gain nothing from a class-name prefix — the prefix is pure noise in front of
an actionable sentence. Those errors subclass :class:`SelfExplanatoryError`,
and ``format_error_message`` renders them bare.

The marker is deliberately opt-in: anything not derived from
``SelfExplanatoryError`` (including plain ``ValueError``/``RuntimeError``
escaping a converter) keeps its class name.
"""

from __future__ import annotations


class SelfExplanatoryError(Exception):
    """Marker base for errors whose message needs no class-name prefix.

    Subclass this only when the message alone tells the user what went wrong
    *and* what to do about it. Never subclass it for internal invariants or
    unexpected states — losing the type there costs diagnosability.
    """


class ConfigFileError(Exception):
    """Raised when a config file contains invalid values.

    Carries an actionable message; the CLI layer (MarkitaiGroup.invoke)
    translates it into a ClickException so users see the message instead
    of a raw ValidationError traceback, and the MCP server returns it as the
    tool error. Lives here, framework-free, so ``markitai.api`` can export
    it without importing the configuration module.
    """


def extra_install_command(*extras: str) -> str:
    """The one command that adds extras to *this* markitai install.

    Every message that tells a user how to unblock themselves goes through
    here. Left to themselves the call sites drifted into five spellings of
    the same instruction, two of which did not work: `pip install
    "markitai[legacy]"` and `uv add playwright` both act on the current
    project, not on the isolated environment the tool actually lives in, so
    following them changed nothing and the next run failed identically.

    The installer is read off ``sys.prefix`` rather than guessed, because a
    pipx user handed a ``uv tool`` command is in the same position as before:
    holding a command that does not apply to them. uv is the fallback when
    the layout says nothing (a plain virtualenv, a checkout, an editable
    install), matching what the docs recommend.

    Several extras give one command: doctor on a plain install used to print
    one reinstall per missing extra. On Windows on Arm the command keeps the
    x64 Python markitai runs on (see :func:`_windows_on_arm_python`).
    """
    import sys
    from pathlib import Path

    installer = "pipx" if "pipx" in Path(sys.prefix).parts else "uv tool"
    if installer == "uv tool":
        pin = _windows_on_arm_python()
        python = f" --python {pin}" if pin else ""
        spec = _uv_tool_spec(Path(sys.prefix), extras)
        if spec is not None:
            return f'uv tool install "{spec}"{python} --force'
        return f'uv tool install "markitai[{",".join(sorted(set(extras)))}]"{python} --force'
    return f'{installer} install "markitai[{",".join(sorted(set(extras)))}]" --force'


def _windows_machine_arch() -> str:
    """The machine's architecture on Windows (``ARM64``, ``AMD64``).

    From the registry's machine environment: an x64 process emulated on
    ARM64 sees ``AMD64`` in its own PROCESSOR_ARCHITECTURE (and so does
    Python 3.11's ``platform.machine()``).
    """
    import os
    import sys

    if sys.platform == "win32":
        import winreg

        try:
            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment",
            ) as key:
                return str(winreg.QueryValueEx(key, "PROCESSOR_ARCHITECTURE")[0])
        except OSError:
            pass
    return os.environ.get("PROCESSOR_ARCHITECTURE", "")


def _windows_on_arm_python() -> str | None:
    """The ``--python`` request that keeps an x64 install on Windows on Arm.

    markitai cannot install on a native ARM64 Python (pymupdf-layout has no
    ARM64 build; litellm needs fastuuid and tiktoken compiled with Rust), so
    setup.ps1 puts it on an emulated x64 one. ``uv tool install --force``
    without ``--python`` builds the tool again on whatever Python uv finds
    first, which can be a native one that then fails to install.
    """
    import sys
    import sysconfig

    if sys.platform != "win32" or sysconfig.get_platform() != "win-amd64":
        return None
    if _windows_machine_arch().upper() != "ARM64":
        return None
    return f"cpython-{sys.version_info.major}.{sys.version_info.minor}-windows-x86_64"


def _uv_tool_spec(prefix: object, extras: tuple[str, ...]) -> str | None:
    """markitai's requirement from this uv tool install, plus ``extras``.

    ``uv tool install --force`` replaces the tool with exactly what it is
    given, so "markitai[legacy]" alone drops every extra already installed
    (setup.sh installs several) and, for an install from a checkout, swaps
    it for the PyPI release. uv's receipt records the original requirement.
    """
    import tomllib
    from pathlib import Path

    try:
        receipt = tomllib.loads(
            (Path(str(prefix)) / "uv-receipt.toml").read_text(encoding="utf-8")
        )
        requirement = next(
            r
            for r in receipt["tool"]["requirements"]
            if str(r.get("name", "")).lower() == "markitai"
        )
    except (OSError, ValueError, KeyError, TypeError, StopIteration):
        return None
    merged = sorted({*requirement.get("extras", []), *extras})
    spec = f"markitai[{','.join(merged)}]"
    if requirement.get("directory"):
        return f"{spec} @ {requirement['directory']}"
    if requirement.get("path"):
        return f"{spec} @ {requirement['path']}"
    if requirement.get("git"):
        return f"{spec} @ git+{requirement['git']}"
    if requirement.get("url"):
        return f"{spec} @ {requirement['url']}"
    return spec + str(requirement.get("specifier", ""))


class MissingDependencyError(SelfExplanatoryError, ImportError):
    """A dependency needed for this operation is not installed.

    Covers optional extras (pillow-heif, rapidocr, cairosvg) as well as core
    packages missing from a broken environment. Also an ``ImportError`` so
    existing ``except ImportError`` handlers keep working. The message must
    name the missing package and the exact install command.
    """


class FileTooLargeError(SelfExplanatoryError, ValueError):
    """An input file exceeds the configured size limit.

    Also a ``ValueError`` for backwards compatibility with callers that
    catch the size check by that type.
    """


class ConversionError(SelfExplanatoryError, RuntimeError):
    """A conversion step failed with an already-rendered message.

    Raised when a ``ConversionStepResult`` failure is turned back into an
    exception: its ``error`` string was built for the user (and already
    carries the inner exception's class name when that exception was
    unexpected), so re-prefixing it with ``RuntimeError:`` adds nothing.
    """


class CliInputRejection(SystemExit):
    """An input rejected before conversion could start.

    Subclasses ``SystemExit`` so every existing ``pytest.raises(SystemExit)``
    and exit-code path keeps working, while the carried ``message`` lets the
    ``--json`` envelope explain a run that produced no work item — otherwise
    it would report ``ok: true`` with an empty list and a non-zero exit.

    This stays out of conversion history: an unsupported file is a usage
    error, not a conversion result, and batch discovery never surfaces one.
    """

    def __init__(self, message: str) -> None:
        super().__init__(1)
        self.message = message

    def __str__(self) -> str:
        """Render the reason, not the exit code (``SystemExit(1)`` -> "1")."""
        return self.message
