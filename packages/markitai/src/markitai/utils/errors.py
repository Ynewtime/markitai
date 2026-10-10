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


def extra_install_command(extra: str) -> str:
    """The one command that adds an extra to *this* markitai install.

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
    """
    import sys
    from pathlib import Path

    installer = "pipx" if "pipx" in Path(sys.prefix).parts else "uv tool"
    if installer == "uv tool":
        spec = _uv_tool_spec(Path(sys.prefix), extra)
        if spec is not None:
            return f'uv tool install "{spec}" --force'
    return f'{installer} install "markitai[{extra}]" --force'


def _uv_tool_spec(prefix: object, extra: str) -> str | None:
    """markitai's requirement from this uv tool install, plus ``extra``.

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
    extras = sorted({*requirement.get("extras", []), extra})
    spec = f"markitai[{','.join(extras)}]"
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
