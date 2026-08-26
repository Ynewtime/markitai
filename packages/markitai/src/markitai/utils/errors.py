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
    return f'{installer} install "markitai[{extra}]" --force'


class MissingDependencyError(SelfExplanatoryError, ImportError):
    """A dependency needed for this operation is not installed.

    Covers optional extras (kreuzberg, pillow-heif, rapidocr) as well as core
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
