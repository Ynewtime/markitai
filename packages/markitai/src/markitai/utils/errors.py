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
