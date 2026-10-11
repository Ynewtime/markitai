"""A child process's environment without the variables that force colour.

Rich and rich-click treat a pipe as a terminal when one of these is set and
write ANSI styles into it: ``NO_COLOR`` drops only the colours, so a dim box
still arrives as ``\\x1b[2m╭─\\x1b[0m``. rich-click also forces it under
``GITHUB_ACTIONS``, which every GitHub Actions step has. The conftest drops
the others from the test process but keeps that one, so in-process tests meet
it on CI; a test that reads a markitai child's output starts the child with
:func:`plain_env` and still compares the text with ``click.unstyle``.
"""

from __future__ import annotations

import os

FORCED_COLOUR_ENV = frozenset(
    {
        "FORCE_COLOR",
        "PY_COLORS",
        "CLICOLOR_FORCE",
        "TTY_COMPATIBLE",
        "TTY_INTERACTIVE",
        "GITHUB_ACTIONS",
    }
)


def plain_env(**overrides: str) -> dict[str, str]:
    """``os.environ`` without :data:`FORCED_COLOUR_ENV`, plus *overrides*."""
    env = {k: v for k, v in os.environ.items() if k not in FORCED_COLOUR_ENV}
    env.update(overrides)
    return env
