"""Dependency-inversion seams between domain layers and the presentation layer.

Code below the CLI (fetch, batch, ...) must never import ``markitai.cli``;
when it needs the user — a consent prompt, a privacy notice — it goes through
the process-wide :class:`InteractionPort`. The default implementation speaks
plain stdio so library use needs no wiring; the CLI injects a richer
implementation at startup (pausing the live progress display around prompts).
"""

from __future__ import annotations

import sys
from typing import Protocol, runtime_checkable


@runtime_checkable
class InteractionPort(Protocol):
    """User-facing interaction seam for code below the presentation layer."""

    def can_prompt(self) -> bool:
        """True when an interactive question can be asked right now."""
        ...

    def notify(self, message: str) -> None:
        """Deliver a user-facing notice to stderr, bypassing log filtering."""
        ...

    def confirm(
        self, question: str, *, default: bool = False, preamble: str | None = None
    ) -> bool:
        """Ask a yes/no question on stderr and return the answer."""
        ...


class StdioInteraction:
    """Default port: plain click stdio, no live-display awareness."""

    def can_prompt(self) -> bool:
        return sys.stdin.isatty()

    def notify(self, message: str) -> None:
        import click

        click.echo(message, err=True)

    def confirm(
        self, question: str, *, default: bool = False, preamble: str | None = None
    ) -> bool:
        import click

        if preamble:
            click.echo(preamble, err=True)
        return bool(click.confirm(question, default=default, err=True))


_interaction: InteractionPort = StdioInteraction()


def get_interaction() -> InteractionPort:
    """Return the process-wide interaction port."""
    return _interaction


def set_interaction(port: InteractionPort) -> None:
    """Replace the process-wide interaction port (called by the CLI at startup)."""
    global _interaction
    _interaction = port
