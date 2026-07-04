"""Custom CLI framework classes for Markitai.

This module contains the custom Click Group class that supports
the main command with arguments and subcommands, with lazy loading
of command modules to minimize startup cost.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import rich_click as click
from click import Context

# Mapping of command name -> (module_path, attribute_name, short_help)
# This allows list_commands() and get_command() to work without importing
# the heavy command modules at module level.  The short_help string is
# used by format_help() so that displaying --help never triggers imports.
_LAZY_COMMANDS: dict[str, tuple[str, str, str]] = {
    "auth": (
        "markitai.cli.commands.auth",
        "auth",
        "Authentication helpers for local providers.",
    ),
    "cache": ("markitai.cli.commands.cache", "cache", "Cache management commands."),
    "config": (
        "markitai.cli.commands.config",
        "config",
        "Configuration management commands.",
    ),
    "doctor": (
        "markitai.cli.commands.doctor",
        "doctor",
        "Check system health, dependencies, and authentication status.",
    ),
    "init": (
        "markitai.cli.commands.init",
        "init",
        "Initialize Markitai configuration.",
    ),
}


class MarkitaiGroup(click.RichGroup):
    """Custom Group that supports main command with arguments and subcommands.

    Subcommands are lazily loaded: their modules are only imported when
    the command is actually invoked, not when help text is rendered.

    This allows:
        markitai document.docx --llm          # Convert file (main command)
        markitai urls.urls -o out             # URL list batch (.urls auto-detected)
        markitai config list                  # Subcommand
        markitai --preset rich document.pdf   # Options before INPUT
    """

    # Options that take a value argument (so we skip their values when looking for INPUT)
    _OPTIONS_WITH_VALUES = {
        "-o",
        "--output",
        "-b",
        "--backend",
        "-c",
        "--config",
        "--config-json",
        "-p",
        "--preset",
        "--no-cache-for",
        "--llm-concurrency",
        "-j",
        "--batch-concurrency",
        "--url-concurrency",
        "-g",
        "--glob",
        "--max-depth",
        "-s",
        "--strategy",
    }

    def list_commands(self, ctx: Context) -> list[str]:
        """Return sorted list of all command names (lazy + eagerly registered)."""
        # Combine lazily-declared names with any eagerly registered commands
        names = set(_LAZY_COMMANDS.keys())
        names.update(super().list_commands(ctx))
        return sorted(names)

    def format_help(
        self, ctx: click.RichContext, formatter: click.RichHelpFormatter
    ) -> None:
        """Render help with lazy-command stubs so --help stays fast.

        Rich rendering resolves every subcommand for its short help; a flag
        makes get_command return lightweight stubs instead of importing the
        real command modules (markitai.cli.commands.cache alone costs ~0.7s).
        """
        self._rendering_help = True
        try:
            super().format_help(ctx, formatter)
        finally:
            self._rendering_help = False

    def get_command(self, ctx: Context, cmd_name: str) -> click.Command | None:
        """Lazily import and return the command, or fall back to eager lookup."""
        # Check eagerly registered commands first (allows runtime add_command)
        cmd = super().get_command(ctx, cmd_name)
        if cmd is not None:
            return cmd

        # Lazy import
        spec = _LAZY_COMMANDS.get(cmd_name)
        if spec is None:
            return None

        if getattr(self, "_rendering_help", False):
            # Help rendering only needs the name + short help — return a
            # stub instead of importing the command module
            _module, _attr, short_help = spec
            return click.RichCommand(
                name=cmd_name, help=short_help, short_help=short_help, params=[]
            )

        module_path, attr_name, _help = spec
        mod = importlib.import_module(module_path)
        cmd = getattr(mod, attr_name)
        # Cache so subsequent calls don't re-import
        self.add_command(cmd, cmd_name)
        return cmd

    def parse_args(self, ctx: Context, args: list[str]) -> list[str]:
        """Parse arguments, detecting if first arg is a subcommand or file path."""
        # Find INPUT: first positional arg that's not:
        # - An option flag (starts with -)
        # - A subcommand
        # - A value for a path option
        #
        # After INPUT is found, keep scanning: a later subcommand token means
        # the invocation is ambiguous (e.g. `markitai note.txt config list`)
        # and must fail loudly instead of silently dropping INPUT.
        ctx.ensure_object(dict)
        skip_next = False
        input_idx: int | None = None
        input_token: str | None = None

        # Use the full set of known command names (lazy + eager) for detection.
        # We use _LAZY_COMMANDS keys + self.commands keys to avoid importing
        # the command modules just to check if an arg is a subcommand name.
        known_commands = set(_LAZY_COMMANDS.keys()) | set(self.commands.keys())

        for i, arg in enumerate(args):
            if skip_next:
                skip_next = False
                continue

            # Check if this is an option that takes a value
            if arg in self._OPTIONS_WITH_VALUES or arg.startswith(
                tuple(f"{opt}=" for opt in self._OPTIONS_WITH_VALUES)
            ):
                if "=" not in arg:
                    skip_next = True  # Next arg is the option's value
                continue

            if arg.startswith("-"):
                # Other options (flags or boolean options)
                continue

            # Positional argument
            if arg in known_commands:
                if input_token is not None and not ctx.resilient_parsing:
                    # Both an INPUT and a subcommand were given - ambiguous
                    raise click.UsageError(
                        "Cannot mix INPUT with a subcommand. "
                        f"Run 'markitai {input_token}' to convert, "
                        f"or 'markitai {arg} ...' for the subcommand."
                    )
                # It's a subcommand - stop looking. If a file with the same
                # name exists in cwd, the subcommand still wins; hint on
                # stderr how to convert the file instead.
                if not ctx.resilient_parsing and Path(arg).is_file():
                    click.echo(
                        f"Note: a file named '{arg}' exists — "
                        f"to convert it, use './{arg}'.",
                        err=True,
                    )
                break

            if input_token is None:
                # It's a file path - store for later use, but keep scanning
                # so a trailing subcommand token is detected as ambiguous
                ctx.obj["_input_path"] = arg
                input_token = arg
                input_idx = i
                continue

            # Second non-command positional: leave it for click to reject
            break

        # Remove INPUT from args so Group doesn't treat it as subcommand
        if input_idx is not None:
            args = args[:input_idx] + args[input_idx + 1 :]

        return super().parse_args(ctx, args)

    def collect_usage_pieces(self, ctx: Context) -> list[str]:
        """Show INPUT in the usage line (it is parsed manually in parse_args,
        not declared as a click Argument)."""
        return ["[OPTIONS]", "INPUT", "[COMMAND]"]
