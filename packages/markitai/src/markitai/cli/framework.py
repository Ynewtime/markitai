"""Custom CLI framework classes for Markitai.

This module contains the custom Click Group class that supports
the main command with arguments and subcommands, with lazy loading
of command modules to minimize startup cost.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

import click
from click import Context

if TYPE_CHECKING:
    pass

# Mapping of command name -> (module_path, attribute_name)
# This allows list_commands() and get_command() to work without importing
# the heavy command modules at module level.
_LAZY_COMMANDS: dict[str, tuple[str, str]] = {
    "cache": ("markitai.cli.commands.cache", "cache"),
    "config": ("markitai.cli.commands.config", "config"),
    "doctor": ("markitai.cli.commands.doctor", "doctor"),
    "init": ("markitai.cli.commands.init", "init"),
}


class MarkitaiGroup(click.Group):
    """Custom Group that supports main command with arguments and subcommands.

    Subcommands are lazily loaded: their modules are only imported when
    the command is actually invoked or when help text needs to be rendered.

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
        "-c",
        "--config",
        "-p",
        "--preset",
        "--no-cache-for",
        "--llm-concurrency",
        "-j",
        "--batch-concurrency",
        "--url-concurrency",
    }

    def list_commands(self, ctx: Context) -> list[str]:
        """Return sorted list of all command names (lazy + eagerly registered)."""
        # Combine lazily-declared names with any eagerly registered commands
        names = set(_LAZY_COMMANDS.keys())
        names.update(super().list_commands(ctx))
        return sorted(names)

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

        module_path, attr_name = spec
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
        ctx.ensure_object(dict)
        skip_next = False
        input_idx = None

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

            # First positional argument
            if arg in known_commands:
                # It's a subcommand - stop looking
                break
            else:
                # It's a file path - store for later use
                ctx.obj["_input_path"] = arg
                input_idx = i
                break

        # Remove INPUT from args so Group doesn't treat it as subcommand
        if input_idx is not None:
            args = args[:input_idx] + args[input_idx + 1 :]

        return super().parse_args(ctx, args)

    def format_usage(
        self,
        ctx: Context,
        formatter: click.HelpFormatter,
    ) -> None:
        """Custom usage line to show INPUT argument."""
        formatter.write_usage(
            ctx.command_path,
            "[OPTIONS] INPUT [COMMAND]",
        )

    def format_help(self, ctx: Context, formatter: click.HelpFormatter) -> None:
        """Custom help formatting to show INPUT argument."""
        # Usage
        self.format_usage(ctx, formatter)

        # Help text
        self.format_help_text(ctx, formatter)

        # Arguments section
        with formatter.section("Arguments"):
            formatter.write_dl(
                [
                    (
                        "INPUT",
                        "File, directory, URL, or .urls file to convert",
                    )
                ]
            )

        # Options (not format_options which may include epilog)
        opts = []
        for param in self.get_params(ctx):
            rv = param.get_help_record(ctx)
            if rv is not None:
                opts.append(rv)
        if opts:
            with formatter.section("Options"):
                formatter.write_dl(opts)

        # Commands
        commands = []
        for name in self.list_commands(ctx):
            cmd = self.get_command(ctx, name)
            if cmd is None or cmd.hidden:
                continue
            commands.append((name, cmd.get_short_help_str(limit=formatter.width)))
        if commands:
            with formatter.section("Commands"):
                formatter.write_dl(commands)
