"""Custom CLI framework classes for Markitai.

This module contains the custom Click Group class that supports
the main command with arguments and subcommands.
"""

from __future__ import annotations

import click
from click import Context


class MarkitaiGroup(click.Group):
    """Custom Group that supports main command with arguments and subcommands.

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

    def parse_args(self, ctx: Context, args: list[str]) -> list[str]:
        """Parse arguments, detecting if first arg is a subcommand or file path."""
        # Find INPUT: first positional arg that's not:
        # - An option flag (starts with -)
        # - A subcommand
        # - A value for a path option
        ctx.ensure_object(dict)
        skip_next = False
        input_idx = None

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
            if arg in self.commands:
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
