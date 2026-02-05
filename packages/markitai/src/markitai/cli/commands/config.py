"""Configuration management CLI commands.

This module provides CLI commands for managing Markitai configuration:
- config list: Show current effective configuration
- config path: Show configuration file paths
- config init: Initialize a configuration file
- config validate: Validate a configuration file
- config get: Get a configuration value
- config set: Set a configuration value
"""

from __future__ import annotations

import json
from pathlib import Path

import click
from rich.syntax import Syntax

from markitai.cli import ui
from markitai.cli.console import get_console
from markitai.cli.i18n import t
from markitai.config import ConfigManager, MarkitaiConfig

console = get_console()


@click.group()
def config() -> None:
    """Configuration management commands."""
    pass


@config.command("list")
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["json", "yaml", "table"], case_sensitive=False),
    default="json",
    help="Output format (json, yaml, or table).",
)
def config_list(output_format: str) -> None:
    """Show current effective configuration."""
    manager = ConfigManager()
    cfg = manager.load()

    config_dict = cfg.model_dump(mode="json", exclude_none=True)

    if output_format == "json":
        config_json = json.dumps(config_dict, indent=2, ensure_ascii=False)
        syntax = Syntax(config_json, "json", theme="monokai", line_numbers=False)
        console.print(syntax)
    elif output_format == "yaml":
        try:
            import yaml

            config_yaml = yaml.dump(
                config_dict,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )
            syntax = Syntax(config_yaml, "yaml", theme="monokai", line_numbers=False)
            console.print(syntax)
        except ImportError:
            console.print("[red]YAML output requires PyYAML: uv add pyyaml[/red]")
            raise SystemExit(1)
    elif output_format == "table":
        from rich.table import Table

        table = Table(title="Markitai Configuration", show_header=True)
        table.add_column("Section", style="cyan")
        table.add_column("Key", style="green")
        table.add_column("Value", style="white")

        for section, values in config_dict.items():
            if isinstance(values, dict):
                for key, value in values.items():
                    if isinstance(value, (dict, list)):
                        value = json.dumps(value, ensure_ascii=False)
                    table.add_row(section, key, str(value))
            else:
                table.add_row("", section, str(values))

        console.print(table)


@config.command("path")
def config_path_cmd() -> None:
    """Show configuration file paths."""
    manager = ConfigManager()
    manager.load()

    ui.title(t("config.title"))

    console.print(
        f"  1. {t('config.cli_args')}      [dim]{ui.MARK_LINE} {t('config.highest')}[/]"
    )
    console.print(f"  2. {t('config.env_vars')}  [dim]{ui.MARK_LINE}[/]")

    # Check local config file
    local_status = ""
    if manager.config_path and "markitai.json" in str(manager.config_path):
        local_status = f" [green]{ui.MARK_SUCCESS} {t('config.loaded')}[/]"
    console.print(f"  3. ./markitai.json{local_status}")

    # Check user config file
    user_config_path = manager.DEFAULT_USER_CONFIG_DIR / "config.json"
    user_status = ""
    if manager.config_path and str(user_config_path) in str(manager.config_path):
        user_status = f" [green]{ui.MARK_SUCCESS} {t('config.loaded')}[/]"
    console.print(f"  4. {user_config_path}{user_status}")

    console.print(
        f"  5. {t('config.defaults')}        [dim]{ui.MARK_LINE} {t('config.lowest')}[/]"
    )
    console.print()

    if manager.config_path:
        ui.success(f"Currently using: {manager.config_path}")
    else:
        ui.warning("Using default configuration (no config file found)")


@config.command("init")
@click.option(
    "--output",
    "-o",
    "output_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Output path for configuration file.",
)
@click.option(
    "--yes",
    "-y",
    is_flag=True,
    default=False,
    help="Overwrite existing config without confirmation.",
)
def config_init(output_path: Path | None, yes: bool) -> None:
    """Initialize a configuration file with defaults."""
    manager = ConfigManager()

    if output_path is None:
        output_path = manager.DEFAULT_USER_CONFIG_DIR / "config.json"
    elif output_path.is_dir():
        # User passed a directory, append default filename
        output_path = output_path / "markitai.json"

    # Check if file exists (not directory)
    if output_path.exists() and output_path.is_file():
        if not yes and not click.confirm(f"{output_path} already exists. Overwrite?"):
            raise click.Abort()

    # Save minimal template config (essential fields only)
    saved_path = manager.save(output_path, minimal=True)
    ui.summary(f"{t('config.created')}: {saved_path}")
    console.print()
    console.print("Edit this file to customize your settings.")
    console.print(
        "[dim]Note: max_tokens, supports_vision are auto-detected from litellm.[/dim]"
    )
    console.print("Run 'markitai config list' to see the current configuration.")


@config.command("validate")
@click.argument(
    "config_file",
    type=click.Path(exists=True, path_type=Path),
    required=False,
)
def config_validate(config_file: Path | None) -> None:
    """Validate a configuration file."""
    manager = ConfigManager()

    try:
        manager.load(config_path=config_file)

        ui.summary(t("config.valid"))

        if manager.config_path:
            console.print(f"[dim]Validated: {manager.config_path}[/dim]")

    except Exception as e:
        console.print(f"[red]Configuration error:[/red] {e}")
        raise SystemExit(2)


@config.command("get")
@click.argument("key")
def config_get(key: str) -> None:
    """Get a configuration value."""
    manager = ConfigManager()
    manager.load()

    value = manager.get(key)
    if value is None:
        console.print(f"[yellow]Key not found:[/yellow] {key}")
        raise SystemExit(1)

    # Format output
    if isinstance(value, (dict, list)):
        console.print(json.dumps(value, indent=2, ensure_ascii=False))
    else:
        console.print(str(value))


@config.command("set")
@click.argument("key")
@click.argument("value")
def config_set(key: str, value: str) -> None:
    """Set a configuration value."""
    from pydantic import ValidationError

    manager = ConfigManager()
    manager.load()

    # Parse value
    parsed_value: bool | int | float | str
    if value.lower() in ("true", "false"):
        parsed_value = value.lower() == "true"
    else:
        try:
            parsed_value = int(value)
        except ValueError:
            try:
                parsed_value = float(value)
            except ValueError:
                parsed_value = value

    # Store old config for rollback on validation failure
    old_config_dict = manager.config.model_dump()

    try:
        manager.set(key, parsed_value)

        # Validate the entire config using Pydantic
        try:
            MarkitaiConfig.model_validate(manager.config.model_dump())
        except ValidationError as ve:
            # Rollback to old config
            manager._config = MarkitaiConfig.model_validate(old_config_dict)
            # Format validation errors nicely
            errors = []
            for err in ve.errors():
                loc = ".".join(str(x) for x in err["loc"])
                msg = err["msg"]
                errors.append(f"  {loc}: {msg}")
            console.print(f"[red]Invalid value for '{key}':[/red]")
            for error in errors:
                console.print(f"[red]{error}[/red]")
            raise SystemExit(1)

        manager.save()
        console.print(f"[green]Set {key} = {parsed_value}[/green]")

    except SystemExit:
        raise
    except Exception as e:
        console.print(f"[red]Error setting value:[/red] {e}")
        raise SystemExit(1)


__all__ = ["config"]
