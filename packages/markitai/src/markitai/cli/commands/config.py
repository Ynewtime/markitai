"""Configuration management CLI commands.

This module provides CLI commands for managing Markitai configuration:
- config list: Show current effective configuration
- config path: Show configuration file paths
- config validate: Validate a configuration file
- config get: Get a configuration value
- config set: Set a configuration value
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import rich_click as click
from rich.syntax import Syntax

from markitai.cli import ui
from markitai.cli.console import get_console
from markitai.cli.i18n import t
from markitai.config import ConfigManager, MarkitaiConfig

console = get_console()

# Sentinel to distinguish "key not found" from "key exists but is null"
_MISSING = object()

_REDACTED = "[REDACTED]"


def _normalize_config_key(key: str) -> str:
    """Normalize snake/camel/header-style keys for policy matching."""
    snake_key = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", key)
    snake_key = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", snake_key)
    return re.sub(r"[^a-z0-9]+", "_", snake_key.lower()).strip("_")


def _is_sensitive_config_key(key: str) -> bool:
    """Return whether a config key conventionally contains secret material."""
    normalized = _normalize_config_key(key)
    parts = set(normalized.split("_"))

    if normalized == "apikey" or normalized.endswith("_apikey"):
        return True
    if "api" in parts and "key" in parts:
        return True
    if "token" in parts:  # Deliberately does not match non-secret ``max_tokens``.
        return True
    return bool(
        parts
        & {
            "authorization",
            "cookie",
            "cookies",
            "credential",
            "credentials",
            "passwd",
            "password",
            "secret",
        }
    )


def _safe_api_base_origin(value: Any) -> Any:
    """Expose only a URL origin, never credentials, paths, query, or fragment."""
    if isinstance(value, str) and value.startswith("env:"):
        return value
    if not isinstance(value, str):
        return _REDACTED
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError:
        return _REDACTED
    if not parsed.scheme or not hostname:
        return _REDACTED
    display_host = f"[{hostname}]" if ":" in hostname else hostname
    if port is not None:
        display_host = f"{display_host}:{port}"
    return f"{parsed.scheme}://{display_host}"


def _redact_header_values(headers: Any) -> Any:
    """Keep header names for diagnosis while hiding every configured value."""
    if not isinstance(headers, dict):
        return _REDACTED
    # Playwright forwards these strings verbatim; unlike api_key/api_token,
    # this field does not resolve ``env:VAR`` references. Treat every value as
    # inline material, including values that merely look like an env reference.
    return dict.fromkeys(headers, _REDACTED)


def _redact_config_secrets(value: Any) -> Any:
    """Recursively redact inline secrets while preserving ``env:VAR`` refs."""
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            normalized = _normalize_config_key(str(key))
            if normalized == "extra_http_headers":
                redacted[key] = _redact_header_values(item)
            elif normalized == "api_base":
                redacted[key] = _safe_api_base_origin(item)
            elif _is_sensitive_config_key(str(key)):
                if isinstance(item, str) and item.startswith("env:"):
                    redacted[key] = item
                else:
                    redacted[key] = _REDACTED
            else:
                redacted[key] = _redact_config_secrets(item)
        return redacted
    if isinstance(value, list):
        return [_redact_config_secrets(item) for item in value]
    return value


def _resolve_field_type(key: str) -> Any:
    """Resolve the declared type for a config key from the Pydantic schema.

    Returns None when the key cannot be resolved (unknown field, dict/list
    containers, non-list indexing, etc.).
    """
    import re
    import types
    from typing import Union, get_args, get_origin

    from pydantic import BaseModel

    def unwrap_optional(annotation: Any) -> Any:
        origin = get_origin(annotation)
        if origin is Union or origin is types.UnionType:
            args = [a for a in get_args(annotation) if a is not type(None)]
            if len(args) == 1:
                return args[0]
        return annotation

    current: Any = MarkitaiConfig
    for part in key.split("."):
        match = re.match(r"^([^\[]+)\[(\d+)\]$", part)
        field_name = match.group(1) if match else part

        if not (isinstance(current, type) and issubclass(current, BaseModel)):
            return None
        field = current.model_fields.get(field_name)
        if field is None:
            return None
        annotation = unwrap_optional(field.annotation)
        if match:
            if get_origin(annotation) is not list:
                return None
            annotation = unwrap_optional(get_args(annotation)[0])
        current = annotation

    return current


def _coerce_value(value: str, key: str) -> bool | int | float | str:
    """Coerce a CLI string value based on the target field's declared type.

    String fields keep the raw string (e.g. api_key "0123456789" stays a
    string); bool fields accept 1/0/true/false; numeric fields parse numbers.
    Falls back to blind bool/int/float inference for unresolvable keys.
    """
    from typing import Literal, get_origin

    target = _resolve_field_type(key)

    if target is str or (target is not None and get_origin(target) is Literal):
        return value
    if target is bool:
        lowered = value.lower()
        if lowered in ("true", "1", "yes", "on"):
            return True
        if lowered in ("false", "0", "no", "off"):
            return False
        return value  # let Pydantic validation report the error
    if target is int:
        try:
            return int(value)
        except ValueError:
            return value
    if target is float:
        try:
            return float(value)
        except ValueError:
            return value

    # Fallback for unresolvable keys: blind bool/int/float/str inference
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


@click.group()
def config() -> None:
    """Configuration management commands.

    View and edit Markitai settings (~/.markitai/config.json or
    ./markitai.json). Keys use dot notation, e.g. llm.enabled.

    Examples:
        markitai config list            # Show effective configuration
        markitai config get llm.enabled # Read a single value
        markitai config set llm.enabled true
        markitai config edit            # Interactive editor
    """


@config.command("list")
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["json", "yaml", "table"], case_sensitive=False),
    default="json",
    help="Output format (json, yaml, or table).",
)
@click.option(
    "--show-secrets",
    is_flag=True,
    help="Show secret values instead of redacting them (unsafe for shared logs).",
)
def config_list(output_format: str, show_secrets: bool) -> None:
    """Show current effective configuration.

    Merges defaults, config file, environment variables — what markitai
    actually uses at runtime.

    Examples:
        markitai config list            # JSON (default)
        markitai config list -f table   # Compact two-column table
        markitai config list -f yaml    # YAML (requires PyYAML)
    """
    manager = ConfigManager()
    cfg = manager.load()

    config_dict = cfg.model_dump(mode="json", exclude_none=True)
    if not show_secrets:
        config_dict = _redact_config_secrets(config_dict)

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
    """Show configuration file paths and precedence.

    Lists all config sources from highest to lowest priority and marks
    the one currently in use.

    Examples:
        markitai config path            # Which config file is in use?
    """
    manager = ConfigManager()
    manager.load()

    ui.title(t("config.title"))

    # Check which config is loaded
    local_loaded = bool(
        manager.config_path and "markitai.json" in str(manager.config_path)
    )
    user_config_path = manager.DEFAULT_USER_CONFIG_DIR / "config.json"
    user_loaded = bool(
        manager.config_path and str(user_config_path) in str(manager.config_path)
    )
    user_config_display = "~/.markitai/config.json"

    # Build rows: (label, annotation)
    rows = [
        (t("config.cli_args"), t("config.highest")),
        (t("config.env_vars"), ""),
        (
            "./markitai.json",
            f"[green]{ui.MARK_SUCCESS} {t('config.loaded')}[/]" if local_loaded else "",
        ),
        (
            user_config_display,
            f"[green]{ui.MARK_SUCCESS} {t('config.loaded')}[/]" if user_loaded else "",
        ),
        (t("config.defaults"), t("config.lowest")),
    ]

    # Align │ column to the longest label
    max_label_len = max(len(label) for label, _ in rows)
    for i, (label, annotation) in enumerate(rows):
        num = i + 1
        padding = " " * (max_label_len - len(label))
        ann = f" {annotation}" if annotation else ""
        console.print(f"  {num}. {label}{padding} [dim]{ui.MARK_LINE}[/]{ann}")

    console.print()

    if manager.config_path:
        ui.success(f"Currently using: {manager.config_path}")
    else:
        ui.warning("Using default configuration (no config file found)")


@config.command("validate")
@click.argument(
    "config_file",
    type=click.Path(exists=True, path_type=Path),
    required=False,
)
def config_validate(config_file: Path | None) -> None:
    """Validate a configuration file.

    Without an argument, validates the currently loaded configuration.

    Examples:
        markitai config validate                # Validate active config
        markitai config validate ./markitai.json
    """
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
    """Get a configuration value.

    KEY uses dot notation; nested sections print as JSON.

    Examples:
        markitai config get llm.enabled        # Single value
        markitai config get llm.model_list     # Whole section as JSON
        markitai config get cache.global_dir
    """
    manager = ConfigManager()
    manager.load()

    value = manager.get(key, _MISSING)
    if value is _MISSING:
        console.print(f"[yellow]Key not found:[/yellow] {key}")
        console.print("[dim]Run 'markitai config list' to see all keys[/dim]")
        raise SystemExit(1)
    if value is None:
        # Key exists but is null (e.g. an unset optional field)
        console.print("null")
        return

    # Serialize Pydantic models to dicts for consistent JSON output
    from pydantic import BaseModel

    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json", exclude_none=True)
    elif isinstance(value, list) and value and isinstance(value[0], BaseModel):
        value = [v.model_dump(mode="json", exclude_none=True) for v in value]

    # Format output
    if isinstance(value, (dict, list)):
        output = json.dumps(value, indent=2, ensure_ascii=False)
        syntax = Syntax(output, "json", theme="monokai", line_numbers=False)
        console.print(syntax)
    else:
        console.print(str(value))


@config.command("set")
@click.argument("key")
@click.argument("value")
def config_set(key: str, value: str) -> None:
    """Set a configuration value.

    Values are validated against the config schema before saving;
    invalid values are rejected and nothing is written.

    Examples:
        markitai config set llm.enabled true
        markitai config set output.dir ./converted
        markitai config set cache.max_size_bytes 104857600
    """
    from pydantic import ValidationError

    manager = ConfigManager()
    manager.load()

    # Parse value based on the target field's declared type
    parsed_value = _coerce_value(value, key)

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

        saved_path = manager.save()
        console.print(f"[green]Set {key} = {parsed_value}[/green]")
        console.print(f"[dim]Saved to {saved_path}[/dim]")

    except SystemExit:
        raise
    except Exception as e:
        console.print(f"[red]Error setting value:[/red] {e}")
        raise SystemExit(1)


@config.command("edit")
def config_edit() -> None:
    """Interactively edit configuration settings.

    Opens a guided editor with arrow-key navigation — no need to
    remember key names.

    Examples:
        markitai config edit
    """
    from markitai.cli.config_editor import run_config_editor

    run_config_editor()


__all__ = ["config"]
