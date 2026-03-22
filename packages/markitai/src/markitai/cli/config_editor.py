"""Interactive config editor — schema introspection and edit loop.

Walks the Pydantic MarkitaiConfig model tree to extract a flat list of
editable scalar settings, then presents them as a searchable questionary
select list for interactive editing.
"""

from __future__ import annotations

import types as _types
import typing
from typing import Any, get_args, get_origin

from pydantic import BaseModel
from pydantic.fields import FieldInfo
from questionary import Choice

from markitai.config import MarkitaiConfig

_SCALAR_TYPES = (str, int, float, bool, type(None))

# Sections to skip entirely (too complex or too advanced for inline editing)
_SKIP_SECTIONS = {"presets", "prompts", "domain_profiles"}

# Sentinel to distinguish "user cancelled" from "user set value to None"
_CANCEL = object()


def extract_editable_settings(cfg: MarkitaiConfig) -> list[dict[str, Any]]:
    """Walk the config model and return a flat list of editable settings.

    Each entry is a dict with keys:
        key: dot-separated path (e.g. "output.dir")
        value: current value
        field_type: "str", "int", "float", "bool", "literal"
        description: human-readable description or ""
        choices: list of valid values (only for "literal" type)

    Complex types (lists, dicts, nested models) are skipped.
    """
    settings: list[dict[str, Any]] = []
    _walk_model(cfg, "", type(cfg).model_fields, settings)
    return settings


def _walk_model(
    obj: BaseModel,
    prefix: str,
    fields: dict[str, FieldInfo],
    settings: list[dict[str, Any]],
) -> None:
    """Recursively walk model fields, collecting scalar settings."""
    for name, field_info in fields.items():
        if name in _SKIP_SECTIONS:
            continue

        key = f"{prefix}{name}" if prefix else name
        value = getattr(obj, name, None)
        annotation = field_info.annotation

        # Unwrap Optional[X] / X | None → X
        origin = get_origin(annotation)
        args = get_args(annotation)
        if (origin is typing.Union or origin is _types.UnionType) and type(
            None
        ) in args:
            non_none = [a for a in args if a is not type(None)]
            if len(non_none) == 1:
                annotation = non_none[0]
                origin = get_origin(annotation)
                args = get_args(annotation)

        # Nested BaseModel → recurse
        if isinstance(value, BaseModel):
            _walk_model(value, f"{key}.", type(value).model_fields, settings)
            continue

        # Skip complex types (list, dict) — check both value AND annotation
        if isinstance(value, (list, dict)):
            continue
        if origin in (list, dict):
            continue
        # Skip None values where the unwrapped annotation is still a BaseModel
        if value is None and annotation is not None:
            try:
                if isinstance(annotation, type) and issubclass(annotation, BaseModel):
                    continue
            except TypeError:
                pass

        # Determine field_type
        description = ""
        if field_info.description:
            description = field_info.description
        elif field_info.title:
            description = field_info.title

        if origin is typing.Literal:
            choices = list(args)
            settings.append(
                {
                    "key": key,
                    "value": value,
                    "field_type": "literal",
                    "description": description,
                    "choices": choices,
                }
            )
        elif annotation is bool or (isinstance(value, bool)):
            settings.append(
                {
                    "key": key,
                    "value": value,
                    "field_type": "bool",
                    "description": description,
                }
            )
        elif annotation is int:
            settings.append(
                {
                    "key": key,
                    "value": value,
                    "field_type": "int",
                    "description": description,
                }
            )
        elif annotation is float:
            settings.append(
                {
                    "key": key,
                    "value": value,
                    "field_type": "float",
                    "description": description,
                }
            )
        elif annotation is str or isinstance(value, str):
            settings.append(
                {
                    "key": key,
                    "value": value,
                    "field_type": "str",
                    "description": description,
                }
            )
        # else: skip unknown types


def format_display_value(value: Any) -> str:
    """Format a config value for display in the choice list."""
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def build_choices(settings: list[dict[str, Any]]) -> list[Choice]:
    """Build questionary Choice objects from settings list."""
    max_key_len = max(len(s["key"]) for s in settings) if settings else 20
    max_val_len = (
        max(len(format_display_value(s["value"])) for s in settings) if settings else 10
    )
    max_val_len = min(max_val_len, 30)

    choices: list[Choice] = []
    for s in settings:
        key = s["key"]
        val_str = format_display_value(s["value"])
        padded_key = key.ljust(max_key_len)
        padded_val = val_str.ljust(max_val_len)
        title = f"{padded_key}  {padded_val}"
        choices.append(
            Choice(
                title=title,
                value=s["key"],
                description=s.get("description", ""),
            )
        )
    return choices


def _prompt_new_value(setting: dict[str, Any]) -> Any:
    """Prompt the user for a new value based on field type.

    Returns the new value, or _CANCEL if the user cancelled (Esc / Ctrl-C).
    """
    import questionary

    key = setting["key"]
    current = setting["value"]
    field_type = setting["field_type"]

    if field_type == "bool":
        result = questionary.confirm(
            f"  {key}",
            default=current if current is not None else False,
        ).ask()
        return _CANCEL if result is None else result

    if field_type == "literal":
        choices = setting.get("choices", [])
        result = questionary.select(
            f"  {key}",
            choices=[str(c) for c in choices],
            default=str(current) if current is not None else None,
        ).ask()
        return _CANCEL if result is None else result

    if field_type == "int":
        result = questionary.text(
            f"  {key}",
            default=str(current) if current is not None else "",
            validate=lambda v: (
                v == ""
                or (v.lstrip("-").isdigit() and v != "-")
                or "Must be an integer"
            ),
        ).ask()
        if result is None:
            return _CANCEL
        return int(result) if result else current

    if field_type == "float":

        def _validate_float(v: str) -> bool | str:
            if v == "":
                return True
            try:
                float(v)
                return True
            except ValueError:
                return "Must be a number"

        result = questionary.text(
            f"  {key}",
            default=str(current) if current is not None else "",
            validate=_validate_float,
        ).ask()
        if result is None:
            return _CANCEL
        return float(result) if result else current

    # str fallback
    result = questionary.text(
        f"  {key}",
        default=str(current) if current is not None else "",
    ).ask()
    return _CANCEL if result is None else result


def run_config_editor() -> None:
    """Run the interactive config editor loop."""
    import questionary
    from rich.console import Console

    from markitai.config import ConfigManager

    console = Console(stderr=True)
    manager = ConfigManager()
    manager.load()
    cfg = manager.config

    console.print()
    config_path = manager.config_path or "defaults"
    console.print(f"  [bold]Markitai Configuration[/]  [dim]({config_path})[/]")
    console.print()

    while True:
        settings = extract_editable_settings(cfg)
        choices = build_choices(settings)

        selected_key = questionary.select(
            "Select a setting to edit:",
            choices=choices,
            use_search_filter=True,
            instruction="(↑↓ move · type to filter · Esc to exit)",
        ).ask()

        if selected_key is None:
            break

        setting = next((s for s in settings if s["key"] == selected_key), None)
        if setting is None:
            continue

        new_value = _prompt_new_value(setting)
        if new_value is _CANCEL:
            continue

        try:
            manager.set(selected_key, new_value)
            manager.save()
            console.print(
                f"  [green]✓[/] {selected_key} = {format_display_value(new_value)}"
            )
            cfg = manager.config
        except Exception as e:
            console.print(f"  [red]✗[/] Error: {e}")

    console.print()
