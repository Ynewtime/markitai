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
