"""Configuration models validated by the native Rust configuration contract.

The adapter preserves ordinary model, list and dictionary access without a
Python validation dependency. Assignment to a declared field is intentionally
unvalidated, matching the original models; construction and conversion validate.
"""

from __future__ import annotations

import copy
import json
import os
from collections.abc import Mapping
from enum import Enum
from functools import lru_cache
from pathlib import Path

from . import _native

# `Any` only appears in annotations, which are strings here; keep `typing` out
# of the import.
TYPE_CHECKING = False
if TYPE_CHECKING:
    from typing import Any

_SCHEMA = json.loads(_native.config_json("{}", schema=True))
_MODELS: dict[str, type[_Section]] = {}
_MISSING = object()


def _json_value(value: Any) -> Any:
    if isinstance(value, _Section):
        return value.model_dump(mode="json")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    raise TypeError(f"Configuration value of type {type(value).__name__} is not JSON compatible")


def _node(name: str) -> dict[str, Any]:
    return _SCHEMA if name == "MarkitaiConfig" else _SCHEMA["$defs"][name]


def _materialize(value: Any, node: dict[str, Any], supplied: Any = _MISSING) -> Any:
    if value is None:
        return None
    if "$ref" in node:
        name = node["$ref"].rsplit("/", 1)[-1]
        if isinstance(supplied, _MODELS[name]):
            return supplied
        return _MODELS[name]._from_normalized(value, supplied)
    if "anyOf" in node:
        for branch in node["anyOf"]:
            if branch.get("type") != "null":
                return _materialize(value, branch, supplied)
    if isinstance(value, list):
        previous = supplied if isinstance(supplied, (list, tuple)) else []
        return [_materialize(item, node.get("items", {}), previous[i] if i < len(previous) else _MISSING)
                for i, item in enumerate(value)]
    if isinstance(value, dict):
        child = node.get("additionalProperties", {})
        if not isinstance(child, dict):
            child = {}
        previous = supplied if isinstance(supplied, Mapping) else {}
        return {key: _materialize(item, child, previous.get(key, _MISSING)) for key, item in value.items()}
    return value


def _filter_child(selection: Any, key: Any, length: int | None = None) -> tuple[bool, Any]:
    if selection is None:
        return False, None
    if selection is True or selection is Ellipsis:
        return True, True
    aliases = [key]
    if length is not None:
        aliases.extend([key - length, "__all__"])
    if isinstance(selection, Mapping):
        for alias in aliases:
            if alias in selection:
                return True, selection[alias]
    elif isinstance(selection, (set, frozenset)):
        if any(alias in selection for alias in aliases):
            return True, True
    else:
        raise TypeError("include and exclude must be a set or dictionary")
    return False, None


@lru_cache(maxsize=None)
def _default_for(model: str, key: str) -> Any:
    node = _node(model)["properties"][key]
    if "default" not in node:
        return _MISSING
    default = node["default"]
    if "$ref" in node:
        name = node["$ref"].rsplit("/", 1)[-1]
        return json.loads(_native.config_json(json.dumps(default), model=name))
    return default


def _dump(value: Any, include: Any, exclude: Any, options: dict[str, Any]) -> Any:
    if isinstance(value, _Section):
        output = {}
        fields = _node(type(value).__name__)["properties"]
        for key in fields:
            if key not in value._data:
                continue
            item = value._data[key]
            if options["exclude_unset"] and key not in value._fields_set:
                continue
            if options["exclude_none"] and item is None:
                continue
            selected, child_include = _filter_child(include, key)
            if include is not None and not selected:
                continue
            rejected, child_exclude = _filter_child(exclude, key)
            if rejected and (child_exclude is True or child_exclude is Ellipsis):
                continue
            if options["exclude_defaults"]:
                default = _default_for(type(value).__name__, key)
                plain = _dump(item, None, None, {**options, "exclude_defaults": False, "exclude_unset": False, "exclude_none": False})
                if default is not _MISSING and plain == default:
                    continue
            output[key] = _dump(item, None if child_include is True or child_include is Ellipsis else child_include,
                                child_exclude, options)
        return output
    if isinstance(value, (Mapping, list, tuple)):
        is_map = isinstance(value, Mapping)
        output = {} if is_map else []
        for key, item in value.items() if is_map else enumerate(value):
            selected, child_include = _filter_child(include, key, None if is_map else len(value))
            if include is not None and not selected:
                continue
            rejected, child_exclude = _filter_child(exclude, key, None if is_map else len(value))
            if rejected and (child_exclude is True or child_exclude is Ellipsis):
                continue
            item = _dump(item, None if child_include is True or child_include is Ellipsis else child_include,
                         child_exclude, options)
            if is_map:
                output[key] = item
            else:
                output.append(item)
        return tuple(output) if isinstance(value, tuple) and options["mode"] == "python" else output
    if options["mode"] == "json":
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, Path):
            return str(value)
        if value is not None and not isinstance(value, (str, int, float, bool)):
            fallback = options.get("fallback")
            return fallback(value) if fallback else _json_value(value)
    return copy.deepcopy(value)


def _check_options(value: Any, node: dict[str, Any], strict: bool, forbid_extra: bool) -> None:
    if "$ref" in node:
        node = _node(node["$ref"].rsplit("/", 1)[-1])
    if "anyOf" in node:
        if value is None and any(branch.get("type") == "null" for branch in node["anyOf"]):
            return
        node = next(branch for branch in node["anyOf"] if branch.get("type") != "null")
        return _check_options(value, node, strict, forbid_extra)
    if isinstance(value, _Section):
        value = value.model_dump()
    kind = node.get("type")
    allowed = {"boolean": type(value) is bool, "integer": type(value) is int,
               "number": type(value) in (int, float), "string": isinstance(value, str),
               "object": isinstance(value, Mapping), "array": isinstance(value, list), "null": value is None}
    if strict and kind in allowed and not allowed[kind]:
        raise ValueError(f"Expected a strict {kind} configuration value")
    if isinstance(value, Mapping):
        properties = node.get("properties")
        if properties is not None:
            if forbid_extra and set(value) - properties.keys():
                raise ValueError("Unknown configuration fields: " + ", ".join(sorted(set(value) - properties.keys())))
            for key, item in value.items():
                if key in properties:
                    _check_options(item, properties[key], strict, forbid_extra)
        elif isinstance(node.get("additionalProperties"), dict):
            for item in value.values():
                _check_options(item, node["additionalProperties"], strict, forbid_extra)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _check_options(item, node.get("items", {}), strict, forbid_extra)


class _Section:
    def __init__(self, **values: Any) -> None:
        supplied = json.loads(json.dumps(values, default=_json_value))
        normalized = json.loads(_native.config_json(json.dumps(supplied), model=type(self).__name__))
        self._initialize(normalized, values)

    def _initialize(self, data: dict[str, Any], supplied: Any = _MISSING) -> None:
        if isinstance(supplied, _Section):
            supplied = supplied._data
        supplied = supplied if isinstance(supplied, Mapping) else {}
        properties = _node(type(self).__name__)["properties"]
        object.__setattr__(self, "_data", {key: _materialize(value, properties.get(key, {}), supplied.get(key, _MISSING))
                                           for key, value in data.items()})
        object.__setattr__(self, "_fields_set", set(supplied) & properties.keys())

    @classmethod
    def _from_normalized(cls, data: dict[str, Any], supplied: Any = _MISSING) -> _Section:
        result = object.__new__(cls)
        result._initialize(data, supplied)
        return result

    def __getattr__(self, key: str) -> Any:
        try:
            return object.__getattribute__(self, "_data")[key]
        except KeyError:
            raise AttributeError(key) from None

    def __setattr__(self, key: str, value: Any) -> None:
        if key.startswith("_"):
            object.__setattr__(self, key, value)
            return
        if key not in _node(type(self).__name__)["properties"]:
            raise ValueError(f'"{type(self).__name__}" object has no field "{key}"')
        self._data[key] = value
        self._fields_set.add(key)

    def __delattr__(self, key: str) -> None:
        if key in self._data:
            del self._data[key]
        else:
            object.__delattr__(self, key)

    def __iter__(self):
        return iter(self._data.items())

    def __repr__(self) -> str:
        return f"{type(self).__name__}(" + ", ".join(f"{key}={value!r}" for key, value in self._data.items()) + ")"

    def __eq__(self, other: Any) -> bool:
        return type(self) is type(other) and self._data == other._data

    def __deepcopy__(self, memo: dict[int, Any]) -> _Section:
        result = object.__new__(type(self))
        memo[id(self)] = result
        object.__setattr__(result, "_data", copy.deepcopy(self._data, memo))
        object.__setattr__(result, "_fields_set", self._fields_set.copy())
        return result

    @property
    def model_fields_set(self) -> set[str]:
        return self._fields_set

    def model_dump(self, *, mode: str = "python", include: Any = None, exclude: Any = None,
                   exclude_unset: bool = False, exclude_defaults: bool = False, exclude_none: bool = False,
                   by_alias: bool | None = None, round_trip: bool = False, warnings: Any = True,
                   context: Any = None, fallback: Any = None, serialize_as_any: bool = False) -> dict[str, Any]:
        if mode not in ("python", "json"):
            raise ValueError("mode must be 'python' or 'json'")
        return _dump(self, include, exclude, {"mode": mode, "exclude_unset": exclude_unset,
                     "exclude_defaults": exclude_defaults, "exclude_none": exclude_none, "fallback": fallback})

    def model_dump_json(self, *, indent: int | None = None, ensure_ascii: bool = False, **kwargs: Any) -> str:
        return json.dumps(self.model_dump(mode="json", **kwargs), ensure_ascii=ensure_ascii, indent=indent,
                          separators=(",", ":") if indent is None else None)

    def model_copy(self, *, update: Mapping[str, Any] | None = None, deep: bool = False) -> _Section:
        result = copy.deepcopy(self) if deep else object.__new__(type(self))
        if not deep:
            object.__setattr__(result, "_data", self._data.copy())
            object.__setattr__(result, "_fields_set", self._fields_set.copy())
        if update:
            result._data.update(update)
            result._fields_set.update(update)
        return result

    @classmethod
    def model_validate(cls, value: Any, *, strict: bool | None = None, extra: str | None = None,
                       from_attributes: bool | None = None, context: Any = None,
                       by_alias: bool | None = None, by_name: bool | None = None) -> _Section:
        if isinstance(value, cls):
            return value
        if by_alias is False and by_name is not True:
            raise ValueError("At least one of by_alias or by_name must be True")
        if extra not in (None, "ignore", "forbid"):
            raise NotImplementedError("Configuration models support extra='ignore' or 'forbid'")
        if from_attributes and not isinstance(value, Mapping):
            value = {key: getattr(value, key) for key in _node(cls.__name__)["properties"] if hasattr(value, key)}
        if not isinstance(value, Mapping):
            raise ValueError(f"{cls.__name__} requires a dictionary or an instance of that model")
        _check_options(value, _node(cls.__name__), bool(strict), extra == "forbid")
        return cls(**value)

    @classmethod
    def model_validate_json(cls, value: str | bytes | bytearray, **kwargs: Any) -> _Section:
        return cls.model_validate(json.loads(value), **kwargs)

    @classmethod
    def model_json_schema(cls, *, by_alias: bool = True, ref_template: str = "#/$defs/{model}",
                          schema_generator: Any = None, mode: str = "validation") -> dict[str, Any]:
        if schema_generator is not None:
            raise NotImplementedError("Custom schema generators require Pydantic")
        if mode not in ("validation", "serialization"):
            raise ValueError("mode must be 'validation' or 'serialization'")
        result = copy.deepcopy(_node(cls.__name__))
        if cls.__name__ != "MarkitaiConfig":
            result["$defs"] = copy.deepcopy(_SCHEMA["$defs"])
        if ref_template != "#/$defs/{model}":
            def rewrite(value: Any) -> None:
                if isinstance(value, dict):
                    if "$ref" in value:
                        value["$ref"] = ref_template.format(model=value["$ref"].rsplit("/", 1)[-1])
                    for item in value.values():
                        rewrite(item)
                elif isinstance(value, list):
                    for item in value:
                        rewrite(item)
            rewrite(result)
        return result


class MarkitaiConfig(_Section):
    """Mutable configuration with native defaults and construction validation."""


_MODELS["MarkitaiConfig"] = MarkitaiConfig
for _name, _definition in _SCHEMA["$defs"].items():
    if "properties" in _definition:
        _MODELS[_name] = type(_name, (_Section,), {"__module__": __name__, "__doc__": f"Native-validated {_name} configuration."})
        globals()[_name] = _MODELS[_name]


class EnvVarNotFoundError(ValueError):
    def __init__(self, var_name: str) -> None:
        self.var_name = var_name
        super().__init__(f"Environment variable {var_name} is not set")


def resolve_env_value(value: str, strict: bool = True) -> str | None:
    if isinstance(value, str) and value.startswith("env:"):
        variable = value[4:]
        if variable not in os.environ:
            if strict:
                raise EnvVarNotFoundError(variable)
            return None
        return os.environ[variable]
    return value


def _resolved_api_key(self: _Section, strict: bool = True) -> str | None:
    return resolve_env_value(self.api_key, strict) if self.api_key else None


def _resolved_api_base(self: _Section, strict: bool = True) -> str | None:
    return resolve_env_value(self.api_base, strict) if self.api_base else None


def _jina_key(self: _Section, strict: bool = False) -> str | None:
    return resolve_env_value(self.api_key, strict) if self.api_key else os.environ.get("JINA_API_KEY")


def _cloudflare_token(self: _Section, strict: bool = False) -> str | None:
    return resolve_env_value(self.api_token, strict) if self.api_token else os.environ.get("CLOUDFLARE_API_TOKEN")


def _cloudflare_account(self: _Section, strict: bool = False) -> str | None:
    return resolve_env_value(self.account_id, strict) if self.account_id else os.environ.get("CLOUDFLARE_ACCOUNT_ID")


LiteLLMParams.get_resolved_api_key = _resolved_api_key
LiteLLMParams.get_resolved_api_base = _resolved_api_base
JinaConfig.get_resolved_api_key = _jina_key
CloudflareConfig.get_resolved_api_token = _cloudflare_token
CloudflareConfig.get_resolved_account_id = _cloudflare_account
PRESET_NAMES = ("minimal", "standard", "rich")
BUILTIN_PRESETS = {"rich": PresetConfig(llm=True, alt=True, desc=True, screenshot=True),
                  "standard": PresetConfig(llm=True, alt=True, desc=True), "minimal": PresetConfig()}
__all__ = list(_MODELS) + ["EnvVarNotFoundError", "resolve_env_value", "PRESET_NAMES", "BUILTIN_PRESETS"]
