"""Guard: a config field and the code that reads it must both exist.

``fetch.py`` read ``getattr(config, "auto_proxy", True)`` in three places, but
no config model has ever declared ``auto_proxy`` — the default won every time,
so a switch that looks configurable was permanently on, invisible to
``config list`` and to the JSON schema.

The pattern is legitimate for genuinely optional attributes, so this test does
not ban it; it only requires that the name be declared somewhere in the config
models, which is what makes it reachable by a user.

The mirror image is just as invisible and bit us the same way:
``screenshot.viewport_width`` and ``viewport_height`` were declared, described,
and published in the JSON schema, but nothing ever read them — every capture
used Playwright's own default and setting them did nothing. So the second
guard walks the models and requires each field to be read somewhere.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import get_args

import pytest
from pydantic import BaseModel

from markitai import config as config_module

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC = REPO_ROOT / "packages" / "markitai" / "src" / "markitai"

# Only config-shaped receivers: `config`, `cfg`, `self.config`, `ctx.config`,
# and their dotted sub-sections (`cfg.fetch`, `ctx.config.llm`, ...).
_GETATTR = re.compile(
    r"""getattr\(\s*
        (?:self\.|ctx\.)?(?:config|cfg)(?:\.\w+)*\s*,\s*
        ["'](?P<name>\w+)["']""",
    re.VERBOSE,
)


def _declared_config_fields() -> set[str]:
    names: set[str] = set()
    for value in vars(config_module).values():
        if isinstance(value, type) and issubclass(value, BaseModel):
            names.update(value.model_fields)
    return names


def _getattr_sites() -> list[tuple[str, int, str]]:
    sites: list[tuple[str, int, str]] = []
    for path in sorted(SRC.rglob("*.py")):
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            for match in _GETATTR.finditer(line):
                rel = path.relative_to(REPO_ROOT).as_posix()
                sites.append((rel, number, match.group("name")))
    return sites


def test_the_scan_finds_something_to_check() -> None:
    """A regex that matches nothing would make the real test vacuously pass."""
    assert _declared_config_fields(), "no config models were discovered"


def test_every_config_getattr_names_a_declared_field() -> None:
    declared = _declared_config_fields()
    phantoms = [
        f"{path}:{number} reads config attribute {name!r}"
        for path, number, name in _getattr_sites()
        if name not in declared
    ]
    assert phantoms == [], (
        "these read a config field no model declares, so the fallback always "
        "wins and the setting is unreachable — declare the field or drop the "
        f"getattr: {phantoms}"
    )


# Fields no code reads by attribute because they are forwarded wholesale.
_CONSUMED_BY_NAME = {
    # RouterSettings.model_dump() is handed to litellm's Router, which reads
    # routing_strategy itself.
    "routing_strategy",
}


def _config_field_paths() -> list[tuple[str, str]]:
    """(dotted path, leaf name) for every leaf field in the config models."""
    leaves: list[tuple[str, str]] = []

    def walk(model: type[BaseModel], prefix: str) -> None:
        for name, field in model.model_fields.items():
            annotation = field.annotation
            nested = next(
                (
                    candidate
                    for candidate in (annotation, *(get_args(annotation) or ()))
                    if isinstance(candidate, type) and issubclass(candidate, BaseModel)
                ),
                None,
            )
            if nested is not None:
                walk(nested, f"{prefix}{name}.")
            else:
                leaves.append((f"{prefix}{name}", name))

    walk(config_module.MarkitaiConfig, "")
    return leaves


def _names_read_in_source() -> set[str]:
    """Attribute and string-key reads, from the AST.

    Parsed rather than grepped: a comment mentioning ``screenshot.viewport_width``
    would otherwise satisfy the guard for the field it is apologising about.
    """
    seen: set[str] = set()
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                seen.add(node.attr)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                seen.add(node.value)
    return seen


def test_every_declared_config_field_is_read_somewhere() -> None:
    read = _names_read_in_source()
    unread = [
        dotted
        for dotted, leaf in _config_field_paths()
        if leaf not in _CONSUMED_BY_NAME and leaf not in read
    ]
    assert unread == [], (
        "these config fields are published to users (config list, JSON "
        "schema) but no code reads them, so setting one does nothing — wire "
        f"it up, drop it, or add it to _CONSUMED_BY_NAME with why: {unread}"
    )


@pytest.mark.parametrize("name", ["auto_proxy"])
def test_known_phantom_stays_gone(name: str) -> None:
    """Regression pin for the phantom this guard was written for."""
    assert name not in {n for _, _, n in _getattr_sites()}
