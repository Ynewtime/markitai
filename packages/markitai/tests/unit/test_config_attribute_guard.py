"""Guard: `getattr(config, "x", default)` must name a field that exists.

``fetch.py`` read ``getattr(config, "auto_proxy", True)`` in three places, but
no config model has ever declared ``auto_proxy`` — the default won every time,
so a switch that looks configurable was permanently on, invisible to
``config list`` and to the JSON schema.

The pattern is legitimate for genuinely optional attributes, so this test does
not ban it; it only requires that the name be declared somewhere in the config
models, which is what makes it reachable by a user.
"""

from __future__ import annotations

import re
from pathlib import Path

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


@pytest.mark.parametrize("name", ["auto_proxy"])
def test_known_phantom_stays_gone(name: str) -> None:
    """Regression pin for the phantom this guard was written for."""
    assert name not in {n for _, _, n in _getattr_sites()}
