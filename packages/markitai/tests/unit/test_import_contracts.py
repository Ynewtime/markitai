"""Guard: every top-level markitai module is covered by the layering contracts.

The import-linter contracts in the root pyproject.toml enumerate source
modules explicitly. A newly added top-level module would silently escape the
"nothing below cli may import it" contract until someone remembers to list
it — this test turns that drift into a red test instead.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import markitai

_REPO_ROOT = Path(__file__).resolve().parents[4]

# Modules legitimately outside the "may not import cli" contract:
# the cli package itself and the __main__ entrypoint that launches it.
_CLI_SIDE = {"markitai.cli", "markitai.__main__"}


def _top_level_modules() -> set[str]:
    src = Path(markitai.__file__).parent
    modules: set[str] = set()
    for entry in src.iterdir():
        if entry.name.startswith("_") and entry.name != "__main__.py":
            continue
        if entry.is_dir() and (entry / "__init__.py").exists():
            modules.add(f"markitai.{entry.name}")
        elif entry.suffix == ".py":
            modules.add(f"markitai.{entry.stem}")
    return modules


def _contracts() -> list[dict]:
    data = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text())
    return data["tool"]["importlinter"]["contracts"]


def test_cli_top_layer_contract_covers_every_module() -> None:
    contract = next(
        c for c in _contracts() if c["forbidden_modules"] == ["markitai.cli"]
    )
    covered = set(contract["source_modules"]) | _CLI_SIDE
    missing = _top_level_modules() - covered
    assert not missing, (
        f"New top-level module(s) {sorted(missing)} escape the "
        "'nothing below cli may import it' contract — add them to "
        "source_modules in [tool.importlinter] (root pyproject.toml)"
    )
