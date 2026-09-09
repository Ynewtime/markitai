#!/usr/bin/env python3
"""Check a built markitai wheel for everything a published artifact must carry.

Run by both CI and the release workflow so the release check cannot drift
away from the one that gates every pull request::

    python scripts/check_wheel.py dist/markitai-1.2.3-py3-none-any.whl
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

REQUIRED_FILES = {
    "markitai/serve/static/index.html",
    "markitai/serve/static/favicon.ico",
    "markitai/serve/static/logo.svg",
}


def check_wheel(wheel_path: Path) -> None:
    """Raise ``SystemExit`` with a readable message on the first problem found."""
    with zipfile.ZipFile(wheel_path) as wheel_file:
        names = set(wheel_file.namelist())
        entry_points_path = next(
            name for name in names if name.endswith(".dist-info/entry_points.txt")
        )
        entry_points = wheel_file.read(entry_points_path).decode()

    missing = REQUIRED_FILES - names
    if missing or not any(
        name.startswith("markitai/serve/static/assets/") for name in names
    ):
        raise SystemExit(f"wheel is missing packaged webapp files: {sorted(missing)}")

    # MIT requires the licence to travel with every copy, and Apache-2.0
    # section 4(d) requires the NOTICE; both land in dist-info/licenses/
    # only while `license-files` resolves inside packages/markitai.
    for suffix in ("LICENSE", "NOTICE"):
        if not any(name.endswith(f".dist-info/licenses/{suffix}") for name in names):
            raise SystemExit(f"wheel is missing dist-info/licenses/{suffix}")

    # "../" in a member path escapes the install directory on extraction.
    traversal = sorted(name for name in names if ".." in name.split("/"))
    if traversal:
        raise SystemExit(f"wheel contains path-traversal entries: {traversal}")

    # The bundled MCP server must ship its console script and module.
    if "markitai-mcp = markitai.mcp.server:main" not in entry_points:
        raise SystemExit(f"wheel is missing the markitai-mcp script:\n{entry_points}")
    if not any(name.startswith("markitai/mcp/") for name in names):
        raise SystemExit("wheel is missing the markitai/mcp/ package")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path, help="path to the built .whl")
    check_wheel(parser.parse_args().wheel)


if __name__ == "__main__":
    main()
