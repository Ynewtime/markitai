#!/usr/bin/env python3
"""Check a built markitai wheel for everything a published artifact must carry.

Run by both CI and the release workflow so the release check cannot drift
away from the one that gates every pull request::

    python scripts/check_wheel.py dist/markitai-1.2.3-py3-none-any.whl
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

WORKBENCH = "markitai/serve/workbench/"
REQUIRED_FILES = {
    "markitai/serve/workbench/manifest.json",
    "markitai/serve/workbench/index.html",
    "markitai/serve/workbench/ui/boot.js",
    "markitai/serve/workbench/ui/app.css",
    "markitai/serve/workbench/ui/logo.svg",
    # Block boxes of a page's screenshots (contract v1.14), run as shipped
    "markitai/serve/workbench/capture/page-blocks.js",
    # Prices of models the pinned LiteLLM does not list (markitai.model_catalog)
    "markitai/litellm_rows.json",
    # Provider groups and brand-mark keys (markitai.serve.provider_catalog)
    "markitai/serve/provider_catalog.json",
    # Commit, build time and dirty flag (markitai.build_info), from the build hook
    "markitai/_build_info.py",
}


def check_wheel(wheel_path: Path) -> None:
    """Raise ``SystemExit`` with a readable message on the first problem found."""
    with zipfile.ZipFile(wheel_path) as wheel_file:
        names = set(wheel_file.namelist())
        entry_points_path = next(
            (name for name in names if name.endswith(".dist-info/entry_points.txt")),
            None,
        )
        if entry_points_path is None:
            raise SystemExit("wheel is missing dist-info/entry_points.txt")
        entry_points = wheel_file.read(entry_points_path).decode()
        manifest = f"{WORKBENCH}manifest.json"
        listed = (
            json.loads(wheel_file.read(manifest))["files"] if manifest in names else {}
        )

    missing = sorted(REQUIRED_FILES - names)
    # Every file of the vendored workbench its manifest lists: brand marks,
    # fonts and licences included, not only the few named above
    missing += sorted(
        f"{WORKBENCH}{name}" for name in listed if f"{WORKBENCH}{name}" not in names
    )
    # The app script ships plain, gzip-only, or both (the bundle decides).
    if not names & {
        "markitai/serve/workbench/ui/app.js",
        "markitai/serve/workbench/ui/app.js.gz",
    }:
        missing.append("markitai/serve/workbench/ui/app.js")
    # The UI's libraries and font are MIT/OFL: their licences travel with them.
    if not any(name.startswith("markitai/serve/workbench/LICENSES/") for name in names):
        missing.append("markitai/serve/workbench/LICENSES/")
    if missing:
        raise SystemExit(f"wheel is missing required files: {missing}")
    # The retired React build must not ride along next to the workbench.
    stale = sorted(name for name in names if name.startswith("markitai/serve/static/"))
    if stale:
        raise SystemExit(f"wheel still carries the retired web UI: {stale[:5]}")

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
