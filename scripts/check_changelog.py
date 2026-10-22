#!/usr/bin/env python3
"""Keep unreleased versions out of CHANGELOG.md and CHANGELOG.zh.md.

The changelogs change only with a release: ``main`` may correct a released
section at any time, but a ``## [X.Y.Z]`` section above the newest ``vX.Y.Z``
tag is an unreleased version and fails the check. The release commit adds
its section and is pushed together with its tag
(``git push --atomic origin main vX.Y.Z``), so the tag exists when this runs.

Website Check runs the guard on every push and pull request::

    python3 scripts/check_changelog.py

The Release workflow runs it on the tagged commit, which must carry the
section for its version in both files::

    python3 scripts/check_changelog.py --release vX.Y.Z
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CHANGELOGS = ("CHANGELOG.md", "CHANGELOG.zh.md")
UNRELEASED_NOTES = (
    ".local/changelog-unreleased.md and .local/changelog-unreleased.zh.md"
)

_VERSION = re.compile(r"(\d+)\.(\d+)\.(\d+)")
_HEADING = re.compile(r"## \[([^\]]*)\]")

Version = tuple[int, int, int]


def parse_version(text: str) -> Version | None:
    m = _VERSION.fullmatch(text.strip())
    return (int(m[1]), int(m[2]), int(m[3])) if m else None


def section_names(text: str) -> list[str]:
    """Every ``## `` heading outside code fences: its ``[name]``, or the whole line."""
    names: list[str] = []
    fenced = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            fenced = not fenced
        if not fenced and line.startswith("## "):
            m = _HEADING.match(line)
            names.append(m[1] if m else line)
    return names


def unreleased_sections(text: str, newest: Version) -> list[str]:
    """Headings that are not a version at or below the newest tag."""
    out = []
    for name in section_names(text):
        version = parse_version(name)
        if version is None or version > newest:
            out.append(name)
    return out


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout


def local_tags() -> list[Version]:
    out = _git("tag", "--list", "v*")
    return [v for t in out.split() if (v := parse_version(t.removeprefix("v")))]


def remote_has_tag(tag: str) -> bool:
    """A second look at origin, for a tag pushed after the checkout fetched."""
    try:
        return bool(_git("ls-remote", "--tags", "origin", f"refs/tags/{tag}").strip())
    except (OSError, subprocess.CalledProcessError):
        return False


def check_unreleased(root: Path = REPO_ROOT) -> list[str]:
    try:
        tags = local_tags()
    except (OSError, subprocess.CalledProcessError) as exc:
        return [f"cannot read the release tags ({exc}); check out with fetch-depth: 0"]
    if not tags:
        return ["no vX.Y.Z tag found; check out with fetch-depth: 0"]
    newest = max(tags)
    errors = []
    for name in CHANGELOGS:
        for section in unreleased_sections(
            (root / name).read_text(encoding="utf-8"), newest
        ):
            version = parse_version(section)
            if version is not None and remote_has_tag(f"v{section}"):
                continue
            what = f"version {section}" if version else f"heading {section!r}"
            errors.append(
                f"{name} has a section for {what}, which has no release tag "
                f"(newest: v{'.'.join(map(str, newest))}). Unreleased notes go to "
                f"{UNRELEASED_NOTES} (gitignored); the release commit moves them into "
                "the CHANGELOG and is pushed with its tag: "
                "git push --atomic origin main vX.Y.Z"
            )
    return errors


def check_release(tag: str, root: Path = REPO_ROOT) -> list[str]:
    version = tag.removeprefix("v")
    if parse_version(version) is None:
        return [f"{tag!r} is not a vX.Y.Z tag"]
    errors = []
    for name in CHANGELOGS:
        if version not in section_names((root / name).read_text(encoding="utf-8")):
            errors.append(
                f"{name} at {tag} has no '## [{version}]' section; add it in the "
                "release commit (the GitHub Release notes are the CHANGELOG.md section)"
            )
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Keep unreleased versions out of the changelogs."
    )
    parser.add_argument(
        "--release", metavar="vX.Y.Z", help="require this version's section instead"
    )
    args = parser.parse_args(argv)
    errors = check_release(args.release) if args.release else check_unreleased()
    for error in errors:
        print(f"::error::{error}")
    if not errors:
        print("CHANGELOG check passed.")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
