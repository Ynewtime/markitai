#!/usr/bin/env python3
"""Fail the build when an installed dependency carries an unacceptable licence.

markitai itself is MIT. Two licence classes are treated as build breakers:

* **Non-commercial / proprietary** — always fatal. `pymupdf-layout` 1.28.0
  shipped as "Polyform Noncommercial or Artifex Commercial", silently making
  the default install unusable in a commercial setting. Nothing in that class
  is ever acceptable, so there is no allowlist for it.
* **Strong copyleft (AGPL/GPL)** — fatal *unless* the distribution is named in
  :data:`ALLOWED_COPYLEFT`. The PyMuPDF trio is there deliberately and is
  disclosed in ``NOTICE`` and ``README.md``; a new AGPL dependency arriving by
  transitive resolution must be a conscious decision, not a surprise.

* **Source-available / weak copyleft** — Elastic-2.0, BUSL, SSPL and LGPL are
  fatal *unless* the distribution is named in :data:`ALLOWED_RESTRICTED`.
  These terms are acceptable for an opt-in extra but must never arrive
  silently: Elastic-2.0 forbids offering the software as a hosted service,
  which is a live concern for a project that ships ``markitai serve``, yet it
  contains no forbidden marker and no GPL spelling, so it would otherwise
  classify as permissive by accident. LGPL is separated from GPL before the
  strong-copyleft test, so it is never mistaken for one.

Run against whatever is installed in the current interpreter::

    uv run python scripts/check_licenses.py          # audit
    uv run python scripts/check_licenses.py --list   # show every classification
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

#: Distributions whose strong-copyleft licence is a reviewed, disclosed choice.
#: Names are PEP 503 normalized. Adding an entry means accepting the licence
#: for every downstream user — update NOTICE and README.md in the same change.
ALLOWED_COPYLEFT: frozenset[str] = frozenset(
    {
        "pymupdf",
        "pymupdf-layout",
        "pymupdf4llm",
    }
)

#: Distributions whose source-available or weak-copyleft licence is a
#: reviewed, disclosed choice. Values are the reason, quoted back in ``--list``.
#: Every entry here is an *optional extra*, never part of a default install.
ALLOWED_RESTRICTED: dict[str, str] = {
    "kreuzberg": (
        "Elastic-2.0, opt-in via markitai[kreuzberg]; forbids offering it to "
        "others as a hosted service — see NOTICE section 4"
    ),
    "cairosvg": (
        "LGPL-3.0-or-later, opt-in via markitai[svg]; used as an unmodified "
        "library, which carries no obligation on our sources"
    ),
}

# Classification results.
PERMISSIVE = "permissive"
COPYLEFT = "copyleft"
RESTRICTED = "restricted"
FORBIDDEN = "forbidden"
UNKNOWN = "unknown"

#: Source-available families: they permit use and modification but attach a
#: real restriction (typically "not as a hosted service"). None of these
#: spellings appear inside a permissive licence name, so substring matching is
#: safe. They are classified rather than forbidden because an opt-in extra may
#: legitimately carry one — but never silently.
_SOURCE_AVAILABLE_MARKERS = (
    "elastic-2",
    "elastic license",
    "busl",
    "business source license",
    "sspl",
    "server side public license",
)

#: Licence families that forbid or restrict commercial use. Substring matching
#: is safe here because these tokens do not occur inside permissive licence
#: names ("commercial" alone is deliberately absent — dual-licensed AGPL
#: packages advertise an "Artifex Commercial License" alternative).
_FORBIDDEN_MARKERS = (
    "polyform",
    "noncommercial",
    "non-commercial",
    "proprietary",
)

# LGPL spellings are stripped before the GPL test so "LGPL-3.0-or-later" does
# not read as GPL. Order matters: longest spellings first.
_LGPL_PATTERN = re.compile(
    r"\bl[-\s]?gpl\b|lesser general public license|library general public license"
)
_AGPL_PATTERN = re.compile(r"\ba[-\s]?gpl\b|affero")
_GPL_PATTERN = re.compile(r"\bgpl\b|general public license")

# "GPL-compatible" is prose from permissive licence texts, not a GPL grant.
_GPL_FALSE_FRIENDS = re.compile(r"gpl[-\s]?compatible")

#: A ``License`` metadata field longer than this (or containing newlines) is
#: the full licence *text*, not an identifier. Such texts quote unrelated
#: licences — pandas ships 50KB of BSD text that mentions "GPL-compatible" —
#: so they are ignored in favour of the trove classifiers.
_MAX_IDENTIFIER_LENGTH = 200


@dataclass(frozen=True)
class PackageLicense:
    """One distribution and the licence text that governs it."""

    name: str
    license: str


def normalize_name(name: str) -> str:
    """Normalize a distribution name per PEP 503."""
    return re.sub(r"[-_.]+", "-", name).strip().lower()


def _is_identifier(value: str) -> bool:
    """Return whether a ``License`` field looks like an SPDX-ish identifier."""
    return "\n" not in value and len(value) <= _MAX_IDENTIFIER_LENGTH


def license_text(
    license_expression: str | None,
    license_field: str | None,
    classifiers: Sequence[str],
) -> str:
    """Reduce a distribution's licence metadata to one authoritative string.

    Precedence follows PEP 639: a ``License-Expression`` is definitive, then a
    short ``License`` identifier, and trove classifiers are the last resort.
    Classifiers go stale (pillow-heif declares ``License: BSD-3-Clause`` while
    still advertising a GPLv2 classifier), so they never override an explicit
    field.
    """
    if license_expression and license_expression.strip():
        return license_expression.strip()
    if license_field and license_field.strip() and _is_identifier(license_field):
        return license_field.strip()
    license_classifiers = [c for c in classifiers if c.startswith("License ::")]
    if license_classifiers:
        return " | ".join(license_classifiers)
    if license_field and license_field.strip():
        # Full licence text with no classifier to fall back on. Only the first
        # line is trustworthy as a name ("BSD 3-Clause License", "MIT License").
        return license_field.strip().splitlines()[0]
    return ""


def classify(text: str) -> str:
    """Classify a licence string into permissive / copyleft / forbidden."""
    if not text or not text.strip():
        return UNKNOWN
    lowered = text.lower()

    if any(marker in lowered for marker in _FORBIDDEN_MARKERS):
        return FORBIDDEN

    if any(marker in lowered for marker in _SOURCE_AVAILABLE_MARKERS):
        return RESTRICTED

    stripped = _GPL_FALSE_FRIENDS.sub(" ", lowered)
    # LGPL is removed before the GPL test so it is never read as strong
    # copyleft; it is reported as RESTRICTED in its own right instead.
    without_lgpl = _LGPL_PATTERN.sub(" ", stripped)
    if _AGPL_PATTERN.search(without_lgpl) or _GPL_PATTERN.search(without_lgpl):
        return COPYLEFT
    if _LGPL_PATTERN.search(stripped):
        return RESTRICTED

    return PERMISSIVE


def audit(
    packages: Iterable[PackageLicense],
    allowed_copyleft: Iterable[str] = ALLOWED_COPYLEFT,
    allowed_restricted: Iterable[str] = tuple(ALLOWED_RESTRICTED),
) -> list[str]:
    """Return one human-readable message per policy violation (empty == clean)."""
    allowed = {normalize_name(name) for name in allowed_copyleft}
    allowed_source_available = {normalize_name(name) for name in allowed_restricted}
    violations: list[str] = []
    for package in packages:
        verdict = classify(package.license)
        if verdict == FORBIDDEN:
            violations.append(
                f"{package.name}: non-commercial or proprietary licence "
                f"({package.license!r}). markitai ships under MIT and cannot "
                f"depend on it — find a replacement or a differently licensed "
                f"release."
            )
        elif verdict == COPYLEFT and normalize_name(package.name) not in allowed:
            violations.append(
                f"{package.name}: strong copyleft licence ({package.license!r}) "
                f"is not on the allowlist. Add it to ALLOWED_COPYLEFT in "
                f"scripts/check_licenses.py and disclose it in NOTICE and "
                f"README.md, or drop the dependency."
            )
        elif (
            verdict == RESTRICTED
            and normalize_name(package.name) not in allowed_source_available
        ):
            violations.append(
                f"{package.name}: source-available or weak-copyleft licence "
                f"({package.license!r}) is not on the allowlist. Such terms are "
                f"acceptable for an opt-in extra but never silently: add it to "
                f"ALLOWED_RESTRICTED in scripts/check_licenses.py with a reason "
                f"and disclose it in NOTICE, or drop the dependency."
            )
    return violations


def iter_installed_licenses() -> Iterator[PackageLicense]:
    """Yield the licence of every distribution installed for this interpreter."""
    from importlib.metadata import distributions

    for dist in distributions():
        metadata = dist.metadata
        name = metadata["Name"]
        if not name:
            continue
        yield PackageLicense(
            name=name,
            license=license_text(
                license_expression=metadata.get("License-Expression"),
                license_field=metadata.get("License"),
                classifiers=metadata.get_all("Classifier") or [],
            ),
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print every distribution and its classification, then audit.",
    )
    args = parser.parse_args(argv)

    packages = sorted(iter_installed_licenses(), key=lambda p: normalize_name(p.name))

    # A gate that grades the wrong interpreter is worse than no gate: run via
    # the shebang and it inspects the system Python, finds three unrelated
    # distributions and prints "audit passed". markitai's own presence is the
    # cheapest proof that this is the environment we mean to audit.
    if not any(normalize_name(p.name) == "markitai" for p in packages):
        print(
            f"License audit ABORTED: markitai is not installed in {sys.executable}, "
            f"so this environment is not the one to audit "
            f"({len(packages)} unrelated distribution(s) found).\n"
            f"  Run it against the project environment:\n"
            f"    uv run python scripts/check_licenses.py"
        )
        return 1

    if args.list:
        for package in packages:
            print(f"{classify(package.license):10} {package.name:32} {package.license}")
        print()

    violations = audit(packages)
    if violations:
        print(f"License audit FAILED ({len(violations)} violation(s)):")
        for violation in violations:
            print(f"  - {violation}")
        return 1

    print(f"License audit passed: {len(packages)} distribution(s) checked.")
    allowed = ", ".join(sorted(ALLOWED_COPYLEFT))
    print(f"Copyleft allowlist: {allowed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
