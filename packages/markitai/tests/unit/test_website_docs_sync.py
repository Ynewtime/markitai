"""Keep the published docs honest about what the CLI actually accepts.

The rot this module exists to prevent: one fact (an option name, a dependency,
a consent rule) is copied by hand into the CLI, `website/guide/*.md`, and
`website/zh/guide/*.md`. Nothing linked those copies, so when six deprecated
backend aliases were deleted from click, both language guides kept advertising
them — including a mutual-exclusion table for flags that no longer parse.

The guard is deliberately cheap and one-directional where it has to be:

* **Fails** when a guide documents a `--flag` the CLI does not define. That is
  the direction that misleads readers, and it is exactly the shape the
  deprecated-alias rot took.
* **Fails** when a flag the CLI *removed* is mentioned outside a section that
  says it was removed, in any page. Migration notes stay legal; "still works"
  prose does not.
* **Fails** when the English and Chinese CLI guides document different option
  sets, which is how half-finished bilingual edits show up.
* **Warns only** when the CLI defines an option no guide mentions. Undocumented
  is a gap, not a lie, and failing on it would make every new flag a
  docs-blocking change.

What it cannot catch: wrong descriptions, stale defaults, wrong values for a
`Choice` option, or prose that contradicts behavior. Those still need review.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import click
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[4]
_WEBSITE = _REPO_ROOT / "website"
_EN_GUIDE = _WEBSITE / "guide"
_ZH_GUIDE = _WEBSITE / "zh" / "guide"
_README = _REPO_ROOT / "README.md"
_SKILLS = _REPO_ROOT / "skills"
_MARKITAI_PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"

pytestmark = pytest.mark.skipif(
    not _WEBSITE.is_dir(), reason="website/ is not part of this checkout"
)

# `--flag`. Trailing-hyphen matches are prose globs (`--no-*`), not options.
_FLAG_RE = re.compile(r"(?<![\w-])--[a-zA-Z][\w-]*")

# Flags that belong to other tools and legitimately appear in install snippets.
_FOREIGN_FLAGS = frozenset(
    {
        "--force",  # uv tool install / pipx install
        "--help",  # click's built-in, not a declared param
    }
)

# A removed flag may only be named under a heading that says it is removed.
_REMOVAL_MARKERS = ("removed", "已移除")


def _iter_docs() -> list[Path]:
    """Return every hand-written Markdown page: both guides and the skills.

    `skills/` teaches agents the same CLI the guides teach humans, so it rots
    the same way and belongs in the same scan — it was the one copy of the
    "deprecated aliases still work" sentence that survived their removal.

    Skips build artifacts: `pnpm docs:build` copies CHANGELOG.md in as
    `changelog.md`, and a changelog legitimately names flags that no longer
    exist. node_modules is skipped for the obvious reason.
    """
    roots = [_WEBSITE, _SKILLS] if _SKILLS.is_dir() else [_WEBSITE]
    return sorted(
        path
        for root in roots
        for path in root.rglob("*.md")
        if path.name != "changelog.md" and "node_modules" not in path.parts
    )


def _cli_option_names() -> set[str]:
    """Reflect every long option name out of the click command tree."""
    from markitai.cli import app

    def walk(command: click.Command) -> set[str]:
        names = {
            option
            for param in command.params
            for option in (*param.opts, *param.secondary_opts)
            if option.startswith("--")
        }
        if isinstance(command, click.Group):
            ctx = click.Context(command)
            for name in command.list_commands(ctx):
                sub = command.get_command(ctx, name)
                if sub is not None:
                    names |= walk(sub)
        return names

    return walk(app)


def _removed_option_names() -> dict[str, str]:
    """The CLI's own removed-option table: removed name -> replacement."""
    from markitai.cli.framework import _REMOVED_OPTIONS

    return dict(_REMOVED_OPTIONS)


def _flags_in(path: Path) -> set[str]:
    """Every long flag named anywhere in one page."""
    return {
        flag
        for flag in _FLAG_RE.findall(path.read_text(encoding="utf-8"))
        if not flag.endswith("-")
    }


def _sectioned_lines(path: Path) -> list[tuple[str, str]]:
    """Yield (nearest preceding heading, line) for each line of a page."""
    heading = ""
    out: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            heading = line
        out.append((heading, line))
    return out


# ============================================================
# Documented options must exist
# ============================================================


@pytest.mark.parametrize("guide", ("en", "zh"))
def test_cli_guide_documents_only_real_options(guide: str) -> None:
    """Every `--flag` in the CLI reference must be a flag the CLI defines."""
    page = (_EN_GUIDE if guide == "en" else _ZH_GUIDE) / "cli.md"
    known = _cli_option_names() | set(_removed_option_names()) | _FOREIGN_FLAGS
    unknown = sorted(_flags_in(page) - known)
    assert not unknown, (
        f"{page.relative_to(_REPO_ROOT)} documents options the CLI does not "
        f"define: {unknown}. Either the docs are stale or the flag was renamed."
    )


def test_en_and_zh_cli_guides_document_the_same_options() -> None:
    """A one-language edit is the usual way the guides drift apart."""
    english = _flags_in(_EN_GUIDE / "cli.md")
    chinese = _flags_in(_ZH_GUIDE / "cli.md")
    assert not sorted(english - chinese), (
        f"documented in guide/cli.md but not in zh/guide/cli.md: "
        f"{sorted(english - chinese)}"
    )
    assert not sorted(chinese - english), (
        f"documented in zh/guide/cli.md but not in guide/cli.md: "
        f"{sorted(chinese - english)}"
    )


def test_removed_flags_are_only_named_in_removal_notes() -> None:
    """Removed aliases may be named in migration notes, nowhere else.

    This is the assertion that would have failed the moment the six backend
    aliases were deleted from click while both guides still taught them.
    """
    removed = set(_removed_option_names())
    offenders: list[str] = []
    for path in [*_iter_docs(), _README]:
        for number, (heading, line) in enumerate(_sectioned_lines(path), start=1):
            if not any(flag in line for flag in removed):
                continue
            context = f"{heading}\n{line}".lower()
            if any(marker in context for marker in _REMOVAL_MARKERS):
                continue
            offenders.append(f"{path.relative_to(_REPO_ROOT)}:{number}: {line.strip()}")
    assert not offenders, (
        "removed CLI flags are still presented as usable:\n" + "\n".join(offenders)
    )


def test_every_cli_option_is_documented() -> None:
    """A flag nobody wrote down is a flag nobody can use."""
    documented: set[str] = set()
    for path in _iter_docs():
        documented |= _flags_in(path)
    missing = sorted(_cli_option_names() - documented - _FOREIGN_FLAGS)
    assert not missing, f"CLI options not mentioned anywhere under website/: {missing}"


# ============================================================
# Extras: one table, one source of truth
# ============================================================


def _declared_extras() -> set[str]:
    metadata = tomllib.loads(_MARKITAI_PYPROJECT.read_text(encoding="utf-8"))
    return set(metadata["project"]["optional-dependencies"])


def test_readme_extras_table_matches_package_metadata() -> None:
    """The README table is the public extras contract; keep it exhaustive.

    `heif` was missing from it for several releases and `ocr` would have been
    missing the moment it stopped being a core dependency.
    """
    readme = _README.read_text(encoding="utf-8")
    documented = set(re.findall(r"^\| `([a-z-]+)` \| ", readme, re.MULTILINE))
    assert documented == _declared_extras(), (
        f"README extras table is out of sync with pyproject: "
        f"missing={sorted(_declared_extras() - documented)}, "
        f"unknown={sorted(documented - _declared_extras())}"
    )


@pytest.mark.parametrize("guide", (_EN_GUIDE, _ZH_GUIDE))
def test_optional_capability_extras_are_documented(guide: Path) -> None:
    """Both guides must name the extras a reader has to install by hand."""
    text = (guide / "getting-started.md").read_text(encoding="utf-8")
    for extra in ("heif", "svg", "browser", "ocr"):
        assert f"markitai[{extra}]" in text, (
            f"{(guide / 'getting-started.md').relative_to(_REPO_ROOT)} never "
            f"tells the reader how to install the {extra!r} extra"
        )


# ============================================================
# Stale prose that outlived the behavior it described
# ============================================================


def test_docs_do_not_advertise_ffmpeg() -> None:
    """No audio/video extension is registered; FFmpeg is not a capability."""
    offenders: list[str] = []
    for path in [*_iter_docs(), _README]:
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if "ffmpeg" in line.lower():
                offenders.append(
                    f"{path.relative_to(_REPO_ROOT)}:{number}: {line.strip()}"
                )
    assert not offenders, "docs still mention FFmpeg:\n" + "\n".join(offenders)


def test_getting_started_pins_no_literal_version() -> None:
    """A version baked into a doc example rots on the next release.

    The pin example claimed to show "the release documented here" while the
    literal was three minor versions behind.
    """
    for guide in (_EN_GUIDE, _ZH_GUIDE):
        page = guide / "getting-started.md"
        text = page.read_text(encoding="utf-8")
        literals = re.findall(r"MARKITAI_VERSION\s*=?\s*\"?(\d+\.\d+\.\d+)", text)
        assert not literals, (
            f"{page.relative_to(_REPO_ROOT)} hardcodes MARKITAI_VERSION="
            f"{literals}; use a placeholder that cannot go stale"
        )


def test_consent_docs_no_longer_exempt_the_x_twitter_path() -> None:
    """`ask` now covers FxTwitter/oEmbed through the shared process decision."""
    stale_en = (
        "This public-URL enrichment does not open its own `ask` prompt",
        "It does not open its own prompt under `ask`",
        "only discloses",
    )
    stale_zh = (
        "不会单独弹出 `ask` 确认",
        "不会为 `ask` 单独弹出确认",
        "例外只揭露、不询问",
    )
    for page in (
        _EN_GUIDE / "fetch-policy.md",
        _EN_GUIDE / "configuration.md",
    ):
        text = page.read_text(encoding="utf-8")
        for stale in stale_en:
            assert stale not in text, (
                f"{page.relative_to(_REPO_ROOT)} still describes the old "
                f"X/Twitter consent exemption: {stale!r}"
            )
    for page in (
        _ZH_GUIDE / "fetch-policy.md",
        _ZH_GUIDE / "configuration.md",
    ):
        text = page.read_text(encoding="utf-8")
        for stale in stale_zh:
            assert stale not in text, (
                f"{page.relative_to(_REPO_ROOT)} still describes the old "
                f"X/Twitter consent exemption: {stale!r}"
            )


def test_doctor_docs_treat_ocr_as_optional() -> None:
    """OCR left the core install, so no page may call RapidOCR required."""
    stale = (
        "Core requirement: RapidOCR",
        "核心要求：RapidOCR",
        "core RapidOCR dependency",
        "核心 RapidOCR 检查",
        "Required Dependencies\n  ✓ RapidOCR",
        "必需依赖\n  ✓ RapidOCR",
    )
    for path in _iter_docs():
        text = path.read_text(encoding="utf-8")
        for phrase in stale:
            assert phrase not in text, (
                f"{path.relative_to(_REPO_ROOT)} still calls OCR a core "
                f"requirement: {phrase!r}"
            )
