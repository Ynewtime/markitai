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
* **Fails** when the CLI defines an option no guide mentions at all, and when
  a top-level command has no reference section. Both are coverage gaps rather
  than lies, but a shipped flag nobody documented is a flag nobody can use.
* **Ignores fenced code blocks** when scanning headings and flags: a shell
  comment such as `# removed in 1.0` inside a snippet must not be able to
  satisfy — or excuse — any rule.

What it cannot catch: wrong descriptions, stale defaults, wrong values for a
`Choice` option, or prose that contradicts behavior. Those still need review.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from urllib.parse import unquote

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
        "--from",  # uvx --from "markitai[mcp]" in MCP client setup snippets
    }
)

# A Markdown heading: `#`, optional level, then a space (a bash comment is not one).
_HEADING_RE = re.compile(r"^#{1,6}\s")

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


def _removed_option_names() -> dict[str, str | None]:
    """The CLI's own removed-option table: removed name -> replacement.

    A ``None`` replacement means the option was dropped with nothing to use
    instead (``--kreuzberg``), and the docs must not promise a successor.
    """
    from markitai.cli.framework import _REMOVED_OPTIONS

    return dict(_REMOVED_OPTIONS)


def _prose_lines(text: str) -> list[str]:
    """Lines with fenced code blocks blanked out, numbering preserved.

    A bash comment (`# removed in 1.0`) or an install snippet inside a fence
    must not count as a heading or as documentation of a flag.
    """
    out: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            out.append("")
            continue
        out.append("" if in_fence else line)
    return out


def _flags_in(path: Path) -> set[str]:
    """Every long flag named anywhere in one page, examples included.

    Unlike the heading scan, code fences count here: a flag shown in a
    runnable example is discoverable. Only headings need the prose-only view,
    because a bash `# comment` otherwise reads as a section title.
    """
    return {
        flag
        for flag in _FLAG_RE.findall(path.read_text(encoding="utf-8"))
        if not flag.endswith("-")
    }


def _sectioned_lines(path: Path) -> list[tuple[str, str]]:
    """Yield (nearest preceding prose heading, line) for every line.

    Lines inside a fence stay in the scan — a stale example is still stale —
    but a bash `# comment` inside one never becomes the heading that would
    excuse the flag it mentions.
    """
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    prose_lines = _prose_lines(path.read_text(encoding="utf-8"))
    heading = ""
    out: list[tuple[str, str]] = []
    for raw_line, prose_line in zip(raw_lines, prose_lines):
        if _HEADING_RE.match(prose_line):
            heading = prose_line
        out.append((heading, raw_line))
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
    """Both guides must name the extras a reader has to install by hand.

    Written from the package metadata so a new extra cannot ship
    undocumented; `all` is included because the guide presents it as the
    one-step option.
    """
    text = (guide / "getting-started.md").read_text(encoding="utf-8")
    for extra in sorted(_declared_extras()):
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
        # The configuration page claimed OCR "works out of the box" long
        # after it moved behind the `ocr` extra; a bare `markitai` install
        # fails on `--ocr`.
        "RapidOCR is included as a dependency and works out of the box",
        "RapidOCR 已作为依赖包含，开箱即用",
    )
    for path in _iter_docs():
        text = path.read_text(encoding="utf-8")
        for phrase in stale:
            assert phrase not in text, (
                f"{path.relative_to(_REPO_ROOT)} still calls OCR a core "
                f"requirement: {phrase!r}"
            )


# ============================================================
# Version, command coverage, anchors, licence: facts that drift silently
# ============================================================


def _project_version() -> str:
    """The version is single-sourced in `__init__.py` (hatch dynamic version)."""
    init = Path(__file__).resolve().parents[2] / "src" / "markitai" / "__init__.py"
    match = re.search(
        r'^__version__\s*=\s*["\']([^"\']+)',
        init.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    assert match, f"no __version__ in {init}"
    return match.group(1)


def test_docs_do_not_describe_the_project_as_pre_1_0() -> None:
    """markitai is 1.x; a stale "0.x" makes a stable API look provisional."""
    assert not _project_version().startswith("0."), (
        "this guard exists for the 1.x era; update it with the version bump"
    )
    offenders: list[str] = []
    for path in _iter_docs():
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if re.search(r"\b0\.x\b", line):
                offenders.append(
                    f"{path.relative_to(_REPO_ROOT)}:{number}: {line.strip()}"
                )
    assert not offenders, "docs still describe the project as 0.x:\n" + "\n".join(
        offenders
    )


def _top_level_commands() -> set[str]:
    """Every subcommand the CLI group actually exposes."""
    from markitai.cli import app

    return set(app.list_commands(click.Context(app)))


@pytest.mark.parametrize("guide", (_EN_GUIDE, _ZH_GUIDE))
def test_cli_reference_covers_every_top_level_command(guide: Path) -> None:
    """A command nobody documents is a command nobody finds.

    `llms.txt` promises "Every command and flag"; `serve` and `mcp` used to
    be named only in passing, so the reference never taught them.
    """
    page = guide / "cli.md"
    headings = [
        line
        for line in _prose_lines(page.read_text(encoding="utf-8"))
        if _HEADING_RE.match(line)
    ]
    missing = sorted(
        name
        for name in _top_level_commands()
        if not any(f"markitai {name}" in heading for heading in headings)
    )
    assert not missing, f"{page.relative_to(_REPO_ROOT)} has no section for: {missing}"


# Targets generated by `bun run docs:build` rather than committed.
_GENERATED_PAGE_TARGETS = {"/changelog", "/zh/changelog"}

_ZH_LINK_RE = re.compile(r"\]\((/zh/[^)\s#]*)(?:#([^)\s]+))?\)")


def _slugify_heading(text: str) -> str:
    """Approximate VitePress/markdown-it-anchor ids (CJK kept as-is).

    ``### `--record-history``` renders as ``id="record-history"``: leading
    and trailing punctuation is stripped, so the slug is stripped too.
    """
    # An explicit ``{#anchor}`` suffix is the heading's real id; drop it from
    # the text so the slug matches what the link points at.
    slug = re.sub(r"\s*\{#[^}]*\}\s*$", "", text).strip().lower()
    slug = re.sub(r"[`*_]", "", slug)
    slug = slug.replace(" ", "-")
    return re.sub(r"[^\w\u4e00-\u9fff-]", "", slug, flags=re.UNICODE).strip("-")


def _heading_slugs(path: Path) -> set[str]:
    """Every id a heading answers to: its explicit `{#id}` or its slug."""
    slugs: set[str] = set()
    for line in _prose_lines(path.read_text(encoding="utf-8")):
        match = re.match(r"^#{1,6}\s+(.*)$", line)
        if not match:
            continue
        text = match.group(1)
        explicit = re.search(r"\{#[^}]*\}\s*$", text)
        if explicit:
            slugs.add(explicit.group(0)[2:-1].strip())
        slugs.add(_slugify_heading(text))
    return slugs


def test_zh_internal_anchors_resolve() -> None:
    """A Chinese link must resolve to a page, and to a heading when it has one."""
    offenders: list[str] = []
    for page in sorted(_ZH_GUIDE.glob("*.md")):
        text = page.read_text(encoding="utf-8")
        for match in _ZH_LINK_RE.finditer(text):
            target, anchor = match.group(1), match.group(2)
            candidate = _WEBSITE / target.lstrip("/")
            if not candidate.suffix:
                candidate = candidate.with_suffix(".md")
            if not candidate.exists():
                # The changelog pages are copied in at build time; their
                # committed sources are the repo-root files.
                if target not in _GENERATED_PAGE_TARGETS:
                    offenders.append(
                        f"{page.relative_to(_REPO_ROOT)} -> {target} "
                        f"(target page does not exist)"
                    )
                continue
            if anchor is not None and unquote(anchor) not in _heading_slugs(candidate):
                offenders.append(
                    f"{page.relative_to(_REPO_ROOT)} -> {target}#{anchor} "
                    f"(no such heading in {candidate.relative_to(_REPO_ROOT)})"
                )
    assert not offenders, "broken Chinese internal anchors:\n" + "\n".join(offenders)


def test_llms_txt_states_the_licence_accurately() -> None:
    """`llms.txt` is what AI agents quote for compliance; it cannot say "MIT".

    The default install bundles Artifex's PyMuPDF stack under AGPL-3.0 or a
    commercial licence, so a bare "MIT licensed" overclaims.
    """
    text = (_WEBSITE / "public" / "llms.txt").read_text(encoding="utf-8")
    assert "MIT licensed" not in text, (
        "llms.txt still claims the whole project is MIT licensed; name the "
        "AGPL PyMuPDF components and point at NOTICE instead"
    )
    assert "AGPL" in text or "NOTICE" in text, (
        "llms.txt must explain the AGPL/commercial PyMuPDF component"
    )


def _comparison_table(text: str) -> list[str]:
    """Rows of the markitai-vs-others table, whitespace-normalised."""
    rows: list[str] = []
    started = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("| |") and "markitai" in stripped:
            started = True
        if not started:
            continue
        if not stripped.startswith("|"):
            break
        rows.append(" ".join(stripped.split()))
    return rows


def test_readme_and_site_publish_the_same_comparison_table() -> None:
    """The same table lives on GitHub and on the site; drift misleads one of them."""
    readme = _comparison_table(_README.read_text(encoding="utf-8"))
    site = _comparison_table((_EN_GUIDE / "comparison.md").read_text(encoding="utf-8"))
    assert len(readme) >= 2, "README.md lost its markitai-vs-others comparison table"
    assert readme == site, (
        "README.md and website/guide/comparison.md publish different comparison "
        "tables; update both from one source"
    )
    zh_site = _comparison_table(
        (_ZH_GUIDE / "comparison.md").read_text(encoding="utf-8")
    )
    assert len(zh_site) == len(site), (
        "the Chinese comparison page has a different number of table rows"
    )


def test_home_feature_cards_stay_bilingual_and_grid_aligned() -> None:
    """One shared feature list feeds both home pages.

    VitePress picks `grid-6` for a multiple of three cards and `grid-4`
    otherwise, so five cards leave one alone in a four-wide row. Each entry
    also carries its own `en`/`zh` copy; a one-language addition is the usual
    way the two home pages drift apart.
    """
    features = (_WEBSITE / ".vitepress" / "theme" / "features.ts").read_text(
        encoding="utf-8"
    )
    cards = len(re.findall(r"icon: svg\(", features))
    assert cards >= 3 and cards % 3 == 0, (
        f"{cards} home feature cards do not fill the default theme's grid "
        f"(use a multiple of three)"
    )
    # 4-space indent: the interface's own `en:`/`zh:` declarations are 2-space.
    english = len(re.findall(r"^    en: \{$", features, re.MULTILINE))
    chinese = len(re.findall(r"^    zh: \{$", features, re.MULTILINE))
    assert english == cards, "a feature card is missing its English copy"
    assert chinese == cards, "a feature card is missing its Chinese copy"
