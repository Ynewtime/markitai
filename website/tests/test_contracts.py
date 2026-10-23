"""Keep markitai.dev honest about the markitai it documents.

The site documents the Python markitai 1.x line, whose source is the `python`
branch. These guards compare the pages with that code, so they run in the
Python checkout's environment, pointed at it by MARKITAI_PYTHON_CHECKOUT:

    git worktree add --detach ../markitai-python origin/python
    (cd ../markitai-python && uv sync --frozen --all-groups)
    MARKITAI_PYTHON_CHECKOUT=../markitai-python \\
        uv run --project ../markitai-python --no-sync pytest website/tests

Without the variable the checks that read the code skip, except in CI, where
they fail. The site-only checks always run.

The rot this module exists to prevent: one fact (an option name, a dependency,
a consent rule) is copied by hand into the CLI, `website/guide/*.md`, and
`website/zh/guide/*.md`. Nothing linked those copies, so when six deprecated
backend aliases were deleted from click, both language guides kept advertising
them, including a mutual-exclusion table for flags that no longer parse.

* **Fails** when a guide documents a `--flag` the CLI does not define, or
  names a removed flag outside a section that says it was removed.
* **Fails** when the English and Chinese CLI guides document different option
  sets, which is how half-finished bilingual edits show up.
* **Fails** when the CLI defines an option no page mentions at all, and when a
  top-level command has no reference section.
* **Ignores fenced code blocks** when scanning headings: a shell comment such
  as `# removed in 1.0` inside a snippet must not excuse any rule.

What it cannot catch: wrong descriptions, stale defaults, wrong values for a
`Choice` option, or prose that contradicts behavior. Those still need review.
"""

from __future__ import annotations

import os
import re
import tomllib
from pathlib import Path
from urllib.parse import unquote

import pytest

_WEBSITE = Path(__file__).resolve().parents[1]
_EN_GUIDE = _WEBSITE / "guide"
_ZH_GUIDE = _WEBSITE / "zh" / "guide"
_CHANGELOGS = sorted((_WEBSITE / "changelog").glob("*.md"))
_PYTHON_ENV = "MARKITAI_PYTHON_CHECKOUT"

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


def _python_checkout() -> Path:
    """The `python` branch checkout whose code the pages describe."""
    value = os.environ.get(_PYTHON_ENV)
    if not value:
        message = f"{_PYTHON_ENV} names no checkout of the python branch"
        if os.environ.get("CI"):
            pytest.fail(message)
        pytest.skip(message)
    root = Path(value).resolve()
    import markitai

    # The guards are worthless against another installed markitai.
    assert root in Path(markitai.__file__).resolve().parents, (
        f"markitai is imported from {markitai.__file__}, not from {root}; run "
        f"pytest with `uv run --project {value} --no-sync`"
    )
    return root


def _package(root: Path) -> Path:
    return root / "packages" / "markitai"


def _iter_docs() -> list[Path]:
    """Every hand-written page.

    Skips the release notes: `docs:build` copies them in as `changelog.md`, and
    a changelog legitimately names flags that no longer exist.
    """
    return sorted(
        path
        for path in _WEBSITE.rglob("*.md")
        if path.name != "changelog.md"
        and "changelog" not in path.relative_to(_WEBSITE).parts[:-1]
        and "node_modules" not in path.parts
    )


def _cli_option_names() -> set[str]:
    """Reflect every long option name out of the click command tree."""
    _python_checkout()
    import click
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
    """The CLI's own removed-option table: removed name -> replacement."""
    _python_checkout()
    from markitai.cli.framework import _REMOVED_OPTIONS

    return dict(_REMOVED_OPTIONS)


def _prose_lines(text: str) -> list[str]:
    """Lines with fenced code blocks blanked out, numbering preserved."""
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
    """Every long flag named anywhere in one page, examples included."""
    return {
        flag
        for flag in _FLAG_RE.findall(path.read_text(encoding="utf-8"))
        if not flag.endswith("-")
    }


def _sectioned_lines(path: Path) -> list[tuple[str, str]]:
    """(nearest preceding prose heading, line) for every line."""
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    prose_lines = _prose_lines(path.read_text(encoding="utf-8"))
    heading = ""
    out: list[tuple[str, str]] = []
    for raw_line, prose_line in zip(raw_lines, prose_lines):
        if _HEADING_RE.match(prose_line):
            heading = prose_line
        out.append((heading, raw_line))
    return out


def _relative(path: Path) -> str:
    return path.relative_to(_WEBSITE.parent).as_posix()


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
        f"{_relative(page)} documents options the CLI does not define: "
        f"{unknown}. Either the docs are stale or the flag was renamed."
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
    """Removed aliases may be named in migration notes, nowhere else."""
    removed = set(_removed_option_names())
    offenders: list[str] = []
    for path in _iter_docs():
        for number, (heading, line) in enumerate(_sectioned_lines(path), start=1):
            if not any(flag in line for flag in removed):
                continue
            context = f"{heading}\n{line}".lower()
            if any(marker in context for marker in _REMOVAL_MARKERS):
                continue
            offenders.append(f"{_relative(path)}:{number}: {line.strip()}")
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


@pytest.mark.parametrize("guide", (_EN_GUIDE, _ZH_GUIDE), ids=("en", "zh"))
def test_optional_capability_extras_are_documented(guide: Path) -> None:
    """Both guides must name the extras a reader has to install by hand."""
    metadata = tomllib.loads(
        (_package(_python_checkout()) / "pyproject.toml").read_text(encoding="utf-8")
    )
    text = (guide / "getting-started.md").read_text(encoding="utf-8")
    for extra in sorted(metadata["project"]["optional-dependencies"]):
        assert f"markitai[{extra}]" in text, (
            f"{_relative(guide / 'getting-started.md')} never tells the reader "
            f"how to install the {extra!r} extra"
        )


def test_docs_name_every_model_the_code_defaults_to() -> None:
    """A reader must be able to tell which model markitai picks by default."""
    _python_checkout()
    from markitai.constants import PROVIDER_DEFAULT_MODELS

    corpus = "".join(path.read_text(encoding="utf-8") for path in _iter_docs())
    missing = sorted(
        f"{provider} -> {model}"
        for provider, model in PROVIDER_DEFAULT_MODELS.items()
        if model not in corpus
    )
    assert not missing, "defaults the docs never name:\n" + "\n".join(missing)


def _top_level_commands() -> set[str]:
    """Every subcommand the CLI group actually exposes."""
    _python_checkout()
    import click
    from markitai.cli import app

    return set(app.list_commands(click.Context(app)))


@pytest.mark.parametrize("guide", (_EN_GUIDE, _ZH_GUIDE), ids=("en", "zh"))
def test_cli_reference_covers_every_top_level_command(guide: Path) -> None:
    """A command nobody documents is a command nobody finds."""
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
    assert not missing, f"{_relative(page)} has no section for: {missing}"


def test_docs_do_not_describe_the_project_as_pre_1_0() -> None:
    """markitai is 1.x; a stale "0.x" makes a stable API look provisional."""
    init = _package(_python_checkout()) / "src" / "markitai" / "__init__.py"
    version = re.search(
        r'^__version__\s*=\s*["\']([^"\']+)', init.read_text(encoding="utf-8"), re.M
    )
    assert version and not version[1].startswith("0."), (
        "this guard exists for the 1.x era; update it with the version bump"
    )
    offenders = [
        f"{_relative(path)}:{number}: {line.strip()}"
        for path in _iter_docs()
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        )
        if re.search(r"\b0\.x\b", line)
    ]
    assert not offenders, "docs still describe the project as 0.x:\n" + "\n".join(
        offenders
    )


# ============================================================
# Stale prose that outlived the behavior it described
# ============================================================


def test_docs_do_not_advertise_ffmpeg() -> None:
    """No audio/video extension is registered; FFmpeg is not a capability."""
    offenders = [
        f"{_relative(path)}:{number}: {line.strip()}"
        for path in _iter_docs()
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        )
        if "ffmpeg" in line.lower()
    ]
    assert not offenders, "docs still mention FFmpeg:\n" + "\n".join(offenders)


def test_getting_started_pins_no_literal_version() -> None:
    """A version baked into a doc example rots on the next release."""
    for guide in (_EN_GUIDE, _ZH_GUIDE):
        page = guide / "getting-started.md"
        literals = re.findall(
            r"MARKITAI_VERSION\s*=?\s*\"?(\d+\.\d+\.\d+)",
            page.read_text(encoding="utf-8"),
        )
        assert not literals, (
            f"{_relative(page)} hardcodes MARKITAI_VERSION={literals}; use a "
            f"placeholder that cannot go stale"
        )


def test_consent_docs_no_longer_exempt_the_x_twitter_path() -> None:
    """`ask` covers FxTwitter/oEmbed through the shared process decision."""
    stale = {
        _EN_GUIDE: (
            "This public-URL enrichment does not open its own `ask` prompt",
            "It does not open its own prompt under `ask`",
            "only discloses",
        ),
        _ZH_GUIDE: (
            "不会单独弹出 `ask` 确认",
            "不会为 `ask` 单独弹出确认",
            "例外只揭露、不询问",
        ),
    }
    for guide, phrases in stale.items():
        for name in ("fetch-policy.md", "configuration.md"):
            text = (guide / name).read_text(encoding="utf-8")
            for phrase in phrases:
                assert phrase not in text, (
                    f"{_relative(guide / name)} still describes the old "
                    f"X/Twitter consent exemption: {phrase!r}"
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
        "RapidOCR is included as a dependency and works out of the box",
        "RapidOCR 已作为依赖包含，开箱即用",
    )
    for path in _iter_docs():
        text = path.read_text(encoding="utf-8")
        for phrase in stale:
            assert phrase not in text, (
                f"{_relative(path)} still calls OCR a core requirement: {phrase!r}"
            )


# ============================================================
# Anchors, licence, layout: facts that drift silently
# ============================================================


# Absolute guide links and relative ones (`./python-api#…`).
_LINK_RE = re.compile(r"\]\((/(?:zh/)?guide/[^)\s#]*|\./[^)\s#]*)(?:#([^)\s]+))?\)")


def _slugify_heading(text: str) -> str:
    """Approximate VitePress/markdown-it-anchor ids (CJK kept as-is)."""
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


@pytest.mark.parametrize("guide", (_EN_GUIDE, _ZH_GUIDE), ids=("en", "zh"))
def test_internal_anchors_resolve(guide: Path) -> None:
    """A guide link must resolve to a page, and to a heading when it has one.

    Both locales: shortening a page renames headings, and an English link
    to a renamed heading lands at the top of the page without a word.
    """
    offenders: list[str] = []
    for page in sorted(guide.glob("*.md")):
        text = page.read_text(encoding="utf-8")
        for match in _LINK_RE.finditer(text):
            target, anchor = match.group(1), match.group(2)
            if target.startswith("./"):
                candidate = page.parent / target[2:]
            else:
                candidate = _WEBSITE / target.lstrip("/")
            if not candidate.suffix:
                candidate = candidate.with_suffix(".md")
            if not candidate.exists():
                offenders.append(
                    f"{_relative(page)} -> {target} (target page does not exist)"
                )
                continue
            if anchor is not None and unquote(anchor) not in _heading_slugs(candidate):
                offenders.append(
                    f"{_relative(page)} -> {target}#{anchor} "
                    f"(no such heading in {_relative(candidate)})"
                )
    assert not offenders, "broken internal anchors:\n" + "\n".join(offenders)


def test_llms_txt_states_the_licence_accurately() -> None:
    """`llms.txt` is what AI agents quote for compliance; it cannot say "MIT"."""
    text = (_WEBSITE / "public" / "llms.txt").read_text(encoding="utf-8")
    assert "MIT licensed" not in text, (
        "llms.txt claims the whole project is MIT licensed; name the AGPL "
        "PyMuPDF components and point at NOTICE instead"
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


def test_both_comparison_pages_carry_the_same_table() -> None:
    site = _comparison_table((_EN_GUIDE / "comparison.md").read_text(encoding="utf-8"))
    zh_site = _comparison_table(
        (_ZH_GUIDE / "comparison.md").read_text(encoding="utf-8")
    )
    assert len(site) >= 2, "website/guide/comparison.md lost its comparison table"
    assert len(zh_site) == len(site), (
        "the Chinese comparison page has a different number of table rows"
    )


def test_home_feature_cards_stay_bilingual_and_grid_aligned() -> None:
    """One shared feature list feeds both home pages.

    VitePress picks `grid-6` for a multiple of three cards and `grid-4`
    otherwise, so five cards leave one alone in a four-wide row.
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


def test_notice_names_every_website_dependency() -> None:
    """The site is built with third-party works; its NOTICE names each of them.

    Pele arrived to draw the Mermaid diagrams; a dependency added to
    `website/package.json` must be credited in `website/NOTICE` too.
    """
    import json

    package = json.loads((_WEBSITE / "package.json").read_text(encoding="utf-8"))
    names = {
        *package.get("dependencies", {}),
        *package.get("devDependencies", {}),
    }
    notice = (_WEBSITE / "NOTICE").read_text(encoding="utf-8")
    section = notice.split("The markitai.dev website", 1)
    assert len(section) == 2, "website/NOTICE does not cover the markitai.dev website"
    missing = sorted(name for name in names if name.lower() not in section[1].lower())
    assert not missing, f"website/NOTICE does not name {missing}"


# ============================================================
# Release notes render through Vue
# ============================================================

# ``<`` followed by a letter starts a tag for Vue; digits (e.g. "<4-page")
# are safe. Only flag simple word-like tags, the exact shape Vue parses.
_TAG_RE = re.compile(r"<[a-zA-Z][a-zA-Z0-9-]*>")


def _strip_code_spans(line: str) -> str:
    # Remove fenced-marker runs first so odd backtick counts (```lang)
    # don't desync the inline-span pairing on the same line.
    line = line.replace("```", "\x00")
    return re.sub(r"`[^`]*`", "", line)


def test_release_notes_exist() -> None:
    assert [path.name for path in _CHANGELOGS] == ["python-1.x.md", "python-1.x.zh.md"]


@pytest.mark.parametrize("changelog", _CHANGELOGS, ids=lambda p: p.name)
def test_changelog_has_no_bare_angle_bracket_tags(changelog: Path) -> None:
    """A bare `<word>` is an unclosed element to Vue and fails the whole build."""
    offenders: list[str] = []
    in_fence = False
    for lineno, line in enumerate(
        changelog.read_text(encoding="utf-8").splitlines(), 1
    ):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for match in _TAG_RE.finditer(_strip_code_spans(line)):
            offenders.append(f"line {lineno}: {match.group(0)}")
    assert not offenders, (
        "Bare angle-bracket tokens break the VitePress/Vue website build; "
        "wrap them in backticks: " + "; ".join(offenders)
    )
