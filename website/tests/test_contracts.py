"""Keep markitai.dev honest about the markitai it documents.

The site documents markitai 2.x, the Rust CLI on `main`. These guards compare
the pages with that CLI: its `--help` text, its configuration schema
(`markitai config list` / `config get`), its presets and the models
`markitai init` picks. Point MARKITAI_BIN at a built `markitai`:

    cargo build --locked -p markitai-cli
    MARKITAI_BIN=target/debug/markitai uv run --no-project --with pytest \\
        pytest website/tests

Without the variable the checks that run the CLI skip, except in CI, where
they fail. The site-only checks always run. Every CLI call gets a private
HOME and MARKITAI_HOME and no provider keys, so a test never reads your
configuration or sends a request.

The rot this module exists to prevent: one fact (an option name, a default, a
preset) is copied by hand into the CLI, `website/guide/*.md`, and
`website/zh/guide/*.md`. Nothing links those copies, so a renamed flag or a
changed default stays advertised in both languages until a reader trips on it.

* **Fails** when a guide documents a `--flag` the CLI does not accept, or names
  a removed flag outside a section that says it was removed.
* **Fails** when the English and Chinese CLI guides document different option
  sets, which is how half-finished bilingual edits show up.
* **Fails** when the CLI shows an option no page mentions at all, and when a
  top-level command has no reference section.
* **Fails** when a page names a configuration key the schema does not have, or
  a reference table gives a key a default the CLI does not use.
* **Fails** when a preset table disagrees with `markitai --help`.
* **Ignores fenced code blocks** when scanning headings: a shell comment such
  as `# removed in 2.0` inside a snippet must not excuse any rule.

What it cannot catch: wrong descriptions, wrong values for a `Choice` option
in prose, or prose that contradicts behavior. Those still need review.
"""

from __future__ import annotations

import atexit
import functools
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import unquote, urlparse

import pytest

_WEBSITE = Path(__file__).resolve().parents[1]
_REPO = _WEBSITE.parent
_EN_GUIDE = _WEBSITE / "guide"
_ZH_GUIDE = _WEBSITE / "zh" / "guide"
_GUIDES = (_EN_GUIDE, _ZH_GUIDE)
# The release notes the changelog pages publish: 2.x, then the Python 1.x line.
_CHANGELOGS = sorted((_WEBSITE / "changelog").glob("*.md")) + [
    _REPO / "CHANGELOG.md",
    _REPO / "CHANGELOG.zh.md",
]
_BIN_ENV = "MARKITAI_BIN"

# `--flag`. Trailing-hyphen matches are prose globs (`--no-*`), not options.
_FLAG_RE = re.compile(r"(?<![\w-])--[a-zA-Z][\w-]*")

# Flags of other tools that legitimately appear in install and setup snippets.
_FOREIGN_FLAGS = frozenset(
    {
        "--help",  # every command's built-in; not an option to document
        "--version",  # likewise
        "--from",  # uvx --from markitai markitai-mcp
        "--uninstall",  # setup.sh -s -- --uninstall
        "--release",  # cargo build --release --locked -p markitai-cli
        "--locked",  # likewise
        "--ignore-missing",  # sha256sum -c SHA256SUMS --ignore-missing
        "--repo",  # gh attestation verify FILE --repo Ynewtime/markitai
        "--prefix",  # npm --prefix bindings/node run build
    }
)

# A Markdown heading: `#`, optional level, then a space (a bash comment is not one).
_HEADING_RE = re.compile(r"^#{1,6}\s")

# A removed flag may only be named under a heading that says it is removed.
_REMOVAL_MARKERS = ("removed", "已移除")

# Top-level configuration sections; a backticked `section.key` names a key.
_CONFIG_SECTIONS = (
    "batch",
    "cache",
    "fetch",
    "history",
    "image",
    "llm",
    "log",
    "ocr",
    "office",
    "output",
    "presets",
    "prompts",
    "screenshot",
    "security",
)


# ============================================================
# The CLI under test
# ============================================================


@functools.cache
def _binary() -> Path:
    """The built markitai 2.x the pages describe."""
    value = os.environ.get(_BIN_ENV)
    if not value:
        message = f"{_BIN_ENV} names no built markitai (cargo build -p markitai-cli)"
        if os.environ.get("CI"):
            pytest.fail(message)
        pytest.skip(message)
    path = Path(value).resolve()
    assert path.is_file(), f"{_BIN_ENV}={value} is not a file"
    version = _run_at(path, "--version").stdout.strip()
    assert version.startswith("markitai 2."), (
        f"{path} reports {version!r}; these guards are for markitai 2.x"
    )
    return path


@functools.cache
def _state() -> Path:
    """One private HOME/MARKITAI_HOME root for every CLI call of this run."""
    root = Path(tempfile.mkdtemp(prefix="markitai-site-tests-"))
    atexit.register(shutil.rmtree, root, ignore_errors=True)
    return root


def _env(**extra: str) -> dict[str, str]:
    """A clean environment: no provider keys, no user configuration."""
    root = _state()
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(root / "home"),
        "USERPROFILE": str(root / "home"),
        "MARKITAI_HOME": str(root / "markitai-home"),
        "MARKITAI_LANG": "en",
        "NO_COLOR": "1",
        "COLUMNS": "200",
    }
    for name in ("SYSTEMROOT", "TEMP", "TMP"):  # Windows needs these to start
        if name in os.environ:
            env[name] = os.environ[name]
    env.update(extra)
    return env


def _run_at(binary: Path, *args: str, **env: str) -> subprocess.CompletedProcess[str]:
    (_state() / "home").mkdir(exist_ok=True)
    (_state() / "markitai-home").mkdir(exist_ok=True)
    return subprocess.run(
        [str(binary), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=_env(**env),
        cwd=_state(),
        timeout=120,
        check=False,
    )


def _run(*args: str, **env: str) -> subprocess.CompletedProcess[str]:
    return _run_at(_binary(), *args, **env)


@functools.cache
def _help(*command: str) -> str:
    result = _run(*command, "--help")
    assert result.returncode == 0, f"markitai {' '.join(command)} --help: {result.stderr}"
    return result.stdout


def _subcommands(text: str) -> list[str]:
    """Names listed under a help text's `Commands:` heading."""
    match = re.search(r"^Commands:\n((?:  \S.*\n)+)", text, re.MULTILINE)
    if not match:
        return []
    return [line.split()[0] for line in match.group(1).splitlines() if line.split()[0] != "help"]


@functools.cache
def _command_tree() -> tuple[tuple[str, ...], ...]:
    """Every command path, the root `()` included."""
    paths: list[tuple[str, ...]] = []
    pending: list[tuple[str, ...]] = [()]
    while pending:
        path = pending.pop()
        paths.append(path)
        pending.extend((*path, name) for name in _subcommands(_help(*path)))
    return tuple(sorted(paths))


@functools.cache
def _cli_option_names() -> frozenset[str]:
    """Every long option some `--help` names, `--no-*` partners included."""
    names: set[str] = set()
    for path in _command_tree():
        names |= set(_FLAG_RE.findall(_help(*path)))
    return frozenset(name for name in names if not name.endswith("-"))


@functools.cache
def _flag_verdict(flag: str) -> str:
    """`known`, `removed` or `unknown`, from the CLI itself.

    `--help` names an option's hidden partner only in prose (`--no-llm
    disables`), and not always, so a flag the help texts miss is put to the
    parser: `markitai FLAG --help` prints help for an accepted switch and
    fails for an unknown or removed one.
    """
    if flag in _cli_option_names():
        return "known"
    result = _run(flag, "--help")
    if result.returncode == 0:
        return "known"
    if "has been removed" in result.stderr:
        return "removed"
    return "unknown"


@functools.cache
def _config_value(key: str) -> object:
    """`markitai config get KEY` as JSON; KeyError for a key the schema lacks."""
    result = _run("config", "get", key)
    if result.returncode != 0:
        raise KeyError(f"{key}: {result.stderr.strip()}")
    text = result.stdout.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text  # plain strings print without quotes


@functools.cache
def _config_defaults() -> dict:
    result = _run("config", "list")
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


# ============================================================
# Reading the pages
# ============================================================


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
    text = path.read_text(encoding="utf-8")
    heading = ""
    out: list[tuple[str, str]] = []
    for raw_line, prose_line in zip(text.splitlines(), _prose_lines(text)):
        if _HEADING_RE.match(prose_line):
            heading = prose_line
        out.append((heading, raw_line))
    return out


def _relative(path: Path) -> str:
    return path.relative_to(_REPO).as_posix()


def _upgrade_page(path: Path) -> bool:
    return path.name == "upgrading.md"


# ============================================================
# Documented options must exist
# ============================================================


@pytest.mark.parametrize("guide", _GUIDES, ids=("en", "zh"))
def test_cli_guide_documents_only_real_options(guide: Path) -> None:
    """Every `--flag` in the CLI reference must be a flag the CLI accepts."""
    page = guide / "cli.md"
    unknown = sorted(
        flag
        for flag in _flags_in(page) - _FOREIGN_FLAGS
        if _flag_verdict(flag) == "unknown"
    )
    assert not unknown, (
        f"{_relative(page)} documents options the CLI does not accept: "
        f"{unknown}. Either the docs are stale or the flag was renamed."
    )


def test_no_page_names_an_unknown_option() -> None:
    """Outside the CLI reference too: a flag a reader copies must parse."""
    offenders = sorted(
        f"{_relative(path)}: {flag}"
        for path in _iter_docs()
        if not _upgrade_page(path)  # names 1.x options that are gone, on purpose
        for flag in _flags_in(path) - _FOREIGN_FLAGS
        if _flag_verdict(flag) == "unknown"
    )
    assert not offenders, "pages name options markitai does not accept:\n" + "\n".join(
        offenders
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
    """Removed spellings may be named in migration notes, nowhere else."""
    offenders: list[str] = []
    for path in _iter_docs():
        removed = {
            flag for flag in _flags_in(path) if _flag_verdict(flag) == "removed"
        }
        if not removed:
            continue
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


@pytest.mark.parametrize("guide", _GUIDES, ids=("en", "zh"))
def test_cli_reference_covers_every_command(guide: Path) -> None:
    """A command nobody documents is a command nobody finds."""
    page = guide / "cli.md"
    headings = [
        line
        for line in _prose_lines(page.read_text(encoding="utf-8"))
        if _HEADING_RE.match(line)
    ]
    missing = sorted(
        name
        for name in _subcommands(_help())
        if not any(f"markitai {name}" in heading for heading in headings)
    )
    assert not missing, f"{_relative(page)} has no section for: {missing}"
    text = page.read_text(encoding="utf-8")
    unnamed = sorted(
        " ".join(path)
        for path in _command_tree()
        if len(path) > 1 and " ".join(path[1:]) not in text
    )
    assert not unnamed, f"{_relative(page)} never names the subcommands: {unnamed}"


def test_docs_name_every_model_init_picks() -> None:
    """A reader must be able to tell which model each API key selects.

    `markitai init --yes` with one provider key set writes that provider's
    default model; it sends no request and saves no key.
    """
    corpus = "".join(path.read_text(encoding="utf-8") for path in _iter_docs())
    missing: list[str] = []
    for variable in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "DEEPSEEK_API_KEY",
        "OPENROUTER_API_KEY",
    ):
        home = Path(tempfile.mkdtemp(dir=_state()))
        result = _run("init", "--yes", MARKITAI_HOME=str(home), **{variable: "test-key"})
        assert result.returncode == 0, f"init with {variable}: {result.stderr}"
        config = json.loads((home / "config.json").read_text(encoding="utf-8"))
        for entry in config["llm"]["model_list"]:
            model = entry["litellm_params"]["model"]
            if model not in corpus:
                missing.append(f"{variable} -> {model}")
    assert not missing, "defaults the docs never name:\n" + "\n".join(missing)


# ============================================================
# Presets and configuration
# ============================================================


def _help_presets() -> dict[str, set[str]]:
    """`Presets (-p):` in `markitai --help`: name -> the flags it turns on."""
    block = re.search(r"^Presets \(-p\):\n((?:  \S.*\n)+)", _help(), re.MULTILINE)
    assert block, "markitai --help lists no presets"
    presets: dict[str, set[str]] = {}
    for line in block.group(1).splitlines():
        name, _, rest = line.strip().partition(" ")
        presets[name] = {flag[2:] for flag in _FLAG_RE.findall(rest)}
    return presets


def _preset_tables(path: Path) -> list[tuple[list[str], list[list[str]]]]:
    """(header, rows) of every table whose first column lists `minimal`."""
    tables: list[tuple[list[str], list[list[str]]]] = []
    lines = _prose_lines(path.read_text(encoding="utf-8"))
    for index, line in enumerate(lines):
        if not line.startswith("| `minimal`"):
            continue
        if index < 2 or not lines[index - 1].startswith("|-") and not lines[index - 1].startswith("| -"):
            continue
        header = [cell.strip() for cell in lines[index - 2].strip("|").split("|")]
        rows = []
        for row in lines[index:]:
            if not row.startswith("|"):
                break
            rows.append([cell.strip() for cell in row.strip().strip("|").split("|")])
        tables.append((header, rows))
    return tables


_PRESET_COLUMNS = {
    "llm": "llm",
    "alt": "alt",
    "desc": "desc",
    "screenshot": "screenshot",
    "截图": "screenshot",
    "ocr": "ocr",
}


def test_preset_tables_match_the_cli() -> None:
    """Every preset table shows what `markitai --help` says each preset turns on."""
    presets = _help_presets()
    seen = 0
    offenders: list[str] = []
    for path in _iter_docs():
        for header, rows in _preset_tables(path):
            seen += 1
            columns = {
                position: _PRESET_COLUMNS[cell.lower()]
                for position, cell in enumerate(header)
                if cell.lower() in _PRESET_COLUMNS
            }
            for row in rows:
                name = row[0].strip("`")
                if name not in presets:
                    continue
                for position, feature in columns.items():
                    shown = row[position] == "✓"
                    if shown != (feature in presets[name]):
                        offenders.append(f"{_relative(path)}: {name} {feature} shows {row[position]!r}")
    assert seen, "no page carries a preset table"
    assert not offenders, "preset tables disagree with markitai --help:\n" + "\n".join(offenders)


# `llm.model_list[0].litellm_params.model`, `fetch.domain_profiles.<host>` and
# keys inside a map are names of entries, not schema keys.
_KEY_RE = re.compile(r"`((?:" + "|".join(_CONFIG_SECTIONS) + r")(?:\.[a-z_]+)+)`")


def test_configuration_keys_exist() -> None:
    """A backticked `section.key` on any page is a key the schema has."""
    offenders: list[str] = []
    for path in _iter_docs():
        if _upgrade_page(path):
            continue  # names keys 2.0 no longer reads, on purpose
        for key in sorted(set(_KEY_RE.findall(path.read_text(encoding="utf-8")))):
            if key.startswith(("presets.", "fetch.domain_profiles.")):
                continue
            if key.endswith((".db", ".json", ".md")):
                continue  # a file name such as `cache.db`, not a key
            try:
                _config_value(key)
            except KeyError as error:
                offenders.append(f"{_relative(path)}: {error}")
    assert not offenders, "pages name configuration keys the CLI does not know:\n" + "\n".join(
        offenders
    )


# | `section.key` | `default` | ...
_DEFAULT_ROW_RE = re.compile(r"^\| `([a-z_]+(?:\.[a-z_]+)+)` \| `([^`]*)` \|")


@pytest.mark.parametrize("guide", _GUIDES, ids=("en", "zh"))
def test_configuration_reference_defaults_match_the_cli(guide: Path) -> None:
    """A reference row `| key | default |` gives the default the CLI uses."""
    page = guide / "configuration.md"
    offenders: list[str] = []
    rows = 0
    for line in _prose_lines(page.read_text(encoding="utf-8")):
        match = _DEFAULT_ROW_RE.match(line)
        if not match or not match.group(1).startswith(_CONFIG_SECTIONS):
            continue
        rows += 1
        key, shown = match.groups()
        actual = _config_value(key)
        expected = json.dumps(actual) if not isinstance(actual, str) else actual
        if shown not in (expected, json.dumps(actual)) and not _same_number(shown, actual):
            offenders.append(f"{key}: page {shown!r}, CLI {expected!r}")
    assert rows >= 40, f"{_relative(page)} lost its key/default reference tables"
    assert not offenders, f"{_relative(page)} defaults differ from the CLI:\n" + "\n".join(offenders)


def _same_number(shown: str, actual: object) -> bool:
    try:
        return isinstance(actual, (int, float)) and float(shown) == float(actual)
    except ValueError:
        return False


@pytest.mark.parametrize("guide", _GUIDES, ids=("en", "zh"))
def test_full_configuration_example_is_the_cli_default(guide: Path) -> None:
    """The page's `markitai.json with all defaults` is `markitai config list`."""
    text = (guide / "configuration.md").read_text(encoding="utf-8")
    blocks = re.findall(r"```json\n(\{\n  \"batch\".*?)\n```", text, re.DOTALL)
    assert blocks, "the configuration page lost its full default configuration"
    page = _drop_nulls(json.loads(blocks[0]))
    assert page == _config_defaults(), (
        "the full configuration example differs from `markitai config list`"
    )


def _drop_nulls(value: object) -> object:
    if isinstance(value, dict):
        return {key: _drop_nulls(item) for key, item in value.items() if item is not None}
    return value


# ============================================================
# Install, setup scripts and stale 1.x wording
# ============================================================


def test_setup_scripts_have_one_source() -> None:
    """markitai.dev serves scripts/install/, copied at build time, never a fork."""
    tracked = subprocess.run(
        ["git", "ls-files", "website/public"],
        capture_output=True,
        text=True,
        cwd=_REPO,
        check=True,
    ).stdout.split()
    assert "website/public/setup.sh" not in tracked
    assert "website/public/setup.ps1" not in tracked
    for name in ("setup.sh", "setup.ps1"):
        assert (_REPO / "scripts" / "install" / name).is_file()
    package = json.loads((_WEBSITE / "package.json").read_text(encoding="utf-8"))
    for script in ("docs:build", "docs:dev"):
        assert "scripts/setup-scripts.mjs" in package["scripts"][script], (
            f"{script} no longer copies the setup scripts into the site"
        )


def test_built_site_serves_the_tested_setup_scripts() -> None:
    """After `docs:build`, /setup.sh and /setup.ps1 are scripts/install/ byte for byte."""
    dist = _WEBSITE / ".vitepress" / "dist"
    if not dist.is_dir():
        pytest.skip("the site is not built (bun run docs:build)")
    for name in ("setup.sh", "setup.ps1"):
        served = dist / name
        assert served.is_file(), f"the built site has no /{name}"
        assert served.read_bytes() == (_REPO / "scripts" / "install" / name).read_bytes(), (
            f"the built /{name} is not scripts/install/{name}"
        )


def _script_usage(name: str, marker: str) -> str:
    """The install command a setup script documents in its own header."""
    text = (_REPO / "scripts" / "install" / name).read_text(encoding="utf-8")
    for line in text.splitlines():
        if marker in line:
            return line.lstrip("# ").split("  #")[0].strip()
    raise AssertionError(f"scripts/install/{name} documents no install command")


def test_pages_install_with_the_scripts_own_commands() -> None:
    """The install commands on the site are the ones the scripts document."""
    unix = _script_usage("setup.sh", "markitai.dev/setup.sh | sh  ")
    windows = _script_usage("setup.ps1", "markitai.dev/setup.ps1 | iex\"")
    snippet = (_WEBSITE / "snippets" / "setup.md").read_text(encoding="utf-8")
    home = (_WEBSITE / "index.md").read_text(encoding="utf-8")
    for text, where in ((snippet, "snippets/setup.md"), (home, "index.md")):
        assert unix in text, f"website/{where} does not show `{unix}`"
        assert windows in text, f"website/{where} does not show `{windows}`"
    for guide in _GUIDES:
        page = guide / "getting-started.md"
        assert "snippets/setup.md" in page.read_text(encoding="utf-8"), (
            f"{_relative(page)} does not start with the setup script"
        )


def test_pages_uninstall_with_the_scripts_own_commands() -> None:
    """The uninstall commands on Getting Started are the ones the scripts document."""
    unix = _script_usage("setup.sh", "markitai.dev/setup.sh | sh -s -- --uninstall")
    windows = _script_usage("setup.ps1", "-Uninstall  ")
    for guide in _GUIDES:
        page = guide / "getting-started.md"
        text = page.read_text(encoding="utf-8")
        assert unix in text, f"{_relative(page)} does not show `{unix}`"
        assert windows in text, f"{_relative(page)} does not show `{windows}`"


def test_upgrading_rolls_back_with_the_scripts_1x_command() -> None:
    """Going back to 1.x uses the command the setup scripts print for a 1.x version."""
    commands = set()
    for name, pattern in (
        ("setup.sh", r"^PYTHON_1X_INSTALL='(.+)'$"),
        ("setup.ps1", r"^\$Python1xInstall = '(.+)'$"),
    ):
        text = (_REPO / "scripts" / "install" / name).read_text(encoding="utf-8")
        found = re.search(pattern, text, re.MULTILINE)
        assert found, f"scripts/install/{name} has no 1.x install command"
        commands.add(found.group(1))
    assert len(commands) == 1, f"the setup scripts disagree on the 1.x command: {commands}"
    command = commands.pop()
    for guide in _GUIDES:
        page = guide / "upgrading.md"
        assert command in page.read_text(encoding="utf-8"), (
            f"{_relative(page)} does not show the 1.x command `{command}`"
        )


def test_getting_started_pins_no_literal_version() -> None:
    """A version baked into a doc example rots on the next release."""
    for guide in _GUIDES:
        page = guide / "getting-started.md"
        literals = re.findall(
            r"MARKITAI_VERSION\s*=?\s*[\"']?(\d+\.\d+\.\d+)",
            page.read_text(encoding="utf-8"),
        )
        assert not literals, (
            f"{_relative(page)} hardcodes MARKITAI_VERSION={literals}; use a "
            f"placeholder that cannot go stale"
        )


# Python markitai 1.x names that only the upgrade page may still mention.
_PYTHON_ONLY = (
    "markitai[",
    "RapidOCR",
    "PyMuPDF",
    "pymupdf",
    "LiteLLM library",
    "loguru",
    "enable_worker_processes",
    "MARKITAI_PDF_WORKERS",
    "MARKITAI_STATIC_HTTP",
    "MARKITAI_USE_MIRROR",
    "MARKITAI_SETUP_LOG",
    "playwright install",
    "uv tool install \"markitai[",
)


def test_python_only_names_stay_on_the_upgrade_page() -> None:
    """1.x extras, engines and knobs are history, told once on Upgrading."""
    offenders = [
        f"{_relative(path)}:{number}: {line.strip()}"
        for path in _iter_docs()
        if not _upgrade_page(path)
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        for name in _PYTHON_ONLY
        if name in line
    ]
    assert not offenders, "pages still describe Python markitai 1.x:\n" + "\n".join(offenders)


def test_upgrade_page_is_linked_from_getting_started() -> None:
    for guide in _GUIDES:
        assert (guide / "upgrading.md").is_file(), f"{_relative(guide)} has no upgrading.md"
        text = (guide / "getting-started.md").read_text(encoding="utf-8")
        assert "upgrading" in text, f"{_relative(guide / 'getting-started.md')} does not link Upgrading"


def test_docs_do_not_advertise_ffmpeg() -> None:
    """No audio/video reader exists; FFmpeg is not a capability."""
    offenders = [
        f"{_relative(path)}:{number}: {line.strip()}"
        for path in _iter_docs()
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if "ffmpeg" in line.lower()
    ]
    assert not offenders, "docs still mention FFmpeg:\n" + "\n".join(offenders)


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


@pytest.mark.parametrize("guide", _GUIDES, ids=("en", "zh"))
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


# The workbench's "Setup guide" links (lib/guides.ts): a page per subscription
# runtime and a section per reason it lists no models, on markitai.dev.
_GUIDES_TS = _REPO / "crates" / "markitai-cli" / "src" / "server" / "web" / "src" / "lib" / "guides.ts"


def _guides_table(text: str, name: str) -> dict[str, str]:
    block = re.search(rf"const {name}: Record<[^>]+> = \{{(.*?)\n\}};", text, re.S)
    assert block, f"{_relative(_GUIDES_TS)} has no {name} table"
    pairs = dict(re.findall(r'"?([\w-]+)"?:\s*"([^"]*)"', block.group(1)))
    assert pairs, f"{_relative(_GUIDES_TS)}: {name} is empty"
    return pairs


def _workbench_guide_links() -> list[str]:
    """Every address `guideLinks()` lists: English and Chinese (Japanese pages use English)."""
    text = _GUIDES_TS.read_text(encoding="utf-8")
    assert 'export const SITE = "https://markitai.dev";' in text
    pages = set(_guides_table(text, "RUNTIME_PAGES").values())
    sections = set(_guides_table(text, "RUNTIME_SECTIONS").values())
    return sorted(
        f"https://markitai.dev{locale}/guide/{page}#{section}"
        for locale in ("", "/zh")
        for page in pages
        for section in sections
    )


def test_workbench_guide_links_reach_a_page_and_its_section() -> None:
    """A provider page's Setup guide opens an existing page of the site, in its
    language, at a heading it has; after `docs:build`, the built page carries
    that id too."""
    dist = _WEBSITE / ".vitepress" / "dist"
    links = _workbench_guide_links()
    offenders: list[str] = []
    for link in links:
        url = urlparse(link)
        page = url.path.lstrip("/")
        source = _WEBSITE / f"{page}.md"
        if not source.is_file():
            offenders.append(f"{link} (no website/{page}.md)")
            continue
        if url.fragment not in _heading_slugs(source):
            offenders.append(f"{link} (no heading #{url.fragment} in website/{page}.md)")
        built = dist / f"{page}.html"
        if dist.is_dir() and f'id="{url.fragment}"' not in (
            built.read_text(encoding="utf-8") if built.is_file() else ""
        ):
            offenders.append(f"{link} (the built site has no {page}.html with id {url.fragment})")
    assert len(links) >= 8, links
    assert not offenders, "workbench guide links that land nowhere:\n" + "\n".join(offenders)


def test_both_locales_have_the_same_pages() -> None:
    english = sorted(path.name for path in _EN_GUIDE.glob("*.md"))
    chinese = sorted(path.name for path in _ZH_GUIDE.glob("*.md"))
    assert english == chinese, f"guide pages differ: en {english}, zh {chinese}"


def test_every_page_has_a_description_and_a_sidebar_entry() -> None:
    """config.ts: a page without a description shares the site's one."""
    config = (_WEBSITE / ".vitepress" / "config.ts").read_text(encoding="utf-8")
    for guide, prefix in ((_EN_GUIDE, "/guide/"), (_ZH_GUIDE, "/zh/guide/")):
        for page in sorted(guide.glob("*.md")):
            route = prefix + page.stem
            assert f"'{route}':" in config, f"config.ts has no description for {route}"
            assert f"link: '{route}'" in config, f"config.ts has no sidebar entry for {route}"


def test_llms_txt_points_at_the_licences() -> None:
    """`llms.txt` is what AI agents quote for compliance."""
    text = (_WEBSITE / "public" / "llms.txt").read_text(encoding="utf-8")
    assert "NOTICE" in text, "llms.txt must point at NOTICE for third-party licences"
    for stale in ("AGPL", "PyMuPDF", "uv tool install", "markitai["):
        assert stale not in text, f"llms.txt still describes Python markitai 1.x: {stale!r}"


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
    zh_site = _comparison_table((_ZH_GUIDE / "comparison.md").read_text(encoding="utf-8"))
    assert len(site) >= 2, "website/guide/comparison.md lost its comparison table"
    assert len(zh_site) == len(site), (
        "the Chinese comparison page has a different number of table rows"
    )


def test_home_feature_cards_stay_bilingual_and_grid_aligned() -> None:
    """One shared feature list feeds both home pages.

    VitePress picks `grid-6` for a multiple of three cards and `grid-4`
    otherwise, so five cards leave one alone in a four-wide row.
    """
    features = (_WEBSITE / ".vitepress" / "theme" / "features.ts").read_text(encoding="utf-8")
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
    assert [path.name for path in _CHANGELOGS] == [
        "python-1.x.md",
        "python-1.x.zh.md",
        "CHANGELOG.md",
        "CHANGELOG.zh.md",
    ]


@pytest.mark.parametrize("changelog", _CHANGELOGS, ids=lambda p: p.name)
def test_changelog_has_no_bare_angle_bracket_tags(changelog: Path) -> None:
    """A bare `<word>` is an unclosed element to Vue and fails the whole build."""
    offenders: list[str] = []
    in_fence = False
    for lineno, line in enumerate(changelog.read_text(encoding="utf-8").splitlines(), 1):
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


# ============================================================
# The changelog pages publish released versions only
# ============================================================

_SECTION_RE = re.compile(r"^## \[(\d+\.\d+\.\d+)\]", re.MULTILINE)


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=Site Test", "-c", "user.email=site@example.invalid", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
        timeout=60,
    ).stdout


def _bun() -> str:
    bun = shutil.which("bun")
    if bun is None:
        message = "bun is not on PATH (the changelog script runs under Bun)"
        if os.environ.get("CI"):
            pytest.fail(message)
        pytest.skip(message)
    return bun


def _notes(title: str, intro: str, *sections: str) -> str:
    body = [f"# {title}", "", intro, ""]
    for version in sections:
        body += [f"## [{version}] - 2026-01-01", "", f"- Fixed: item of {version}", ""]
    return "\n".join(body)


def _changelog_repo(root: Path, current: tuple[str, ...], python: tuple[str, ...]) -> Path:
    """A repository with release notes, the changelog script and tags v1.0.0, v2.0.0."""
    site = root / "website"
    (site / "scripts").mkdir(parents=True)
    (site / "changelog").mkdir()
    (site / "zh").mkdir()
    shutil.copy(_WEBSITE / "scripts" / "changelog.mjs", site / "scripts")
    link = "[1.0.0 changelog](https://github.com/Ynewtime/markitai/blob/v1.0.0/CHANGELOG{}.md)"
    for suffix in ("", ".zh"):
        (root / f"CHANGELOG{suffix}.md").write_text(
            _notes("Changelog", "1.x notes: " + link.format(suffix) + ".", *current),
            encoding="utf-8",
        )
        (site / "changelog" / f"python-1.x{suffix}.md").write_text(
            _notes("Changelog", "Python intro, not shown.", *python), encoding="utf-8"
        )
    _git(root, "init", "-q")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "notes")
    _git(root, "tag", "v1.0.0")
    _git(root, "tag", "v2.0.0")
    return site


def _changelog(site: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [_bun(), "scripts/changelog.mjs", *args],
        cwd=site,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        timeout=60,
    )


def test_changelog_pages_put_released_2x_first_and_gate_by_tag(tmp_path: Path) -> None:
    """2.x then 1.x, newest first; a 2.x section waits for its tag.

    The release commit adds `## [2.0.1]` (or keeps `## [Unreleased]`) at the
    top of CHANGELOG.md before the tag exists: --check passes and the page
    leaves it out until `v2.0.1` is tagged.
    """
    site = _changelog_repo(tmp_path, ("Unreleased", "2.0.1", "2.0.0"), ("1.0.0", "0.9.0"))
    check = _changelog(site, "--check")
    assert check.returncode == 0, check.stdout + check.stderr
    build = _changelog(site)
    assert build.returncode == 0, build.stdout + build.stderr
    for page in ("changelog.md", "zh/changelog.md"):
        text = (site / page).read_text(encoding="utf-8")
        assert _SECTION_RE.findall(text) == ["2.0.0", "1.0.0", "0.9.0"], page
        assert "Unreleased" not in text, page
        assert "Python intro" not in text, page
        assert text.index("[2.0.0]") < text.index("{#python-1x}") < text.index("[1.0.0]"), page
        assert "](#python-1x)" in text and "github.com" not in text, (
            f"{page}: the intro's 1.x link should point into the page"
        )

    _git(tmp_path, "tag", "v2.0.1")
    assert _changelog(site).returncode == 0
    text = (site / "changelog.md").read_text(encoding="utf-8")
    assert _SECTION_RE.findall(text) == ["2.0.1", "2.0.0", "1.0.0", "0.9.0"]


def test_changelog_check_fails_on_lost_or_one_language_sections(tmp_path: Path) -> None:
    """An untagged section below a released one would vanish; so would a one-language one."""
    site = _changelog_repo(tmp_path, ("2.0.0", "2.0.0-typo"), ("1.0.0",))
    check = _changelog(site, "--check")
    assert check.returncode == 1 and "2.0.0-typo" in check.stderr, check.stderr

    zh = tmp_path / "CHANGELOG.zh.md"
    zh.write_text(_notes("更新日志", "简介。", "1.9.9"), encoding="utf-8")
    (tmp_path / "CHANGELOG.md").write_text(_notes("Changelog", "Intro.", "2.0.0"), encoding="utf-8")
    _git(tmp_path, "tag", "v1.9.9")
    check = _changelog(site, "--check")
    assert check.returncode == 1 and "publishes" in check.stderr, check.stderr


def test_built_changelog_pages_carry_the_released_2x_notes() -> None:
    """After `docs:build`: every tagged 2.x section, then the 1.x heading, then 1.x."""
    dist = _WEBSITE / ".vitepress" / "dist"
    if not dist.is_dir():
        pytest.skip("the site is not built (bun run docs:build)")
    tags = {line.strip().removeprefix("v") for line in _git(_REPO, "tag", "--list", "v*").split()}
    for page, current in (
        ("changelog.html", _REPO / "CHANGELOG.md"),
        ("zh/changelog.html", _REPO / "CHANGELOG.zh.md"),
    ):
        html = (dist / page).read_text(encoding="utf-8")
        headings = re.findall(r"<h2[^>]*\bid=\"([^\"]+)\"[^>]*>(.*?)</h2>", html, re.DOTALL)
        shown = [
            "1.x" if anchor == "python-1x" else m.group(1)
            for anchor, text in headings
            if anchor == "python-1x" or (m := re.search(r"\[(\d+\.\d+\.\d+)\]", text))
        ]
        assert "1.x" in shown, f"{page} has no heading where the 1.x notes start"
        written = _SECTION_RE.findall(current.read_text(encoding="utf-8"))
        released = [version for version in written if version in tags]
        assert released, f"{current.name} has no tagged section (fetch the tags)"
        assert shown[: shown.index("1.x")] == released, (
            f"{page} shows 2.x {shown[: shown.index('1.x')]}, the tagged sections are {released}"
        )
        older = shown[shown.index("1.x") + 1 :]
        assert older and older[0].startswith("1."), f"{page}: the 1.x part starts with {older[:1]}"
