"""Pytest configuration and fixtures."""

import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from markitai.config import LLMConfig, PromptsConfig


# =============================================================================
# Path Fixtures
# =============================================================================


@pytest.fixture
def fixtures_dir() -> Path:
    """Return the path to the fixtures directory."""
    return Path(__file__).parent / "fixtures"


# =============================================================================
# Sample Content Fixtures
# =============================================================================


@pytest.fixture
def sample_config_dict() -> dict:
    """Return sample configuration dictionary."""
    return {
        "output": {
            "dir": "./custom_output",
            "on_conflict": "overwrite",
        },
        "llm": {
            "enabled": True,
            "concurrency": 5,
        },
        "image": {
            "compress": True,
            "quality": 90,
        },
    }


# Rich and rich-click read the terminal from the environment: FORCE_COLOR
# (and its kin) forces ANSI colour even into CliRunner's captured output,
# and the width comes from COLUMNS, else from the terminal pytest runs in,
# so the same message wrapped differently on a laptop and a CI runner. The
# CLI modules build their shared Console at import, which fixes both there,
# so this runs before any test module imports markitai rather than in a
# fixture. Tests that check colour or wrapping set their own environment.
# GITHUB_ACTIONS stays, so CI still exercises it (rich-click forces colour
# under it; assertions on rendered help strip ANSI). A markitai child process
# gets tests.terminal_env.plain_env(), without any of them.
_FORCED_COLOUR_ENV = ("FORCE_COLOR", "PY_COLORS", "CLICOLOR_FORCE", "TTY_COMPATIBLE")


# On Windows a pipe or file gets the Unicode marks only while the console's
# output code page is UTF-8 (markitai.utils.term._windows_unicode). The test
# process's own output is a pipe, and the console of a CI step or an SSH
# session is at the OEM code page (437), so the marks this process and the
# CLI it starts print would follow the machine. The session runs at 65001 and
# gives the console its code page back at the end; tests that check the other
# code pages fake them, or start a console of their own.
_ORIGINAL_CODE_PAGE: int | None = None


def _console_code_page(page: int | None = None) -> int:
    """The console's output code page (0 without one); set it first if given."""
    import ctypes

    kernel32 = ctypes.WinDLL("kernel32")  # type: ignore[attr-defined]
    if page is not None:
        kernel32.SetConsoleOutputCP(page)
    return int(kernel32.GetConsoleOutputCP())


def pytest_configure(config: pytest.Config) -> None:
    global _ORIGINAL_CODE_PAGE
    for name in _FORCED_COLOUR_ENV:
        os.environ.pop(name, None)
    os.environ["COLUMNS"] = "80"
    # TERM=dumb (some editors' consoles) picks the ASCII marks at import;
    # tests that want them set it themselves
    if os.environ.get("TERM") == "dumb":
        del os.environ["TERM"]
    # LiteLLM reads its model table once, at its first import: a test module
    # that imports litellm before markitai.llm would otherwise give the whole
    # session the upstream table fetched over the network (no openrouter/
    # rows on 2026-10-18) instead of the bundled one markitai and doctor read
    from markitai.utils.litellm_import import prepare_litellm_import

    prepare_litellm_import()
    # That import brought loguru in, whose default sink keeps the sys.stderr
    # of the moment: here the process's own, as capture is suspended between
    # loading conftests and collecting. Its INFO/DEBUG lines then reached
    # file descriptor 2, so capfd saw them. Dropped here, the sink is added
    # again during collection (below), where it writes to pytest's capture
    # as it did when the first import happened while collecting.
    from loguru import logger

    logger.remove()
    if sys.platform == "win32":
        page = _console_code_page()
        if page not in (0, 65001):
            _ORIGINAL_CODE_PAGE = page
            _console_code_page(65001)


_loguru_sink_added = False


@pytest.hookimpl(wrapper=True, trylast=True)
def pytest_make_collect_report(collector: pytest.Collector) -> Iterator[Any]:
    """Give loguru its stderr sink back inside the capture that collecting a
    file runs under (trylast: inside the capture plugin's own wrapper, which
    resumes capture for files only)."""
    global _loguru_sink_added
    if not _loguru_sink_added and isinstance(collector, pytest.File):
        _loguru_sink_added = True
        from loguru import logger

        logger.add(sys.stderr)
    return (yield)


@pytest.fixture(autouse=True)
def _litellm_registrations_stay_in_their_test(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    """Give back the LiteLLM model rows a test's registrations changed.

    A LiteLLM Router registers each deployment's ``model_info`` into the
    process-wide ``litellm.model_cost`` (and markitai caches what it reads
    there in ``get_model_info_cached``): one test's
    ``ModelInfo(supports_vision=False)`` for deepseek/deepseek-flash made the
    model text-only for every later test. Rows that existed before the test
    are put back; new rows stay, as markitai's own catalog rows may first be
    registered inside a test.
    """
    litellm = sys.modules.get("litellm")
    if litellm is None or not hasattr(litellm, "register_model"):
        yield
        return
    real = litellm.register_model
    before: dict[str, dict[str, Any]] = {}

    def register_model(*args: Any, **kwargs: Any) -> Any:
        if not before:
            before.update({k: dict(v) for k, v in litellm.model_cost.items()})
        return real(*args, **kwargs)

    monkeypatch.setattr(litellm, "register_model", register_model)
    yield
    if not before:
        return
    for key, row in before.items():
        if litellm.model_cost.get(key) != row:
            litellm.model_cost[key] = row
    # What register_model itself calls after changing the table: drops
    # get_model_info's caches and the case-insensitive index
    invalidate = getattr(
        sys.modules.get("litellm.utils"), "_invalidate_model_cost_lowercase_map", None
    )
    if invalidate is not None:
        invalidate()
    models = sys.modules.get("markitai.llm.models")
    if models is not None:
        models._model_info_cache.clear()


@pytest.fixture(autouse=True)
def _loguru_sinks_stay_in_their_test(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    """Remove the loguru sinks a test added (setup_logging run by the CLI).

    A console sink left behind by a CLI test wrote every later test's log
    lines too, so a test that adds its own sink saw each line twice.
    """
    from loguru import logger

    real = logger.add
    added: list[int] = []

    def add(*args: Any, **kwargs: Any) -> int:
        handler_id = real(*args, **kwargs)
        added.append(handler_id)
        return handler_id

    monkeypatch.setattr(logger, "add", add)
    yield
    for handler_id in added:
        try:
            logger.remove(handler_id)
        except ValueError:  # already removed by the test (or setup_logging)
            pass


def pytest_unconfigure(config: pytest.Config) -> None:
    if sys.platform == "win32" and _ORIGINAL_CODE_PAGE is not None:
        _console_code_page(_ORIGINAL_CODE_PAGE)


# pytest puts the running test's id in PYTEST_CURRENT_TEST; Windows refuses
# an environment variable over 32,767 characters, and a 1 MiB id there failed
# setup and teardown, then took 50 minutes to report. Parametrize long values
# with ``ids=``. An id is measured from the package folder, so the limit is
# the same whether pytest runs in packages/markitai (CI) or at the repository
# root (CONTRIBUTING's ``uv run pytest``), where every id gains a
# ``packages/markitai/`` prefix.
MAX_TEST_ID = 255
_ROOT_PREFIX = "packages/markitai/"


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    long_ids = [
        item.nodeid
        for item in items
        if len(item.nodeid.removeprefix(_ROOT_PREFIX)) > MAX_TEST_ID
    ]
    if long_ids:
        shown = "\n".join(f"  {nodeid[:120]}... ({len(nodeid)})" for nodeid in long_ids)
        raise pytest.UsageError(
            f"test ids over {MAX_TEST_ID} characters (give them short ids=):\n{shown}"
        )


@pytest.fixture(autouse=True)
def _isolate_global_cache_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Keep the default PersistentCache out of the real ``~/.markitai``.

    Many tests build ``LLMProcessor`` without ``cache_global_dir``; the
    default would open — and write test entries into — the user's real
    ``~/.markitai/cache.db``. Explicit ``global_dir`` arguments are
    unaffected.
    """
    monkeypatch.setattr(
        "markitai.llm.cache.DEFAULT_GLOBAL_CACHE_DIR",
        str(tmp_path / "global-cache"),
    )


@pytest.fixture(autouse=True)
def _no_pdf_worker_processes(monkeypatch: pytest.MonkeyPatch):
    """Keep PDF extraction in-process unless a test turns the pool on.

    ``markitai.cli.main.main()`` enables the worker pool for the whole
    process; without this, one test calling it would make every later PDF
    test in the same worker spawn extraction processes.
    """
    from markitai.converter import pdf_parallel

    monkeypatch.setattr(pdf_parallel, "_enabled", False)
    # shutdown_pool below (and in tests) marks a stop that lasts until
    # enable(): each test starts with the pool not stopped
    monkeypatch.setattr(pdf_parallel, "_stopping", False)
    yield
    pdf_parallel.shutdown_pool()


@pytest.fixture(autouse=True)
def _x_posts_use_the_default_browser(monkeypatch: pytest.MonkeyPatch):
    """Keep X posts on the browser a test mocks.

    The default ``X_CAPTURE = "direct"`` launches a second, real Chromium for
    x.com; a test that mocks only the default browser would then reach the
    network wherever Chromium is installed. Tests of the direct path set it
    back themselves.
    """
    from markitai import fetch_playwright

    monkeypatch.setattr(fetch_playwright, "X_CAPTURE", "embed")


@pytest.fixture(autouse=True)
def _isolate_user_config_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point ``Path.home()`` at a temp dir so no test reads the developer's
    real ``~/.markitai``.

    ``cli/main.py`` and ``serve/app.py`` load ``~/.markitai/.env`` (into
    ``os.environ``) at import/startup and ``ConfigManager`` reads
    ``~/.markitai/config.json``. A developer with real LLM keys and a model
    config configured would otherwise pollute provider-detection,
    capabilities, and doctor tests that assume a clean environment — the
    app re-loads the keys after a test deletes them, so deleting env vars
    alone is not enough. Redirecting home makes the whole suite hermetic.

    Tests that intentionally probe config resolution set ``MARKITAI_CONFIG``
    or a config path themselves and are unaffected (their own monkeypatch
    runs after this fixture).
    """
    fake_home = tmp_path / "fake-home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", staticmethod(lambda: fake_home))
    monkeypatch.setenv("HOME", str(fake_home))  # expanduser("~") follows too
    monkeypatch.setenv("USERPROFILE", str(fake_home))  # ...on Windows, via this one
    # DEFAULT_USER_CONFIG_DIR is computed at import time from Path.home();
    # re-point it so ConfigManager/auth/init never see the developer's real
    # ~/.markitai/config.json either.
    monkeypatch.setattr(
        "markitai.config.ConfigManager.DEFAULT_USER_CONFIG_DIR",
        fake_home / ".markitai",
    )
    # Scrub any keys the module-level load_dotenv already placed in os.environ.
    # The provider map is the source of truth; the rest are read directly by
    # provider code that is not in it.
    #
    # The subscription providers take credentials from the environment ahead of
    # their config file, so an ambient token short-circuits every config-file
    # test into the env branch: `GH_TOKEN` or `GITHUB_TOKEN` is set on any
    # machine with the gh CLI configured, and in CI. A test that exercises the
    # env branch sets the variable itself, after this fixture.
    from markitai.constants import PROVIDER_API_KEY_ENV, PROVIDER_API_KEY_ENV_NAMES

    for key in {
        *PROVIDER_API_KEY_ENV.values(),
        # The other names LiteLLM reads the same keys from (GOOGLE_API_KEY,
        # OR_API_KEY, ANTHROPIC_AUTH_TOKEN): a model with one of them set
        # is not held back for a missing key
        *(name for names in PROVIDER_API_KEY_ENV_NAMES.values() for name in names),
        "MISTRAL_API_KEY",
        "CLOUDFLARE_API_TOKEN",
        # Copilot: `copilot login --help`'s own precedence order.
        "COPILOT_GITHUB_TOKEN",
        "GH_TOKEN",
        "GITHUB_TOKEN",
        # Claude Code: cloud-backend selectors, checked before the CLI's state.
        "CLAUDE_CODE_USE_BEDROCK",
        "CLAUDE_CODE_USE_VERTEX",
        "CLAUDE_CODE_USE_FOUNDRY",
    }:
        monkeypatch.delenv(key, raising=False)


# =============================================================================
# Fetch Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def _reset_remote_fetch_consent(monkeypatch: pytest.MonkeyPatch):
    """Isolate remote consent and keep unit tests independent of live DNS."""
    from markitai import fetch, fetch_policy

    async def resolve_public_test_host(_hostname: str) -> tuple[str, ...]:
        return ("93.184.216.34",)

    monkeypatch.delenv("MARKITAI_NO_REMOTE_FETCH", raising=False)
    monkeypatch.setattr(
        fetch_policy,
        "resolve_hostname_addresses",
        resolve_public_test_host,
    )
    fetch.reset_remote_consent()
    fetch.reset_explicit_fallback_decision()
    yield
    fetch.reset_remote_consent()
    fetch.reset_explicit_fallback_decision()


@pytest.fixture(autouse=True)
def _reset_fetch_cache_policy():
    """The CLI registers cfg.cache's fetch-cache policy on the process-wide
    session; keep one test's --no-cache-for/TTL from leaking into the next."""
    from markitai.constants import DEFAULT_FETCH_CACHE_TTL_SECONDS
    from markitai.fetch_session import get_default_session

    def reset() -> None:
        get_default_session().configure_fetch_cache(
            ttl_seconds=DEFAULT_FETCH_CACHE_TTL_SECONDS, no_cache_patterns=[]
        )

    reset()
    yield
    reset()


# =============================================================================
# CLI Fixtures
# =============================================================================


@pytest.fixture
def cli_runner():
    """Return a CLI test runner."""
    from click.testing import CliRunner

    return CliRunner()


# =============================================================================
# LLM Configuration Fixtures
# =============================================================================


@pytest.fixture
def llm_config() -> "LLMConfig":
    """Return a test LLM configuration."""
    from markitai.config import LiteLLMParams, LLMConfig, ModelConfig

    return LLMConfig(
        enabled=True,
        model_list=[
            ModelConfig(
                model_name="default",
                litellm_params=LiteLLMParams(
                    model="openai/gpt-4o-mini",
                    api_key="test-key",
                ),
            ),
        ],
        concurrency=2,
    )


@pytest.fixture
def prompts_config() -> "PromptsConfig":
    """Return a test prompts configuration."""
    from markitai.config import PromptsConfig

    return PromptsConfig()


# =============================================================================
# Test File Fixtures
# =============================================================================


@pytest.fixture
def sample_txt_file(tmp_path: Path) -> Path:
    """Create a sample text file for testing."""
    txt_file = tmp_path / "sample.txt"
    txt_file.write_text("# Test Document\n\nThis is test content.", encoding="utf-8")
    return txt_file


@pytest.fixture
def sample_md_file(tmp_path: Path) -> Path:
    """Create a sample markdown file for testing."""
    md_file = tmp_path / "sample.md"
    md_file.write_text(
        "# Test Document\n\nThis is test content.\n\n## Section 1\n\nSome text here.\n",
        encoding="utf-8",
    )
    return md_file


# =============================================================================
# Image Test Utilities
# =============================================================================


@pytest.fixture
def create_test_image():
    """Factory fixture for creating test images.

    Usage:
        def test_something(create_test_image):
            png_bytes = create_test_image(100, 100, "red")
    """
    import io

    from PIL import Image

    def _create(width: int = 100, height: int = 100, color: str = "red") -> bytes:
        img = Image.new("RGB", (width, height), color)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return buffer.getvalue()

    return _create


@pytest.fixture
def sample_png_bytes() -> bytes:
    """Return minimal valid PNG bytes for testing."""
    # Minimal 1x1 red PNG
    return bytes(
        [
            0x89,
            0x50,
            0x4E,
            0x47,
            0x0D,
            0x0A,
            0x1A,
            0x0A,  # PNG signature
            0x00,
            0x00,
            0x00,
            0x0D,
            0x49,
            0x48,
            0x44,
            0x52,  # IHDR chunk
            0x00,
            0x00,
            0x00,
            0x01,
            0x00,
            0x00,
            0x00,
            0x01,  # 1x1 pixel
            0x08,
            0x02,
            0x00,
            0x00,
            0x00,
            0x90,
            0x77,
            0x53,
            0xDE,
            0x00,
            0x00,
            0x00,
            0x0C,
            0x49,
            0x44,
            0x41,
            0x54,
            0x08,
            0xD7,
            0x63,
            0xF8,
            0xCF,
            0xC0,
            0x00,
            0x00,
            0x00,
            0x03,
            0x00,
            0x01,
            0x00,
            0x05,
            0xFE,
            0xD4,
            0xEF,
            0x00,
            0x00,
            0x00,
            0x00,
            0x49,
            0x45,
            0x4E,
            0x44,
            0xAE,
            0x42,
            0x60,
            0x82,
        ]
    )
