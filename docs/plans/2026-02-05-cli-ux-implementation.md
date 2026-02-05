# CLI UX 统一视觉系统实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 实现 CLI 统一视觉系统，减少日志噪音，创建一致的用户体验

**Architecture:** 创建 ui.py 组件模块 + i18n.py 多语言支持，逐步改造各命令，保持向后兼容

**Tech Stack:** Rich（已有依赖），Click（CLI 框架），Loguru（日志）

---

## Task 1: 创建 UI 组件模块

**Files:**
- Create: `packages/markitai/src/markitai/cli/ui.py`
- Create: `packages/markitai/tests/unit/cli/test_ui.py`

**Step 1: 编写失败测试**

```python
# packages/markitai/tests/unit/cli/test_ui.py
"""Tests for unified UI components."""

from __future__ import annotations

import io

import pytest
from rich.console import Console

from markitai.cli import ui


class TestUIComponents:
    """Test UI component functions."""

    def test_symbols_defined(self):
        """Test that all UI symbols are defined."""
        assert ui.MARK_SUCCESS == "✓"
        assert ui.MARK_ERROR == "✗"
        assert ui.MARK_WARNING == "!"
        assert ui.MARK_INFO == "•"
        assert ui.MARK_TITLE == "◆"
        assert ui.MARK_LINE == "│"

    def test_title_output(self, capsys):
        """Test title function output."""
        test_console = Console(file=io.StringIO(), force_terminal=True)
        ui.title("测试标题", console=test_console)
        output = test_console.file.getvalue()
        assert "◆" in output
        assert "测试标题" in output

    def test_success_output(self, capsys):
        """Test success function output."""
        test_console = Console(file=io.StringIO(), force_terminal=True)
        ui.success("操作成功", console=test_console)
        output = test_console.file.getvalue()
        assert "✓" in output
        assert "操作成功" in output

    def test_error_with_detail(self, capsys):
        """Test error function with detail."""
        test_console = Console(file=io.StringIO(), force_terminal=True)
        ui.error("操作失败", detail="详细信息", console=test_console)
        output = test_console.file.getvalue()
        assert "✗" in output
        assert "操作失败" in output
        assert "│" in output
        assert "详细信息" in output

    def test_warning_output(self, capsys):
        """Test warning function output."""
        test_console = Console(file=io.StringIO(), force_terminal=True)
        ui.warning("警告信息", console=test_console)
        output = test_console.file.getvalue()
        assert "!" in output
        assert "警告信息" in output

    def test_info_output(self, capsys):
        """Test info function output."""
        test_console = Console(file=io.StringIO(), force_terminal=True)
        ui.info("信息项", console=test_console)
        output = test_console.file.getvalue()
        assert "•" in output
        assert "信息项" in output

    def test_step_output(self, capsys):
        """Test step function output."""
        test_console = Console(file=io.StringIO(), force_terminal=True)
        ui.step("处理中...", console=test_console)
        output = test_console.file.getvalue()
        assert "│" in output
        assert "处理中..." in output

    def test_summary_output(self, capsys):
        """Test summary function output."""
        test_console = Console(file=io.StringIO(), force_terminal=True)
        ui.summary("完成：3 成功", console=test_console)
        output = test_console.file.getvalue()
        assert "✓" in output
        assert "完成：3 成功" in output

    def test_section_output(self, capsys):
        """Test section function output."""
        test_console = Console(file=io.StringIO(), force_terminal=True)
        ui.section("必需依赖", console=test_console)
        output = test_console.file.getvalue()
        assert "必需依赖" in output
```

**Step 2: 运行测试验证失败**

Run: `uv run pytest packages/markitai/tests/unit/cli/test_ui.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'markitai.cli.ui'"

**Step 3: 编写最小实现**

```python
# packages/markitai/src/markitai/cli/ui.py
"""Unified UI components for Markitai CLI.

This module provides consistent visual elements across all CLI commands,
implementing the Clack-style design language.

Usage:
    from markitai.cli import ui

    ui.title("系统检查")
    ui.success("Python 3.13.1")
    ui.error("文件不存在", detail="请检查路径")
    ui.summary("检查完成：3 通过")
"""

from __future__ import annotations

from rich.console import Console

from markitai.cli.console import get_console

# Unified symbols
MARK_SUCCESS = "✓"
MARK_ERROR = "✗"
MARK_WARNING = "!"
MARK_INFO = "•"
MARK_TITLE = "◆"
MARK_LINE = "│"


def title(text: str, *, console: Console | None = None) -> None:
    """Display a section title.

    Args:
        text: Title text
        console: Optional console instance (for testing)
    """
    c = console or get_console()
    c.print(f"[cyan]{MARK_TITLE}[/] [bold]{text}[/]")
    c.print()


def success(text: str, *, console: Console | None = None) -> None:
    """Display a success item.

    Args:
        text: Success message
        console: Optional console instance (for testing)
    """
    c = console or get_console()
    c.print(f"  [green]{MARK_SUCCESS}[/] {text}")


def error(
    text: str, *, detail: str | None = None, console: Console | None = None
) -> None:
    """Display an error item.

    Args:
        text: Error message
        detail: Optional detail text (shown indented below)
        console: Optional console instance (for testing)
    """
    c = console or get_console()
    c.print(f"  [red]{MARK_ERROR}[/] {text}")
    if detail:
        c.print(f"    [dim]{MARK_LINE} {detail}[/]")


def warning(
    text: str, *, detail: str | None = None, console: Console | None = None
) -> None:
    """Display a warning item.

    Args:
        text: Warning message
        detail: Optional detail text (shown indented below)
        console: Optional console instance (for testing)
    """
    c = console or get_console()
    c.print(f"  [yellow]{MARK_WARNING}[/] {text}")
    if detail:
        c.print(f"    [dim]{MARK_LINE} {detail}[/]")


def info(text: str, *, console: Console | None = None) -> None:
    """Display an info item.

    Args:
        text: Info message
        console: Optional console instance (for testing)
    """
    c = console or get_console()
    c.print(f"  [dim]{MARK_INFO}[/] {text}")


def step(text: str, *, console: Console | None = None) -> None:
    """Display a progress step.

    Args:
        text: Step message
        console: Optional console instance (for testing)
    """
    c = console or get_console()
    c.print(f"  [dim]{MARK_LINE}[/] {text}")


def section(text: str, *, console: Console | None = None) -> None:
    """Display a section header (without symbol).

    Args:
        text: Section name
        console: Optional console instance (for testing)
    """
    c = console or get_console()
    c.print(f"[bold]{text}[/]")


def summary(text: str, *, console: Console | None = None) -> None:
    """Display a summary line.

    Args:
        text: Summary text
        console: Optional console instance (for testing)
    """
    c = console or get_console()
    c.print()
    c.print(f"[green]{MARK_SUCCESS}[/] {text}")
```

**Step 4: 运行测试验证通过**

Run: `uv run pytest packages/markitai/tests/unit/cli/test_ui.py -v`
Expected: PASS

**Step 5: 提交**

```bash
git add packages/markitai/src/markitai/cli/ui.py packages/markitai/tests/unit/cli/test_ui.py
git commit -m "feat(cli): add unified UI components module"
```

---

## Task 2: 创建 i18n 模块

**Files:**
- Create: `packages/markitai/src/markitai/cli/i18n.py`
- Create: `packages/markitai/tests/unit/cli/test_i18n.py`

**Step 1: 编写失败测试**

```python
# packages/markitai/tests/unit/cli/test_i18n.py
"""Tests for i18n module."""

from __future__ import annotations

import os
from unittest import mock

import pytest

from markitai.cli import i18n


class TestLanguageDetection:
    """Test language detection."""

    def test_detect_zh_from_lang(self):
        """Test detecting Chinese from LANG env."""
        with mock.patch.dict(os.environ, {"LANG": "zh_CN.UTF-8"}, clear=False):
            # Reset cached language
            i18n._lang = None
            assert i18n.detect_language() == "zh"

    def test_detect_en_from_lang(self):
        """Test detecting English from LANG env."""
        with mock.patch.dict(os.environ, {"LANG": "en_US.UTF-8"}, clear=False):
            i18n._lang = None
            assert i18n.detect_language() == "en"

    def test_markitai_lang_override(self):
        """Test MARKITAI_LANG takes priority."""
        with mock.patch.dict(
            os.environ, {"MARKITAI_LANG": "zh", "LANG": "en_US.UTF-8"}, clear=False
        ):
            i18n._lang = None
            assert i18n.detect_language() == "zh"

    def test_default_to_en(self):
        """Test default to English when unknown."""
        with mock.patch.dict(os.environ, {"LANG": "fr_FR.UTF-8"}, clear=False):
            i18n._lang = None
            assert i18n.detect_language() == "en"


class TestTranslation:
    """Test translation function."""

    def test_t_returns_correct_language(self):
        """Test t() returns correct translation."""
        # Force English
        with mock.patch.dict(os.environ, {"MARKITAI_LANG": "en"}, clear=False):
            i18n._lang = None
            i18n._lang = i18n.detect_language()
            assert i18n.t("success") == "completed"

        # Force Chinese
        with mock.patch.dict(os.environ, {"MARKITAI_LANG": "zh"}, clear=False):
            i18n._lang = None
            i18n._lang = i18n.detect_language()
            assert i18n.t("success") == "完成"

    def test_t_with_format_args(self):
        """Test t() with format arguments."""
        with mock.patch.dict(os.environ, {"MARKITAI_LANG": "zh"}, clear=False):
            i18n._lang = None
            i18n._lang = i18n.detect_language()
            result = i18n.t("doctor.summary", passed=3, optional=1)
            assert "3" in result
            assert "1" in result

    def test_t_returns_key_for_unknown(self):
        """Test t() returns key for unknown translations."""
        result = i18n.t("unknown.key.here")
        assert result == "unknown.key.here"
```

**Step 2: 运行测试验证失败**

Run: `uv run pytest packages/markitai/tests/unit/cli/test_i18n.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'markitai.cli.i18n'"

**Step 3: 编写最小实现**

```python
# packages/markitai/src/markitai/cli/i18n.py
"""Internationalization (i18n) support for Markitai CLI.

This module provides language detection and translation functions
for CLI output messages.

Usage:
    from markitai.cli.i18n import t

    ui.title(t("doctor.title"))  # "System Check" or "系统检查"
"""

from __future__ import annotations

import os

# Text definitions: key -> {lang -> text}
TEXTS: dict[str, dict[str, str]] = {
    # Common
    "success": {"en": "completed", "zh": "完成"},
    "failed": {"en": "failed", "zh": "失败"},
    "warning": {"en": "warning", "zh": "警告"},
    "error": {"en": "error", "zh": "错误"},
    "enabled": {"en": "Enabled", "zh": "已启用"},
    "disabled": {"en": "Disabled", "zh": "已禁用"},
    "not_found": {"en": "not found", "zh": "未找到"},
    "installed": {"en": "installed", "zh": "已安装"},
    "missing": {"en": "missing", "zh": "缺失"},
    "total": {"en": "Total", "zh": "总计"},
    # Doctor command
    "doctor.title": {"en": "System Check", "zh": "系统检查"},
    "doctor.required": {"en": "Required Dependencies", "zh": "必需依赖"},
    "doctor.optional": {"en": "Optional Dependencies", "zh": "可选依赖"},
    "doctor.auth": {"en": "Authentication", "zh": "认证状态"},
    "doctor.summary": {
        "en": "Check complete ({passed} required passed, {optional} optional missing)",
        "zh": "检查完成（{passed} 必需通过，{optional} 可选缺失）",
    },
    "doctor.all_good": {
        "en": "All dependencies configured correctly",
        "zh": "所有依赖配置正确",
    },
    "doctor.fix_hint": {"en": "To fix missing dependencies:", "zh": "修复缺失依赖："},
    # Cache command
    "cache.title": {"en": "Cache Statistics", "zh": "缓存统计"},
    "cache.llm": {"en": "LLM responses", "zh": "LLM 响应"},
    "cache.spa": {"en": "SPA domains", "zh": "SPA 域名"},
    "cache.proxy": {"en": "Proxy detection", "zh": "代理检测"},
    "cache.entries": {"en": "entries", "zh": "条"},
    "cache.cleared": {"en": "Cleared {count} cache entries", "zh": "已清理 {count} 条缓存"},
    "cache.no_entries": {"en": "No cache entries to clear", "zh": "无缓存可清理"},
    # Config command
    "config.title": {"en": "Configuration Sources", "zh": "配置来源"},
    "config.cli_args": {"en": "CLI arguments", "zh": "命令行参数"},
    "config.env_vars": {"en": "Environment variables", "zh": "环境变量"},
    "config.local_file": {"en": "Local config file", "zh": "本地配置文件"},
    "config.user_file": {"en": "User config file", "zh": "用户配置文件"},
    "config.defaults": {"en": "Default values", "zh": "默认值"},
    "config.highest": {"en": "highest priority", "zh": "最高优先级"},
    "config.lowest": {"en": "lowest priority", "zh": "最低优先级"},
    "config.loaded": {"en": "loaded", "zh": "已加载"},
    "config.created": {"en": "Configuration file created", "zh": "配置文件已创建"},
    "config.valid": {"en": "Configuration is valid", "zh": "配置有效"},
    # Convert
    "convert.title": {"en": "Converting {file}", "zh": "转换 {file}"},
    "convert.parsing": {"en": "Parsing document...", "zh": "解析文档..."},
    "convert.extracting": {"en": "Extracting images ({n})...", "zh": "提取图片（{n} 张）..."},
    "convert.generating": {"en": "Generating Markdown...", "zh": "生成 Markdown..."},
    "convert.llm": {"en": "LLM enhancing...", "zh": "LLM 增强中..."},
    "convert.done": {"en": "Done", "zh": "完成"},
    "convert.output": {"en": "Output", "zh": "输出"},
    "convert.usage": {"en": "Usage", "zh": "用量"},
    # Batch
    "batch.title": {"en": "Batch Converting {dir}", "zh": "批量转换 {dir}"},
    "batch.progress": {"en": "Progress: {done}/{total} completed", "zh": "进度：{done}/{total} 完成"},
    "batch.summary": {
        "en": "Completed: {success}/{total} succeeded, {failed} failed",
        "zh": "完成：{success}/{total} 成功，{failed} 失败",
    },
    "batch.details": {"en": "Details", "zh": "详情"},
    "batch.waiting": {"en": "waiting", "zh": "等待中"},
}

# Cached language
_lang: str | None = None


def detect_language() -> str:
    """Detect user language preference.

    Priority: MARKITAI_LANG > LANG/LC_ALL > default (en)

    Returns:
        Language code: "en" or "zh"
    """
    lang = os.environ.get("MARKITAI_LANG", "")
    if not lang:
        lang = os.environ.get("LANG", "") or os.environ.get("LC_ALL", "")

    if lang.lower().startswith("zh"):
        return "zh"
    return "en"


def get_language() -> str:
    """Get current language (cached).

    Returns:
        Current language code
    """
    global _lang
    if _lang is None:
        _lang = detect_language()
    return _lang


def set_language(lang: str) -> None:
    """Override language setting.

    Args:
        lang: Language code ("en" or "zh")
    """
    global _lang
    _lang = lang


def t(key: str, **kwargs: str | int) -> str:
    """Get translated text.

    Args:
        key: Translation key (e.g., "doctor.title")
        **kwargs: Format arguments

    Returns:
        Translated text, or key if not found
    """
    lang = get_language()
    texts = TEXTS.get(key, {})
    text = texts.get(lang, texts.get("en", key))
    return text.format(**kwargs) if kwargs else text
```

**Step 4: 运行测试验证通过**

Run: `uv run pytest packages/markitai/tests/unit/cli/test_i18n.py -v`
Expected: PASS

**Step 5: 提交**

```bash
git add packages/markitai/src/markitai/cli/i18n.py packages/markitai/tests/unit/cli/test_i18n.py
git commit -m "feat(cli): add i18n module for multi-language support"
```

---

## Task 3: 改造 doctor 命令

**Files:**
- Modify: `packages/markitai/src/markitai/cli/commands/doctor.py`
- Modify: `packages/markitai/tests/unit/cli/test_doctor.py`

**Step 1: 编写测试更新**

更新现有测试以验证新 UI 格式：

```python
# 在 test_doctor.py 中添加新测试
def test_doctor_unified_ui_output(cli_runner, mocker):
    """Test doctor command uses unified UI components."""
    # Mock all dependencies as OK
    mocker.patch("shutil.which", return_value="/usr/bin/soffice")
    mocker.patch(
        "markitai.cli.commands.doctor.is_playwright_available", return_value=True
    )
    mocker.patch(
        "markitai.cli.commands.doctor.is_playwright_browser_installed",
        return_value=True,
    )

    result = cli_runner.invoke(doctor)

    # Should use unified symbols
    assert "◆" in result.output  # Title marker
    assert "✓" in result.output  # Success marker
    # Should not use Rich table format
    assert "Dependency Status" not in result.output
```

**Step 2: 运行测试验证失败**

Run: `uv run pytest packages/markitai/tests/unit/cli/test_doctor.py::test_doctor_unified_ui_output -v`
Expected: FAIL (output still uses Rich table)

**Step 3: 修改 doctor.py 实现**

修改 `_doctor_impl` 函数中的输出部分（约第566-616行），替换 Rich Table 为统一 UI：

```python
# 在 doctor.py 中的 _doctor_impl 函数末尾，替换 Rich table 输出部分

    # Output results
    if as_json:
        click.echo(json.dumps(results, indent=2))
        return

    # Unified UI output (replaces Rich table)
    from markitai.cli import ui
    from markitai.cli.i18n import t

    ui.title(t("doctor.title"))

    # Group by status: required (ok/error) vs optional (warning/missing)
    required_deps = ["playwright", "libreoffice", "rapidocr"]
    optional_deps = ["ffmpeg"]
    auth_deps = ["claude-agent-auth", "copilot-auth"]
    sdk_deps = ["claude-agent-sdk", "copilot-sdk"]

    # Required dependencies
    ui.section(t("doctor.required"))
    passed = 0
    for key in required_deps:
        if key not in results:
            continue
        info = results[key]
        if info["status"] == "ok":
            ui.success(f"{info['name']} {info['message']}")
            passed += 1
        elif info["status"] == "warning":
            ui.warning(info["name"], detail=info["message"])
        else:
            ui.error(info["name"], detail=info["message"])
    console.print()

    # Optional dependencies
    optional_missing = 0
    has_optional = any(k in results for k in optional_deps)
    if has_optional:
        ui.section(t("doctor.optional"))
        for key in optional_deps:
            if key not in results:
                continue
            info = results[key]
            if info["status"] == "ok":
                ui.success(f"{info['name']}")
            else:
                ui.warning(f"{info['name']} ({t('missing')})", detail=info.get("install_hint", ""))
                optional_missing += 1
        console.print()

    # LLM & SDK status
    llm_keys = ["llm-api", "vision-model"] + sdk_deps
    has_llm = any(k in results for k in llm_keys)
    if has_llm:
        ui.section("LLM")
        for key in llm_keys:
            if key not in results:
                continue
            info = results[key]
            if info["status"] == "ok":
                ui.success(f"{info['name']}: {info['message']}")
            elif info["status"] == "warning":
                ui.warning(f"{info['name']}: {info['message']}")
            else:
                ui.error(f"{info['name']}: {info['message']}")
        console.print()

    # Auth status
    has_auth = any(k in results for k in auth_deps)
    if has_auth:
        ui.section(t("doctor.auth"))
        for key in auth_deps:
            if key not in results:
                continue
            info = results[key]
            if info["status"] == "ok":
                ui.success(f"{info['name']}: {info['message']}")
            else:
                ui.error(f"{info['name']}: {info['message']}")
        console.print()

    # Summary
    if optional_missing == 0:
        ui.summary(t("doctor.all_good"))
    else:
        ui.summary(t("doctor.summary", passed=passed, optional=optional_missing))

    # Show install hints for missing/error items
    hints = [
        (info["name"], info["install_hint"])
        for info in results.values()
        if info["status"] in ("missing", "error") and info.get("install_hint")
    ]

    if hints:
        console.print()
        console.print(f"[yellow]{t('doctor.fix_hint')}[/yellow]")
        for name, hint in hints:
            console.print(f"  [dim]•[/dim] {name}: {hint}")
```

**Step 4: 运行测试验证通过**

Run: `uv run pytest packages/markitai/tests/unit/cli/test_doctor.py -v`
Expected: PASS

**Step 5: 运行完整测试**

Run: `uv run pytest packages/markitai/tests/unit -v -x`
Expected: PASS

**Step 6: 提交**

```bash
git add packages/markitai/src/markitai/cli/commands/doctor.py packages/markitai/tests/unit/cli/test_doctor.py
git commit -m "refactor(cli): update doctor command to use unified UI"
```

---

## Task 4: 改造 cache 命令

**Files:**
- Modify: `packages/markitai/src/markitai/cli/commands/cache.py`
- Modify: `packages/markitai/tests/unit/test_cache_cli.py`

**Step 1: 编写测试更新**

```python
# 添加到 test_cache_cli.py
def test_cache_stats_unified_ui(cli_runner, tmp_path, mocker):
    """Test cache stats uses unified UI."""
    # Mock config
    mock_cfg = mocker.MagicMock()
    mock_cfg.cache.enabled = True
    mock_cfg.cache.global_dir = str(tmp_path)
    mock_cfg.cache.max_size_bytes = 100 * 1024 * 1024
    mocker.patch(
        "markitai.cli.commands.cache.ConfigManager"
    ).return_value.load.return_value = mock_cfg

    result = cli_runner.invoke(cache, ["stats"])

    # Should use unified title
    assert "◆" in result.output
    assert "•" in result.output  # Info markers
```

**Step 2: 运行测试验证失败**

Run: `uv run pytest packages/markitai/tests/unit/test_cache_cli.py::test_cache_stats_unified_ui -v`
Expected: FAIL

**Step 3: 修改 cache.py 实现**

更新 `cache_stats` 函数的非 JSON 输出部分：

```python
# 在 cache_stats 函数中替换非 JSON 输出部分

    if as_json:
        console.print(
            json.dumps(stats_data, indent=2, ensure_ascii=False), soft_wrap=True
        )
    else:
        from markitai.cli import ui
        from markitai.cli.i18n import t

        ui.title(t("cache.title"))

        console.print(f"  {t('enabled')}: {cfg.cache.enabled}")
        console.print()

        if stats_data["cache"]:
            c = stats_data["cache"]
            if "error" in c:
                ui.error(f"Cache: {c['error']}")
            else:
                ui.info(f"{t('cache.llm')}: {c['count']} {t('cache.entries')} ({c['size_mb']} MB)")
        else:
            ui.info(f"{t('cache.llm')}: 0 {t('cache.entries')}")
```

类似地更新 `cache_clear` 和 `cache_spa_domains` 函数。

**Step 4: 运行测试验证通过**

Run: `uv run pytest packages/markitai/tests/unit/test_cache_cli.py -v`
Expected: PASS

**Step 5: 提交**

```bash
git add packages/markitai/src/markitai/cli/commands/cache.py packages/markitai/tests/unit/test_cache_cli.py
git commit -m "refactor(cli): update cache commands to use unified UI"
```

---

## Task 5: 改造 config 命令

**Files:**
- Modify: `packages/markitai/src/markitai/cli/commands/config.py`
- Modify: `packages/markitai/tests/unit/test_config_cli.py`

**Step 1: 编写测试更新**

```python
# 添加到 test_config_cli.py
def test_config_path_unified_ui(cli_runner, mocker):
    """Test config path uses unified UI."""
    mocker.patch(
        "markitai.cli.commands.config.ConfigManager"
    ).return_value.config_path = None

    result = cli_runner.invoke(config, ["path"])

    assert "◆" in result.output  # Title marker
```

**Step 2: 运行测试验证失败**

Run: `uv run pytest packages/markitai/tests/unit/test_config_cli.py::test_config_path_unified_ui -v`
Expected: FAIL

**Step 3: 修改 config.py 实现**

更新 `config_path_cmd` 函数：

```python
@config.command("path")
def config_path_cmd() -> None:
    """Show configuration file paths."""
    from markitai.cli import ui
    from markitai.cli.i18n import t

    manager = ConfigManager()
    manager.load()

    ui.title(t("config.title"))

    console.print(f"  1. {t('config.cli_args')}      [dim]│ {t('config.highest')}[/]")
    console.print(f"  2. {t('config.env_vars')}    [dim]│[/]")
    console.print(f"  3. ./markitai.json  [dim]│[/]", end="")
    if manager.config_path and "markitai.json" in str(manager.config_path):
        console.print(f" [green]✓ {t('config.loaded')}[/]")
    else:
        console.print()
    console.print(f"  4. {manager.DEFAULT_USER_CONFIG_DIR / 'config.json'}")
    console.print(f"  5. {t('config.defaults')}        [dim]│ {t('config.lowest')}[/]")
    console.print()

    if manager.config_path:
        ui.success(f"Currently using: {manager.config_path}")
    else:
        ui.warning("Using default configuration (no config file found)")
```

**Step 4: 运行测试验证通过**

Run: `uv run pytest packages/markitai/tests/unit/test_config_cli.py -v`
Expected: PASS

**Step 5: 提交**

```bash
git add packages/markitai/src/markitai/cli/commands/config.py packages/markitai/tests/unit/test_config_cli.py
git commit -m "refactor(cli): update config commands to use unified UI"
```

---

## Task 6: 更新 CLI 导出

**Files:**
- Modify: `packages/markitai/src/markitai/cli/__init__.py`

**Step 1: 更新导出**

```python
# 在 cli/__init__.py 中添加导出
from markitai.cli import ui
from markitai.cli import i18n

__all__ = [..., "ui", "i18n"]
```

**Step 2: 运行完整测试**

Run: `uv run pytest packages/markitai/tests/unit -v`
Expected: PASS

**Step 3: 运行代码质量检查**

Run: `uv run ruff check --fix && uv run ruff format && uv run pyright`
Expected: No errors

**Step 4: 提交**

```bash
git add packages/markitai/src/markitai/cli/__init__.py
git commit -m "chore(cli): export ui and i18n modules"
```

---

## Task 7: 端到端验证

**Step 1: 手动测试各命令**

```bash
# 测试 doctor 命令
uv run markitai doctor

# 测试 cache 命令
uv run markitai cache stats

# 测试 config 命令
uv run markitai config path

# 测试 JSON 输出仍然正常
uv run markitai doctor --json
```

**Step 2: 测试多语言**

```bash
# 测试中文
MARKITAI_LANG=zh uv run markitai doctor

# 测试英文
MARKITAI_LANG=en uv run markitai doctor
```

**Step 3: 运行完整测试套件**

Run: `uv run pytest -v`
Expected: PASS

**Step 4: 最终提交**

如果有任何修复：
```bash
git add -A
git commit -m "fix(cli): address integration issues in unified UI"
```

---

## 完成标准

- [ ] ui.py 模块创建完成，所有测试通过
- [ ] i18n.py 模块创建完成，所有测试通过
- [ ] doctor 命令改造完成，使用统一 UI
- [ ] cache 命令改造完成，使用统一 UI
- [ ] config 命令改造完成，使用统一 UI
- [ ] 所有测试通过
- [ ] Ruff lint 通过
- [ ] Pyright 类型检查通过
- [ ] 手动验证各命令输出符合设计
