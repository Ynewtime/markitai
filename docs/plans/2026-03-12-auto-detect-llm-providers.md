# Auto-Detect LLM Providers (Zero-Config) Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 当无配置文件时，自动检测已鉴权的 local provider 和环境变量中的 API Key，填充 `model_list`，使 `markitai file.pdf --llm` 零配置可用。

**Architecture:** 将 `interactive.py` 中的 provider 检测逻辑提取为共享模块 `cli/providers_detect.py`，在 `main.py` 的 model_list 填充逻辑中复用。检测优先级：`MODEL` env var > config file > auto-detect（env keys + local providers）。Interactive 模式也改用共享模块。

**Tech Stack:** Python, Pydantic, Click CLI

---

## 文件结构

| 文件 | 操作 | 职责 |
|------|------|------|
| `src/markitai/cli/providers_detect.py` | **新建** | 共享 provider 检测逻辑（从 `interactive.py` 提取） |
| `src/markitai/cli/interactive.py` | 修改 | 移除检测逻辑，改为导入 `providers_detect` |
| `src/markitai/cli/main.py` | 修改 | 替换 `MODEL` env var 逻辑为完整的 auto-detect |
| `tests/unit/cli/test_providers_detect.py` | **新建** | 共享检测逻辑的单元测试 |
| `tests/unit/cli/test_interactive.py` | 修改 | 更新导入路径 |
| `tests/unit/test_model_env_var.py` | 修改 | 适配新的 auto-detect 行为 |

## 设计约束

1. **优先级链**（从高到低）：
   - `MODEL` 环境变量（显式指定，最高优先级）
   - 配置文件 `markitai.json` 中 `weight > 0` 的模型
   - Auto-detect：已鉴权的 local providers（claude-agent, copilot, chatgpt, gemini-cli）
   - Auto-detect：环境变量中的 API Key（`ANTHROPIC_API_KEY`, `GEMINI_API_KEY` 等）
2. **Auto-detect 仅在 `model_list` 为空时触发**——有配置文件时不覆盖用户配置
3. **Local provider 鉴权检查有 I/O 开销**——仅在 `--llm` 启用且无 config 时才执行
4. **向后兼容**：`MODEL` env var 仍然有效且优先级最高

---

## Chunk 1: 提取共享检测模块

### Task 1: 创建 `providers_detect.py` 并迁移检测逻辑

**Files:**
- Create: `src/markitai/cli/providers_detect.py`
- Create: `tests/unit/cli/test_providers_detect.py`

- [ ] **Step 1: 写 `detect_all_providers()` 的测试**

在 `tests/unit/cli/test_providers_detect.py` 中创建测试文件：

```python
"""Tests for shared LLM provider auto-detection."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from markitai.cli.providers_detect import (
    ProviderDetectionResult,
    detect_all_providers,
    get_active_models_from_config,
)


class TestDetectAllProviders:
    """Tests for detect_all_providers function."""

    def test_detects_anthropic_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should detect ANTHROPIC_API_KEY from environment."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        # Disable CLI providers and other env vars
        with patch("markitai.cli.providers_detect.shutil.which", return_value=None):
            results = detect_all_providers()
        anthropic = [r for r in results if r.provider == "anthropic"]
        assert len(anthropic) == 1
        assert anthropic[0].model == "anthropic/claude-sonnet-4-5-20250929"

    def test_detects_gemini_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should detect GEMINI_API_KEY from environment."""
        monkeypatch.setenv("GEMINI_API_KEY", "AIza-test")
        with patch("markitai.cli.providers_detect.shutil.which", return_value=None):
            results = detect_all_providers()
        gemini = [r for r in results if r.provider == "gemini"]
        assert len(gemini) == 1

    def test_detects_claude_cli_when_authenticated(self) -> None:
        """Should detect claude-agent when CLI is installed and authenticated."""
        with (
            patch(
                "markitai.cli.providers_detect.shutil.which",
                side_effect=lambda cmd: "/usr/bin/claude" if cmd == "claude" else None,
            ),
            patch(
                "markitai.cli.providers_detect._check_claude_auth",
                return_value=True,
            ),
        ):
            results = detect_all_providers()
        claude = [r for r in results if r.provider == "claude-agent"]
        assert len(claude) == 1

    def test_returns_empty_when_nothing_available(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should return empty list when no providers detected."""
        # Clear all relevant env vars
        for var in [
            "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY",
            "DEEPSEEK_API_KEY", "OPENROUTER_API_KEY",
        ]:
            monkeypatch.delenv(var, raising=False)
        with (
            patch("markitai.cli.providers_detect.shutil.which", return_value=None),
            patch("markitai.cli.providers_detect._check_chatgpt_auth", return_value=False),
            patch("markitai.cli.providers_detect._check_gemini_cli_auth", return_value=False),
        ):
            results = detect_all_providers()
        assert results == []


class TestGetActiveModelsFromConfig:
    """Tests for get_active_models_from_config (moved from interactive)."""

    def test_returns_models_with_positive_weight(self) -> None:
        model_list = [
            {"model_name": "default", "litellm_params": {"model": "gemini/flash", "weight": 10}},
            {"model_name": "default", "litellm_params": {"model": "claude/sonnet", "weight": 0}},
        ]
        result = get_active_models_from_config(model_list)
        assert result == ["gemini/flash"]

    def test_returns_empty_when_all_disabled(self) -> None:
        model_list = [
            {"model_name": "default", "litellm_params": {"model": "a/b", "weight": 0}},
        ]
        assert get_active_models_from_config(model_list) == []

    def test_default_weight_is_enabled(self) -> None:
        """Models without explicit weight should be included."""
        model_list = [
            {"model_name": "default", "litellm_params": {"model": "anthropic/claude"}},
        ]
        assert get_active_models_from_config(model_list) == ["anthropic/claude"]
```

- [ ] **Step 2: 运行测试，确认 ImportError 失败**

Run: `uv run pytest tests/unit/cli/test_providers_detect.py -x`
Expected: `ImportError: cannot import name 'detect_all_providers' from 'markitai.cli.providers_detect'`

- [ ] **Step 3: 创建 `providers_detect.py`，从 `interactive.py` 提取检测逻辑**

创建 `src/markitai/cli/providers_detect.py`，将以下内容从 `interactive.py` 迁移过来：
- `ProviderDetectionResult` dataclass
- `_check_claude_auth()`, `_check_copilot_auth()`, `_check_chatgpt_auth()`, `_check_gemini_cli_auth()` 函数
- `detect_all_llm_providers()` → 重命名为 `detect_all_providers()`
- `detect_llm_provider()` → 重命名为 `detect_first_provider()`
- `get_active_models_from_config()` 函数
- `_format_model_list()` 函数

```python
"""Shared LLM provider auto-detection.

Used by both interactive mode and CLI main to discover available
LLM providers from authenticated CLI tools and environment variables.
"""

from __future__ import annotations

import asyncio
import os
import shutil
from dataclasses import dataclass
from typing import Any


@dataclass
class ProviderDetectionResult:
    """Result of LLM provider auto-detection."""

    provider: str
    model: str
    authenticated: bool
    source: str  # "cli", "env"


def _check_claude_auth() -> bool:
    """Check if Claude CLI is authenticated."""
    from markitai.providers.auth import AuthManager

    auth_manager = AuthManager()
    try:
        status = asyncio.run(auth_manager.check_auth("claude-agent"))
        return status.authenticated
    except Exception:
        return False


def _check_copilot_auth() -> bool:
    """Check if Copilot CLI is authenticated."""
    from markitai.providers.auth import AuthManager

    auth_manager = AuthManager()
    try:
        status = asyncio.run(auth_manager.check_auth("copilot"))
        return status.authenticated
    except Exception:
        return False


def _check_chatgpt_auth() -> bool:
    """Check if ChatGPT provider is authenticated."""
    from markitai.providers.auth import _check_chatgpt_auth as check_fn

    try:
        status = check_fn()
        return status.authenticated
    except Exception:
        return False


def _check_gemini_cli_auth() -> bool:
    """Check if Gemini CLI is authenticated."""
    from markitai.providers.auth import _check_gemini_cli_auth as check_fn

    try:
        status = check_fn()
        return status.authenticated
    except Exception:
        return False


def detect_all_providers() -> list[ProviderDetectionResult]:
    """Auto-detect all available LLM providers.

    Checks each provider independently and returns all that are available,
    ordered by priority:
    1. Claude CLI (if installed and authenticated)
    2. Copilot CLI (if installed and authenticated)
    3. ChatGPT (if authenticated via OAuth)
    4. Gemini CLI (if authenticated via OAuth)
    5-9. Environment variables (ANTHROPIC, OPENAI, GEMINI, DEEPSEEK, OPENROUTER)

    Returns:
        List of all detected providers (may be empty).
    """
    results: list[ProviderDetectionResult] = []

    # 1. Check Claude CLI
    if shutil.which("claude"):
        if _check_claude_auth():
            results.append(
                ProviderDetectionResult(
                    provider="claude-agent",
                    model="claude-agent/sonnet",
                    authenticated=True,
                    source="cli",
                )
            )

    # 2. Check Copilot CLI
    if shutil.which("copilot"):
        if _check_copilot_auth():
            results.append(
                ProviderDetectionResult(
                    provider="copilot",
                    model="copilot/claude-sonnet-4.5",
                    authenticated=True,
                    source="cli",
                )
            )

    # 3. Check ChatGPT (OAuth)
    if _check_chatgpt_auth():
        results.append(
            ProviderDetectionResult(
                provider="chatgpt",
                model="chatgpt/gpt-5.2",
                authenticated=True,
                source="cli",
            )
        )

    # 4. Check Gemini CLI (OAuth)
    if _check_gemini_cli_auth():
        results.append(
            ProviderDetectionResult(
                provider="gemini-cli",
                model="gemini-cli/gemini-2.5-pro",
                authenticated=True,
                source="cli",
            )
        )

    # 5-9. Check environment variables
    env_providers = [
        ("ANTHROPIC_API_KEY", "anthropic", "anthropic/claude-sonnet-4-5-20250929"),
        ("OPENAI_API_KEY", "openai", "openai/gpt-5.2"),
        ("GEMINI_API_KEY", "gemini", "gemini/gemini-2.5-flash"),
        ("DEEPSEEK_API_KEY", "deepseek", "deepseek/deepseek-chat"),
        ("OPENROUTER_API_KEY", "openrouter", "openrouter/google/gemini-2.5-flash"),
    ]
    for env_var, provider, model in env_providers:
        if os.environ.get(env_var):
            results.append(
                ProviderDetectionResult(
                    provider=provider,
                    model=model,
                    authenticated=True,
                    source="env",
                )
            )

    return results


def detect_first_provider() -> ProviderDetectionResult | None:
    """Auto-detect the highest-priority available LLM provider.

    Returns:
        Highest-priority detected provider, or None.
    """
    results = detect_all_providers()
    return results[0] if results else None


def get_active_models_from_config(
    model_list: list[dict[str, Any]],
) -> list[str]:
    """Extract active model names (weight > 0) from config model_list.

    Args:
        model_list: Raw model_list dicts from config (each has litellm_params).

    Returns:
        List of model identifiers with positive weight.
    """
    active: list[str] = []
    for entry in model_list:
        params = entry.get("litellm_params", {})
        model = params.get("model", "")
        weight = params.get("weight", 1)  # default weight is 1 (enabled)
        if model and weight > 0:
            active.append(model)
    return active


def format_model_list(models: list[str], max_show: int = 3) -> str:
    """Format a list of model names for display.

    Shows up to *max_show* names, with a "+N more" suffix if there are extras.
    """
    shown = ", ".join(models[:max_show])
    extra = len(models) - max_show
    if extra > 0:
        shown += f" (+{extra} more)"
    return shown
```

- [ ] **Step 4: 运行测试，确认通过**

Run: `uv run pytest tests/unit/cli/test_providers_detect.py -x -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/markitai/cli/providers_detect.py tests/unit/cli/test_providers_detect.py
git commit -m "refactor: extract shared provider detection to providers_detect.py"
```

### Task 2: 更新 `interactive.py` 使用共享模块

**Files:**
- Modify: `src/markitai/cli/interactive.py`
- Modify: `tests/unit/cli/test_interactive.py`

- [ ] **Step 1: 更新 `interactive.py` 的导入**

在 `interactive.py` 中：
1. 删除 `ProviderDetectionResult` dataclass 定义
2. 删除 `_check_claude_auth()`, `_check_copilot_auth()`, `_check_chatgpt_auth()`, `_check_gemini_cli_auth()` 函数
3. 删除 `detect_all_llm_providers()` 和 `detect_llm_provider()` 函数
4. 删除 `get_active_models_from_config()` 和 `_format_model_list()` 函数
5. 改为从 `providers_detect` 导入：

```python
from markitai.cli.providers_detect import (
    ProviderDetectionResult,
    detect_all_providers,
    detect_first_provider,
    format_model_list,
    get_active_models_from_config,
)
```

6. 更新 `prompt_enable_llm()` 中的函数调用：
   - `detect_all_llm_providers()` → `detect_all_providers()`
   - `_format_model_list()` → `format_model_list()`
7. 更新 `_print_summary()` 中的 `_format_model_list()` → `format_model_list()`

为保持向后兼容，在 `interactive.py` 文件末尾保留 re-export：
```python
# Re-export for backward compatibility with existing imports
detect_all_llm_providers = detect_all_providers
detect_llm_provider = detect_first_provider
```

- [ ] **Step 2: 更新 `test_interactive.py` 的 mock 路径**

测试中所有 `patch("markitai.cli.interactive.xxx")` 路径需要更新为 `patch("markitai.cli.providers_detect.xxx")`，因为检测函数已经迁移。涉及的 mock：
- `_check_claude_auth` → `markitai.cli.providers_detect._check_claude_auth`
- `_check_copilot_auth` → `markitai.cli.providers_detect._check_copilot_auth`
- `_check_chatgpt_auth` → `markitai.cli.providers_detect._check_chatgpt_auth`
- `_check_gemini_cli_auth` → `markitai.cli.providers_detect._check_gemini_cli_auth`
- `shutil.which` → `markitai.cli.providers_detect.shutil.which`

同时：
- `TestGetActiveModelsFromConfig` 类的导入改为从 `providers_detect` 导入（或保留从 `interactive` 导入，因为有 re-export）
- 删除 `test_interactive.py` 中已迁移到 `test_providers_detect.py` 的重复测试

- [ ] **Step 3: 运行测试，确认 interactive 测试全部通过**

Run: `uv run pytest tests/unit/cli/test_interactive.py tests/unit/cli/test_providers_detect.py -x -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add src/markitai/cli/interactive.py tests/unit/cli/test_interactive.py
git commit -m "refactor: interactive.py uses shared providers_detect module"
```

---

## Chunk 2: main.py auto-detect 集成

### Task 3: 新增 `auto_detect_model_list()` 函数并集成到 main.py

**Files:**
- Modify: `src/markitai/cli/providers_detect.py`
- Modify: `src/markitai/cli/main.py`
- Modify: `tests/unit/cli/test_providers_detect.py`
- Modify: `tests/unit/test_model_env_var.py`

- [ ] **Step 1: 写 `providers_to_model_configs()` 的测试**

在 `tests/unit/cli/test_providers_detect.py` 中新增：

```python
from markitai.config import ModelConfig

from markitai.cli.providers_detect import providers_to_model_configs


class TestProvidersToModelConfigs:
    """Tests for converting detected providers to ModelConfig list."""

    def test_converts_single_provider(self) -> None:
        """Should create ModelConfig from detected provider."""
        providers = [
            ProviderDetectionResult(
                provider="anthropic",
                model="anthropic/claude-sonnet-4-5-20250929",
                authenticated=True,
                source="env",
            )
        ]
        configs = providers_to_model_configs(providers)
        assert len(configs) == 1
        assert configs[0].litellm_params.model == "anthropic/claude-sonnet-4-5-20250929"
        assert configs[0].model_name == "default"

    def test_converts_multiple_providers(self) -> None:
        """Should create ModelConfig for each detected provider."""
        providers = [
            ProviderDetectionResult("claude-agent", "claude-agent/sonnet", True, "cli"),
            ProviderDetectionResult("gemini", "gemini/gemini-2.5-flash", True, "env"),
        ]
        configs = providers_to_model_configs(providers)
        assert len(configs) == 2
        models = [c.litellm_params.model for c in configs]
        assert "claude-agent/sonnet" in models
        assert "gemini/gemini-2.5-flash" in models

    def test_empty_providers_returns_empty(self) -> None:
        assert providers_to_model_configs([]) == []
```

- [ ] **Step 2: 运行测试，确认 ImportError 失败**

Run: `uv run pytest tests/unit/cli/test_providers_detect.py::TestProvidersToModelConfigs -x`
Expected: `ImportError`

- [ ] **Step 3: 实现 `providers_to_model_configs()`**

在 `providers_detect.py` 中新增：

```python
def providers_to_model_configs(
    providers: list[ProviderDetectionResult],
) -> list[ModelConfig]:
    """Convert detected providers to ModelConfig list for LLM router.

    Args:
        providers: Detected provider results from detect_all_providers().

    Returns:
        List of ModelConfig instances ready for cfg.llm.model_list.
    """
    from markitai.config import LiteLLMParams, ModelConfig

    return [
        ModelConfig(
            model_name="default",
            litellm_params=LiteLLMParams(model=p.model),
        )
        for p in providers
    ]
```

- [ ] **Step 4: 运行测试，确认通过**

Run: `uv run pytest tests/unit/cli/test_providers_detect.py::TestProvidersToModelConfigs -x -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/markitai/cli/providers_detect.py tests/unit/cli/test_providers_detect.py
git commit -m "feat: add providers_to_model_configs conversion function"
```

- [ ] **Step 6: 修改 `main.py` 的 model_list 填充逻辑**

替换 `main.py` 中 393-410 行的 `MODEL` env var 逻辑：

```python
    # Auto-populate model_list when LLM enabled but no models configured
    if cfg.llm.enabled and not cfg.llm.model_list:
        # Priority 1: MODEL env var (explicit single-model override)
        model_env = os.environ.get("MODEL")
        if model_env:
            from markitai.config import LiteLLMParams, ModelConfig

            cfg.llm.model_list = [
                ModelConfig(
                    model_name="default",
                    litellm_params=LiteLLMParams(model=model_env),
                )
            ]
            logger.info(f"[Config] Using MODEL env var: {model_env}")
        else:
            # Priority 2: Auto-detect from env keys and authenticated CLI providers
            from markitai.cli.providers_detect import (
                detect_all_providers,
                providers_to_model_configs,
            )

            detected = detect_all_providers()
            if detected:
                cfg.llm.model_list = providers_to_model_configs(detected)
                names = [d.model for d in detected]
                logger.info(
                    f"[Config] Auto-detected {len(detected)} provider(s): "
                    + ", ".join(names)
                )
            else:
                logger.warning(
                    "[Config] LLM enabled but no models configured. "
                    "Set MODEL env var or add models to llm.model_list in config file."
                )
```

- [ ] **Step 7: 更新 `test_model_env_var.py` 中受影响的测试**

`test_warning_when_no_model_env_and_no_model_list` 测试需要更新：原来 `MODEL` env var 不设就会 warning，但现在会先尝试 auto-detect。需要同时 mock 掉 auto-detect 使其返回空列表，才能触发 warning。

在该测试中新增 mock：
```python
with patch("markitai.cli.providers_detect.detect_all_providers", return_value=[]):
    ...
```

同时新增一个测试验证 auto-detect 生效：
```python
def test_auto_detect_populates_model_list_when_no_config(
    self, tmp_path: Path, cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Auto-detected providers should fill model_list when no config exists."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("content")
    output_dir = tmp_path / "out"

    monkeypatch.delenv("MODEL", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test-key")

    result = cli_runner.invoke(
        app, [str(test_file), "-o", str(output_dir), "--llm", "--dry-run"]
    )
    assert result.exit_code == 0
    assert "no models configured" not in result.output.lower()
```

- [ ] **Step 8: 运行完整测试套件**

Run: `uv run pytest tests/unit/cli/test_providers_detect.py tests/unit/cli/test_interactive.py tests/unit/test_model_env_var.py -x -v`
Expected: All PASS

- [ ] **Step 9: 运行全量测试**

Run: `uv run pytest tests/ -x -q`
Expected: All PASS

- [ ] **Step 10: Commit**

```bash
git add src/markitai/cli/main.py tests/unit/test_model_env_var.py
git commit -m "feat: auto-detect LLM providers when no config file exists"
```

---

## Chunk 3: Interactive 模式适配

### Task 4: Interactive 模式无 config 时也使用 auto-detect 的 model 名称

**Files:**
- Modify: `src/markitai/cli/interactive.py`

当前的 `prompt_enable_llm()` 在无 config 时走 `detect_all_providers()` 后，summary 显示的是 `provider` 名称（如 `claude-agent`）。但实际执行时 `main.py` 也会 auto-detect 并使用 `model` 名称（如 `claude-agent/sonnet`）。

- [ ] **Step 1: 更新无 config 分支的 active_models**

在 `prompt_enable_llm()` 的 fallback 分支中，除了设置 `provider_result`，也填充 `session.active_models`：

```python
    # 2. Fallback: auto-detect providers if no config models
    if not has_provider:
        all_providers = detect_all_providers()
        session.provider_result = all_providers[0] if all_providers else None
        if all_providers:
            has_provider = True
            session.active_models = [p.model for p in all_providers]
            console.print(
                f"[green]\u2713[/green] Detected: {format_model_list(session.active_models)}"
            )
```

这样 summary 中就会显示 `Models: claude-agent/sonnet, gemini/gemini-2.5-flash (+3 more)` 而不是 `Provider: claude-agent`。

- [ ] **Step 2: 运行测试确认通过**

Run: `uv run pytest tests/unit/cli/test_interactive.py -x -v`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
git add src/markitai/cli/interactive.py
git commit -m "fix: interactive summary shows actual model names instead of provider"
```

- [ ] **Step 4: 运行全量测试**

Run: `uv run pytest tests/ -x -q`
Expected: All PASS
