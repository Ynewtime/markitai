# CLI UX 全面改造实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 实现 CLI UX 全面改造设计，统一输出风格，优化日志系统，新增 init 命令

**Architecture:** 分阶段改造，先基础设施（日志），再核心输出（批量/单文件），最后新功能（init）

**Tech Stack:** Rich, Click, Loguru, Python 3.11+

**Design Document:** `docs/plans/2026-02-05-cli-ux-full-redesign.md`

---

## Phase 1: 日志系统改造

### Task 1.1: 修改日志文件格式

**Files:**
- Modify: `packages/markitai/src/markitai/cli/logging_config.py`
- Modify: `packages/markitai/src/markitai/config.py`

**Step 1: 更新配置模型添加 log.format**

在 `config.py` 的 `LogConfig` 类中添加 format 字段：

```python
class LogConfig(BaseModel):
    """Logging configuration."""

    level: str = "DEBUG"
    dir: str | None = "./logs"
    rotation: str = "10 MB"
    retention: str = "7 days"
    format: str = "text"  # 新增: "text" | "json"
```

**Step 2: 修改 logging_config.py 支持 text 格式**

在 `setup_logging` 函数中，根据 format 配置选择日志格式：

```python
# 日志文件格式
if log_format == "json":
    # 紧凑 JSON 格式
    logger.add(
        log_file_path,
        level=log_level,
        rotation=rotation,
        retention=retention,
        format='{{"ts":"{time:YYYY-MM-DDTHH:mm:ss}","lvl":"{level.name}","src":"{module}:{line}","msg":"{message}"}}',
    )
else:
    # 人类可读格式（默认）
    logger.add(
        log_file_path,
        level=log_level,
        rotation=rotation,
        retention=retention,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <5} | {module}:{line: <3} | {message}",
    )
```

**Step 3: 运行测试验证**

Run: `uv run pytest packages/markitai/tests/unit/test_logging*.py -v`

**Step 4: 手动验证日志文件格式**

```bash
rm -rf logs/ && uv run markitai doctor && head -5 logs/*.log
```

Expected: 人类可读格式，非 JSON

**Step 5: 提交**

```bash
git add packages/markitai/src/markitai/cli/logging_config.py packages/markitai/src/markitai/config.py
git commit -m "refactor(log): change default log format to human-readable text"
```

---

### Task 1.2: 添加环境变量覆盖

**Files:**
- Modify: `packages/markitai/src/markitai/cli/logging_config.py`

**Step 1: 支持 MARKITAI_LOG_FORMAT 环境变量**

```python
def setup_logging(...):
    # 检查环境变量覆盖
    log_format = os.environ.get("MARKITAI_LOG_FORMAT", log_format)
```

**Step 2: 测试环境变量**

```bash
MARKITAI_LOG_FORMAT=json uv run markitai doctor && head -5 logs/*.log
```

**Step 3: 提交**

```bash
git add packages/markitai/src/markitai/cli/logging_config.py
git commit -m "feat(log): support MARKITAI_LOG_FORMAT env override"
```

---

## Phase 2: 批量处理输出改造

### Task 2.1: 改造批量摘要表格

**Files:**
- Modify: `packages/markitai/src/markitai/batch.py`

**Step 1: 找到 Batch Processing Summary 表格**

搜索 `Table()` 和 "Batch Processing Summary"，定位到输出代码。

**Step 2: 替换为紧凑单列格式**

使用 ui.py 的组件替换 Rich Table：

```python
from markitai.cli import ui
from markitai.cli.i18n import t

# 替换 Table 输出
ui.summary(f"Done: {files_done} files, {urls_done} URLs ({duration}, ${cost:.4f})")
console.print()
console.print(f"  {ui.MARK_INFO} Files: {files_done}/{files_total} {ui.MARK_SUCCESS if files_done == files_total else ''}  Cache: {file_cache_hits}")
console.print(f"  {ui.MARK_INFO} URLs:  {urls_done}/{urls_total} {ui.MARK_SUCCESS if urls_done == urls_total else ''}  Cache: {url_cache_hits}")
console.print()
console.print(f"  Output: {output_dir}/")
```

**Step 3: 运行测试**

Run: `uv run pytest packages/markitai/tests/unit/test_batch*.py -v`

**Step 4: 手动验证**

```bash
uv run markitai packages/markitai/tests/fixtures --preset rich --no-cache -o output-test --dry-run
```

**Step 5: 提交**

```bash
git add packages/markitai/src/markitai/batch.py
git commit -m "refactor(batch): replace summary table with compact unified UI"
```

---

### Task 2.2: 改造进度条样式

**Files:**
- Modify: `packages/markitai/src/markitai/batch.py`

**Step 1: 修改 Progress 显示当前处理文件**

在进度条中添加当前处理的文件名：

```python
progress = Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    TaskProgressColumn(),
    TextColumn("({task.fields[current]})"),  # 显示当前文件
    console=console,
)
```

**Step 2: 更新任务时设置 current 字段**

```python
progress.update(task_id, advance=1, current=filename)
```

**Step 3: 测试**

Run: `uv run markitai packages/markitai/tests/fixtures -o output-test --no-cache`

**Step 4: 提交**

```bash
git add packages/markitai/src/markitai/batch.py
git commit -m "refactor(batch): show current file in progress bar"
```

---

### Task 2.3: 改造完成后的结果显示

**Files:**
- Modify: `packages/markitai/src/markitai/batch.py`

**Step 1: 进度完成后显示统一格式结果**

```python
# 完成后显示
console.print()
ui.title("Converting")
console.print()
console.print(f"  {ui.MARK_SUCCESS} {files_done} files completed ({file_duration})")
console.print(f"  {ui.MARK_SUCCESS} {urls_done} URLs completed ({url_duration})")

if warnings:
    console.print()
    for w in warnings:
        console.print(f"  {ui.MARK_WARNING} {w}")
```

**Step 2: 测试并提交**

---

### Task 2.4: 移除 LogPanel Rich 面板

**Files:**
- Modify: `packages/markitai/src/markitai/batch.py`

**Step 1: 找到 LogPanel 和 Panel 使用**

**Step 2: 替换为统一 UI 格式的日志输出**

使用 `ui.step()` 或直接的 `│` 缩进格式。

**Step 3: 测试并提交**

---

## Phase 3: 单文件处理改造

### Task 3.1: 改造单文件输出逻辑

**Files:**
- Modify: `packages/markitai/src/markitai/cli/processors/file.py`

**Step 1: 实现分层输出逻辑**

```python
async def process_single_file(
    input_path: Path,
    output_dir: Path,
    cfg: MarkitaiConfig,
    dry_run: bool,
    log_file_path: Path | None = None,
    verbose: bool = False,
    quiet: bool = False,
) -> None:
    # quiet 模式：完全静默
    if quiet:
        # 只处理，不输出
        ...
        return

    # 有 output_dir：显示保存路径
    if output_dir:
        if verbose:
            ui.title(f"Converting {input_path.name}")
            # 显示详细步骤...

        # 处理完成后
        ui.success(str(output_file))
        return

    # 无 output_dir：输出内容到 stdout
    console.print(markdown_content, markup=False)
```

**Step 2: 运行测试**

Run: `uv run pytest packages/markitai/tests/unit/test_cli_main.py -v`

**Step 3: 手动验证各场景**

```bash
# 无 -o：输出内容
uv run markitai packages/markitai/tests/fixtures/concise.md

# 有 -o：显示路径
uv run markitai packages/markitai/tests/fixtures/concise.md -o output-test/

# 有 -o -v：详细
uv run markitai packages/markitai/tests/fixtures/concise.md -o output-test/ -v

# quiet：静默
uv run markitai packages/markitai/tests/fixtures/concise.md -o output-test/ -q
```

**Step 4: 提交**

---

### Task 3.2: 改造 URL 处理输出

**Files:**
- Modify: `packages/markitai/src/markitai/cli/processors/url.py`

**Step 1: 替换 Panel 为统一 UI**

找到所有 `Panel()` 使用，替换为 `ui.error()` / `ui.warning()` 格式。

**Step 2: 测试并提交**

---

## Phase 4: Dry-run 和 Panel 改造

### Task 4.1: 改造 Dry-run 预览

**Files:**
- Modify: `packages/markitai/src/markitai/cli/processors/batch.py`

**Step 1: 替换 Dry-run Panel 为统一 UI**

```python
ui.title("Dry Run")
console.print()
console.print(f"  Files ({len(files)})")
for f in files:
    console.print(f"    {ui.MARK_INFO} {f.name}")
console.print()
console.print(f"  URLs ({len(urls)})")
for u in urls:
    console.print(f"    {ui.MARK_INFO} {u}")
console.print()
console.print("  " + "─" * 20)
console.print(f"  Total: {len(files)} files, {len(urls)} URLs")
```

**Step 2: 测试**

```bash
uv run markitai packages/markitai/tests/fixtures --dry-run
```

**Step 3: 提交**

---

### Task 4.2: 改造验证器警告面板

**Files:**
- Modify: `packages/markitai/src/markitai/cli/processors/validators.py`

**Step 1: 替换所有 Panel 为统一 UI**

```python
# 旧
console.print(Panel("LLM is required...", title="Warning"))

# 新
ui.warning("LLM is required for this feature")
console.print(f"  {ui.MARK_LINE} Enable with: markitai --llm ...")
```

**Step 2: 测试并提交**

---

## Phase 5: 新用户引导

### Task 5.1: 创建跨平台提示工具

**Files:**
- Create: `packages/markitai/src/markitai/cli/hints.py`

**Step 1: 创建提示工具模块**

```python
"""Cross-platform hints for CLI commands."""

from __future__ import annotations

import sys


def get_env_set_command(var: str, value: str = "your_key") -> str:
    """Get the appropriate environment variable set command for current OS."""
    if sys.platform == "win32":
        # Check if running in PowerShell or CMD
        import os
        if os.environ.get("PSModulePath"):
            return f'$env:{var}="{value}"'
        return f"set {var}={value}"
    return f"export {var}={value}"


def get_llm_not_configured_hint() -> str:
    """Get hint message when LLM is not configured."""
    from markitai.cli import ui

    env_cmd = get_env_set_command("GEMINI_API_KEY")

    lines = [
        f"{ui.MARK_WARNING} LLM not configured",
        "",
        "  Quick setup (choose one):",
        "",
        "  • Claude CLI:  claude login",
        "  • Copilot CLI: copilot auth login",
        f"  • API Key:     {env_cmd}",
        "  • Wizard:      markitai init",
        "",
        "  Run 'markitai doctor' to check status.",
    ]
    return "\n".join(lines)
```

**Step 2: 测试跨平台检测**

**Step 3: 提交**

---

### Task 5.2: 创建 init 命令

**Files:**
- Create: `packages/markitai/src/markitai/cli/commands/init.py`
- Modify: `packages/markitai/src/markitai/cli/main.py`

**Step 1: 创建 init.py**

```python
"""Init command for Markitai CLI."""

from __future__ import annotations

from pathlib import Path

import click

from markitai.cli import ui
from markitai.cli.console import get_console
from markitai.cli.i18n import t
from markitai.config import ConfigManager

console = get_console()


@click.command("init")
@click.option(
    "--yes",
    "-y",
    is_flag=True,
    default=False,
    help="Quick mode, generate default config without prompts.",
)
@click.option(
    "--output",
    "-o",
    "output_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Output path for configuration file.",
)
def init(yes: bool, output_path: Path | None) -> None:
    """Initialize Markitai configuration."""
    if yes:
        # 快速模式
        _quick_init(output_path)
    else:
        # 交互式向导
        _wizard_init(output_path)


def _quick_init(output_path: Path | None) -> None:
    """Quick init - generate default config."""
    manager = ConfigManager()

    if output_path is None:
        output_path = Path.cwd() / "markitai.json"
    elif output_path.is_dir():
        output_path = output_path / "markitai.json"

    saved_path = manager.save(output_path, minimal=True)
    ui.success(f"Configuration created: {saved_path}")


def _wizard_init(output_path: Path | None) -> None:
    """Interactive wizard init."""
    ui.title("Markitai Setup")
    console.print()
    console.print("  Detecting available LLM providers...")
    console.print()

    # 检测可用的 provider
    # ... 实现检测逻辑

    # 显示选项
    # ... 实现交互逻辑
```

**Step 2: 在 main.py 注册命令**

```python
from markitai.cli.commands.init import init as init_cmd

app.add_command(init_cmd, name="init")
```

**Step 3: 测试**

```bash
uv run markitai init --help
uv run markitai init -y
```

**Step 4: 提交**

---

### Task 5.3: 移除 config init 命令

**Files:**
- Modify: `packages/markitai/src/markitai/cli/commands/config.py`

**Step 1: 删除 config_init 函数和装饰器**

**Step 2: 更新测试**

**Step 3: 提交**

---

## Phase 6: 交互式模式改造

### Task 6.1: 改造交互式欢迎界面

**Files:**
- Modify: `packages/markitai/src/markitai/cli/interactive.py`

**Step 1: 替换 Panel 欢迎为统一 UI**

```python
ui.title("Markitai Interactive")
console.print()
console.print(f"  Output: {output_dir}")
console.print(f"  LLM: {model_name}")
console.print(f"  Preset: {preset}")
console.print()
console.print("  Type 'help' for commands, 'q' to quit.")
console.print()
```

**Step 2: 测试**

```bash
uv run markitai -i
```

**Step 3: 提交**

---

## Phase 7: 测试与验证

### Task 7.1: 更新受影响的测试

**Files:**
- Modify: `packages/markitai/tests/unit/test_batch*.py`
- Modify: `packages/markitai/tests/unit/test_cli_main.py`

**Step 1: 更新断言以匹配新输出格式**

**Step 2: 运行全部测试**

```bash
uv run pytest packages/markitai/tests/unit -v
```

**Step 3: 提交**

---

### Task 7.2: 端到端验证

**Step 1: 验证各场景**

```bash
# 批量处理
uv run markitai packages/markitai/tests/fixtures --preset rich --no-cache -o output-final

# 单文件
uv run markitai packages/markitai/tests/fixtures/concise.md
uv run markitai packages/markitai/tests/fixtures/concise.md -o output-final/
uv run markitai packages/markitai/tests/fixtures/concise.md -o output-final/ -v

# 子命令
uv run markitai doctor
uv run markitai cache stats
uv run markitai config path
uv run markitai init --help

# Dry-run
uv run markitai packages/markitai/tests/fixtures --dry-run

# 日志文件
head -10 logs/*.log
```

**Step 2: 确认所有输出符合设计**

**Step 3: 最终提交**

---

## 完成标准

- [ ] 日志文件默认人类可读格式
- [ ] 批量处理摘要使用紧凑单列
- [ ] 进度条显示当前处理文件
- [ ] 单文件输出逻辑正确（无-o输出内容，有-o显示路径）
- [ ] Dry-run 使用统一 UI
- [ ] 所有 Panel 替换为统一 UI
- [ ] init 命令可用
- [ ] config init 已移除
- [ ] 交互式欢迎使用统一 UI
- [ ] 所有测试通过
- [ ] 跨平台提示正确
