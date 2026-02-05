# Markitai CLI UX 统一视觉系统设计

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 减少 CLI 日志噪音，创建统一视觉主题

**Architecture:** 创建 ui.py 组件模块 + i18n.py 多语言支持，逐步改造各命令

**Tech Stack:** Rich（已有依赖），无需新增库

---

## 核心视觉组件

### 统一符号系统

```
✓  成功（绿色）
✗  失败（红色）
!  警告（黄色）
•  信息项（灰色）
◆  标题标记
│  引导线
```

### 颜色方案

- 主色：青色（cyan）- 标题和重点
- 成功：绿色（green）
- 失败：红色（red）
- 警告：黄色（yellow）
- 辅助：暗淡灰（dim）- 次要信息

### 信息层级

1. **标题层**：`◆ 标题` - 用于命令开始
2. **结果层**：`✓/✗/! 内容` - 用于检查结果
3. **详情层**：`  │ 详情` - 缩进+引导线，仅在需要时显示
4. **摘要层**：底部统计汇总

---

## 命令改造方案

### 1. `markitai doctor`

**当前问题**：
- 使用重型 Rich 表格
- FFmpeg 版本信息过于冗长
- 没有区分必需/可选依赖

**改造后**：
```
◆ 系统检查

必需依赖
  ✓ Python 3.13.1
  ✓ LibreOffice 24.8.3
  ✓ Playwright 已安装

可选依赖
  ! FFmpeg 未安装
    │ 用于音视频处理，如不需要可忽略

✓ 检查完成（3 必需通过，1 可选缺失）
```

### 2. `markitai cache stats`

**当前**：散乱的文本输出

**改造后**：
```
◆ 缓存统计

  • LLM 响应：  123 条（2.4 MB）
  • SPA 域名：   15 条
  • 代理检测：    8 条

总计：156 条缓存
```

### 3. `markitai config list`

**改造后**：
```
◆ 配置来源

  1. 命令行参数      │ 最高优先级
  2. 环境变量        │
  3. ./markitai.json │ ✓ 已加载
  4. ~/.markitai/    │ 未找到
  5. 默认值          │ 最低优先级
```

---

## 转换进度显示

### 单文件转换

```
◆ 转换 document.pdf

  │ 解析文档...
  │ 提取图片（3 张）...
  │ 生成 Markdown...
  ✓ 完成

输出：output/document.pdf.md
```

### 带 LLM 增强

```
◆ 转换 document.pdf

  │ 解析文档...
  │ 提取图片（3 张）...
  │ 生成 Markdown...
  │ LLM 增强中...
    │ 清理格式
    │ 生成元数据
    │ 图片描述（1/3）
    │ 图片描述（2/3）
    │ 图片描述（3/3）
  ✓ 完成

输出：output/document.pdf.llm.md
用量：1,234 tokens（$0.002）
```

### 批量转换

```
◆ 批量转换 ./docs

  ✓ report.docx
  ✓ slides.pptx
  │ data.xlsx...
  • meeting.pdf（等待中）
  • notes.md（等待中）

进度：2/5 完成
```

### 失败情况

```
◆ 批量转换 ./docs

  ✓ report.docx
  ✗ slides.pptx
    │ 错误：文件损坏
  ✓ data.xlsx

完成：2/3 成功，1 失败
详情：output/reports/markitai.xxx.report.json
```

---

## 多语言支持（i18n）

### 检测策略

优先级：`MARKITAI_LANG` > `LANG`/`LC_ALL` > 系统默认

### 文本定义结构

```python
TEXTS = {
    "doctor.title": {"en": "System Check", "zh": "系统检查"},
    "doctor.required": {"en": "Required Dependencies", "zh": "必需依赖"},
    "doctor.optional": {"en": "Optional Dependencies", "zh": "可选依赖"},
    # ...
}
```

### CLI 参数

可选添加 `--lang` 参数覆盖自动检测：

```bash
markitai doctor --lang zh
markitai doctor --lang en
```

---

## 实现方案

### 核心 UI 组件

```python
# packages/markitai/src/markitai/cli/ui.py

from rich.console import Console

console = Console()

MARK_SUCCESS = "✓"
MARK_ERROR = "✗"
MARK_WARNING = "!"
MARK_INFO = "•"
MARK_TITLE = "◆"
MARK_LINE = "│"

def title(text: str) -> None:
    console.print(f"[cyan]{MARK_TITLE}[/] [bold]{text}[/]")
    console.print()

def success(text: str) -> None:
    console.print(f"  [green]{MARK_SUCCESS}[/] {text}")

def error(text: str, detail: str | None = None) -> None:
    console.print(f"  [red]{MARK_ERROR}[/] {text}")
    if detail:
        console.print(f"    [dim]{MARK_LINE} {detail}[/]")

def warning(text: str, detail: str | None = None) -> None:
    console.print(f"  [yellow]{MARK_WARNING}[/] {text}")
    if detail:
        console.print(f"    [dim]{MARK_LINE} {detail}[/]")

def info(text: str) -> None:
    console.print(f"  [dim]{MARK_INFO}[/] {text}")

def step(text: str) -> None:
    console.print(f"  [dim]{MARK_LINE}[/] {text}")

def summary(text: str) -> None:
    console.print()
    console.print(f"[green]{MARK_SUCCESS}[/] {text}")
```

### i18n 模块

```python
# packages/markitai/src/markitai/cli/i18n.py

import locale
import os

def detect_language() -> str:
    lang = os.environ.get("MARKITAI_LANG", "")
    if not lang:
        lang = os.environ.get("LANG", "") or os.environ.get("LC_ALL", "")
    return "zh" if lang.startswith("zh") else "en"

TEXTS = {
    # 通用
    "success": {"en": "completed", "zh": "完成"},
    "failed": {"en": "failed", "zh": "失败"},
    # ... 完整文本定义
}

_lang = detect_language()

def t(key: str, **kwargs) -> str:
    text = TEXTS.get(key, {}).get(_lang, key)
    return text.format(**kwargs) if kwargs else text
```

---

## 文件影响

- **新增**：`cli/ui.py`（~100 行）
- **新增**：`cli/i18n.py`（~150 行）
- **修改**：`cli/commands/doctor.py`
- **修改**：`cli/commands/cache.py`
- **修改**：`cli/commands/config.py`
- **修改**：`workflow/core.py`（进度显示）
- **修改**：`utils/progress.py`（进度报告器）

---

## 重构策略

1. 创建 `cli/ui.py` 统一组件
2. 创建 `cli/i18n.py` 多语言支持
3. 逐个命令改造（doctor → cache → config → 转换流程）
4. 移除 Rich 表格，改用轻量列表
5. 统一日志级别：默认静默，`--verbose` 显示详情
