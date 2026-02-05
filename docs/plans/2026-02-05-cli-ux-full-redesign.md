# Markitai CLI UX 全面改造设计

> **Goal:** 全面改造 markitai CLI 输出体验，建立统一视觉系统，减少噪音，提升新用户上手体验。

## 背景

当前 CLI 存在以下问题：
- 输出风格不统一（Rich Table、Panel、loguru 日志混杂）
- 日志文件过于冗余（JSON 序列化，298 行 = 251KB）
- 批量处理输出噪音大
- 新用户缺乏引导

## 设计原则

1. **统一**：所有输出使用一致的符号系统（✓/✗/!/•/◆/│）
2. **分层**：默认简洁，--verbose 显示详细
3. **跨平台**：考虑 Linux/macOS/Windows 差异
4. **管道友好**：支持 stdout 输出、--quiet 静默模式

---

## 视觉系统

### 符号系统

| 符号 | 名称 | 用途 |
|------|------|------|
| ✓ | MARK_SUCCESS | 成功、完成 |
| ✗ | MARK_ERROR | 错误、失败 |
| ! | MARK_WARNING | 警告、注意 |
| • | MARK_INFO | 信息项、列表 |
| ◆ | MARK_TITLE | 标题、章节 |
| │ | MARK_LINE | 缩进、详情行 |

### 颜色方案

| 元素 | 颜色 | Rich 标记 |
|------|------|-----------|
| 标题 | 青色 | `[cyan]` |
| 成功 | 绿色 | `[green]` |
| 错误 | 红色 | `[red]` |
| 警告 | 黄色 | `[yellow]` |
| 次要信息 | 灰色 | `[dim]` |

### 示例

```
◆ System Check

  ✓ Playwright installed
  ✗ LibreOffice not found
    │ Install: sudo apt install libreoffice
  ! FFmpeg missing (optional)

✓ Check complete (2 passed, 1 optional missing)
```

---

## 日志系统

### 终端日志

**分层显示策略：**

| 模式 | 显示内容 |
|------|----------|
| 默认 | 仅关键结果（完成、错误） |
| --verbose | 处理步骤、详细信息 |
| --quiet | 完全静默 |

**默认模式示例：**
```
✓ Done: 7 files, 3 URLs (2m 25s, $0.058)
```

**--verbose 模式示例：**
```
◆ Converting

  │ Processing file_example.pdf...
  │ LLM enhance: haiku, 806 tokens
  ✓ file_example.pdf (4.2s)

✓ Done: 7 files, 3 URLs (2m 25s, $0.058)
```

### 日志文件

**格式：人类可读（默认）**

```
2026-02-05 20:48:16 | INFO  | batch:234 | Done: 7 files
2026-02-05 20:48:16 | DEBUG | llm:89   | Model: haiku, tokens: 806
```

**可选 JSON（通过配置）：**
```json
{"ts":"2026-02-05T20:48:16","lvl":"INFO","src":"batch:234","msg":"Done: 7 files"}
```

**配置项：**
```json
{
  "log": {
    "format": "text",
    "level": "INFO"
  }
}
```

---

## 批量处理输出

### 进度条

**处理中：**
```
◆ Converting

  Files  ━━━━━━━━━━━━━━━━━━━━━━━━━━  5/7  (processing: example.pdf)
  URLs   ━━━━━━━━━━━━━━━━━━━━━━━━━━  2/3
```

**完成后：**
```
◆ Converting

  ✓ 7 files completed (1:58)
  ✓ 3 URLs completed (2:25)

  ! 1 warning: rate limit on example.com
```

### 摘要（紧凑单列）

```
✓ Done: 7 files, 3 URLs (2m 25s, $0.058)

  • Files: 7/7 ✓  Cache: 0
  • URLs:  3/3 ✓  Cache: 0

  Output: output-3/
```

### Panel 替换（无边框）

**旧样式：**
```
╭─────────────────── Error ───────────────────╮
│ Failed to fetch https://example.com         │
│ Connection timeout after 30s                │
╰─────────────────────────────────────────────╯
```

**新样式：**
```
✗ Failed to fetch https://example.com
  │ Connection timeout after 30s
```

---

## 单文件处理

| 参数 | 行为 |
|------|------|
| 无 `-o` | 输出 Markdown 内容到 stdout |
| `-o output/` | 保存文件，打印保存路径 |
| `-o output/ -v` | 保存文件，显示详细处理信息 |
| `--quiet` | 完全静默 |

**无 -o（输出内容）：**
```bash
markitai file.pdf
```
```markdown
# File Content

This is the converted content...
```

**有 -o（显示路径）：**
```bash
markitai file.pdf -o output/
```
```
✓ output/file.pdf.md
```

**有 -o + verbose：**
```bash
markitai file.pdf -o output/ -v
```
```
◆ Converting file.pdf

  ✓ Parsed (0.8s)
  ✓ Images: 3 extracted

✓ output/file.pdf.md (4.2s)
```

---

## Dry-run 预览

```bash
markitai input/ --dry-run
```
```
◆ Dry Run

  Files (2)
    • file1.pdf
    • file2.docx

  URLs (1)
    • https://example.com

  ─────────────────
  Total: 2 files, 1 URL
```

---

## 新用户引导

### 未配置 LLM 时的提示

根据操作系统显示不同的命令：

**Linux/macOS：**
```
! LLM not configured

  Quick setup (choose one):

  • Claude CLI:  claude login
  • Copilot CLI: copilot auth login
  • API Key:     export GEMINI_API_KEY=your_key
  • Wizard:      markitai init

  Run 'markitai doctor' to check status.
```

**Windows PowerShell：**
```
! LLM not configured

  Quick setup (choose one):

  • Claude CLI:  claude login
  • Copilot CLI: copilot auth login
  • API Key:     $env:GEMINI_API_KEY="your_key"
  • Wizard:      markitai init

  Run 'markitai doctor' to check status.
```

**Windows CMD：**
```
! LLM not configured

  Quick setup (choose one):

  • Claude CLI:  claude login
  • Copilot CLI: copilot auth login
  • API Key:     set GEMINI_API_KEY=your_key
  • Wizard:      markitai init

  Run 'markitai doctor' to check status.
```

### markitai init 命令

**新的顶级命令，替代原 `config init`：**

```bash
markitai init        # 交互式向导
markitai init -y     # 快速模式，生成默认配置
```

**交互流程：**
```
◆ Markitai Setup

  Detecting available LLM providers...

  ✓ Claude CLI found
  ✓ GitHub Copilot CLI found

  Choose default provider:

  1. Claude CLI (recommended)
  2. GitHub Copilot CLI
  3. Enter API key manually
  4. Skip for now

  Choice [1]:
```

---

## 交互式模式

```bash
markitai -i
```
```
◆ Markitai Interactive

  Output: ./output
  LLM: claude-agent/haiku
  Preset: rich

  Type 'help' for commands, 'q' to quit.

>
```

---

## CLI 命令变更汇总

| 变更类型 | 旧命令 | 新命令 |
|----------|--------|--------|
| 移除 | `markitai config init` | - |
| 新增 | - | `markitai init` |
| 新增 | - | `markitai init -y` |
| 保留 | `markitai config list` | 不变 |
| 保留 | `markitai config path` | 不变 |
| 保留 | `markitai config get` | 不变 |
| 保留 | `markitai config validate` | 不变 |
| 保留 | `markitai doctor` | 不变 |
| 保留 | `markitai cache stats` | 不变 |

### 参数行为

| 参数 | 行为 |
|------|------|
| `-v, --verbose` | 显示详细处理信息 |
| `-q, --quiet` | 完全静默（仅错误输出到 stderr） |
| `--dry-run` | 预览模式，不实际执行 |
| `-o, --output` | 指定输出目录，不输出内容到 stdout |

---

## 实现范围

### 核心 UI 模块（已完成 ✓）
- `cli/ui.py` - 统一 UI 组件
- `cli/i18n.py` - 多语言支持

### 子命令（已完成 ✓）
- `cli/commands/doctor.py`
- `cli/commands/cache.py`
- `cli/commands/config.py`

### 批量处理（待改造）
- `batch.py` - 进度条、摘要表格、日志面板
- `cli/processors/batch.py` - 批量处理输出

### 单文件处理（待改造）
- `cli/processors/file.py` - 单文件输出逻辑
- `cli/processors/url.py` - URL 处理输出

### 日志系统（待改造）
- `cli/logging_config.py` - 日志格式配置

### 新用户引导（待新增）
- `cli/main.py` - 添加 `markitai init` 命令
- `cli/commands/init.py` - init 命令实现（新文件）

### 验证器和面板（待改造）
- `cli/processors/validators.py` - 警告面板
- `cli/interactive.py` - 交互式欢迎界面

### 不改造的部分
- JSON 输出（`--json` 参数）- 保持结构化输出
- 内部日志（logger.debug）- 仅改格式，不改调用点
- 测试文件 - 随对应模块更新

---

## 向后兼容

### 保持兼容
- 所有现有命令行参数
- `--json` 输出格式不变
- 退出码语义不变（0=成功，非0=失败）
- stdout/stderr 分离（内容到 stdout，日志到 stderr）

### 破坏性变更

| 变更 | 影响 | 迁移方案 |
|------|------|----------|
| 移除 `config init` | 脚本中使用的命令 | 改用 `markitai init` |
| 日志文件格式 | 解析 JSON 日志的脚本 | 配置 `log.format: "json"` 恢复 |
| 终端输出格式 | 解析终端输出的脚本 | 使用 `--json` 获取结构化输出 |

### 配置文件新增项

```json
{
  "log": {
    "level": "INFO",
    "format": "text",
    "dir": "./logs"
  }
}
```

### 环境变量

| 变量 | 用途 |
|------|------|
| `MARKITAI_LOG_FORMAT` | 覆盖日志格式（text/json） |
| `MARKITAI_LANG` | 覆盖界面语言（en/zh） |
| `NO_COLOR` | 禁用颜色输出（标准） |
