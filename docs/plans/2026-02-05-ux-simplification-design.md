# Markitai UX 简化设计方案

## 背景

从产品经理视角，当前项目存在以下问题：

1. **配置复杂**（优先级最高）— 用户不知道如何配置 .env / API 密钥
2. **功能理解成本高** — `--llm`、`--alt`、`--desc`、`--ocr` 等选项太多，不知道怎么组合
3. **依赖过多** — 安装脚本输出混乱，依赖安装日志刷屏

## 设计目标

- 零配置即可使用基础功能（格式转换）
- 新用户可通过交互式引导快速上手
- 安装过程简洁清晰，类似 opencode.ai / skills.sh 的体验

---

## 方案一：交互式模式 (`-I/--interactive`)

### 触发方式

```bash
markitai              # 无参数 → 显示帮助
markitai file.pdf     # 有参数 → 直接执行
markitai -I           # 进入交互式引导
markitai --interactive
```

### 交互流程

```
$ markitai -I

┌─ Markitai Interactive Mode ─────────────────────┐
│                                                 │
│  ? What would you like to convert?              │
│    ○ Single file                                │
│    ● Directory                                  │
│    ○ URL                                        │
│                                                 │
└─────────────────────────────────────────────────┘

? Enter path: ./docs

? Enable LLM enhancement? (better formatting, metadata)
  ● Yes (recommended)
  ○ No (basic conversion only)

? No API key detected. How would you like to configure?
  ● Auto-detect (Claude CLI / Copilot CLI)
  ○ Enter API key manually
  ○ Use .env file
  ○ Skip for now

✓ Detected: Claude CLI (authenticated)
✓ Using model: claude-agent/sonnet

? Additional options:
  ☑ Generate alt text for images
  ☐ Generate image descriptions (JSON)
  ☑ Enable OCR for scanned documents
  ☐ Take page screenshots

Starting conversion...
```

### 技术实现

- 使用 `questionary` 或 `rich.prompt` 实现交互式界面
- 保存用户偏好到 `~/.markitai/config.json`
- 下次运行时记住上次的选择

---

## 方案二：三层 LLM 配置策略

### 层级设计

```
┌─────────────────────────────────────────────────┐
│ Layer 1: 零配置                                  │
│ - 默认只做格式转换，不需要任何 API              │
│ - markitai file.pdf → 直接输出 Markdown         │
└─────────────────────────────────────────────────┘
                    ↓ 用户启用 --llm
┌─────────────────────────────────────────────────┐
│ Layer 2: 智能检测                                │
│ - 自动检测已安装的 Claude CLI / Copilot CLI     │
│ - 自动检测环境变量中的 API 密钥                  │
│ - 按优先级选择：CLI > 环境变量 > 配置文件       │
└─────────────────────────────────────────────────┘
                    ↓ 未检测到任何配置
┌─────────────────────────────────────────────────┐
│ Layer 3: 交互式引导                              │
│ - 提示用户选择配置方式                          │
│ - 引导完成后保存配置                            │
└─────────────────────────────────────────────────┘
```

### 检测优先级

1. **Claude CLI** — `claude` 命令存在且已认证
2. **Copilot CLI** — `copilot` 命令存在且已认证
3. **环境变量** — `ANTHROPIC_API_KEY`、`OPENAI_API_KEY`、`GEMINI_API_KEY` 等
4. **配置文件** — `~/.markitai/config.json` 中的 `llm.model_list`

### 自动检测逻辑

```python
def auto_detect_llm_provider() -> str | None:
    """自动检测可用的 LLM 提供商"""
    # 1. 检测 Claude CLI
    if shutil.which("claude"):
        if check_claude_auth():
            return "claude-agent/sonnet"

    # 2. 检测 Copilot CLI
    if shutil.which("copilot"):
        if check_copilot_auth():
            return "copilot/gpt-4o"

    # 3. 检测环境变量
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic/claude-sonnet-4-20250514"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai/gpt-4o"
    if os.environ.get("GEMINI_API_KEY"):
        return "gemini/gemini-2.0-flash"

    return None
```

---

## 方案三：安装脚本优化

### 目标输出效果

```bash
$ curl -fsSL https://markitai.dev/install | sh

  Markitai Setup

  ✓ uv 0.5.14
  ✓ Python 3.13.2
  ✓ markitai 0.4.2

  ? Select optional components: (space to toggle, enter to confirm)
    ☑ Playwright (browser automation for JS pages)
    ☐ LibreOffice (legacy .doc/.xls/.ppt files)
    ☐ FFmpeg (audio/video processing)
    ☐ Claude CLI (use Claude subscription)
    ☐ Copilot CLI (use GitHub Copilot)

  ✓ Playwright browser

  Setup complete!

  Get started:
    markitai -I          # Interactive mode
    markitai file.pdf    # Convert a file
    markitai --help      # Show all options
```

### 关键改进

1. **静默化子命令输出**
   ```bash
   # Before
   uv tool install markitai  # 大量依赖安装日志

   # After
   uv tool install markitai 2>&1 | grep -v "^  "  # 或完全静默
   ```

2. **单次 checkbox 选择**
   - 使用类似 `gum choose --no-limit` 的交互式多选
   - 一次性选择所有需要的组件

3. **进度显示**
   - 使用 spinner 显示正在安装的组件
   - 安装完成后显示 ✓

### 实现方式

可选方案：
- **A) 纯 Shell** — 使用 `dialog` 或 `whiptail`（需要额外依赖）
- **B) 使用 gum** — [charmbracelet/gum](https://github.com/charmbracelet/gum)（单二进制，跨平台）
- **C) 简化为非交互** — 默认安装推荐组件，`--minimal` 最小安装

**推荐方案 B + C 结合**：
- 检测到 TTY 时使用 gum 交互
- 非 TTY（如 CI）时静默安装默认组件

---

## 方案四：`markitai doctor` 增强

### 当前问题

- 安装提示不够详细（未区分平台）
- 无法自动修复

### 改进设计

```bash
$ markitai doctor

  ┌─ System Health ───────────────────────────────┐
  │                                               │
  │  ✓ Playwright      Chromium browser ready     │
  │  ✓ RapidOCR        v3.5.0, lang: en           │
  │  ✗ LibreOffice     Not installed              │
  │  ✗ FFmpeg          Not installed              │
  │                                               │
  │  LLM Providers:                               │
  │  ✓ Claude CLI      Authenticated              │
  │  ○ Copilot CLI     Not installed              │
  │  ○ API Keys        No keys detected           │
  │                                               │
  └───────────────────────────────────────────────┘

  Missing components:

  LibreOffice (for .doc/.xls/.ppt conversion)
    macOS:   brew install --cask libreoffice
    Ubuntu:  sudo apt install libreoffice
    Windows: winget install LibreOffice.LibreOffice

  FFmpeg (for audio/video processing)
    macOS:   brew install ffmpeg
    Ubuntu:  sudo apt install ffmpeg
    Windows: winget install FFmpeg.FFmpeg

  Run 'markitai doctor --fix' to install missing components.
```

### 新增选项

```bash
markitai doctor           # 检查状态
markitai doctor --fix     # 自动安装缺失组件
markitai doctor --json    # JSON 输出（已有）
```

---

## 实现计划

### Phase 1: 交互式模式
1. 添加 `-I/--interactive` 选项到 CLI
2. 实现交互式流程（使用 questionary/rich）
3. 实现 LLM 自动检测逻辑

### Phase 2: 安装脚本优化
1. 静默化依赖安装输出
2. 实现 checkbox 组件选择（使用 gum 或 fallback）
3. 优化完成后的引导信息

### Phase 3: Doctor 增强
1. 改进跨平台安装提示
2. 实现 `--fix` 自动安装功能

### Phase 4: 文档更新
1. 更新 README 的快速开始部分
2. 添加交互式模式的使用说明

---

## 附录：参考项目

- [opencode.ai](https://opencode.ai/docs#install) — 安装后通过 `/connect` 配置
- [skills.sh](https://skills.sh/docs) — 简洁的单行安装
- [gum](https://github.com/charmbracelet/gum) — 终端交互式组件
