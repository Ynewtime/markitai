# 快速开始

## 一键安装（推荐）

一条命令装好 Python（如缺少）、uv 和 markitai：

::: code-group
```bash [Linux/macOS]
curl -fsSL https://markitai.dev/setup.sh | sh
```

```powershell [Windows]
powershell -ExecutionPolicy ByPass -c "irm https://markitai.dev/setup.ps1 | iex"
```
:::

可选组件会逐项询问再装。Playwright 浏览器、网页工作台和 OCR 默认装，Claude 和 Copilot CLI 默认不装。

没有交互终端时只装核心。自动化场景设 `MARKITAI_INSTALL_OPTIONAL=1` 补装可选组件，设 `MARKITAI_VERSION=X.Y.Z` 固定版本。国内网络加速见[中国大陆用户指南](/zh/guide/configuration#中国大陆用户指南)。

## 第一次转换

转一个真实网页，就是你正在读的这篇：

```bash
mkai https://markitai.dev/zh/guide/getting-started --pure
```

`mkai` 是 `markitai` 的短别名。`--pure` 把纯 Markdown 打印到终端。想保存成文件，给一个输出目录：

```bash
markitai document.docx -o output/
markitai https://example.com/article -o output/
markitai ./docs -o output/
```

### 开启 LLM 增强

导出任意一家提供商的 API key，加上 `--llm`：

```bash
export GEMINI_API_KEY=...    # 或 OPENAI_ / ANTHROPIC_ / DEEPSEEK_ / OPENROUTER_API_KEY
markitai report.pdf -o output/ --llm
```

markitai 会按找到的 key 自动选一个模型。想自己指定，设 `MODEL=provider/model`。

需要配置文件、订阅制提供商（ChatGPT、Claude Code、Copilot）或多模型回退时，跑一遍引导配置：

```bash
markitai init
markitai doctor     # 查看装了什么、配了什么
```

## 可选能力

核心安装已经能转文档。用到时再装对应 extra：

```bash
uv tool install 'markitai[browser]' --force
```

| Extra | 启用能力 |
|-------|---------|
| `markitai[browser]` | 浏览器渲染（`-s playwright`），用于重 JS 页面和 URL 截图 |
| `markitai[ocr]` | 本地 OCR（`--ocr`），用于扫描版 PDF 和图片 |
| `markitai[serve]` | [网页工作台](/zh/guide/serve)及其 REST API |
| `markitai[mcp]` | 供 AI Agent 使用的 [MCP 服务器](/zh/guide/mcp) |
| `markitai[claude-agent]` | 用 Claude Code 订阅作为 LLM 提供商 |
| `markitai[copilot]` | 用 GitHub Copilot 订阅作为 LLM 提供商 |
| `markitai[legacy]` | 旧版 Office `.doc`、`.ppt` |
| `markitai[heif]` | HEIC、HEIF、AVIF 图片 |
| `markitai[svg]` | 高质量 SVG 渲染 |
| `markitai[extra-fetch]` | curl-cffi 客户端，应对有 TLS 指纹检测的站点 |
| `markitai[all]` | 以上全部 |

两个远程抓取策略不需要 extra，只需要凭据：`-s jina` 读 `JINA_API_KEY`，`-s cloudflare` 读 `CLOUDFLARE_API_TOKEN` 和 `CLOUDFLARE_ACCOUNT_ID`。

装了 browser extra 之后，装一次 Chromium：

```bash
markitai doctor --fix
```

## 手动安装

已有 Python 3.11 到 3.13 时：

```bash
uv tool install markitai     # 推荐
pipx install markitai
uv pip install markitai      # 装进当前虚拟环境
```

手动安装不带任何可选组件。用 `markitai doctor` 查看可用能力，用 `markitai init` 配置 LLM 提供商。

## 功能说明

**预设**打包了常用参数。`minimal` 只做基础转换，`standard` 加 LLM 清洗和图片分析，`rich` 再加页面截图。任何一项都能用 `--no-*` 关掉，例如 `--preset rich --no-desc`。

**URL** 先在本机抓取。公开页面本机抓不到时，markitai 可能回退到远程阅读服务（Defuddle、Jina 或 Cloudflare），并在 stderr 提示一次。私有、内网和带凭据的 URL 永远不出本机。`MARKITAI_NO_REMOTE_FETCH=1` 强制全部本地。

**目录**按批量转换，带进度显示和 JSON 报告。中断后加 `--resume` 接着跑。

## 输出结构

```text
output/
├── document.pdf.md          # Markdown（带 --llm 时默认只写 .llm.md，除非 --keep-base）
├── document.pdf.llm.md      # LLM 增强版
└── .markitai/
    ├── assets/              # 源文档里的图片，以及图片描述 images.json
    ├── screenshots/         # 页面、幻灯片或整页截图（--screenshot）
    ├── reports/             # 批量运行的 JSON 报告
    └── states/              # 供 --resume 使用的批量状态
```

输出文件名是完整输入名加 `.md`，所以 `report.pdf` 和 `report.docx` 不会互相覆盖。

## 支持的格式

| 格式 | 扩展名 |
|------|--------|
| Office | `.docx`、`.doc`、`.pptx`、`.ppt`、`.xlsx`、`.xls`、`.odt`、`.ods`、`.numbers` |
| PDF | `.pdf` |
| 文本与标记 | `.txt`、`.md`、`.markdown`、`.html`、`.htm`、`.xhtml`、`.xml`、`.csv`、`.tsv`、`.rtf`、`.rst`、`.org`、`.tex` |
| 图片 | `.jpg`、`.jpeg`、`.png`、`.webp`、`.svg`、`.gif`、`.bmp`、`.tiff`、`.tif`、`.heic`、`.heif`、`.avif`（后三种需要 `markitai[heif]`） |
| 其他文档 | `.epub`、`.eml`、`.msg`、`.ipynb` |
| URL | `http://`、`https://` |

## 平台特定功能

Windows、Linux、macOS 功能一致，只有两点要留意：

- **EMF/WMF 图片**是 Windows 专有格式，只在 Windows 上转换。
- **PPTX 幻灯片截图**需要渲染器。Windows 用 Microsoft Office 或 LibreOffice；Linux 需要 LibreOffice（`apt-get install libreoffice`）；macOS 优先 LibreOffice（`brew install --cask libreoffice`），否则调用已装的 PowerPoint，首次会弹一次授权对话框，且需要桌面会话。无头 Mac 上在配置里设 `"office": { "macos_fallback": false }` 关掉。

旧版 `.doc`、`.ppt` 需要 `markitai[legacy]`，任何平台都不用装 Office。

## 下一步

- [配置说明](/zh/guide/configuration)：LLM 提供商与全部设置
- [CLI 命令](/zh/guide/cli)：每个命令和参数
