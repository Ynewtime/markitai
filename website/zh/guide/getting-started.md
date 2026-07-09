# 快速开始

## 环境要求

### 必需依赖

- **Python 3.11-3.14** - 运行时环境
- **[uv](https://docs.astral.sh/uv/)** - 包管理器（推荐）

### 可选依赖

以下依赖用于特定功能：

| 依赖 | 用途 | 安装方式 |
|------|------|----------|
| **Playwright** | `-s playwright`（SPA 渲染） | `uv pip install markitai[browser]`，浏览器需运行 `uv run playwright install chromium` |
| **FFmpeg** | `doctor`/`init` 会检测（属于 `markitdown[all]` 的间接依赖）；markitai 目前尚未注册任何音视频转换格式 | `apt install ffmpeg` (Linux) / `brew install ffmpeg` (macOS) |
| **Jina API 密钥** | `-s jina`（URL 转换） | 设置 `JINA_API_KEY` 环境变量 |
| **LLM API 密钥** | `--llm`（AI 增强） | 设置 `OPENAI_API_KEY` 或对应提供商的密钥。订阅制提供商（`chatgpt/`、`claude-agent/`、`copilot/`）使用 CLI/OAuth 认证 |
| **Cloudflare** | `-s cloudflare`（云端渲染与转换） | 设置 `CLOUDFLARE_API_TOKEN` 和 `CLOUDFLARE_ACCOUNT_ID` 环境变量 |
| **CairoSVG** | 高质量 SVG 渲染 | `uv pip install markitai[svg]` |
| **pillow-heif** | HEIC/HEIF/AVIF 图片输入 | `uv pip install markitai[heif]` |
| **kreuzberg** | `.xml`、`.tsv`、`.rtf`、`.rst`、`.org`、`.tex`、`.odt`、`.ods` 转换 | `uv pip install markitai[kreuzberg]`（已包含在 `[all]` 中） |

::: tip 浏览器自动化
对于 SPA 网站（Twitter、React 应用等），会自动使用 Playwright。首次使用前需安装浏览器：
```bash
uv run playwright install chromium
# Linux 还需安装系统依赖：
uv run playwright install-deps chromium
```
然后使用 `-s playwright` 强制启用浏览器渲染。
:::

## 安装

### 一键安装（推荐）

运行安装脚本，自动安装 Python、UV 和 markitai：

::: code-group
```bash [Linux/macOS]
curl -fsSL https://markitai.dev/setup.sh | sh
```

```powershell [Windows]
powershell -ExecutionPolicy ByPass -c "irm https://markitai.dev/setup.ps1 | iex"
```
:::

::: warning 安全提示
- 以 root/管理员 身份运行时，脚本会先检测并询问是否继续
- 每个组件安装前都会询问确认——uv 和 Playwright 浏览器默认 Yes；LibreOffice、FFmpeg 和 Claude/Copilot CLI 默认 No
:::

脚本会：
- 检测 / 自动安装 Python 3.11-3.14（无需确认）
- 安装 [uv](https://docs.astral.sh/uv/) 包管理器（需确认，默认 Yes）
- 安装 markitai 本体及其纯 pip 依赖的可选组件（浏览器自动化、`extra-fetch`、`kreuzberg`、`svg`、`heif`），无需额外确认；LibreOffice、FFmpeg 和 Claude/Copilot CLI 会在随后单独询问确认（默认 No）

#### 版本固定

使用环境变量固定特定版本：

::: code-group
```bash [Linux/macOS]
export MARKITAI_VERSION="0.14.0"
export UV_VERSION="0.9.27"
curl -fsSL https://markitai.dev/setup.sh | sh
```

```powershell [Windows]
$env:MARKITAI_VERSION = "0.14.0"
$env:UV_VERSION = "0.9.27"
powershell -ExecutionPolicy ByPass -c "irm https://markitai.dev/setup.ps1 | iex"
```
:::

### 手动安装

如果你已有 Python 3.11–3.14，只想要最小安装：

```bash
# 使用 uv（推荐，隔离环境）
uv tool install "markitai[all]"

# 或使用 uv pip（安装到虚拟环境）
uv pip install "markitai[all]"
```

与一键安装不同，手动安装**不会**帮你配置可选组件和配置文件，剩余步骤需自行完成：

```bash
markitai doctor           # 查看已装 / 缺失的组件
markitai doctor --fix     # 安装 Playwright Chromium 浏览器（用于 --playwright）
markitai init             # 创建配置并设置 LLM 提供方
```

::: tip markitai 和 mkai 都可用
每次安装都会同时提供 `markitai` 命令**和更短的 `mkai` 别名**——两者是完全相同的命令（`mkai --help` 等同 `markitai --help`）。若你的 `PATH` 上已存在别的 `mkai`，请使用完整的 `markitai` 以避免歧义。
:::

## 快速上手

### 首次运行

新用户推荐使用交互模式，引导完成初始设置：

```bash
markitai -I
```

或使用配置向导初始化：

```bash
# 交互式配置向导
markitai init

# 快速模式（生成默认配置）
markitai init --yes
```

### 基础转换

将单个文档转换为 Markdown：

```bash
markitai document.docx
```

不指定 `-o` 时，输出打印到 stdout。使用 `-o` 可指定输出目录：

```bash
markitai document.docx -o output/
```

### URL 转换

直接转换网页：

```bash
markitai https://example.com/article -o output/
```

### LLM 增强

启用 AI 驱动的格式清洗和优化：

```bash
markitai document.docx --llm
```

这需要设置 API 密钥（参见[配置说明](/zh/guide/configuration)）。

### 使用预设

Markitai 提供三种预设，适用于常见场景：

```bash
# Rich: LLM + alt 文本 + 描述 + 截图
markitai document.pdf --preset rich

# Standard: LLM + alt 文本 + 描述
markitai document.pdf --preset standard

# Minimal: 仅基础转换
markitai document.pdf --preset minimal
```

### 批量处理

转换目录中的多个文件：

```bash
markitai ./docs -o ./output
```

恢复中断的批量处理：

```bash
markitai ./docs -o ./output --resume
```

### 系统检查

验证所有依赖项，自动修复缺失组件：

```bash
# 检查系统状态
markitai doctor

# 自动修复缺失组件
markitai doctor --fix
```

## 输出结构

```
output/
├── document.pdf.md           # 基础 Markdown（--llm 模式下默认跳过，除非加 --keep-base）
├── document.pdf.llm.md       # LLM 增强版本（使用 --llm 时）
├── .markitai/                  # 元数据命名空间
│   ├── assets/
│   │   ├── document.docx.0001.jpg   # 源文档内嵌的图片
│   │   └── images.json       # 图片描述
│   ├── screenshots/           # 页面/幻灯片截图（仅 PDF/PPTX；URL 为整页截图；--screenshot）
│   │   └── document.pdf.page0001.jpg
│   ├── reports/                # 转换报告（JSON）——批量/URL 批量任务默认生成，或 output.report = true 时生成
│   └── states/                 # 批处理状态文件（用于 --resume）
```

::: tip
输出文件名在完整输入文件名后追加 `.md`：`document.docx` → `document.docx.md`（`--llm` 模式下为 `document.docx.llm.md`）。源文件格式在输出名中保持可见，不同输入（例如 `report.pdf` 和 `report.docx`）永远不会命名冲突。
:::

::: tip
在 `--llm` 模式下，默认只写入 `.llm.md`。使用 `--keep-base` 可以同时写入基础 `.md` 文件。
:::

## 支持的格式

| 格式 | 扩展名 |
|------|--------|
| Office 文档 | `.docx`, `.doc`, `.pptx`, `.ppt`, `.xlsx`, `.xls`, `.odt`, `.ods`, `.numbers` |
| PDF | `.pdf` |
| 文本 / 标记 / 结构化数据 | `.txt`, `.md`, `.markdown`, `.html`, `.htm`, `.xhtml`, `.xml`, `.csv`, `.tsv`, `.rtf`, `.rst`, `.org`, `.tex` |
| 图片 | `.jpg`, `.jpeg`, `.png`, `.webp`, `.svg`, `.gif`, `.bmp`, `.tiff`, `.tif`, `.heic`, `.heif`, `.avif`（后三种需要 `markitai[heif]`） |
| 其他文档 | `.epub`, `.eml`, `.msg`, `.ipynb` |
| URL | `http://`, `https://` |

## 平台特定功能

部分功能在不同平台上有差异：

### Windows

| 功能 | 支持 | 说明 |
|------|------|------|
| 旧版 Office（`.doc`、`.xls`、`.ppt`） | ✅ 完全支持 | 使用 COM 自动化 |
| PPTX 幻灯片渲染 | ✅ 完全支持 | 优先使用 MS Office，LibreOffice 备选 |
| EMF/WMF 图片 | ✅ 完全支持 | 原生支持 |
| 浏览器自动化 | ✅ 完全支持 | 隐藏窗口模式 |

### Linux

| 功能 | 支持 | 说明 |
|------|------|------|
| 旧版 Office（`.doc`、`.xls`、`.ppt`） | ✅ 完全支持 | 需要 LibreOffice（Windows 使用 COM，LibreOffice 作为备选） |
| PPTX 幻灯片渲染 | ✅ 完全支持 | 需要 LibreOffice |
| EMF/WMF 图片 | ❌ 不支持 | Windows 专有格式 |
| 浏览器自动化 | ✅ 完全支持 | 需要系统依赖 |

**安装 LibreOffice：**
```bash
# Ubuntu/Debian
sudo apt-get install libreoffice

# Fedora/RHEL
sudo dnf install libreoffice
```

**安装 Playwright 浏览器：**
```bash
uv run playwright install chromium
uv run playwright install-deps chromium  # 安装系统依赖
```

### macOS

| 功能 | 支持 | 说明 |
|------|------|------|
| 旧版 Office（`.doc`、`.xls`、`.ppt`） | ✅ 完全支持 | 需要 LibreOffice |
| PPTX 幻灯片渲染 | ✅ 完全支持 | 需要 LibreOffice |
| EMF/WMF 图片 | ❌ 不支持 | Windows 专有格式 |
| 浏览器自动化 | ✅ 完全支持 | - |

**安装 LibreOffice：**
```bash
brew install --cask libreoffice
```

## 下一步

- [配置说明](/zh/guide/configuration) - 配置 LLM 提供商和其他设置
- [CLI 命令](/zh/guide/cli) - 完整命令参考
