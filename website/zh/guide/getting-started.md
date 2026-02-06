# 快速开始

## 环境要求

### 必需依赖

- **Python 3.11-3.13** - 运行时环境（由于 onnxruntime 限制，暂不支持 3.14）
- **[uv](https://docs.astral.sh/uv/)** - 包管理器（推荐）

### 可选依赖

以下依赖用于特定功能：

| 依赖 | 用途 | 安装方式 |
|------|------|----------|
| **Playwright** | `--playwright`（SPA 渲染） | 包随 markitai 安装，浏览器需运行 `uv run playwright install chromium` |
| **FFmpeg** | 音视频处理 | `apt install ffmpeg` (Linux) / `brew install ffmpeg` (macOS) |
| **Jina API 密钥** | `--jina`（URL 转换） | 设置 `JINA_API_KEY` 环境变量 |
| **LLM API 密钥** | `--llm`（AI 增强） | 设置 `OPENAI_API_KEY` 或对应提供商的密钥 |

::: tip 浏览器自动化
对于 SPA 网站（Twitter、React 应用等），会自动使用 Playwright。首次使用前需安装浏览器：
```bash
uv run playwright install chromium
# Linux 还需安装系统依赖：
uv run playwright install-deps chromium
```
然后使用 `--playwright` 参数强制启用浏览器渲染。
:::

## 安装

### 一键安装（推荐）

运行安装脚本，自动安装 Python、UV 和 markitai：

::: code-group
```bash [Linux/macOS]
curl -fsSL https://raw.githubusercontent.com/Ynewtime/markitai/main/scripts/setup.sh | sh
```

```powershell [Windows]
irm https://raw.githubusercontent.com/Ynewtime/markitai/main/scripts/setup.ps1 | iex
```
:::

::: warning 安全提示
- 以 root/管理员 身份运行时脚本会发出警告
- 所有安装操作都需要明确确认（默认: 否）
- 远程脚本执行需要两步确认
:::

脚本会：
- 检测 Python 3.11-3.13
- 安装 [uv](https://docs.astral.sh/uv/) 包管理器（需要确认）
- 安装 markitai 及所有可选依赖

#### 版本固定

使用环境变量固定特定版本：

::: code-group
```bash [Linux/macOS]
export MARKITAI_VERSION="0.5.0"
export UV_VERSION="0.9.27"
curl -fsSL https://raw.githubusercontent.com/Ynewtime/markitai/main/scripts/setup.sh | sh
```

```powershell [Windows]
$env:MARKITAI_VERSION = "0.5.0"
$env:UV_VERSION = "0.9.27"
irm https://raw.githubusercontent.com/Ynewtime/markitai/main/scripts/setup.ps1 | iex
```
:::

### 手动安装

```bash
# 使用 uv（推荐）
uv tool install markitai

# 或使用 uv pip（用于虚拟环境）
uv pip install markitai
```

## 快速上手

### 基础转换

将单个文档转换为 Markdown：

```bash
markitai document.docx
```

输出保存到当前目录，同时打印到 stdout。使用 `-o` 指定其他输出目录：

```bash
markitai document.docx -o output/
```

### URL 转换

直接转换网页：

```bash
markitai https://example.com/article
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

## 输出结构

```
output/
├── document.docx.md        # 基础 Markdown 输出
├── document.docx.llm.md    # LLM 增强版本（使用 --llm 时）
├── assets/
│   ├── document.docx.0001.jpg
│   └── images.json         # 图片描述
```

## 支持的格式

| 格式 | 扩展名 |
|------|--------|
| Word | `.docx`, `.doc` |
| PowerPoint | `.pptx`, `.ppt` |
| Excel | `.xlsx`, `.xls` |
| PDF | `.pdf` |
| 文本 | `.txt`, `.md` |
| 图片 | `.jpg`, `.jpeg`, `.png`, `.webp` |
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

::: tip Windows 性能
在 Windows 上，markitai 自动将并发数限制为 4 个线程，以应对较高的线程切换开销。
:::

### Linux

| 功能 | 支持 | 说明 |
|------|------|------|
| 旧版 Office（`.doc`、`.xls`、`.ppt`） | ❌ 不支持 | 需要 Windows COM |
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
| 旧版 Office（`.doc`、`.xls`、`.ppt`） | ❌ 不支持 | 需要 Windows COM |
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
