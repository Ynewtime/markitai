# 快速开始

## 环境要求

### 必需依赖

- **Python 3.11+** - 运行时环境
- **[uv](https://docs.astral.sh/uv/)** - 包管理器（推荐）或 pip

### 可选依赖

以下依赖用于特定功能：

| 依赖 | 用途 | 安装方式 |
|------|------|----------|
| **[Node.js 22+](https://nodejs.org/)** | `--agent-browser`（SPA 渲染） | 参见 [nodejs.org](https://nodejs.org/) |
| **[agent-browser](https://www.npmjs.com/package/agent-browser)** | `--agent-browser` | `pnpm add -g agent-browser && agent-browser install` |
| **Jina API 密钥** | `--jina`（URL 转换） | 设置 `JINA_API_KEY` 环境变量 |
| **LLM API 密钥** | `--llm`（AI 增强） | 设置 `OPENAI_API_KEY` 或对应提供商的密钥 |

::: tip 浏览器自动化
对于 SPA 网站（Twitter、React 应用等），需要安装 `agent-browser`：
```bash
pnpm add -g agent-browser
agent-browser install              # 下载 Chromium 浏览器
agent-browser install --with-deps  # Linux: 同时安装系统依赖
```
然后使用 `--agent-browser` 参数启用浏览器渲染。
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

脚本会：
- 检测 Python 3.11+
- 安装 [uv](https://docs.astral.sh/uv/) 包管理器（需要确认）
- 安装 markitai 及所有可选依赖
- 可选安装 `agent-browser` 用于浏览器自动化

### 手动安装

```bash
# 使用 uv（推荐）
uv tool install markitai

# 或使用 pip
pip install --user markitai
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

## 下一步

- [配置说明](/zh/guide/configuration) - 配置 LLM 提供商和其他设置
- [CLI 命令](/zh/guide/cli) - 完整命令参考
