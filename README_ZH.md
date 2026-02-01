# Markitai

[English](./README.md) | 简体中文

开箱即用的 Markdown 转换器，原生支持 LLM 增强。

## 功能特性

- **多格式支持** - DOCX/DOC、PPTX/PPT、XLSX/XLS、PDF、TXT、MD、JPG/PNG/WebP、URLs
- **LLM 增强** - 格式清洗、元数据生成、图像分析
- **批量处理** - 并发转换、断点续传、进度显示
- **OCR 识别** - 扫描件 PDF 和图片的文字提取
- **URL 转换** - 直接转换网页，支持 SPA 浏览器渲染
- **智能缓存** - LLM 结果缓存、SPA 域名学习、自动代理检测

## 安装

### 一键安装（推荐）

```bash
# Linux/macOS
curl -fsSL https://raw.githubusercontent.com/Ynewtime/markitai/main/scripts/setup-zh.sh | sh

# Windows (PowerShell)
irm https://raw.githubusercontent.com/Ynewtime/markitai/main/scripts/setup-zh.ps1 | iex
```

### 手动安装

```bash
# 需要 Python 3.11-3.13（暂不支持 3.14）
uv tool install markitai

# 或使用 uv pip（用于虚拟环境）
uv pip install markitai
```

## 快速开始

```bash
# 基础转换
markitai document.docx

# URL 转换
markitai https://example.com/article

# LLM 增强
markitai document.docx --llm

# 使用预设
markitai document.pdf --preset rich      # LLM + alt + desc + 截图
markitai document.pdf --preset standard  # LLM + alt + desc
markitai document.pdf --preset minimal   # 仅基础转换

# 批量处理
markitai ./docs -o ./output

# 断点续传
markitai ./docs -o ./output --resume

# 批量 URL 处理（自动识别 .urls 文件）
markitai urls.urls -o ./output
```

## 输出结构

```
output/
├── document.docx.md        # 基础 Markdown
├── document.docx.llm.md    # LLM 增强版本
├── assets/
│   ├── document.docx.0001.jpg
│   └── images.json         # 图片描述
├── screenshots/            # 页面截图（使用 --screenshot）
│   └── example_com.full.jpg
```

## 配置

优先级：CLI 参数 > 环境变量 > 配置文件 > 默认值

```bash
# 查看配置
markitai config list

# 初始化配置文件
markitai config init -o .

# 查看缓存状态
markitai cache stats

# 清除缓存
markitai cache clear

# 检查系统健康状态和依赖
markitai doctor
```

配置文件位置：`./markitai.json` 或 `~/.markitai/config.json`

### 本地 Provider（基于订阅）

使用您现有的 Claude Code 或 GitHub Copilot 订阅：

```bash
# Claude Agent（需要 Claude Code CLI）
markitai document.pdf --llm  # 在配置中设置 claude-agent/sonnet

# GitHub Copilot（需要 Copilot CLI）
markitai document.pdf --llm  # 在配置中设置 copilot/gpt-5.2
```

安装 CLI 工具：
```bash
# Claude Code CLI
curl -fsSL https://claude.ai/install.sh | bash

# GitHub Copilot CLI
curl -fsSL https://gh.io/copilot-install | bash
```

## 环境变量

| 变量 | 说明 |
|------|------|
| `OPENAI_API_KEY` | OpenAI API 密钥 |
| `GEMINI_API_KEY` | Google Gemini API 密钥 |
| `DEEPSEEK_API_KEY` | DeepSeek API 密钥 |
| `ANTHROPIC_API_KEY` | Anthropic API 密钥 |
| `JINA_API_KEY` | Jina Reader API 密钥（URL 转换） |

## 依赖项目

- [pymupdf4llm](https://github.com/pymupdf/RAG) - PDF 转换
- [markitdown](https://github.com/microsoft/markitdown) - Office 文档和 URL 转换
- [LiteLLM](https://github.com/BerriAI/litellm) - LLM 网关
- [RapidOCR](https://github.com/RapidAI/RapidOCR) - OCR 识别

## 文档

- [快速开始](https://markitai.ynewtime.com/zh/guide/getting-started)
- [配置说明](https://markitai.ynewtime.com/zh/guide/configuration)
- [CLI 命令](https://markitai.ynewtime.com/zh/guide/cli)

## 许可证

MIT
