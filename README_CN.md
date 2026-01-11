# MarkIt

文档转 Markdown 工具，支持 LLM 增强。

## 特性

- **多格式支持**：Word (.docx/.doc)、PowerPoint (.pptx/.ppt)、Excel (.xlsx/.xls)、PDF、HTML、图片
- **LLM 增强**：清理页眉页脚、修复标题层级、添加 frontmatter、生成摘要
- **图像处理**：自动压缩、格式转换、去重、LLM 智能分析
- **批量处理**：递归目录转换，支持断点续传
- **多 LLM 提供商**：OpenAI、Anthropic、Google Gemini、Ollama、OpenRouter，支持并发 fallback
- **基于能力路由**：自动将文本任务路由到文本模型，视觉任务路由到视觉模型

## 安装

```bash
# 使用 uv（推荐）
uv pip install markit

pip install markit
```

**系统依赖**（处理 .doc/.ppt/.xls）：

```bash
# Ubuntu/Debian
sudo apt install libreoffice-core

# macOS
brew install --cask libreoffice

# Windows
scoop install libreoffice
```

**可选**（更好性能）：

```bash
# Ubuntu/Debian
sudo apt install pandoc
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
cargo install oxipng
# or: brew install pandoc oxipng

# macOS
brew install pandoc oxipng

# Windows
scoop install pandoc oxipng
#   或下载安装:
#   Pandoc: https://pandoc.org/installing.html
#   oxipng: https://github.com/shssoichiro/oxipng/releases
```

## 使用

```bash
# 基础转换
markit convert document.docx

# LLM 增强（格式清洗、添加 frontmatter、生成摘要）
markit convert document.docx --llm

# 图片分析（生成 alt 文本）
markit convert document.pdf --analyze-image

# 生成图片描述 markdown 文件
markit convert document.pdf --analyze-image-with-md

# 批量转换
markit batch ./docs -o ./output -r

# 恢复中断的批处理
markit batch ./docs -o ./output --resume

# 极速模式（跳过验证、最少重试）
markit batch ./docs -o ./output --fast
```

## 命令

| 命令 | 说明 |
|------|------|
| `markit convert <file>` | 转换单个文件 |
| `markit batch <dir>` | 批量转换目录 |
| `markit config init` | 创建配置文件 |
| `markit config test` | 验证配置 |
| `markit config list` | 显示当前配置 |
| `markit config locations` | 显示配置文件搜索路径 |
| `markit provider add` | 添加 LLM 提供商凭证 |
| `markit provider test` | 测试 LLM 连接 |
| `markit provider list` | 列出已配置的凭证 |
| `markit provider fetch` | 获取提供商可用模型列表 |
| `markit model add` | 添加模型到配置 |
| `markit model list` | 列出已配置的模型 |

## 主要选项

| 选项 | 说明 |
|------|------|
| `-o, --output` | 输出目录 |
| `--llm` | 启用 LLM Markdown 增强 |
| `--analyze-image` | 用 LLM 生成图片 alt 文本 |
| `--analyze-image-with-md` | 同时生成 `.md` 描述文件 |
| `--no-compress` | 禁用图片压缩 |
| `--pdf-engine` | PDF 引擎：pymupdf4llm（默认）、pymupdf、pdfplumber |
| `--llm-provider` | 覆盖提供商：openai、anthropic、gemini、ollama、openrouter |
| `--llm-model` | 覆盖模型名称 |
| `-r, --recursive` | 递归处理子目录（batch） |
| `--resume` | 恢复中断的批处理 |
| `--fast` | 极速模式（跳过验证、最少重试） |
| `--dry-run` | 预览执行计划 |
| `-v, --verbose` | 详细日志 |

## 支持格式

| 格式 | 扩展名 | 引擎 |
|------|--------|------|
| Word | .docx, .doc | MarkItDown（.doc 需 LibreOffice） |
| PowerPoint | .pptx, .ppt | MarkItDown（.ppt 需 LibreOffice） |
| Excel | .xlsx, .xls | MarkItDown（.xls 需 LibreOffice） |
| PDF | .pdf | PyMuPDF4LLM / PyMuPDF / pdfplumber |
| HTML | .html, .htm | MarkItDown |
| 文本 | .txt | MarkItDown |
| 图片 | .png, .jpg, .gif, .webp, .bmp | LLM 分析 |

## 配置

直接运行 `markit config init` 创建 `markit.yaml`

```yaml
log_level: "INFO"
state_file: ".markit-state.json"

image:
  enable_compression: true
  png_optimization_level: 2  # 0-6，越高越慢
  jpeg_quality: 85
  max_dimension: 2048
  filter_small_images: true

concurrency:
  file_workers: 4      # 并发文件转换数
  image_workers: 8     # 并发图像处理数
  llm_workers: 5       # 并发 LLM 请求数

pdf:
  engine: "pymupdf4llm"  # pymupdf4llm, pymupdf, pdfplumber

enhancement:
  enabled: false
  remove_headers_footers: true
  fix_heading_levels: true
  add_frontmatter: true
  generate_summary: true

output:
  default_dir: "output"
  on_conflict: "rename"  # skip, overwrite, rename

prompt:
  output_language: "zh"  # zh, en, auto

# LLM 配置
# 将凭证和模型分开定义，更加灵活
llm:
  concurrent_fallback_enabled: true
  concurrent_fallback_timeout: 60  # 触发备用模型的超时秒数
  max_request_timeout: 300

  credentials:
    - id: "openai-main"
      provider: "openai"
      # api_key: "sk-..."  # 或使用 OPENAI_API_KEY 环境变量
    - id: "deepseek"
      provider: "openai"
      base_url: "https://api.deepseek.com"
      api_key_env: "DEEPSEEK_API_KEY"

  models:
    - name: "GPT-4o"
      model: "gpt-4o"
      credential_id: "openai-main"
      capabilities: ["text", "vision"]
      cost:  # 可选：用于成本统计
        input_per_1m: 2.50
        output_per_1m: 10.00
    - name: "deepseek-chat"
      model: "deepseek-chat"
      credential_id: "deepseek"
      capabilities: ["text"]  # 纯文本模型
```

**环境变量：**

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="..."
export GOOGLE_API_KEY="..."
export OPENROUTER_API_KEY="..."
export DEEPSEEK_API_KEY="..."
```

## 输出结构

```
output/
  document.docx.md              # 转换后的 markdown
  assets/
    document.docx.001.png       # 提取的图片
    document.docx.001.png.md    # 图片描述（使用 --analyze-image-with-md 时）
```

## 架构

MarkIt 采用模块化、面向服务的架构：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ConversionPipeline                                │
│                                                                             │
│  ┌─────────────────┐  ┌──────────────────┐  ┌────────────────────────────┐  │
│  │ FormatRouter    │  │ ImageProcessing  │  │ LLMOrchestrator            │  │
│  │                 │  │ Service          │  │                            │  │
│  │ - 路由文件       │  │ - 压缩            │  │ - ProviderManager          │  │
│  │ - 选择转换器     │  │ - 去重            │  │ - MarkdownEnhancer         │  │
│  │                 │  │ - 格式转换        │  │ - ImageAnalyzer            │  │
│  └─────────────────┘  └──────────────────┘  └────────────────────────────┘  │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         OutputManager                               │    │
│  │                                                                     │    │
│  │  - 冲突处理 (rename/overwrite/skip)                                  │    │
│  │  - 写入 markdown + 资源                                              │    │
│  │  - 生成图片描述 .md 文件                                              │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

**核心设计：**
- **基于能力的模型路由**：文本任务使用文本模型，视觉任务使用视觉模型
- **并发 fallback**：主模型超时后并发启动备用模型
- **LibreOffice 配置池**：隔离配置实现 .doc/.ppt/.xls 并行转换
- **进程池图像处理**：大量图像压缩使用进程池绕过 GIL

## 许可证

MIT
