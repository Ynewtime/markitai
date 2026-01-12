# MarkIt - 产品需求文档

**版本**: 1.0.2 | **状态**: Active

## 概述

MarkIt 是一个命令行工具，用于批量将办公文档转换为 Markdown，支持 LLM 增强。

**目标用户**：技术文档工程师、知识库管理员、内容迁移工程师

**核心价值**：
- 多格式支持（Office、PDF、HTML、图片）
- LLM 驱动的格式清洗和图片分析
- 并发处理，支持断点续传

## 架构

```
┌─────────────────────────────────────────────────────────────┐
│                    CLI 层 (Typer)                           │
├─────────────────────────────────────────────────────────────┤
│               配置管理 (pydantic-settings)                   │
├─────────────────────────────────────────────────────────────┤
│                      处理管道                                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐     │
│  │ 转换引擎  │→ │ 图片处理  │→ │ LLM 增强  │→ │ Markdown │     │
│  │          │  │          │  │          │  │   输出   │     │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘     │
├──────────────────┬──────────────────────────────────────────┤
│ 转换器            │ LLM 提供商                                │
│ • MarkItDown     │ • OpenAI      • Anthropic                │
│ • PyMuPDF4LLM    │ • Gemini      • Ollama                   │
│ • pdfplumber     │ • OpenRouter                             │
│ • LibreOffice    │                                          │
└──────────────────┴──────────────────────────────────────────┘
```

## 项目结构

```
markit/
├── cli/
│   ├── main.py              # Typer 应用
│   └── commands/
│       ├── convert.py       # 单文件转换
│       ├── batch.py         # 批量转换
│       ├── config.py        # 配置管理
│       └── provider.py      # LLM 提供商命令
├── config/
│   ├── settings.py          # pydantic-settings 配置
│   └── constants.py         # 常量定义
├── core/
│   ├── pipeline.py          # 主处理管道
│   ├── router.py            # 格式路由
│   └── state.py             # 批处理状态（断点续传）
├── converters/
│   ├── base.py              # 转换器基类
│   ├── markitdown.py        # MarkItDown 封装
│   ├── office.py            # LibreOffice 转换
│   └── pdf/
│       ├── pymupdf4llm.py   # 默认 PDF 引擎
│       ├── pymupdf.py
│       └── pdfplumber.py
├── image/
│   ├── extractor.py         # 图片提取
│   ├── compressor.py        # PNG/JPEG 压缩
│   └── analyzer.py          # LLM 图片分析
├── llm/
│   ├── base.py              # 提供商基类
│   ├── manager.py           # 提供商管理 + 故障转移
│   ├── enhancer.py          # Markdown 增强
│   ├── openai.py
│   ├── anthropic.py
│   ├── gemini.py
│   ├── ollama.py
│   └── openrouter.py
├── markdown/
│   ├── formatter.py
│   ├── frontmatter.py
│   └── chunker.py           # 大文件切块
└── utils/
    ├── logging.py           # structlog 配置
    ├── concurrency.py
    └── fs.py
```

## 支持格式

| 格式 | 扩展名 | 主引擎 | 回退 |
|------|--------|--------|------|
| Word | .docx | MarkItDown | - |
| Word（旧版） | .doc | LibreOffice → MarkItDown | - |
| PowerPoint | .pptx | MarkItDown | - |
| PowerPoint（旧版） | .ppt | LibreOffice → MarkItDown | - |
| Excel | .xlsx | MarkItDown | - |
| Excel（旧版） | .xls | LibreOffice → MarkItDown | - |
| PDF | .pdf | PyMuPDF4LLM | PyMuPDF, pdfplumber |
| HTML | .html, .htm | MarkItDown | - |
| CSV | .csv | MarkItDown | - |
| 图片 | .png, .jpg, .gif, .webp, .bmp | LLM 分析 | - |

## CLI 命令

### convert

```bash
markit convert [选项] INPUT_FILE
```

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `-o, --output` | 输出目录 | `./output` |
| `--llm` | 启用 LLM 增强 | 禁用 |
| `--analyze-image` | 生成 alt 文本 | 禁用 |
| `--analyze-image-with-md` | 生成 .md 描述文件 | 禁用 |
| `--no-compress` | 禁用图片压缩 | 启用压缩 |
| `--pdf-engine` | pymupdf4llm, pymupdf, pdfplumber, markitdown | pymupdf4llm |
| `--llm-provider` | openai, anthropic, gemini, ollama, openrouter | 第一个可用 |
| `--llm-model` | 模型名称 | 提供商默认 |
| `-v, --verbose` | 详细输出 | 禁用 |
| `--dry-run` | 仅显示计划 | 禁用 |

### batch

```bash
markit batch [选项] INPUT_DIR
```

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `-r, --recursive` | 递归处理子目录 | 禁用 |
| `--include` | 包含模式（glob） | 所有支持格式 |
| `--exclude` | 排除模式（glob） | 无 |
| `--file-concurrency` | 文件并发数 | 4 |
| `--image-concurrency` | 图片并发数 | 8 |
| `--llm-concurrency` | LLM 请求并发数 | 5 |
| `--on-conflict` | skip, overwrite, rename | rename |
| `--resume` | 恢复中断的批处理 | 禁用 |
| `--state-file` | 状态文件路径 | `.markit-state.json` |

### provider

```bash
markit provider add       # 添加 LLM 提供商凭证
markit provider test      # 测试 LLM 连接
markit provider list      # 列出已配置的凭证
markit provider fetch     # 获取提供商可用模型列表
```

### model

```bash
markit model add          # 交互式添加模型到配置
markit model list         # 列出已配置的模型
```

## 配置

**优先级**：命令行参数 > 环境变量 > markit.yaml > 默认值

### markit.yaml

```yaml
log_level: "INFO"
log_dir: ".logs"

output:
  default_dir: "output"
  on_conflict: "rename"

image:
  enable_compression: true
  filter_small_images: true

concurrency:
  file_workers: 4
  image_workers: 8
  llm_workers: 5

pdf:
  engine: "pymupdf4llm"

# 新配置结构（推荐用于多模型场景）
# 将凭证和模型分开定义
llm:
  credentials:
    - id: "openai-main"
      provider: "openai"
      # api_key: "sk-..."  # 或使用 OPENAI_API_KEY 环境变量

  models:
    - name: "GPT-4o"
      model: "gpt-4o"
      credential_id: "openai-main"
      capabilities: ["text", "vision"]
      timeout: 120
```

### 环境变量

```bash
# API 密钥（无 MARKIT_ 前缀）
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=...
GOOGLE_API_KEY=...
OPENROUTER_API_KEY=...

# 设置项（MARKIT_ 前缀）
MARKIT_LOG_LEVEL=DEBUG
MARKIT_LOG_FILE=logs/markit.log
MARKIT_CONCURRENCY__FILE_WORKERS=8
```

## LLM 功能

### Markdown 增强（`--llm`）

- 插入 YAML frontmatter（标题、来源、处理时间）
- 移除页眉页脚和垃圾内容
- 修复标题层级（从 h2 开始）
- 规范化空行
- 遵循 GFM 规范

### 图片分析（`--analyze-image`）

- 为每张提取的图片生成 alt 文本
- 每张图片单次 LLM 调用

### 图片描述文件（`--analyze-image-with-md`）

在每张图片旁生成 `<image>.md`：

```markdown
---
source_image: diagram_001.png
image_type: diagram
generated_at: 2026-01-08T12:00:00Z
---

## Alt Text
展示数据处理流程的流程图。

## Detailed Description
该图展示了一个三阶段的处理流程...

## Detected Text
"输入" -> "处理" -> "输出"
```

## 技术栈

| 组件 | 技术 |
|------|------|
| CLI | Typer |
| 配置 | pydantic-settings |
| 转换 | MarkItDown, PyMuPDF4LLM, LibreOffice |
| 图片 | Pillow, oxipng |
| 异步 | anyio, httpx |
| LLM SDK | openai, anthropic, google-genai, ollama |
| 文本切块 | langchain-text-splitters, tiktoken |
| 日志 | structlog |

## 依赖

- Python 3.12+
- LibreOffice（处理 .doc, .ppt, .xls）
- oxipng（可选，PNG 压缩）
