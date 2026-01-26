# Markitai 技术规格文档

> 版本: 0.3.0
> 最后更新: 2026-01-26

---

## 1. 概述

### 1.1 项目定位

Markitai 是一个开箱即用的 Markdown 转换器，原生支持 LLM 增强。核心设计理念：

- **程序转换 + LLM 优化**：基础转换只做格式转换，数据清洗、格式优化、排版等交由大模型处理
- **不造轮子**：基于社区优秀依赖实现，避免重复造轮子
- **测试驱动**：所有特性都需要测试覆盖

### 1.2 设计原则

| 原则 | 说明 |
|------|------|
| 最小依赖 | 基础转换不引入或仅引入最少的数据清洗规则 |
| 可选 LLM | LLM 功能为可选增强，不强制依赖 |
| 配置优先 | 所有行为可通过配置自定义 |
| 优雅降级 | LLM 失败时保留基础转换结果 |

### 1.3 技术栈

| 组件 | 库 | 版本 | 用途 |
|------|------|------|------|
| 包管理 | uv | >=0.9.25 | Monorepo workspace 管理 |
| PDF 转换 | pymupdf4llm | >=0.2.9 | PDF → Markdown + 图片提取 |
| Office 转换 | markitdown[all] | >=0.1.4 | Word/PPT/Excel 基础转换 |
| LLM 网关 | litellm | >=1.80.16 | 统一 LLM 调用、成本追踪、负载均衡 |
| LLM 结构化输出 | instructor | >=1.14.0 | LLM 结构化输出解析 |
| OCR | rapidocr | >=3.5.0 | 扫描版 PDF 和图片文字识别 |
| CLI | click | >=8.1.0 | 命令行接口 |
| 日志 | loguru | >=0.7.3 | 日志记录 |
| 进度条 | rich | >=14.2.0 | 终端进度显示 |
| 图片处理 | Pillow | >=12.1.0 | 图片压缩和格式转换 |
| 异步 | asyncio, aiofiles | stdlib | 异步 IO 和并发处理 |

**Python 版本要求**: >=3.11（支持 3.11 及以上版本）

### 1.4 项目结构（Monorepo）

采用 uv workspace 管理 monorepo，便于后续 fork 依赖源码进行修改。

```
markitai/
├── packages/
│   ├── markitai/                 # 主包（CLI + 核心逻辑）
│   │   ├── src/markitai/
│   │   ├── tests/
│   │   └── pyproject.toml
│   └── <forked-dep>/           # Fork 的依赖（按需添加）
│       ├── src/
│       └── pyproject.toml
├── pyproject.toml              # Workspace 根配置
├── uv.lock                     # 锁文件
└── README.md
```

**Workspace 根配置** (`pyproject.toml`):

```toml
[project]
name = "markitai-workspace"
version = "0.1.0"
requires-python = "==3.13.*"

[tool.uv.workspace]
members = ["packages/*"]

[dependency-groups]
dev = ["pytest-cov>=7.0.0"]
```

> **注意**：Workspace 使用 `==3.13.*` 锁定开发环境，而主包 `>=3.11` 是发布版本的最低要求。

**主包配置** (`packages/markitai/pyproject.toml`):

```toml
[project]
name = "markitai"
version = "0.3.0"
requires-python = ">=3.11"
dependencies = [
    "pymupdf4llm>=0.2.9",
    "markitdown[all]>=0.1.4",
    "litellm>=1.80.16",
    "instructor>=1.14.0",
    "rapidocr>=3.5.0",
    "click>=8.1.0",
    "loguru>=0.7.3",
    "rich>=14.2.0",
    "Pillow>=12.1.0",
    "aiofiles>=25.1.0",
    "pydantic>=2.10.0",
    "python-dotenv>=1.2.1",
]

[project.scripts]
markitai = "markitai.cli:app"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/markitai"]
```

**Fork 依赖说明**：

当需要 fork 某个依赖（如 markitdown）时：
1. 将源码克隆到 `packages/<fork-name>/`
2. 修改其 `pyproject.toml` 中的包名（避免与上游冲突）
3. 在主包中将依赖改为 workspace 引用
4. 具体命名规则在 fork 时再定

**图片提取策略**：
- PDF：pymupdf4llm 原生支持 `write_images=True`，直接写入磁盘
- Office：markitdown `keep_data_uris=True` + 后处理提取 base64 图片

---

## 2. 系统架构

### 2.1 整体架构图

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              CLI Layer                                       │
│                     (Click + Rich Progress/Live)                             │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
              ┌───────────────────────┴────────────────────────┐
              ▼                                                ▼
┌──────────────────────────────┐               ┌────────────────────────────────┐
│      Local File Input        │               │        URL Input (.urls)       │
│  (PDF/DOCX/PPTX/XLSX/Image)  │               │  (Static/Browser/Jina Fetch)   │
└───────────────┬──────────────┘               └───────────────┬────────────────┘
                │                                              │
                ▼                                              ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                           Workflow Layer                                     │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │  convert_document_core (ConversionContext)                             │  │
│  │    - validate_and_detect_format                                        │  │
│  │    - convert_document → process_embedded_images → write_base_markdown  │  │
│  │    - process_with_vision_llm / process_with_standard_llm               │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │  SingleFileWorkflow                                                    │  │
│  │    - process_document_with_llm (clean + frontmatter)                   │  │
│  │    - analyze_images (alt text + description)                           │  │
│  │    - enhance_with_vision (screenshot mode)                             │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │  BatchProcessor                                                        │  │
│  │    - discover_files / process_batch / resume capability                │  │
│  │    - save_state / save_report / print_summary                          │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         Core Processor Layer                                 │
│  ┌────────────────┐  ┌──────────────────┐  ┌────────────────────────────┐    │
│  │   Converter    │  │   LLMProcessor   │  │    ImageProcessor          │    │
│  │  - PDF         │  │  - LLMRuntime    │  │  - extract_base64_images   │    │
│  │  - Office      │  │  - process_doc   │  │  - compress / filter       │    │
│  │  - Image       │  │  - analyze_image │  │  - download_url_images     │    │
│  │  - Text        │  │  - enhance_doc   │  │  - process_and_save        │    │
│  │  - Legacy      │  │  - LLMCache      │  └────────────────────────────┘    │
│  └────────────────┘  └──────────────────┘                                    │
│  ┌────────────────┐  ┌──────────────────┐  ┌────────────────────────────┐    │
│  │  FetchModule   │  │   PromptManager  │  │      OCR Engine            │    │
│  │  - static      │  │  - cleaner       │  │  - RapidOCR                │    │
│  │  - browser     │  │  - frontmatter   │  │  - Vision LLM (optional)   │    │
│  │  - jina        │  │  - image_*       │  └────────────────────────────┘    │
│  │  - FetchCache  │  │  - document_*    │                                    │
│  └────────────────┘  └──────────────────┘                                    │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                            Utils & Infra Layer                               │
│  ┌────────────────┐  ┌──────────────────┐  ┌────────────────────────────┐    │
│  │    executor    │  │     security     │  │        paths/output        │    │
│  │ (ThreadPool)   │  │ (atomic_write,   │  │  (ensure_dir, resolve)     │    │
│  └────────────────┘  │  path_validate)  │  └────────────────────────────┘    │
│  ┌────────────────┐  └──────────────────┘  ┌────────────────────────────┐    │
│  │   json_order   │  ┌──────────────────┐  │    text/mime helpers       │    │
│  │ (report/state) │  │   urls parser    │  │  (normalize, detect)       │    │
│  └────────────────┘  │  (.urls file)    │  └────────────────────────────┘    │
│                      └──────────────────┘                                    │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                          External Dependencies                               │
│  ┌────────────────┐  ┌──────────────────┐  ┌────────────────────────────┐    │
│  │  MarkItDown    │  │  LiteLLM Router  │  │        Pillow              │    │
│  │  pymupdf4llm   │  │  (multi-model)   │  │   (image processing)       │    │
│  │  (conversion)  │  │  instructor      │  └────────────────────────────┘    │
│  └────────────────┘  └──────────────────┘  ┌────────────────────────────┐    │
│  ┌────────────────┐  ┌──────────────────┐  │     agent-browser          │    │
│  │   RapidOCR     │  │   Rich/Loguru    │  │  (browser automation)      │    │
│  │  (OCR engine)  │  │  (UI/logging)    │  └────────────────────────────┘    │
│  └────────────────┘  └──────────────────┘                                    │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 核心模块

| 模块 | 职责 |
|------|------|
| `ConfigManager` | 配置加载与合并 (config.py) |
| `Converter` | 文档基础转换 (converter/) - PDF/Office/Image/Text |
| `LLMProcessor` | LLM 调用与处理 (llm.py) |
| `LLMRuntime` | 全局 LLM 并发控制（共享 Semaphore） |
| `LLMCache` | 持久化 LLM 缓存（双层查找：精确 hash + 语义相似） |
| `ImageProcessor` | 图片提取、压缩、分析 (image.py) |
| `PromptManager` | 提示词管理 (prompts/) |
| `ConversionContext` | 单文件转换上下文 (workflow/core.py) |
| `SingleFileWorkflow` | 单文件 LLM 处理流程封装 (workflow/single.py) |
| `BatchProcessor` | 批量处理与断点恢复 (batch.py) |
| `FetchModule` | URL 抓取模块 - static/browser/jina 三策略 (fetch.py) |
| `FetchCache` | URL 抓取结果缓存（SQLite） |
| `UrlParser` | .urls 文件解析（JSON/纯文本格式）(urls.py) |
| `Logger` | 日志系统 (Loguru) |

### 2.3 数据流

**本地文件转换流程**:
```
输入文件 → 格式检测 → 基础转换 → 图片提取 → 图片压缩 → 输出 Markdown
```

**URL 抓取流程**:
```
.urls 文件 → 解析 URL 列表 → 选择抓取策略 (static/browser/jina)
     ↓
  static: MarkItDown 直接抓取
  browser: agent-browser 渲染抓取 + 可选截图
  jina: Jina Reader API 抓取
     ↓
抓取结果 → FetchCache 缓存 → 转换为 Markdown → 后续 LLM 处理
```

**LLM 增强流程**:
```
基础 Markdown → LLM 清洗优化 → 生成 frontmatter → 输出 .llm.md
```

**Vision 增强流程** (PDF 截图模式):
```
PDF → 页面截图 → Vision LLM 结合原始文本增强 → 输出 .llm.md
```

**图片分析流程**:
```
提取的图片 → 压缩（保存到 assets 目录） → Vision LLM/OCR 分析 → 生成 alt 文本/描述 → 输出 images.json
```

---

## 3. 接口设计

### 3.1 CLI 命令

```bash
# 主命令
markitai [OPTIONS] INPUT

# 子命令
markitai config [SUBCOMMAND]   # 配置管理
```

#### markitai config 子命令

```bash
# 显示当前生效的配置（合并后）
markitai config list

# 显示配置文件路径
markitai config path

# 初始化配置文件（交互式）
markitai config init

# 初始化配置文件到指定路径
markitai config init --output ~/.markitai/config.json

# 验证配置文件
markitai config validate [CONFIG_PATH]

# 设置单个配置项（写入用户配置文件）
markitai config set llm.enabled true
markitai config set llm.concurrency 20

# 获取单个配置项
markitai config get llm.enabled
```

| 子命令 | 说明 |
|--------|------|
| `list` | 显示当前生效的完整配置（合并命令行、环境变量、配置文件后） |
| `path` | 显示配置文件查找路径和当前使用的配置文件 |
| `init` | 交互式生成配置文件，支持 `--output` 指定路径 |
| `validate` | 验证配置文件格式，可选传入路径，默认验证当前生效的配置 |
| `set` | 设置配置项，使用点号分隔的路径（如 `llm.enabled`） |
| `get` | 获取配置项值 |

### 3.2 命令行参数

| 参数 | 短参数 | 类型 | 默认值 | 说明 |
|------|--------|------|--------|------|
| `INPUT` | - | PATH/URL | 必填 | 输入文件、目录或 URL |
| `--output` | `-o` | PATH | `./output` | 输出目录 |
| `--preset` | `-p` | CHOICE | - | 使用预设配置（rich/standard/minimal） |
| `--llm/--no-llm` | - | FLAG | False | 启用/禁用 LLM 处理 |
| `--alt/--no-alt` | - | FLAG | False | 启用/禁用图片 alt 文本生成 |
| `--desc/--no-desc` | - | FLAG | False | 启用/禁用图片描述文件生成 |
| `--screenshot/--no-screenshot` | - | FLAG | False | 启用/禁用 PDF/PPTX/URL 页面截图 |
| `--batch-concurrency` | `-j` | INT | 10 | 批处理文件并发数 |
| `--llm-concurrency` | - | INT | 10 | LLM 请求并发数 |
| `--url-concurrency` | - | INT | 5 | URL 抓取并发数 |
| `--ocr/--no-ocr` | - | FLAG | False | 启用/禁用 OCR（用于扫描版 PDF） |
| `--resume` | - | FLAG | False | 恢复中断的批处理 |
| `--config` | `-c` | PATH | - | 指定配置文件路径 |
| `--verbose` | - | FLAG | False | 详细日志输出 |
| `--version` | `-v` | FLAG | - | 打印版本信息 |
| `--dry-run` | - | FLAG | False | 预览执行计划 |
| `--no-compress` | - | FLAG | False | 禁用图片压缩 |
| `--no-cache` | - | FLAG | False | 全局禁用 LLM 缓存（强制重新调用 API） |
| `--no-cache-for` | - | TEXT | - | 指定文件/模式禁用缓存（逗号分隔，支持 glob） |
| `--agent-browser` | - | FLAG | False | 强制使用 agent-browser 渲染抓取 URL |
| `--jina` | - | FLAG | False | 强制使用 Jina Reader API 抓取 URL |

**--preset 预设说明**：

| 预设 | 等效参数 | 适用场景 |
|------|----------|----------|
| `rich` | `--llm --alt --desc --screenshot` | 复杂文档，需要完整分析 |
| `standard` | `--llm --alt --desc` | 普通文档，不需要截图 |
| `minimal` | 无增强 | 仅基础转换 |

预设可与单独参数组合使用，单独参数会覆盖预设：
```bash
markitai doc.pdf --preset rich --no-desc  # rich 但不生成描述文件
```

**--ocr 与 --llm 组合说明**：
- `--ocr` 单独使用：使用 RapidOCR 提取扫描版 PDF/图片中的文字
- `--ocr --llm` 组合使用：使用 LLM Vision 直接分析扫描页面（更智能，但成本更高）

### 3.3 环境变量

| 变量名 | 说明 | 示例 |
|--------|------|------|
| `MARKITAI_CONFIG` | 配置文件路径 | `~/.markitai/config.json` |
| `MARKITAI_LOG_DIR` | 日志目录 | `~/.markitai/logs` |
| `MARKITAI_PROMPT_DIR` | 提示词目录 | `~/.markitai/prompts` |
| `OPENAI_API_KEY` | OpenAI API Key | `YOUR_OPENAI_KEY` |
| `ANTHROPIC_API_KEY` | Anthropic API Key | `YOUR_ANTHROPIC_KEY` |
| `GEMINI_API_KEY` | Google Gemini API Key | `YOUR_GEMINI_KEY` |
| `DEEPSEEK_API_KEY` | DeepSeek API Key | `YOUR_DEEPSEEK_KEY` |
| `LITELLM_PROXY_URL` | LiteLLM Proxy 地址 | `http://localhost:4000` |

### 3.4 退出码

| 退出码 | 含义 |
|--------|------|
| 0 | 成功 |
| 1 | 一般错误 |
| 2 | 配置错误 |
| 3 | 输入文件不存在 |
| 4 | 输出目录无法创建 |
| 5 | LLM 调用失败（已降级处理） |
| 10 | 部分文件处理失败 |

---

## 4. 配置管理

### 4.1 配置文件格式

配置文件采用 JSON 格式，文件名为 `markitai.json`。

### 4.2 配置文件查找顺序

优先级从高到低：

1. 命令行参数 `--config` 指定的路径
2. 环境变量 `MARKITAI_CONFIG` 指定的路径
3. 当前工作目录 `./markitai.json`
4. 用户目录 `~/.markitai/config.json`
5. 系统默认值

### 4.3 配置项详解

```json
{
  "output": {
    "dir": "./output",
    "on_conflict": "rename"
  },
  "llm": {
    "enabled": false,
    "model_list": [
      {
        "model_name": "default",
        "litellm_params": {
          "model": "deepseek/deepseek-chat",
          "api_key": "env:DEEPSEEK_API_KEY",
          "weight": 7
        }
      },
      {
        "model_name": "default",
        "litellm_params": {
          "model": "gemini/gemini-2.5-flash",
          "api_key": "env:GEMINI_API_KEY",
          "weight": 3
        }
      },
      {
        "model_name": "vision",
        "litellm_params": {
          "model": "gemini/gemini-2.5-flash",
          "api_key": "env:GEMINI_API_KEY",
          "weight": 7
        }
      },
      {
        "model_name": "vision",
        "litellm_params": {
          "model": "openai/gpt-4o",
          "api_key": "env:OPENAI_API_KEY",
          "weight": 3
        }
      }
    ],
    "router_settings": {
      "routing_strategy": "simple-shuffle",
      "num_retries": 2,
      "timeout": 120,
      "fallbacks": []
    },
    "concurrency": 10
  },
  "image": {
    "alt_enabled": false,
    "desc_enabled": false,
    "compress": true,
    "quality": 85,
    "format": "jpeg",
    "max_width": 1920,
    "max_height": 1080,
    "filter": {
      "min_width": 50,
      "min_height": 50,
      "min_area": 5000,
      "deduplicate": true
    }
  },
  "ocr": {
    "enabled": false,
    "lang": "zh"
  },
  "screenshot": {
    "enabled": false
  },
  "prompts": {
    "dir": "~/.markitai/prompts",
    "cleaner": null,
    "frontmatter": null,
    "image_caption": null,
    "image_description": null,
    "image_analysis": null,
    "page_content": null,
    "document_enhance": null
  },
  "batch": {
    "concurrency": 10,
    "state_flush_interval_seconds": 10,
    "scan_max_depth": 5,
    "scan_max_files": 10000
  },
  "log": {
    "level": "DEBUG",
    "dir": "~/.markitai/logs",
    "rotation": "10 MB",
    "retention": "7 days"
  }
}
```

**注意**：配置文件需维护 JSON Schema (`config.schema.json`) 用于 IDE 自动补全和验证。

### 4.4 配置项说明

#### output 配置

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `dir` | string | `./output` | 输出目录 |
| `on_conflict` | string | `rename` | 冲突处理策略：`skip`（跳过）、`overwrite`（覆盖）、`rename`（重命名） |
| `allow_symlinks` | bool | false | 是否允许输出目录为软链接 |

**on_conflict 策略说明**：
- `skip`：如果输出文件已存在，跳过处理
- `overwrite`：直接覆盖已存在的文件
- `rename`：自动添加版本号，如 `file.pdf.md` → `file.pdf.v2.md`，`file.pdf.llm.md` → `file.pdf.v2.llm.md`

#### llm 配置

采用 LiteLLM Router 兼容的配置格式，支持多模型资源池和负载均衡。

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | bool | false | 是否启用 LLM 处理 |
| `model_list` | array | [] | 模型资源池（兼容 LiteLLM Router 格式） |
| `router_settings` | object | {} | LiteLLM Router 设置 |
| `concurrency` | int | 10 | LLM 并发请求数 |

**model_list 数组元素格式**（兼容 LiteLLM）：

| 字段 | 类型 | 说明 |
|------|------|------|
| `model_name` | string | 逻辑模型名（同名实现负载均衡） |
| `litellm_params` | object | LiteLLM 参数（model, api_key, api_base, weight 等） |
| `model_info` | object | 模型元信息，可选（所有字段均自动从 litellm 检测） |

**litellm_params 常用字段**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `model` | string | 实际模型标识（如 `deepseek/deepseek-chat`） |
| `api_key` | string | API Key（支持 `env:VAR_NAME` 格式读取环境变量） |
| `api_base` | string | 自定义 API 地址 |
| `weight` | int | 流量权重（相对值，仅 simple-shuffle 策略生效） |

**model_info 字段**（均为可选，不设置时从 litellm 自动检测）：

| 字段 | 类型 | 说明 |
|------|------|------|
| `supports_vision` | bool \| null | 标记模型支持视觉输入 |
| `max_tokens` | int \| null | 模型最大输出 token 数 |
| `max_input_tokens` | int \| null | 上下文窗口大小 |

**逻辑模型名约定**：
- `default`：文本处理任务（Markdown 清洗、frontmatter 生成）
- `vision`：图片分析任务（自动从 litellm 检测视觉能力，或通过 `model_info.supports_vision: true` 显式指定）

**router_settings 配置**：

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `routing_strategy` | string | `simple-shuffle` | 路由策略（simple-shuffle/least-busy/usage-based-routing/latency-based-routing） |
| `num_retries` | int | 2 | 最大重试次数 |
| `timeout` | int | 120 | 请求超时时间（秒） |
| `fallbacks` | array | [] | 故障转移配置 |

#### image 配置

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `alt_enabled` | bool | false | 是否启用图片 alt 文本生成 |
| `desc_enabled` | bool | false | 是否启用图片描述输出 |
| `compress` | bool | true | 是否压缩图片 |
| `quality` | int | 85 | JPEG 压缩质量 (1-100) |
| `format` | string | `jpeg` | 输出格式 (jpeg/png/webp) |
| `max_width` | int | 1920 | 最大宽度（节省 LLM Token 成本） |
| `max_height` | int | 1080 | 最大高度（节省 LLM Token 成本） |
| `filter` | object | {} | 图片过滤配置 |

**filter 配置**（过滤小图片、图标和重复图片）：

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `min_width` | int | 50 | 最小宽度（像素），小于此值的图片被过滤 |
| `min_height` | int | 50 | 最小高度（像素），小于此值的图片被过滤 |
| `min_area` | int | 5000 | 最小面积（width×height），小于此值的图片被过滤 |
| `deduplicate` | bool | true | 是否去重（基于图片 hash） |

#### ocr 配置

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | bool | false | 是否启用 OCR |
| `lang` | string | `zh` | OCR 语言（zh/en，内部映射到 RapidOCR） |

#### screenshot 配置

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | bool | false | 是否启用 PDF/PPTX/URL 页面截图 |
| `viewport_width` | int | 1280 | URL 截图视口宽度（像素） |
| `viewport_height` | int | 720 | URL 截图视口高度（像素） |
| `quality` | int | 85 | JPEG 压缩质量 (1-100) |
| `max_height` | int | 10000 | URL 全页截图最大高度（像素） |

#### prompts 配置

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `dir` | string | `~/.markitai/prompts` | 提示词目录 |
| `cleaner` | string | null | 清洗提示词文件路径 |
| `frontmatter` | string | null | frontmatter 生成提示词 |
| `image_caption` | string | null | 图片 alt 文本提示词 |
| `image_description` | string | null | 图片描述提示词 |
| `image_analysis` | string | null | 图片分析提示词（caption + description 组合） |
| `page_content` | string | null | 页面内容提取提示词 |
| `document_enhance` | string | null | 文档增强提示词（OCR+LLM/PPTX+LLM 模式） |
| `url_enhance` | string | null | URL/网页内容增强提示词 |

#### batch 配置

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `concurrency` | int | 10 | 文件处理并发数 |
| `url_concurrency` | int | 5 | URL 抓取并发数（独立于文件处理） |
| `state_flush_interval_seconds` | int | 10 | 状态/报告文件写入节流间隔（秒） |
| `scan_max_depth` | int | 5 | 扫描目录最大深度（相对输入目录） |
| `scan_max_files` | int | 10000 | 扫描文件数上限 |

> 注：批处理状态文件为 `states/markitai.<hash>.state.json`，报告文件为 `reports/markitai.<hash>.report.json`，均支持断点恢复。

#### log 配置

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `level` | string | `DEBUG` | 日志级别 |
| `dir` | string | `~/.markitai/logs` | 日志目录 |
| `rotation` | string | `10 MB` | 日志轮转大小 |
| `retention` | string | `7 days` | 日志保留时间 |

#### cache 配置

LLM 调用结果缓存配置，用于避免重复调用 API。

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | bool | true | 是否启用缓存 |
| `no_cache` | bool | false | 跳过读取缓存但仍写入 |
| `no_cache_patterns` | array | [] | 跳过缓存的文件模式（glob，相对于输入目录） |
| `max_size_bytes` | int | 1073741824 | 缓存大小上限（默认 1GB） |
| `global_dir` | string | `~/.markitai/cache` | 全局缓存目录 |

**缓存行为说明**：

- **`--no-cache`**：全局禁用缓存读取，强制重新调用 LLM API，但结果仍会写入缓存
- **`--no-cache-for`**：细粒度控制，仅对匹配的文件禁用缓存

**`--no-cache-for` 模式示例**：

```bash
# 单个文件
markitai docs/ --llm --no-cache-for file.pdf

# Glob 模式
markitai docs/ --llm --no-cache-for "*.pdf"

# 递归模式
markitai docs/ --llm --no-cache-for "**/reports/*.pdf"

# 多模式组合（逗号分隔）
markitai docs/ --llm --no-cache-for "*.pdf,reports/**,specific.docx"
```

**模式匹配规则**：
- 使用增强的 `fnmatch` 语法（跨平台一致）
- 模式相对于输入目录匹配
- `*` 匹配单层目录内的任意字符
- `**` 匹配零个或多个目录层级
- Windows 路径分隔符 `\` 自动转换为 `/`

**`**` 匹配示例**：

| 模式 | 匹配 | 不匹配 |
|------|------|--------|
| `**/*.pdf` | `file.pdf`, `a/file.pdf`, `a/b/file.pdf` | `file.docx` |
| `src/**/test.py` | `src/test.py`, `src/unit/test.py` | `lib/test.py` |
| `**/reports/*.pdf` | `reports/a.pdf`, `foo/reports/a.pdf` | `other/a.pdf` |

**缓存 Hash 计算**：

缓存键基于以下内容计算 SHA256：
- Prompt 内容
- 文档内容长度
- 文档头部 25000 字符
- 文档尾部 25000 字符

这确保文档任何部分的修改都会使缓存失效。

#### fetch 配置

URL 抓取配置，用于处理静态页面和 JS 渲染页面。

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `strategy` | string | `auto` | 抓取策略（auto/static/browser/jina） |
| `agent_browser` | object | {} | agent-browser 配置 |
| `jina` | object | {} | Jina Reader API 配置 |
| `fallback_patterns` | array | `["x.com", "twitter.com"]` | 需要浏览器渲染的域名模式 |

**strategy 策略说明**：

| 策略 | 说明 | 适用场景 |
|------|------|----------|
| `auto` | 自动检测，优先静态抓取，JS 页面回退到浏览器/Jina | 默认，兼顾速度和兼容性 |
| `static` | MarkItDown 直接 HTTP 抓取 | 静态页面，最快 |
| `browser` | agent-browser 无头浏览器渲染 | JS 重度页面（SPA、Twitter 等） |
| `jina` | Jina Reader API（云端渲染） | 无本地依赖，有 API 限制 |

**agent_browser 配置**：

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `command` | string | `agent-browser` | agent-browser 命令路径 |
| `timeout` | int | 30000 | 页面加载超时（毫秒） |
| `wait_for` | string | `domcontentloaded` | 等待状态（load/domcontentloaded/networkidle） |
| `extra_wait_ms` | int | 2000 | 加载后额外等待时间（毫秒，用于 SPA 渲染） |
| `session` | string | null | 隔离的浏览器会话名称 |

**jina 配置**：

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `api_key` | string | null | Jina API Key（支持 `env:JINA_API_KEY` 语法） |
| `timeout` | int | 30 | 请求超时（秒） |

**CLI 使用示例**：

```bash
# 自动检测策略（默认）
markitai https://example.com --llm

# 强制使用浏览器渲染
markitai https://x.com/user/status/123 --agent-browser --llm

# 强制使用 Jina API
markitai https://example.com --jina --llm

# URL 带截图
markitai https://example.com --agent-browser --screenshot --llm
```

#### presets 配置

预设系统允许用户通过单个参数启用一组功能，简化常用场景的命令行使用。

**内置预设**：

| 预设名称 | llm | alt | desc | screenshot | 适用场景 |
|----------|-----|-----|------|------------|----------|
| `rich` | true | true | true | true | 复杂文档，需要完整 LLM 分析和页面截图 |
| `standard` | true | true | true | false | 普通文档，LLM 分析但不截图 |
| `minimal` | false | false | false | false | 仅基础格式转换 |

**CLI 使用**：

```bash
markitai document.pdf --preset rich          # 使用 rich 预设
markitai document.pdf --preset rich --no-desc  # 覆盖预设的 desc 设置
```

**自定义预设配置**：

用户可在配置文件中定义自定义预设：

```json
{
  "presets": {
    "custom": {
      "llm": true,
      "alt": true,
      "desc": false,
      "screenshot": false,
      "ocr": false
    }
  }
}
```

**PresetConfig 字段**：

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `llm` | bool | false | 启用 LLM 处理 |
| `ocr` | bool | false | 启用 OCR |
| `alt` | bool | false | 启用图片 alt 文本生成 |
| `desc` | bool | false | 启用图片描述文件生成 |
| `screenshot` | bool | false | 启用页面截图 |

**优先级**：CLI 参数 > 预设 > 配置文件默认值

---

## 5. 输入输出

### 5.1 支持的输入格式

| 格式 | 扩展名 | 处理方式 |
|------|--------|----------|
| Word (新) | `.docx` | MarkItDown |
| Word (旧) | `.doc` | 先转换为 DOCX，再用 MarkItDown |
| PowerPoint (新) | `.pptx` | MarkItDown |
| PowerPoint (旧) | `.ppt` | 先转换为 PPTX，再用 MarkItDown |
| Excel (新) | `.xlsx` | MarkItDown |
| Excel (旧) | `.xls` | 先转换为 XLSX，再用 MarkItDown |
| PDF | `.pdf` | pymupdf4llm + RapidOCR (可选) |
| 文本 | `.txt` | 直接读取 |
| Markdown | `.md` | 直接读取 |
| JPEG | `.jpg`, `.jpeg` | RapidOCR / LLM Vision |
| PNG | `.png` | RapidOCR / LLM Vision |
| WebP | `.webp` | RapidOCR / LLM Vision |
| URL | `http://`, `https://` | MarkItDown / agent-browser / Jina |

**旧版 Office 格式转换策略**（DOC/PPT/XLS → DOCX/PPTX/XLSX）：

| 平台 | 首选方案 | 回退方案 | 依赖 |
|------|----------|----------|------|
| Windows | MS Office (COM) | LibreOffice CLI | pywin32 |
| macOS | LibreOffice CLI | - | soffice |
| Linux | LibreOffice CLI | - | soffice |

**转换实现**（Windows pywin32 COM）：
```python
import win32com.client

def convert_legacy_office(input_path: str, output_path: str, file_type: str):
    """Windows 使用 MS Office COM 转换旧版 Office 格式"""
    if file_type == "doc":
        app = win32com.client.Dispatch("Word.Application")
        doc = app.Documents.Open(input_path)
        doc.SaveAs(output_path, FileFormat=16)  # wdFormatDocumentDefault
        doc.Close()
        app.Quit()
    elif file_type == "ppt":
        app = win32com.client.Dispatch("PowerPoint.Application")
        pres = app.Presentations.Open(input_path)
        pres.SaveAs(output_path, FileFormat=24)  # ppSaveAsOpenXMLPresentation
        pres.Close()
        app.Quit()
    elif file_type == "xls":
        app = win32com.client.Dispatch("Excel.Application")
        wb = app.Workbooks.Open(input_path)
        wb.SaveAs(output_path, FileFormat=51)  # xlOpenXMLWorkbook
        wb.Close()
        app.Quit()
```

**转换实现**（LibreOffice CLI 回退）：
```python
import subprocess

def convert_with_libreoffice(input_path: str, output_dir: str, target_format: str):
    """使用 LibreOffice CLI 转换（跨平台回退方案）"""
    subprocess.run([
        "soffice", "--headless", "--convert-to", target_format,
        "--outdir", output_dir, input_path
    ], check=True)
```

### 5.2 输出目录结构

```
output/                              # 输出目录
├── document.docx.md                 # 基础转换后的 Markdown
├── document.docx.llm.md             # LLM 优化后的 Markdown (--llm)
├── assets/                          # 资源目录
│   ├── document.docx.0001.jpg       # 提取的图片（已压缩）
│   ├── document.docx.0002.jpg
│   └── images.json                  # 图片描述文件 (--desc)，合并模式
├── sub_dir/                         # 子目录（保持输入目录结构）
│   ├── file.pdf.md
│   └── assets/
│       └── file.pdf-0-0.png         # PDF 嵌入图片（pymupdf4llm 命名）
├── screenshots/                     # 页面截图目录（--screenshot 模式）
│   ├── document.pdf.page0001.jpg    # PDF 页面截图
│   └── presentation.pptx.slide0001.jpg  # PPTX 幻灯片截图
├── states/                          # 状态目录（用于断点恢复）
│   └── markitai.<hash>.state.json     # 批处理状态文件
└── reports/                         # 报告目录
    └── markitai.<hash>.report.json    # 处理报告（支持 on_conflict 策略）
```

**目录设计说明**：
- `assets/images.json`：采用合并模式，多次运行的描述会按 path 合并，不受 `on_conflict` 策略影响
- `states/`：状态文件用于断点恢复，文件名包含基于输入/输出路径计算的 hash
- `reports/`：报告文件遵循 `on_conflict` 策略（skip/overwrite/rename）

### 5.3 文件命名规则

| 输出类型 | 命名格式 | 示例 |
|----------|----------|------|
| 基础 Markdown | `{原文件名}.md` | `document.docx.md` |
| LLM Markdown | `{原文件名}.llm.md` | `document.docx.llm.md` |
| 提取的图片（一般） | `{原文件名}.{4位序号}.{格式}` | `document.docx.0001.jpg` |
| 提取的图片（PDF） | `{原文件名}-{页码}-{图片索引}.{格式}` | `report.pdf-0-0.png` |
| PDF 页面截图 | `{原文件名}.page{4位页码}.jpg` | `report.pdf.page0001.jpg` |
| PPTX 幻灯片截图 | `{原文件名}.slide{4位序号}.jpg` | `presentation.pptx.slide0001.jpg` |
| 图片描述 | `assets/images.json`（合并存储） | 见下方格式说明 |
| 状态文件 | `states/markitai.{hash}.state.json` | `states/markitai.a1b2c3d4e5f6.state.json` |
| 处理报告 | `reports/markitai.{hash}.report.json` | `reports/markitai.a1b2c3d4e5f6.report.json` |

> 注：PDF 嵌入图片由 pymupdf4llm 命名，页码和图片索引均从 0 开始；页面/幻灯片截图序号从 1 开始。

**Hash 计算说明**：
- Hash 基于以下参数计算：
  - 输入路径（resolved）
  - 输出目录路径（resolved）
  - 关键选项：`llm_enabled`、`ocr_enabled`、`screenshot_enabled`、`image_alt_enabled`、`image_desc_enabled`
- 使用 MD5 取前 6 位十六进制字符
- 不同参数组合会生成不同的 hash，确保不同任务的状态/报告文件相互独立

**图片描述文件格式** (`assets/images.json`)：

```json
{
  "version": "1.0",
  "created": "2026-01-18T10:30:00.000000+08:00",
  "updated": "2026-01-18T10:30:45.123456+08:00",
  "images": [
    {
      "path": "/path/to/output/assets/input.pdf-0-0.png",
      "source": "/path/to/input.pdf",
      "alt": "简短描述（用于 Markdown alt 文本）",
      "desc": "详细描述（Markdown 格式）",
      "text": "提取的文字内容（如有）",
      "created": "2026-01-18T10:30:45.123456+00:00"
    }
  ]
}
```

**文件冲突处理**（由 `output.on_conflict` 控制）：

- `skip`：跳过已存在的文件
- `overwrite`：直接覆盖
- `rename`（默认）：添加版本号 `file.pdf.md` → `file.pdf.v2.md` → `file.pdf.v3.md`

---

## 6. 核心模块

### 6.1 转换引擎 (Converter)

负责将各种文档格式转换为基础 Markdown。

```python
class Converter:
    """文档转换引擎"""

    def convert(self, input_path: Path) -> ConvertResult:
        """
        转换单个文件

        Args:
            input_path: 输入文件路径

        Returns:
            ConvertResult: 包含 markdown 内容和提取的图片列表
        """
        pass

    def detect_format(self, input_path: Path) -> FileFormat:
        """检测文件格式"""
        pass
```

**ConvertResult 数据结构**：

```python
@dataclass
class ConvertResult:
    markdown: str                    # Markdown 内容
    images: list[ExtractedImage]     # 提取的图片列表
    metadata: dict                   # 元数据（页数、字数等）
```

**异步阻塞处理**：`win32com`、`subprocess.run` 等阻塞操作须通过 `loop.run_in_executor()` 放入线程池执行。

**外部依赖检查**：启动时检查 LibreOffice/MS Office 可用性，缺失时给出安装引导。

### 6.2 LLM 集成 (LLMProcessor)

负责 LLM 调用和文本处理。

```python
class LLMProcessor:
    """LLM 处理器"""

    async def clean_markdown(self, content: str) -> str:
        """清洗和优化 Markdown 内容"""
        pass

    async def generate_frontmatter(self, content: str, source: str) -> dict:
        """生成 YAML frontmatter"""
        pass

    async def analyze_image(self, image_path: Path) -> ImageAnalysis:
        """分析图片内容"""
        pass
```

### 6.3 图片处理 (ImageProcessor)

负责图片提取、压缩和分析。

```python
class ImageProcessor:
    """图片处理器"""

    def compress(self, image: Image, quality: int = 85) -> Image:
        """压缩图片"""
        pass

    def extract_from_document(self, doc_path: Path) -> list[ExtractedImage]:
        """从文档中提取图片"""
        pass

    async def generate_caption(self, image_path: Path) -> str:
        """生成图片 alt 文本"""
        pass

    async def generate_description(self, image_path: Path) -> str:
        """生成图片详细描述"""
        pass
```

### 6.4 提示词管理 (PromptManager)

负责提示词的加载和管理。

```python
class PromptManager:
    """提示词管理器"""

    def get_prompt(self, name: str, **variables) -> str:
        """
        获取提示词

        Args:
            name: 提示词名称 (cleaner/frontmatter/image_caption/image_description)
            **variables: 模板变量

        Returns:
            str: 渲染后的提示词
        """
        pass

    def load_custom_prompt(self, path: Path) -> str:
        """加载自定义提示词文件"""
        pass
```

---

## 7. 转换引擎

### 7.1 格式处理器

每种输入格式对应一个处理器：

| 格式 | 处理器 | 依赖 |
|------|--------|------|
| DOCX | `DocxConverter` | markitdown |
| DOC | `DocConverter` | pywin32 / soffice → markitdown |
| PPTX | `PptxConverter` | markitdown |
| PPT | `PptConverter` | pywin32 / soffice → markitdown |
| XLSX | `XlsxConverter` | markitdown |
| XLS | `XlsConverter` | pywin32 / soffice → markitdown |
| PDF | `PdfConverter` | pymupdf4llm + rapidocr |
| TXT | `TxtConverter` | built-in |
| MD | `MarkdownConverter` | built-in |
| Image | `ImageConverter` | rapidocr / litellm |

### 7.2 OCR 集成

**何时启用 OCR**：

1. `--ocr` 参数显式启用
2. 配置文件中 `ocr.enabled: true`

**注意**：OCR 模式需要手动启用，适用于扫描版 PDF 或图片文件。启用后：
- PDF 文件将逐页进行 OCR 识别
- 图片文件将直接进行 OCR 文字提取

**OCR 配置**：

```python
from rapidocr import RapidOCR

engine = RapidOCR(params={
    "Rec.lang_type": config.ocr.lang,  # 内部映射 zh→ch
})

result = engine(image_path)
text = "\n".join(result.txts)
```

### 7.3 Conversion Pipeline

```
                    ┌─────────────────────┐
                    │  Detect Input File  │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Select Converter   │
                    └──────────┬──────────┘
                               │
            ┌──────────────────┼──────────────────┐
            │                  │                  │
   ┌────────▼────────┐ ┌───────▼──────┐ ┌─────────▼────────┐
   │  MarkItDown     │ │ pymupdf4llm  │ │  Direct Read     │
   │ (DOCX/PPTX/XLS) │ │ (PDF) + OCR  │ │   (TXT/MD)       │
   └────────┬────────┘ └───────┬──────┘ └─────────┬────────┘
            │                  │                  │
            └──────────────────┼──────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │ Extract & Compress  │
                    │       Images        │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Output Base .md    │
                    └─────────────────────┘
```

---

## 8. LLM 集成

### 8.1 LiteLLM Router 配置

采用 LiteLLM Router 实现多模型资源池和负载均衡：

```python
from litellm import Router

# 从配置构建 Router
router = Router(
    model_list=config.llm.model_list,
    **config.llm.router_settings
)

# 调用时使用逻辑模型名
response = await router.acompletion(
    model="default",  # 逻辑模型名，自动负载均衡
    messages=[{"role": "user", "content": prompt}],
)
```

**支持的路由策略**：
- `simple-shuffle`: 加权随机分配（默认，支持 weight 参数）
- `least-busy`: 最少繁忙
- `usage-based-routing`: 基于用量
- `latency-based-routing`: 基于延迟

**Proxy 模式**（可选）：

```python
from litellm import acompletion

# 连接外部 LiteLLM Proxy Server
response = await acompletion(
    model="default",
    messages=[{"role": "user", "content": prompt}],
    api_base="http://localhost:4000",  # LiteLLM Proxy
)
```

### 8.2 模型配置要求

**重要**：必须在配置文件或环境变量中指定模型名称，无默认模型。

支持的模型格式示例：
- `deepseek/deepseek-chat` - DeepSeek
- `gemini/gemini-2.5-flash` - Google Gemini
- `gpt-4o-mini` - OpenAI
- `anthropic/claude-sonnet-4-20250514` - Anthropic Claude
- `ollama/llama3.2` - 本地 Ollama

### 8.3 成本追踪

使用 LiteLLM 内置的成本追踪功能：

```python
from litellm import completion_cost

response = await acompletion(...)
cost = completion_cost(completion_response=response)
logger.info(f"LLM cost: ${cost:.6f}")
```

### 8.4 并发控制

使用 `LLMRuntime` 实现全局并发控制，确保跨多个 `LLMProcessor` 实例共享 Semaphore：

```python
from dataclasses import dataclass, field
import asyncio

@dataclass
class LLMRuntime:
    """全局 LLM 并发控制器"""
    concurrency: int
    _semaphore: asyncio.Semaphore | None = field(default=None, init=False)

    @property
    def semaphore(self) -> asyncio.Semaphore:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.concurrency)
        return self._semaphore

# 批处理中创建共享 Runtime
runtime = LLMRuntime(concurrency=config.llm.concurrency)
processor = LLMProcessor(config.llm, config.prompts, runtime=runtime)

# LLMProcessor 内部使用共享 semaphore
async def call_llm_with_limit(prompt: str) -> str:
    async with self.semaphore:  # 来自 runtime 或本地
        return await acompletion(...)
```

### 8.5 错误处理与重试

**重试配置**：
- 最大重试次数：2（由 `llm.router_settings.num_retries` 配置）
- 重试间隔：指数退避（1s, 2s, 4s, ...，最大 60s）
- 可重试错误：`RateLimitError`, `APIConnectionError`, `Timeout`, `ServiceUnavailableError`

**降级策略**（重试耗尽后）：
1. 保留基础转换结果（输出 `file.md`）
2. 不生成 LLM 文件（无 `file.llm.md`）
3. 记录警告日志
4. 继续处理下一个文件
5. 最终退出码为 5（部分成功）或 10（全部失败）

---

## 9. 图片处理

### 9.1 图片提取

从文档中提取嵌入的图片：

```python
class ImageExtractor:
    def extract(self, doc_path: Path, output_dir: Path) -> list[ExtractedImage]:
        """
        提取文档中的图片

        Returns:
            list[ExtractedImage]: 提取的图片列表，包含路径和原始位置信息
        """
        pass
```

**ExtractedImage 数据结构**：

```python
@dataclass
class ExtractedImage:
    path: Path           # 图片文件路径
    index: int           # 图片序号
    original_name: str   # 原始文件名
    mime_type: str       # MIME 类型
    width: int           # 宽度
    height: int          # 高度
```

### 9.2 图片压缩

默认压缩参数：
- 格式：JPEG
- 质量：85%
- 最大尺寸：1920x1080（保持比例缩放，用于节省 LLM Vision Token 成本）

```python
from PIL import Image

def compress_image(
    image: Image.Image,
    quality: int = 85,
    max_size: tuple[int, int] = (1920, 1080),
    format: str = "JPEG"
) -> Image.Image:
    # 缩放
    image.thumbnail(max_size, Image.Resampling.LANCZOS)

    # 转换为 RGB（JPEG 不支持透明通道）
    if format == "JPEG" and image.mode in ("RGBA", "P"):
        image = image.convert("RGB")

    return image
```

### 9.3 图片分析

图片分析有两种模式：

**OCR 模式**（默认）：
```python
from rapidocr import RapidOCR

engine = RapidOCR()
result = engine(image_path)
text = "\n".join(result.txts)
```

**LLM Vision 模式**（--llm + --alt / --desc）：
```python
import base64
from litellm import acompletion

def image_to_base64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

async def analyze_with_llm(image_path: Path, prompt: str) -> str:
    base64_image = image_to_base64(image_path)

    # 使用 router 调用视觉模型（通过 model_name="vision" 区分）
    response = await router.acompletion(
        model="vision",  # 逻辑模型名，自动负载均衡到 supports_vision=true 的模型
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }]
    )
    return response.choices[0].message.content
```

### 9.4 图片分析输出

**alt 文本**：生成简短的图片描述，用于 `![alt](path)` 中的 alt 属性。

**图片描述文件**（`images.json`）：生成详细的图片描述，保存到 `assets/images.json`，格式见 5.3 节。

#### PDF/PPTX 图片抽取说明

**PDF 文档**会产生两类图片，存放在不同目录：

| 类型 | 目录 | 命名格式 | 触发条件 | 用途 |
|------|------|----------|----------|------|
| 嵌入图片 | `assets/` | `{filename}.pdf-{page}-{index}.{ext}` | 自动（PDF 含图时） | 文档内嵌的图像资源 |
| 页面截图 | `screenshots/` | `{filename}.pdf.page{page:04d}.jpg` | `--screenshot` | Vision LLM 整页分析 |

**PDF 嵌入图片**（Embedded Images）：
- 由 pymupdf4llm 自动抽取 PDF 内嵌的图像资源
- 命名示例：`report.pdf-0-0.png`（第 1 页第 1 张图）、`report.pdf-2-1.jpg`（第 3 页第 2 张图）
- 页码和图片索引均从 0 开始
- 会进行压缩和过滤（根据 `image.filter` 配置）
- 在 Markdown 中以 `![alt](assets/report.pdf-0-0.png)` 形式引用

**PDF 页面截图**（Page Screenshots）：
- 仅在 `--screenshot` 模式下生成
- 将每页渲染为 JPEG 图片（300 DPI）
- 命名示例：`report.pdf.page0001.jpg`、`report.pdf.page0002.jpg`（4 位数字，从 1 开始）
- 用于 Vision LLM 分析整页布局和内容
- 在 `.llm.md` 中以注释形式引用：`<!-- ![Page 1](screenshots/report.pdf.page0001.jpg) -->`

**PPTX 文档**仅产生幻灯片截图（无单独的嵌入图片提取）：

| 类型 | 目录 | 命名格式 | 触发条件 |
|------|------|----------|----------|
| 幻灯片截图 | `screenshots/` | `{filename}.pptx.slide{slide:04d}.jpg` | `--screenshot` |

- 命名示例：`presentation.pptx.slide0001.jpg`、`presentation.pptx.slide0002.jpg`
- 在 `.llm.md` 中以注释形式引用：`<!-- ![Slide 1](screenshots/presentation.pptx.slide0001.jpg) -->`

---

## 10. 提示词管理

### 10.1 内置提示词

系统内置以下提示词：

| 名称 | 文件 | 用途 |
|------|------|------|
| `cleaner` | `cleaner.md` | Markdown 清洗和格式优化 |
| `frontmatter` | `frontmatter.md` | 生成 YAML frontmatter 元数据 |
| `image_caption` | `image_caption.md` | 生成图片 alt 文本（简短描述） |
| `image_description` | `image_description.md` | 生成图片详细描述 |
| `image_analysis` | `image_analysis.md` | 合并的图片分析（caption + description） |
| `page_content` | `page_content.md` | OCR 页面内容提取 |
| `document_process` | `document_process.md` | 文档处理（cleaner + frontmatter 合并调用） |
| `document_enhance` | `document_enhance.md` | 文档增强（结合提取文本和页面截图） |
| `document_enhance_complete` | `document_enhance_complete.md` | 完整文档增强（含 frontmatter 生成） |
| `url_enhance` | `url_enhance.md` | URL/网页内容增强（多源内容合并） |

> 内置提示词路径：`packages/markitai/src/markitai/prompts/`

### 10.2 自定义覆盖

用户可以通过以下方式覆盖默认提示词：

**方式 1**：在配置文件中指定路径

```json
{
  "prompts": {
    "cleaner": "/path/to/custom/cleaner.md"
  }
}
```

**方式 2**：在提示词目录中放置同名文件

```
~/.markitai/prompts/
├── cleaner.md          # 覆盖 cleaner 提示词
├── frontmatter.md      # 覆盖 frontmatter 提示词
├── image_caption.md    # 覆盖 image_caption 提示词
└── image_description.md # 覆盖 image_description 提示词
```

### 10.3 提示词模板变量

| 变量 | 说明 | 可用于 |
|------|------|--------|
| `{content}` | 文档内容 | cleaner, frontmatter |
| `{source}` | 源文件名 | frontmatter |
| `{timestamp}` | 处理时间 | frontmatter |

---

## 11. 批量处理

### 11.1 并发模型

使用 asyncio 实现异步并发处理：

```python
async def process_batch(
    files: list[Path],
    concurrency: int = 10
) -> list[ProcessResult]:
    semaphore = asyncio.Semaphore(concurrency)

    async def process_with_limit(file: Path) -> ProcessResult:
        async with semaphore:
            return await process_file(file)

    tasks = [process_with_limit(f) for f in files]
    return await asyncio.gather(*tasks, return_exceptions=True)
```

### 11.2 进度追踪

使用 rich 库显示进度条（URL 进度 + 文件进度）：

```python
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn

progress = Progress(
    SpinnerColumn(),
    TextColumn("[bold blue]{task.fields[filename]:<30}"),
    BarColumn(),
    TaskProgressColumn(),
    TimeElapsedColumn(),
)

# URL 进度（如果有 .urls 文件）
if total_urls > 0:
    url_task = progress.add_task("URLs", total=total_urls, filename=f"[URLs:0/{total_urls}]")

# 文件进度
file_task = progress.add_task("Files", total=total_files, filename=f"[Files:0/{total_files}]")
```

**控制台显示效果**：
```
⠋ [URLs:3/10]                    ━━━━━━━━━━━━━━━━━━━━  30%  0:00:45
⠋ [Files:25/100]                 ━━━━━━━━━━━━━━━━━━━━  25%  0:02:30
```

> 注：`--verbose` 模式下会额外显示日志面板，实时显示处理详情。

### 11.3 断点恢复

批处理使用独立的状态文件和报告文件：
- **状态文件**：`states/markitai.<hash>.state.json`，用于断点恢复
- **报告文件**：`reports/markitai.<hash>.report.json`，遵循 `on_conflict` 策略

**状态文件格式** (`states/markitai.<hash>.state.json`)：

状态文件仅保存断点恢复所需的最小信息：

```json
{
  "version": "1.0",
  "options": {
    "input_dir": "/path/to/input",
    "output_dir": "/path/to/output",
    "llm_enabled": true,
    "image_alt_enabled": true,
    "image_desc_enabled": false
  },
  "documents": {
    "file1.pdf": {
      "status": "completed",
      "output": "file1.pdf.md"
    },
    "file2.docx": {
      "status": "failed",
      "error": "LLM timeout"
    },
    "file3.xlsx": {
      "status": "pending"
    }
  },
  "urls": {
    "https://example.com/page1": {
      "status": "completed",
      "source_file": "links.urls",
      "output": "example.com_page1.md"
    }
  }
}
```

**说明**：
- 状态文件使用最小化格式，仅包含恢复所需字段
- `documents`/`urls` 的 key 使用相对于 `input_dir` 的相对路径或 URL
- `output` 为相对于 `output_dir` 的相对路径
- 状态文件在每次处理完成后更新（节流间隔由 `batch.state_flush_interval_seconds` 控制）

**报告文件格式** (`reports/markitai.<hash>.report.json`)：

报告文件在批处理完成后生成，包含完整状态信息、`summary` 和 `llm_usage` 统计：

```json
{
  "version": "1.0",
  "started_at": "2026-01-14T10:00:00+08:00",
  "updated_at": "2026-01-14T10:30:00+08:00",
  "generated_at": "2026-01-14T10:30:00+08:00",
  "log_file": "/home/user/.markitai/logs/markitai_20260114_100000_123456.log",
  "options": { "..." },
  "documents": {
    "file1.pdf": {
      "status": "completed",
      "output": "file1.pdf.md",
      "started_at": "2026-01-14T10:00:05+08:00",
      "completed_at": "2026-01-14T10:00:17+08:00",
      "duration": 12.5,
      "images": 5,
      "screenshots": 0,
      "cost_usd": 0.0012,
      "llm_usage": { "..." },
      "cache_hit": false
    }
  },
  "urls": {
    "https://example.com/page": {
      "source_file": "links.urls",
      "status": "completed",
      "output": "example.com_page.md",
      "fetch_strategy": "static",
      "images": 3,
      "duration": 5.2,
      "cost_usd": 0.0008,
      "llm_usage": { "..." },
      "cache_hit": false
    }
  },
  "url_sources": ["links.urls"],
  "summary": {
    "total_documents": 100,
    "completed_documents": 95,
    "failed_documents": 5,
    "pending_documents": 0,
    "total_urls": 20,
    "completed_urls": 18,
    "failed_urls": 2,
    "pending_urls": 0,
    "url_cache_hits": 5,
    "url_sources": 2,
    "duration": 932.5,
    "processing_time": 1250.5
  },
  "llm_usage": {
    "models": {
      "gemini/gemini-2.5-flash": {
        "requests": 200,
        "input_tokens": 167000,
        "output_tokens": 60000,
        "cost_usd": 0.0390
      }
    },
    "total_requests": 200,
    "total_input_tokens": 167000,
    "total_output_tokens": 60000,
    "total_cost_usd": 0.0390
  }
}
```

**恢复逻辑**：

使用 `--resume` 参数时，系统自动从报告文件加载状态，跳过已完成的文件：

```python
# BatchState.get_pending_files() 内部实现
def get_pending_files(self) -> list[Path]:
    return [
        Path(f.path)
        for f in self.files.values()
        if f.status in (FileStatus.PENDING, FileStatus.FAILED)
    ]
```

---

## 12. 日志系统

### 12.1 日志级别

| 级别 | 说明 | 使用场景 |
|------|------|----------|
| DEBUG | 调试信息 | 开发调试 |
| INFO | 一般信息 | 默认级别 |
| WARNING | 警告信息 | LLM 降级、跳过文件 |
| ERROR | 错误信息 | 处理失败 |
| CRITICAL | 严重错误 | 程序无法继续 |

### 12.2 日志分离架构

日志系统采用控制台/文件分离架构，确保批处理期间文件日志不受进度条显示影响：

```
┌─────────────────────────────────────────────────────────────────┐
│                         setup_logging()                         │
├─────────────────────────────────────────────────────────────────┤
│  console_handler = logger.add(stderr)                           │
│      - level: DEBUG if verbose else INFO                        │
│      - disabled during batch processing (progress bar conflict) │
│                                                                 │
│  file_handler = logger.add(file)                                │
│      - level: config.log.level (default: DEBUG)                 │
│      - serialize=True (JSON)                                    │
│      - runs independently, unaffected by batch processing       │
└─────────────────────────────────────────────────────────────────┘
```

### 12.3 批处理日志显示

**普通模式**：只显示进度条，日志写入文件

**Verbose 模式**：进度条 + LogPanel（8行滚动日志）

```
┌──────────────────────────────────────────────────────────────────┐
│  [Overall Progress]  ━━━━━━━━━━━━━━━━━━━━  50%  0:03:22          │
├────────────────────────────────── Logs ──────────────────────────┤
│  18:01:15 | [LLM:file.pdf:1] gemini-2.5-flash tokens=1500+200    │
│           | time=1234ms cost=$0.001200                           │
│  18:01:16 | [LLM:file.pdf:2] Retry #1: RateLimitError status=429 │
│  18:01:18 | [LLM:file.pdf:2] gemini-2.5-flash tokens=800+150     │
│           | time=980ms cost=$0.000800                            │
│  18:01:20 | [DONE] file.pdf: 45.2s (images=7, cost=$0.025839)    │
│  18:01:21 | [START] document.docx                                │
│  18:01:22 | [LLM:document.docx:1] deepseek-chat tokens=2000+300  │
└──────────────────────────────────────────────────────────────────┘
```

### 12.4 日志事件格式

| 事件 | 级别 | 格式 | 触发时机 |
|------|------|------|---------|
| `file_start` | INFO | `[START] {filename}` | 文件处理开始 |
| `llm_request` | DEBUG | `[LLM:{file}:{n}] Request to {model}` | LLM 调用开始 |
| `llm_success` | INFO | `[LLM:{file}:{n}] {model} tokens={in}+{out} time={ms}ms cost=${cost}` | LLM 调用成功 |
| `llm_retry` | WARNING | `[LLM:{file}:{n}] Retry #{attempt}: {error_type} status={code}` | LLM 重试 |
| `llm_error` | ERROR | `[LLM:{file}:{n}] Failed: {error_type} status={code}` | LLM 最终失败 |
| `file_done` | INFO | `[DONE] {filename}: {time}s (images={n}, cost=${cost})` | 文件处理完成 |
| `file_error` | ERROR | `[FAIL] {filename}: {error}` | 文件处理失败 |

### 12.5 文件日志格式（JSON）

```json
{"time": "2026-01-16T18:01:15.123+08:00", "level": "INFO", "message": "[START] file.pdf"}
{"time": "2026-01-16T18:01:16.456+08:00", "level": "DEBUG", "message": "[LLM:file.pdf:1] Request to default"}
{"time": "2026-01-16T18:01:17.789+08:00", "level": "INFO", "message": "[LLM:file.pdf:1] gemini-2.5-flash tokens=1500+200 time=1234ms cost=$0.001200"}
{"time": "2026-01-16T18:01:18.500+08:00", "level": "WARNING", "message": "[LLM:file.pdf:2] Retry #1: RateLimitError status=429"}
```

### 12.6 LLM 重试追踪

禁用 LiteLLM Router 内部重试，使用自定义重试逻辑追踪每次重试：

```python
RETRYABLE_ERRORS = (RateLimitError, APIConnectionError, Timeout, ServiceUnavailableError)

async def _call_llm_with_retry(self, model, messages, call_id, max_retries=2):
    for attempt in range(max_retries + 1):
        try:
            response = await self.router.acompletion(...)
            logger.info(f"[LLM:{call_id}] {model} tokens=... time=...ms cost=$...")
            return response
        except RETRYABLE_ERRORS as e:
            if attempt < max_retries:
                logger.warning(f"[LLM:{call_id}] Retry #{attempt+1}: {type(e).__name__} status={e.status_code}")
                await asyncio.sleep(min(2 ** attempt, 60))
            else:
                logger.error(f"[LLM:{call_id}] Failed after {max_retries+1} attempts")
                raise
```

---

## 13. 错误处理

### 13.1 错误分类

| 错误类型 | 处理方式 |
|----------|----------|
| 配置错误 | 立即退出，提示修复 |
| 输入文件不存在 | 跳过，记录警告 |
| 格式不支持 | 跳过，记录警告 |
| 转换失败 | 跳过，记录错误 |
| LLM 调用失败 | 重试后降级 |
| 图片处理失败 | 跳过图片，继续处理 |

---

## 14. 测试策略

### 14.1 单元测试

测试框架：pytest + pytest-xdist（并行加速）

```bash
# 安装
uv add --dev pytest pytest-xdist

# 运行测试（自动并行）
uv run pytest -n auto

# 指定并行数
uv run pytest -n 4
```

```
tests/
├── unit/
│   ├── test_converter.py      # 转换引擎测试
│   ├── test_llm_processor.py  # LLM 处理测试
│   ├── test_image_processor.py # 图片处理测试
│   ├── test_prompt_manager.py  # 提示词管理测试
│   └── test_config.py          # 配置管理测试
├── integration/
│   ├── test_cli.py             # CLI 集成测试
│   └── test_batch.py           # 批处理测试
└── fixtures/
    ├── sample.docx
    ├── sample.pdf
    └── sample.png
```

### 14.2 集成测试

```python
def test_basic_conversion():
    """测试基础转换流程"""
    result = subprocess.run(
        ["markitai", "tests/fixtures/sample.docx", "-o", "tmp/output"],
        capture_output=True
    )
    assert result.returncode == 0
    assert Path("tmp/output/sample.docx.md").exists()
```

### 14.3 SKILL.md 测试

手动测试步骤见 `packages/markitai/tests/SKILL.md`。

---

## 附录

### A. 配置文件完整示例

```json
{
  "output": {
    "dir": "./output",
    "on_conflict": "rename",
    "allow_symlinks": false
  },
  "llm": {
    "enabled": false,
    "model_list": [
      {
        "model_name": "default",
        "litellm_params": {
          "model": "deepseek/deepseek-chat",
          "api_key": "env:DEEPSEEK_API_KEY",
          "weight": 7
        }
      },
      {
        "model_name": "default",
        "litellm_params": {
          "model": "gemini/gemini-2.5-flash",
          "api_key": "env:GEMINI_API_KEY",
          "weight": 3
        }
      }
    ],
    "router_settings": {
      "routing_strategy": "simple-shuffle",
      "num_retries": 2,
      "timeout": 120,
      "fallbacks": []
    },
    "concurrency": 10
  },
  "image": {
    "alt_enabled": false,
    "desc_enabled": false,
    "compress": true,
    "quality": 75,
    "format": "jpeg",
    "max_width": 1920,
    "max_height": 99999,
    "filter": {
      "min_width": 50,
      "min_height": 50,
      "min_area": 5000,
      "deduplicate": true
    }
  },
  "ocr": {
    "enabled": false,
    "lang": "en"
  },
  "screenshot": {
    "enabled": false,
    "viewport_width": 1920,
    "viewport_height": 1080,
    "quality": 75,
    "max_height": 10000
  },
  "prompts": {
    "dir": "~/.markitai/prompts",
    "cleaner": null,
    "frontmatter": null,
    "image_caption": null,
    "image_description": null,
    "image_analysis": null,
    "page_content": null,
    "document_enhance": null,
    "url_enhance": null
  },
  "batch": {
    "concurrency": 10,
    "url_concurrency": 5,
    "state_flush_interval_seconds": 10,
    "scan_max_depth": 5,
    "scan_max_files": 10000
  },
  "log": {
    "level": "INFO",
    "dir": "~/.markitai/logs",
    "rotation": "10 MB",
    "retention": "7 days"
  },
  "cache": {
    "enabled": true,
    "no_cache": false,
    "no_cache_patterns": [],
    "max_size_bytes": 536870912,
    "global_dir": "~/.markitai"
  },
  "fetch": {
    "strategy": "auto",
    "agent_browser": {
      "command": "agent-browser",
      "timeout": 30000,
      "wait_for": "domcontentloaded",
      "extra_wait_ms": 1000,
      "session": null
    },
    "jina": {
      "api_key": null,
      "timeout": 30
    },
    "fallback_patterns": ["x.com", "twitter.com"]
  },
  "presets": {
    "rich": {
      "llm": true,
      "ocr": false,
      "alt": true,
      "desc": true,
      "screenshot": true
    },
    "standard": {
      "llm": true,
      "ocr": false,
      "alt": true,
      "desc": true,
      "screenshot": false
    },
    "minimal": {
      "llm": false,
      "ocr": false,
      "alt": false,
      "desc": false,
      "screenshot": false
    }
  }
}
```

### B. YAML Frontmatter 示例

```yaml
---
title: 2024年度销售报告
source: sales_report_2024.docx
description: 本报告总结了2024年度的销售业绩，包括季度分析、区域对比和产品线表现。
tags:
  - 销售报告
  - 年度总结
  - 数据分析
markitai_processed: "2026-01-14T10:30:00+08:00"
---
```

**注意**：`markitai_processed` 使用本地时间（带时区偏移），而非 UTC 时间。

**tags 生成规则**（在提示词中明确）：
- 数量：3-10 个
- 字数：每个标签 2-8 个字
- 格式：使用名词或名词短语
- 避免：过于宽泛的标签（如"文档"、"报告"）

### C. 错误码列表

| 错误码 | 名称 | 说明 |
|--------|------|------|
| 0 | SUCCESS | 成功 |
| 1 | GENERAL_ERROR | 一般错误 |
| 2 | CONFIG_ERROR | 配置错误 |
| 3 | INPUT_NOT_FOUND | 输入文件不存在 |
| 4 | OUTPUT_DIR_ERROR | 输出目录无法创建 |
| 5 | LLM_DEGRADED | LLM 调用失败（已降级处理） |
| 10 | PARTIAL_FAILURE | 部分文件处理失败 |

### D. 支持的 LLM 模型

Markitai 支持所有 LiteLLM 兼容的模型（100+ providers），完整列表见 [LiteLLM Providers](https://docs.litellm.ai/docs/providers)。

**推荐模型配置**（参考 `markitai.json`）：

| Provider | 模型标识 | 说明 |
|----------|----------|------|
| Google | `gemini/gemini-2.5-flash-lite` | 最快最便宜，日常任务首选 |
| Google | `gemini/gemini-3-flash-preview` | 多模态，高质量 |
| DeepSeek | `deepseek/deepseek-chat` | 低成本，中文优秀 |
| Anthropic | `claude-haiku-4-5` | 快速响应，低成本 |
| Anthropic | `claude-sonnet-4-5` | 代码/写作优秀 |
| OpenRouter | `openrouter/openai/gpt-5.2` | OpenAI GPT-5.2 |
| Ollama | `ollama/llama3.2` | 本地部署，隐私优先 |

---

*文档结束*
