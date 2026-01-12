# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 构建与开发命令

**重要**: 运行 uv/Python/markit 相关命令前，先激活虚拟环境：

```bash
# Linux/macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

```bash
# 安装依赖 (需要 Python 3.12+，优先使用 uv)
uv sync --all-extras

# 运行所有测试
pytest

# 运行单个测试文件
pytest tests/unit/test_pipeline.py

# 运行特定测试
pytest tests/unit/test_pipeline.py::test_function_name -v

# 代码检查和格式化
ruff check .
ruff format .

# 类型检查
mypy markit
```

## CLI 使用

```bash
# 单文件转换
markit convert document.docx

# 启用 LLM 增强
markit convert document.docx --llm

# 批量转换
markit batch ./docs -o ./output -r

# 配置管理
markit config init     # 创建 markit.yaml
markit config test     # 验证配置
markit provider list   # 列出已配置的 LLM 提供商
```

## 架构概述

MarkIt 是一个文档转 Markdown 工具，支持可选的 LLM 增强。代码库采用面向服务的架构，职责分离清晰。

### 核心管道流程

```
输入文件 → FormatRouter → 预处理器 → 转换器 → ImageProcessingService → LLMOrchestrator → OutputManager → 输出
```

1. **FormatRouter** (`markit/core/router.py`): 根据文件扩展名路由到对应转换器
   - PDF: 根据配置路由到 pymupdf4llm/pymupdf/pdfplumber
   - 旧格式 (.doc, .ppt, .xls): 添加 OfficePreprocessor 先通过 LibreOffice 转换
   - 现代 Office/HTML: 使用 MarkItDown 转换器

2. **ConversionPipeline** (`markit/core/pipeline.py`): 主协调器，负责：
   - 带回退支持的文档转换
   - 委托 ImageProcessingService 处理图片
   - 委托 LLMOrchestrator 处理 LLM 操作
   - 通过 OutputManager 写入输出

### 服务层

- **ImageProcessingService** (`markit/services/image_processor.py`): 处理图片格式转换、压缩（通过 oxipng/Pillow）、去重，并准备图片供 LLM 分析

- **LLMOrchestrator** (`markit/services/llm_orchestrator.py`): 集中所有 LLM 操作：
  - 管理 ProviderManager 支持多提供商
  - 创建 MarkdownEnhancer 进行文本清理
  - 创建 ImageAnalyzer 处理视觉任务
  - 实现基于能力的路由（文本模型 vs 视觉模型）

- **OutputManager** (`markit/services/output_manager.py`): 处理文件写入、冲突解决，生成图片描述 markdown 文件

### LLM 提供商系统

**ProviderManager** (`markit/llm/manager.py`): 管理多个 LLM 提供商：
- 延迟初始化（按需验证提供商）
- 基于能力的路由（文本 vs 视觉任务）
- 失败时自动回退
- 并发回退（主模型超时时启动备用模型）
- 轮询负载均衡
- 按模型成本追踪

支持的提供商: OpenAI, Anthropic, Gemini, Ollama, OpenRouter（均在 `markit/llm/` 模块中）

### 配置系统

设置定义在 `markit/config/settings.py`，使用 pydantic-settings：
- **LLMConfig**: 支持旧版单提供商和新版凭证/模型分离两种模式
- **LLMCredentialConfig**: 提供商凭证（可引用环境变量）
- **LLMModelConfig**: 引用凭证的模型实例，带能力声明

配置从当前目录或父目录的 `markit.yaml` 加载。

### 关键设计模式

1. **分阶段管道**: 批处理时，管道分为多个阶段：
   - 阶段 1: `convert_document_only()` - CPU 密集型转换，提前释放文件信号量
   - 阶段 2: `create_llm_tasks()` - 为 LLM 队列创建协程
   - 阶段 3: `finalize_output()` - 合并结果并写入输出

2. **基于能力的路由**: 模型声明能力（`["text"]` 或 `["text", "vision"]`）。纯文本任务路由到更便宜的文本模型；视觉任务只路由到具备视觉能力的模型。

3. **LibreOffice 配置文件池** (`markit/converters/libreoffice_pool.py`): 并行转换 .doc/.ppt/.xls 时，使用隔离的 LibreOffice 配置文件目录避免冲突。

4. **图片处理进程池**: 重度图片压缩使用进程池绕过 Python GIL。

### 文档同步规则

**重要**：更新文档时必须同步更新关联文档：
- `README.md` ↔ `README_CN.md`：内容保持同步
- `CLAUDE.md` ↔ `AGENTS.md`：开发规范保持一致
- `docs/ROADMAP.md`：任务进展及时更新

### 文件结构

```
markit/
├── cli/           # Typer CLI 命令 (convert, batch, config, provider, model)
├── config/        # 设置、常量
├── converters/    # 格式转换器 (markitdown, pandoc, pdf/)
├── core/          # 管道、路由、状态管理
├── image/         # 图片分析、压缩、提取
├── llm/           # LLM 提供商 (openai, anthropic, gemini, ollama, openrouter)
├── markdown/      # Markdown 处理 (chunker, formatter, frontmatter)
├── services/      # 服务层 (image_processor, llm_orchestrator, output_manager)
└── utils/         # 工具类 (concurrency, fs, logging, stats)
```
