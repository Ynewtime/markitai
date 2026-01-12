# 贡献指南

本指南涵盖 MarkIt 的开发环境配置、架构概述和贡献指南。

## 环境要求

- Python 3.12+
- [uv](https://github.com/astral-sh/uv)（推荐）或 pip

## 开发环境配置

```bash
# 克隆仓库
git clone https://github.com/user/markit.git
cd markit

# 创建虚拟环境并安装依赖
uv sync --all-extras

# 激活虚拟环境
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
```

## 开发命令

```bash
# 运行所有测试
pytest

# 运行单个测试文件
pytest tests/unit/test_pipeline.py

# 运行特定测试（详细输出）
pytest tests/unit/test_pipeline.py::test_function_name -v

# 代码检查和格式化
ruff check .
ruff format .

# 类型检查
pyright src/markit

# Pre-commit hooks（推荐）
pre-commit install                    # 安装 pre-commit hook
pre-commit install --hook-type pre-push  # 安装 pre-push hook
pre-commit run --all-files            # 手动运行全部检查

# Justfile 命令（跨平台任务运行器）
# 安装: cargo install just / brew install just / choco install just / scoop install just
just --list      # 列出所有可用命令
just ci          # 运行全部 CI 检查（lint + typecheck + test）
just test-cov    # 运行测试并生成覆盖率报告
just clean       # 清理构建产物
just build       # 构建包
```

## 架构概述

MarkIt 采用模块化、面向服务的架构，职责分离清晰。

### 核心管道流程

```
输入文件 → FormatRouter → 预处理器 → 转换器 → ImageProcessingService → LLMOrchestrator → OutputManager → 输出
```

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

### 核心组件

1. **FormatRouter** (`src/markit/core/router.py`)：根据文件扩展名路由到对应转换器
   - PDF：根据配置路由到 pymupdf4llm/pymupdf/pdfplumber
   - 旧格式 (.doc, .ppt, .xls)：添加 OfficePreprocessor 先通过 LibreOffice 转换
   - 现代 Office/HTML：使用 MarkItDown 转换器

2. **ConversionPipeline** (`src/markit/core/pipeline.py`)：主协调器
   - 带回退支持的文档转换
   - 委托 ImageProcessingService 处理图片
   - 委托 LLMOrchestrator 处理 LLM 操作
   - 通过 OutputManager 写入输出

### 服务层

- **ImageProcessingService** (`src/markit/services/image_processor.py`)：处理图片格式转换、压缩（通过 oxipng/Pillow）、去重，并准备图片供 LLM 分析

- **LLMOrchestrator** (`src/markit/services/llm_orchestrator.py`)：集中所有 LLM 操作
  - 管理 ProviderManager 支持多提供商
  - 创建 MarkdownEnhancer 进行文本清理
  - 创建 ImageAnalyzer 处理视觉任务
  - 实现基于能力的路由（文本模型 vs 视觉模型）

- **OutputManager** (`src/markit/services/output_manager.py`)：处理文件写入、冲突解决，生成图片描述 markdown 文件

### LLM 提供商系统

**ProviderManager** (`src/markit/llm/manager.py`)：管理多个 LLM 提供商
- 延迟初始化（按需验证提供商）
- 基于能力的路由（文本 vs 视觉任务）
- 失败时自动回退
- 并发回退（主模型超时时启动备用模型）
- 轮询负载均衡
- 按模型成本追踪

支持的提供商：OpenAI、Anthropic、Gemini、Ollama、OpenRouter（均在 `src/markit/llm/` 模块中）

### 配置系统

设置定义在 `src/markit/config/settings.py`，使用 pydantic-settings：
- **LLMConfig**：支持旧版单提供商和新版凭证/模型分离两种模式
- **LLMCredentialConfig**：提供商凭证（可引用环境变量）
- **LLMModelConfig**：引用凭证的模型实例，带能力声明

配置从当前目录或父目录的 `markit.yaml` 加载。

## 关键设计模式

### 1. 分阶段管道

批处理时，管道分为多个阶段：
- **阶段 1**：`convert_document_only()` - CPU 密集型转换，提前释放文件信号量
- **阶段 2**：`create_llm_tasks()` - 为 LLM 队列创建协程
- **阶段 3**：`finalize_output()` - 合并结果并写入输出

### 2. 基于能力的路由

模型声明能力（`["text"]` 或 `["text", "vision"]`）。纯文本任务路由到更便宜的文本模型；视觉任务只路由到具备视觉能力的模型。

### 3. AIMD 自适应并发

实现加性增、乘性减算法：
- 连续 N 次成功后，并发数 +1
- 遇到 429 限流，并发数 ×0.5
- 冷却期防止振荡

### 4. 死信队列 (DLQ)

跟踪每个文件的失败次数：
- 记录失败次数和最后错误
- 达到最大重试后标记为永久失败
- 防止"毒药文件"在恢复时反复卡死队列

### 5. LibreOffice 配置文件池

`src/markit/converters/libreoffice_pool.py`：并行转换 .doc/.ppt/.xls 时，使用隔离的 LibreOffice 配置文件目录避免冲突。

### 6. 进程池图像处理

重度图片压缩使用进程池绕过 Python GIL。

## 项目结构

```
.
├── src/markit/        # 源码 (src layout)
│   ├── cli/           # Typer CLI 命令 (convert, batch, config, provider, model)
│   ├── config/        # 设置、常量
│   ├── converters/    # 格式转换器 (markitdown, pandoc, pdf/)
│   ├── core/          # 管道、路由、状态管理
│   ├── image/         # 图片分析、压缩、提取
│   ├── llm/           # LLM 提供商 (openai, anthropic, gemini, ollama, openrouter)
│   ├── markdown/      # Markdown 处理 (chunker, formatter, frontmatter)
│   ├── services/      # 服务层 (image_processor, llm_orchestrator, output_manager)
│   └── utils/         # 工具类 (concurrency, fs, logging, stats)
├── tests/
│   ├── unit/          # 单元测试
│   ├── integration/   # 集成测试
│   ├── e2e/           # 端到端测试（需要外部服务）
│   └── fixtures/      # 测试固件
├── docs/              # 文档
└── .github/workflows/ # CI/CD
```

## 测试

### 单元测试

```bash
# 运行所有测试（默认排除 e2e）
pytest

# 带覆盖率运行
pytest --cov=src/markit --cov-report=html
```

### 集成测试

```bash
# 运行集成测试
pytest tests/integration/
```

### E2E 测试（端到端）

```bash
# 运行 e2e 测试（需要 API Key 或本地服务）
pytest -m e2e
```

## 文档同步规则

更新文档时必须同步更新关联文档：
- `README.md` ↔ `docs/README_ZH.md`：内容保持同步
- `CLAUDE.md` ↔ `AGENTS.md`：开发规范保持一致
- `docs/ROADMAP.md`：任务进展及时更新
- `docs/CONTRIBUTING.md` ↔ `docs/CONTRIBUTING_ZH.md`：保持同步

## 代码风格

- 遵循现有代码模式
- 使用 `ruff` 进行代码检查和格式化
- 使用 `mypy` 进行类型检查
- 所有异步 I/O 使用 `anyio`
- 使用 `structlog` 进行带上下文的日志记录
