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

# 运行所有测试（排除 e2e 测试）
pytest

# 运行 e2e 测试（需要 API Key 或本地服务）
pytest -m e2e

# 运行单个测试文件
pytest tests/unit/test_pipeline.py

# 运行特定测试
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

---

## 核心设计原则

### 1. 职责分离：程序提取，LLM 清理

这是本项目最重要的架构原则：

| 职责 | 程序（Converter） | LLM（Enhancer） |
|------|------------------|-----------------|
| 文本提取 | ✅ | |
| 图片提取 | ✅ | |
| 版式/格式保留 | ✅ | |
| 识别无效信息 | | ✅ |
| 数据清理 | | ✅ |
| 格式增强 | | ✅ |
| 元数据抽取 | | ✅ |

**禁止事项**：
- 禁止在程序中用正则表达式清理"无效内容"（如图表残留、页眉页脚）
- 禁止在程序中做内容判断（如"这行是否有意义"）
- 如果 LLM 没有清理干净，应该优化 Prompt，而不是用正则补救

**原因**：
- 正则无法理解语义，会误删有效内容
- 用户文档内容不可预测（可能包含 YAML 代码块、数字列表、技术术语等）
- LLM 具备语义理解能力，是清理内容的正确工具

### 2. Prompt 集中管理

所有 LLM Prompt 集中在 `src/markit/config/prompts/` 目录，通过 `PromptConfig.get_prompt()` 加载。

**当前结构**（v0.1.6）：
```
src/markit/config/prompts/
├── __init__.py
├── enhancement_zh.md          # 中文文档增强 prompt
├── enhancement_en.md          # 英文文档增强 prompt
├── enhancement_continuation_zh.md  # 续段 prompt（无 frontmatter）
├── enhancement_continuation_en.md
├── summary_zh.md              # 摘要生成 prompt
├── summary_en.md
├── image_analysis_zh.md       # 图片分析 prompt
└── image_analysis_en.md
```

**加载方式**：
```python
from markit.config.settings import PromptConfig

config = PromptConfig(output_language="zh")
prompt = config.get_prompt("enhancement")  # 自动加载 enhancement_zh.md
```

**自定义 prompt**（优先级从高到低）：
1. `prompt.enhancement_prompt_file` - 指定文件路径
2. `prompt.prompts_dir` - 用户 prompts 目录
3. 内置 package prompts（上述目录）

### 3. 多 Chunk 处理策略

当文档过大需要分块处理时：

```
┌─────────────────────────────────────────────────────────────┐
│                      多 Chunk 处理流程                        │
├─────────────────────────────────────────────────────────────┤
│  Chunk 1  →  LLM (清理 + 提取 entities/topics)  →  结果1     │
│  Chunk 2  →  LLM (清理 + 提取 entities/topics)  →  结果2     │
│  Chunk N  →  LLM (清理 + 提取 entities/topics)  →  结果N     │
├─────────────────────────────────────────────────────────────┤
│                         合并阶段                              │
│  1. 合并清理后的内容                                          │
│  2. 合并所有 entities（去重）                                 │
│  3. 合并所有 topics（去重）                                   │
│  4. 生成最终 frontmatter（只在文档开头）                       │
└─────────────────────────────────────────────────────────────┘
```

**关键点**：
- 每个 chunk 都提取元数据，最后合并去重
- 不要只让第一个 chunk 提取元数据（会丢失后半部分信息）
- Frontmatter 只在最终合并时生成一次

### 4. Prompt 优化优先于代码补救

当 LLM 输出不符合预期时：

| 问题 | 错误做法 | 正确做法 |
|------|---------|---------|
| LLM 在续段生成了 frontmatter | 用正则删除中间的 frontmatter | 优化 Prompt，明确告诉 LLM 不要生成 |
| LLM 没有清理图表残留 | 用正则匹配 "Row 1", "Column 1" 等 | 在 Prompt 中添加具体示例 |
| LLM 标题层级混乱 | 用 ensure_h2_start 后处理 | 在 Prompt 中明确层级规则 |

**原则**：相信 LLM 的能力，通过清晰的指令引导它，而不是用代码"修补"它的输出。

### 5. PPT/演示文稿标题层级规则

PPT 转换时的标题层级应在 Prompt 中明确：

```markdown
- 每页标题：## Slide N / ## 第 N 页
- 页内一级标题：###
- 页内二级标题：####
- 页内三级标题：#####
```

**禁止**：让 `ensure_h2_start` 后处理破坏 LLM 已处理好的层级结构。

### 6. 错误处理策略

对于 API 错误（如 `httpx.ConnectError`、`httpx.ReadError`）：

1. **先分析根因**：是网络问题？代理配置？服务端限流？
2. **增加重试次数是治标不治本**，应该：
   - 优化错误日志，帮助用户定位问题
   - 检查代理配置
   - 考虑增加连接超时时间
   - 提供清晰的错误提示

---

## 架构概述

MarkIt 是一个文档转 Markdown 工具，支持可选的 LLM 增强。代码库采用 src layout 和面向服务的架构，职责分离清晰。

### 核心管道流程

```
输入文件 → FormatRouter → 预处理器 → 转换器 → ImageProcessingService → LLMOrchestrator → OutputManager → 输出
```

1. **FormatRouter** (`src/markit/core/router.py`): 根据文件扩展名路由到对应转换器
   - PDF: 根据配置路由到 pymupdf4llm/pymupdf/pdfplumber
   - 旧格式 (.doc, .ppt, .xls): 添加 OfficePreprocessor 先通过 LibreOffice 转换
   - 现代 Office/HTML: 使用 MarkItDown 转换器

2. **ConversionPipeline** (`src/markit/core/pipeline.py`): 主协调器，负责：
   - 带回退支持的文档转换
   - 委托 ImageProcessingService 处理图片
   - 委托 LLMOrchestrator 处理 LLM 操作
   - 通过 OutputManager 写入输出

### 服务层

- **ImageProcessingService** (`src/markit/services/image_processor.py`): 处理图片格式转换、压缩（通过 oxipng/Pillow）、去重，并准备图片供 LLM 分析

- **LLMOrchestrator** (`src/markit/services/llm_orchestrator.py`): 集中所有 LLM 操作：
  - 管理 ProviderManager 支持多提供商
  - 创建 MarkdownEnhancer 进行文本清理
  - 创建 ImageAnalyzer 处理视觉任务
  - 实现基于能力的路由（文本模型 vs 视觉模型）

- **OutputManager** (`src/markit/services/output_manager.py`): 处理文件写入、冲突解决，生成图片描述 markdown 文件

### LLM 提供商系统

**ProviderManager** (`src/markit/llm/manager.py`): 管理多个 LLM 提供商：
- 延迟初始化（按需验证提供商）
- 基于能力的路由（文本 vs 视觉任务）
- 失败时自动回退
- 并发回退（主模型超时时启动备用模型）
- 按模型成本追踪

**路由策略** (`llm.routing.strategy`)：
- `cost_first`: 优先使用最便宜的模型，失败时回退
- `least_pending`: 综合考虑成本和负载，分散请求
- `round_robin`: 简单轮询，依次使用各模型

**AIMD 限流** (`llm.adaptive`): 按凭证自动调整并发：
- 成功时线性增加并发（Additive Increase）
- 失败时指数降低并发（Multiplicative Decrease）
- 每个凭证独立限流，避免一个 API 问题影响其他

支持的提供商: OpenAI, Anthropic, Gemini, Ollama, OpenRouter（均在 `src/markit/llm/` 模块中）

### 配置系统

设置定义在 `src/markit/config/settings.py`，使用 pydantic-settings：
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

3. **LibreOffice 配置文件池** (`src/markit/converters/libreoffice_pool.py`): 并行转换 .doc/.ppt/.xls 时，使用隔离的 LibreOffice 配置文件目录避免冲突。

4. **图片处理进程池**: 重度图片压缩使用进程池绕过 Python GIL。

---

## 代码风格与规范

### 格式化
- **行长**: 100 字符（由 `ruff` 强制执行）
- **引号**: 首选双引号 `"`
- **缩进**: 4 个空格

### 导入 (Imports)
- **排序**: `isort`（通过 `ruff`）
- **顺序**: 标准库 -> 第三方库 -> 本地库 (`markit`)
- **风格**: 首选绝对导入（例如 `from markit.core import ...`）

### 类型提示 (Type Hinting)
- **严格程度**: 高 (`disallow_untyped_defs = true`)
- **语法**: 使用现代 Python 3.10+ 语法（例如使用 `str | None` 代替 `Optional[str]`，`list[str]` 代替 `List[str]`）
- **库**: Typer CLI 参数使用 `typing.Annotated`

### 命名规范
- **变量/函数**: `snake_case`（例如 `convert_document`, `file_path`）
- **类**: `PascalCase`（例如 `ConversionPipeline`, `MarkitError`）
- **常量**: `UPPER_SNAKE_CASE`（例如 `PROJECT_ROOT`）
- **私有**: 以 `_` 为前缀（例如 `_process_image`）

### 错误处理
- **基础异常**: `MarkitError`（位于 `src/markit/exceptions.py`）
- **继承体系**: 继承 `MarkitError` 以实现特定领域的错误（例如 `ConversionError`, `LLMError`, `ConfigurationError`）
- **模式**: 捕获特定异常；将低级错误（IOError 等）包装在带有上下文的 `MarkitError` 子类中

### 路径处理
- **库**: **始终**使用 `pathlib.Path`。**不要**使用 `os.path`
- **类型**: 类型提示应使用 `Path`（例如 `def process(path: Path) -> None:`）

### 库与框架
- **CLI**: `typer` 用于命令定义，`rich` 用于输出格式化
- **配置**: `pydantic-settings` 和 `pydantic` 模型
- **异步**: `anyio` 用于异步操作
- **日志**: `structlog`

### 文档字符串 (Docstrings)
- **风格**: 首选 Google 风格文档字符串
- **内容**: 简短的摘要行，后跟详细信息，以及 `Args:`、`Returns:` 和 `Raises:`（如果适用）

```python
def convert_document(file_path: Path, output_dir: Path) -> Path:
    """Convert a document to Markdown.

    Args:
        file_path: Absolute path to the source file.
        output_dir: Directory where the output should be saved.

    Returns:
        Path to the generated Markdown file.

    Raises:
        ConversionError: If the file format is not supported or conversion fails.
    """
    ...
```

---

## 测试规范

### 测试框架
- **框架**: `pytest`
- **配置**: `pyproject.toml` 中的 `[tool.pytest.ini_options]`
- **位置**: `tests/` 目录（分为 `unit/`、`integration/` 和 `e2e/`）

### 运行测试

```bash
# 运行单个测试文件
pytest tests/unit/test_cli.py

# 运行特定测试用例
pytest tests/unit/test_cli.py::test_version_command

# 运行匹配关键字的测试
pytest -k "conversion"

# 运行并显示详细输出和简短回溯
pytest -v --tb=short
```

### 关键 Fixtures (`tests/conftest.py`)
- `temp_dir`: 提供临时目录路径（`Path` 对象）
- `sample_text_file`: 创建一个示例 `.txt` 文件
- `sample_markdown_file`: 创建一个示例 `.md` 文件
- `cleanup_test_outputs`: 每次测试后自动清理 `tests/fixtures/documents/output`
- `documents_dir`: 返回测试文档目录路径 (`tests/fixtures/documents`)

---

## 文档同步规则

**重要**：更新文档时必须同步更新关联文档：

| 主文档 | 关联文档 | 说明 |
|--------|----------|------|
| `README.md` | `docs/README_ZH.md` | 中英文内容保持同步 |
| `docs/CONTRIBUTING.md` | `docs/CONTRIBUTING_ZH.md` | 贡献指南保持同步 |
| `docs/ROADMAP.md` | - | 任务进展及时更新 |

---

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
│   └── fixtures/      # 测试数据
├── docs/              # 文档
└── .github/workflows/ # CI/CD
```
