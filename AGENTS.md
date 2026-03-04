# Markitai

带有原生 LLM 增强能力的 Markdown 转换器。支持 DOCX、PDF、PPTX、XLSX、图片、URL 等格式转 Markdown，可选 LLM 清洗、元数据生成和图片分析。

## 原则

- 测试先行。处理问题、新增需求，严格遵循 TDD 原则，请务必调用 `test-driven-development` 技能
- 测试行为而非实现细节。重构内部逻辑时测试代码应可以完全不改
- Mock 系统边界（网络、第三方 API），不要对内部层级做 mock
- 覆盖率本身是虚荣指标，但可用它来发现死代码和隐藏 bug

## 项目结构

```
packages/markitai/
├── src/markitai/           # 源代码
│   ├── cli/                # Click CLI (入口: main.py)
│   │   ├── commands/       # 子命令 (cache, config, doctor, init)
│   │   └── processors/     # CLI 处理器 (batch, file, url, llm, validators)
│   ├── converter/          # 格式转换器 (base, pdf, office, image, text, legacy, cloudflare)
│   ├── llm/                # LLM 集成 (processor, cache, content, document, models, vision)
│   ├── providers/          # 自定义 LLM 提供商 (claude-agent, copilot)
│   ├── prompts/            # LLM 提示词模板
│   ├── utils/              # 工具模块 (executor, frontmatter, mime, output, paths, progress, text)
│   └── workflow/           # 转换流水线 (core, single, helpers)
├── tests/
│   ├── unit/               # 单元测试
│   ├── integration/        # 集成测试
│   ├── fixtures/           # 测试数据
│   └── conftest.py         # 共享 fixtures
```

## 技术栈

- **Python**: 3.11-3.13 (target: 3.13)
- **构建**: uv + hatchling
- **CLI**: Click
- **LLM**: LiteLLM (路由/网关) + Instructor (结构化输出)
- **PDF**: pymupdf4llm
- **Office**: markitdown (Microsoft)
- **OCR**: rapidocr
- **浏览器**: Playwright (可选，用于 JS 渲染页面)
- **校验**: Pydantic v2
- **日志**: Loguru
- **终端 UI**: Rich

## 开发命令

```bash
# 安装依赖
uv sync              # 核心依赖
uv sync --all-extras # 所有可选依赖

# 运行测试
uv run pytest                              # 全部测试
uv run pytest -m "not slow and not network" # 仅快速测试
uv run pytest -n auto                      # 并行执行
uv run pytest --cov=markitai               # 带覆盖率

# 代码质量
uv run ruff check          # lint
uv run ruff check --fix    # 自动修复
uv run ruff format         # 格式化
uv run pyright             # 类型检查

# Pre-commit hooks
uv run pre-commit install
```

## 代码风格

- **格式化/Lint**: Ruff (line-length: 88)
- **类型检查**: Pyright (basic mode)
- **类型注解**: 所有函数必须有类型注解。使用 `str | None`（不用 `Optional[str]`），文件头加 `from __future__ import annotations`
- **文档字符串**: Google style
- **导入顺序**: 由 Ruff isort 管理（标准库 -> 第三方 -> 本地模块）

## 架构

### 配置优先级（由高到低）

1. CLI 参数
2. 环境变量（配置中使用 `env:VAR_NAME` 语法）
3. 配置文件（`./markitai.json` 或 `~/.markitai/config.json`）
4. `constants.py` 中的默认值

### 核心流水线 (workflow/core.py)

1. 校验并检测文件格式
2. 初始化对应格式的转换器
3. 转换为 Markdown（使用格式特定转换器）
4. 提取并处理图片
5. 可选 LLM 增强（清洗、截图+LLM、元数据）
6. 生成 YAML frontmatter
7. 写入输出文件（Markdown + 图片资源）

### URL 抓取策略 (fetch.py)

- `static` - 直接 HTTP 请求（最快，默认）
- `playwright` - 无头浏览器（JS 渲染页面）
- `cloudflare` - 云端渲染
- `jina` - Jina Reader API
- `auto` - 自动检测带回退链

策略选择由 `FetchPolicyEngine` (fetch_policy.py) 驱动。

### LLM 集成

- `LLMProcessor` (llm/processor.py) - 编排所有 LLM 调用
- `HybridRouter` - 在标准 LiteLLM 模型和本地提供商 (claude-agent, copilot) 间路由
- `DocumentMixin` (llm/document.py) - 文档处理、Vision 增强、分页批处理
- `VisionMixin` (llm/vision.py) - 图片分析（alt text、描述、OCR）
- `PersistentCache` (llm/cache.py) - 基于 SQLite 的 LLM 结果缓存，LRU 淘汰
- `ContentProtection` (llm/content.py) - 保护代码块、智能截断

### 批处理 (batch.py)

- `BatchState` 跟踪文件/URL 处理状态，支持断点续处理
- `FileStatus`: PENDING -> IN_PROGRESS -> COMPLETED/FAILED
- 状态持久化到 `state.json`，报告持久化到 `report.json`
- 可配置并发度的并发处理
- 按模型和文件跟踪费用

### 安全 (security.py)

- `atomic_write_text/json()` - 原子文件写入（临时文件 + rename）
- `validate_path_within_base()` - 路径遍历攻击防护
- `check_symlink_safety()` - 符号链接检测
- `validate_file_size()` - 文件大小校验
- Windows 文件锁重试逻辑

## 关键常量 (constants.py)

- MAX_DOCUMENT_SIZE: 500 MB
- DEFAULT_MAX_CONTENT_CHARS: 32000（截断阈值）
- DEFAULT_MAX_OUTPUT_TOKENS: 8192
- DEFAULT_LLM_CONCURRENCY: 10
- DEFAULT_BATCH_CONCURRENCY: 10
- DEFAULT_PLAYWRIGHT_TIMEOUT: 30000 ms
- DEFAULT_PLAYWRIGHT_EXTRA_WAIT_MS: 5000

## 测试约定

- **标记**: `@pytest.mark.slow`（>10s 的测试）、`@pytest.mark.network`（需要网络）
- **异步**: pytest 配置中 `asyncio_mode = "auto"`
- **常用 Fixtures**: `fixtures_dir`, `tmp_output`, `sample_markdown`, `cli_runner`, `llm_config`, `create_test_image`, `mock_llm_response`
- **CLI 测试**: 使用 Click 的 `CliRunner`
- **Mock 原则**: 在系统边界（网络、文件 I/O、第三方 API）做 mock，不对内部层级做 mock

## 编码模式

- 广泛使用 `asyncio` 处理 I/O 密集型操作
- Thread pool executor (`utils/executor.py`) 处理 CPU 密集型工作（LibreOffice、pymupdf）
- 基于 `converter/base.py` 的格式无关转换器架构
- 基于 `providers/` 的插件式 LLM 提供商支持
- 每个文件头部加 `from __future__ import annotations`
- Loguru 结构化日志，带模块上下文
