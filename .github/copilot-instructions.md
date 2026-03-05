# Markitai

带有原生 LLM 增强能力的 Markdown 转换器。支持 DOCX、PDF、PPTX、XLSX、图片、URL 等格式转 Markdown，可选 LLM 清洗、元数据生成和图片分析。

## 原则

- **TDD**：处理问题、新增需求，严格遵循 TDD 原则，请务必调用 `test-driven-development` 技能
- **测试行为而非实现**：重构内部逻辑时测试代码应可以完全不改
- **Mock 系统边界**：Mock 网络、第三方 API、文件 I/O；不对内部层级做 mock
- **可选依赖优雅降级**：所有 SDK 通过 `importlib.util.find_spec()` 运行时检查，缺失时跳过注册而非报错
- **原子性写入**：关键文件写入使用 `security.py` 的原子操作（临时文件 + `os.replace()`）
- **跨平台兼容**：路径操作使用 `pathlib.Path`；Windows 文件锁重试；子进程命令引号包裹路径

## 项目结构

```
packages/markitai/
├── src/markitai/
│   ├── cli/                # Click CLI
│   │   ├── main.py         # 入口 (MarkitaiGroup, 全局选项, .env 加载)
│   │   ├── commands/       # 子命令 (cache, config, doctor, init)
│   │   ├── processors/     # 处理器 (batch, file, url, llm, validators)
│   │   ├── framework.py    # CLI 基础框架
│   │   ├── hints.py        # 错误提示与建议
│   │   ├── interactive.py  # 交互模式向导
│   │   └── console.py, ui.py, i18n.py, logging_config.py
│   ├── converter/          # 格式转换器
│   │   ├── base.py         # BaseConverter 抽象类, ConvertResult, 注册机制
│   │   ├── pdf.py          # PDF (pymupdf4llm, OCR, LLM vision)
│   │   ├── office.py       # DOCX/PPTX/XLSX (markitdown)
│   │   ├── image.py        # 图片直传
│   │   ├── text.py         # 纯文本
│   │   ├── cloudflare.py   # Cloudflare Workers AI
│   │   └── legacy.py       # 旧格式兜底
│   ├── llm/                # LLM 集成
│   │   ├── processor.py    # LLMProcessor, HybridRouter, 路由逻辑
│   │   ├── cache.py        # PersistentCache (SQLite, LRU)
│   │   ├── content.py      # ContentProtection (代码块保护, 智能截断)
│   │   ├── document.py     # DocumentMixin (分页批处理, Vision 增强)
│   │   ├── vision.py       # VisionMixin (alt text, 描述, OCR)
│   │   ├── models.py       # Pydantic 模型
│   │   └── types.py        # 类型定义
│   ├── providers/          # 自定义 LLM 提供商 (插件式)
│   │   ├── __init__.py     # register_providers(), 模型解析, 公共 API
│   │   ├── claude_agent.py # claude-agent/* (Claude Agent SDK)
│   │   ├── copilot.py      # copilot/* (GitHub Copilot SDK)
│   │   ├── chatgpt.py      # chatgpt/* (OAuth Device Code + Responses API)
│   │   ├── gemini_cli.py   # gemini-cli/* (Gemini CLI OAuth)
│   │   ├── auth.py         # AuthManager 单例, AuthStatus, 凭据检查
│   │   ├── errors.py       # 错误层次: ProviderError -> Auth/Quota/Timeout/SDK
│   │   ├── timeout.py      # 自适应超时计算
│   │   ├── json_mode.py    # StructuredOutputHandler (JSON 提取/验证)
│   │   └── common.py       # sync_completion(), has_images()
│   ├── prompts/            # LLM 提示词模板
│   ├── utils/              # 工具模块
│   │   ├── executor.py     # 共享 ThreadPoolExecutor (CPU 密集型任务)
│   │   ├── frontmatter.py  # YAML frontmatter 生成
│   │   ├── mime.py, output.py, paths.py, progress.py, text.py
│   │   ├── cli_helpers.py  # CLI 辅助
│   │   └── office.py       # LibreOffice 集成
│   ├── workflow/           # 转换流水线
│   │   ├── core.py         # ConversionContext, 主流水线编排
│   │   ├── single.py       # 单文件处理
│   │   └── helpers.py      # 流水线辅助
│   ├── config.py           # Pydantic v2 配置模型, env:VAR_NAME 展开
│   ├── constants.py        # 全局常量与默认值
│   ├── security.py         # 原子写入, 路径校验, 符号链接检测
│   ├── batch.py            # BatchState, 断点续处理, 并发控制
│   ├── fetch.py            # URL 抓取 (策略模式, 自动检测, 内容校验)
│   ├── fetch_policy.py     # FetchPolicyEngine (域名画像, 策略排序)
│   ├── fetch_http.py       # 可插拔 HTTP 后端 (httpx / curl-cffi)
│   ├── fetch_playwright.py # Playwright 封装
│   ├── image.py            # 图片处理与压缩
│   ├── ocr.py              # RapidOCR 集成
│   └── types.py            # 全局类型定义
├── tests/
│   ├── unit/               # 单元测试
│   ├── integration/        # 集成测试
│   ├── fixtures/           # 测试数据
│   └── conftest.py         # 共享 fixtures
```

## 技术栈

- **Python**: 3.11-3.13 (target: 3.13)
- **构建**: uv workspace + hatchling
- **CLI**: Click
- **LLM**: LiteLLM (路由/网关) + Instructor (结构化输出)
- **PDF**: pymupdf4llm
- **Office**: markitdown (Microsoft)
- **OCR**: rapidocr
- **浏览器**: Playwright (可选)
- **校验**: Pydantic v2
- **日志**: Loguru
- **终端 UI**: Rich
- **安全扫描**: Bandit
- **Pre-commit**: ruff check, ruff format, pyright, bandit

## 开发命令

```bash
# 安装
uv sync                    # 核心依赖
uv sync --all-extras       # 所有可选依赖

# 测试
uv run pytest                              # 默认: 并行 + 跳过 slow/network
uv run pytest -m "not slow and not network" # 等价于默认 addopts
uv run pytest -m ""                        # 全部测试 (含 slow/network)
uv run pytest --cov=markitai               # 带覆盖率 (fail_under=40)
uv run pytest -n auto                      # 并行执行 (已在默认 addopts)

# 代码质量
uv run ruff check --fix    # lint + 自动修复
uv run ruff format         # 格式化
uv run pyright             # 类型检查 (basic mode)
uv run bandit -r packages/markitai/src -c pyproject.toml  # 安全扫描

# Pre-commit hooks (CI 等价)
uv run pre-commit install
uv run pre-commit run --all-files
```

## 代码风格

- **格式化/Lint**: Ruff (line-length: 88, 双引号, 4 空格缩进)
- **类型检查**: Pyright basic mode; 可选依赖报 warning 不报 error
- **类型注解**: 所有函数必须有类型注解。`str | None`（不用 `Optional[str]`）
- **文件头**: 每个 `.py` 文件第一行 `from __future__ import annotations`
- **文档字符串**: Google style（公开函数必须有）
- **导入顺序**: Ruff isort 管理（`from __future__` -> 标准库 -> 第三方 -> `TYPE_CHECKING` 块 -> 本地模块）
- **日志格式**: `logger.debug("[ModuleName] message")`，方括号标注模块名
- **安全注释**: Bandit 误报用 `# nosec BXXX` 抑制，需注明原因

## 架构

### 配置模型 (config.py)

- **Pydantic v2** 模型，字段带 `Field(description=...)`
- **env:VAR_NAME** 语法支持环境变量展开（惰性解析）
- **weight 语义**: `weight=0` 禁用模型（排除出路由）；`weight>0` 参与负载均衡
- **优先级**: CLI 参数 > 环境变量 > 配置文件 (`markitai.json`) > `constants.py` 默认值

### 核心流水线 (workflow/core.py)

```
ConversionContext → validate_and_detect_format → initialize_converter
→ convert_to_markdown → extract_and_process_images
→ optional_llm_enhancement → generate_frontmatter → write_output
```

- `ConversionContext` 持有全部处理状态和中间结果
- 每步返回 `ConversionStepResult`（success/error/skip_reason），首错即停
- CPU 密集型任务（PDF、LibreOffice）通过共享 `ThreadPoolExecutor` 执行

### 转换器 (converter/)

- **注册机制**: `@register_converter(FileFormat.PDF)` 装饰器注册
- **统一接口**: `BaseConverter.convert(input_path, output_dir) -> ConvertResult`
- **ConvertResult**: `markdown: str`, `images: list[ExtractedImage]`, `metadata: dict`
- 新增格式只需实现 `BaseConverter` 并注册

### 提供商系统 (providers/)

- **插件式注册**: `register_providers()` 启动时调用，通过 `litellm.custom_provider_map` 注册
- **条件加载**: 每个提供商通过 `importlib.util.find_spec()` 检查 SDK，缺失则跳过
- **4 个自定义提供商**: claude-agent (SDK), copilot (SDK), chatgpt (OAuth), gemini-cli (OAuth)
- **模型解析**: 别名动态解析（如 `claude-agent/sonnet` → LiteLLM 数据库最新版本）

**错误层次**:
```
ProviderError (base, 携带 provider + retryable 属性)
├── AuthenticationError (不可重试, 需用户操作)
├── QuotaError (不可重试, 订阅/配额问题)
├── ProviderTimeoutError (可重试, 携带 timeout_seconds)
└── SDKNotAvailableError (不可重试, 需安装 SDK)
```

**认证**: `AuthManager` 单例缓存认证状态；支持 CLI 认证和环境变量认证两种模式

### LLM 集成 (llm/)

- `LLMProcessor` (processor.py) — 编排所有 LLM 调用
- `HybridRouter` — 在标准 LiteLLM 模型和 4 个本地提供商间路由；weight=0 模型在 Router 创建前过滤
- `DocumentMixin` — 文档处理、Vision 增强、分页批处理
- `VisionMixin` — 图片分析（alt text、描述、OCR）
- `PersistentCache` — SQLite LLM 缓存，LRU 淘汰，100MB 限制
- `ContentProtection` — 代码块保护、智能截断

### URL 抓取 (fetch.py, fetch_policy.py)

- **策略模式**: static (httpx/curl-cffi) → playwright → cloudflare → jina
- **FetchPolicyEngine**: 基于域名特征和历史成功率动态排序策略
- **内容校验门**: 所有策略的结果需通过内容质量校验才被接受
- **自动 SPA 检测**: `JS_REQUIRED_PATTERNS` 识别需要浏览器渲染的站点

### 批处理 (batch.py)

- `BatchState` 跟踪 `FileStatus`: PENDING → IN_PROGRESS → COMPLETED/FAILED
- 状态持久化 `state.json`，报告持久化 `report.json`，支持 `--resume`
- 可配置并发度，按模型和文件跟踪费用

### 安全 (security.py)

- `atomic_write_text/json()` — 临时文件 + `os.replace()` 原子写入（同步/异步版本）
- `validate_path_within_base()` — 路径遍历攻击防护
- `check_symlink_safety()` — 符号链接检测
- `validate_file_size()` — 文件大小校验
- Windows: 文件锁重试（5 次指数退避），temp 文件清理

## 关键常量 (constants.py)

- MAX_DOCUMENT_SIZE: 500 MB
- DEFAULT_MAX_CONTENT_CHARS: 32000（截断阈值）
- DEFAULT_MAX_OUTPUT_TOKENS: 8192
- DEFAULT_LLM_CONCURRENCY: 10
- DEFAULT_BATCH_CONCURRENCY: 10
- DEFAULT_MODEL_WEIGHT: 1（weight=0 禁用模型）
- DEFAULT_PLAYWRIGHT_TIMEOUT: 30000 ms
- DEFAULT_PLAYWRIGHT_EXTRA_WAIT_MS: 5000

## 测试约定

- **标记**: `@pytest.mark.slow`（>10s）、`@pytest.mark.network`（需要网络）
- **默认跳过**: pytest addopts 已配置 `-m "not slow and not network"`
- **异步**: `asyncio_mode = "auto"`，测试和 fixture 直接 async
- **并行**: `-n auto`（pytest-xdist）
- **覆盖率**: `fail_under = 40`；`TYPE_CHECKING` 块排除
- **常用 Fixtures**: `fixtures_dir`, `tmp_output`, `sample_markdown`, `cli_runner`, `llm_config`, `create_test_image`, `mock_llm_response`
- **CLI 测试**: Click 的 `CliRunner`
- **Mock 原则**: 只在系统边界做 mock（网络、文件 I/O、第三方 API），不对内部层级做 mock

## 编码模式

- `asyncio` 处理 I/O 密集型操作；`ThreadPoolExecutor` (`utils/executor.py`) 处理 CPU 密集型（共享实例，避免 executor 泛滥）
- 转换器通过 `@register_converter` 注册，新格式只需实现 `BaseConverter`
- 提供商通过 `litellm.custom_provider_map` 注册，实现 `CustomLLM` 接口
- 每个 `.py` 文件头 `from __future__ import annotations`
- `TYPE_CHECKING` 块解决循环导入（运行时不导入，仅供类型检查器使用）
- Loguru 结构化日志 `logger.debug/warning/error("[Module] msg")`
- 配置不可变：作为参数传递，不在函数内修改
- 错误处理：bug 用异常；可预期失败用 result 对象返回
