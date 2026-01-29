# Markitai 架构文档

## 概述

Markitai 是一个专业的 Markdown 转换器，原生支持 LLM 增强功能。本文档描述了整体架构、模块依赖关系和关键设计决策。

---

## 系统架构

```
                                    +------------------------------------------+
                                    |              CLI Layer                    |
                                    |           cli/main.py                     |
                                    |  (Click commands, argument parsing)       |
                                    +--------------------+---------------------+
                                                         |
                    +------------------------------------+------------------------------------+
                    |                                    |                                    |
                    v                                    v                                    v
        +-------------------+           +-------------------+           +-------------------+
        |   Single File     |           |   Batch Process   |           |   URL Fetch       |
        |   workflow/       |           |   batch.py        |           |   fetch.py        |
        |   single.py       |           |   (Concurrency)   |           |   (4 strategies)  |
        +---------+---------+           +---------+---------+           +---------+---------+
                  |                               |                               |
                  +-------------------------------+-------------------------------+
                                                  |
                                                  v
                              +------------------------------------------+
                              |           Converter Layer                |
                              |           converter/                     |
                              |  +----------+----------+----------+      |
                              |  | PDF      | Office   | Image    |      |
                              |  | Legacy   | Text     | ...      |      |
                              |  +----------+----------+----------+      |
                              +--------------------+---------------------+
                                                   |
                                                   v
                              +------------------------------------------+
                              |           LLM Enhancement                |
                              |           llm/processor.py               |
                              |  (LiteLLM Router, cost tracking)         |
                              +--------------------+---------------------+
                                                   |
                    +------------------------------+------------------------------+
                    |                              |                              |
                    v                              v                              v
        +-------------------+       +-------------------+       +-------------------+
        |   Claude Agent    |       |   Copilot SDK     |       |   LiteLLM Native  |
        |   providers/      |       |   providers/      |       |   (100+ models)   |
        |   claude_agent.py |       |   copilot.py      |       |                   |
        +-------------------+       +-------------------+       +-------------------+
```

---

## 模块依赖关系

```
cli/
├── main.py                    # 主 CLI 入口
│   ├── config.py              # 配置管理
│   ├── batch.py               # 批量处理
│   │   └── workflow/single.py
│   ├── workflow/single.py     # 单文件处理
│   │   ├── converter/*        # 格式转换器
│   │   ├── llm/processor.py   # LLM 增强
│   │   └── image.py           # 图像处理
│   └── fetch.py               # URL 抓取
└── commands/                  # 子命令组
    ├── config.py              # config 子命令
    └── cache.py               # cache 子命令

llm/
├── __init__.py                # 包导出
├── processor.py               # LLMProcessor 类
├── types.py                   # 类型定义
├── cache.py                   # 缓存类
└── models.py                  # 模型工具函数

providers/
├── __init__.py                # 提供商注册
├── claude_agent.py            # Claude Code CLI 提供商
└── copilot.py                 # GitHub Copilot CLI 提供商

workflow/
├── __init__.py                # 工作流导出
├── core.py                    # 核心转换逻辑
├── single.py                  # 单文件处理
└── helpers.py                 # 辅助函数

utils/
├── __init__.py                # 工具导出
├── cli_helpers.py             # CLI 工具函数
├── executor.py                # 线程池执行器
├── mime.py                    # MIME 类型检测
├── office.py                  # Office 软件检测
├── output.py                  # 输出路径解析
├── paths.py                   # 路径工具
├── progress.py                # 进度报告器
└── text.py                    # 文本处理工具
```

---

## 核心模块

### 1. CLI 层 (`cli/`)

**职责**: 命令行界面、参数解析、用户交互

**核心组件**:
- `main.py`: 使用 Click 框架的主入口
- `commands/config.py`: 配置管理命令
- `commands/cache.py`: 缓存管理命令

**设计决策**:
- 使用 Click 框架实现命令分组
- 支持文件、目录和 URL 三种输入类型
- 提供丰富的选项和预设配置

### 2. 配置系统 (`config.py`)

**职责**: 配置加载、验证、合并

**优先级顺序**:
```
1. CLI 参数（最高）
2. 环境变量
3. 配置文件（markitai.json）
4. 默认值（最低）
```

**核心类**:
- `MarkitaiConfig`: 根配置模型
- `LLMConfig`: LLM 相关设置
- `OutputConfig`: 输出设置
- `PromptsConfig`: Prompt 模板
- `FetchConfig`: URL 抓取设置
- `BatchConfig`: 批量处理设置
- `CacheConfig`: 缓存设置

**功能特性**:
- Pydantic v2 验证
- `env:VAR_NAME` 语法支持环境变量引用
- JSON Schema 自动生成
- ConfigManager 单例管理

### 3. 转换器层 (`converter/`)

**职责**: 将各种文件格式转换为 Markdown

**支持格式**:
| 转换器 | 格式 | 依赖 |
|--------|------|------|
| `pdf.py` | PDF | pymupdf4llm |
| `office.py` | PPTX, DOCX, XLSX | markitdown, python-pptx |
| `legacy.py` | DOC, XLS, PPT | LibreOffice 或 pywin32 |
| `image.py` | PNG, JPG, WebP | rapidocr, opencv |
| `text.py` | TXT, MD | - |

**设计模式**: 模板方法
```python
class BaseConverter(ABC):
    supported_formats: list[FileFormat] = []

    @abstractmethod
    def convert(
        self, input_path: Path, output_dir: Path | None = None
    ) -> ConvertResult:
        pass

    def can_convert(self, path: Path | str) -> bool:
        return detect_format(path) in self.supported_formats
```

**注册机制**:
```python
@register_converter(FileFormat.PDF)
class PDFConverter(BaseConverter):
    ...
```

### 4. LLM 集成 (`llm/`)

**职责**: LLM 调用、成本跟踪、缓存管理

**包结构**:
```
llm/
├── __init__.py      # 统一导出
├── processor.py     # LLMProcessor 类
├── types.py         # LLMResponse, ImageAnalysis, Frontmatter 等
├── cache.py         # SQLiteCache, PersistentCache, ContentCache
└── models.py        # 模型信息、成本计算、日志
```

**核心类**:
- `LLMProcessor`: 使用 LiteLLM Router 的主处理器
- `LLMRuntime`: 全局并发控制（信号量共享）
- `SQLiteCache`: 持久化 LRU 缓存（基于 SQLite）
- `PersistentCache`: 全局缓存包装器（支持模式跳过）
- `ContentCache`: 内存 TTL LRU 缓存

**类型定义** (`types.py`):
- `LLMResponse`: LLM 响应数据
- `ImageAnalysis`: 图像分析结果
- `Frontmatter`: 文档元数据
- `DocumentProcessResult`: 文档处理结果

**成本跟踪**:
- Token 计数（tiktoken / 字符估算）
- 成本计算（基于模型定价）
- 累计统计报告

### 5. 提供商系统 (`providers/`)

**职责**: 自定义 LLM 提供商集成

**架构设计**:
```python
# 使用 LiteLLM CustomLLM 接口
def register_providers() -> None:
    """注册所有自定义提供商到 LiteLLM"""
    litellm.custom_provider_map.append({
        "provider": "claude-agent",
        "custom_handler": ClaudeAgentProvider()
    })
```

**支持的提供商**:
- `claude-agent/*`: Claude Code CLI（订阅制，无额外 API 费用）
  - 别名: `sonnet`, `opus`, `haiku`, `inherit`
  - 动态版本解析：自动查找 LiteLLM 数据库中的最新版本
- `copilot/*`: GitHub Copilot CLI（订阅制）
  - 直接使用模型名: `gpt-4.1`, `claude-sonnet-4.5`, `gemini-2.5-pro` 等

**辅助函数**:
- `count_tokens()`: Token 计数（tiktoken 或估算）
- `calculate_copilot_cost()`: Copilot 成本估算
- `validate_local_provider_deps()`: 依赖检查
- `get_local_provider_model_info()`: 获取模型信息

### 6. URL 抓取 (`fetch.py`)

**职责**: Web 内容获取

**策略模式**:
```
AUTO（默认）
├── STATIC: 静态 HTML（requests + BeautifulSoup）
├── BROWSER: 动态渲染（agent-browser）
└── JINA: Jina Reader API
```

**SPA 检测与缓存**:
- JavaScript 框架检测（React, Vue, Angular 等）
- 域名 SPA 标记持久化
- 智能策略选择

**配置选项**:
- `strategy`: 抓取策略（auto/static/browser/jina）
- `fallback_patterns`: 回退策略的 URL 模式匹配
- `agent_browser`: 浏览器自动化配置
- `jina`: Jina API 配置

### 7. 批量处理 (`batch.py`)

**职责**: 并发多文件处理

**功能特性**:
- 可配置并发数（默认 10）
- 断点续传支持（状态持久化）
- 实时进度显示（Rich Live）
- 错误隔离（单文件失败不影响整体）

**状态管理**:
- `FileState`: 文件处理状态
- `UrlState`: URL 处理状态
- `BatchState`: 批量任务状态
- 定期状态刷新（防止数据丢失）

**状态转换**:
```
PENDING -> IN_PROGRESS -> COMPLETED
                       -> FAILED
```

### 8. 工作流层 (`workflow/`)

**职责**: 统一的文档转换流程

**组件**:
- `core.py`: 核心转换逻辑和上下文
- `single.py`: 单文件处理流程
- `helpers.py`: 辅助函数（frontmatter 合并、LLM usage 统计等）

**ConversionContext 数据类**:
```python
@dataclass
class ConversionContext:
    input_path: Path
    output_dir: Path
    config: MarkitaiConfig
    shared_processor: LLMProcessor | None = None
    # ... 处理状态和跟踪字段
```

---

## 数据流

### 单文件处理流程

```
Input File
    |
    v
+---------------+
| Format Detect |
+-------+-------+
        |
        v
+---------------+
| Select        |
| Converter     |
+-------+-------+
        |
        v
+---------------+
| Convert       |
| Content       |
+-------+-------+
        |
        v
+---------------+     +---------------+
| LLM Enhance?  |---->| LLM Process   |
+-------+-------+     +-------+-------+
        |                     |
        v                     v
+----------------------------------+
|        Output Markdown           |
+----------------------------------+
```

### URL 抓取流程

```
URL Input
    |
    v
+---------------+
| Strategy      |
| Selection     |
| (AUTO)        |
+-------+-------+
        |
        +---> STATIC ---> requests ---> HTML
        |
        +---> BROWSER ---> agent-browser ---> Rendered HTML
        |
        +---> JINA ---> Jina API ---> Markdown
        |
        v
+---------------+
| Content       |
| Extraction    |
+-------+-------+
        |
        v
+---------------+
| Markdown      |
| Conversion    |
+---------------+
```

---

## 关键设计决策

### 1. 为什么选择 LiteLLM？

**决策**: 使用 LiteLLM 作为 LLM 网关

**原因**:
- 统一 API 接口（支持 100+ 模型）
- 内置重试和错误处理
- 成本跟踪和 Token 计数
- 路由和负载均衡

### 2. 为什么选择 Pydantic？

**决策**: 使用 Pydantic v2 进行配置管理

**原因**:
- 运行时类型验证
- 自动生成 JSON Schema
- 环境变量集成
- 清晰的错误信息

### 3. 为什么支持本地提供商？

**决策**: 创建 LocalProviderWrapper 集成 Claude/Copilot CLI

**原因**:
- 利用现有订阅（无额外 API 费用）
- 本地认证（无需管理 API 密钥）
- 与 LiteLLM 生态系统集成

### 4. 为什么使用策略模式处理 URL 抓取？

**决策**: 实现 AUTO -> STATIC -> BROWSER -> JINA 策略链

**原因**:
- 不同站点需要不同策略
- 静态抓取速度快、成本低
- 浏览器渲染处理 SPA
- Jina API 作为后备方案

### 5. 为什么进行模块重构？

**决策**: 将 `llm.py` 和 `cli.py` 拆分为包

**原因**:
- 更好的代码组织（关注点分离）
- 提高可测试性
- 更易于维护和扩展
- 清晰的依赖边界

---

## 扩展点

### 添加新的转换器

```python
# converter/new_format.py
from markitai.converter.base import BaseConverter, FileFormat, register_converter

@register_converter(FileFormat.NEW)
class NewFormatConverter(BaseConverter):
    supported_formats = [FileFormat.NEW]

    def convert(self, input_path: Path, output_dir: Path | None = None) -> ConvertResult:
        # 实现转换逻辑
        return ConvertResult(markdown=content)
```

### 添加新的 LLM 提供商

```python
# providers/new_provider.py
from litellm.llms.custom_llm import CustomLLM

class NewProvider(CustomLLM):
    def completion(self, model: str, messages: list, **kwargs):
        # 实现调用逻辑
        return ModelResponse(...)

# 在 providers/__init__.py 中注册
litellm.custom_provider_map.append({
    "provider": "new-provider",
    "custom_handler": NewProvider()
})
```

---

## 性能考量

### 并发控制

- 批量处理默认 10 个并发任务
- Windows 限制为 4（线程切换开销）
- 通过配置可调整
- LLMRuntime 共享信号量控制 LLM 并发

### 缓存策略

- LLM 结果缓存（SQLite，基于内容哈希）
- SPA 域名缓存（持久化）
- HTTP 条件请求（ETag/Last-Modified）
- 模式匹配跳过缓存

### 内存管理

- 大文件流式处理
- 及时释放图像缓冲区
- 单文件大小限制（500MB）
- 线程池执行器（转换器隔离）

---

## 安全考量

- 路径遍历保护（`validate_path_within_base`）
- 符号链接检查（`check_symlink_safety`）
- 文件大小限制
- 原子文件写入（`atomic_write_text`, `atomic_write_json`）
- 敏感信息过滤
- Glob 模式转义（`escape_glob_pattern`）

---

## 相关文档

- [CLI 参考](../website/guide/cli.md)
- [配置指南](../website/guide/configuration.md)
- [快速入门](../website/guide/getting-started.md)
