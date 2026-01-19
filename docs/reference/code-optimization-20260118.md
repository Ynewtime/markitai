# Markit 代码优化报告 (2026-01-18)

## 概述

本次代码审计和优化覆盖了 Markit 项目的核心模块，包括配置管理、LLM 集成、图像处理、安全模块和 CLI 接口。

**项目统计**：
- 源代码：~7,400 行
- 测试代码：~3,600 行
- Python 版本：3.13+

---

## 已完成的修复

### 1. 环境变量缺失处理 (高优先级)

**问题**：`resolve_env_value()` 在环境变量不存在时返回空字符串，可能导致静默失败。

**位置**：`config.py:13-18`

**修复**：
```python
# 修复前
def resolve_env_value(value: str) -> str:
    if value.startswith("env:"):
        return os.environ.get(env_var, "")  # 返回空字符串
    return value

# 修复后
class EnvVarNotFoundError(ValueError):
    """环境变量未找到异常"""
    pass

def resolve_env_value(value: str, strict: bool = True) -> str | None:
    if value.startswith("env:"):
        env_value = os.environ.get(env_var)
        if env_value is None:
            if strict:
                raise EnvVarNotFoundError(env_var)
            return None
        return env_value
    return value
```

**影响**：配置 `api_key: "env:MISSING_VAR"` 时会立即报错，便于调试。

---

### 2. Instructor 使用量追踪修复 (高优先级)

**问题**：`ImageProcessor._analyze_with_instructor()` 使用 `create()` 方法只返回 Pydantic 模型，没有原始 API 响应，导致无法获取 token 使用量和成本信息。批处理报告中所有 `llm_cost_usd` 为 0。

**位置**：`llm.py:283-310`

**修复**：
```python
# 修复前
response = await client.chat.completions.create(
    model=model,
    messages=messages,
    response_model=ImageAnalysisResult,
)
# response 是 Pydantic 模型，没有 usage 信息

# 修复后
(response, raw_response) = await client.chat.completions.create_with_completion(
    model=model,
    messages=messages,
    response_model=ImageAnalysisResult,
    max_retries=2,
)
# raw_response 包含完整的 API 响应，含 usage 信息
if hasattr(raw_response, "usage") and raw_response.usage is not None:
    input_tokens = getattr(raw_response.usage, "prompt_tokens", 0) or 0
    output_tokens = getattr(raw_response.usage, "completion_tokens", 0) or 0
    cost = completion_cost(completion_response=raw_response)
    self._track_usage(model, input_tokens, output_tokens, cost)
```

**验证结果**：
- 批处理 7 个文件，总成本正确显示为 $0.1062
- 各文件成本细分准确记录

---

### 3. 图片处理输出分离修复 (高优先级)

**问题**：独立图片处理时，`.md` 和 `.llm.md` 文件内容几乎相同。正确行为应该是：
- `.md`：简洁占位符（仅标题和图片引用，保持原始 alt 文本）
- `.llm.md`：完整的 LLM 分析内容（包含 LLM 生成的 alt 文本和详细描述）

**位置**：`cli.py:analyze_images_with_llm()` 1116-1149 行

**修复**：
```python
# 修复前 - alt 文本更新影响了 .md 文件
if alt_enabled:
    markdown = re.sub(old_pattern, new_ref, markdown)  # 所有情况都更新

# 修复后 - 独立图片不更新返回的 markdown
# Check if this is a standalone image file
is_standalone_image = (
    input_path is not None
    and input_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
    and len(image_paths) == 1
)

# IMPORTANT: For standalone images, do NOT update the base markdown
if alt_enabled and not is_standalone_image:
    markdown = re.sub(old_pattern, new_ref, markdown)
```

**验证结果**：
```markdown
# candy.JPG.md (保持原始占位符)
---
title: candy
source: candy.JPG
---
# candy
![candy](assets/candy.JPG)   <-- 原始 alt 文本

# candy.JPG.llm.md (LLM 分析)
---
title: candy
description: Curious cat gazing outdoors
tags: [image, analysis]
---
# candy
![Curious cat gazing outdoors](assets/candy.JPG)   <-- LLM 生成的 alt 文本
## Scene Overview
This image captures a curious cat standing on a ledge...
```

---

### 4. 语言检测增强 (高优先级)

**问题**：英文文档生成了中文的 frontmatter（title/description/tags）。

**原因**：提示词本身是中文的，模型倾向于使用中文输出。

**位置**：`llm.py:_detect_language()` 和 `prompts/*.md`

**修复**：
1. 在 `LLMProcessor` 中添加语言检测方法：
```python
def _detect_language(self, content: str) -> str:
    """Detect language: returns 'English' or 'Chinese'."""
    cjk_count = sum(1 for c in content if "\u4e00" <= c <= "\u9fff")
    total = sum(1 for c in content if c.isalpha())
    return "Chinese" if total > 0 and cjk_count / total > 0.1 else "English"
```

2. 更新提示词，在开头注入语言要求：
```markdown
**⚠️ CRITICAL LANGUAGE RULE: Output language = {language}**
If English → ALL output (title/description/tags) must be English.
If Chinese → 所有输出必须使用中文。
```

**验证结果**：
```yaml
# 英文文档现在正确输出英文 frontmatter
title: Data Overview
description: This document presents a detailed overview...
tags: [data, overview, statistics, demographics, table]
```

---

### 5. LLM 缓存哈希粒度 (中优先级 - 已修复)

**问题**：`ContentCache._compute_hash()` 只取前 500/2000 字符，可能导致哈希冲突。

**位置**：`llm.py:206-211`

**修复**：
```python
# 修复前
combined = f"{prompt[:500]}|{content[:2000]}"

# 修复后
combined = f"{prompt}|{content}"
```

**影响**：缓存键更精确，消除潜在的缓存冲突。

---

### 5. 异步文件 I/O 支持 (低优先级 - 新增)

**新增功能**：`security.py`

```python
async def atomic_write_text_async(path: Path, content: str, encoding: str = "utf-8") -> None:
    """异步原子写入文本文件"""

async def atomic_write_json_async(path: Path, obj: Any, indent: int = 2) -> None:
    """异步原子写入 JSON 文件"""

async def write_bytes_async(path: Path, data: bytes) -> None:
    """异步写入二进制文件"""
```

**用途**：批量处理时可利用 aiofiles 并发写入文件。

---

### 6. 图像处理并发优化 (低优先级 - 新增)

**新增功能**：`image.py`

```python
async def process_and_save_async(
    self,
    images: list[tuple[str, str, bytes]],
    output_dir: Path,
    base_name: str,
    max_concurrency: int = 4,
) -> ImageProcessResult:
    """并发处理和保存图像"""
```

**优化策略**：
1. CPU 密集型处理（解码、压缩）保持顺序执行
2. I/O 密集型操作（写入磁盘）并发执行
3. 信号量控制并发数

---

### 7. 日志处理器生命周期管理 (低优先级 - 新增)

**新增功能**：`cli.py`

```python
class LoggingContext:
    """日志处理器上下文管理器"""

    def suspend_console(self) -> "LoggingContext":
        """暂停控制台日志"""

    def __enter__(self) -> "LoggingContext":
        """移除控制台处理器"""

    def __exit__(self, ...):
        """恢复控制台处理器"""
```

**用途**：在批量处理时，避免日志输出与 Rich 进度条冲突。

---

## 待改进事项

### 1. Workflow 模块重构

**现状**：`cli.py` 中的 `process_with_llm()` 和 `analyze_images_with_llm()` 与 `workflow/single.py` 中的 `SingleFileWorkflow` 有代码重复。

**建议**：
- 将 CLI 中的处理逻辑迁移到 `SingleFileWorkflow` 类
- CLI 仅负责参数解析和调度
- 减少代码重复，提高可维护性

**优先级**：中

---

### 2. 类型标注完善

**现状**：部分函数参数和返回值类型标注不完整，Pyright 报告多处警告。

**建议**：
- 修复 `llm.py` 中 LiteLLM 相关的类型问题
- 使用 `typing.cast()` 处理第三方库类型不完整的情况

**优先级**：低

---

### 3. 配置 Schema 自动生成

**现状**：`config.schema.json` 手工维护，需要与 Pydantic 模型保持同步。

**建议**：
- 使用 Pydantic 的 `model_json_schema()` 自动生成
- 在 CI 中添加 schema 同步检查

**优先级**：低

---

### 4. LLM 结果持久化缓存

**现状**：LLM 缓存仅在内存中，重启后失效。

**建议**：
- 可选的磁盘缓存（基于内容哈希）
- 配置项控制缓存位置和过期时间

**优先级**：低

---

## 架构亮点

### 安全设计
- 路径遍历防护：`validate_path_within_base()`
- Glob 注入防护：`escape_glob_pattern()`
- 文件大小限制：防止 DoS 攻击
- 原子写入：防止部分写入

### 并发控制
- 信号量限制 LLM 并发
- 队列+worker 模式处理图像分析
- 批处理支持恢复机制

### 错误处理
- 可重试错误分类（速率限制、连接错误等）
- 指数退避重试策略
- 优雅降级（LLM 失败返回原始内容）

---

## 测试覆盖

| 模块 | 测试文件 | 测试数量 |
|------|----------|----------|
| LLM | test_llm.py | ~40 |
| 批处理 | test_batch.py | ~30 |
| 配置 | test_config.py | ~25 |
| 安全 | test_security.py | ~20 |
| 图像 | test_image.py | ~15 |
| CLI | test_cli.py | ~17 |

---

## 版本信息

- 优化版本：0.2.0
- 优化日期：2026-01-18
- 审计范围：核心模块（config, llm, image, security, cli, batch, workflow）
