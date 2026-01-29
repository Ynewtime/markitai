# markitai 问题修复任务

> 基于 `logs/markitai_20260126_235833_480871.log` 深度分析
> 更新时间: 2026-01-27 (深度分析 v2)

---

## 任务总览

| # | 问题 | 严重性 | 状态 |
|---|------|--------|------|
| 1 | Browser 打开可见 Terminal 窗口 | Medium | **已修复** ✅ |
| 2 | **Prompt 泄漏到 LLM 输出** | **Critical** | **已修复** ✅ |
| 3 | x.com 超时及错误消息不准确 | High | **已修复** ✅ |
| 4 | max_tokens 超出 deepseek 限制 | High | **已修复** ✅ |
| 5 | 图片下载失败 (外部资源) | Low | 不修复 |

### Issue #2 子任务进度

| 子任务 | Prompt 文件 | 使用方法 | 状态 |
|--------|-------------|----------|------|
| 2.1 | `document_process` | `_process_document_combined` | **已修复** ✅ |
| 2.2 | `cleaner` | `clean_markdown` | **已修复** ✅ |
| 2.3 | `frontmatter` | `generate_frontmatter` | **已修复** ✅ |

**修复详情**:
- 每个 prompt 拆分为 `*_system.md` (角色定义+规则) 和 `*_user.md` (用户内容)
- LLM 调用使用 `[{"role": "system", ...}, {"role": "user", ...}]` 消息结构
- 新增 `_validate_no_prompt_leakage` 函数检测并处理 prompt 泄漏

---

## Issue #1: Browser 打开可见 Terminal 窗口

### 问题描述

Windows 上运行 agent-browser 时会打开单独的 Terminal 窗口，影响用户体验。

### 根因

`fetch.py:730` 的 `asyncio.create_subprocess_exec` 在 Windows 上默认显示控制台窗口。

### 修复方案

```python
# fetch.py:_run_agent_browser_command
import subprocess

kwargs: dict = {
    "stdout": asyncio.subprocess.PIPE,
    "stderr": asyncio.subprocess.PIPE,
}
if sys.platform == "win32":
    kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

proc = await asyncio.create_subprocess_exec(*effective_args, **kwargs)
```

---

## Issue #2: Prompt 泄漏到 LLM 输出 (Critical)

### 问题描述

`concise.llm.md` 的 `cleaned_markdown` 字段包含完整的 prompt 文本，而非处理后的内容。

### 深度根因分析

**核心问题：Prompt 与内容混合在同一个 user message 中**

```python
# llm.py:4040-4042 (当前实现)
messages = cast(
    list[ChatCompletionMessageParam],
    [{"role": "user", "content": prompt}],  # ❌ Prompt 作为 user 内容
)
```

**问题链**：
1. `document_process.md` prompt 包含详细的处理规则（【核心原则】【清理规范】等）
2. Prompt 和文档内容作为单个 user message 传给 LLM
3. LLM 处理时难以区分"指令"和"内容"
4. 在生成 `cleaned_markdown` 时，LLM 可能复制整个输入（包括 prompt）
5. Instructor 只验证 JSON 结构，不验证内容合理性

**为什么某些模型会泄漏 prompt**：
- deepseek-v3.2 在处理长文本时可能会"引用"输入
- 没有明确的 system role 来隔离指令
- Pydantic model 的 Field description 未被充分利用

### 修复方案：拆分 System Prompt 和 User Prompt

**Step 1**: 创建 system prompt 文件

**文件**: `packages/markitai/src/markitai/prompts/document_process_system.md`

```markdown
你是一个专业的 Markdown 文档处理助手。

## 你的任务
1. **格式优化**：清理 Markdown 格式，保持原文语言不变
2. **元数据生成**：提取标题、摘要、标签

## 处理规则
- 禁止翻译：保留原文语言
- 禁止改写：只做格式调整
- 保留代码块、表格、链接、图片语法
- 保留所有 `__MARKITAI_*__` 占位符

## 输出格式
返回 JSON，包含：
- cleaned_markdown: 优化后的 Markdown（只包含文档内容，不要包含任何处理指令）
- frontmatter: { title, description, tags }

重要：cleaned_markdown 必须只包含优化后的文档内容本身，绝对不要包含任何任务说明或 prompt 文本。
```

**Step 2**: 创建 user prompt 文件

**文件**: `packages/markitai/src/markitai/prompts/document_process_user.md`

```markdown
请处理以下文档（使用 {language} 生成元数据）：

源文件: {source}

---

{content}
```

**Step 3**: 修改 `_process_document_combined` 方法

```python
# llm.py:_process_document_combined
async def _process_document_combined(
    self,
    markdown: str,
    source: str,
) -> DocumentProcessResult:
    # ... cache checks ...

    language = get_language_name(detect_language(markdown))
    truncated_content = self._smart_truncate(markdown, DEFAULT_MAX_CONTENT_CHARS)

    # 获取分离的 system 和 user prompt
    system_prompt = self._prompt_manager.get_prompt("document_process_system")
    user_prompt = self._prompt_manager.get_prompt(
        "document_process_user",
        content=truncated_content,
        source=source,
        language=language,
    )

    # 构建消息：分离 system 和 user role
    messages = cast(
        list[ChatCompletionMessageParam],
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    # ... rest of the method ...
```

**Step 4**: 增强 Pydantic model 的 Field descriptions

```python
# llm.py:DocumentProcessResult
class DocumentProcessResult(BaseModel):
    """LLM document processing result."""

    cleaned_markdown: str = Field(
        description=(
            "格式优化后的 Markdown 文档内容。"
            "只包含实际的文档内容，不要包含任何处理指令或 prompt 文本。"
        )
    )
    frontmatter: Frontmatter = Field(
        description="文档元数据：标题、摘要、标签"
    )
```

**Step 5**: 添加输出验证

```python
# llm.py: 在 _process_document_combined 返回前添加
def _validate_no_prompt_leakage(self, cleaned: str, source: str) -> str:
    """检测并处理 prompt 泄漏。"""
    prompt_markers = [
        "## 任务 1:",
        "## 任务 2:",
        "【核心原则】",
        "【清理规范】",
        "请处理以下",
        "你是一个专业的",
    ]

    for marker in prompt_markers:
        if marker in cleaned:
            logger.warning(f"[{source}] Prompt leakage detected, attempting recovery")
            # 尝试提取 "---" 分隔符之后的内容
            if "---" in cleaned:
                parts = cleaned.split("---", 2)
                if len(parts) > 2:
                    return parts[2].strip()
            raise ValueError("LLM returned prompt text in cleaned_markdown")

    return cleaned
```

### 验证方法

```bash
markitai "https://stephango.com/concise" --preset rich --no-cache
# 检查 output/*.llm.md 不应包含 "请处理以下" "【核心原则】" 等 prompt 文本
grep -l "请处理以下\|【核心原则】" output/*.llm.md  # 应无匹配
```

---

## Issue #3: x.com 超时及错误消息不准确

### 问题描述

1. x.com 使用 browser 获取超时 (30000ms)
2. 超时后错误消息建议"安装 agent-browser"，但 browser 实际已安装且尝试过

### 深度根因分析

**超时的根本原因**：

1. **反爬虫检测**：x.com 主动检测自动化浏览器
   - 检测 Playwright/Puppeteer 特征
   - 要求 JavaScript 动态加载内容
   - 可能显示验证页面

2. **等待策略不当**：
   - 当前使用 `wait_for: "domcontentloaded"`
   - x.com 的核心内容通过 GraphQL API 异步加载
   - `domcontentloaded` 触发时内容可能尚未加载

3. **超时时间不足**：
   - 默认 30 秒对社交媒体站点不够
   - x.com 可能需要 45-60 秒

**错误消息不准确的原因**：

```python
# fetch.py:1698-1704
if static_reason in CRITICAL_INVALID_REASONS:
    raise FetchError(
        f"URL requires browser rendering: {url}. "
        f"Reason: {static_reason}. "
        f"Please install agent-browser..."  # ❌ 未区分超时情况
    )
```

代码没有追踪 browser 是否已尝试过以及失败原因。

### 修复方案

**方案 A: 使用 Jina Reader API 作为首选策略（推荐）**

Jina Reader 对社交媒体有特殊优化，无反爬虫问题。

```python
# fetch.py: 添加 Jina 策略优先级
SOCIAL_MEDIA_JINA_PRIORITY = {
    "x.com",
    "twitter.com",
    "threads.net",
}

async def _fetch_with_fallback(
    url: str,
    config: FetchConfig,
    start_with_browser: bool = False,
) -> FetchResult:
    domain = extract_domain(url)

    # 对于社交媒体，优先使用 Jina
    if domain in SOCIAL_MEDIA_JINA_PRIORITY and config.jina.api_key:
        try:
            result = await fetch_with_jina(url, config.jina)
            if result.content and not _is_invalid_content(result.content)[0]:
                return result
        except Exception as e:
            logger.debug(f"Jina fetch failed for {url}: {e}, falling back to browser")

    # ... 继续原有的 browser/static 策略 ...
```

**方案 B: 增加社交媒体的超时和等待策略**

```python
# fetch.py: 社交媒体特殊配置
SOCIAL_MEDIA_BROWSER_CONFIG = {
    "x.com": {"timeout": 60000, "wait_for": "networkidle", "extra_wait_ms": 3000},
    "twitter.com": {"timeout": 60000, "wait_for": "networkidle", "extra_wait_ms": 3000},
    "instagram.com": {"timeout": 45000, "wait_for": "networkidle", "extra_wait_ms": 2000},
}

async def fetch_with_browser(url: str, ...) -> FetchResult:
    domain = extract_domain(url)

    # 使用社交媒体特殊配置
    if domain in SOCIAL_MEDIA_BROWSER_CONFIG:
        override = SOCIAL_MEDIA_BROWSER_CONFIG[domain]
        timeout = override.get("timeout", timeout)
        wait_for = override.get("wait_for", wait_for)
        extra_wait_ms = override.get("extra_wait_ms", extra_wait_ms)

    # ... 继续执行 ...
```

**方案 C: 实现 Nitter fallback（备选）**

```python
# fetch.py: Nitter 作为 Twitter 的 fallback
NITTER_INSTANCES = [
    "nitter.net",
    "nitter.poast.org",
    "nitter.privacydev.net",
]

def _convert_twitter_to_nitter(url: str) -> str:
    """Convert x.com/twitter.com URL to nitter instance."""
    parsed = urlparse(url)
    instance = random.choice(NITTER_INSTANCES)
    return f"https://{instance}{parsed.path}"

# 在 browser 超时后尝试 nitter
if "x.com" in url or "twitter.com" in url:
    try:
        nitter_url = _convert_twitter_to_nitter(url)
        return await fetch_with_static(nitter_url)
    except Exception:
        pass  # nitter 也失败，继续原有错误处理
```

**方案 D: 修复错误消息（必须）**

```python
# fetch.py:_fetch_multi_source 中追踪 browser 状态
browser_attempted = False
browser_timed_out = False
browser_error_msg = ""

# 在 browser 尝试后记录状态
if browser_task and not browser_task.cancelled():
    browser_attempted = True
    try:
        browser_result = await browser_task
    except asyncio.TimeoutError:
        browser_timed_out = True
        browser_error_msg = f"Browser fetch timed out after {timeout}ms"
    except Exception as e:
        browser_error_msg = str(e)

# 修改错误消息
if static_reason in CRITICAL_INVALID_REASONS:
    if browser_timed_out:
        raise FetchError(
            f"URL requires browser rendering: {url}. "
            f"{browser_error_msg}. "
            f"Try: 1) Increase timeout with --fetch-timeout 60000, "
            f"2) Check network connectivity, "
            f"3) Use Jina API with --jina-api-key"
        )
    elif browser_attempted:
        raise FetchError(
            f"URL requires browser rendering: {url}. "
            f"Browser fetch failed: {browser_error_msg}"
        )
    else:
        raise FetchError(
            f"URL requires browser rendering: {url}. "
            f"Reason: {static_reason}. "
            f"Please install agent-browser: npm install -g agent-browser && agent-browser install"
        )
```

### 推荐的实施优先级

1. **立即修复**：错误消息不准确（方案 D）
2. **短期**：启用 Jina API 优先策略（方案 A）
3. **中期**：增加社交媒体超时配置（方案 B）
4. **长期**：实现 Nitter fallback（方案 C）

### 验证方法

```bash
# 测试 x.com，应显示超时错误而非安装提示
markitai "https://x.com/user/status/123" --preset rich --no-cache
# 期望: "Browser fetch timed out" 或 "Try Jina API"
# 不期望: "Please install agent-browser"
```

---

## Issue #4: max_tokens 超出 deepseek 限制

### 问题描述

```
ERROR: Invalid max_tokens value, the valid range of max_tokens is [1, 8192]
```

### 深度根因分析

**问题链**：

1. `_get_router_primary_model()` 返回配置中第一个模型 (`gemini/gemini-2.5-flash-lite`)
2. 代码基于该模型计算 `max_tokens`
3. Router 实际选择了 `openrouter/deepseek/deepseek-v3.2`
4. LiteLLM `get_model_info('deepseek/deepseek-v3.2')` 返回错误的 `max_output_tokens=163840`
5. 实际 API 限制是 `8192`
6. 请求被拒绝

**LiteLLM 模型信息不准确**：

| 模型 | LiteLLM 返回值 | 实际限制 |
|------|---------------|----------|
| `deepseek/deepseek-v3.2` | 163840 | 8192 |
| `deepseek/deepseek-chat` | 8192 | 8192 |
| `openrouter/deepseek/deepseek-v3.2` | (未知) | 8192 |

### 修复方案

**策略 1: 模型限制覆盖表（推荐）**

```python
# llm.py: 已知 LiteLLM 信息不准确的模型
MODEL_MAX_OUTPUT_OVERRIDES = {
    "deepseek/deepseek-v3.2": 8192,
    "openrouter/deepseek/deepseek-v3.2": 8192,
    "openrouter/deepseek/deepseek-chat": 8192,
}

def get_model_info_cached(model: str) -> dict[str, Any]:
    info = _cached_get_model_info(model)

    # 应用已知的覆盖
    if model in MODEL_MAX_OUTPUT_OVERRIDES:
        info = dict(info)  # 创建副本
        info["max_output_tokens"] = MODEL_MAX_OUTPUT_OVERRIDES[model]
        logger.debug(f"[ModelInfo] Applied override for {model}: max_output_tokens={info['max_output_tokens']}")

    return info
```

**策略 2: 使用所有可能模型的最小值**

```python
# llm.py:_calculate_dynamic_max_tokens
def _calculate_dynamic_max_tokens(
    self, messages: list[Any], target_model_id: str | None = None
) -> int | None:
    # 收集所有可能被选中的模型的 max_output_tokens
    all_max_outputs = []
    for model_config in self.router.model_list:
        model_id = model_config.get("litellm_params", {}).get("model")
        if model_id:
            info = get_model_info_cached(model_id)
            max_out = info.get("max_output_tokens")
            if max_out:
                all_max_outputs.append(max_out)

    if not all_max_outputs:
        return None  # 让 LiteLLM 处理

    # 使用最小值确保兼容所有可能被选中的模型
    max_output = min(all_max_outputs)
    logger.debug(f"[DynamicTokens] Using min max_output from all models: {max_output}")

    # ... 继续计算 ...
```

**策略 3: 捕获 max_tokens 错误并重试**

```python
# llm.py:_call_llm_with_retry 中添加
except litellm.BadRequestError as e:
    error_msg = str(e)
    if "max_tokens" in error_msg.lower() and "invalid" in error_msg.lower():
        # 解析错误中的有效范围
        import re
        match = re.search(r'\[(\d+),\s*(\d+)\]', error_msg)
        if match:
            valid_max = int(match.group(2))
            logger.warning(
                f"[LLM:{call_id}] max_tokens exceeded, retrying with {valid_max}"
            )
            # 用更小的 max_tokens 重试
            return await self._call_llm_with_retry(
                model, messages, call_id, context, max_retries=0,
                max_tokens_override=valid_max
            )
    raise
```

### 验证方法

```bash
markitai packages/markitai/tests/fixtures/file_example_XLSX_100.xlsx --preset rich --no-cache
grep "Invalid max_tokens" logs/markitai_*.log  # 应无匹配
```

---

## Issue #5: 图片下载失败 (外部资源)

### 问题描述

```
Failed to download image: https://yenwtime-1255970624.cos.ap-guangzhou.myqcloud.com/JPG/unit.jpg
```

### 分析

- COS bucket URL 不可访问（可能已过期或权限问题）
- **这是外部资源问题，非代码 bug**
- 当前代码已有 fallback 处理，图片下载失败不会阻断整体流程

### 处理

不需要代码修复。

---

## 执行计划

### Phase 1: Critical 修复（并行执行）

| 任务 | 优先级 | 修改文件 |
|------|--------|----------|
| Prompt 拆分 system/user | P0 | `llm.py`, `prompts/*.md` |
| max_tokens 覆盖表 | P0 | `llm.py` |
| 错误消息修复 | P1 | `fetch.py` |
| Terminal 窗口隐藏 | P1 | `fetch.py` |

### Phase 2: 增强修复

| 任务 | 优先级 | 修改文件 |
|------|--------|----------|
| Jina API 优先策略 | P2 | `fetch.py` |
| 社交媒体超时配置 | P2 | `fetch.py`, `constants.py` |
| Nitter fallback | P3 | `fetch.py` |

### Phase 3: 验证测试

```bash
# 完整测试
markitai packages/markitai/tests/fixtures --no-cache --preset rich -o ./output-test --verbose

# 检查项:
# - [ ] 无可见 Terminal 窗口
# - [ ] .llm.md 文件无 prompt 泄漏
# - [ ] 无 max_tokens 错误
# - [ ] x.com 超时显示正确错误消息
```

---

## 已完成的修复（历史）

| 问题 | 修复状态 |
|------|----------|
| agent-browser Windows 执行失败 | **已修复** - 使用 native exe |
| PDF 图片路径错误 | **已修复** - 使用 `as_posix()` |
| Alt text 条件逻辑错误 | **已修复** - `alt_enabled or desc_enabled` |
| JS 站点静默回退 | **已修复** - 严格模式报错 |
| Page marker 恢复逻辑 | **已修复** - 尊重 LLM 结果 |
| Symlink 测试失败 | **已修复** - `@requires_symlink` 装饰器 |

---

## 备注

- 修复顺序按优先级执行
- P0 任务并行执行
- 每个修复完成后运行 ruff/pyright 验证
- 完成后手动提交

---
---

# Windows 性能优化任务

> 基于 `docs/reference/windows-opt-1.md` 深度分析报告
> 创建时间: 2026-01-27
> 预期收益: Windows 批处理性能提升 2-4 倍

---

## 优化任务总览

| # | 任务 | 难度 | 优先级 | 预期收益 | 状态 |
|---|------|------|--------|----------|------|
| W1 | 线程池配置调优 | ⭐ | 🔴 High | -10~20% 切换开销 | ✅ 已完成 |
| W2 | ONNX Runtime 全局单例 + 预热 | ⭐⭐ | 🔴 High | -3~8s 首次调用 | ✅ 已完成 |
| W3 | 图像处理 OpenCV 优化 | ⭐⭐ | 🟡 Medium | CPU 处理提速 20-40% | ✅ 已完成 |
| W4 | asyncio 子进程命令批量化 | ⭐⭐ | 🟡 Medium | 每页面 -200~500ms | ✅ 已完成 |
| W5 | LibreOffice UNO 守护进程模式 | ⭐⭐⭐⭐ | 🟢 Low | 每文件 -2~3s | ⏸️ 推迟 |

---

## W1: 线程池配置调优

### 问题背景

**位置**: `packages/markitai/src/markitai/utils/executor.py` L14-58

当前配置:
```python
_CONVERTER_MAX_WORKERS = min(os.cpu_count() or 4, 8)
```

Windows 线程上下文切换开销约 2-8 μs（Linux 为 1-3 μs），高线程数下差异累积明显。

### 实施方案

**文件**: `packages/markitai/src/markitai/utils/executor.py`

```python
import os
import platform

def _get_optimal_workers():
    cpu_count = os.cpu_count() or 4
    if platform.system() == "Windows":
        # Windows: 降低默认值，减少线程切换开销
        return min(cpu_count, 4)
    else:
        # Linux/macOS: 可以使用更高并发
        return min(cpu_count, 8)

_CONVERTER_MAX_WORKERS = _get_optimal_workers()
```

### 验证方法

```bash
# 运行批处理测试，对比修改前后耗时
markitai packages/markitai/tests/fixtures --preset rich -o ./output-perf-test --verbose
```

### 预期收益

- 减少线程切换开销 10-20%
- Windows 上更稳定的并发性能

---

## W2: ONNX Runtime 全局单例 + 预热

### 问题背景

**位置**: `packages/markitai/src/markitai/ocr.py` L39-85

RapidOCR 基于 ONNX Runtime，冷启动延迟源于:
1. DLL 加载开销（Windows 特有）
2. DirectML/CUDA 初始化
3. 模型加载和图优化

实测影响: CPU 模式 1-3s，DirectML 3-8s，CUDA 5-15s

### 实施方案

**文件**: `packages/markitai/src/markitai/ocr.py`

```python
import threading
import numpy as np

class OCRProcessor:
    _global_engine = None
    _init_lock = threading.Lock()

    @classmethod
    def get_shared_engine(cls, config: OCRConfig | None = None):
        """Get or create global singleton engine (thread-safe)."""
        if cls._global_engine is None:
            with cls._init_lock:
                if cls._global_engine is None:
                    cls._global_engine = cls._create_engine_impl(config)
        return cls._global_engine

    @classmethod
    def preheat(cls, config: OCRConfig | None = None):
        """Preheat engine at application startup."""
        engine = cls.get_shared_engine(config)
        # Execute dummy inference to complete GPU compilation
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        try:
            engine(dummy_image)
        except Exception:
            pass  # Ignore errors from dummy image
        return engine

    @property
    def engine(self):
        """Use shared engine instead of instance engine."""
        return self.get_shared_engine(self.config)
```

### 额外修改

**文件**: `packages/markitai/src/markitai/cli.py` 或 `batch.py`

在批处理模式入口添加预热调用:
```python
if batch_mode and ocr_enabled:
    from markitai.ocr import OCRProcessor
    OCRProcessor.preheat()
```

### 验证方法

```bash
# 首次 OCR 调用不应有明显延迟
markitai packages/markitai/tests/fixtures/image.png --preset rich --verbose
# 检查日志中 OCR 初始化时间
```

### 预期收益

- 消除首次调用 1-8 秒延迟
- 批处理时所有文件共享同一引擎

---

## W3: 图像处理 OpenCV 优化

### 问题背景

**位置**: `packages/markitai/src/markitai/image.py` L37-95 (`_compress_image_worker`)

Pillow 在 Python 层处理，受 GIL 限制。OpenCV 在 C++ 层释放 GIL，更适合多线程。

### 实施方案

#### Step 1: 添加依赖

**文件**: `packages/markitai/pyproject.toml`

```toml
dependencies = [
    # ... existing deps ...
    "opencv-python>=4.8.0",
]
```

#### Step 2: 实现 OpenCV 压缩函数

**文件**: `packages/markitai/src/markitai/image.py`

```python
import cv2
import numpy as np

def _compress_image_cv2(
    image_data: bytes,
    quality: int,
    max_size: tuple[int, int],
    output_format: str = "JPEG",
) -> tuple[bytes, int, int]:
    """Compress image using OpenCV (releases GIL in C++ layer)."""
    # Decode
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image")

    h, w = img.shape[:2]

    # Resize if needed
    if w > max_size[0] or h > max_size[1]:
        scale = min(max_size[0] / w, max_size[1] / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        w, h = new_w, new_h

    # Encode
    if output_format.upper() == "JPEG":
        encode_param = [cv2.IMWRITE_JPEG_QUALITY, quality]
        _, buffer = cv2.imencode('.jpg', img, encode_param)
    elif output_format.upper() == "PNG":
        # PNG compression level 0-9, map quality 0-100 to 9-0
        compression = max(0, min(9, 9 - quality // 11))
        encode_param = [cv2.IMWRITE_PNG_COMPRESSION, compression]
        _, buffer = cv2.imencode('.png', img, encode_param)
    elif output_format.upper() == "WEBP":
        encode_param = [cv2.IMWRITE_WEBP_QUALITY, quality]
        _, buffer = cv2.imencode('.webp', img, encode_param)
    else:
        raise ValueError(f"Unsupported format: {output_format}")

    return buffer.tobytes(), w, h
```

#### Step 3: 修改 worker 函数

```python
def _compress_image_worker(...):
    # 优先使用 OpenCV，失败时回退到 Pillow
    try:
        return _compress_image_cv2(image_data, quality, max_size, output_format)
    except Exception:
        return _compress_image_pillow(image_data, quality, max_size, output_format)
```

### 验证方法

```bash
# 运行图像压缩性能测试
python -c "
from markitai.image import _compress_image_cv2
import time
with open('test.jpg', 'rb') as f:
    data = f.read()
start = time.time()
for _ in range(100):
    _compress_image_cv2(data, 85, (1920, 1080))
print(f'OpenCV: {time.time()-start:.2f}s')
"
```

### 预期收益

- CPU 密集型图像处理提速 20-40%
- 多线程场景下效果更明显

---

## W4: asyncio 子进程命令批量化

### 问题背景

**位置**: `packages/markitai/src/markitai/fetch.py` L645-686

每次 `agent-browser` 命令调用增加约 50-100ms 开销。URL 批量抓取时多次调用（open, wait, snapshot, get）影响累积。

### 实施方案

**文件**: `packages/markitai/src/markitai/fetch.py`

```python
async def _run_agent_browser_batch(
    session: str,
    commands: list[tuple[str, list[str]]],  # [(command, args), ...]
    timeout_seconds: float,
) -> list[tuple[bytes, bytes, int]]:
    """Execute multiple agent-browser commands in batch."""
    # 方案 A: 使用 agent-browser 的 batch/script 功能（如果支持）
    # 方案 B: 合并为单个 shell 脚本执行
    # 方案 C: 使用 agent-browser 的持久连接模式

    # 当前实现: 复用 session，减少浏览器启动开销
    results = []
    for cmd, args in commands:
        full_args = ["--session", session, cmd] + args
        result = await _run_agent_browser_command(full_args, timeout_seconds)
        results.append(result)
    return results
```

**优化 session 复用**:

```python
async def fetch_page_with_browser(url: str, ...) -> BrowserFetchResult:
    session = f"markitai-{hash(url) % 10000}"  # 使用固定 session 名

    # 批量执行命令
    commands = [
        ("open", [url]),
        ("wait", ["--load", "domcontentloaded"]),
        ("snapshot", ["-c", "--json"]),
        ("get", ["title"]),
    ]

    results = await _run_agent_browser_batch(session, commands, timeout)
    # ... parse results ...
```

### 验证方法

```bash
# 批量处理 URL，对比修改前后耗时
markitai "https://example.com" "https://httpbin.org/html" --preset rich --verbose
```

### 预期收益

- 减少 3-5 次子进程创建
- 每页面节省 200-500ms

---

## W5: LibreOffice UNO 守护进程模式

### 问题背景

**位置**:
- `packages/markitai/src/markitai/converter/office.py` L378-402
- `packages/markitai/src/markitai/converter/legacy.py` L517-531

每次调用 LibreOffice 需要:
1. 启动 `soffice.exe`（2-3s）
2. 加载 UNO 运行时
3. 初始化文档处理框架

### 实施方案（需评估可行性）

#### Step 1: 启动 LibreOffice 守护进程

```bash
# Windows
soffice.exe --accept="socket,host=localhost,port=2002;urp;" --headless

# Linux
soffice --accept="socket,host=localhost,port=2002;urp;" --headless &
```

#### Step 2: 实现 UNO 连接池

**新文件**: `packages/markitai/src/markitai/utils/libreoffice_pool.py`

```python
import uno
from com.sun.star.beans import PropertyValue

class LibreOfficePool:
    def __init__(self, port: int = 2002):
        self._port = port
        self._desktop = None
        self._connected = False

    def connect(self):
        """Connect to running LibreOffice instance."""
        if self._connected:
            return

        local_context = uno.getComponentContext()
        resolver = local_context.ServiceManager.createInstanceWithContext(
            "com.sun.star.bridge.UnoUrlResolver", local_context
        )
        ctx = resolver.resolve(
            f"uno:socket,host=localhost,port={self._port};urp;StarOffice.ComponentContext"
        )
        smgr = ctx.ServiceManager
        self._desktop = smgr.createInstanceWithContext(
            "com.sun.star.frame.Desktop", ctx
        )
        self._connected = True

    def convert_to_pdf(self, input_path: str, output_path: str) -> bool:
        """Convert document to PDF using UNO API."""
        if not self._connected:
            self.connect()

        url = uno.systemPathToFileUrl(input_path)
        doc = self._desktop.loadComponentFromURL(url, "_blank", 0, ())

        filter_props = (
            PropertyValue("FilterName", 0, "writer_pdf_Export", 0),
        )
        output_url = uno.systemPathToFileUrl(output_path)
        doc.storeToURL(output_url, filter_props)
        doc.close(True)
        return True

    def close(self):
        """Close connection."""
        if self._desktop:
            try:
                self._desktop.terminate()
            except Exception:
                pass
        self._connected = False
```

### 评估要点

1. **可行性**: UNO API 在 Windows 上是否稳定？
2. **依赖**: 需要额外安装 `uno` 包？（LibreOffice 自带 Python 绑定）
3. **复杂度**: 进程管理、连接重试、错误恢复
4. **收益**: 批处理 10+ 文件时显著，单文件收益有限

### 评估结论（2026-01-27）

**决定：推迟到后续版本实现**

**原因：**

1. **依赖问题**：`uno` 模块是 LibreOffice 自带的 Python 绑定，无法通过 pip 安装。需要配置 `PYTHONPATH` 指向 LibreOffice 安装目录中的 Python 库，或使用 LibreOffice 自带的 Python 解释器。这增加了部署复杂度。

2. **守护进程管理**：需要实现：
   - 程序启动时自动启动 LibreOffice 守护进程
   - 程序退出时清理守护进程
   - 守护进程崩溃时自动重启
   - 超时和健康检查机制

3. **并发访问**：UNO 连接不是线程安全的，需要实现：
   - 连接池管理
   - 请求排队或序列化
   - 连接失效检测和重建

4. **跨平台差异**：Windows、Linux、macOS 上的 LibreOffice 安装路径和 Python 绑定位置不同，需要分别处理。

5. **投入产出比**：当前子进程模式虽然每次启动有 2-3s 开销，但实现简单可靠。UNO 模式的复杂度（⭐⭐⭐⭐）与预期收益不成正比。

**替代方案**：
- 当前已通过 W1-W4 优化获得显著性能提升
- 如需进一步优化 LibreOffice 性能，可考虑使用 Docker 容器预热 LibreOffice 实例

### 验证方法（仅供参考）

```bash
# 启动 LibreOffice 守护进程
soffice.exe --accept="socket,host=localhost,port=2002;urp;" --headless

# 测试 UNO 连接
python -c "
from markitai.utils.libreoffice_pool import LibreOfficePool
pool = LibreOfficePool()
pool.connect()
pool.convert_to_pdf('test.docx', 'test.pdf')
"
```

### 预期收益

- 每文件节省 2-3 秒启动时间
- 批处理 10+ 文件时提速显著

---

## 实施计划

### Phase 1: 快速优化（简单实现）

| 序号 | 任务 | 文件 | 难度 |
|------|------|------|------|
| 1.1 | W1 线程池配置调优 | `utils/executor.py` | ⭐ |
| 1.2 | W2 ONNX 全局单例 | `ocr.py` | ⭐⭐ |
| 1.3 | W2 ONNX 预热调用 | `cli.py` / `batch.py` | ⭐ |

### Phase 2: 依赖升级（OpenCV 集成）

| 序号 | 任务 | 文件 | 难度 |
|------|------|------|------|
| 2.1 | 添加 opencv-python 依赖 | `pyproject.toml` | ⭐ |
| 2.2 | W3 实现 OpenCV 压缩 | `image.py` | ⭐⭐ |
| 2.3 | W3 worker 函数切换 | `image.py` | ⭐ |

### Phase 3: 流程优化（子进程批量化）

| 序号 | 任务 | 文件 | 难度 |
|------|------|------|------|
| 3.1 | W4 agent-browser 命令批量化 | `fetch.py` | ⭐⭐ |
| 3.2 | W4 session 复用优化 | `fetch.py` | ⭐ |

### Phase 4: 高级优化（待评估）

| 序号 | 任务 | 文件 | 难度 |
|------|------|------|------|
| 4.1 | W5 评估 UNO 可行性 | - | ⭐⭐ |
| 4.2 | W5 实现 LibreOffice 连接池 | `utils/libreoffice_pool.py` | ⭐⭐⭐⭐ |
| 4.3 | W5 集成到 converter | `converter/office.py`, `converter/legacy.py` | ⭐⭐⭐ |

---

## 验收标准

### 性能指标

```bash
# 基准测试命令
markitai packages/markitai/tests/fixtures --preset rich -o ./output-benchmark --verbose

# 对比指标:
# - 总处理时间
# - OCR 首次调用延迟
# - 图片压缩耗时
# - URL 抓取耗时
```

### 检查项

- [x] Windows 线程池默认 max_workers=4
- [x] OCR 引擎全局单例，首次调用无明显延迟
- [x] 图片压缩使用 OpenCV
- [x] agent-browser 命令复用 session
- [x] 无新增 bug 或回归（484 passed, 8 skipped）
- [x] ruff/pyright 检查通过

---

## 完成记录

### 2026-01-27 Windows 性能优化完成

**实现内容：**

1. **W1 线程池配置** (`utils/executor.py`)
   - 新增 `_get_optimal_workers()` 函数
   - Windows 限制 max_workers=4，Linux/macOS 限制 max_workers=8

2. **W2 ONNX Runtime 单例 + 预热** (`ocr.py`, `batch.py`)
   - 新增 `_global_engine`、`_global_config` 类变量
   - 新增 `get_shared_engine()` 类方法（双重检查锁定）
   - 新增 `preheat()` 类方法（执行 dummy inference 预热）
   - 新增 `_create_engine_impl()` 类方法
   - `batch.py` 批处理入口添加 OCR 预热调用

3. **W3 OpenCV 图像压缩** (`image.py`, `pyproject.toml`)
   - 新增 `opencv-python>=4.8.0` 依赖
   - 新增 `_compress_image_cv2()` 函数
   - 重命名原函数为 `_compress_image_pillow()`
   - `_compress_image_worker()` 优先使用 OpenCV，失败回退 Pillow

4. **W4 asyncio 子进程批量化** (`fetch.py`)
   - 新增 `_get_effective_agent_browser_args()` 函数
   - 新增 `_run_agent_browser_batch()` 函数
   - 新增 `_url_to_session_id()` 函数（稳定 session ID 生成）

5. **W5 LibreOffice UNO 守护进程** - 评估后推迟到未来版本

**单元测试：**

- `test_executor.py`: 新增 `TestGetOptimalWorkers` 测试类（5 个测试用例）
- `test_ocr.py`: 新增 `TestOCRProcessorSingleton`、`TestOCRProcessorPreheat` 测试类
- `test_image.py`: 新增 `TestCompressImageWorkerFunctions` 测试类
- `test_fetch.py`: 新增 `TestUrlToSessionId`、`TestGetEffectiveAgentBrowserArgs` 测试类

**代码质量：**

- ruff: All checks passed!
- pyright: 0 errors, 0 warnings
- pytest: 484 passed, 8 skipped

### 2026-01-27 Bug 修复（第二批）

**问题分析：**

| # | 问题 | 根因 | 状态 |
|---|------|------|------|
| 1 | 终端窗口弹出 (chrome-headless-shell.exe) | `verify_agent_browser_ready` 使用 `subprocess.run()` 未设置 `CREATE_NO_WINDOW` | ✅ 已修复 |
| 2 | agent-browser 启动延迟 70s | `open about:blank` 测试超时 30s | ✅ 已修复 |
| 3 | PDF LLM 增强失败 (max_tokens 超限) | Router 选择 deepseek 但 max_tokens 基于 gemini 计算 | ✅ 已修复 |
| 4 | x.com 超时 | 代理未配置 (中国大陆环境) | ⚠️ 需配置 |
| 5 | URL screenshot 失败 | browser 超时导致 | ⚠️ 需配置 |

**修复内容：**

1. **终端窗口隐藏** (`fetch.py`)
   - `verify_agent_browser_ready` 的 `subprocess.run()` 添加 `creationflags=CREATE_NO_WINDOW`
   - 统一使用 `run_kwargs` 字典传递参数

2. **启动延迟优化** (`fetch.py`)
   - `open about:blank` 测试超时从 30s 改为 10s
   - 减少启动阻塞时间

3. **max_tokens 兼容性修复** (`llm.py`)
   - `_calculate_dynamic_max_tokens()` 新增 `router` 参数
   - 当使用 Router 时，获取所有模型中**最小的** `max_output_tokens`
   - 确保与 Router 可能选择的任何模型兼容
   - 更新所有调用点：
     - `_analyze_images_batch_instructor()`
     - `_analyze_single_image_instructor()`
     - `_analyze_single_image_json_mode()`
     - `enhance_url_with_vision()`
     - `_enhance_with_frontmatter()`

4. **代理自动检测** (`fetch.py`) - **新增功能**
   - 新增 `_detect_proxy()` 函数，自动检测代理设置
   - 检测顺序：环境变量 → 探测本地常见代理端口
   - 支持的环境变量：`HTTPS_PROXY`, `HTTP_PROXY`, `ALL_PROXY`
   - 探测端口：7890 (Clash), 10808 (V2Ray), 1080 (SOCKS5), 8080, 8118, 9050
   - 新增 `get_proxy_for_url()` 函数，为需要代理的 URL 返回代理
   - `_get_jina_client()` 自动应用检测到的代理
   - `_run_agent_browser_command()` 自动设置代理环境变量给 Playwright

**代理自动检测说明：**

程序现在会自动检测代理设置：

1. **优先使用环境变量**：`HTTPS_PROXY`, `HTTP_PROXY`, `ALL_PROXY`
2. **自动探测本地代理**：探测 127.0.0.1 上的常见代理端口
   - Clash: 7890, 7891
   - V2Ray: 10808, 10809
   - 其他: 1080, 8080, 8118, 9050

如果运行 Clash 等代理软件，程序会自动使用 `http://127.0.0.1:7890` 无需手动配置。

**代码质量：**

- ruff: All checks passed!
- pyright: 0 errors, 0 warnings
- pytest: 490 passed, 8 skipped（新增 6 个代理检测测试）
