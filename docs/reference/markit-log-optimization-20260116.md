# Markit 日志系统优化方案

> 日期: 2026-01-16
> 状态: 已实施

## 一、现状问题

### 1.1 日志几乎为空

开启 `--verbose` 选项运行批处理后，日志文件只有 2 条记录：

```json
{"message": "Processing 6 files with concurrency 10", ...}
{"message": "Report saved: output4/reports/fixtures.report.json", ...}
```

整个 6 分 44 秒的处理过程中间没有任何日志输出。

### 1.2 根本原因

`batch.py:407-413` 在批处理期间禁用了整个日志命名空间：

```python
logger.disable("markit")  # 禁用所有日志（包括文件日志）
try:
    with Live(progress, ...):
        # 所有处理都在这里，日志全部丢失
finally:
    logger.enable("markit")
```

### 1.3 缺失的关键信息

- 每次 LLM 调用的详情（model、tokens、耗时、成本）
- LLM 重试信息（重试次数、原因、状态码）
- HTTP 状态码和错误消息
- 文件处理的阶段性日志

---

## 二、LiteLLM Callback API 调研结果

### 2.1 Callback 注册方式

```python
from litellm.integrations.custom_logger import CustomLogger

class MarkitLLMLogger(CustomLogger):
    def log_pre_api_call(self, model, messages, kwargs): ...
    def log_success_event(self, kwargs, response_obj, start_time, end_time): ...
    def log_failure_event(self, kwargs, response_obj, start_time, end_time): ...
    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time): ...
    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time): ...

litellm.callbacks = [MarkitLLMLogger()]
```

### 2.2 可获取的信息

**kwargs 字段：**

| 字段 | 说明 |
|------|------|
| `model` | 模型名 |
| `messages` | 输入消息 |
| `response_cost` | 成本 |
| `cache_hit` | 是否缓存命中 |
| `litellm_params.metadata` | 自定义元数据 |
| `litellm_params.api_key` | API Key |
| `litellm_params.api_base` | API Base URL |
| `standard_logging_object` | 标准日志 payload |

**standard_logging_object 字段：**

| 字段 | 说明 |
|------|------|
| `id` | 唯一标识 |
| `model` | 实际使用的模型 |
| `api_base` | API 端点 |
| `prompt_tokens` | 输入 tokens |
| `completion_tokens` | 输出 tokens |
| `response_time` | 响应时间（秒） |
| `status` | 状态 |
| `error_code` | 错误码 |
| `error_class` | 错误类型 |

**异常对象属性：**

| 属性 | 说明 |
|------|------|
| `status_code` | HTTP 状态码 |
| `message` | 错误消息 |
| `llm_provider` | 提供商 |

### 2.3 重试时的 Callback 行为

**关键发现：LiteLLM callback 只在最终成功/失败时触发，中间重试不触发。**

因此需要自己实现重试逻辑来追踪每次重试详情。

---

## 三、优化方案设计

### 3.1 架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                         setup_logging()                             │
├─────────────────────────────────────────────────────────────────────┤
│  console_handler = logger.add(stderr)                               │
│      - level: DEBUG if verbose else INFO                            │
│      - 可被批处理临时禁用                                             │
│                                                                     │
│  file_handler = logger.add(file)                                    │
│      - level: config.log.level (默认 DEBUG)                          │
│      - serialize=True (JSON)                                        │
│      - 独立运行，不受批处理禁用影响                                    │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    批处理模式                                        │
├─────────────────────────────────────────────────────────────────────┤
│  非 verbose: 禁用 console handler，文件日志正常                       │
│  verbose: 显示 LogPanel（进度条 + 日志面板）                          │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 日志事件设计

| 事件 | 级别 | 格式 | 触发时机 |
|------|------|------|---------|
| `file_start` | INFO | `[START] {filename}` | 文件处理开始 |
| `file_convert` | INFO | `[CONVERT] {filename}: {time}s` | 格式转换完成 |
| `llm_request` | DEBUG | `[LLM:{file}:{n}] Request to {model}` | LLM 调用开始 |
| `llm_success` | INFO | `[LLM:{file}:{n}] {model} tokens={in}+{out} time={ms}ms cost=${cost}` | LLM 调用成功 |
| `llm_retry` | WARNING | `[LLM:{file}:{n}] Retry #{attempt}: {error_type} status={code}` | LLM 重试 |
| `llm_error` | ERROR | `[LLM:{file}:{n}] Failed: {error_type} status={code} msg={msg}` | LLM 最终失败 |
| `ocr_complete` | DEBUG | `[OCR] {filename}: {blocks} blocks, confidence={avg}` | OCR 完成 |
| `file_done` | INFO | `[DONE] {filename}: {time}s (images={n}, cost=${cost})` | 文件处理完成 |
| `file_error` | ERROR | `[FAIL] {filename}: {error}` | 文件处理失败 |

### 3.3 Verbose 模式 LogPanel

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

- 日志面板显示 8 行
- 自动滚动显示最新日志

### 3.4 重试追踪方案

**采用方案 B1：禁用 Router 内部重试，自己实现重试循环**

```python
async def _call_llm_with_retry(
    self,
    model: str,
    messages: list[dict],
    context: str,
    max_retries: int = 3,
) -> LLMResponse:
    """带重试追踪的 LLM 调用"""

    call_index = self._get_next_call_index(context)
    call_id = f"{context}:{call_index}"

    last_exception = None

    for attempt in range(max_retries + 1):
        start_time = time.perf_counter()

        try:
            if attempt == 0:
                logger.debug(f"[LLM:{call_id}] Request to {model}")
            else:
                logger.warning(
                    f"[LLM:{call_id}] Retry #{attempt}: "
                    f"{type(last_exception).__name__} "
                    f"status={getattr(last_exception, 'status_code', 'N/A')}"
                )

            response = await self.router.acompletion(
                model=model,
                messages=messages,
                metadata={"call_id": call_id, "attempt": attempt},
            )

            # 成功
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            # ... 记录成功日志
            return response

        except RETRYABLE_ERRORS as e:
            last_exception = e
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            if attempt == max_retries:
                # 最终失败
                logger.error(
                    f"[LLM:{call_id}] Failed after {max_retries + 1} attempts: "
                    f"{type(e).__name__} status={getattr(e, 'status_code', 'N/A')}"
                )
                raise

            # 指数退避
            wait_time = min(2 ** attempt, 60)
            await asyncio.sleep(wait_time)

        except Exception as e:
            # 不可重试的错误
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                f"[LLM:{call_id}] Failed: {type(e).__name__} "
                f"status={getattr(e, 'status_code', 'N/A')} "
                f"msg={str(e)[:200]}"
            )
            raise
```

### 3.5 LiteLLM CustomLogger 集成

```python
class MarkitLLMLogger(CustomLogger):
    """LiteLLM callback 用于获取额外信息"""

    def __init__(self):
        self.last_call_details: dict = {}

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        slo = kwargs.get("standard_logging_object", {})
        self.last_call_details = {
            "api_base": slo.get("api_base"),
            "response_time": slo.get("response_time"),
            "cache_hit": kwargs.get("cache_hit", False),
        }

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self.log_success_event(kwargs, response_obj, start_time, end_time)
```

---

## 四、实现清单

| 序号 | 文件 | 改动内容 | 复杂度 |
|------|------|---------|--------|
| 1 | `cli.py` | `setup_logging()` 返回 handler ID，支持分离控制 | 低 |
| 2 | `cli.py` | 传递 verbose 参数给 BatchProcessor | 低 |
| 3 | `batch.py` | 只禁用 console handler，保留 file handler | 低 |
| 4 | `batch.py` | verbose 模式添加 LogPanel（8行） | 中 |
| 5 | `llm.py` | 添加 `_call_llm_with_retry()` 自定义重试逻辑 | 中 |
| 6 | `llm.py` | 添加 context 参数和调用计数器 | 低 |
| 7 | `llm.py` | 注册 LiteLLM CustomLogger callback | 低 |
| 8 | `llm.py` | Router 配置 `num_retries=0` 禁用内部重试 | 低 |
| 9 | `config.py` | LogConfig.level 默认改为 DEBUG | 低 |
| 10 | 各调用点 | 传递 context 参数给 LLM 调用 | 低 |

---

## 五、确认的设计决策

| 决策项 | 选择 |
|--------|------|
| 文件日志默认级别 | DEBUG |
| LogPanel 行数 | 8 行 |
| 重试追踪方案 | B1: 禁用 Router 重试，自己实现 |
| LiteLLM CustomLogger | 需要注册 |
| 日志文件命名 | 保持原样 `markit_{time}.log` |

---

## 六、预期效果

### 文件日志示例（JSON 格式）

```json
{"time": "2026-01-16T18:01:15.123+08:00", "level": "INFO", "message": "[START] file.pdf"}
{"time": "2026-01-16T18:01:16.456+08:00", "level": "DEBUG", "message": "[LLM:file.pdf:1] Request to default"}
{"time": "2026-01-16T18:01:17.789+08:00", "level": "INFO", "message": "[LLM:file.pdf:1] gemini-2.5-flash tokens=1500+200 time=1234ms cost=$0.001200"}
{"time": "2026-01-16T18:01:18.012+08:00", "level": "DEBUG", "message": "[LLM:file.pdf:2] Request to vision"}
{"time": "2026-01-16T18:01:18.500+08:00", "level": "WARNING", "message": "[LLM:file.pdf:2] Retry #1: RateLimitError status=429"}
{"time": "2026-01-16T18:01:20.123+08:00", "level": "INFO", "message": "[LLM:file.pdf:2] gemini-2.5-flash tokens=800+150 time=980ms cost=$0.000800"}
{"time": "2026-01-16T18:01:25.456+08:00", "level": "INFO", "message": "[DONE] file.pdf: 10.3s (images=7, cost=$0.002000)"}
```

### 控制台输出（verbose 模式）

进度条 + 日志面板实时滚动显示。

---

## 七、参考资源

- [LiteLLM Custom Callbacks](https://docs.litellm.ai/docs/observability/custom_callback)
- [LiteLLM Router - Load Balancing](https://docs.litellm.ai/docs/routing)
- [LiteLLM Exception Mapping](https://docs.litellm.ai/docs/exception_mapping)
- [LiteLLM Standard Logging Payload](https://docs.litellm.ai/docs/proxy/logging_spec)
- [GitHub Issue #8648 - Retry Logging](https://github.com/BerriAI/litellm/issues/8648)
