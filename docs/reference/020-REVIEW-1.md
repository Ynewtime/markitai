# v0.2.0 规划文档审查报告

**审查日期**: 2026-01-14
**审查范围**: ROADMAP.md [任务批次 2026011402 - v0.2.0] + 020-SPEC-1.md
**代码库版本**: v0.1.6

---

## 一、背景

自 ROADMAP 和 SPEC 文档撰写以来，代码库又经历了三个版本迭代（v0.1.4 → v0.1.5 → v0.1.6），引入了大量架构改进。本分析旨在识别规划与现状的差异、矛盾点，并提出刷新建议。

---

## 二、关键发现概要

| 模块 | SPEC 假设 | 当前现状 (v0.1.6) | 差距评估 |
|------|----------|-------------------|----------|
| **LLM Provider 层** | 计划全部删除，迁移到 LiteLLM | 5 个 Provider 实现高度成熟，含复杂路由 | **重大偏差** |
| **AIMD 限流** | 无规划 | 已实现 per-credential AIMD | SPEC 缺失 |
| **并发回退** | 无规划 | `complete_with_concurrent_fallback` 已实现 | SPEC 缺失 |
| **路由策略** | 简单 LiteLLM 封装 | 三种策略 (cost_first/least_pending/round_robin) | SPEC 过于简化 |
| **OCR 模块** | 新建 `src/markit/ocr/` | 不存在 | **符合预期** |
| **Prompt 管理** | 硬编码 | 已文件化到 `config/prompts/` | SPEC 需更新 |
| **chunk_workers** | 无 | 已新增，默认 6 | SPEC 需补充 |
| **llm_workers** | 默认 5 | 默认 20 | SPEC 需更新 |
| **Google SDK** | google-generativeai | 已迁移到 google-genai | SPEC 需更新 |

---

## 三、LLM 层重构 (LiteLLM Migration) 矛盾点分析

### 3.1 SPEC 假设 vs 现状

**SPEC Section 2.4 假设**：

```python
# 删除旧代码：
# - src/markit/llm/openai.py
# - src/markit/llm/anthropic.py
# - src/markit/llm/gemini.py
# - src/markit/llm/ollama.py
# - src/markit/llm/openrouter.py

# 新增 LiteLLMProvider 简单封装
class LiteLLMProvider(BaseLLMProvider):
    async def complete(self, ...) -> LLMResponse:
        response = await acompletion(model=self.model, ...)
        return self._convert_response(response)
```

**当前现状**（v0.1.6 五个 Provider 实现的复杂度）：

| Provider | 特殊实现 | 难以用 LiteLLM 简单替代的原因 |
|----------|---------|------------------------------|
| `openai.py` | GPT-5.2+ 使用 `max_completion_tokens` | LiteLLM 需验证是否支持此参数 |
| `anthropic.py` | Tool-based 结构化输出 (`IMAGE_ANALYSIS_TOOL`) | Anthropic 特殊的图片分析方式 |
| `gemini.py` | 超时转换（秒→毫秒）、`usage_metadata` 提取 | Google SDK 差异处理 |
| `ollama.py` | 本地服务器连接、自定义 token 字段 | `prompt_eval_count`/`eval_count` |
| `openrouter.py` | 继承 `OpenAIProvider`，覆盖 `_use_max_completion_tokens` | OpenRouter 特殊兼容性 |

### 3.2 ProviderManager 复杂性 - SPEC 未充分考虑

**当前 `ProviderManager` 已实现但 SPEC 中未规划的功能**：

1. **Per-Credential AIMD 限流** (`_credential_limiters`)
   ```python
   # 每个凭证独立的自适应限流
   self._credential_limiters: dict[str, AdaptiveRateLimiter] = {}
   self._credential_pending: dict[str, int] = {}  # 请求计数追踪
   ```

2. **三种路由策略** (`_select_best_provider`)
   ```python
   if strategy == "cost_first":
       return candidates[0]  # 最便宜
   elif strategy == "round_robin":
       # 轮询
   elif strategy == "least_pending":
       # 综合成本和负载
       score = cost_weight * cost_score + load_weight * load_score
   ```

3. **并发回退机制** (`complete_with_concurrent_fallback`)
   ```python
   # 主模型超时后启动备用模型并发
   primary_task = asyncio.create_task(...)
   await asyncio.wait_for(shield(primary_task), timeout=timeout)
   # 超时后：
   fallback_task = asyncio.create_task(...)
   done, pending = await asyncio.wait([primary_task, fallback_task], ...)
   ```

4. **双锁机制**（Provider 级 + Credential 级）
   ```python
   self._provider_init_locks: dict[str, asyncio.Lock] = {}
   self._credential_init_locks: dict[str, asyncio.Lock] = {}
   ```

### 3.3 迁移到 LiteLLM 的风险评估

| 风险项 | 影响 | 建议 |
|--------|------|------|
| 路由策略丢失 | LiteLLM 无 least_pending 路由 | 需在 LiteLLM 上层保留路由逻辑 |
| AIMD 限流丢失 | 无法按凭证自动调整并发 | 需保留 `AdaptiveRateLimiter` 层 |
| 并发回退丢失 | 超时竞争机制失效 | 需在上层重新实现 |
| Provider 特殊处理 | Anthropic tool-based 输出、Gemini 超时 | 需验证 LiteLLM 兼容性 |
| 测试 Provider | `ChaosMockProvider` 无法迁移 | 需实现 LiteLLM mock |

**建议决策点**：LiteLLM 迁移是否值得？需重新评估：
- 当前 5 个 Provider 已稳定运行，代码成熟
- 迁移需重新实现大量上层逻辑
- 收益：统一接口、更多提供商支持（但当前 5 个已足够）

---

## 四、配置系统矛盾点

### 4.1 ConcurrencyConfig 差异

| 字段 | SPEC 假设 | 现状 |
|------|----------|------|
| `file_workers` | 4 | 4 ✓ |
| `image_workers` | 8 | 8 ✓ |
| `llm_workers` | **5** | **20** |
| `chunk_workers` | **无** | **6 (新增)** |
| `ocr_workers` | 2 | 无 |

**需刷新**：
- `llm_workers` 默认值已从 5 调整为 20
- 新增 `chunk_workers` 字段（Per-file chunk 并发控制）
- `ocr_workers` 仍需在 v0.2.0 中添加

### 4.2 LLMConfig 新增字段

SPEC 中未提及但已存在的配置：

```python
class AdaptiveConfig(BaseModel):
    """AIMD rate limiting (v0.1.2 新增)"""
    enabled: bool = True
    initial_concurrency: int = 15
    max_concurrency: int = 50
    min_concurrency: int = 3
    success_threshold: int = 15
    multiplicative_decrease: float = 0.5
    cooldown_seconds: float = 5.0

class RoutingConfig(BaseModel):
    """LLM routing strategy (v0.1.6 新增)"""
    strategy: Literal["cost_first", "least_pending", "round_robin"] = "least_pending"
    cost_weight: float = 0.6
    load_weight: float = 0.4
```

---

## 五、依赖变更矛盾点

### 5.1 SPEC 计划 vs 现状

**SPEC 计划删除**：
```toml
# "openai>=1.0.0"
# "anthropic>=0.20.0"
# "google-generativeai>=0.5.0"
```

**当前 pyproject.toml**：
```toml
"openai>=2.14.0",         # 版本已大幅升级
"anthropic>=0.75.0",      # 版本已大幅升级
"google-genai>=1.56.0",   # ⚠️ 已迁移到新 SDK！
"ollama>=0.6.1",          # 版本已升级
```

**关键发现**：Google AI SDK 已从 `google-generativeai` 迁移到 `google-genai`，这在 SPEC 中未反映。

### 5.2 OCR 依赖 - 符合预期

SPEC 计划新增的 OCR 依赖当前不存在（符合预期）：
```toml
# 计划新增
"rapidocr-onnxruntime>=1.4.0"
"litellm>=1.50.0"

[project.optional-dependencies]
ocr-full = ["paddlepaddle>=2.6.0", "paddleocr>=2.9.0"]
```

---

## 六、CLI 命令矛盾点

### 6.1 现有 CLI 结构

```
markit
├── convert     # 单文件转换
├── batch       # 批量转换
├── config      # 配置管理 (init/test/list/locations)
├── provider    # Provider 管理 (add/list/test/fetch)
└── model       # Model 管理 (add/list)
```

### 6.2 SPEC 计划新增但未实现

| 命令 | SPEC 规划 | 现状 |
|------|----------|------|
| `markit check` | 检查环境依赖（OCR 引擎状态） | ❌ 不存在 |
| `--ocr` 参数 | convert/batch 命令新增 | ❌ 不存在 |

---

## 七、OCR 模块 - 符合预期的缺失

SPEC 规划的 OCR 模块完全不存在（符合预期，因为这是 v0.2.0 的新功能）：

```
❌ src/markit/ocr/          # 目录不存在
❌ src/markit/ocr/base.py   # BaseOCREngine
❌ src/markit/ocr/rapid.py  # RapidOCREngine
❌ src/markit/ocr/paddle.py # PaddleOCREngine
❌ src/markit/ocr/dual.py   # DualOCREngine
```

**PDFConfig 现状**：
```python
class PDFConfig(BaseModel):
    engine: Literal["pymupdf4llm", "pymupdf", "pdfplumber", "markitdown"] = "pymupdf4llm"
    extract_images: bool = True
    ocr_enabled: bool = False  # 字段存在但未实现
    # ❌ 缺少: ocr_auto_detect, ocr_dpi, ocr_engine
```

---

## 八、Prompt 管理 - SPEC 需更新

**SPEC 假设**：Prompt 硬编码在 `enhancer.py`

**现状**（v0.1.5 已重构）：
```
src/markit/config/prompts/
├── enhancement_zh.md           # 主增强 prompt
├── enhancement_en.md
├── enhancement_continuation_zh.md  # 续段 prompt
├── enhancement_continuation_en.md
├── summary_zh.md
├── summary_en.md
├── image_analysis_zh.md
└── image_analysis_en.md
```

加载机制：
```python
class PromptConfig(BaseModel):
    output_language: Literal["zh", "en", "auto"] = "zh"
    prompts_dir: str = DEFAULT_PROMPTS_DIR
    image_analysis_prompt_file: str | None = None
    enhancement_prompt_file: str | None = None
    description_strategy: Literal["first_chunk", "separate_call", "none"] = "first_chunk"
```

---

## 九、综合建议

### 9.1 SPEC 文档需刷新的内容

| Section | 需刷新内容 |
|---------|-----------|
| 2.2 现有接口分析 | 补充 `LLMTaskResultWithStats`、`_handle_api_error` 等新增类 |
| 2.4.4 ProviderManager 适配 | 补充 AIMD 限流、路由策略、并发回退的迁移方案 |
| 2.5 依赖变更 | 更新 google-genai 替代 google-generativeai |
| 附录 A.2 修改文件 | 补充 `chaos.py`, `adaptive_limiter.py`, `flow_control.py`, `prompts/` |
| 附录 B.1 当前项目结构 | 更新 `config/prompts/` 目录结构 |

### 9.2 ROADMAP 需刷新的内容

| 内容 | 建议 |
|------|------|
| LiteLLM 迁移决策 | 需重新评估收益/成本比，考虑保留现有 Provider 实现 |
| 依赖删除列表 | 更新 Google SDK 名称为 google-genai |
| ConcurrencyConfig | 更新默认值（llm_workers: 20, 新增 chunk_workers: 6） |
| 测试计划 | 补充 AIMD 限流、路由策略的迁移测试 |

### 9.3 潜在的重大决策变更

**核心问题**：LiteLLM 迁移是否仍然值得？

**赞成迁移**：
- 统一接口，减少维护成本
- LiteLLM 支持 100+ 提供商
- 简化未来的提供商扩展

**反对迁移**：
- 当前 5 个 Provider 已高度稳定（经过 v0.1.x 多个版本验证）
- AIMD 限流、路由策略、并发回退等复杂逻辑需要在 LiteLLM 上层重新实现
- 迁移风险高，测试覆盖需要全面重写
- 可能引入新的兼容性问题

**建议**：考虑**保留现有 Provider 实现**，仅在 v0.2.0 中聚焦 OCR 功能。LiteLLM 迁移可作为 v0.3.0 的独立重构目标。

---

## 十、文档自相矛盾点汇总

| 位置 | 矛盾描述 |
|------|---------|
| SPEC 2.5 | 称"移除 google-generativeai"，但代码已迁移到 google-genai |
| SPEC 2.4.4 | ProviderManager 适配仅展示简单封装，未考虑 AIMD 限流等复杂逻辑 |
| ROADMAP ConcurrencyConfig | llm_workers 默认 5，实际为 20；未提及 chunk_workers |
| SPEC 附录 A.2 | 修改文件列表不完整，缺少 v0.1.2-v0.1.6 新增的关键文件 |

---

## 附录：代码库探索详情

### A. LLM 模块现状 (`src/markit/llm/`)

**现有文件**：
- `base.py` - BaseLLMProvider, LLMMessage, LLMResponse, TokenUsage, LLMTaskResultWithStats
- `manager.py` - ProviderManager (延迟加载, AIMD, 路由策略, 并发回退)
- `enhancer.py` - MarkdownEnhancer (Prompt 文件化加载, 多 chunk 处理)
- `queue.py` - LLMTaskQueue (背压队列, AIMD 集成)
- `openai.py` - OpenAIProvider
- `anthropic.py` - AnthropicProvider
- `gemini.py` - GeminiProvider
- `ollama.py` - OllamaProvider
- `openrouter.py` - OpenRouterProvider (继承 OpenAIProvider)
- `chaos.py` - ChaosMockProvider (测试用)

### B. 配置系统现状 (`src/markit/config/`)

**settings.py 主要配置类**：
- `PDFConfig` - engine, extract_images, ocr_enabled
- `ConcurrencyConfig` - file_workers(4), image_workers(8), llm_workers(20), chunk_workers(6)
- `LLMConfig` - credentials[], models[], validation, adaptive, routing
- `AdaptiveConfig` - AIMD 参数
- `RoutingConfig` - strategy, cost_weight, load_weight
- `PromptConfig` - output_language, prompts_dir, *_prompt_file

### C. 服务层现状 (`src/markit/services/`)

- `image_processor.py` - ImageProcessingService (格式转换, 压缩, 去重)
- `llm_orchestrator.py` - LLMOrchestrator (LLM 操作协调, per-file semaphore)
- `output_manager.py` - OutputManager (文件写入, 冲突解决)

### D. 工具层现状 (`src/markit/utils/`)

- `adaptive_limiter.py` - AdaptiveRateLimiter (AIMD 算法)
- `flow_control.py` - BoundedQueue, DeadLetterQueue
- `concurrency.py` - ConcurrencyManager (三层信号量)
