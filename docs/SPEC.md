# Markit 设计规格与开发计划
> 基于 `docs/ROADMAP.md` (批次 2026011102 & 2026011103) 生成

本文档概述了即将进行的可靠性和可观测性改进的详细设计规格和开发计划。

---

## 第一部分：日志优化与可观测性
**来源：** 批次 2026011102

### 1.1 设计目标
- **上下文感知**：每个与特定文件或 LLM 事务相关的日志条目必须携带上下文标识符（`file`、`provider_id`、`model_id`）。
- **结构一致性**：标准化日志消息中的键值对，以便机器解析（例如文本日志中的 `key=value` 格式，或适用的结构化 JSON 字段）。
- **逻辑修正**：确保日志反映实际的架构事件（例如 Provider 初始化 vs. 模型选择）。

### 1.2 详细规格

#### A. 提供商与模型初始化
*   **当前问题**：日志记录“Provider <model> initialized”，混淆了模型名称与提供商实例。
*   **重构**：
    *   修改 `LLMRegistry` 或 `ProviderFactory`，按 `provider_id`（`markit.yaml` 中的键）跟踪初始化。
    *   日志格式：`[info] Provider initialized | provider_id=<id> type=<type> base_url=<url>`
    *   验证逻辑：`validate_provider` 应针对每个 Provider 验证凭据/连接一次，而不是针对每个模型。

#### B. LLM 请求上下文
*   **需求**：将 `provider_id` 和 `model_id` 注入所有底层 HTTP/请求日志中。
*   **实现**：
    *   更新 `LLMClient` 方法（`acompletion` 等）以接受或保留上下文。
    *   **请求开始**：`[debug] Sending LLM Request | provider=<id> model=<id> request_id=<uuid> method=POST url=...`
    *   **响应/错误**：`[debug] LLM Response received | provider=<id> model=<id> status=200 duration=1.2s`
    *   **Token 用量**：`[info] LLM Usage | provider=<id> model=<id> input_tokens=... output_tokens=... cost=$...`

#### C. 转换管道日志
*   **需求**：所有管道步骤必须包含 `file=<path>`。
*   **标准化格式**：
    *   **计划**：`[debug] Conversion plan | file=<path> primary=<converter> fallback=<converter>`
    *   **尝试**：`[debug] Executing converter | file=<path> converter=<name> attempt=1/2`
    *   **预处理器**：`[debug] Running pre-processor | file=<path> processor=<name>`
    *   **Office 转换**：`[info] Converting Office format | file=<path> from=.doc to=.docx engine=LibreOffice`
    *   **完成**：`[info] Processing complete | file=<path> status=success total_duration=...`

### 1.3 实施计划（第一阶段）
1.  **重构 LLM 日志**：修改 `markit/llm/client.py` 和 `markit/llm/provider.py`。
2.  **重构管道日志**：更新 `markit/core/pipeline.py` 和 `markit/services/llm_orchestrator.py`。
3.  **重构转换器日志**：更新 `markit/converters/*.py`。
4.  **验证**：运行单个文件转换并根据规格手动验证日志输出。

---

## 第二部分：高容量弹性与架构演进
**来源：** 批次 2026011103 (v0.1.2)

### 2.1 概述
本阶段旨在解决极端条件下（10k+ 文件、API 速率限制、网络不稳定）的系统稳定性问题。它引入了一个专用的测试沙盒（“混沌模式”）和用于流控制的架构模式。

### 2.2 子阶段：高负载沙盒与混沌测试

#### A. 负载生成 (`tests/fixtures/heavy_load/`)
*   **脚本**：`generate_dataset.py`
*   **规格**：
    *   生成独立的文件夹：`1k_mix`, `10k_text`, `deep_nested`。
    *   内容：包含随机 Lorem Ipsum 内容的合成 Markdown/文本文件，用于模拟 Token 消耗。
    *   元数据：用于跟踪的可预测文件名（例如 `doc_0001.md` 到 `doc_9999.md`）。

#### B. 混沌模拟提供商 (`markit/llm/providers/chaos.py`)
*   **类**：`ChaosMockProvider(LLMProvider)`
*   **配置**：
    *   `latency_mean`: 5.0s
    *   `latency_stddev`: 2.0s
    *   `failure_rate`: 0.1 (10% 随机 500 错误)
    *   `rate_limit_prob`: 0.3 (30% 几率出现 429 错误)
    *   `oom_trigger`: 布尔值（模拟海量通用负载以测试内存）。
*   **行为**：拦截 `complete` 调用并在返回模拟文本之前应用模拟的混沌。

#### C. 弹性测试套件
*   **场景 1：持久战**：针对混沌提供商运行 1000 个文件。断言 0 崩溃，最终 100% 完成。
*   **场景 2：中断者**：
    *   启动 500 个文件的批处理。
    *   在 10 秒时发送 `SIGINT`。
    *   验证 `state.json` 完整性。
    *   使用 `--resume` 恢复。
    *   断言未重复处理已完成的文件。

### 2.3 子阶段：架构演进

#### A. 自适应并发控制 (AIMD)
*   **问题**：静态 `--concurrency` 不是最优的。太高 = 429 循环；太低 = 慢。
*   **组件**：`AdaptiveRateLimiter`
*   **算法 (AIMD)**：
    *   *加性增*：每 `N` 个成功请求，并发限制增加 1（上限为 `max_concurrency`）。
    *   *乘性减*：收到 429 时，立即将并发限制减少 0.5 倍（最小值为 1）。
*   **集成**：封装 `LLMTaskQueue` 中的 `asyncio.Semaphore`。

#### B. 优先级背压队列
*   **问题**：文件读取（生产者）比 LLM 处理（消费者）快得多。将 10k 文件加载到内存会导致 OOM。
*   **设计**：
    *   **加载的有界信号量**：仅允许 `N * 2` 个文件同时处于“加载中”或“处理中”状态。
    *   **管道阶段**：
        1.  **发现**：懒加载生成器（已存在）。
        2.  **摄入**：读取文件内容 -> **如果队列已满则阻塞**。
        3.  **处理**：LLM/转换。
        4.  **完成**：写入磁盘/更新状态 -> **最高优先级**。
*   **重构**：从 `asyncio.gather(all_tasks)` 迁移到具有有界大小的 `生产者-消费者` 队列模式。

#### C. 死信队列 (DLQ) 策略
*   **逻辑**：
    *   在 `state.json` 中跟踪每个文件的 `failure_count`。
    *   如果 `failure_count > max_retries`（例如 3），将状态标记为 `permanent_failure`。
    *   **隔离**：将这些文件移动或记录到 `failed_report.json`，以便下次恢复时不会无限重试它们。

#### D. 可观测性仪表板 (控制台)
*   **增强**：更新 `rich` Live 显示。
*   **指标**：
    *   当前并发数（动态）。
    *   吞吐量（文档/分钟）。
    *   错误率（429/5xx 计数）。
    *   预估成本（实时累积）。

### 2.4 实施计划（第二阶段）
1.  **步骤 1**：构建 `ChaosMockProvider` 和负载生成器。（测试的前提条件）。
2.  **步骤 2**：实现 `state.json` 增强（重试计数、DLQ 逻辑）。
3.  **步骤 3**：重构 `ConversionPipeline` 以使用有界队列（背压）。
4.  **步骤 4**：实现 `AdaptiveRateLimiter`。
5.  **步骤 5**：运行“持久战”测试并调整参数。

---

## 交付物摘要

| 批次 | 组件 | 关键交付物 |
| :--- | :--- | :--- |
| **2026011102** | 日志 | 跨所有模块的标准化、富上下文日志。 |
| **2026011103** | 测试 | `tests/fixtures/heavy_load`, `ChaosMockProvider`。 |
| **2026011103** | 核心 | `AdaptiveRateLimiter`, 有界管道队列, DLQ 逻辑。 |
