# Markit Overall Optimization Plan (v2)

> 目标：在不明显改变用户交互（CLI 参数、默认行为、输出结构尽量保持）的前提下，同时提升性能、成本可控性、稳定性与可维护性。
>
> 适用场景（来自你的确认）：
> - best-effort：LLM/vision 出错时尽量降级继续，不因为 LLM 问题导致整批失败
> - 缓存：默认关闭，但允许可选启用（你认为 resume 足够，但缓存仍可在“重复运行/重复图片/重复文档”场景显著省钱）
> - 主要工作负载：批处理为主，Office + PDF 为主，经常使用 `--preset rich`
>
> 本文是“超级详细”的实施计划文档；不直接改代码，但每一项都写清楚：为什么做、改哪些点、怎么做、风险/兼容性、验收方式。

---

## 0. 不改用户交互的约束边界

- CLI 与输出：
  - 尽量不新增/不改动用户必须理解的概念；如需新增，必须做到“默认无感”。
  - 现有输出目录结构（`output/*.md`, `output/*.llm.md`, `output/assets/*`, `output/screenshots/*`, `output/reports/*.json`, `output/assets.desc.json`）保持。
  - 现有 `--preset rich/standard/minimal` 行为保持。
- 行为策略：
  - best-effort：LLM/vision/OCR 等失败时，尽量输出“已转换的内容”，并在 report/state 记录降级原因。
- 性能策略：
  - 批处理吞吐优先于单文件极致速度：要确保长批处理不会被瞬时并发/资源爆炸拖垮。

---

## 1. 当前实现的关键问题（基线）

### 1.1 并发是乘法放大，导致性能抖动与成本不可控

- 文件级并发（`BatchProcessor`）和 LLM 级并发（`LLMProcessor` 内 semaphore）在批处理中会叠加。
- `--preset rich` 下每个文件可能触发多次 LLM：
  - 文档 clean/frontmatter
  - OCR+LLM / PPTX+LLM 的 vision enhance
  - 图片 alt/desc 分析（多图时 N 次 vision call）
- 结果：峰值请求数远超配置直觉，触发 rate limit 后重试/排队导致整体吞吐下降。

### 1.2 图片 alt 替换方式对长文 + 多图退化明显

- 现行逻辑对每张图对全文 `re.sub`，复杂度接近 `O(N_images * len(markdown))`。

### 1.3 输出写入不是原子化，中断会留下“半成品”

- `.md`、`.llm.md`、`assets.desc.json`、`reports/*.json`，以及 batch state 在不同阶段写入。
- 一旦崩溃或中断，各类文件的相互一致性不可保证。

### 1.4 可维护性：核心 workflow 逻辑堆在 CLI

- `packages/markit/src/markit/cli.py` 既负责参数解析，也负责转换 pipeline，且单文件/批处理逻辑存在大量重复。

---

## 2. 总体优化路线图（P0-P3）

- P0（立刻做，收益最大，风险低）：并发治理 + 资源预算 + 产物原子写入
- P1（短期做，收益大，风险中）：workflow 分层 + 复用 LLMProcessor + 统一统计/降级记录
- P2（中期做，收益中，风险中）：图片/vision 成本优化（去重、压缩、可选缓存）+ 文件扫描提速
- P3（持续做，收益长期，风险低）：可观测性与回归体系（性能/成本/稳定性）

每个阶段都给出：改动点、实现细节、验收标准。

---

## 3. P0：并发治理与资源预算（性能 + 成本 + 稳定）

### 3.1 建立“全局 LLM 并发限制器”（必须项）

**目标**：无论批处理启动多少文件任务，LLM/vision 调用都用同一把全局 semaphore，避免并发乘法爆炸。

**设计**：
- 新增一个轻量对象 `LLMRuntime`（或 `GlobalLLMLimiter`），包含：
  - `semaphore: asyncio.Semaphore`（全局
  - 可选：限速器（token bucket）用于更平滑的 QPS
- `LLMProcessor` 不再“每实例 new semaphore”，而是接收外部传入的 limiter；如果未传入则使用本地 semaphore（保持兼容）。

**改动点建议**：
- `packages/markit/src/markit/llm.py`
  - `LLMProcessor.__init__` 增加可选参数：`runtime: LLMRuntime | None = None`
  - `_call_llm_with_retry`/`_analyze_with_instructor`/`_analyze_with_json_mode`/`extract_page_content`/`enhance_document_with_vision` 等所有会消耗并发的入口，都用 `runtime.semaphore`。
- `packages/markit/src/markit/cli.py`
  - 在 `run_workflow()` 或 batch/single 顶层创建一个 `runtime = LLMRuntime(concurrency=cfg.llm.concurrency)` 并传给各处。

**兼容性**：
- CLI 不变；仅内部调度改变。

**验收**：
- 批处理 100 文件、`--preset rich`：
  - 观测峰值并发 LLM 请求数不超过 `cfg.llm.concurrency`（或略高但可解释）
  - rate limit/timeout 大幅降低

### 3.2 引入“每文件 LLM 成本预算/熔断”（强烈建议，默认可不启用）

**目标**：在 rich 模式中避免某个文件因为图片太多或异常重试导致成本失控、拖慢整批。

**策略**：
- 新配置（默认不启用，CLI 不增加新参数也可）：
  - `llm.budget.max_cost_usd_per_file: float | null`
  - `llm.budget.max_requests_per_file: int | null`
  - `llm.budget.max_images_per_file: int | null`（超过就只做前 N 张）
- best-effort：预算触发时不报错，跳过后续 LLM/vision 步骤，并在 report/state 记录为 `skipped_reason=budget_exceeded`。

**实现要点**：
- 在 `LLMProcessor` 中维护“当前 context（文件）”级别的 usage 计数器（已有 `_call_counter`，可复用思路）。
- 每次 call 结束检查预算，超过则抛出内部可识别异常（例如 `BudgetExceeded`），在 workflow 层捕获并降级。

**验收**：
- 人工构造多图/超长 PDF，预算触发时：
  - `.md` 仍存在
  - `.llm.md` 可能缺失或仅部分
  - report 明确记录预算触发

### 3.3 统一“文件级 best-effort 兜底边界”（必须项）

**目标**：任何单文件失败都不应拖垮整批；失败文件应被标记 FAILED 并写入 error（已存在），但不影响其他文件继续。

**要点**：
- 约束：
  - Converter 硬失败（打不开文件、格式错误） => 文件失败
  - LLM/vision 失败 => 降级继续（保留 converter 输出、保留 screenshots、保留 assets），并记录

**落地**：
- 把 `cli.py` 中散落的 try/except 收敛为：
  - `convert` 必须成功才算 file success
  - quality enhancements（LLM、image analysis）属于 optional stage

---

## 4. P0：输出写入原子化与一致性（稳定 + 可维护）

### 4.1 引入统一的原子写工具（必须项）

**目标**：避免中断留下半写文件。

**技术方案**：
- 新增工具函数：
  - `atomic_write_text(path: Path, content: str, encoding="utf-8")`
  - `atomic_write_json(path: Path, obj: Any)`
- 实现：同目录写 `.{name}.tmp`，写完后 `replace`。

**替换范围**：
- `packages/markit/src/markit/cli.py`
  - `.md`、`.llm.md`、单文件 report json
  - `assets.desc.json`
- `packages/markit/src/markit/batch.py`
  - batch state 文件写入（`save_state`）
  - batch report 写入（`save_report`）

**验收**：
- 在处理中途 kill 进程：
  - 不出现 `*.tmp` 残留（或可在启动时清理）
  - 重要输出文件不应出现截断内容

### 4.2 产物状态机与可恢复语义（建议）

**目标**：让 `--resume` 不只是“跳过完成文件”，还可以基于产物判断是否需要重做某阶段。

**最小化改造**（不改变用户交互）：
- 在 state 的每个 file 里记录 stage：
  - `converted` / `llm_processed` / `image_analyzed` / `reported`
- resume 时：
  - 如果 `.md` 存在但 `.llm.md` 缺失且 `cfg.llm.enabled` => 只补做 LLM stage
  - 如果 `assets.desc.json` 缺失但 desc_enabled => 只补做 desc 聚合

**注意**：这会改变 resume 语义（更聪明），但对用户是正向增强，且无需新参数。

---

## 5. P1：Workflow 分层与复用（可维护 + 性能 + 稳定）

### 5.1 从 CLI 抽出 workflow 层（必须项）

**目标**：保持 CLI 交互不变，但把业务编排移到可测试的模块。

**建议模块结构**（示例，不要求一次到位）：
- `packages/markit/src/markit/workflow/__init__.py`
- `packages/markit/src/markit/workflow/single.py`
- `packages/markit/src/markit/workflow/batch.py`
- `packages/markit/src/markit/workflow/reporting.py`
- `packages/markit/src/markit/workflow/assets_desc.py`

**拆分原则**：
- CLI（click）：只做参数/配置/日志初始化 + 调用 workflow
- workflow：
  - 接收 `MarkitConfig`、`runtime`、`converter`/`processor` 等依赖
  - 返回结构化结果（用于 report/state）

**验收**：
- unit test 能直接调用 workflow，不需要 click runner
- `cli.py` 文件行数显著下降（比如从 1500+ -> 400-600）

### 5.2 复用 LLMProcessor（必须项）

**目标**：避免每个阶段/每文件反复创建 Router/注册 callback。

**方式**：
- 在 batch 开始创建一个 `processor = LLMProcessor(cfg.llm, cfg.prompts, runtime=runtime)`
- 每文件复用此 processor，但 usage 记录要按 file context 打标签（已有 `context` 参数设计）。

**细节**：
- `LLMProcessor` 的 `_usage` 当前是全局累计；对于 report 需要“每文件 usage”。
- 方案：
  1) 在 processor 内维护 `usage_by_context`（推荐）
  2) 或在 workflow 层每文件开始前记录“usage snapshot”，结束后做差分

---

## 6. P2：图片与 vision 成本优化（成本 + 性能 + 稳定）

### 6.1 alt 替换算法改为单次扫描（必须项）

**目标**：对多图长文避免退化。

**实现建议**：
- 解析 markdown 中 `![](...assets/<name>)` 形式，构建引用表：`name -> list[span]`。
- 拿到 vision 输出后，只对命中的 spans 替换 alt 文本。

**兼容性**：
- 输出 markdown 结构不变。

### 6.2 跨文件图片去重（强烈建议）

**目标**：rich 模式常见大量重复图片（logo、页眉页脚图标），跨文件去重能直接省钱。

**策略**：
- 计算图片哈希（建议 sha256 或感知哈希 phash，先 sha256 足够）。
- 在 batch 生命周期里维护 `hash -> analysis_result` 缓存（内存）。
- 可选持久化缓存（默认关闭）：
  - `output/.markit-cache/image_analysis.json`
  - 仅存 hash 与 caption/desc，不存原图与敏感路径。

**你提到 resume 足够**：
- resume 的 state 只能保证“同一次输出目录没跑完继续”，不能覆盖“同批重复运行/修改配置/对不同 output_dir 重复处理”。
- 因此缓存是“额外的成本优化”，默认关闭即可，且不影响现有逻辑。

### 6.3 vision 输入预压缩（强烈建议）

**目标**：减少 base64 payload，降低 token/错误率。

**建议新增配置**（默认保持现状也行，但建议给默认值）：
- `image.vision_max_width: 1024`
- `image.vision_max_height: 1024`
- `image.vision_quality: 70`

实现：对要送 vision 的图片生成临时压缩副本再 base64。

### 6.4 图片分析“挑选策略”

**目标**：rich 下图片很多时，做“最有价值”的那部分。

策略（保持默认兼容，可通过 config 控制）：
- 过滤太小/疑似装饰图：已有 `ImageFilterConfig`，建议默认更严格一点（先不改默认，提供 preset rich 内的 override）
- 优先分析：大图/包含文本的图（可做轻量 OCR 检测或像素特征）

---

## 7. P2：批处理文件扫描与 I/O 优化（性能 + 稳定）

### 7.1 discover_files 改为单次遍历

**目标**：大目录场景减少 glob 组合与重复候选。

实现：
- 使用 `os.walk`：
  - 用当前的 `scan_max_depth` 截断下钻
  - 用 `scan_max_files` 限制
  - 每个文件扩展名在 set 内则收集
- 仍然调用 `validate_path_within_base` 确保安全。

### 7.2 state 写入节流策略再加强

现有已有 `state_flush_interval_seconds`；进一步建议：
- 在高并发场景中把 state 写入放到一个单独的“节流队列”或 background task
- 主流程只更新内存状态

---

## 8. P3：可观测性、报告与回归体系（长期收益）

### 8.1 报告字段标准化

- 对 LLM usage 标注 `token_status`：`actual | estimated | unknown`
- 每文件记录：
  - `stages`: {converted, llm_processed, image_analyzed}
  - `degraded`: true/false
  - `degraded_reasons`: ["llm_failed", "vision_rate_limited", "budget_exceeded", ...]

### 8.2 性能基准与回归测试

**建议最小基准集**（Office+PDF 为主）：
- 10 个文件：快速回归（CI 可跑）
- 100 个文件：性能回归（本地/夜间）

指标：
- wall time、平均每文件耗时、P95 耗时
- LLM 请求数、失败率、重试次数
- cost 汇总（如果 available）

### 8.3 错误分类与用户可理解提示

- best-effort 下不要把 stacktrace 直接打印到终端（可写入 log），终端给简洁原因。
- 使用 `markit.security.sanitize_error_message()` 对外输出。

---

## 9. 具体实施清单（按文件逐条）

> 这一节更像“变更说明书”，便于实际落地时逐项勾选。

### 9.1 `packages/markit/src/markit/llm.py`

- [ ] 增加 `LLMRuntime`（全局 semaphore + 可选限速）
- [ ] `LLMProcessor` 支持注入 runtime，所有调用统一走 runtime semaphore
- [ ] 增加 budget 检查点（可选）
- [ ] 支持 usage 按 context 统计（用于 per-file report）

### 9.2 `packages/markit/src/markit/cli.py`

- [ ] CLI 不变：仍然支持 preset，仍然输出 `.md/.llm.md/reports/assets.desc.json`
- [ ] 抽离 workflow：CLI 只做参数解析与调用
- [ ] 批处理与单文件都复用同一 `LLMProcessor` + `LLMRuntime`
- [ ] 统一 best-effort：LLM/vision/OCR 失败时写入 report/stage，不中断批处理
- [ ] 全部写文件改为 atomic write

### 9.3 `packages/markit/src/markit/batch.py`

- [ ] state 写入改为 atomic write
- [ ] 可以记录 stage 与 degrade 信息（用于 resume/报告）
- [ ] discover_files 改为单次遍历（配合 scan_max_depth/scan_max_files）

### 9.4 `packages/markit/src/markit/image.py`

- [ ] 为 vision 加“临时缩略图”输出能力（不影响现有 assets）
- [ ] 支持可选跨文件 hash cache（batch 生命周期）

### 9.5 `packages/markit/src/markit/security.py`

- [ ] 对外输出错误统一走 `sanitize_error_message`
- [ ] 为 atomic write 增加“tmp 文件清理”帮助函数（可选）

---

## 10. 风险评估与回滚策略

- 并发治理（全局 semaphore）风险：
  - 可能降低峰值吞吐，但整体 wall time 通常更稳定（少 rate limit）。
  - 回滚：runtime 注入可开关，回退到“每实例 semaphore”。

- 原子写风险：
  - Windows 下 rename 语义差异需注意，但当前环境主要 Linux。
  - 回滚：仅对关键文件启用（report/state）再逐步扩大。

- 更聪明的 resume（按 stage 补做）风险：
  - 可能改变用户对 resume 的预期；但属于增强。
  - 回滚：默认仍按旧逻辑，仅当检测到 stage 信息存在时启用。

---

## 11. 验收清单（你关心的“批处理 + rich + Office/PDF”）

- 功能不变：
  - `markit ./input -o ./output --preset rich --resume` 行为一致
  - `.md` 一定生成（converter 成功时）
  - `.llm.md` 尽量生成，失败则记录降级
  - `output/reports/*.report.json` 保持

- 稳定性：
  - kill 进程后 resume 可继续，且不会产生截断文件
  - 批处理 100 文件中，某些文件 LLM 失败不会导致其他文件失败

- 性能：
  - 峰值 LLM 并发不超过配置
  - 在 rate limit 环境下 wall time 波动显著降低

- 成本：
  - 同一批次重复图片减少 vision 调用（启用跨文件去重后）
  - vision 输入压缩后 cost/token 下降

---

## 12. 建议的执行顺序（最小风险落地）

1) 先做 atomic write（report/state/md/llm.md）
2) 再做全局 LLM semaphore（runtime 注入）
3) 再做 LLMProcessor 复用与 per-context usage
4) 再做 alt 替换单次扫描 + 跨文件图片去重
5) 最后做 workflow 分层与 discover_files 重构

---

## 13. 附：不改变 CLI 的前提下，给 rich preset 的“默认策略建议”

- rich 模式默认仍启用：LLM + alt + desc + screenshot
- 但内部策略建议：
  - 图片分析并发 <= `llm.concurrency`
  - 对小图/装饰图更严格过滤（可作为 rich preset 的内置 override，而不是改全局默认）
  - vision 输入使用较小分辨率副本

---

## 结语

这份 v2 计划的核心思想是：
- 用“全局资源治理”解决批处理 rich 模式的吞吐与稳定性问题
- 用“原子写 + stage 记录”保证随时可中断、可恢复
- 用“workflow 分层 + processor 复用”降低复杂度、提升可测试性
- 用“图片与 vision 成本策略”在保持体验的同时显著降本
