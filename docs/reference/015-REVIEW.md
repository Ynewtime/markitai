# 任务批次 2026011302 - v0.1.5 深度复盘分析报告

## 1) 现象复核（用交付件直接举证）

### 1.1 “大量无效 Chunk YAML / 多段 frontmatter”属实，且非常严重
在 `docs/fail_output/2.Hello OCS源码剖析.doc.md` 中，除了开头的系统 frontmatter 外，还出现了大量 “`--- ... ---` + YAML 字段”块，例如：

- `docs/fail_output/2.Hello OCS源码剖析.doc.md:160` 起出现一段新的 YAML（`description/entities/topics/domain`），这就是典型的“续段 chunk 也生成了 frontmatter”。

同样问题也出现在：
- `docs/fail_output/OCS计费原理与实现(排版后).doc.md:120`

并且 grep 结果显示这两个文件里 `---` 出现次数异常高（这和 chunk 数量、LLM 多次返回 frontmatter高度一致）。

### 1.2 “PPTX 标题空行不规范 / 规整效果不明显”属实
`docs/fail_output/CBS架构演进规划201605 V0.5.pptx.md` 在 `## 第 1 页` 后面没有空行：

- `docs/fail_output/CBS架构演进规划201605 V0.5.pptx.md:28` 是标题行
- `docs/fail_output/CBS架构演进规划201605 V0.5.pptx.md:29` 立刻接内容（未留空行）

这与 ROADMAP 要求“header 前后各一行空行”不符，也解释了“LLM 规整看起来没起作用”。

### 1.3 “大量 API 报错”属实，但不应仅归因于“网络不稳定”
在 `docs/fail_logs/20260113_windows_powershell_test.txt` 中可以看到大量网络异常堆栈，例如：
- `docs/fail_logs/20260113_windows_powershell_test.txt:1207` `httpcore.ConnectError: All connection attempts failed`
- `docs/fail_logs/20260113_windows_powershell_test.txt:1244` `httpx.ConnectError: All connection attempts failed`
- `docs/fail_logs/20260113_windows_powershell_test.txt:1567` `httpcore.ReadError`
- `docs/fail_logs/20260113_windows_powershell_test.txt:1604` `httpx.ReadError`

这些当然可能是网络问题，但结合“chunk=41/20”等大拆分（见下节），“并发策略”同样是关键因素。

---

## 2) 关键根因分析（对照现有代码，指出 ROADMAP 里遗漏点）

### 2.1 Chunk frontmatter 的根因：prompt + merge 只是表面，真正的致命点是“多 chunk 并发放大”
日志显示文档被拆分为很多 chunk：
- `docs/fail_logs/20260113_windows_powershell_test.txt:877`：`2.Hello OCS源码剖析.doc` 被 split 成 **41** chunks
- `docs/fail_logs/20260113_windows_powershell_test.txt:1083`：`OCS计费原理与实现(排版后).doc` 被 split 成 **20** chunks

如果 continuation chunk 也带“生成 frontmatter”的指令，必然产生“中间 frontmatter”。

但更重要的是：**这些 chunk 的 LLM 调用很可能是一次性并发打出去的**（并发放大），导致：
- 更容易触发 provider 限流/连接失败/读失败
- 更容易出现 LLM 行为飘（某些 chunk 仍生成 frontmatter）

从实现上看：`src/markit/services/llm_orchestrator.py:325` 的 `create_enhancement_task()` 调 `enhancer.enhance()` **没有传入 semaphore**；而 `src/markit/llm/enhancer.py:501` 在 `enhance()` 内部对 chunks `asyncio.gather()` 并发处理（如果 semaphore 为空则不做限制）。这解释了为什么在 batch 场景里，即使 `llm_concurrency=10`，仍可能出现“一个文档内部瞬间 20~40 个 LLM 请求并发”。

> 这点在 ROADMAP v0.1.5 的“API 错误频繁（低）根因：网络不稳定”里没有被充分识别；实际上它更像是 **“网络不稳定 + 并发放大 + 重试堆叠”** 的组合问题。

### 2.2 “格式未规整”的根因：不仅是“没跑 formatter”，还包括 formatter 规则缺失/不生效
ROADMAP 提到“LLM 增强后未经过 MarkdownFormatter 处理”，但目前实现里仍有两个隐患：

1) **只有 `ConversionPipeline._convert_file_async()` 路径做了 post-processing**  
你现在的 post-processing 逻辑在 `src/markit/core/pipeline.py:655`（`clean_markdown()` + `format_markdown()`）——但 Windows 日志显示它跑的是 “Starting streaming batch processing”，对应的是新的三阶段批处理路径（`create_llm_tasks()` / `finalize_output()`），而 `finalize_output()`（`src/markit/core/pipeline.py:535` 起）并没有调用 `clean_markdown/format_markdown`。

=> 这意味着：**batch 模式很可能仍然产出未规整 Markdown**（与 fail_output 一致）。

2) **`MarkdownFormatter` 本身没有补“标题下空行”**
`src/markit/markdown/formatter.py:124` 的 `_normalize_headings()`只会“标题前加空行”，并没有实现“标题后加空行”。所以即使补上 post-processing，`## 第 1 页` 这种仍可能不满足要求。

反而 `SimpleMarkdownCleaner`（`src/markit/llm/enhancer.py:826`）里明确有“heading 后补空行”的正则，但它只在 LLM enhancement 失败 fallback 时用，不在成功路径里用。

### 2.3 清洗图表残留：规则有了，但 Windows CRLF 可能让 regex 失效
`MarkdownCleaner` 的 chart residue regex（`src/markit/markdown/formatter.py:308`）是基于 `\n` 的。  
但 fail_output 文件明显是 CRLF（Windows），例如 `docs/fail_output/2.Hello OCS源码剖析.doc.md:1` 行尾显示 `\r`。

如果在进入 cleaner 前没有统一把 `\r\n` 归一成 `\n`，则：
- `chart_axis_numbers` 这类依赖 `\n` 的模式可能匹配不到
- frontmatter 清理的 regex（`src/markit/markdown/chunker.py:245`、`src/markit/llm/enhancer.py:704`）也可能因为只写了 `---\n` 而漏删

---

## 3) 修复建议（按优先级，尽量“一次修到位”）

### P0：把“batch 模式输出质量”补齐（这是 fail_output 的直接根因）
1) **在 batch/streaming 路径也做 post-processing**  
建议位置二选一（推荐 1）：
- (推荐) 在 `src/markit/services/llm_orchestrator.py:325` 的 `create_enhancement_task()` 成功返回前，对 `result.content` 做 `clean_markdown()` + `format_markdown()`。
- 或在 `src/markit/core/pipeline.py:535` 的 `finalize_output()` 写文件前统一 clean/format。

这样可以保证 convert/batch 两条路径一致，不会出现“单文件好、批处理差”。

2) **让 `MarkdownFormatter` 真正实现“标题后空行”**
在 `src/markit/markdown/formatter.py:124` 里补齐 heading spacing（标题下方至少 1 个空行），确保 `docs/fail_output/CBS架构演进规划201605 V0.5.pptx.md:28` 这种结构能被自动修正。

### P0：限制“chunk 内部并发”，防止 API 错误被并发放大
3) **给 `MarkdownEnhancer.enhance()` 提供并发上限（semaphore）**
- `src/markit/llm/enhancer.py:468` 已经支持 `semaphore: asyncio.Semaphore | None` 参数
- 但 `src/markit/services/llm_orchestrator.py:325` 调用时没传

建议：在 orchestrator 里持有/创建一个 “chunk-level semaphore”，大小建议：
- `min(llm_concurrency, 4)` 或可配置（更稳）
- 目标是让“一个文档的 chunk 并发”不会把全局并发打爆

这会明显减少 `docs/fail_logs/20260113_windows_powershell_test.txt:1207` 一类错误的发生概率（尤其在 Windows 下）。

### P0：frontmatter 清理要对 CRLF 免疫，且避免误伤
4) **frontmatter 清理前先 normalize line endings**
对以下函数都建议先 `content = content.replace("\r\n", "\n").replace("\r", "\n")`：
- `src/markit/markdown/chunker.py:228` `_remove_chunk_frontmatter()`
- `src/markit/llm/enhancer.py:686` `_remove_intermediate_frontmatter()`
- `src/markit/markdown/formatter.py:326` `MarkdownCleaner.clean()`（否则 chart residue 清理也可能失效）

5) **frontmatter regex 适当收窄，减少误删风险**
当前模式是“看到 `--- ... ---` 就删”，建议加一个轻量判断：块内至少包含 `description:`/`entities:`/`topics:` 等典型字段再判定为 frontmatter，避免误删正文里合法的 `---` 分隔内容。

### P1：清洗策略与提示词进一步落地
6) **将“图表残留清理”从 prompt 变成 deterministic 规则优先**
提示词只是“建议”，最终要靠清洗器兜底。你已经在 `src/markit/markdown/formatter.py:308` 加了 regex，是正确方向；建议再补两类常见垃圾：
- “只有 1~2 个 token 的散点行”（如 `0.0`、`—`、`•`），但要小心误删列表/公式
- 图片前后紧邻的 “坐标轴单位/图例短词”（在图片引用附近局部清理更安全）

### P1：API 错误处理从“重试次数”升级为“策略”
7) **把错误分层：连接类错误快速 fallback，读超时类错误走并发 fallback**
- ConnectError：可以更早触发 fallback（不要在同一个 provider 上堆叠太多重试）
- ReadError：倾向并发 fallback 或降低并发后重试
- 这能降低日志里堆栈刷屏（例如 `docs/fail_logs/20260113_windows_powershell_test.txt:1180` 起那种长堆栈），也能减少整体耗时波动

---

## 4) 建议补充的验证点（防止回归）
- 单测：构造含 CRLF 的多 chunk 内容，验证 `_remove_intermediate_frontmatter()` 和 `_remove_chunk_frontmatter()` 仍能删除中间 YAML。
- 单测：输入 `## Heading\nText`，验证 formatter 输出 `## Heading\n\nText`。
- 集成：走 batch streaming 路径，确认增强后产物不再出现类似 `docs/fail_output/2.Hello OCS源码剖析.doc.md:160` 的中间 frontmatter。

---

如果你希望我下一步继续，我建议你选一个方向：
- 只做“ROADMAP 修复建议补全”（把我上面 P0/P1 精简成可执行 checklist）；或
- 进一步帮你把“batch 模式缺失 post-processing + chunk 并发失控”这两条根因定位到具体调用链（从 `markit batch` 入口一路追到 orchestrator/enhancer）。