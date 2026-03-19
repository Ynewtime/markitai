# HTML URL Extraction Parity 分支提交评审报告

评审时间: 2026-03-19

评审对象:

- 基线提交: `cd44f95837e01e5deda8ffa6a713330a451026a3`
- 评审范围: `cd44f95..feature/html-url-extraction-parity`
- 说明: 当前 `main` 停在 `cd44f95`，后续 15 个提交位于 `feature/html-url-extraction-parity`

## 1. 评审范围与方法

本次评审聚焦 `webextract/` 新增的 resolver、semantic、quality、enricher、thread policy、YouTube/Reddit/GitHub/Hacker News 提取器，以及它们与 `fetch_playwright.py` 的集成闭环。

提交序列如下:

1. `943ed45` test: freeze semantic parity expectations for threaded pages
2. `ed85f36` refactor: add typed extraction result and frontmatter builder
3. `b8ba63a` refactor: add resolver layer for structured site extraction
4. `1098ea7` feat: add shared semantic model for threaded extractions
5. `b0b36cc` feat: rebuild x extraction on shared thread semantics
6. `a7d7fc3` feat: validate shared thread extraction with github discussions
7. `4ddcef1` feat: add typed native extraction quality profiles
8. `6b22a15` feat: add policy-aware async enrichers to resolver pipeline
9. `3a94e5d` feat: split raw html preprocess from browser dom normalization
10. `3b6f686` feat: add narrow markdown fidelity layer for canonical content
11. `1b2853e` feat: expand threaded extraction coverage on shared abstractions
12. `622b8bf` feat: add youtube page extraction on native resolver contract
13. `31501e2` test: add parity, diagnostics, and benchmark guardrails for native extraction
14. `0d08249` docs: explain native extraction architecture and migration policy
15. `277e68e` feat: add thread policy module for threaded page inclusion rules

本次评审基于:

- 静态审查:
  - `packages/markitai/src/markitai/webextract/`
  - `packages/markitai/src/markitai/fetch_playwright.py`
  - 对应 unit/integration tests 与 fixtures
- 新鲜测试:
  - `uv run pytest packages/markitai/tests/unit/webextract/test_resolver.py packages/markitai/tests/unit/webextract/test_reddit_post.py packages/markitai/tests/unit/webextract/test_youtube_page.py packages/markitai/tests/unit/webextract/test_frontmatter_builder.py packages/markitai/tests/unit/webextract/test_quality_profiles.py packages/markitai/tests/unit/webextract/test_async_resolver.py`
    - 结果: `88 passed`
  - `uv run pytest packages/markitai/tests/integration/test_defuddle_parity.py packages/markitai/tests/integration/test_webextract_benchmarks.py`
    - 结果: `11 passed`
- 定向事实核查:
  - 用最小 HTML 复现 resolver 清洗缺口
  - 用最小 HTML 复现 `ResolvedPage.content_root` 未生效
  - 用最小 old Reddit 结构复现嵌套回复丢失
  - 用源码搜索核查 async enricher、frontmatter builder、thread policy 是否接入生产路径

## 2. 总体结论

这个分支在架构方向上是对的:

- resolver/semantic/render/type/profile 的分层比旧的站点特判更清楚
- 站点级 fixture、parity test 和 benchmark guardrail 让后续演进更有边界
- YouTube 非线程页、GitHub/Reddit/HN 线程页的抽象方向也基本成立

但当前实现仍存在几处“单元测试通过、真实产品路径没有闭环”的问题，集中在两类:

1. resolver 新增契约与主流水线之间存在集成断层
2. 新增能力已经落成模块和测试，但尚未真正接入 URL 转换主路径

从优先级看，最该先修的是:

1. resolver 分支未走标准化/清洗
2. `ResolvedPage.content_root` 契约无效
3. Reddit 嵌套回复在真实 DOM 形态下会漏抽

## 3. 详细问题

### High-1: resolver 分支绕过标准化和清洗，导致 `clean_html` 与 Markdown 语义不一致

严重度: High

引入位置:

- 主要由 `b8ba63a` 的 resolver 路径引入
- 受 `3b6f686` 的 markdown fidelity 层影响更明显

现象:

- generic 路径会在 `_extract_once()` 中对根节点执行:
  - `standardize_content(...)`
  - `sanitize_tag_tree(...)`
- resolver 路径 `_build_from_resolved()` 直接把 `ResolvedPage.content_html` 传给 `render_markdown()`，没有任何等价清洗步骤

证据:

- `packages/markitai/src/markitai/webextract/pipeline.py:73-77`
- `packages/markitai/src/markitai/webextract/pipeline.py:203-207`

本地复现:

- 以 `content_html='<article><script>alert(1)</script><p>Hello <a href="/rel">world</a></p></article>'` 构造 `ResolvedPage`
- `_build_from_resolved()` 返回结果中:
  - `clean_html` 仍包含 `<script>`
  - Markdown 仍保留相对链接 `[world](/rel)`

影响:

- resolver-backed 页面不会得到 generic 路径已有的 HTML 规范化收益
- `clean_html` 作为“canonical representation”的定义被破坏
- 站点提取器只要产出相对链接或未清洗片段，最终输出就会与 generic 路径行为分叉

为什么是问题:

- 这不是“风格不一致”，而是同一 pipeline 对两条输入路径执行了不同的数据安全和规范化语义
- 文档和类型注释都把 `clean_html` 描述成规范化后的 canonical HTML，但 resolver 路径目前达不到这个承诺

建议:

- 在 `_build_from_resolved()` 内对 resolver 产出的 HTML 做和 generic 路径等价的标准化/清洗
- 至少补 2 类回归测试:
  - resolver 输出相对链接时应被标准化
  - resolver 输出危险标签时 `clean_html` 不应原样保留

### High-2: `ResolvedPage.content_root` 合约当前无效

严重度: High

引入位置:

- `b8ba63a`

现象:

- `ResolvedPage` 数据模型和注释明确允许 resolver 返回 `content_root` 或 `content_html`
- 但 `extract_web_content()` 只有在 `resolved is not None and resolved.content_html` 时才走 resolver 分支
- 这意味着 root-only resolver 会被静默当成“没有 resolver 结果”，直接回退到 generic pipeline

证据:

- `packages/markitai/src/markitai/webextract/resolver.py:35-43`
- `packages/markitai/src/markitai/webextract/pipeline.py:41-47`

本地复现:

- 自定义一个只返回 `ResolvedPage(content_root=soup.find("article"), metadata_overrides={"title": "Override"})` 的 resolver
- `resolve_page()` 确实返回了 `content_root`
- 但主流水线不会消费它，最后仍使用 generic 路径结果

影响:

- `metadata_overrides`、`semantic`、`diagnostics` 会一并失效
- 当前 contract 对未来扩展者具有误导性: 写出合法 root-only resolver，也不会生效

为什么是问题:

- 这是“类型和实现不一致”
- 当前不是未完成接口那么简单，而是已经对外暴露了一个不会工作的能力面

建议:

- 二选一:
  - 实现 `content_root` 路径
  - 或收缩 contract，只允许 `content_html`
- 若保留双形态 contract，必须补一条 root-only resolver 的端到端测试

### High-3: Reddit 嵌套回复在真实 old Reddit DOM 形态下会漏抽

严重度: High

引入位置:

- `1b2853e`

现象:

- `_collect_comment_nodes()` 递归时从 `entry.find("div", class_="child")` 向下走
- 但 old Reddit 常见结构里，`.child` 是 `.entry` 的兄弟节点，而不是子节点
- 当前测试 fixture 把 `.child` 放进了 `.entry` 内，因此未暴露这个问题

证据:

- `packages/markitai/src/markitai/webextract/extractors/reddit_post.py:257-283`
- `packages/markitai/tests/fixtures/web/reddit_post.playwright.html:54`

本地复现:

- 构造一个更贴近 old Reddit 的最小结构:
  - 父评论 `div.thing.comment`
  - 其下 `div.entry`
  - `div.child` 作为 `div.entry` 的兄弟节点
- 结果只抽出父评论，子评论完全丢失

影响:

- 分支声称支持 Reddit thread 语义，但在真实 DOM 形态下只能得到浅层 thread
- 这会直接破坏 `ConversationThread.items` 的完整性和 `parent_id` 关系

为什么是问题:

- 这是实际内容丢失，不是排版问题
- 它会把“线程提取”退化为“只提取顶层评论”

建议:

- 递归入口不应只从 `entry` 内寻找 `.child`
- 应支持:
  - `.child` 是 `entry` 子节点
  - `.child` 是 comment container 下、与 `entry` 并列的兄弟节点
- 补一条更贴近 old Reddit 实际结构的 fixture/test，避免继续被当前 fixture 误导

### Medium-1: async enricher 已实现并有测试，但未接入生产主路径

严重度: Medium

引入位置:

- `6b22a15`

现象:

- 分支新增了 `resolve_page_async()`、`EnrichmentPolicy` 和 `XOEmbedEnricher`
- 但生产代码路径仍使用同步 `resolve_page()`:
  - `extract_web_content()` 只调用 `resolve_page()`
  - `fetch_playwright.py` 只调用同步 `extract_web_content()`
- 源码搜索显示 `resolve_page_async()` 的调用点仅存在于测试和模块内部

证据:

- `packages/markitai/src/markitai/webextract/resolver.py:107-168`
- `packages/markitai/src/markitai/webextract/pipeline.py:41-47`
- `packages/markitai/src/markitai/fetch_playwright.py:528-553`

影响:

- commit message 和模块设计中“policy-aware async enrichers”已落地，但真实 URL 抓取不会触发它们
- 当前 enricher 体系的产品价值仍停留在测试级，而不是运行时级

为什么是问题:

- 这属于已实现模块未被编排层消费
- 如果团队以为 oEmbed 或其他 enrichers 已能在真实抓取中生效，会形成错误预期

建议:

- 明确二选一:
  - 近期接入 `extract_web_content` / fetch orchestration
  - 或在文档/注释中降级表述，避免暗示已生效
- 补一条接入层测试，而不是只测 resolver 单元测试

### Medium-2: `build_source_frontmatter()` 未接入真实 URL 转换链路

严重度: Medium

引入位置:

- `ed85f36`

现象:

- `build_source_frontmatter()` 明确会导出:
  - `word_count`
  - `content_profile`
- 但当前 URL 主路径仍然只对 `extracted.metadata` 调 `coerce_source_frontmatter()`
- 全仓库搜索显示 `build_source_frontmatter()` 只在模块导出和单测中出现，没有生产调用点

证据:

- `packages/markitai/src/markitai/webextract/frontmatter.py:8-36`
- `packages/markitai/src/markitai/fetch_playwright.py:544-548`
- `packages/markitai/src/markitai/webextract/__init__.py:22-43`

影响:

- 新增类型化 frontmatter builder 的核心增量没有进入真实输出
- 用户经由 URL 转换得到的 frontmatter 仍缺少本分支新增的 `word_count` / `content_profile`

为什么是问题:

- 这意味着“新增 builder 的测试是绿的”并不等于“真实输出变了”
- 从产品角度看，这个提交的可见效果尚未兑现

建议:

- 在 URL 转换真实路径里使用 `build_source_frontmatter(result)`
- `coerce_source_frontmatter()` 只保留给兼容旧对象的兜底路径

### Medium-3: thread policy 模块已落地，但当前没有任何生产消费点

严重度: Medium

引入位置:

- `277e68e`

现象:

- `thread_policy.py` 新增了 `ThreadPolicy` 和 `get_thread_policy(url)`
- 但源码搜索显示 `get_thread_policy()` 只在自身模块和测试中出现，extractor、pipeline、render、fetch 均未消费它

证据:

- `packages/markitai/src/markitai/webextract/thread_policy.py:43-60`
- `packages/markitai/tests/unit/webextract/test_thread_policy.py:28-49`

影响:

- 当前“thread inclusion policy”仍是声明式能力，不影响真实提取结果
- 对 X/Reddit/GitHub/HN 的 include/exclude 规则没有实际运行时约束

为什么是问题:

- 该模块不是单纯预留类型，而是已经以 `feat:` 形式交付
- 但目前它只能证明“默认值可以返回”，不能证明“线程包含规则已生效”

建议:

- 若这是后续阶段能力，建议在文档中标注“已定义 contract，尚未接入 extractor”
- 若希望本分支就算交付，至少应让一个 extractor 或 pipeline 层实际读取 policy

## 4. 事实核查

本报告只保留了经过以下任一方式确认的结论:

1. 源码直接确认:
   - async enricher 无生产调用点
   - frontmatter builder 无生产调用点
   - thread policy 无生产调用点
2. 最小复现确认:
   - resolver 分支未清洗/未标准化
   - `content_root` contract 无效
   - Reddit 嵌套回复漏抽
3. 新鲜测试确认:
   - 本分支新增 unit/integration 测试当前均为绿

本次评审还做了反向核查，排除了一个最初怀疑但最终不成立的点:

- “YouTube `RICH_MEDIA_PAGE` 可能会因为 quality profile 未注册而被 generic 规则误拒”
  - 事实核查结果: 当前实现下，典型 YouTube resolver 输出能通过 generic profile
  - 结论: 这是架构一致性欠佳，但不是已确认的当前故障，因此未列为正式 finding

## 5. 风险优先级建议

P0:

1. 修复 resolver 路径缺失的标准化/清洗
2. 统一 `ResolvedPage` contract 与 pipeline 实现
3. 修复 Reddit 嵌套回复递归逻辑，并补真实结构 fixture

P1:

1. 把 async enricher 接入真实抓取路径，或明确降级为“实验性/未接线”
2. 把 `build_source_frontmatter()` 接进 URL 输出链路
3. 明确 thread policy 是 contract-only 还是 runtime feature

## 6. 结论

这个分支最大的价值不是“已经完全达到 parity”，而是把 native extraction 的骨架搭出来了。当前主要问题不是抽象方向错，而是若干新增抽象还停留在“模块层成立、产品层未闭环”的状态。

如果要把这一轮工作视为可合并状态，我建议至少先解决 3 个 High 问题；否则后续迭代会在一个已经分层但运行时语义不一致的基础上继续叠功能，修复成本会越来越高。
