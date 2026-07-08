# Markitai 重设计思路（If We Started Over）

> **日期：** 2026-07-08
> **性质：** 架构反思笔记，非执行计划。回答的问题是"如果从零重来会怎么设计"，
> 以及由此推导出"当前仓库最值得渐进演进的方向"。基于 v0.16.0 时点的代码
> （145 个 Python 文件，约 5.5 万行）。

---

## 1. 现状诊断：一个仓库里藏着四个产品

Markitai 当前实际上在同时维护四个各自足以成为独立产品的子系统：

| 子系统 | 位置 | 规模信号 | 本质 |
|---|---|---|---|
| 反爬取引擎 | `fetch.py`（2810 行）、`fetch_playwright.py`、`converter/cloudflare.py`、`domain_profiles.py` | 五级升级链 static → playwright → defuddle → jina → cloudflare | curl-cffi / 浏览器自动化产品的活 |
| Readability 引擎 | `webextract/`（约 40 个文件） | Python 版 defuddle + 站点提取器（X、Bilibili、GitHub、HN、Reddit、YouTube、Steam） | 与 defuddle/trafilatura 同赛道 |
| 文档格式转换器 | `converter/` | 包装 markitdown / pymupdf4llm / rapidocr / kreuzberg | 格式适配层 |
| LLM 编排层 | `llm/`、`providers/` | litellm + instructor + 多家订阅 OAuth + 退化检测 + 缓存 | 多提供商编排产品的活 |

每一块的维护成本都是真实的。**gemini-cli provider 在 2026-07-08 被整体移除**（Google
关闭个人层 onboarding）就是信号：订阅 OAuth 逆向这条线腐烂速度最快，而它与
"把文档转成好 Markdown"这个核心价值的距离最远。

## 2. 如果重来，会改的三件事

### 2.1 核心：引入文档 IR，所有路径汇合到它

**现状问题。** 各条转换路径各自直接产出 Markdown 字符串：

- `webextract/markdown.py` 有一套自己的 DOM→Markdown 渲染器；
- `converter/` 各模块（pdf、office、eml……）各产各的 Markdown；
- 然后 `llm/document.py`（2143 行）再把 Markdown **反向解析回来**做分块和增强——
  等于先扔掉结构再重建结构；
- 质量启发式散落三处：`utils/markdown_quality.py`、`webextract/quality.py`、
  `llm/content.py`，逻辑相似但实现各异。

**重设计。** 定义一个块级中间表示（IR）：

```
Document
├── metadata      (title, source, fetched_at, extractor, …)
├── assets        (images with local path + remote origin)
└── blocks[]      heading | paragraph | code | table | image | embed | footnote
```

所有输入路径（HTML DOM、PDF、docx、eml）先转成 IR；Markdown 只是最后一个
序列化器。收益：

- 质量评分、图片处理、frontmatter、LLM 分块**只写一遍**，只对 IR 操作；
- LLM 增强不再需要"解析自己刚生成的 Markdown"；
- 未来若要支持第二种输出格式（如带 wikilink 的 Obsidian 风格），只加一个序列化器。

### 2.2 站点提取器从代码降级为数据

**现状问题。** `webextract/extractors/x_common.py` 751 行、
`webextract/elements/footnotes.py` 1641 行。站点适配是全项目腐烂最快的部分：
X 改一次 DOM 就要发一个版本。2026-07-08 的 Bilibili opus 事件也暴露了另一面——
当 fetch 链落到 cloudflare `/markdown` 端点时，提取器整个被绕过（见
`handoff-providers-logging-fetch-2026-07-08.md`）。

**重设计。** 提取器拆成两层：

1. **声明层（数据）**：选择器 + 字段映射 + 清洗规则，YAML/JSON 配方。
   改选择器不用发版，社区能贡献，坏了容易 diff 出是哪条选择器失效；
2. **过程层（Python 钩子）**：只有真正需要逻辑的部分（thread 展开、oEmbed
   enrich、分页）才落到代码。

配套约束：**任何 fetch 策略都必须交出 HTML 给同一条 `webextract` 管线**，
不允许出现"某个策略返回已转换的 Markdown 从而绕过提取器"的旁路。

### 2.3 LLM 从"重写者"降级为"修复者"

**现状问题。** 当前模式是 LLM 全文增强，再用 `llm/degeneration.py` 事后检测
它有没有把文档改坏——这是在给自己制造的风险打补丁。

**重设计。** 确定性转换永远产出 baseline；LLM 只做**有界的、可校验的**小任务：

- alt-text 生成（对单张图）；
- 表格修复（对单个 table 块）;
- OCR 纠错（对单个文本块）；
- frontmatter / 摘要（只新增字段，不改正文）。

每个任务的输出都能对着 IR 做结构 diff 校验；全文重写变成显式 opt-in。
退化检测从"必需的安全网"降级为"可选审计"；provider 挂了只是降级而非失败。

## 3. 会刻意保留的设计

这些是项目真正的护城河，重来也照抄：

- **fetch 升级阶梯**（由质量分驱动逐级升级）——方向正确，只需修掉 2.2 说的旁路；
- **doctor / init / auth 这套 CLI 体验**——同类工具里少见的完成度；
- **CAPTCHA 检测并拒绝**而非静默返回挑战页——正确的失败方式；
- **"opinionated 默认值"的定位**——但要言行一致：`config.py` 现有 25+ 个
  config class，很多 knob 是"工具本可自己判断却推给了用户"的决定。
  重来会砍掉约一半配置面。

## 4. 务实路径：不推倒重写

5.5 万行里大部分启发式是真实世界喂出来的经验，重写等于把学费再交一遍。
如果问"当前仓库该怎么走"，最高杠杆的顺序是：

1. **抽出 IR**（唯一的结构性手术）：让 `webextract` 和 `converter` 汇合到
   IR，让 LLM 层不再反向解析 Markdown；
2. **封死提取器旁路**：所有 fetch 策略统一交 HTML（cloudflare 走 `/content`
   而非 `/markdown` 即属此类）；
3. **提取器配方化**：在 IR 骨架上把站点适配逐个迁移为数据；
4. **收缩配置面**：每个 release 挑几个 knob，能自动判断的就删。

其中 1 是前置依赖，2 可独立先做（见 handoff 文档中的待决议线），3、4 可无限期渐进。
