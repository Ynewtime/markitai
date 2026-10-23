---
pageClass: compact-tables
---

# 为什么选 Markitai

没有哪个转换器是全面最优的，下面这些工具各自优化的目标不同。本页说明 markitai 做了什么、什么时候该选别的工具以及原因。

| | **markitai** | markitdown | docling | anydoc |
| --- | --- | --- | --- | --- |
| 引擎 | Rust，一个原生程序；规则读取器 + 可选 LLM 管线 | Python；轻量规则转换器 + 插件 | Python；ML 版面/表格/VLM 文档结构模型 | Rust；零 ML 解析器 |
| LLM 增强 | 内置：格式清洗、frontmatter、视觉分析、每次运行的 JSON 成本/用量报告 | 可选：图片说明、转写、OCR 插件 | 用 VLM 还原结构（DocTags），不做正文清洗 | 无 |
| 网页 | 本地优先的抓取级联：自带的正文抽取器，然后是本机 Chrome；3 个远程阅读服务只在你主动开启时使用 | 整份 DOM 直接 HTML→Markdown，没有主内容抽取 | 把文档 URL 下载进同一套文件管线 | 不支持 URL 输入；只处理本地文件/字节 |
| 扫描件 | 内置本地 OCR（PaddleOCR 模型或 macOS Vision），或 `--ocr --llm` 让视觉模型直接读页面 | 可选插件（LLM 视觉或 Azure OCR） | 内置扫描 PDF/图片 OCR | 开源库里没有 |
| 定位 | 独立项目；CLI、本地网页工作台（EN/中文/日本語）、MCP 服务器、Python/Node/Go 绑定 | Microsoft（AutoGen 团队）；生态与插件采用面最广 | 源自 IBM Research，现由 LF AI & Data Foundation 治理；企业 RAG 构建块 | Firecrawl 开源；零依赖、毫秒级、14 种格式，提供 Node/Python/WASM 绑定 |

anydoc 追求零依赖的速度，docling 追求 RAG 管线里的 ML 文档结构，markitdown 追求生态覆盖。markitai 在原生读取器之上加了内置 LLM 管线、实时网页抓取、OCR 和本地界面。

anydoc 也跑在 markitai 内部：它读取 Word、Excel、OpenDocument、RTF、EPUB 和旧版 `.doc`/`.ppt`/`.xls` 文件；PDF 由 Firecrawl 的 pdf-inspector 读取，PowerPoint 由 markitai 自己的读取器处理。

## 该选哪一个？

| 如果你需要…… | 用 |
|---|---|
| 无 ML、无模型依赖的最快转换 | anydoc |
| 面向 RAG 管线的文档结构（表格、阅读顺序、版面） | docling |
| 最广的插件生态与靠近 Microsoft 的集成 | markitdown |
| 从文件**和**实时 URL 得到干净的 Markdown，可选 LLM 清洗、OCR，以及本地工作台 | markitai |

## 许可

markitai 采用 [MIT](https://github.com/Ynewtime/markitai/blob/main/LICENSE) 许可。它的依赖都是宽松许可（MIT、Apache-2.0、BSD、ISC、Zlib、Unicode 等），其 CSS 解析器的四个 crate 采用文件级的 MPL-2.0；没有任何 AGPL 或 GPL 依赖。每个发布包和 wheel 都附带这些声明；完整的归属见 [NOTICE](https://github.com/Ynewtime/markitai/blob/main/NOTICE)。
