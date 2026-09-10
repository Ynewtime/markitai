---
pageClass: compact-tables
---

# 为什么选 Markitai

没有哪个转换器是全面最优的——下面这些工具各自优化的目标不同。本页说明 markitai 做了什么、什么时候该选别的工具以及原因，不列那些很快就会过期的 star 数和下载量。

| | **markitai** | markitdown | docling | anydoc |
| --- | --- | --- | --- | --- |
| 引擎 | Python；规则转换 + 可选 LLM 管线 | Python；轻量规则转换器 + 插件 | Python；ML 版面/表格/VLM 文档结构模型 | Rust；零 ML 解析器 |
| LLM 增强 | 内置：格式清洗、frontmatter、视觉分析、每次运行的 JSON 成本/用量报告 | 可选：图片说明、转写、OCR 插件 | 用 VLM 还原结构（DocTags），不做正文清洗 | 无 |
| 网页 | 5 级抓取级联、本地优先；static 先跑一份从零移植的 [defuddle](https://github.com/kepano/defuddle) 可读性算法，再回退到浏览器或 3 个远程 API | 整份 DOM 直接 HTML→Markdown，没有主内容抽取 | 把文档 URL 下载进同一套文件管线 | 不支持 URL 输入——只处理本地文件/字节 |
| 扫描件 | 可选本地 OCR（`markitai[ocr]`，RapidOCR），或 `--ocr --llm` 让视觉模型直接读页面 | 可选插件（LLM 视觉或 Azure OCR） | 内置扫描 PDF/图片 OCR | 开源库里没有 |
| 定位 | 独立项目；CLI + 本地双语（EN/中文）网页工作台 | Microsoft（AutoGen 团队）；生态与插件采用面最广 | 源自 IBM Research，现由 LF AI & Data Foundation 治理；企业 RAG 构建块 | Firecrawl 开源；零依赖、毫秒级、14 种格式，提供 Node/Python/WASM 绑定 |

每个工具优化的目标不同：anydoc 追求零依赖的速度，docling 追求 RAG 管线里的 ML 文档结构，markitdown 追求生态覆盖——markitai 用这些换来内置 LLM 管线、实时网页抓取和本地界面。

其中两个还同时是依赖而非仅是对手：Office 格式由 markitdown 转换，旧版 `.doc`/`.ppt` 由 `markitai[legacy]` 背后的 anydoc 处理。

## 该选哪一个？

| 如果你需要…… | 用 |
|---|---|
| 无 ML、无模型依赖的最快转换 | anydoc |
| 面向 RAG 管线的文档结构（表格、阅读顺序、版面） | docling |
| 最广的插件生态与靠近 Microsoft 的集成 | markitdown |
| 从文件**和**实时 URL 得到干净的 Markdown，可选 LLM 清洗、OCR，以及本地工作台 | markitai |

## 许可

markitai 自身源码为 [MIT](https://github.com/Ynewtime/markitai/blob/main/LICENSE)。默认安装并非全部 MIT：PDF 引擎是 Artifex Software 的 PyMuPDF，采用 **AGPL-3.0 或商业**双许可。本地使用不受影响；再分发这一组合作品，或通过网络向他人提供服务，则触发 AGPL-3.0 义务。完整的第三方归属与依赖清单见 [NOTICE](https://github.com/Ynewtime/markitai/blob/main/NOTICE)。
