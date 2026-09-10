---
pageClass: compact-tables
---

# Why Markitai

There is no single best converter — each of the tools below optimizes for a different job. This page states what markitai does, where another tool is the better choice, and why, without star counts or download numbers that go stale.

| | **markitai** | markitdown | docling | anydoc |
| --- | --- | --- | --- | --- |
| Engine | Python; rule-based conversion + optional LLM pipeline | Python; lightweight rule-based converters + plugins | Python; ML layout/table/VLM document-structure models | Rust; zero-ML parsers |
| LLM enhancement | Built-in: format cleaning, frontmatter, vision analysis, per-run JSON cost/usage reports | Optional: image captions, transcription, an OCR plugin | VLM for structure (DocTags), not prose cleanup | None |
| Web pages | 5-strategy fetch cascade, local-first; static runs a from-scratch port of [defuddle](https://github.com/kepano/defuddle)'s readability algorithm before falling back to a browser or 3 remote APIs | Whole-DOM HTML→Markdown, no main-content pass | Downloads a document URL into the same file pipeline | No URL input — local files/bytes only |
| Scanned docs | Optional local OCR (`markitai[ocr]`, RapidOCR), or `--ocr --llm` to have the vision model read the pages | Optional plugin (LLM-vision or Azure OCR) | Built-in OCR for scanned PDFs/images | None in the OSS library |
| Positioning | Independent project; CLI + local bilingual (EN/中文) web workspace | Microsoft (AutoGen team); widest ecosystem/plugin adoption | IBM Research origin, now governed by the LF AI & Data Foundation; enterprise RAG building block | Firecrawl open-source; dependency-free, millisecond-scale, 14 formats, Node/Python/WASM bindings |

Each optimizes for a different job: anydoc for dependency-free speed, docling for ML-driven document structure in RAG pipelines, markitdown for ecosystem reach — markitai trades those for a built-in LLM pipeline, live web fetching, and a local UI.

Two of them are also dependencies rather than only alternatives: markitdown converts the Office formats, and anydoc handles legacy `.doc`/`.ppt` behind `markitai[legacy]`.

## Which one should you use?

| If you need… | Use |
|---|---|
| The fastest possible conversion with no ML or model dependencies | anydoc |
| Document structure (tables, reading order, layout) for a RAG pipeline | docling |
| The widest plugin ecosystem and Microsoft-adjacent integration | markitdown |
| Clean Markdown from files **and** live URLs, optional LLM cleanup, OCR, and a local workspace | markitai |

## License

markitai's own source code is [MIT](https://github.com/Ynewtime/markitai/blob/main/LICENSE). The default installation is not uniformly MIT: the PDF engine is PyMuPDF from Artifex Software, dual-licensed **AGPL-3.0 or commercial**. Local use is unaffected; redistributing the combined work or offering it to others over a network triggers AGPL-3.0 obligations. Full attribution and the exact dependency list are in [NOTICE](https://github.com/Ynewtime/markitai/blob/main/NOTICE).
