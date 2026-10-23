---
pageClass: compact-tables
---

# Why Markitai

There is no single best converter; each of the tools below optimizes for a different job. This page states what markitai does, where another tool is the better choice, and why.

| | **markitai** | markitdown | docling | anydoc |
| --- | --- | --- | --- | --- |
| Engine | Rust, one native program; rule-based readers + optional LLM pipeline | Python; lightweight rule-based converters + plugins | Python; ML layout/table/VLM document-structure models | Rust; zero-ML parsers |
| LLM enhancement | Built-in: format cleaning, frontmatter, vision analysis, per-run JSON cost/usage reports | Optional: image captions, transcription, an OCR plugin | VLM for structure (DocTags), not prose cleanup | None |
| Web pages | Local-first fetch cascade: its own article extractor, then a local Chrome; 3 remote readers only when you opt in | Whole-DOM HTML→Markdown, no main-content pass | Downloads a document URL into the same file pipeline | No URL input; local files/bytes only |
| Scanned docs | Built-in local OCR (PaddleOCR models, or macOS Vision), or `--ocr --llm` to have the vision model read the pages | Optional plugin (LLM-vision or Azure OCR) | Built-in OCR for scanned PDFs/images | None in the OSS library |
| Positioning | Independent project; CLI, local web workspace (EN/中文/日本語), MCP server, Python/Node/Go bindings | Microsoft (AutoGen team); widest ecosystem/plugin adoption | IBM Research origin, now governed by the LF AI & Data Foundation; enterprise RAG building block | Firecrawl open-source; dependency-free, millisecond-scale, 14 formats, Node/Python/WASM bindings |

anydoc goes for dependency-free speed, docling for ML-driven document structure in RAG pipelines, markitdown for ecosystem reach. markitai adds a built-in LLM pipeline, live web fetching, OCR and a local UI on top of native readers.

anydoc also runs inside markitai: it reads Word, Excel, OpenDocument, RTF, EPUB and the legacy `.doc`/`.ppt`/`.xls` files, next to Firecrawl's pdf-inspector for PDFs and markitai's own PowerPoint reader.

## Which one should you use?

| If you need… | Use |
|---|---|
| The fastest possible conversion with no ML or model dependencies | anydoc |
| Document structure (tables, reading order, layout) for a RAG pipeline | docling |
| The widest plugin ecosystem and Microsoft-adjacent integration | markitdown |
| Clean Markdown from files **and** live URLs, optional LLM cleanup, OCR, and a local workspace | markitai |

## License

markitai is [MIT](https://github.com/Ynewtime/markitai/blob/main/LICENSE). Its dependencies are under permissive licences (MIT, Apache-2.0, BSD, ISC, Zlib, Unicode and similar) and, for the four crates of its CSS parser, the file-level MPL-2.0; none is AGPL or GPL. Every release archive and wheel carries the notices; [NOTICE](https://github.com/Ynewtime/markitai/blob/main/NOTICE) has the full attribution.
