# Markitai documentation

Markitai 1.3.0-dev converts documents, web pages and images to Markdown. A
single Rust core powers the `markitai` command (alias `mkai`), a local REST
service with a browser workspace, an MCP server and in-process Node.js, Python
and Go bindings. It is the development build of a rewrite of reference release
1.2.0 and is not yet release-ready; [compatibility](compatibility.md) lists the
reference contracts and [project status](STATUS.md) the delivered scope.

## Start here

- [Quick start](quickstart.md): build, first conversion, folders, web pages,
  OCR, language models, workspace and MCP.
- [Troubleshooting](troubleshooting.md): `doctor`, exit codes and common errors.
- [CLI](cli.md): every option and subcommand.
- [Configuration](configuration.md): file locations, `config`/`init`, environment
  variables and defaults.
- [Bindings](bindings.md): Node.js, Python and Go installation and API.

## Common tasks

| Task | Command | Read more |
|---|---|---|
| Convert one file to stdout | `markitai report.docx` | [Quick start](quickstart.md#3-convert-your-first-document) |
| Write files instead | `markitai report.docx -o out/` | [Output](output.md) |
| Convert a folder | `markitai ./docs -o out/ -g '**/*.pdf'` | [CLI](cli.md), [reports](reports.md) |
| Resume an interrupted batch | `markitai ./docs -o out/ --resume` | [Recovery storage](state-storage.md) |
| Convert a web page | `markitai https://example.com -o out/` | [URL fetching](fetch.md) |
| Render a JavaScript page | `markitai URL -s playwright -o out/` | [Browser](browser.md), [installation](browser-installation.md) |
| OCR a scan (macOS) | `markitai scan.pdf --ocr -o out/` | [Local OCR](ocr.md), [PDF OCR](pdf-ocr.md) |
| Page or slide images | `markitai deck.pdf --screenshot -o out/` | [PDF rendering](pdf-rendering.md), [Office rendering](office-rendering.md) |
| Enhance with a model | `markitai report.pdf --llm -o out/` | [LLM](llm.md), [pricing](pricing.md) |
| Image captions | `markitai report.docx --llm --alt --desc -o out/` | [Image enrichment](image-enrichment.md) |
| RAG / Obsidian layout | `markitai report.docx --profile rag -o out/` | [Output](output.md) |
| Machine-readable result | `markitai report.docx -o out/ --json` | [CLI](cli.md) |
| Browser workspace | `markitai serve` | [REST service](serve.md), [workspace](web-ui.md) |
| MCP server | `markitai mcp` | [MCP](mcp.md) |
| Check the installation | `markitai doctor` | [Troubleshooting](troubleshooting.md) |
| Inspect or clear caches | `markitai cache stats` | [Cache](cache.md) |

## Formats

- [Formats](formats.md): native readers, fidelity and known gaps.
- [PDF layout](pdf.md), [PDF rendering](pdf-rendering.md) and [page OCR](pdf-ocr.md).
- [Office page rendering](office-rendering.md): optional LibreOffice export,
  full-page images and OCR.
- [Numbers](numbers.md): tables from Apple Numbers files.
- [HTML](html.md), [HTML code blocks](html-code.md),
  [article boundaries](html-article.md), [streamed HTML](html-stream.md) and
  [CSS resources](css-resources.md).
- [EML](eml.md) and [MSG](msg.md) e-mail.
- [Markup](markup.md): reStructuredText, Org and TeX.
- [Images](images.md): raster, SVG, HEIF/AVIF and shared asset handling.
- [Local OCR](ocr.md): macOS Vision recognition and languages.

## Web pages

- [URL fetching](fetch.md): strategies, page cache, PDF downloads and proxies.
- [Browser fetching](browser.md) and [browser installation](browser-installation.md).

## Language models

- [LLM](llm.md): providers, routing, retries, budgets and structured output.
- [Image enrichment](image-enrichment.md): captions and descriptions.
- [Pricing](pricing.md): price catalog and dollar limits.
- [Subscriptions](subscriptions.md) and [ChatGPT adapter](subscription-chatgpt.md):
  GitHub Copilot, Claude and ChatGPT through their official runtimes.
- [Provider Batch](provider-batch.md): OpenAI Batch API for folders.
- [Cache](cache.md): reuse of model answers.

## Output and records

- [Output](output.md): frontmatter, pure mode, file names and assets.
- [Grouped publication](grouped-publication.md) and
  [output ownership](output-ownership.md): how results are written safely.
- [Reports](reports.md): batch reports.
- [History](history.md): optional run archives shown by `serve`.
- [Recovery storage](state-storage.md): batch resume state.

## Services and bindings

- [REST service](serve.md) and [browser workspace](web-ui.md).
- [Service settings](service-settings.md) and
  [provider management](provider-management.md): saved connections, model
  discovery and connection tests.
- [MCP](mcp.md): stdio tools for AI assistants.
- [Bindings](bindings.md): Node.js, Python, Go and the C ABI.

## Project and maintenance

These pages are for contributors and reviewers rather than users.

- [Control center](CONTROL.md), [status](STATUS.md) and
  [remaining work](remaining-work.md): scope, checkpoints and the work queue.
- [Architecture](architecture.md) and [development](development.md).
- [Compatibility](compatibility.md): contracts of reference release 1.2.0.
- [Native CI](ci.md): package builds and installed-package checks.
- [Validation](validation/README.md): measured evidence for each round.
- [Performance plan](performance-plan.md).
- Decisions: [native core](decisions/0001-native-core.md),
  [document LLM cache](decisions/0002-persistent-llm-cache.md),
  [static-page fetch cache](decisions/0003-persistent-fetch-cache.md) and
  [run reports, recovery and history](decisions/0004-run-persistence.md).
- [Control history through R31](control-history-through-r31.md).

Pages describe verified behavior and name the gaps that remain; a documented
option that is not implemented returns an explicit error.
