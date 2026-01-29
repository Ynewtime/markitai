# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-01-28

### Added

- **Claude Agent SDK**: `claude-agent/sonnet|opus|haiku` via Claude Code CLI
- **GitHub Copilot SDK**: `copilot/claude-sonnet-4.5|gpt-4o|o1` models
- **URL HTTP caching**: ETag/Last-Modified conditional requests
- **Quiet mode**: `--quiet` / `-q` flag (auto-enabled for single file)
- **Module refactoring**: `cli.py` → `cli/`, `llm.py` → `llm/`, new `providers/`
- **Setup scripts hardening**: default N for high-impact ops, version pinning
- **Docs**: CONTRIBUTING.md, architecture.md, ai-tools-setup.md, dependabot.yml

### Changed

- Python 3.13, docs reorganized to `docs/archive/`
- agent-browser locked to 0.7.6 (Windows bug in 0.8.x)
- Default `extra_wait_ms`: 1000 → 3000, Instructor mode: `JSON` → `MD_JSON`

### Fixed

- **Windows**: UTF-8 console, Copilot CLI path discovery, script argument quoting
- **LLM**: Frontmatter regex fallback, `source` field fix, vision/frontmatter error handling
- **Prompts**: Enhanced prompt leakage prevention, placeholder protection rules
- **Content**: Social media cleanup rules (X/Twitter, Facebook, Instagram)
- **Setup**: WSL detection, Python pymanager support, PATH refresh order

## [0.3.2] - 2026-01-27

### Added

- Chinese README (`README_ZH.md`) with language toggle
- Chinese setup scripts: `setup-zh.sh`, `setup-zh.ps1`, `setup-dev-zh.sh`, `setup-dev-zh.ps1`

### Changed

- Improved setup scripts with better error handling and user feedback
- Updated Python version note: 3.11-3.13 (3.14 not yet supported)
- Updated documentation language toggle links

## [0.3.1] - 2026-01-27

### Fixed

#### Prompt Leakage Prevention
- Split all prompts into `*_system.md` (role definition) and `*_user.md` (content template)
- Added `_validate_no_prompt_leakage()` to detect and handle prompt leakage in LLM output
- Updated LLM calls to use proper `[{"role": "system"}, {"role": "user"}]` message structure

#### LLM Compatibility
- Fixed `max_tokens` exceeding deepseek limit by using minimum across all router models
- Fixed terminal window popup on Windows when running agent-browser verification

#### URL Fetching
- Improved error messages for browser fetch timeout (no longer suggests installing when already attempted)
- Added auto-proxy detection for Jina API and browser fetching
  - Checks environment variables: `HTTPS_PROXY`, `HTTP_PROXY`, `ALL_PROXY`
  - Auto-detects local proxy ports: 7890 (Clash), 10808 (V2Ray), 1080 (SOCKS5), etc.

### Added

#### SPA Domain Learning
- New `SPADomainCache` for automatic detection and caching of JavaScript-heavy sites
- `markitai cache spa-domains` command to view/manage learned domains
- `markitai cache clear --include-spa-domains` option

#### Windows Performance Optimizations
- Thread pool optimization: Windows defaults to 4 workers (vs 8 on Linux/macOS)
- ONNX Runtime global singleton with preheat for OCR engine
- OpenCV-based image compression (releases GIL, 20-40% faster)
- Batch subprocess execution for agent-browser commands

### Changed

- Default image quality: 85 → 75
- Default image max_height: 1080 → 99999 (effectively unlimited)
- Default image min_area filter: 2500 → 5000
- Default URL concurrency: 3 → 5
- Default scan_max_depth: 10 → 5
- Extended fallback_patterns with more social media domains

## [0.3.0] - 2026-01-26

### Added

#### URL Conversion Support
- **Direct URL conversion**: `markitai <url>` converts web pages to Markdown
- **URL batch processing**: Support `.urls` file format (text or JSON), auto-detected from input
- **URL image downloading**: `download_url_images()` with concurrent downloads (5 parallel)
- Automatic relative URL resolution for images
- Cross-platform filename sanitization (Windows illegal characters handling)

#### Multi-Source URL Fetching (fetch.py)
- **Three fetch strategies**: `--static` / `--agent-browser` / `--jina`
  - `static`: MarkItDown direct HTTP fetch (default, fastest)
  - `browser`: agent-browser headless rendering (for JS-heavy pages)
  - `jina`: Jina Reader API (cloud-based, no local deps)
  - `auto`: Smart fallback (static → browser/jina if JS detected)
- **FetchCache**: SQLite-based URL cache with LRU eviction (100MB default)
- **Screenshot capture**: `--screenshot` for full-page screenshots via browser
- **Multi-source content**: Parallel static + browser fetch with quality validation
- Domain pattern matching for auto-browser fallback (x.com, twitter.com, etc.)
- `FetchResult` with `static_content`, `browser_content`, `screenshot_path`

#### agent-browser Integration
- Headless browser automation via `agent-browser` CLI
- Configurable wait states: `load`, `domcontentloaded`, `networkidle`
- Extra wait time for SPA rendering (`extra_wait_ms`)
- Session isolation for concurrent fetches
- `verify_agent_browser_ready()` with cached readiness check
- Screenshot compression with Pillow (JPEG quality + max height)

#### URL LLM Enhancement
- New `prompts/url_enhance.md` for URL-specific content cleaning
- Multi-source LLM processing: combine static + browser + screenshot
- Smart content selection based on validity detection

#### Cache Enhancements
- **`--no-cache-for <pattern>`**: Selective cache bypass with glob patterns
  - Single file: `--no-cache-for file1.pdf`
  - Glob pattern: `--no-cache-for "*.pdf"`
  - Mixed: `--no-cache-for "*.pdf,reports/**"`
- **`markitai cache stats -v`**: Verbose mode with detailed cache entries
- **`--limit N`**: Control number of entries in verbose output (default: 20)
- **`--scope project|global|all`**: Filter cache statistics by scope
- `SQLiteCache.list_entries()`: List cache entries with metadata
- `SQLiteCache.stats_by_model()`: Per-model cache statistics
- Improved cache hash: head + tail + length algorithm for better invalidation

#### Workflow Core Refactor (workflow/core.py)
- **`ConversionContext`**: Unified single-file conversion context
- **`convert_document_core()`**: Main conversion pipeline
  - `validate_and_detect_format()` → `convert_document()` → `process_embedded_images()`
  - `write_base_markdown()` → `process_with_vision_llm()` / `process_with_standard_llm()`
- Parallel document + image processing with proper dependency handling
- Alt text injection after LLM processing completes (race condition fix)

#### Official Website
- VitePress 2.x documentation site with bilingual support (English/Chinese)
- Custom theme with brand colors matching logo
- Local search integration
- GitHub Actions auto-deployment to GitHub Pages

#### Project
- **MIT License**: Added LICENSE file

#### CI/CD
- **`.github/workflows/ci.yml`**: Automated testing on push/PR
- **`.github/workflows/deploy-website.yml`**: Website deployment to GitHub Pages

#### Code Architecture
- New `utils/paths.py`: `ensure_dir()`, `ensure_subdir()`, `ensure_assets_dir()`
- New `utils/mime.py`: `get_mime_type()`, `get_extension_from_mime()`
- New `utils/text.py`: `normalize_markdown_whitespace()`, text utilities
- New `utils/executor.py`: `run_in_executor()` with shared ThreadPoolExecutor
- New `utils/output.py`: Output formatting helpers
- New `json_order.py`: Ordered JSON serialization for reports/state files
- New `urls.py`: `.urls` file parser (JSON and plain text formats)
- `LLMUsageAccumulator` class for centralized cost tracking
- `create_llm_processor()` factory function
- Unified `detect_language()` with `get_language_name()` helper
- Centralized `IMAGE_EXTENSIONS`, `JS_REQUIRED_PATTERNS` constants

#### Configuration
- **`supports_vision` now optional**: Auto-detected from litellm when not explicitly set
  - No need to manually configure for most models (GPT-4o, Gemini, Claude, etc.)
  - Explicit `supports_vision: true/false` overrides auto-detection if needed

### Changed

#### Package Rename
- **`markit` → `markitai`**: Package renamed for clarity
- CLI command remains `markitai`

#### Python Version
- **Python 3.11+ support**: Lowered minimum Python version from 3.13 to 3.11

#### CLI Behavior
- **Single file mode**: Direct stdout output (no logging by default)
- **`--verbose`**: Show logs before output in single file mode
- Batch processing behavior unchanged

#### Code Quality
- Refactored PowerShell COM conversion scripts (~18% code reduction)
- Unified MIME type mapping across codebase
- Extracted common fixtures to `conftest.py`
- Improved error messages for network failures (SSL/connection/proxy)
- Architecture diagram updated in `docs/spec.md`

### Fixed
- URL filename cross-platform compatibility
- Cache invalidation for large documents (tail changes now detected)
- Image analysis race condition with `.llm.md` file writing

## [0.2.4] - 2026-01-21

### Changed
- Restructured `assets.json` format with flat asset array
- Extract Live display management for early log capture
- Improved MS Office detection with file path fallback

### Fixed
- Add openpyxl FileVersion compatibility patch
- Add pptx XMLSyntaxError compatibility patch
- Enhanced `check_symlink_safety` with nested symlink detection
- LLM empty response retry logic
- `normalize_frontmatter` for consistent YAML field order

## [0.2.3] - 2026-01-20

### Added

#### Persistent LLM Cache
- SQLite-based cache with LRU eviction and size limits (default 1GB)
- Dual-layer lookup: project cache + global cache
- `CacheConfig` in `MarkitaiConfig` with enabled/no_cache/max_size options
- **`--no-cache` CLI flag**: Skip reading but still write (Bun semantics)
- **`markitai cache stats [--json]`**: View cache statistics
- **`markitai cache clear [--scope]`**: Clear cache by scope

#### Vision Router Optimization
- Smart router selection: auto-detect image content in messages
- `vision_router` property filtering only `supports_vision=true` models
- Replace hardcoded "vision" model name with "default" + smart routing

#### Legacy Office Conversion
- MS Office COM batch conversion: one app launch per file type
- `check_ms_word/excel/powerpoint_available()` registry-based detection
- Pre-convert legacy files before batch processing to reduce overhead

#### Performance (Phase 3)
- **Parallel PDF processing**: Concurrent page OCR & rendering
- **Parallel image processing**: `ProcessPoolExecutor` for CPU-bound compression
- Adaptive worker count based on file size
- LRU eviction and byte-size limits for image cache
- Batch semaphore for memory pressure control

### Changed
- OCR optimization: `recognize_numpy()` and `recognize_pixmap()` for direct array processing
- Reuse already-rendered pixmap in PDF OCR (avoid re-rendering)

### Fixed
- EMF/WMF format detection and PNG conversion support
- `DATA_URI_PATTERN` regex for hyphenated MIME types (x-emf, x-wmf)
- Base64 stripping: remove hallucinated images instead of replacing
- Batch timing: record `start_at` before pre-conversion for accurate duration
- Pyright venv detection: add venvPath/venv to pyproject.toml

## [0.2.2] - 2026-01-20

### Added
- `constants.py` module to consolidate hardcoded values
- Unit tests for image and llm modules
- `convert_to_markdown.py` reference script

### Changed
- Centralized constants usage across config.py, llm.py, batch.py, image.py
- Improved LLM content restoration with garbage detection logic
- Enable parallel batch processing for image analysis
- Move state saving outside semaphore to reduce blocking

### Fixed
- Rich Panel markup parsing issue (escape file paths)

## [0.2.1] - 2026-01-20

### Added

#### LLM Usage Tracking
- Context-based usage tracking (per-file instead of global)
- `get_context_cost()` and `get_context_usage()` for per-file stats
- Thread-safe lock for concurrent access to usage dictionaries

#### Type System
- `types.py` with TypedDict definitions (ModelUsageStats, LLMUsageByModel, AssetDescription)
- `ImageAnalysis.llm_usage` for multi-model tracking (renamed from `model`)

#### Model Configuration
- `get_model_max_output_tokens()` using litellm.get_model_info()
- Auto-inject max_tokens with fallback to conservative default (8192)

#### Office Detection
- `utils/office.py` module with cross-platform detection
- `has_ms_office()`: Windows COM-based MS Office detection
- `find_libreoffice()`: PATH + common paths search with `@lru_cache`

#### Image Processing
- `strip_base64_images()` method
- `remove_nonexistent_images()` to clean LLM-hallucinated references
- Normalize whitespace for standalone image `.llm.md` output

### Changed
- File conflict rename strategy: `.2.md` → `.v2.md` for natural sort order
- Batch state: add `screenshots` field (separate from embedded images)
- Batch state: add `log_file` field for run traceability
- Store file paths as relative to input_dir in batch state

## [0.2.0] - 2026-01-19

### Added
- **Monorepo architecture** with uv workspace (`packages/markitai/`)
- **LiteLLM integration** for unified LLM provider access
- New converter modules: `pdf`, `office`, `image`, `text`, `legacy`
- Workflow system for single file processing (`workflow/single.py`)
- Markdown-based prompt management system (`prompts/*.md`)
- Unified config with JSON schema validation (`config.schema.json`)
- Security module for path validation (`security.py`)
- Comprehensive test suite with fixtures

### Changed
- CLI rewritten with Click (replaced Typer)
- Requires Python 3.13+

### Removed
- Old `src/markitai/` structure and all legacy code
- Complex pipeline/router/state machine architecture
- Individual LLM provider implementations (OpenAI, Anthropic, etc.)
- Docker and CI scripts (to be re-added later)

### Breaking Changes
- Configuration format changed (see migration guide)
- CLI command syntax updated
- Python 3.12 and below no longer supported

## [0.1.6] - 2026-01-14

### Fixed
- Model routing strategy bugs
- Documentation accuracy improvements

## [0.1.5] - 2026-01-13

### Changed
- Refactored prompt management system for better maintainability
- Simplified cleaner module logic

## [0.1.4] - 2026-01-13

### Fixed
- JSON parsing edge cases in LLM responses
- Log formatting improvements for readability

## [0.1.3] - 2026-01-12

### Added
- Test coverage improved to 81%

### Changed
- Adopted `src` layout for project structure
- Reorganized documentation to `docs/reference/`
- Added GitHub Actions CI workflow

### Fixed
- Provider-specific bugs in fallback handling

## [0.1.2] - 2026-01-12

### Added
- Resilience features for network failures (retry logic, timeout handling)
- `CLAUDE.md` and `AGENTS.md` documentation for AI assistants

### Changed
- Log optimization for cleaner, more informative output

## [0.1.1] - 2026-01-11

### Changed
- Major architecture refactoring with service layer pattern
- Enhanced LLM support with better error handling and retries

## [0.1.0] - 2026-01-10

### Added

#### Capability-Based Model Routing
- `required_capability` and `prefer_capability` parameters for LLM calls
- Text tasks prioritize text-only models for cost efficiency
- Vision tasks automatically use vision-capable models
- Backward compatible: parameters default to None (round-robin behavior)

#### Lazy Model Initialization
- Providers loaded on-demand instead of all at startup
- Significantly reduced initialization time for single-file conversions
- `warmup()` method for batch mode to validate providers upfront
- `required_capabilities` parameter in `initialize()`

#### Concurrent Fallback Mechanism
- Primary model timeout triggers parallel backup model execution
- Neither model is interrupted - first response wins
- Configurable via `llm.concurrent_fallback_timeout` (default: 180s)
- Handles Gemini 504 timeout scenarios gracefully

#### Execution Mode Support
- `--fast` flag for speed-optimized batch processing
- Fast mode: skips validation, limits fallback attempts, reduces logging
- Default mode: full validation, detailed logging, comprehensive retries
- Configurable via `execution.mode` in config file

#### Enhanced Statistics
- `BatchStats` class for comprehensive processing metrics
- Per-model tracking: calls, tokens, duration, estimated cost
- `ModelCostConfig` for optional cost estimation
- Summary format: "Complete: X success, Y failed | Total: Xs | Tokens: N"

### Changed
- CLI architecture refactored for better modularity
- Config format migrated from JSON to YAML

## [0.0.1] - 2026-01-08

### Added
- **Initial release**
- CLI commands: `convert`, `batch`, `config`, `provider`
- Multi-format support: Word (.doc, .docx), PowerPoint (.ppt, .pptx), Excel (.xls, .xlsx), PDF, HTML
- LLM enhancement: markdown formatting, frontmatter generation, image alt text
- 5 LLM providers with fallback: OpenAI, Anthropic, Gemini, Ollama, OpenRouter
- 3 PDF engines: pymupdf4llm (default), pymupdf, pdfplumber
- Image processing: extraction, compression (oxipng/mozjpeg), LLM analysis
- Batch processing with resume capability and concurrency control
- Unit and integration tests
- Docker multi-stage build
- Chinese and English documentation

[0.4.0]: https://github.com/Ynewtime/markitai/compare/v0.3.2...v0.4.0
[0.3.2]: https://github.com/Ynewtime/markitai/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/Ynewtime/markitai/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/Ynewtime/markitai/compare/v0.2.4...v0.3.0
[0.2.4]: https://github.com/Ynewtime/markitai/compare/v0.2.3...v0.2.4
[0.2.3]: https://github.com/Ynewtime/markitai/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/Ynewtime/markitai/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/Ynewtime/markitai/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/Ynewtime/markitai/compare/v0.1.6...v0.2.0
[0.1.6]: https://github.com/Ynewtime/markitai/compare/v0.1.5...v0.1.6
[0.1.5]: https://github.com/Ynewtime/markitai/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/Ynewtime/markitai/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/Ynewtime/markitai/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/Ynewtime/markitai/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/Ynewtime/markitai/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/Ynewtime/markitai/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/Ynewtime/markitai/releases/tag/v0.0.1
