# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.4] - 2026-01-13

### Fixed

- **JSON Extraction**: Implemented bracket-counting algorithm (`_extract_first_json_object`) in ImageAnalyzer to precisely extract JSON objects, fixing intermittent JSON parsing failures
- **Concurrent Fallback**: Fixed `convert` command not reading `concurrent_fallback_enabled` config, enabling concurrent fallback mechanism properly
- **Timeout Configuration**: Unified timeout settings by removing `DEFAULT_LLM_TIMEOUT` (120s), now using `DEFAULT_MAX_REQUEST_TIMEOUT` (300s) for httpx timeout, fixing concurrent fallback triggering issues

### Changed

- **Log Formatting Improvements**
  - Simplified HTTP request/response logs with `method=<method> url=<url>` format
  - Added file context injection via `set_request_context(file_path=...)`
  - Removed redundant logs (e.g., `Enhancing Markdown`)
  - Filtered out `request_id: None` logs via `RequestIdNoneFilter`
  - Fixed provider field in LLM logs to use context-injected credential_id instead of hardcoded provider name
  - Simplified concurrent fallback logs by removing redundant provider/model fields

## [0.1.3] - 2026-01-12

### Added

- **Comprehensive Test Suite** (46% → 81% coverage, 1363 tests)
  - LLM providers: openai 94%, anthropic 90%, gemini 97%, ollama 98%, openrouter 100%, enhancer 90%
  - CLI commands: config 95%, convert 85%+, batch 75%+, provider 70%+, model 70%+
  - Converters: pdfplumber 95%, pandoc 95%, pymupdf 99%, office 63%
  - Markdown: formatter 97%, chunker 94%, frontmatter 100%
  - Utils/Image: fs 93%, stats 97%, concurrency 99%, analyzer 94%, extractor 96%

- **LiteLLM Integration Analysis** (`docs/reference/litellm_analysis.md`, `docs/020-SPEC.md`)
  - Compatibility analysis with existing provider architecture
  - Migration path evaluation and risk assessment

### Fixed

- **ProviderManager Race Condition**: Added credential-level locking (`_credential_init_locks`) to prevent duplicate validation when multiple models share the same credential
- **Provider Logging**: Fixed hardcoded provider name in LLM log output, now uses `self.name` for correct inheritance (OpenRouterProvider shows `provider=openrouter`)

### Changed

- Migrated e2e tests to unit tests (aimd_limiter, bounded_queue, dead_letter_queue, chaos_provider)
- Reorganized test structure with dedicated directories for cli/commands, converters, image, llm, markdown, utils

## [0.1.2] - 2026-01-12

### Added

- **Resilience Testing Framework**
  - Test fixture generator (`tests/fixtures/heavy_load/generate_dataset.py`) supporting 1k/10k/nested presets
  - Chaos Mock Provider (`markit/llm/chaos.py`) for simulating delays, rate limits, failures, and timeouts
  - Interrupt recovery verification (`tests/integration/test_resilience.py`) with SIGINT + state.json + --resume

- **AIMD Adaptive Concurrency Control**
  - Core implementation in `markit/utils/adaptive_limiter.py` (additive increase / multiplicative decrease / cooldown)
  - Queue integration in `markit/llm/queue.py` (enable via `use_adaptive=True`)

- **Priority Backpressure Queue**
  - Backpressure mechanism: `markit/utils/flow_control.py` → BoundedQueue
  - Queue integration in `markit/llm/queue.py` with `max_pending` limit

- **Dead Letter Queue (DLQ)**
  - Generic DLQ implementation: `markit/utils/flow_control.py` → DeadLetterQueue
  - State integration in `markit/core/state.py` with `failure_count` and `permanent_failure` fields

- **Observability Enhancements**
  - Dry-run token/cost estimation in `markit/cli/commands/batch.py`
  - Three-scenario estimates: conversion only / LLM enhancement / full analysis with images

### Fixed

- **AIMD Deadlock Bug**: Fixed semaphore replacement causing deadlock under high concurrency
  - Increase concurrency: Call `release()` on existing semaphore to add slots
  - Decrease concurrency: Lazy shrinking via `_pending_reductions` counter

### Changed

- Optimized log field ordering with `sort_keys=False` in ConsoleRenderer
- Enhanced HTTP request logging context with provider/model injection
- Improved conversion plan, converter, and enhancement logging with file context

## [0.1.1] - 2026-01-11

### Added

- **Service Layer Architecture**
  - `ImageProcessingService` (`markit/services/image_processor.py`): Image format conversion, compression (oxipng/Pillow), deduplication
  - `LLMOrchestrator` (`markit/services/llm_orchestrator.py`): Centralized LLM operations with capability-based routing
  - `OutputManager` (`markit/services/output_manager.py`): File writing, conflict resolution, image description markdown generation

- **LibreOffice Profile Pool** (`markit/converters/libreoffice_pool.py`)
  - Isolated profile directories for parallel .doc/.ppt/.xls conversion
  - Borrow/return mechanism to avoid lock file conflicts

- **Process Pool for Images**
  - CPU-intensive image compression now uses ProcessPoolExecutor to bypass Python GIL

- **Enhanced Prompts**
  - Chinese language output support in enhancement and summary prompts
  - Improved image analysis prompts with better classification and OCR handling

### Changed

- Refactored `ConversionPipeline` to use new service layer
- Moved hardcoded default models to `markit/config/constants.py`

## [0.1.0] - 2026-01-10

### Added

- **Capability-Based Model Routing**
  - Models declare capabilities (`["text"]` or `["text", "vision"]`)
  - Text tasks route to cheaper text models; vision tasks only route to vision-capable models
  - `complete_with_fallback` supports `required_capability` and `prefer_capability` parameters

- **Lazy Model Initialization**
  - On-demand provider validation reduces startup time
  - Batch mode pre-analyzes task types for targeted initialization

- **Model Cost Configuration**
  - Optional cost tracking per model (`input_per_1m`, `output_per_1m`)
  - Cost estimation in logs and statistics

- **Concurrent Fallback**
  - Primary model timeout triggers concurrent backup model execution
  - First response wins; other tasks cancelled
  - Configurable via `concurrent_fallback_timeout`

- **Fast Mode** (`--fast`)
  - Skip validation, minimal retries, error-only logging
  - Single fallback attempt for maximum speed

- **Enhanced Logging Statistics**
  - Per-model usage tracking
  - Token consumption and cost estimates
  - Timing breakdown (LLM / Convert / Init)

### Changed

- Renamed `markit config show` to `markit config list`
- Renamed `markit config validate` to `markit config test`
- Renamed `markit provider models` to `markit provider fetch`
- Renamed `markit provider select` to `markit model add`
- Reordered config subcommands: init/test/list/locations

## [0.0.1] - 2026-01-07

### Added

- Initial release of MarkIt
- **Multi-format Support**: Word (.docx/.doc), PowerPoint (.pptx/.ppt), Excel (.xlsx/.xls), PDF, HTML, Images
- **LLM Enhancement**: Clean headers/footers, fix headings, add frontmatter, generate summaries
- **Image Processing**: Auto compression, format conversion, deduplication
- **Batch Processing**: Recursive directory conversion with resume capability
- **Multi-Provider LLM**: OpenAI, Anthropic, Google Gemini, Ollama, OpenRouter
- **CLI Commands**: convert, batch, config, provider, model
- **Configuration**: YAML-based config with credential/model separation

[0.1.4]: https://github.com/Ynewtime/markit/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/Ynewtime/markit/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/Ynewtime/markit/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/Ynewtime/markit/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/Ynewtime/markit/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/Ynewtime/markit/releases/tag/v0.0.1
