# MarkIt State Machine Documentation

This document describes the state transitions and processing flow of the MarkIt document conversion system.

## Overview

MarkIt implements a multi-phase pipeline architecture for converting documents to Markdown. The pipeline is designed for parallelism, separating CPU-bound (document conversion, image processing) and I/O-bound (LLM calls) operations.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ConversionPipeline                                 │
│                                                                              │
│  ┌─────────────────┐  ┌──────────────────┐  ┌────────────────────────────┐  │
│  │ FormatRouter    │  │ ImageProcessing  │  │ LLMOrchestrator            │  │
│  │                 │  │ Service          │  │                            │  │
│  │ - Route files   │  │ - Compression    │  │ - ProviderManager          │  │
│  │ - Select        │  │ - Deduplication  │  │ - MarkdownEnhancer         │  │
│  │   converter     │  │ - Format convert │  │ - ImageAnalyzer            │  │
│  └─────────────────┘  └──────────────────┘  └────────────────────────────┘  │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         OutputManager                                │    │
│  │                                                                      │    │
│  │  - Conflict resolution (rename/overwrite/skip)                       │    │
│  │  - Write markdown + assets                                           │    │
│  │  - Generate image description .md files                              │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Single File Conversion State Machine

```
                              ┌───────────────┐
                              │    START      │
                              │ (input file)  │
                              └───────┬───────┘
                                      │
                                      ▼
                         ┌────────────────────────┐
                         │   ROUTING              │
                         │   FormatRouter.route() │
                         └────────────┬───────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    │                 │                 │
                    ▼                 ▼                 ▼
            ┌───────────┐     ┌───────────┐     ┌───────────┐
            │   PDF     │     │  Office   │     │  Other    │
            │ .pdf      │     │ .docx/ppt │     │ .md/.txt  │
            └─────┬─────┘     └─────┬─────┘     └─────┬─────┘
                  │                 │                 │
                  │                 ▼                 │
                  │    ┌────────────────────────┐     │
                  │    │   PRE-PROCESSING       │     │
                  │    │   LibreOffice → PDF    │     │
                  │    │   (uses ProfilePool)   │     │
                  │    └────────────┬───────────┘     │
                  │                 │                 │
                  └────────────────►│◄────────────────┘
                                    │
                                    ▼
                         ┌────────────────────────┐
                         │   CONVERSION           │
                         │   Primary Converter    │
                         └────────────┬───────────┘
                                      │
                         ┌────────────┴────────────┐
                         │ Success?                │
                         └────────────┬────────────┘
                              │               │
                           Yes│               │No
                              ▼               ▼
                    ┌─────────────┐  ┌─────────────────┐
                    │ Continue    │  │ FALLBACK        │
                    └──────┬──────┘  │ Try fallback    │
                           │         │ converter       │
                           │         └────────┬────────┘
                           │                  │
                           ◄──────────────────┘
                           │
                           ▼
              ┌────────────────────────────────┐
              │   IMAGE PROCESSING             │
              │   (via ImageProcessingService) │
              │                                │
              │   1. Deduplicate (MD5 hash)    │
              │   2. Format conversion         │
              │      (EMF/WMF/TIFF → PNG)      │
              │   3. Compression (Pillow/      │
              │      oxipng)                   │
              │   4. Filename standardization  │
              └────────────────┬───────────────┘
                               │
               ┌───────────────┴───────────────┐
               │ LLM Features Enabled?         │
               └───────────────┬───────────────┘
                        │              │
                     Yes│              │No
                        ▼              │
          ┌─────────────────────────┐  │
          │   LLM PROCESSING        │  │
          │   (via LLMOrchestrator) │  │
          │                         │  │
          │   ┌───────────────────┐ │  │
          │   │ Image Analysis?   │ │  │
          │   │ (vision models)   │ │  │
          │   └─────────┬─────────┘ │  │
          │             │           │  │
          │   ┌───────────────────┐ │  │
          │   │ Enhancement?      │ │  │
          │   │ (text models)     │ │  │
          │   └─────────┬─────────┘ │  │
          │             │           │  │
          └─────────────┼───────────┘  │
                        │              │
                        ◄──────────────┘
                        │
                        ▼
             ┌──────────────────────────┐
             │   OUTPUT                  │
             │   (via OutputManager)     │
             │                           │
             │   1. Conflict resolution  │
             │   2. Write .md file       │
             │   3. Write assets/        │
             │   4. Image desc .md files │
             └──────────────┬────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │    COMPLETE   │
                    │ PipelineResult│
                    └───────────────┘
```

## Batch Processing State Machine

```
                              ┌───────────────┐
                              │    START      │
                              │ (file list)   │
                              └───────┬───────┘
                                      │
                                      ▼
                         ┌────────────────────────┐
                         │   WARMUP               │
                         │   - LLM providers      │
                         │   - LibreOffice pool   │
                         └────────────┬───────────┘
                                      │
                                      ▼
              ┌───────────────────────────────────────────┐
              │            PARALLEL PROCESSING            │
              │                                           │
              │   ┌─────────────────────────────────┐    │
              │   │  Phase 1: Document Conversion   │    │
              │   │  (file semaphore controlled)    │    │
              │   │                                 │    │
              │   │  ┌───┐  ┌───┐  ┌───┐  ┌───┐    │    │
              │   │  │F1 │  │F2 │  │F3 │  │F4 │    │    │
              │   │  └─┬─┘  └─┬─┘  └─┬─┘  └─┬─┘    │    │
              │   │    │      │      │      │      │    │
              │   │    ▼      ▼      ▼      ▼      │    │
              │   │   DocumentConversionResult     │    │
              │   └─────────────────────────────────┘    │
              │                   │                      │
              │                   ▼                      │
              │   ┌─────────────────────────────────┐    │
              │   │  Phase 2: LLM Task Collection   │    │
              │   │                                 │    │
              │   │  Collect all LLM tasks from     │    │
              │   │  DocumentConversionResults      │    │
              │   └─────────────────────────────────┘    │
              │                   │                      │
              │                   ▼                      │
              │   ┌─────────────────────────────────┐    │
              │   │  Phase 3: LLM Execution         │    │
              │   │  (LLM semaphore controlled)     │    │
              │   │                                 │    │
              │   │  ┌──────┐  ┌──────┐  ┌──────┐  │    │
              │   │  │Enhnc │  │ImgAn │  │ImgAn │  │    │
              │   │  │Task1 │  │Task1 │  │Task2 │  │    │
              │   │  └──────┘  └──────┘  └──────┘  │    │
              │   └─────────────────────────────────┘    │
              │                   │                      │
              │                   ▼                      │
              │   ┌─────────────────────────────────┐    │
              │   │  Phase 4: Finalization          │    │
              │   │  (per file)                     │    │
              │   │                                 │    │
              │   │  - Apply LLM results            │    │
              │   │  - Write output files           │    │
              │   └─────────────────────────────────┘    │
              │                                           │
              └───────────────────────────────────────────┘
                                      │
                                      ▼
                              ┌───────────────┐
                              │   COMPLETE    │
                              │ BatchResult   │
                              └───────────────┘
```

## LibreOffice Profile Pool State Machine

The `LibreOfficeProfilePool` manages isolated user profiles for concurrent LibreOffice conversions:

```
                    ┌──────────────────────┐
                    │     UNINITIALIZED    │
                    │  (pool not created)  │
                    └──────────┬───────────┘
                               │
                      initialize() or
                      first acquire()
                               │
                               ▼
                    ┌──────────────────────┐
                    │      INITIALIZED     │
                    │  Queue: [P0, P1...PN]│
                    │  All profiles free   │
                    └──────────┬───────────┘
                               │
                     ┌─────────┴─────────┐
                     │                   │
            acquire()│                   │cleanup()
                     ▼                   ▼
           ┌─────────────────┐  ┌─────────────────┐
           │   ACTIVE        │  │   CLEANED UP    │
           │                 │  │   (removed)     │
           │   Some profiles │  └─────────────────┘
           │   in use        │
           └────────┬────────┘
                    │
      ┌─────────────┼─────────────┐
      │             │             │
  success       failure      reset
      │             │        threshold
      ▼             ▼             │
  release &    release &          │
  reset        increment          ▼
  failure      failure      ┌───────────┐
  count        count        │ RESET     │
      │             │       │ Profile   │
      │             │       │ cleared   │
      └─────────────┴───────┴───────────┘
                    │
                    ▼
            (return to queue)
```

### Profile Lifecycle:

```
Profile N: [AVAILABLE] → [IN_USE] → [SUCCESS/FAILURE] → [MAYBE_RESET] → [AVAILABLE]

Counters per profile:
- usage_count: increments on success, triggers reset at threshold (default: 100)
- failure_count: increments on failure, triggers reset at threshold (default: 3)
```

## LLM Provider State Machine

```
                    ┌──────────────────────┐
                    │     NOT LOADED       │
                    │  (lazy initialization)│
                    └──────────┬───────────┘
                               │
                    get_provider_manager()
                               │
                               ▼
                    ┌──────────────────────┐
                    │      LOADING         │
                    │  - Parse config      │
                    │  - Build provider    │
                    │    chain             │
                    └──────────┬───────────┘
                               │
               ┌───────────────┴───────────────┐
               │                               │
          lazy=True                       lazy=False
               │                               │
               ▼                               ▼
    ┌──────────────────────┐    ┌──────────────────────┐
    │  LAZY_INITIALIZED    │    │  EAGER_VALIDATED     │
    │                      │    │                      │
    │  Providers not yet   │    │  All providers       │
    │  validated           │    │  network-tested      │
    └──────────┬───────────┘    └──────────┬───────────┘
               │                           │
               │                           │
               └─────────────┬─────────────┘
                             │
                         complete()
                             │
                             ▼
              ┌─────────────────────────────┐
              │         ACTIVE              │
              │                             │
              │  ┌─────────────────────┐    │
              │  │ Primary Provider    │◄───┼─── First call
              │  └─────────┬───────────┘    │
              │            │                │
              │      success │ failure      │
              │            │    │           │
              │            ▼    ▼           │
              │  ┌─────────────────────┐    │
              │  │ Fallback Chain      │    │
              │  │ [P1 → P2 → P3 ...]  │    │
              │  └─────────────────────┘    │
              │                             │
              │  Concurrent Fallback Mode:  │
              │  Start backup after timeout │
              │  Use first successful       │
              └─────────────────────────────┘
```

## Image Processing State Machine

```
ExtractedImage → [DEDUP] → [CONVERT] → [COMPRESS] → [RENAME] → ProcessedImage

                    ┌──────────────────────┐
                    │   EXTRACTED IMAGES   │
                    │   [img1, img2, ...]  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   DEDUPLICATION      │
                    │   MD5 hash-based     │
                    │   Keep first only    │
                    └──────────┬───────────┘
                               │
                               ▼
              ┌────────────────────────────────┐
              │        PARALLEL PROCESSING     │
              │                                │
              │   For each unique image:       │
              │                                │
              │   ┌────────────────────────┐   │
              │   │ 1. FORMAT CONVERSION   │   │
              │   │    EMF/WMF → PNG       │   │
              │   │    TIFF → PNG/JPEG     │   │
              │   └────────────┬───────────┘   │
              │                │               │
              │                ▼               │
              │   ┌────────────────────────┐   │
              │   │ 2. COMPRESSION         │   │
              │   │    PNG: oxipng         │   │
              │   │    JPEG: quality       │   │
              │   │    Resize if > max_dim │   │
              │   └────────────┬───────────┘   │
              │                │               │
              │                ▼               │
              │   ┌────────────────────────┐   │
              │   │ 3. RENAME              │   │
              │   │    doc.pdf.001.png     │   │
              │   └────────────────────────┘   │
              │                                │
              │   Executor: Thread/Process     │
              │   (auto-selected by count)     │
              └────────────────┬───────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   FILENAME MAP       │
                    │   old → new mappings │
                    │   Update markdown    │
                    └──────────────────────┘
```

## Error Handling States

```
┌─────────────────────────────────────────────────────────────────┐
│                      ERROR RECOVERY FLOW                        │
│                                                                 │
│   ┌─────────────┐      ┌─────────────┐      ┌─────────────┐    │
│   │ CONVERSION  │      │ LLM         │      │ OUTPUT      │    │
│   │ ERROR       │      │ ERROR       │      │ ERROR       │    │
│   └──────┬──────┘      └──────┬──────┘      └──────┬──────┘    │
│          │                    │                    │            │
│          ▼                    ▼                    ▼            │
│   ┌─────────────┐      ┌─────────────┐      ┌─────────────┐    │
│   │ Try fallback│      │ Try fallback│      │ Skip/Rename │    │
│   │ converter   │      │ provider    │      │ per config  │    │
│   └──────┬──────┘      └──────┬──────┘      └──────┬──────┘    │
│          │                    │                    │            │
│          ▼                    ▼                    ▼            │
│   ┌─────────────┐      ┌─────────────┐      ┌─────────────┐    │
│   │ Success?    │      │ Success?    │      │ Success?    │    │
│   └──────┬──────┘      └──────┬──────┘      └──────┬──────┘    │
│       Y  │  N            Y   │  N            Y   │  N          │
│       │  │               │   │               │   │             │
│       ▼  ▼               ▼   ▼               ▼   ▼             │
│    [OK][FAIL]         [OK][SimpleClean]   [OK][FAIL]           │
│                                                                 │
│   Fallback Strategies:                                          │
│   - Conversion: primary → fallback → MarkItDown                │
│   - LLM: primary → [fallback chain] → SimpleMarkdownCleaner    │
│   - Output: rename (add _N suffix) / overwrite / skip          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Configuration State

```
┌─────────────────────────────────────────────────────────────────┐
│                    CONFIGURATION RESOLUTION                      │
│                                                                  │
│   Priority (highest to lowest):                                  │
│                                                                  │
│   1. CLI Arguments      --llm-provider anthropic                 │
│         │                                                        │
│         ▼                                                        │
│   ┌─────────────────────┐                                        │
│   │ LLMConfigResolver   │  Merge CLI → Config                    │
│   └─────────┬───────────┘                                        │
│             │                                                    │
│             ▼                                                    │
│   2. Environment Vars   MARKIT_LLM__PROVIDERS__0__MODEL=...     │
│         │                                                        │
│         ▼                                                        │
│   3. markit.yaml        llm: providers: [...]                   │
│         │                                                        │
│         ▼                                                        │
│   4. Defaults           from constants.py                        │
│         │                                                        │
│         ▼                                                        │
│   ┌─────────────────────┐                                        │
│   │ MarkitSettings      │  Final resolved config                 │
│   └─────────────────────┘                                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Service Dependency Graph

```
                         ┌──────────────────┐
                         │ ConversionPipeline│
                         └────────┬─────────┘
                                  │
            ┌─────────────────────┼─────────────────────┐
            │                     │                     │
            ▼                     ▼                     ▼
   ┌────────────────┐   ┌────────────────┐   ┌────────────────┐
   │ImageProcessing │   │ LLMOrchestrator│   │ OutputManager  │
   │    Service     │   │                │   │                │
   └───────┬────────┘   └───────┬────────┘   └───────┬────────┘
           │                    │                    │
           ▼                    ▼                    │
   ┌────────────────┐   ┌────────────────┐           │
   │ImageCompressor │   │ProviderManager │           │
   │                │   │                │           │
   └────────────────┘   └───────┬────────┘           │
                                │                    │
                    ┌───────────┼───────────┐        │
                    │           │           │        │
                    ▼           ▼           ▼        │
           ┌────────────┐ ┌────────────┐ ┌─────────┐ │
           │ Markdown   │ │  Image     │ │Provider │ │
           │ Enhancer   │ │ Analyzer   │ │ Chain   │ │
           └────────────┘ └────────────┘ └─────────┘ │
                                                     │
                                                     ▼
                                             ┌──────────────┐
                                             │   Frontmatter│
                                             │   Generator  │
                                             └──────────────┘
```

## Key Constants and Thresholds

| Constant | Default | Description |
|----------|---------|-------------|
| `LIBREOFFICE_POOL_SIZE` | 8 | Concurrent LibreOffice instances |
| `PROCESS_POOL_THRESHOLD` | 5 | Min images to use process pool |
| `RESET_AFTER_FAILURES` | 3 | Profile reset after N failures |
| `RESET_AFTER_USES` | 100 | Profile reset after N uses |
| `CHUNK_SIZE` | 32000 | LLM enhancement chunk size (tokens) |
| `CHUNK_OVERLAP` | 500 | Token overlap between chunks |
| `MAX_IMAGE_DIMENSION` | 2048 | Max image width/height |
| `CONCURRENT_FALLBACK_TIMEOUT` | 60s | Fallback trigger timeout |
| `MAX_REQUEST_TIMEOUT` | 300s | Absolute request timeout |
| `FILE_WORKERS` | 4 | Concurrent file conversions |
| `IMAGE_WORKERS` | 8 | Concurrent image processing |
| `LLM_WORKERS` | 5 | Concurrent LLM requests |
