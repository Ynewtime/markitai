# Markit 性能优化分析报告 (Phase 3)

基于 `logs\markit_20260120_175255_943396.log` 日志文件及核心代码库的深度分析，本文档详细阐述了当前系统的性能瓶颈，并提出了针对性的优化建议。

## 1. 性能现状摘要

通过分析日志，我们观察到以下关键性能数据：

| 任务阶段 | 耗时范围 | 典型案例 | 瓶颈等级 |
| :--- | :--- | :--- | :--- |
| **Office 转换 (PPTX/DOC)** | **6s - 14s** | `Free_Test_Data_500KB_PPTX.pptx` (6.3s), `file_example_PPT_250kB.ppt` (14.15s) | 🔴 Critical (IO/CPU) |
| **Vision 增强 (LLM)** | **25s - 44s** | PDF 增强 5 页耗时 44s ($0.05), PPTX 增强 8 页耗时 25s ($0.01) | 🔴 Critical (Latency) |
| **标准文本清洗 (LLM)** | **22s - 38s** | 100KB Excel/Word 文档清洗耗时 30s+ | 🟡 High |
| **图片分析 (Embedded)** | **4s - 22s** | 单图 4s, 2图 22s | 🟡 High |
| **总处理时长** | **40s - 70s** | 单个中等大小文档 (500KB) | 🔴 Critical |

## 2. 核心瓶颈分析

### A. 视觉任务 (Vision) 无持久化缓存 (High Impact)
虽然 `llm.py` 中存在 `ContentCache`，但分析证实这是一个纯内存 (`OrderedDict`) 缓存。
*   **现象**: 每次运行 CLI 命令，进程重启，内存缓存清空。
*   **后果**: 对于 Vision 增强这种昂贵（~$0.05/次）且缓慢（40s+）的操作，每次重跑都要重新调用 API。
*   **影响**: 极大地阻碍了调试和增量处理（Resume 功能仅基于文件存在与否，无法做到 Asset 级别的增量）。

### B. 串行阻塞的流水线 (Pipeline Blocking)
当前 `process_file` 内部逻辑是严格串行的：
```mermaid
graph LR
    A[Converter (6-14s)] --> B[Extract Images] --> C[Vision Enhance (25-44s)] --> D[Embedded Img Analysis (10-20s)]
```
*   **问题**: 
    1. **转换阻塞**: Office 转换（依赖外部 COM/LibreOffice）必须完全完成后，LLM 才能开始。
    2. **LLM 串行**: Vision Enhancement（修复排版）完成后，才开始分析文档内的嵌入图片。两者其实互不依赖（除非 Vision 结果严重改变了图片上下文，但通常图片本身内容分析是独立的）。
    3. **资源争抢**: 多个文件并发时，长尾的 Vision 任务会长时间占用 `LLMRuntime` 的 Semaphore，导致短任务（如 Frontmatter 生成）被阻塞。

### C. Office 转换开销 (External Dependency)
PPT/PPTX 转换极慢（6-14s）。
*   **原因**: 依赖本地安装的 Microsoft Office (COM) 或 LibreOffice 启动开销。
*   **现状**: 虽然使用了 `ThreadPoolExecutor` 将其移出主线程，但在处理大量 Office 文档时，这仍然是显著的初始延迟。
*   **Windows 平台缺陷**: 当前代码即使在 Windows 环境下，对于 `.doc` 和 `.ppt` 旧格式转换也强制使用 LibreOffice CLI (`soffice.exe`)，未利用本地安装的 MS Office COM 接口。LibreOffice 的启动和转换通常比原生 COM 调用更慢且兼容性稍差。

### D. 路由策略与模型选择 (Configuration)
日志显示 `qwen/qwen3-coder-30b-a3b-instruct` 被用于通用文档清洗 (`document_process`)。
*   **数据**: 处理 100KB Excel 耗时 31s (Line 106)。
*   **问题**: 虽然 Qwen 是优秀的 Code 模型，但用于纯文本/Markdown 清洗可能并非最优（速度 vs 质量）。其响应延迟显著高于 `gemini-flash` 等模型。
*   **Vision**: Vision 任务主要路由到 `gemini-2.5-*`，表现尚可（~3s/页），但 PDF 处理每页高达 8s，可能受限于 Context Window 或网络抖动。

## 3. 优化方案 (Roadmap)

### 第一阶段：持久化缓存 (Quick Win)
**目标**: 消除重复的昂贵计算，实现秒级重跑。

1.  **引入 SQLite 缓存**:
    *   替换 `ContentCache` 的内存实现，使用 `sqlite3`。
    *   Key: `hash(prompt + image_bytes)`。
    *   Value: `json(response)`。
    *   Scope: `analyze_image`, `enhance_document_with_vision`, `clean_markdown`.
2.  **缓存层级**:
    *   实现 Global Cache (`~/.markit/cache.db`)，跨项目共享通用图片分析结果（如相同的 Logo/Icon）。

### 第二阶段：Windows 平台 COM 优化 (Platform Specific)
**目标**: 在 Windows 环境下利用原生 Office 提升转换速度和兼容性。

1.  **COM 转换实现**:
    *   参考 `convert_to_markdown.py`，在 `LegacyOfficeConverter` 中增加 COM 调用逻辑。
    *   使用 PowerShell 脚本封装 COM 操作（`Word.Application` 和 `PowerPoint.Application`），通过 `subprocess` 调用，实现 `.doc -> .docx` 和 `.ppt -> .pptx` 的无缝转换。
2.  **自动回退机制**:
    *   优先级：`MS Office COM (Windows)` > `LibreOffice CLI` > `Fail`。
    *   即使在 Windows 上，如果 COM 调用失败（如许可证弹窗、未激活），自动回退到 LibreOffice。

### 第三阶段：流水线并行化 (Throughput)
**目标**: 提升吞吐量，减少 CPU/IO 等待。

1.  **解耦转换与分析**:
    *   将 `process_file` 拆分为 `convert_stage` 和 `analyze_stage`。
    *   `convert_stage` 产生的中间产物（图片/Markdown）存入磁盘。
    *   `analyze_stage` 独立扫描中间产物进行 LLM 处理。
2.  **任务内并行**:
    *   在 `process_file` 内部，使用 `asyncio.gather` 并行执行 "文档清洗" 和 "嵌入图片分析"。
    *   注：需要处理合并冲突（分别修改 Markdown 的不同部分）。

### 第三阶段：精细化模型路由 (Cost/Latency)
**目标**: 在质量和速度之间取得平衡。

1.  **模型分层 (Tiering)**:
    *   `tier:fast`: Gemini Flash, Haiku (用于简单 OCR 修正, Alt Text)。
    *   `tier:smart`: GPT-4o, Sonnet, Qwen-Max (用于复杂排版修复, 代码提取)。
2.  **动态降级**:
    *   如果 `smart` 模型超时，自动降级到 `fast` 模型（牺牲质量换取完成）。
    *   当前代码已有 Fallback，但主要是为了错误处理，而非性能优化。

### 第五阶段：大文档分块策略 (Scalability)
**目标**: 防止 Context Window 溢出，提升 TTFT。

1.  **智能分块**:
    *   对于 `clean_markdown`，实现基于 Header/Paragraph 的流式分块处理。
    *   避免一次性将 100页文档塞入 Context。
2.  **增量 Frontmatter**:
    *   仅使用文档前 10k token 生成 Frontmatter，而非全量。

## 4. 立即行动建议 (Next Steps)

1.  **修改 `llm.py`**: 实现基于 SQLite 的持久化缓存装饰器。
2.  **实现 COM 转换**: 在 `legacy.py` 中移植 `convert_to_markdown.py` 的 PowerShell COM 调用代码。
3.  **调整 `markit.json`**: 优化模型权重，确保 `default` 组优先使用高吞吐模型（如 `gemini-flash` 或 `deepseek-v3`），将 `qwen-coder` 留给特定代码任务或作为 Backup。
4.  **代码优化**: 在 `process_file` 中尝试并行化 Vision Enhance 和 Embedded Image Analysis。
