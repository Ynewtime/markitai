# Proposal

Refactor MarkIt Pipeline for Maximum Parallelism

Project: markit - Intelligent document to Markdown conversion tool with LLM enhancement

Goal: Refactor the conversion pipeline to achieve maximum parallelism and minimize total processing time for batch conversions.

Current Architecture Issues:
1. File semaphore blocks entire processing (markit/utils/concurrency.py:208):
   - When map_file_tasks() processes files, the semaphore is held for the entire duration of pipeline.convert_file_async()
   - A slow file (e.g., large document with many LLM calls) blocks one file slot for minutes
   - Other files cannot start even if their document conversion is fast
2. LLM semaphore is defined but never used (markit/utils/concurrency.py:68-72):
   - ConcurrencyManager has llm_workers semaphore
   - But run_llm_task() is never called in actual LLM paths
   - Image analysis and markdown enhancement have no rate limiting
3. Sequential processing within files (markit/core/pipeline.py:250-310):
   - Within a single file: convert → process images → enhance markdown → write output
   - Each stage blocks before the next can start
   - LLM enhancement is particularly slow and blocks file completion
4. Chunked enhancement is partially parallel (markit/llm/enhancer.py:114):
   - Chunks within a document ARE processed in parallel via asyncio.gather()
   - But summary generation (line 122) is sequential after all chunks
   - No concurrency control for the number of simultaneous LLM calls
   
Proposed Architecture:
┌─────────────────────────────────────────────────────────────────────┐
│                        BATCH PROCESSOR                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Phase 1: Document Conversion (file_semaphore)                      │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐                               │
│  │File 1│ │File 2│ │File 3│ │File 4│  → Parallel (8 workers)       │
│  └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘                               │
│     │        │        │        │                                     │
│     ▼        ▼        ▼        ▼                                     │
│  ┌──────────────────────────────────┐                               │
│  │     Conversion Results Queue      │                               │
│  └──────────────────────────────────┘                               │
│                     │                                                │
│  Phase 2: LLM Processing (llm_semaphore)                            │
│  ┌──────────────────────────────────┐                               │
│  │  Global LLM Task Queue           │                               │
│  │  - Image analysis tasks          │                               │
│  │  - Markdown enhancement tasks    │                               │
│  │  - Summary generation tasks      │                               │
│  └──────────────────────────────────┘                               │
│     │        │        │        │                                     │
│     ▼        ▼        ▼        ▼                                     │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐                               │
│  │LLM 1 │ │LLM 2 │ │LLM 3 │ │...   │  → Parallel (10 workers)      │
│  └──────┘ └──────┘ └──────┘ └──────┘                               │
│                                                                      │
│  Phase 3: Output Writing (io_semaphore)                             │
│  - Write markdown files                                              │
│  - Write images                                                      │
│  - Write image descriptions                                          │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘

Specific Refactoring Tasks:
1. Decouple file processing stages (markit/core/pipeline.py):
      # Instead of:
   async def _convert_file_async(self, input_file, output_dir):
       result = await self._convert_with_fallback(...)  # Document conversion
       result = await self._process_images(result)       # Image processing + LLM
       markdown = await self._enhance_markdown(...)      # LLM enhancement
       await self._write_output(...)                     # Write output
   
   # Refactor to:
   async def _convert_file_async(self, input_file, output_dir):
       # Phase 1: Quick document conversion (release file semaphore early)
       conversion_result = await self._convert_document(input_file)
       
       # Phase 2: Queue LLM tasks (don't await, return futures)
       llm_tasks = self._create_llm_tasks(conversion_result)
       
       # Phase 3: Return intermediate result, let batch processor handle LLM
       return IntermediateResult(conversion_result, llm_tasks)
   
2. Create global LLM task queue (markit/llm/queue.py - new file):
      class LLMTaskQueue:
       """Global queue for all LLM tasks across files."""
       
       def __init__(self, max_concurrent: int = 10):
           self.semaphore = asyncio.Semaphore(max_concurrent)
           self.tasks: list[asyncio.Task] = []
       
       async def submit(self, coro: Coroutine) -> asyncio.Task:
           """Submit an LLM task with rate limiting."""
           async def rate_limited():
               async with self.semaphore:
                   return await coro
           
           task = asyncio.create_task(rate_limited())
           self.tasks.append(task)
           return task
       
       async def wait_all(self) -> list[Any]:
           """Wait for all submitted tasks to complete."""
           return await asyncio.gather(*self.tasks, return_exceptions=True)
   
3. Modify batch processor (markit/cli/commands/batch.py):
      async def _execute_batch(...):
       llm_queue = LLMTaskQueue(max_concurrent=llm_concurrency)
       
       # Phase 1: Start all document conversions
       conversion_tasks = []
       for file in files:
           task = asyncio.create_task(
               pipeline.convert_document_only(file, output_dir)
           )
           conversion_tasks.append(task)
       
       # Phase 2: As conversions complete, queue LLM tasks
       for task in asyncio.as_completed(conversion_tasks):
           result = await task
           for image in result.images:
               await llm_queue.submit(analyzer.analyze(image))
           if llm_enabled:
               await llm_queue.submit(enhancer.enhance(result.markdown))
       
       # Phase 3: Wait for all LLM tasks and write outputs
       llm_results = await llm_queue.wait_all()
       # ... write outputs
   
4. Add LLM response caching (markit/llm/cache.py - new file):
      class LLMCache:
       """Cache LLM responses based on content hash."""
       
       def __init__(self, cache_dir: Path):
           self.cache_dir = cache_dir
       
       def get_key(self, content: bytes | str) -> str:
           """Generate cache key from content."""
           if isinstance(content, str):
               content = content.encode()
           return hashlib.sha256(content).hexdigest()[:16]
       
       async def get(self, key: str) -> str | None:
           """Get cached response."""
           cache_file = self.cache_dir / f"{key}.json"
           if cache_file.exists():
               return json.loads(cache_file.read_text())
           return None
       
       async def set(self, key: str, response: str) -> None:
           """Cache response."""
           cache_file = self.cache_dir / f"{key}.json"
           cache_file.write_text(json.dumps(response))
   
5. Key files to modify:
   - markit/core/pipeline.py: Split _convert_file_async() into phases
   - markit/cli/commands/batch.py: Implement phased batch processing
   - markit/llm/manager.py: Add rate limiting via semaphore
   - markit/llm/enhancer.py: Support async task submission
   - markit/image/analyzer.py: Support async task submission
   - New: markit/llm/queue.py: Global LLM task queue
   - New: markit/llm/cache.py: Response caching
   
Expected Performance Improvement:
- Current: ~80 seconds for 6 files
- Target: ~30-40 seconds for 6 files
- Key insight: LLM calls from different files should run in parallel, not wait for each file to complete

Testing:
- Run markit batch input/ -o output/ --llm --analyze-image-with-md --verbose
- Monitor logs/markit.log for timing gaps
- Success criteria: No single LLM call should block other files' processing

Constraints:
- Maintain backward compatibility with existing CLI options
- Keep unit tests passing
- Don't break single-file markit convert command


# Review by Gemini 3 Pro

结合项目源码分析，docs/refactor_pipeline.md 提出的重构方案是非常合理且必要的。

## 1. 现状问题确认

通过查阅源码，我确认了文档中提到的几个核心问题确实存在：
*   File Semaphore 阻塞问题 (markit/utils/concurrency.py):
    *   map_file_tasks 使用 file_semaphore 包裹了整个任务函数。
    *   在 markit/cli/commands/batch.py 中，传递给 batch_processor 的任务是 pipeline.convert_file_async。
    *   markit/core/pipeline.py 的 _convert_file_async 是顺序执行的：转换文档 -> 处理图片(含LLM分析) -> 增强Markdown(含LLM) -> 写入输出。
    *   结论： 这意味着一个文件的 LLM I/O 等待时间（通常很长）会一直占用一个“文件处理名额”，导致后续文件无法开始 CPU 密集的本地转换工作。
*   LLM Semaphore 未生效 (markit/utils/concurrency.py, markit/llm/manager.py):
    *   ConcurrencyManager 虽然定义了 _llm_semaphore，但只在 run_llm_task 中使用。
    *   markit/image/analyzer.py 和 markit/llm/enhancer.py 直接调用 ProviderManager。
    *   ProviderManager 内部只是简单地遍历 provider 进行调用，没有任何并发控制或信号量机制。
    *   结论： 目前的架构确实没有对 LLM 请求进行全局限流，容易触发 API Rate Limit，或者在单文件内部并行（如 asyncio.gather 处理 chunks）时瞬间发起过多请求。
    
## 2. 方案可行性分析

文档提出的 "三阶段流水线" (Phased Pipeline) 方案非常切中痛点：
1.  阶段解耦 (Decoupling):
    *   将 CPU 密集的 文档转换 (Phase 1) 与 I/O 密集的 LLM 处理 (Phase 2) 分离。
    *   一旦 Phase 1 完成，立即释放 file_semaphore，让下一个文件进入转换阶段。这最大化了 CPU 利用率。
2.  全局任务队列 (Global Task Queue):
    *   使用全局队列接管所有文件的 LLM 任务（图片分析、文本增强）。
    *   配合独立的 llm_semaphore，可以在全局范围内精确控制并发请求数，避免 API 超限，同时保证只要有配额，LLM 任务就能持续运行，不会被“慢文件”阻塞。
3.  中间结果传递 (Intermediate Result):
    *   通过返回 IntermediateResult (包含转换结果和待执行的 LLM future)，让 BatchProcessor 能够调度后续任务。这是标准的异步生产-消费模式，非常适合 Python asyncio。
    
## 3. 补充建议与注意事项

虽然方案整体很棒，但在实施时建议考虑以下几点以增强健壮性：
*   保持单文件模式的兼容性:
    *   提案建议修改 _convert_file_async 返回 IntermediateResult。这会破坏 markit/cli/commands/convert.py 中单文件 convert 命令的逻辑，因为它期望直接得到最终结果。
    *   建议： 将新逻辑实现为 convert_file_phased。原有的 _convert_file_async 保留作为 wrapper，在内部调用 phased 版本并 await 所有 LLM 任务完成，最后返回标准的 PipelineResult。这样单文件模式无需大改。
*   内存控制 (Backpressure):
    *   如果文档转换（Phase 1）极快（例如每秒处理10个），而 LLM 处理（Phase 2）很慢（每秒1个），内存中会堆积大量待处理的图片数据和文本。
    *   虽然对于普通批次（几十个文件）这不是问题，但如果是大批量任务，建议给 LLM 队列设置一个最大长度（或限制 Phase 1 领先 Phase 2 的文件数量），防止内存爆涨。
*   错误处理:
    *   在全局队列模式下，某个文件的 LLM 任务失败需要正确地反馈回该文件的最终状态。BatchProcessor 需要能够将全局队列中的异常关联回具体的文件任务，以便生成正确的最终报告。
    *   

总结： 该方案逻辑清晰，直击当前并发模型的痛点，建议采纳并按计划实施。


# Review by Claude Opus 4.5

深度评审: docs/refactor_pipeline.md 重构方案
1. 现状问题确认 ✅
通过分析源码，我确认文档中提到的问题确实存在且分析准确：
1.1 File Semaphore 阻塞问题 (✅ 确认)
位置: markit/utils/concurrency.py:207-210
async def process_item(item: T) -> TaskResult[T]:
    async with semaphore:  # <-- 信号量在整个任务期间持有
        try:
            result = await func(item)  # <-- func = pipeline.convert_file_async (包含所有LLM调用)
影响: 在 batch.py:503 调用 concurrency.map_file_tasks(files, process_file) 时，process_file 内部调用 pipeline.convert_file_async。该方法顺序执行：
1. 文档转换 (快速, CPU密集)
2. 图片处理含LLM分析 (慢, I/O密集) - pipeline.py:277
3. Markdown增强LLM (慢, I/O密集) - pipeline.py:283
4. 写输出 (快, I/O密集)
结论: 一个有10张图片的文件可能持有 file_semaphore 长达30-60秒，期间其他文件无法开始转换。
1.2 LLM Semaphore 未生效 (✅ 确认)
位置:
- concurrency.py:68-72: _llm_semaphore 已定义
- concurrency.py:104-117: run_llm_task() 方法存在
但从未被调用:
- enhancer.py:147: 直接调用 provider_manager.complete_with_fallback()，无信号量
- analyzer.py:126: 直接调用 provider_manager.analyze_image_with_fallback()，无信号量
- analyzer.py:221: batch_analyze() 使用 asyncio.gather() 无限并发
风险: 
- 如果一个文档有20张图片，会同时发起20个LLM请求
- 如果8个文件同时处理，每个有10张图片，理论上可能同时发起80个LLM请求
- 容易触发API Rate Limit
1.3 Chunk并行但无全局控制 (✅ 确认)
位置: enhancer.py:114
enhanced_chunks = await asyncio.gather(*[self._process_chunk(chunk) for chunk in chunks])
文档内部的chunks是并行的，但：
1. 没有全局并发控制
2. summary 生成是顺序的 (enhancer.py:122)
---
2. 方案评估
2.1 三阶段流水线架构 (✅ 优秀设计)
Phase 1: Document Conversion (快速释放 file_semaphore)
    ↓
Phase 2: LLM Processing (全局队列 + llm_semaphore)
    ↓  
Phase 3: Output Writing (io_semaphore)
优点:
- CPU密集任务与I/O密集任务解耦
- LLM任务跨文件共享配额，最大化吞吐量
- 符合生产者-消费者模式
2.2 IntermediateResult 设计 (⚠️ 需调整)
文档建议:
async def _convert_file_async(self, input_file, output_dir):
    conversion_result = await self._convert_document(input_file)
    llm_tasks = self._create_llm_tasks(conversion_result)
    return IntermediateResult(conversion_result, llm_tasks)
问题: 这会破坏现有的单文件API:
- pipeline.convert_file() (sync) 在 pipeline.py:214 调用 _convert_file_async()
- convert.py 命令直接依赖 PipelineResult 返回类型
建议: 采用 Gemini 3 Pro 的建议，创建新方法而非修改现有方法:
# 新增分阶段方法
async def convert_document_only(self, input_file, output_dir) -> DocumentConversionResult:
    """Phase 1: Only document conversion, no LLM processing."""
    
async def create_llm_tasks(self, doc_result) -> list[Coroutine]:
    """Phase 2: Create LLM task coroutines without executing."""
# 保留现有方法作为wrapper
async def _convert_file_async(self, input_file, output_dir) -> PipelineResult:
    """原有方法，内部调用分阶段方法，保持向后兼容。"""
2.3 LLMTaskQueue 设计 (✅ 可行，需完善)
文档方案:
class LLMTaskQueue:
    def __init__(self, max_concurrent: int = 10):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.tasks: list[asyncio.Task] = []
改进建议:
1. 任务关联追踪: 需要将LLM任务结果关联回源文件
@dataclass
class LLMTask:
    source_file: Path
    task_type: Literal["image_analysis", "markdown_enhancement", "summary"]
    coro: Coroutine
    result: Any = None
2. 优先级支持: 可考虑优先处理已转换完成的文件的LLM任务
from asyncio import PriorityQueue
class LLMTaskQueue:
    def __init__(self, max_concurrent: int = 10):
        self.queue = PriorityQueue()
        self.semaphore = asyncio.Semaphore(max_concurrent)
3. Backpressure机制: 防止转换过快导致内存堆积
def __init__(self, max_concurrent: int = 10, max_pending: int = 100):
    self.semaphore = asyncio.Semaphore(max_concurrent)
    self.pending_limit = asyncio.Semaphore(max_pending)  # 限制待处理任务数
2.4 LLMCache 设计 (⚠️ 建议简化)
文档方案使用 SHA256 hash 作为缓存key。
问题:
1. 图片分析缓存意义有限 - 同一批次处理不太可能遇到完全相同的图片
2. Markdown增强缓存更复杂 - chunk边界可能不同
建议: 
- 首版不实现缓存，专注并发优化
- 如需缓存，考虑文件级别而非内容级别:
cache_key = f"{file_path.stem}_{file_stat.st_mtime}_{file_stat.st_size}"
---
3. 实施风险分析
| 风险 | 严重程度 | 缓解措施 |
|------|----------|----------|
| 破坏单文件命令 | 高 | 保留 _convert_file_async 原有签名，新增分阶段方法 |
| 内存堆积 | 中 | 添加 backpressure 机制，限制待处理任务数 |
| 错误追踪困难 | 中 | LLMTask 结构包含 source_file 字段 |
| 现有测试失败 | 低 | 保持公共API不变，新增API单独测试 |
| LLM异步上下文 | 中 | 确保 ProviderManager 在同一事件循环内初始化 |
---
4. 建议的实施步骤
Phase 1: 基础设施 (低风险)
1. ✅ 创建 markit/llm/queue.py - LLMTaskQueue
2. ✅ 修改 ConcurrencyManager - 暴露 get_llm_semaphore() 方法供外部使用
Phase 2: Pipeline分阶段 (中风险)
3. 在 pipeline.py 新增:
   - convert_document_only() - 只做文档转换
   - create_image_analysis_tasks() - 返回图片分析协程列表
   - create_enhancement_task() - 返回增强协程
4. 保留 _convert_file_async() 作为兼容层
Phase 3: Batch重构 (高风险)
5. 重写 _execute_batch():
   - Phase 1: 并行文档转换，快速释放 file_semaphore
   - Phase 2: 所有LLM任务进入全局队列
   - Phase 3: 等待所有任务完成，写入输出
Phase 4: 测试与验证
6. 添加集成测试验证并发行为
7. 性能基准测试对比
---
5. 预期性能提升
当前瓶颈分析 (6文件，每文件平均3张图片):
- 文档转换: ~2s/file × 6 = 12s (但受 file_semaphore 限制)
- 图片分析: ~3s/image × 18 = 54s (顺序执行)
- Markdown增强: ~5s/file × 6 = 30s (顺序执行)
- 总计: ~80-90s
重构后预期:
- 文档转换: ~12s (并行8个文件)
- 图片分析: 18 images / 10 workers × 3s = ~6s
- Markdown增强: 6 files / 10 workers × 5s = ~3s
- 预期总计: ~25-35s
预期提升: 2.5x - 3x
---
6. 结论
方案评分: 8.5/10
优点:
- 问题分析准确，直击痛点
- 三阶段设计架构清晰
- 保留了向后兼容性考虑
需完善:
- IntermediateResult 需要更详细的数据结构设计
- LLMTaskQueue 需要任务追踪和错误关联机制
- 建议首版移除 LLMCache，降低复杂度
建议: 采纳此方案，按上述分阶段实施。