# Markit 性能优化分析报告 v2

> 分析日期: 2026-01-20
> 版本: markit v0.2.0
> 重点: LLM 调用优化

## 1. 执行摘要

当前 markit 在批量处理文档时，LLM 调用是主要性能瓶颈。本报告深入分析了 LLM 调用流程，识别了关键优化点，并提供了详细的实施方案。

### 关键发现

| 优化项 | 当前耗时 | 优化后 | 收益 | 优先级 |
|--------|---------|--------|------|--------|
| 批量图片分析串行 | ~20s/10张 | ~2s/10张 | **10x** | HIGH |
| 多批文档串行处理 | ~12s/50页 | ~10s/50页 | **20%** | HIGH |
| Fallback 链串行重试 | ~5s/失败 | ~2s/失败 | **60%** | MEDIUM |
| I/O 阻塞事件循环 | +10-15% | 0% | **10-15%** | LOW |
| State 保存阻塞并发 | +5-15% | 0% | **5-15%** | MEDIUM |

---

## 2. LLM 调用流程分析

### 2.1 完整调用流程图

```
process_batch() / process_single_file()
│
├─ [CONVERT] converter.convert()                    # I/O 密集
├─ [IMAGE] image_processor.process_and_save()       # I/O + CPU
│
├─ 【分支1】OCR+LLM / PPTX+LLM 模式 (有页面截图)
│   │
│   ├─ enhance_document_with_vision()
│   │   └─ enhance_document_complete()              # llm.py:1904
│   │       │
│   │       ├─ [≤10页] _enhance_with_frontmatter()  # 🟢 1次 LLM 调用
│   │       │
│   │       └─ [>10页] 串行执行:                     # 🔴 性能问题
│   │           ├─ _enhance_document_batched()      # N 次调用 (每批10页)
│   │           └─ generate_frontmatter()           # +1 次调用 (串行等待!)
│   │
│   └─ analyze_images_with_llm()                    # 内嵌图片分析
│       └─ analyze_images_batch()                   # llm.py:1257
│           └─ for batch in batches:                # 🔴 串行循环!
│               await analyze_batch()               # 每批10张, 串行等待
│
├─ 【分支2】标准图片文件 (*.jpg/*.png)
│   └─ analyze_images_with_llm()                    # 🟢 1次调用
│
└─ 【分支3】标准文档处理 (无截图)
    │
    ├─ process_with_llm()
    │   └─ process_document()                       # llm.py:2190
    │       │
    │       ├─ [优先] _process_document_combined()  # 🟢 1次调用
    │       │
    │       └─ [降级] asyncio.gather(               # 🟢 并行 (已优化)
    │              clean_markdown(),
    │              generate_frontmatter()
    │          )
    │
    └─ analyze_images_with_llm()                    # 同分支1
```

### 2.2 并发控制机制

```
┌─────────────────────────────────────────────────────────────────┐
│ Batch Processing                                                 │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ File Semaphore (batch_concurrency=10)                    │    │
│  │  ├─ File 1 ──┐                                           │    │
│  │  ├─ File 2 ──┼──► ┌─────────────────────────────────┐   │    │
│  │  ├─ ...     ─┤    │ LLM Semaphore (concurrency=10)  │   │    │
│  │  └─ File 10 ─┘    │  ├─ LLM Call 1                  │   │    │
│  │                   │  ├─ LLM Call 2                  │   │    │
│  │  File 11+ 等待    │  ├─ ...                         │   │    │
│  └───────────────────│  └─ LLM Call 10                 │───┘    │
│                      │                                  │        │
│                      │  LLM Call 11+ 等待 semaphore    │        │
│                      └─────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────┘

问题: 10个文件共享10个LLM并发位，但每个文件内的LLM调用是串行的
```

---

## 3. 性能瓶颈详细分析

### 3.1 🔴 HIGH: 批量图片分析串行执行

**位置**: `llm.py:1285-1296` `analyze_images_batch()`

**现状代码**:
```python
# llm.py:1285-1296
for batch_num in range(num_batches):
    batch_start = batch_num * max_images_per_batch
    batch_end = min(batch_start + max_images_per_batch, len(image_paths))
    batch_paths = image_paths[batch_start:batch_end]

    logger.info(f"[{context}] Processing {len(batch_paths)} images in batch {batch_num + 1}/{num_batches}")

    batch_results = await self.analyze_batch(  # ❌ 串行等待!
        batch_paths, language, context
    )
    all_results.extend(batch_results)
```

**问题**:
- 每批最多10张图片，但批次之间是串行执行
- 假设每批需要2秒，20张图片 = 4批 = 8秒串行等待
- LLM Semaphore 有10个并发位，但只用了1个

**影响**:
- 20张图片: 8秒 → 应该只需要2秒
- **潜在加速: 4x**

**优化方案**:
```python
# 并行处理所有批次
async def analyze_images_batch(self, image_paths, language, context):
    # ... 分批逻辑 ...

    # 创建所有批次的任务
    tasks = []
    for batch_num in range(num_batches):
        batch_paths = image_paths[batch_start:batch_end]
        task = asyncio.create_task(
            self.analyze_batch(batch_paths, language, context)
        )
        tasks.append(task)

    # 并行执行所有批次
    batch_results_list = await asyncio.gather(*tasks, return_exceptions=True)

    # 合并结果，处理异常
    for batch_results in batch_results_list:
        if isinstance(batch_results, Exception):
            logger.warning(f"Batch failed: {batch_results}")
            continue
        all_results.extend(batch_results)

    return all_results
```

---

### 3.2 🔴 HIGH: 多批文档处理串行执行

**位置**: `llm.py:1957-1960` `enhance_document_complete()`

**现状代码**:
```python
# llm.py:1957-1960
# 多批文档处理 (>10页)
cleaned = await self._enhance_document_batched_simple(  # ❌ 先等待清理
    extracted_text, page_images, source, protected
)
frontmatter = await self.generate_frontmatter(cleaned, source)  # ❌ 再等待frontmatter
```

**问题**:
- 文档清理和 Frontmatter 生成是串行的
- Frontmatter 可以基于原始文本生成，不需要等待清理完成

**影响**:
- 50页文档: 清理10秒 + Frontmatter 2秒 = 12秒
- **潜在加速: 17%**

**优化方案**:
```python
# 并行执行清理和 Frontmatter 生成
clean_task = asyncio.create_task(
    self._enhance_document_batched_simple(
        extracted_text, page_images, source, protected
    )
)
# Frontmatter 基于原始文本生成 (提取足够的上下文即可)
frontmatter_task = asyncio.create_task(
    self.generate_frontmatter(extracted_text[:5000], source)  # 前5000字符
)

cleaned, frontmatter = await asyncio.gather(clean_task, frontmatter_task)
```

---

### 3.3 🟡 MEDIUM: Fallback 链串行重试

**位置**: `llm.py:1249-1251` `analyze_image()` 及相关方法

**现状流程**:
```
_analyze_with_instructor()  ──失败──► _analyze_with_json_mode()  ──失败──► _analyze_with_two_calls()
        ↓                                    ↓                                    ↓
     等待超时                              等待超时                              等待超时
      (~2s)                                (~2s)                                (~2s)
```

**问题**:
- 每个 fallback 方法失败都要等待完整超时
- 最坏情况: 3次超时 = 6秒

**优化方案 A - 快速超时**:
```python
async def _analyze_image_with_fallback(self, ...):
    # 设置较短的超时，快速转向下一个方法
    try:
        return await asyncio.wait_for(
            self._analyze_with_instructor(...),
            timeout=3.0  # 3秒超时
        )
    except (asyncio.TimeoutError, InstructorError):
        pass

    try:
        return await asyncio.wait_for(
            self._analyze_with_json_mode(...),
            timeout=3.0
        )
    except (asyncio.TimeoutError, JSONDecodeError):
        pass

    # 最后一个方法，使用完整超时
    return await self._analyze_with_two_calls(...)
```

**优化方案 B - 竞争模式** (更激进):
```python
async def _analyze_image_with_fallback(self, ...):
    # 同时启动多个方法，谁先成功用谁
    tasks = [
        asyncio.create_task(self._analyze_with_instructor(...)),
        asyncio.create_task(self._analyze_with_json_mode(...)),
    ]

    done, pending = await asyncio.wait(
        tasks,
        return_when=asyncio.FIRST_COMPLETED
    )

    # 取消未完成的任务
    for task in pending:
        task.cancel()

    # 返回第一个成功的结果
    for task in done:
        if not task.exception():
            return task.result()

    # 都失败了，使用最后的方法
    return await self._analyze_with_two_calls(...)
```

---

### 3.4 🟡 MEDIUM: State 保存阻塞并发

**位置**: `batch.py:711` `process_with_limit()`

**现状代码**:
```python
# batch.py 内 process_with_limit
async with self.semaphore:  # 文件级 semaphore
    result = await self._process_file(file_info)
    self.save_state()  # ❌ 在 semaphore 内保存状态!
```

**问题**:
- `save_state()` 是同步 I/O 操作
- 在 semaphore 内执行，阻塞其他文件的处理

**优化方案**:
```python
async with self.semaphore:
    result = await self._process_file(file_info)

# 状态保存移到 semaphore 外
await asyncio.to_thread(self.save_state)  # 非阻塞
```

**更好的方案 - 批量保存**:
```python
# 使用计数器，每处理 N 个文件保存一次
self._processed_count += 1
if self._processed_count % 10 == 0:  # 每10个文件保存一次
    await asyncio.to_thread(self.save_state)
```

---

### 3.5 🟢 LOW: I/O 操作阻塞事件循环

**位置**: `cli.py:1708, 1759, 1826` 等多处

**现状代码**:
```python
# cli.py 多处
atomic_write_text(output_file, result.markdown)  # 同步写入
```

**优化方案**:
```python
await asyncio.to_thread(atomic_write_text, output_file, result.markdown)
```

---

### 3.6 🟢 LOW: 图像缓存容量过小

**位置**: `llm.py:399`

**现状**:
```python
self._image_cache_max_size = 50  # 最多缓存50张图片
```

**问题**:
- 处理大量图片的文档时，频繁缓存淘汰
- 导致重复的文件读取和 base64 编码

**优化方案**:
```python
# 根据可用内存动态设置
self._image_cache_max_size = 200  # 或根据内存计算

# 更好的方案: 基于内存大小限制
self._image_cache_max_bytes = 100 * 1024 * 1024  # 100MB
```

---

## 4. 其他发现

### 4.1 已优化的部分 (Good)

1. **process_document 降级路径**: `llm.py:2245-2251`
   - 使用 `asyncio.gather()` 并行执行 `clean_markdown()` 和 `generate_frontmatter()`
   - ✅ 已是最佳实践

2. **LLMRuntime 共享**: `llm.py:119-156`
   - 批处理模式下，所有文件共享同一个 LLM semaphore
   - ✅ 避免了资源浪费

3. **Router 负载均衡**: 使用 litellm Router
   - 支持多模型 fallback
   - ✅ 自动重试和负载分配

### 4.2 需要进一步调查

1. **PDF OCR 重复处理**: `converter/pdf.py:59`, `ocr.py:228-236`
   - `is_scanned_pdf()` 检测时已提取文字
   - 后续 OCR 再次提取，可能存在重复

2. **PPTX 截图生成**: `converter/office.py`
   - 每页生成截图是否可以并行化？

---

## 5. 实施路线图

### Phase 1: Quick Wins (1-2小时)

| 优化项 | 文件 | 行号 | 预期收益 |
|--------|------|------|----------|
| 增大图像缓存 | llm.py | 399 | 5-10% |
| 增大 state flush 间隔 | markit.json | batch.state_flush_interval_seconds | 5-10% |
| routing_strategy 改为 simple-shuffle | markit.json | router_settings | 0-5% |

### Phase 2: Core Optimizations (4-8小时)

| 优化项 | 文件 | 预期收益 | 复杂度 |
|--------|------|----------|--------|
| 批量图片分析并行化 | llm.py:1285-1296 | **4x** | 中 |
| 多批文档并行处理 | llm.py:1957-1960 | 20% | 低 |
| State 保存移出 semaphore | batch.py:711 | 5-15% | 低 |

### Phase 3: Advanced Optimizations (可选)

| 优化项 | 预期收益 | 复杂度 |
|--------|----------|--------|
| Fallback 竞争模式 | 30-50% (失败路径) | 高 |
| 图像预加载并行化 | 10% | 中 |
| PDF OCR 去重 | 30% (扫描PDF) | 中 |

---

## 6. 配置建议

### 6.1 markit.json 优化配置

```json
{
  "llm": {
    "concurrency": 15,  // 增加 LLM 并发 (原10)
    "router_settings": {
      "routing_strategy": "simple-shuffle",  // 最佳性能
      "num_retries": 2,
      "timeout": 180  // 稍微减少超时
    }
  },
  "batch": {
    "concurrency": 15,  // 增加文件并发 (原10)
    "state_flush_interval_seconds": 30  // 减少刷盘频率 (原5)
  }
}
```

### 6.2 代码常量优化

```python
# llm.py
_image_cache_max_size = 200  # 原50

# 可选: 增加批量图片分析的批大小
max_images_per_batch = 15  # 原10 (需测试API限制)
```

---

## 7. 监控指标

实施优化后，建议监控以下指标:

1. **LLM 调用耗时分布**
   - `[LLM:*] ... time=Xms`
   - 关注 P50, P95, P99

2. **批量图片分析耗时**
   - `[LLM] *: Embedded image analysis Xs (N images)`
   - 计算每张图片平均耗时

3. **文档处理总耗时**
   - `[DONE] *: Xs`
   - 对比优化前后

4. **LLM 成本**
   - 优化不应显著增加 API 调用成本

---

## 附录: 代码位置索引

| 功能 | 文件 | 行号 | 函数 |
|------|------|------|------|
| 批量图片分析 | llm.py | 1257-1298 | `analyze_images_batch()` |
| 单批图片分析 | llm.py | 1300-1450 | `analyze_batch()` |
| 单张图片分析 | llm.py | 1187-1255 | `analyze_image()` |
| Fallback 链 | llm.py | 1249-1251 | `_analyze_image_with_fallback()` |
| 文档增强(完整) | llm.py | 1904-1962 | `enhance_document_complete()` |
| 文档处理 | llm.py | 2190-2268 | `process_document()` |
| 批处理入口 | cli.py | 1545-2014 | `process_batch()` |
| 单文件入口 | cli.py | 801-1170 | `process_single_file()` |
| State 保存 | batch.py | 428-459 | `save_state()` / flush 逻辑 |
| 图像缓存 | llm.py | 395-399 | `_image_cache_max_size` |
| LLM Semaphore | llm.py | 440-450 | `semaphore` property |
