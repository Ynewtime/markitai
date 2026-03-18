# P0-A: 并发安全与数据完整性修复

Date: 2026-03-18
Updated: 2026-03-18 (纳入 code-reviewer 评审反馈)
Parent: [2026-03-18-codebase-review-report.md](2026-03-18-codebase-review-report.md)

---

## 目标

修复代码库中的并发安全和数据完整性问题。经 code-reviewer 交叉评审后，重新校准了各 Task 的真实优先级。

## 评审校准

| Task | 初始评估 | 评审结论 | 说明 |
|------|---------|---------|------|
| Task 1 (io_semaphore) | Important | **Critical — 真实 bug** | 非 batch 路径下 `runtime=None`，每次访问创建新 Semaphore，I/O 并发限制完全失效 |
| Task 2 (ContentCache) | Critical | Defensive | asyncio 事件循环单线程保护，无活跃 bug；锁是防御 Python 3.13+ free-threaded 的前瞻措施 |
| Task 3 (cooldowns) | Critical | Defensive | 同上；同时需保护 `_image_cache`（同样的 OrderedDict 模式） |
| Task 4 (ConfigManager) | Critical | Low | 仅在 CLI 命令中单次调用（`markitai config --set`、`markitai init`），非批处理路径 |
| Task 5 (write_bytes_async) | Important | Low | 仅影响派生图片文件，可重新生成 |

---

## Task 1: io_semaphore 缓存 [Critical — 真实 bug]

### 问题

`llm/processor.py:758-766` — `io_semaphore` 属性在 `runtime=None` 时每次返回新的 `asyncio.Semaphore(DEFAULT_IO_CONCURRENCY)`，导致并发控制完全失效。

**受影响的代码路径：**
- `workflow/core.py:613, 673` — `create_llm_processor(ctx.config)` 不传 runtime
- `workflow/single.py:114` — `create_llm_processor(temp_config)` 不传 runtime
- 所有非 batch 的单文件/URL 处理路径

**对比：** `semaphore` 属性（line 748-756）正确缓存在 `self._semaphore`，`io_semaphore` 遗漏了这一模式。

### 修复方案

```python
# llm/processor.py — LLMProcessor.__init__
def __init__(self, ...):
    ...
    self._semaphore: asyncio.Semaphore | None = None
    self._io_semaphore: asyncio.Semaphore | None = None  # 新增

# llm/processor.py — io_semaphore property
@property
def io_semaphore(self) -> asyncio.Semaphore:
    """Get the I/O concurrency semaphore for file operations."""
    if self._runtime is not None:
        return self._runtime.io_semaphore
    if self._io_semaphore is None:
        self._io_semaphore = asyncio.Semaphore(DEFAULT_IO_CONCURRENCY)
    return self._io_semaphore
```

### 影响范围

- 修改文件：`llm/processor.py`（`__init__` + `io_semaphore` 属性）
- 测试：验证多次访问返回同一实例；验证非 batch 路径下 semaphore 生效

---

## Task 2: ContentCache + _image_cache 线程安全 [Defensive]

### 问题

`llm/cache.py:555-618` — `ContentCache` 使用 `OrderedDict` 无同步保护。
`llm/processor.py:693` — `_image_cache: OrderedDict` 同样无保护。

**评审结论：** asyncio 事件循环的单线程模型保护了当前 CPython 下的正确性。锁的目的是：
1. 防御 Python 3.13+ free-threaded 模式（PEP 703）
2. 防御未来代码变更引入真正的跨线程访问
3. 与 `LLMProcessor._usage_lock` 模式保持一致

### 修复方案

**ContentCache（`llm/cache.py`）：**

```python
class ContentCache:
    def __init__(self, maxsize=..., ttl_seconds=...):
        self._cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._lock = threading.Lock()
        ...

    def get(self, prompt: str, content: str) -> Any | None:
        key = self._compute_hash(prompt, content)  # 锁外计算 hash
        with self._lock:
            if key not in self._cache:
                return None
            result, timestamp = self._cache[key]
            if time.time() - timestamp > self._ttl:
                del self._cache[key]
                return None
            self._cache.move_to_end(key)
            return result

    def set(self, prompt: str, content: str, result: Any) -> None:
        key = self._compute_hash(prompt, content)  # 锁外计算 hash
        with self._lock:
            if key in self._cache:
                self._cache[key] = (result, time.time())
                self._cache.move_to_end(key)
                return
            if len(self._cache) >= self._maxsize:
                self._cache.popitem(last=False)
            self._cache[key] = (result, time.time())

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    @property
    def size(self) -> int:
        return len(self._cache)  # len() 是原子的，无需锁
```

**_image_cache（`llm/processor.py`）：**

在 `LLMProcessor.__init__` 中添加 `self._image_cache_lock = threading.Lock()`，在所有 `_image_cache` 读写处加锁。

### 设计决策

- `_compute_hash()` 在锁外执行：CPU 密集操作不应持锁
- `size` 属性不加锁：`len(OrderedDict)` 在 CPython 中是原子读
- **不解决** get-miss-llm-set 之间的重复调用：锁只保护数据结构完整性，消除重复需要 key-level locking（复杂度高，收益有限）

### 影响范围

- 修改文件：`llm/cache.py`、`llm/processor.py`
- 测试：新增 `test_content_cache_thread_safety` 用并发线程验证数据完整性

---

## Task 3: `_model_cooldowns` 并发保护 [Defensive]

### 问题

`llm/processor.py:123` (LocalProviderWrapper) 和 `400` (HybridRouter) — `_model_cooldowns` 字典在并发 `asyncio.gather` 下被无保护读写。

### 修复方案

为两个类添加 `self._cooldown_lock = threading.Lock()`：

```python
def __init__(self, ...):
    ...
    self._model_cooldowns: dict[str, float] = {}
    self._cooldown_lock = threading.Lock()

def record_cooldown(self, model_id: str, seconds: float) -> None:
    with self._cooldown_lock:
        self._model_cooldowns[model_id] = time.monotonic() + seconds
    logger.info(...)

def _select_model(self, model_name: str, has_images: bool = False) -> str:
    ...
    # 快照读取，释放锁后用 snapshot 做过滤
    with self._cooldown_lock:
        cooldowns = dict(self._model_cooldowns)
    # 后续过滤逻辑使用 cooldowns snapshot
    ...
```

HybridRouter 同理。

### ~~清理死分支~~ ⚠️ 评审修正：保留 weight<=0 分支

初始方案提议删除 `_select_model()` 中 `weight<=0` 的过滤分支（`processor.py:196-199` 和 `474-475`），理由是 `_create_router()` 已在创建时过滤。

**评审指出这是错误的：** 这些分支是有效的防御性检查，保护：
- 直接构造 `LocalProviderWrapper` / `HybridRouter` 的场景
- 未来 `_create_router()` 逻辑变更后的安全回退

**结论：保留这些分支不动。**

### 影响范围

- 修改文件：`llm/processor.py`
- 测试：新增并发 cooldown 读写测试

---

## Task 4: ConfigManager.save() 原子写入 [Low]

### 问题

`config.py:762` — 使用 `open()` + `json.dump()` 写配置文件。

**评审结论：** `save()` 仅在 `markitai config --set` 和 `markitai init` 等 CLI 命令中单次调用，不在批处理路径中。进程中断的概率极低。但与代码库其他位置的原子写入模式不一致。

### 修复方案

```python
# Before
with open(save_path, "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)
    f.write("\n")

# After
from markitai.security import atomic_write_text
content = json.dumps(output_data, indent=2, ensure_ascii=False) + "\n"
atomic_write_text(save_path, content)
```

使用 `atomic_write_text` + 手动追加换行，而非 `atomic_write_json`，因为后者不带尾部换行，且不应修改其全局默认行为。

### 影响范围

- 修改文件：`config.py`
- 测试：验证保存后文件末尾有换行符

---

## Task 5: write_bytes_async 原子写入 [Low]

### 问题

`security.py:222-234` — `write_bytes_async` 直接写入目标路径，不使用 temp+rename 模式。

**评审结论：** 仅用于保存派生的图片资产（`image.py:1046, 1227`），文件可重新生成。风险低但与项目原子写入模式不一致。

### 修复方案

将 `write_bytes_async` 内部改为原子写入：

```python
async def write_bytes_async(path: Path, data: bytes) -> None:
    """Write bytes to file atomically using temp file + rename."""
    import aiofiles

    path = Path(path)
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(
        suffix=".tmp",
        prefix=f".{path.name}.",
        dir=parent,
    )
    try:
        async with aiofiles.open(fd, "wb", closefd=True) as f:
            await f.write(data)
        await _replace_with_retry_async(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
```

直接修改 `write_bytes_async` 函数体，无需新增函数或别名。

### 影响范围

- 修改文件：`security.py`
- 测试：验证写入后文件内容正确

---

## 执行顺序

```
Task 1 (io_semaphore) ──── 唯一真实 bug，第一个修
    ↓
Task 2 (ContentCache) ─┐
Task 3 (cooldowns)     ├── 防御性加锁，可并行
    ↓                  ┘
Task 4 (ConfigManager) ─┐
Task 5 (write_bytes)    ├── 低优先，可并行
                        ┘
```

## 测试策略

- 每个 Task 遵循 TDD：先写失败测试，再改实现，再验证通过
- Task 1：直接测试 semaphore 对象身份（`assert processor.io_semaphore is processor.io_semaphore`）
- Task 2-3：并发测试使用 `threading.Thread` + `concurrent.futures` 验证数据完整性
- Task 4-5：验证文件写入行为的正确性
- 不 mock 内部实现，测试真实行为
