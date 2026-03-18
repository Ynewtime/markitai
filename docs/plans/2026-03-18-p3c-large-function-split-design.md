# P3-C: 超大函数拆分设计

Date: 2026-03-18

## 问题

5 个 CLI 处理函数超过 350 行，逻辑阶段混杂在一个函数体内，可读性差、难以单独测试。

| 函数 | 文件 | 行数 | 主要问题 |
|------|------|------|----------|
| `process_url()` | cli/processors/url.py | 574 | 6 个阶段挤在一起 |
| `process_batch()` | cli/processors/batch.py | 412 | 嵌套闭包 + nonlocal |
| `create_url_processor()` | cli/processors/batch.py | 390 | LLM 分支树重复 |
| `fetch_url()` | fetch.py | 367 | 重复的策略分发链 |
| `process_url_batch()` | cli/processors/url.py | 373 | 嵌套函数 + nonlocal |

## 设计原则

1. **纯结构重构**：不改变任何行为，只提取子函数
2. **最小变量穿透**：子函数通过参数/返回值通信，不用 nonlocal
3. **文件内提取**：子函数提取为同文件的 `_private_function`，不创建新文件
4. **测试安全网**：现有测试零改动零回归

## 范围（Phase 1）

聚焦 `fetch_url()`（367 行），因为它：
- 有最清晰的阶段边界
- 有重复的策略分发链可合并
- 改动影响面最小（不涉及 CLI、闭包）
- 已有 235 个测试作为安全网

`process_url()`、`process_batch()`、`create_url_processor()`、`process_url_batch()` 留待 Phase 2，它们涉及 CLI 层的闭包和 nonlocal，风险更高。

## fetch_url() 拆分设计

### 现状分析（367 行）

```
fetch_url() 内部阶段：
├── 1. Setup + cache key (45 行)
├── 2. Conditional cache 优化 (55 行)
├── 3. Explicit strategy dispatch (87 行) ← 大 if/elif
├── 4. AUTO strategy routing (14 行)
├── 5. Non-explicit strategy dispatch (81 行) ← 与 3 几乎重复
└── 6. Post-fetch processing (40 行)
```

**核心问题**：阶段 3 和阶段 5 是几乎相同的 if/elif 策略分发链，代码重复 ~160 行。

### 拆分方案

```
fetch_url()                        # 编排器 (~80 行)
├── _resolve_cache_and_check()     # 缓存查询 (~55 行)
├── _dispatch_strategy()           # 统一策略分发 (~90 行，合并重复)
└── _finalize_fetch_result()       # 后处理 (~40 行)
```

#### `_resolve_cache_and_check()`

**职责**：处理缓存查询和条件缓存（ETag/Last-Modified）

**签名**：
```python
async def _resolve_cache_and_check(
    url: str,
    strategy: FetchStrategy,
    config: FetchConfig,
    cache: FetchCache | None,
    skip_read_cache: bool,
) -> tuple[FetchResult | None, tuple[str | None, str | None] | None]:
    """Check cache and return (cached_result, validators_to_write_later).

    Returns:
        (result, None) — cache hit, done
        (None, (etag, last_modified)) — cache miss, try conditional fetch
        (None, None) — no cache
    """
```

#### `_dispatch_strategy()`

**职责**：根据策略执行抓取，合并原来重复的两个 if/elif 链

**签名**：
```python
async def _dispatch_strategy(
    url: str,
    strategy: FetchStrategy,
    config: FetchConfig,
    explicit_strategy: bool,
    screenshot_kwargs: dict,
    renderer: Any | None,
    spa_cache: SPADomainCache | None,
    policy_config: FetchPolicyConfig | None,
) -> FetchResult:
    """Dispatch URL fetch to the appropriate strategy implementation."""
```

**关键设计**：合并 explicit/non-explicit 的重复逻辑。差异点（explicit 时不做 SPA 检测和 fallback）通过参数控制。

#### `_finalize_fetch_result()`

**职责**：截图捕获 + 缓存写入

**签名**：
```python
async def _finalize_fetch_result(
    result: FetchResult,
    url: str,
    cache: FetchCache | None,
    cache_strategy: str | None,
    validators_to_write: tuple[str | None, str | None] | None,
    screenshot_kwargs: dict,
    screenshot_config: ScreenshotConfig | None,
    renderer: Any | None,
) -> FetchResult:
    """Post-process: capture screenshot if needed, write to cache."""
```

### 变量穿透分析

跨阶段共享的变量：

| 变量 | 生产者 | 消费者 |
|------|--------|--------|
| `result` | `_dispatch_strategy` | `_finalize_fetch_result` |
| `cache_strategy` | `fetch_url` setup | `_resolve_cache_and_check`, `_finalize_fetch_result` |
| `screenshot_kwargs` | `fetch_url` setup | `_dispatch_strategy`, `_finalize_fetch_result` |
| `validators_to_write` | `_resolve_cache_and_check` | `_finalize_fetch_result` |
| `_renderer` | `fetch_url` setup | `_dispatch_strategy`, `_finalize_fetch_result` |

所有共享变量通过参数/返回值传递，无 nonlocal 或全局状态。

### 预期结果

- `fetch_url()` 从 367 行降到 ~80 行（编排逻辑）
- 消除 ~160 行重复代码（两个策略分发链合并为一个）
- 总代码行数因去重而减少
- 每个子函数可独立理解

## 实施步骤

1. 写测试验证子函数行为（如果可以，用现有 235 个测试作为安全网）
2. 提取 `_resolve_cache_and_check()`
3. 提取 `_dispatch_strategy()`（合并重复逻辑）
4. 提取 `_finalize_fetch_result()`
5. 简化 `fetch_url()` 为编排器
6. 跑全量测试 + 真实测试
