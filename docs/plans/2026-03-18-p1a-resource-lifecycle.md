# P1-A: 资源生命周期修复

Date: 2026-03-18
Parent: [2026-03-18-codebase-review-report.md](2026-03-18-codebase-review-report.md)
Priority: P1

---

## 目标

修复全局资源的生命周期问题，主要影响测试环境和多次 `asyncio.run()` 场景。

## 评审校准

| 问题 | 初始评估 | 重新评估 | 原因 |
|------|---------|---------|------|
| I-3 全局 Semaphore 事件循环绑定 | Important | **Important** | 影响测试环境，semaphore 跨 loop 使用会报错 |
| I-4 FetchCache 初始化竞争 | Important | **已修复/不存在** | `_init_db` 已在 `self._lock` 内调用 `_get_connection()` |
| I-5 ProcessPoolExecutor 每次重建 | Important | **Low** | spawn 上下文的 ProcessPool 跨调用复用有复杂的生命周期问题，当前设计合理 |

**结论：** 只有 I-3 需要实施。I-4 经验证不存在，I-5 经分析当前设计合理（spawn 上下文 ProcessPool 的生命周期管理比 ThreadPool 复杂得多，共享 pool 需要处理 worker 死亡、孤儿进程等问题，收益不足以支撑复杂度）。

---

## Task 1: 全局 asyncio.Semaphore 事件循环重置

### 问题

`utils/executor.py:152-155` 和 `fetch.py:836-845` 的全局 `asyncio.Semaphore` 在事件循环销毁后不会重置。

**受影响场景：**
- 测试环境中 pytest-asyncio 为每个测试创建新事件循环
- CLI 中 `asyncio.run()` 多次调用（如先处理文件再处理 URL）
- 旧 semaphore 绑定到已销毁的 loop，新 loop 中 `acquire()` 会报 `RuntimeError`

### 修复方案

在 `close_shared_clients()` 中重置这两个全局 semaphore：

```python
# fetch.py — close_shared_clients()
async def close_shared_clients() -> None:
    global _cf_br_semaphore
    ...
    _cf_br_semaphore = None

# utils/executor.py — 新增 reset 函数
def reset_heavy_task_semaphore() -> None:
    global _HEAVY_TASK_SEMAPHORE
    _HEAVY_TASK_SEMAPHORE = None
```

同时在 `close_shared_clients()` 中调用 `reset_heavy_task_semaphore()`。

### 影响范围

- 修改文件：`fetch.py`、`utils/executor.py`
- 测试：验证重置后再次获取返回新实例

---

## Task 2: close_shared_clients() 补充遗漏的全局状态

### 问题

`close_shared_clients()` 未重置以下全局变量：
- `_spa_domain_cache`
- `_markitdown_instance` (line 959)
- `_detected_proxy` (line 968)

### 修复方案

在 `close_shared_clients()` 中补充重置：

```python
global _spa_domain_cache, _cf_br_semaphore
if _spa_domain_cache is not None:
    _spa_domain_cache = None
_cf_br_semaphore = None
```

`_markitdown_instance` 和 `_detected_proxy` 也一并重置。

### 影响范围

- 修改文件：`fetch.py`
- 测试：验证 cleanup 后全局状态已重置

---

## 执行顺序

1. Task 1 + Task 2 合并实施（都在同一函数中修改）

## 测试策略

- 验证 `close_shared_clients()` 后全局变量为 None
- 验证 semaphore 重置后 `get_cf_semaphore()` 返回新实例
- 验证 `reset_heavy_task_semaphore()` 重置状态
