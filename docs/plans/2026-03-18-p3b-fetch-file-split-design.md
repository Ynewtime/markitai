# P3-B: fetch.py 大文件拆分设计

Date: 2026-03-18
Updated: 2026-03-18 (纳入 code-reviewer 评审反馈)

## 问题

`fetch.py` 达 3149 行，混合了 6 种不同职责：URL 抓取策略、HTTP 缓存、代理检测、截图工具、客户端生命周期、SPA 域名追踪。文件过大使得定位问题困难、代码审查负担重、IDE 导航不便。

## 设计原则

1. **零破坏**：所有现有 import 路径继续工作（`from markitai.fetch import X`）
2. **职责单一**：每个新模块有一个清晰的职责
3. **无循环导入**：通过 `fetch_types.py` 解耦共享类型
4. **最小改动**：只搬运代码 + 补 re-export，不重构逻辑

## 评审校准

初始方案提出 6 模块拆分，reviewer 发现两个 Critical 问题：

1. **循环导入**：`fetch_strategies.py` 需要 `FetchResult`（在 fetch.py 中定义），而 fetch.py 需要 re-export strategies → 双向依赖
2. **re-export 清单不完整**：遗漏 15+ 个被测试代码引用的名称

**解决方案**：采纳 reviewer 建议的分阶段策略 + `fetch_types.py` 解耦。

## 拆分方案（分阶段）

### Phase 1：最安全的 2 文件提取（本次实施）

```
src/markitai/
├── fetch.py               # 保留大部分代码，减少 ~800 行 (~2350 行)
├── fetch_types.py          # 共享类型/枚举/异常/数据类 (~100 行)  [新]
├── fetch_cache.py          # FetchCache + SPADomainCache (~700 行)  [新]
├── fetch_http.py           # (已存在，不变)
├── fetch_playwright.py     # (已存在，不变)
└── fetch_policy.py         # (已存在，不变)
```

**收益：**
- fetch.py 从 3149 行减至 ~2350 行（-25%）
- 最大的独立代码块（缓存 ~700 行）提取为独立模块
- `fetch_types.py` 为后续拆分消除循环导入障碍
- 测试零改动，业务零影响

### Phase 2：进一步拆分（后续按需）

Phase 1 完成后，如果仍需进一步拆分，可独立提取：
- `fetch_proxy.py` (~250 行)
- `fetch_clients.py` (~250 行)
- `fetch_strategies.py` (~900 行) — 依赖 `fetch_types.py` 解耦
- `fetch_utils.py` (~200 行)

Phase 2 的每个模块都可独立实施，不必一次性完成。

## Phase 1 详细设计

### fetch_types.py（新，~100 行）

**搬入：**
- `FetchStrategy` 枚举
- `FetchError` / `JinaRateLimitError` / `JinaAPIError` 异常类
- `FetchResult` / `ConditionalFetchResult` 数据类
- `CRITICAL_INVALID_REASONS` 常量

**依赖：** 仅标准库 + dataclasses。零外部依赖。

**设计意义：** 所有模块（fetch.py、fetch_cache.py、未来的 fetch_strategies.py）都从这里 import 类型，消除循环依赖。

### fetch_cache.py（新，~700 行）

**搬入：**
- `FetchCache` 类（全部 15 个方法）
- `SPADomainCache` 类（全部 6 个方法）
- `_make_json_safe()` 辅助函数

**依赖：**
- `fetch_types.py` — `FetchResult`, `ConditionalFetchResult`
- `markitai.security` — `atomic_write_json`
- `markitai.constants` — cache 相关常量

**全局状态：** 无。实例由 fetch.py 的 getter 管理。

### fetch.py 变更

**删除：** FetchCache、SPADomainCache、_make_json_safe 的代码（~700 行）
**删除：** FetchStrategy、FetchResult 等类型定义（~100 行）

**添加 re-export：**
```python
# Re-export shared types for backward compatibility
from markitai.fetch_types import (  # noqa: F401
    CRITICAL_INVALID_REASONS,
    ConditionalFetchResult,
    FetchError,
    FetchResult,
    FetchStrategy,
    JinaAPIError,
    JinaRateLimitError,
)

# Re-export cache classes for backward compatibility
from markitai.fetch_cache import (  # noqa: F401
    FetchCache,
    SPADomainCache,
)
```

**内部引用更新：**
- fetch.py 中所有使用 `FetchResult`、`FetchStrategy` 等的地方改为从 `fetch_types` import
- 所有使用 `FetchCache`、`SPADomainCache` 的地方改为从 `fetch_cache` import
- `_make_json_safe` 的调用方（`fetch_url()` 中使用）改为从 `fetch_cache` import

### 完整 re-export 清单

基于实际 import 图分析，以下名称必须从 fetch.py re-export：

**从 fetch_types.py：**
- `FetchStrategy`, `FetchError`, `JinaRateLimitError`, `JinaAPIError`
- `FetchResult`, `ConditionalFetchResult`
- `CRITICAL_INVALID_REASONS`

**从 fetch_cache.py：**
- `FetchCache`, `SPADomainCache`

**保留在 fetch.py（无需 re-export）：**
- `fetch_url`, `close_shared_clients`, `get_fetch_cache`, `get_spa_domain_cache`
- `get_cf_semaphore`, `get_proxy_for_url`, `fetch_with_*`（5 个策略）
- `detect_js_required`, `should_use_browser_for_domain`
- 所有 `_` 前缀的内部函数（`_detect_proxy`, `_get_system_proxy`, `_fetch_with_fallback` 等）

## 测试策略

- 不修改任何测试 import 路径
- 拆分完跑 `uv run pytest` 确认零回归
- 跑真实测试 `markitai fixtures/ --preset rich` 确认业务正常

## 风险评估

| 风险 | 缓解 |
|------|------|
| 循环导入 | fetch_types.py 无外部依赖，所有模块单向 import 它 |
| re-export 遗漏 | Phase 1 只搬 types + cache，留在 fetch.py 的名称无需 re-export |
| FetchCache 内部依赖 | FetchCache 自包含，唯一外部依赖是 `_make_json_safe`（一起搬） |
| 性能影响 | 无——只是代码组织变化，运行时行为不变 |

## 实施步骤

1. 创建 `fetch_types.py` — 搬运类型定义
2. 创建 `fetch_cache.py` — 搬运 FetchCache + SPADomainCache + _make_json_safe
3. 更新 `fetch.py` — 删除搬走的代码，添加 re-export，内部 import 指向新模块
4. 跑全量测试 + 真实测试
5. 提交
