# Jina Reader API 充分利用方案

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 充分利用 Jina Reader API，提升 URL 抓取的成功率、速度和可靠性。

**Architecture:** 从 5 个维度增强 Jina 集成——策略升级（有 API key 时提升优先级）、并行竞争抓取（Jina 与 static 并行）、RPM 限流保护、域名级策略配置、`_fetch_multi_source` 三路并行。

**Tech Stack:** Python asyncio, httpx, Pydantic config, FetchPolicyEngine

---

## 现状分析

### 当前 Jina 使用方式
1. **策略排序中永远垫底** — `FetchPolicyEngine.decide()` 无论是否配置了 API key，Jina 都在最后位置（`["static", "playwright", "cloudflare", "jina"]`）
2. **仅作 `_fetch_multi_source` 的 last-resort** — 只有 static+playwright 都拿到无效内容时才尝试 Jina
3. **无 RPM 管理** — 批量场景下多个 URL 同时触发 Jina 调用，极易触发 20 RPM 限制（free tier）
4. **`DomainProfileConfig` 不支持策略偏好** — 不能按域名配置 "优先用 Jina"
5. **`_fetch_multi_source` 只做 static+playwright 并行** — Jina 完全不参与并行竞争

### 期望效果
- 有 Jina API key 时，Jina 在策略排序中升至第二（static → jina → playwright → cloudflare）
- 批量处理时自动限流，不超过 Jina RPM 限制
- 用户可按域名配置 `prefer_strategy: "jina"` 指定优先策略
- `_fetch_multi_source` 中 Jina 参与三路并行竞争
- 所有改动有完整测试覆盖

---

## Task 1: 策略引擎感知 Jina API Key

**目标：** 当配置了 Jina API key 时，自动提升 Jina 优先级。

**Files:**
- Modify: `packages/markitai/src/markitai/fetch_policy.py`
- Modify: `packages/markitai/src/markitai/fetch.py` (传递 `has_jina_key` 参数)
- Test: `packages/markitai/tests/unit/test_fetch_policy.py` (新建或追加)

**设计：**

`FetchPolicyEngine.decide()` 增加 `has_jina_key: bool = False` 参数：

```python
def decide(
    self,
    domain: str,
    known_spa: bool,
    explicit_strategy: str | None,
    fallback_patterns: list[str],
    policy_enabled: bool,
    has_jina_key: bool = False,  # 新增
) -> FetchDecision:
    if explicit_strategy and explicit_strategy != "auto":
        return FetchDecision(
            order=[explicit_strategy], reason=f"explicit_{explicit_strategy}"
        )

    if not policy_enabled:
        # 即使 policy 禁用，有 key 时也提升 Jina
        if has_jina_key:
            return FetchDecision(
                order=["static", "jina", "playwright", "cloudflare"],
                reason="disabled_jina_key",
            )
        return FetchDecision(
            order=["static", "playwright", "cloudflare", "jina"],
            reason="disabled",
        )

    is_fallback_domain = any(
        domain == p or domain.endswith(f".{p}") for p in fallback_patterns
    )

    if known_spa or is_fallback_domain:
        if has_jina_key:
            return FetchDecision(
                order=["playwright", "jina", "cloudflare", "static"],
                reason="spa_jina_key",
            )
        return FetchDecision(
            order=["playwright", "cloudflare", "jina", "static"],
            reason="spa_or_pattern",
        )

    # 默认策略：有 key 时 Jina 升至第二
    if has_jina_key:
        return FetchDecision(
            order=["static", "jina", "playwright", "cloudflare"],
            reason="default_jina_key",
        )

    return FetchDecision(
        order=["static", "playwright", "cloudflare", "jina"],
        reason="default",
    )
```

**调用方修改** (`fetch.py:_fetch_with_fallback`):

```python
# 在 engine.decide() 调用处增加 has_jina_key
jina_key = config.jina.get_resolved_api_key()
decision = engine.decide(
    domain=domain,
    known_spa=start_with_browser,
    explicit_strategy=config.strategy if config.strategy != "auto" else None,
    fallback_patterns=config.fallback_patterns,
    policy_enabled=config.policy.enabled,
    has_jina_key=bool(jina_key),  # 新增
)
```

**测试要点：**
- 无 API key → 默认排序不变
- 有 API key → Jina 排在第二
- SPA 域名 + 有 key → playwright, jina, cloudflare, static
- 显式策略 → 忽略 key 状态
- policy_enabled=False + 有 key → jina 仍然提升

---

## Task 2: DomainProfileConfig 支持 `prefer_strategy`

**目标：** 允许用户按域名配置首选抓取策略。

**Files:**
- Modify: `packages/markitai/src/markitai/config.py` (`DomainProfileConfig`)
- Modify: `packages/markitai/src/markitai/fetch_policy.py` (接受 `domain_prefer_strategy`)
- Modify: `packages/markitai/src/markitai/fetch.py` (从 profile 传递 prefer)
- Test: 现有 config 和 policy 测试文件

**设计：**

`DomainProfileConfig` 增加字段：

```python
class DomainProfileConfig(BaseModel):
    """Domain-specific overrides for fetch settings."""
    wait_for_selector: str | None = None
    wait_for: Literal["load", "domcontentloaded", "networkidle"] | None = None
    extra_wait_ms: int | None = Field(default=None, ge=0, le=30000)
    prefer_strategy: Literal["static", "playwright", "jina", "cloudflare"] | None = None  # 新增
```

`FetchPolicyEngine.decide()` 增加 `domain_prefer_strategy: str | None = None`：

```python
def decide(
    self,
    domain: str,
    known_spa: bool,
    explicit_strategy: str | None,
    fallback_patterns: list[str],
    policy_enabled: bool,
    has_jina_key: bool = False,
    domain_prefer_strategy: str | None = None,  # 新增
) -> FetchDecision:
    if explicit_strategy and explicit_strategy != "auto":
        return FetchDecision(...)

    # 域名偏好最高优先级（在 explicit 之后）
    if domain_prefer_strategy:
        all_strategies = ["static", "playwright", "cloudflare", "jina"]
        remaining = [s for s in all_strategies if s != domain_prefer_strategy]
        return FetchDecision(
            order=[domain_prefer_strategy] + remaining,
            reason=f"domain_prefer_{domain_prefer_strategy}",
        )
    # ... 其余逻辑不变
```

**调用方修改** (`fetch.py:_fetch_with_fallback`):

```python
from urllib.parse import urlparse

domain = urlparse(url).netloc.lower()
profile = config.domain_profiles.get(domain)
domain_prefer = profile.prefer_strategy if profile else None

decision = engine.decide(
    domain=domain,
    known_spa=start_with_browser,
    explicit_strategy=...,
    fallback_patterns=...,
    policy_enabled=...,
    has_jina_key=bool(jina_key),
    domain_prefer_strategy=domain_prefer,
)
```

**用户配置示例** (`markitai.json`):

```json
{
  "fetch": {
    "domain_profiles": {
      "docs.python.org": {
        "prefer_strategy": "jina"
      },
      "medium.com": {
        "prefer_strategy": "jina",
        "extra_wait_ms": 0
      }
    }
  }
}
```

**测试要点：**
- 域名 prefer_strategy=jina → Jina 排第一
- 域名偏好覆盖 SPA 检测
- 无 prefer_strategy → 不影响现有逻辑
- config schema 同步验证

---

## Task 3: Jina RPM 限流器

**目标：** 批量场景下自动控制 Jina 调用频率，避免触发 429。

**Files:**
- Modify: `packages/markitai/src/markitai/fetch.py` (添加限流器 + 集成到 `fetch_with_jina`)
- Modify: `packages/markitai/src/markitai/config.py` (`JinaConfig` 增加 `rpm` 字段)
- Modify: `packages/markitai/src/markitai/constants.py` (添加默认 RPM)
- Test: `packages/markitai/tests/unit/test_fetch.py` (测试限流逻辑)

**设计：**

使用 `asyncio.Semaphore` + 滑动窗口令牌桶的简化实现：

```python
# constants.py
DEFAULT_JINA_RPM = 20  # Jina free tier: 20 RPM

# config.py JinaConfig
class JinaConfig(BaseModel):
    api_key: str | None = None
    timeout: int = DEFAULT_JINA_TIMEOUT
    rpm: int = DEFAULT_JINA_RPM  # 新增: requests per minute limit
```

```python
# fetch.py — 限流器实现
import time

class _JinaRateLimiter:
    """Simple sliding-window rate limiter for Jina API calls."""

    def __init__(self, rpm: int = DEFAULT_JINA_RPM) -> None:
        self._rpm = rpm
        self._timestamps: list[float] = []
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until a request slot is available."""
        async with self._lock:
            now = time.monotonic()
            # Remove timestamps older than 60s
            cutoff = now - 60.0
            self._timestamps = [t for t in self._timestamps if t > cutoff]

            if len(self._timestamps) >= self._rpm:
                # Need to wait until the oldest timestamp expires
                wait_time = self._timestamps[0] - cutoff
                if wait_time > 0:
                    logger.debug(f"[Jina] Rate limit: waiting {wait_time:.1f}s")
                    await asyncio.sleep(wait_time)
                    # Re-clean after sleep
                    now = time.monotonic()
                    cutoff = now - 60.0
                    self._timestamps = [t for t in self._timestamps if t > cutoff]

            self._timestamps.append(time.monotonic())

# 全局限流器
_jina_rate_limiter: _JinaRateLimiter | None = None

def _get_jina_rate_limiter(rpm: int = DEFAULT_JINA_RPM) -> _JinaRateLimiter:
    global _jina_rate_limiter
    if _jina_rate_limiter is None:
        _jina_rate_limiter = _JinaRateLimiter(rpm)
    return _jina_rate_limiter
```

**集成到 `fetch_with_jina()`:**

```python
async def fetch_with_jina(
    url: str,
    api_key: str | None = None,
    timeout: int = 30,
    rpm: int = DEFAULT_JINA_RPM,  # 新增
) -> FetchResult:
    limiter = _get_jina_rate_limiter(rpm)
    await limiter.acquire()
    # ... 现有逻辑不变
```

**调用方传递 rpm** (`_fetch_with_fallback` 和 `_fetch_multi_source` 中的 Jina 调用)：

```python
result = await fetch_with_jina(url, api_key, config.jina.timeout, config.jina.rpm)
```

**测试要点：**
- 限流器在窗口内不超过 RPM 上限
- 超过 RPM 后自动等待
- 全局单例不重复创建
- `close_shared_clients()` 重置限流器

---

## Task 4: `_fetch_multi_source` 三路并行 (static + playwright + jina)

**目标：** 在 screenshot 模式下，Jina 参与三路并行竞争，而非仅作 last-resort fallback。

**Files:**
- Modify: `packages/markitai/src/markitai/fetch.py` (`_fetch_multi_source`)
- Test: `packages/markitai/tests/unit/test_fetch.py`

**设计：**

当有 Jina API key 时，将 Jina 加入 `asyncio.gather` 并行抓取：

```python
async def _fetch_multi_source(
    url: str,
    config: FetchConfig,
    screenshot_dir: Path | None = None,
    screenshot_config: ScreenshotConfig | None = None,
    cache: FetchCache | None = None,
    skip_read_cache: bool = False,
    renderer: Any | None = None,
) -> FetchResult:
    # ... 现有 fetch_static 和 fetch_browser 定义不变 ...

    # Task 3 (新增): Jina 并行抓取
    jina_key = config.jina.get_resolved_api_key()
    jina_content: str | None = None

    async def fetch_jina() -> str | None:
        if not jina_key:
            return None
        try:
            result = await fetch_with_jina(url, jina_key, config.jina.timeout, config.jina.rpm)
            logger.debug(f"[URL] Jina fetch success: {len(result.content)} chars")
            return result.content
        except Exception as e:
            logger.debug(f"[URL] Jina fetch failed: {e}")
            return None

    # 三路并行
    static_content, browser_fetch_result, jina_content = await asyncio.gather(
        fetch_static(), fetch_browser(), fetch_jina()
    )

    # ... 现有 browser_result 解包逻辑不变 ...

    # 验证三路内容质量
    jina_invalid, jina_reason = (
        _is_invalid_content(jina_content) if jina_content else (True, "not_attempted")
    )

    # 选择最佳内容的优先级：
    # 1. static 有效 → 用 static
    # 2. browser 有效 → 用 browser
    # 3. jina 有效 → 用 jina（无截图但有内容）
    # 4. 都无效 → 原有 fallback 逻辑
    if not static_invalid:
        primary_content = static_content
        strategy_used = "static"
    elif not browser_invalid:
        primary_content = browser_content
        strategy_used = "browser"
    elif not jina_invalid:
        # Jina 内容有效 — 使用 Jina 内容 + 如有截图则附加
        primary_content = jina_content
        strategy_used = "jina"
    elif browser_content:
        # 三路都无效 — 走原有的严格失败逻辑（不再需要额外 Jina fallback 调用）
        raise FetchError(
            f"All fetch strategies returned invalid content for {url}. "
            f"Static: {static_reason}, Browser: {browser_reason}, "
            f"Jina: {jina_reason}."
        )
    elif static_content:
        # 无 browser 但有 static — 原有逻辑
        # ...
    else:
        raise FetchError(f"All strategies failed for {url}")
```

**关键改动：**
- Jina 和 static/playwright 同时发起，不额外增加延迟
- 无 API key 时 `fetch_jina()` 直接返回 None（零开销）
- 不再需要 `_fetch_multi_source` 末尾的两处 Jina last-resort 调用（已在并行中完成）
- 截图仍由 playwright 提供，Jina 贡献内容

**测试要点：**
- 有 key 时三路并行都发起
- 无 key 时 Jina 不调用
- static/browser 都无效但 Jina 有效 → 使用 Jina 内容
- 三路都无效 → FetchError
- Jina rate limit 不影响 static/browser 结果

---

## Task 5: JinaConfig 增加高级选项

**目标：** 增加 `return_format` 和 `include_links` 等 Jina Reader API 高级参数。

**Files:**
- Modify: `packages/markitai/src/markitai/config.py` (`JinaConfig`)
- Modify: `packages/markitai/src/markitai/constants.py` (新增默认值)
- Modify: `packages/markitai/src/markitai/fetch.py` (`fetch_with_jina` 增加 headers)
- Test: `packages/markitai/tests/unit/test_fetch.py`

**设计：**

```python
# config.py
class JinaConfig(BaseModel):
    api_key: str | None = None
    timeout: int = DEFAULT_JINA_TIMEOUT
    rpm: int = DEFAULT_JINA_RPM
    no_cache: bool = False  # 跳过 Jina 服务端缓存 (X-No-Cache: true)
    include_links: bool = False  # 在 Markdown 中保留链接 (X-Retain-Images: none)
    target_selector: str | None = None  # CSS 选择器提取特定内容 (X-Target-Selector)
    wait_for_selector: str | None = None  # 等待特定元素加载 (X-Wait-For-Selector)
```

```python
# fetch.py fetch_with_jina() headers 增强
headers = {
    "Accept": "application/json",
}
if api_key:
    headers["Authorization"] = f"Bearer {api_key}"
if no_cache:
    headers["X-No-Cache"] = "true"
if target_selector:
    headers["X-Target-Selector"] = target_selector
if wait_for_selector:
    headers["X-Wait-For-Selector"] = wait_for_selector
```

**用户配置示例：**

```json
{
  "fetch": {
    "jina": {
      "api_key": "env:JINA_API_KEY",
      "rpm": 200,
      "target_selector": "article, main, .content"
    },
    "domain_profiles": {
      "medium.com": {
        "prefer_strategy": "jina"
      }
    }
  }
}
```

**测试要点：**
- 默认值不添加额外 headers
- 设置 no_cache=True → 包含 X-No-Cache header
- target_selector 正确传递到 header
- config schema 同步

---

## Task 6: config.schema.json 同步 + 测试覆盖

**目标：** 确保所有配置变更反映在 JSON schema 中，运行全量测试。

**Files:**
- Modify: `packages/markitai/src/markitai/config.schema.json`
- Run: 全量测试

**步骤：**

1. 运行 schema 生成命令或手动同步 `JinaConfig` 和 `DomainProfileConfig` 的新字段
2. 运行 `test_schema_sync.py` 确保 schema 与 Pydantic model 一致
3. 运行全量 `uv run pytest packages/markitai/tests/ -x` 确认无回归

---

## 实施优先级

| 优先级 | Task | 影响 | 复杂度 |
|--------|------|------|--------|
| P0 | Task 1: 策略引擎感知 API key | 高 — 立即提升 Jina 使用率 | 低 |
| P0 | Task 3: RPM 限流器 | 高 — 避免批量 429 | 中 |
| P1 | Task 4: 三路并行 | 高 — 提升抓取成功率 | 中 |
| P1 | Task 2: 域名策略偏好 | 中 — 用户精细控制 | 低 |
| P2 | Task 5: 高级选项 | 低 — 增值功能 | 低 |
| P2 | Task 6: Schema 同步 | 必要 — 配置校验 | 低 |

---

## 风险与缓解

1. **Jina 免费层 20 RPM 限制** — Task 3 的限流器直接解决
2. **Jina 不支持截图** — Task 4 中 Jina 贡献内容，截图由 playwright 提供
3. **向后兼容** — 所有新功能默认关闭（无 key = 行为不变），零破坏性改动
4. **网络开销** — 三路并行中 Jina 是轻量 HTTP GET，额外开销极小
