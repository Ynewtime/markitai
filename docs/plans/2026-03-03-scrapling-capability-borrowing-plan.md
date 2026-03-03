# Scrapling-Inspired Capability Borrowing Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Improve markitai URL fetching resilience and extraction quality by borrowing Scrapling architecture ideas (policy-driven strategy selection, session persistence, adaptive targeting) while keeping current dependencies and user workflow stable.

**Architecture:** Keep the existing `markitdown + Playwright + Cloudflare + Jina` stack, then add an internal capability layer: `FetchPolicyEngine`, domain profile driven wait/selector tuning, and optional persistent browser contexts. All defaults remain backward-compatible and require zero new CLI input from users.

**Tech Stack:** Python 3.11+, asyncio, pytest, pydantic, Playwright, httpx, optional extras (`curl-cffi`, `playwright-stealth`).

**Execution Discipline:** `@test-driven-development` for every task, `@verification-before-completion` before completion claims.

---

## Non-Goals

- Do not replace markitai fetch backends with Scrapling runtime.
- Do not add required new CLI flags for basic users.
- Do not introduce crawling/spider features.

## User Experience Guardrails

- Existing commands must keep working unchanged (`--playwright`, `--jina`, `--cloudflare`, `auto`).
- New behavior must be opt-in by config defaults that are safe and conservative.
- Missing optional dependencies must gracefully degrade without hard failure.
- Error messages should suggest one actionable next step only.

## Success Metrics (for verification)

- URL fetch success rate on internal URL matrix improves by >= 10% on JS-heavy domains.
- p95 URL fetch latency does not regress by > 15% on static domains.
- No regression in current fetch unit tests.
- No new required user configuration for default behavior.

---

### Task 1: Add Backward-Compatible Config Surface (Policy + Session)

**Files:**
- Modify: `packages/markitai/src/markitai/config.py`
- Test: `packages/markitai/tests/unit/test_config.py`

**Step 1: Write the failing tests**

```python
def test_fetch_policy_defaults_are_user_friendly() -> None:
    from markitai.config import MarkitaiConfig

    cfg = MarkitaiConfig()
    assert cfg.fetch.policy.enabled is True
    assert cfg.fetch.policy.max_strategy_hops == 4
    assert cfg.fetch.playwright.session_mode == "isolated"
    assert cfg.fetch.playwright.session_ttl_seconds == 600


def test_fetch_config_accepts_domain_profile_overrides() -> None:
    from markitai.config import MarkitaiConfig

    cfg = MarkitaiConfig.model_validate(
        {
            "fetch": {
                "domain_profiles": {
                    "x.com": {
                        "wait_for_selector": '[data-testid="tweetText"]',
                        "wait_for": "domcontentloaded",
                        "extra_wait_ms": 1200,
                    }
                }
            }
        }
    )

    assert cfg.fetch.domain_profiles["x.com"].wait_for_selector == '[data-testid="tweetText"]'
```

**Step 2: Run test to verify it fails**

Run: `cd packages/markitai && uv run pytest tests/unit/test_config.py -k "fetch_policy_defaults_are_user_friendly or domain_profile_overrides" -v`
Expected: FAIL with missing `policy` / `domain_profiles` fields.

**Step 3: Write minimal implementation**

```python
class FetchPolicyConfig(BaseModel):
    enabled: bool = True
    max_strategy_hops: int = Field(default=4, ge=1, le=6)


class DomainProfileConfig(BaseModel):
    wait_for_selector: str | None = None
    wait_for: Literal["load", "domcontentloaded", "networkidle"] | None = None
    extra_wait_ms: int | None = Field(default=None, ge=0, le=30000)


class PlaywrightConfig(BaseModel):
    ...
    session_mode: Literal["isolated", "domain_persistent"] = "isolated"
    session_ttl_seconds: int = Field(default=600, ge=60, le=7200)


class FetchConfig(BaseModel):
    ...
    policy: FetchPolicyConfig = Field(default_factory=FetchPolicyConfig)
    domain_profiles: dict[str, DomainProfileConfig] = Field(default_factory=dict)
```

**Step 4: Run test to verify it passes**

Run: `cd packages/markitai && uv run pytest tests/unit/test_config.py -k "fetch_policy_defaults_are_user_friendly or domain_profile_overrides" -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add packages/markitai/src/markitai/config.py packages/markitai/tests/unit/test_config.py
git commit -m "feat(fetch): add policy and domain profile config with safe defaults"
```

---

### Task 2: Introduce FetchPolicyEngine (Design Borrowing, No New Backend)

**Files:**
- Create: `packages/markitai/src/markitai/fetch_policy.py`
- Create: `packages/markitai/tests/unit/test_fetch_policy.py`
- Modify: `packages/markitai/src/markitai/fetch.py`

**Step 1: Write the failing tests**

```python
def test_policy_prefers_browser_for_known_spa_domain() -> None:
    from markitai.fetch_policy import FetchPolicyEngine

    engine = FetchPolicyEngine()
    decision = engine.decide(
        domain="x.com",
        known_spa=True,
        explicit_strategy=None,
        fallback_patterns=["x.com"],
        policy_enabled=True,
    )
    assert decision.order[:2] == ["playwright", "cloudflare"]


def test_policy_keeps_static_first_for_normal_domain() -> None:
    from markitai.fetch_policy import FetchPolicyEngine

    engine = FetchPolicyEngine()
    decision = engine.decide(
        domain="example.com",
        known_spa=False,
        explicit_strategy=None,
        fallback_patterns=["x.com"],
        policy_enabled=True,
    )
    assert decision.order[0] == "static"
```

**Step 2: Run test to verify it fails**

Run: `cd packages/markitai && uv run pytest tests/unit/test_fetch_policy.py -v`
Expected: FAIL because module/file does not exist.

**Step 3: Write minimal implementation**

```python
@dataclass
class FetchDecision:
    order: list[str]
    reason: str


class FetchPolicyEngine:
    def decide(...)-> FetchDecision:
        if not policy_enabled:
            return FetchDecision(order=["static", "playwright", "cloudflare", "jina"], reason="disabled")
        if known_spa or domain in fallback_patterns:
            return FetchDecision(order=["playwright", "cloudflare", "jina", "static"], reason="spa_or_pattern")
        return FetchDecision(order=["static", "playwright", "cloudflare", "jina"], reason="default")
```

Wire it into `_fetch_with_fallback()` by replacing hardcoded `strategies` assignment with policy output.

**Step 4: Run test to verify it passes**

Run: `cd packages/markitai && uv run pytest tests/unit/test_fetch_policy.py tests/unit/test_fetch.py -k "fallback or policy" -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add packages/markitai/src/markitai/fetch_policy.py packages/markitai/src/markitai/fetch.py packages/markitai/tests/unit/test_fetch_policy.py
git commit -m "feat(fetch): add policy engine for strategy ordering"
```

---

### Task 3: Domain Profile-Driven Wait Selector Optimization (User-Friendly)

**Files:**
- Modify: `packages/markitai/src/markitai/fetch.py`
- Modify: `packages/markitai/src/markitai/config.py`
- Test: `packages/markitai/tests/unit/test_fetch.py`

**Step 1: Write the failing tests**

```python
def test_domain_profile_applies_wait_for_selector() -> None:
    from markitai.fetch import _resolve_playwright_profile_overrides

    overrides = _resolve_playwright_profile_overrides(
        url="https://x.com/user/status/1",
        domain_profiles={
            "x.com": {
                "wait_for_selector": '[data-testid="tweetText"]',
                "wait_for": "domcontentloaded",
                "extra_wait_ms": 1200,
            }
        },
    )

    assert overrides["wait_for_selector"] == '[data-testid="tweetText"]'
    assert overrides["wait_for"] == "domcontentloaded"
```

**Step 2: Run test to verify it fails**

Run: `cd packages/markitai && uv run pytest tests/unit/test_fetch.py -k "domain_profile_applies_wait_for_selector" -v`
Expected: FAIL with missing helper.

**Step 3: Write minimal implementation**

```python
def _resolve_playwright_profile_overrides(url: str, domain_profiles: dict[str, Any]) -> dict[str, Any]:
    domain = urlparse(url).netloc.lower()
    profile = domain_profiles.get(domain)
    if not profile:
        return {}
    out: dict[str, Any] = {}
    if profile.wait_for_selector:
        out["wait_for_selector"] = profile.wait_for_selector
    if profile.wait_for:
        out["wait_for"] = profile.wait_for
    if profile.extra_wait_ms is not None:
        out["extra_wait_ms"] = profile.extra_wait_ms
    return out
```

Merge these overrides in Playwright fetch call path only when user has not explicitly set that field.

**Step 4: Run test to verify it passes**

Run: `cd packages/markitai && uv run pytest tests/unit/test_fetch.py -k "domain_profile_applies_wait_for_selector or playwright_advanced_kwargs" -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add packages/markitai/src/markitai/fetch.py packages/markitai/src/markitai/config.py packages/markitai/tests/unit/test_fetch.py
git commit -m "feat(fetch): add domain profile wait/selector overrides"
```

---

### Task 4: Add Playwright Domain-Persistent Session Mode

**Files:**
- Modify: `packages/markitai/src/markitai/fetch_playwright.py`
- Modify: `packages/markitai/src/markitai/fetch.py`
- Test: `packages/markitai/tests/unit/test_fetch_playwright.py`
- Test: `packages/markitai/tests/unit/test_fetch.py`

**Step 1: Write the failing tests**

```python
@pytest.mark.asyncio
async def test_renderer_reuses_context_in_domain_persistent_mode():
    from markitai.fetch_playwright import PlaywrightRenderer

    renderer = PlaywrightRenderer()
    renderer.enable_domain_session_cache(ttl_seconds=600, max_contexts=8)

    # fetch twice with same session_key should call browser.new_context once
    ...
    assert mock_browser.new_context.await_count == 1
```

```python
def test_fetch_builds_session_key_from_domain() -> None:
    from markitai.fetch import _url_to_session_key

    assert _url_to_session_key("https://x.com/a") == _url_to_session_key("https://x.com/b")
```

**Step 2: Run test to verify it fails**

Run: `cd packages/markitai && uv run pytest tests/unit/test_fetch_playwright.py -k "domain_persistent" -v`
Expected: FAIL because cache/session API not implemented.

**Step 3: Write minimal implementation**

```python
class PlaywrightRenderer:
    def __init__(...):
        ...
        self._context_cache: dict[str, CachedContext] = {}

    def enable_domain_session_cache(self, ttl_seconds: int, max_contexts: int) -> None:
        self._session_cache_enabled = True
        self._session_ttl_seconds = ttl_seconds
        self._max_contexts = max_contexts

    async def fetch(..., session_key: str | None = None, persist_context: bool = False):
        if persist_context and session_key:
            context = await self._get_or_create_cached_context(session_key, ctx_options)
        else:
            context = await browser.new_context(**ctx_options)
```

In `fetch.py`, pass `session_key` and `persist_context=True` only when `config.playwright.session_mode == "domain_persistent"`.

**Step 4: Run test to verify it passes**

Run: `cd packages/markitai && uv run pytest tests/unit/test_fetch_playwright.py tests/unit/test_fetch.py -k "session_key or domain_persistent" -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add packages/markitai/src/markitai/fetch_playwright.py packages/markitai/src/markitai/fetch.py packages/markitai/tests/unit/test_fetch_playwright.py packages/markitai/tests/unit/test_fetch.py
git commit -m "feat(playwright): add optional domain-persistent context reuse"
```

---

### Task 5: Optional Static HTTP Adapter (httpx default, curl-cffi optional)

**Files:**
- Create: `packages/markitai/src/markitai/fetch_http.py`
- Modify: `packages/markitai/src/markitai/fetch.py`
- Modify: `packages/markitai/pyproject.toml`
- Test: `packages/markitai/tests/unit/test_fetch.py`

**Step 1: Write the failing tests**

```python
@pytest.mark.asyncio
async def test_conditional_fetch_falls_back_when_curl_cffi_missing(monkeypatch):
    from markitai.fetch_http import get_static_http_client

    monkeypatch.setenv("MARKITAI_STATIC_HTTP", "curl_cffi")
    client = get_static_http_client()
    assert client.name in {"httpx", "curl_cffi"}
```

```python
@pytest.mark.asyncio
async def test_conditional_fetch_forwards_accept_header():
    # verify adapter receives text/markdown accept header
    ...
```

**Step 2: Run test to verify it fails**

Run: `cd packages/markitai && uv run pytest tests/unit/test_fetch.py -k "curl_cffi_missing or forwards_accept_header" -v`
Expected: FAIL because adapter layer does not exist.

**Step 3: Write minimal implementation**

```python
class StaticHttpClient(Protocol):
    name: str
    async def get(self, url: str, headers: dict[str, str], timeout_s: float, proxy: str | None) -> StaticHttpResponse: ...


def get_static_http_client() -> StaticHttpClient:
    mode = os.getenv("MARKITAI_STATIC_HTTP", "httpx")
    if mode == "curl_cffi":
        try:
            return CurlCffiClient()
        except Exception:
            return HttpxClient()
    return HttpxClient()
```

Use adapter in `fetch_with_static_conditional()` without changing default user flow.

**Step 4: Run test to verify it passes**

Run: `cd packages/markitai && uv run pytest tests/unit/test_fetch.py -k "conditional_fetch or curl_cffi" -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add packages/markitai/src/markitai/fetch_http.py packages/markitai/src/markitai/fetch.py packages/markitai/pyproject.toml packages/markitai/tests/unit/test_fetch.py
git commit -m "feat(fetch): add optional static http adapter with graceful fallback"
```

---

### Task 6: User-Friendly Error/Telemetry Layer and Documentation

**Files:**
- Modify: `packages/markitai/src/markitai/fetch.py`
- Modify: `packages/markitai/src/markitai/cli/processors/url.py`
- Modify: `docs/architecture.md`
- Create: `docs/guide/fetch-policy.md`
- Test: `packages/markitai/tests/unit/test_fetch.py`

**Step 1: Write the failing tests**

```python
def test_fetch_metadata_contains_policy_reason_without_user_noise():
    from markitai.fetch import FetchResult

    r = FetchResult(content="ok", strategy_used="static", metadata={"policy_reason": "default"})
    assert r.metadata["policy_reason"] == "default"
```

```python
def test_user_error_message_single_action_hint():
    # assert only one actionable suggestion in fetch failure text
    ...
```

**Step 2: Run test to verify it fails**

Run: `cd packages/markitai && uv run pytest tests/unit/test_fetch.py -k "policy_reason or single_action_hint" -v`
Expected: FAIL before metadata/hint normalization.

**Step 3: Write minimal implementation**

- Add `policy_reason`, `policy_order`, `profile_applied` fields to fetch metadata.
- In CLI-facing errors, keep one concise remediation hint.
- Add docs with examples that require no new flags for normal use.

**Step 4: Run test to verify it passes**

Run: `cd packages/markitai && uv run pytest tests/unit/test_fetch.py tests/unit/test_config.py tests/unit/test_fetch_playwright.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add packages/markitai/src/markitai/fetch.py packages/markitai/src/markitai/cli/processors/url.py docs/architecture.md docs/guide/fetch-policy.md packages/markitai/tests/unit/test_fetch.py
git commit -m "docs(fetch): document policy/session behavior and improve user-facing hints"
```

---

## Final Verification Gate (@verification-before-completion)

Run full checks:

```bash
cd packages/markitai
uv run pytest tests/unit/test_fetch.py tests/unit/test_fetch_playwright.py tests/unit/test_config.py -v
uv run ruff check src tests
uv run pyright
```

Expected:
- All tests PASS
- No new lint/type errors

---

## Rollout Plan

- Release as minor version feature enhancement.
- Keep default behavior equivalent to current flow.
- Mark `domain_persistent` and `MARKITAI_STATIC_HTTP=curl_cffi` as advanced/optional.
- Add changelog note with "no migration required".

## Risks and Mitigations

- Risk: Context cache leaks resources.
- Mitigation: TTL + max size + explicit `close_shared_clients()` cleanup tests.

- Risk: Policy over-optimizes and hurts static pages.
- Mitigation: policy toggle + regression matrix on static corpus.

- Risk: Optional dependency confusion.
- Mitigation: default to httpx and silent fallback.
