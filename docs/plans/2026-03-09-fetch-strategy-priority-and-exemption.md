# Fetch Strategy Priority Override & Domain Exemption — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Allow users to customize URL fetch strategy priority order (globally and per-domain) and configure domain/IP exemptions that restrict fetching to local-only strategies for information security.

**Architecture:** Extend `FetchPolicyConfig` with `strategy_priority`, `local_only_patterns`, and `inherit_no_proxy` fields. Extend `DomainProfileConfig` with `strategy_priority`. The `FetchPolicyEngine.decide()` method applies these overrides in a clear priority chain. Domain exemption matching follows NO_PROXY syntax conventions with CIDR support.

**Tech Stack:** Pydantic v2 validators, Python `ipaddress` stdlib for CIDR matching.

---

## Task 1: Add strategy constants to `constants.py`

**Files:**
- Modify: `packages/markitai/src/markitai/constants.py`
- Test: `packages/markitai/tests/unit/test_fetch_policy.py`

**Step 1: Write the failing test**

Add to `test_fetch_policy.py`:

```python
from markitai.constants import ALL_FETCH_STRATEGIES, EXTERNAL_STRATEGIES, LOCAL_STRATEGIES


def test_strategy_constants_are_consistent() -> None:
    """LOCAL + EXTERNAL = ALL, no overlap."""
    assert set(LOCAL_STRATEGIES) | set(EXTERNAL_STRATEGIES) == set(ALL_FETCH_STRATEGIES)
    assert set(LOCAL_STRATEGIES) & set(EXTERNAL_STRATEGIES) == set()
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest packages/markitai/tests/unit/test_fetch_policy.py::test_strategy_constants_are_consistent -v`
Expected: FAIL with `ImportError`

**Step 3: Write minimal implementation**

Add to `constants.py` after the existing URL Fetch Settings section (around line 236, before `# Local LLM Provider Settings`):

```python
# Fetch strategy categories
ALL_FETCH_STRATEGIES: tuple[str, ...] = (
    "defuddle", "jina", "static", "playwright", "cloudflare",
)
EXTERNAL_STRATEGIES: tuple[str, ...] = ("defuddle", "jina", "cloudflare")
LOCAL_STRATEGIES: tuple[str, ...] = ("static", "playwright")
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest packages/markitai/tests/unit/test_fetch_policy.py::test_strategy_constants_are_consistent -v`
Expected: PASS

**Step 5: Update `fetch_policy.py` to use new constants**

Replace the module-level constants in `fetch_policy.py`:

```python
# Old (lines 21-22):
ALL_STRATEGIES = ["defuddle", "jina", "static", "playwright", "cloudflare"]
LOCAL_ONLY_STRATEGIES = ["static", "playwright"]

# New:
from markitai.constants import ALL_FETCH_STRATEGIES, LOCAL_STRATEGIES

ALL_STRATEGIES = list(ALL_FETCH_STRATEGIES)
LOCAL_ONLY_STRATEGIES = list(LOCAL_STRATEGIES)
```

**Step 6: Run all fetch_policy tests**

Run: `uv run pytest packages/markitai/tests/unit/test_fetch_policy.py -v`
Expected: All existing tests PASS

---

## Task 2: Add `local_only_patterns` matching to `fetch_policy.py`

**Files:**
- Modify: `packages/markitai/src/markitai/fetch_policy.py`
- Test: `packages/markitai/tests/unit/test_fetch_policy.py`

**Step 1: Write the failing tests**

Add to `test_fetch_policy.py`:

```python
from markitai.fetch_policy import match_local_only, parse_no_proxy


class TestMatchLocalOnly:
    """Tests for NO_PROXY-style local-only pattern matching."""

    def test_exact_domain_match(self) -> None:
        assert match_local_only("internal.corp.com", ["internal.corp.com"]) is True

    def test_domain_no_match(self) -> None:
        assert match_local_only("example.com", ["internal.corp.com"]) is False

    def test_suffix_with_leading_dot(self) -> None:
        assert match_local_only("app.internal.com", [".internal.com"]) is True

    def test_suffix_does_not_match_exact(self) -> None:
        assert match_local_only("internal.com", [".internal.com"]) is False

    def test_wildcard_prefix(self) -> None:
        assert match_local_only("app.internal.com", ["*.internal.com"]) is True

    def test_ip_exact_match(self) -> None:
        assert match_local_only("10.0.1.5", ["10.0.1.5"]) is True

    def test_cidr_match(self) -> None:
        assert match_local_only("10.0.1.5", ["10.0.0.0/8"]) is True

    def test_cidr_no_match(self) -> None:
        assert match_local_only("192.168.1.1", ["10.0.0.0/8"]) is False

    def test_ipv6_cidr(self) -> None:
        assert match_local_only("fd12::1", ["fd00::/8"]) is True

    def test_localhost_special(self) -> None:
        assert match_local_only("localhost", ["localhost"]) is True

    def test_localhost_with_port(self) -> None:
        assert match_local_only("localhost:3000", ["localhost"]) is True

    def test_empty_patterns(self) -> None:
        assert match_local_only("example.com", []) is False

    def test_multiple_patterns_any_match(self) -> None:
        patterns = [".corp.com", "10.0.0.0/8"]
        assert match_local_only("app.corp.com", patterns) is True
        assert match_local_only("10.1.2.3", patterns) is True
        assert match_local_only("example.com", patterns) is False


class TestParseNoProxy:
    """Tests for NO_PROXY environment variable parsing."""

    def test_comma_separated(self) -> None:
        result = parse_no_proxy("localhost,.internal.com,10.0.0.0/8")
        assert result == ["localhost", ".internal.com", "10.0.0.0/8"]

    def test_strips_whitespace(self) -> None:
        result = parse_no_proxy(" localhost , .corp.com ")
        assert result == ["localhost", ".corp.com"]

    def test_empty_string(self) -> None:
        assert parse_no_proxy("") == []

    def test_none(self) -> None:
        assert parse_no_proxy(None) == []

    def test_filters_empty_entries(self) -> None:
        result = parse_no_proxy("localhost,,,.corp.com")
        assert result == ["localhost", ".corp.com"]

    def test_wildcard_star_only(self) -> None:
        """NO_PROXY=* means match everything — we preserve it as-is."""
        result = parse_no_proxy("*")
        assert result == ["*"]
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/markitai/tests/unit/test_fetch_policy.py::TestMatchLocalOnly -v`
Expected: FAIL with `ImportError: cannot import name 'match_local_only'`

**Step 3: Write the implementation**

Add to `fetch_policy.py` (after `is_private_or_local_domain`, before `FetchDecision`):

```python
def parse_no_proxy(value: str | None) -> list[str]:
    """Parse a NO_PROXY-style comma-separated string into a list of patterns.

    Strips whitespace, filters empty entries.
    """
    if not value:
        return []
    return [p.strip() for p in value.split(",") if p.strip()]


def match_local_only(domain: str, patterns: list[str]) -> bool:
    """Check if domain matches any local-only exemption pattern.

    Supports NO_PROXY syntax:
    - Domain exact: ``internal.corp.com``
    - Suffix: ``.internal.com`` (matches subdomains only)
    - Wildcard: ``*.internal.com`` (same as ``.internal.com``)
    - IP exact: ``10.0.1.5``
    - CIDR: ``10.0.0.0/8``, ``fd00::/8``
    - Star: ``*`` (matches everything)
    - Special: ``localhost``
    """
    if not patterns:
        return False

    host = _extract_host(domain).strip().lower()
    if not host:
        return False

    for pattern in patterns:
        p = pattern.strip().lower()
        if not p:
            continue

        # Wildcard: match everything
        if p == "*":
            return True

        # Normalize *.foo.com → .foo.com
        if p.startswith("*."):
            p = p[1:]  # "*.foo.com" → ".foo.com"

        # Suffix match: .foo.com matches sub.foo.com but not foo.com
        if p.startswith("."):
            if host.endswith(p):
                return True
            continue

        # CIDR match: try parsing pattern as network
        if "/" in p:
            try:
                network = ipaddress.ip_network(p, strict=False)
                host_ip = ipaddress.ip_address(host)
                if host_ip in network:
                    return True
            except ValueError:
                pass
            continue

        # Exact match (domain or IP)
        if host == p:
            return True

    return False
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/markitai/tests/unit/test_fetch_policy.py::TestMatchLocalOnly packages/markitai/tests/unit/test_fetch_policy.py::TestParseNoProxy -v`
Expected: All PASS

---

## Task 3: Add config fields with Pydantic validators

**Files:**
- Modify: `packages/markitai/src/markitai/config.py:341-356`
- Test: `packages/markitai/tests/unit/test_config.py`

**Step 1: Write the failing tests**

Add to `test_config.py`:

```python
import pytest
from pydantic import ValidationError

from markitai.config import DomainProfileConfig, FetchPolicyConfig


class TestFetchPolicyConfigValidation:
    """Tests for FetchPolicyConfig new fields."""

    def test_defaults(self) -> None:
        cfg = FetchPolicyConfig()
        assert cfg.strategy_priority is None
        assert cfg.local_only_patterns == []
        assert cfg.inherit_no_proxy is True

    def test_valid_strategy_priority(self) -> None:
        cfg = FetchPolicyConfig(strategy_priority=["static", "playwright"])
        assert cfg.strategy_priority == ["static", "playwright"]

    def test_invalid_strategy_name(self) -> None:
        with pytest.raises(ValidationError, match="invalid_strategy"):
            FetchPolicyConfig(strategy_priority=["static", "invalid"])

    def test_duplicate_strategies(self) -> None:
        with pytest.raises(ValidationError, match="duplicate"):
            FetchPolicyConfig(strategy_priority=["static", "static"])

    def test_empty_strategy_priority_rejected(self) -> None:
        with pytest.raises(ValidationError, match="empty"):
            FetchPolicyConfig(strategy_priority=[])

    def test_valid_local_only_patterns(self) -> None:
        cfg = FetchPolicyConfig(
            local_only_patterns=[".corp.com", "10.0.0.0/8", "localhost"]
        )
        assert len(cfg.local_only_patterns) == 3

    def test_empty_pattern_rejected(self) -> None:
        with pytest.raises(ValidationError, match="empty"):
            FetchPolicyConfig(local_only_patterns=[""])

    def test_invalid_cidr_rejected(self) -> None:
        with pytest.raises(ValidationError, match="CIDR"):
            FetchPolicyConfig(local_only_patterns=["999.0.0.0/8"])


class TestDomainProfileStrategyPriority:
    """Tests for DomainProfileConfig.strategy_priority."""

    def test_default_none(self) -> None:
        cfg = DomainProfileConfig()
        assert cfg.strategy_priority is None

    def test_valid_priority(self) -> None:
        cfg = DomainProfileConfig(strategy_priority=["static"])
        assert cfg.strategy_priority == ["static"]

    def test_invalid_strategy(self) -> None:
        with pytest.raises(ValidationError, match="invalid_strategy"):
            DomainProfileConfig(strategy_priority=["bogus"])

    def test_duplicate_rejected(self) -> None:
        with pytest.raises(ValidationError, match="duplicate"):
            DomainProfileConfig(strategy_priority=["static", "static"])
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/markitai/tests/unit/test_config.py::TestFetchPolicyConfigValidation -v`
Expected: FAIL (`strategy_priority` field does not exist)

**Step 3: Write the implementation**

Modify `FetchPolicyConfig` in `config.py` (lines 341-345):

```python
class FetchPolicyConfig(BaseModel):
    """Configuration for fetch strategy policy engine."""

    enabled: bool = True
    max_strategy_hops: int = Field(default=5, ge=1, le=6)
    strategy_priority: list[str] | None = Field(
        default=None,
        description="Custom global strategy order. Overrides the default priority.",
    )
    local_only_patterns: list[str] = Field(
        default_factory=list,
        description="Domain/IP patterns that must use local-only strategies (NO_PROXY syntax).",
    )
    inherit_no_proxy: bool = Field(
        default=True,
        description="Merge NO_PROXY env var into local_only_patterns at runtime.",
    )

    @field_validator("strategy_priority")
    @classmethod
    def validate_strategy_priority(
        cls, v: list[str] | None,
    ) -> list[str] | None:
        if v is None:
            return None
        if len(v) == 0:
            raise ValueError("strategy_priority must not be empty if set")
        _validate_strategy_list(v)
        return v

    @field_validator("local_only_patterns")
    @classmethod
    def validate_local_only_patterns(cls, v: list[str]) -> list[str]:
        for pattern in v:
            _validate_local_only_pattern(pattern)
        return v
```

Modify `DomainProfileConfig` (lines 348-356):

```python
class DomainProfileConfig(BaseModel):
    """Domain-specific overrides for fetch settings."""

    wait_for_selector: str | None = None
    wait_for: Literal["load", "domcontentloaded", "networkidle"] | None = None
    extra_wait_ms: int | None = Field(default=None, ge=0, le=30000)
    prefer_strategy: (
        Literal["static", "defuddle", "playwright", "jina", "cloudflare"] | None
    ) = None
    strategy_priority: list[str] | None = Field(
        default=None,
        description="Custom strategy order for this domain. Overrides global and prefer_strategy.",
    )

    @field_validator("strategy_priority")
    @classmethod
    def validate_strategy_priority(
        cls, v: list[str] | None,
    ) -> list[str] | None:
        if v is None:
            return None
        if len(v) == 0:
            raise ValueError("strategy_priority must not be empty if set")
        _validate_strategy_list(v)
        return v
```

Add the shared validation helpers near the top of `config.py` (after imports, before model classes — around line 40):

```python
import ipaddress as _ipaddress

from markitai.constants import ALL_FETCH_STRATEGIES

def _validate_strategy_list(strategies: list[str]) -> None:
    """Validate a list of strategy names (shared by FetchPolicyConfig and DomainProfileConfig)."""
    valid = set(ALL_FETCH_STRATEGIES)
    for s in strategies:
        if s not in valid:
            raise ValueError(
                f"invalid_strategy: '{s}'. Must be one of {sorted(valid)}"
            )
    if len(strategies) != len(set(strategies)):
        raise ValueError("duplicate strategies in strategy_priority")


def _validate_local_only_pattern(pattern: str) -> None:
    """Validate a single local-only pattern (NO_PROXY syntax)."""
    if not pattern or not pattern.strip():
        raise ValueError("empty pattern in local_only_patterns")
    p = pattern.strip()
    # CIDR validation
    if "/" in p:
        try:
            _ipaddress.ip_network(p, strict=False)
        except ValueError as e:
            raise ValueError(f"invalid CIDR in local_only_patterns: '{p}' — {e}") from e
```

Add `field_validator` to the imports at the top of `config.py`:

```python
from pydantic import BaseModel, Field, field_validator
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/markitai/tests/unit/test_config.py::TestFetchPolicyConfigValidation packages/markitai/tests/unit/test_config.py::TestDomainProfileStrategyPriority -v`
Expected: All PASS

---

## Task 4: Update `FetchPolicyEngine.decide()` to apply overrides

**Files:**
- Modify: `packages/markitai/src/markitai/fetch_policy.py`
- Test: `packages/markitai/tests/unit/test_fetch_policy.py`

**Step 1: Write the failing tests**

Add to `test_fetch_policy.py`:

```python
class TestStrategyPriorityOverride:
    """Tests for global and per-domain strategy_priority override."""

    def setup_method(self) -> None:
        self.engine = FetchPolicyEngine()

    def test_global_priority_overrides_default(self) -> None:
        decision = self.engine.decide(
            "example.com", False, None, [], True,
            global_strategy_priority=["static", "playwright"],
        )
        assert decision.order == ["static", "playwright"]
        assert decision.reason == "global_priority"

    def test_domain_priority_overrides_global(self) -> None:
        decision = self.engine.decide(
            "example.com", False, None, [], True,
            global_strategy_priority=["static", "playwright"],
            domain_strategy_priority=["defuddle", "static"],
        )
        assert decision.order == ["defuddle", "static"]
        assert decision.reason == "domain_priority"

    def test_domain_priority_overrides_prefer(self) -> None:
        decision = self.engine.decide(
            "example.com", False, None, [], True,
            domain_prefer_strategy="jina",
            domain_strategy_priority=["static"],
        )
        assert decision.order == ["static"]
        assert decision.reason == "domain_priority"

    def test_explicit_still_wins_over_all(self) -> None:
        decision = self.engine.decide(
            "example.com", False, "playwright", [], True,
            global_strategy_priority=["static"],
            domain_strategy_priority=["defuddle"],
        )
        assert decision.order == ["playwright"]

    def test_global_priority_still_respects_spa(self) -> None:
        """Global priority replaces base order, SPA/fallback detection is bypassed."""
        decision = self.engine.decide(
            "twitter.com", True, None, ["twitter.com"], True,
            global_strategy_priority=["static", "playwright"],
        )
        assert decision.order == ["static", "playwright"]
        assert decision.reason == "global_priority"


class TestLocalOnlyExemption:
    """Tests for local_only_patterns domain exemption."""

    def setup_method(self) -> None:
        self.engine = FetchPolicyEngine()

    def test_matching_domain_forces_local_strategies(self) -> None:
        decision = self.engine.decide(
            "app.internal.com", False, None, [], True,
            local_only_patterns=[".internal.com"],
        )
        assert decision.order == ["static", "playwright"]
        assert decision.reason == "local_only_pattern"

    def test_cidr_match_forces_local(self) -> None:
        decision = self.engine.decide(
            "10.0.1.5", False, None, [], True,
            local_only_patterns=["10.0.0.0/8"],
        )
        assert decision.order == ["static", "playwright"]
        assert decision.reason == "local_only_pattern"

    def test_non_matching_domain_uses_normal_order(self) -> None:
        decision = self.engine.decide(
            "example.com", False, None, [], True,
            local_only_patterns=[".internal.com"],
        )
        assert decision.order[0] == "defuddle"  # normal default

    def test_local_only_overrides_global_priority(self) -> None:
        """Security exemption takes precedence over custom priority."""
        decision = self.engine.decide(
            "app.internal.com", False, None, [], True,
            global_strategy_priority=["defuddle", "jina", "static"],
            local_only_patterns=[".internal.com"],
        )
        assert decision.order == ["static", "playwright"]
        assert decision.reason == "local_only_pattern"

    def test_local_only_overrides_domain_priority(self) -> None:
        decision = self.engine.decide(
            "app.internal.com", False, None, [], True,
            domain_strategy_priority=["defuddle"],
            local_only_patterns=[".internal.com"],
        )
        assert decision.order == ["static", "playwright"]

    def test_explicit_strategy_overrides_local_only(self) -> None:
        """Explicit CLI flag always wins (user knows what they're doing)."""
        decision = self.engine.decide(
            "app.internal.com", False, "defuddle", [], True,
            local_only_patterns=[".internal.com"],
        )
        assert decision.order == ["defuddle"]

    def test_wildcard_star_matches_all(self) -> None:
        decision = self.engine.decide(
            "example.com", False, None, [], True,
            local_only_patterns=["*"],
        )
        assert decision.order == ["static", "playwright"]

    def test_spa_domain_with_local_only_stays_local(self) -> None:
        decision = self.engine.decide(
            "spa.internal.com", True, None, [], True,
            local_only_patterns=[".internal.com"],
        )
        assert decision.order == ["playwright", "static"]
        assert "local_only" in decision.reason
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/markitai/tests/unit/test_fetch_policy.py::TestStrategyPriorityOverride -v`
Expected: FAIL with `TypeError: decide() got an unexpected keyword argument 'global_strategy_priority'`

**Step 3: Update `decide()` signature and implementation**

Replace the entire `decide()` method in `fetch_policy.py`:

```python
class FetchPolicyEngine:
    """Engine to decide the order of fetch strategies based on URL and config."""

    def decide(
        self,
        domain: str,
        known_spa: bool,
        explicit_strategy: str | None,
        fallback_patterns: list[str],
        policy_enabled: bool,
        has_jina_key: bool = False,
        domain_prefer_strategy: str | None = None,
        global_strategy_priority: list[str] | None = None,
        domain_strategy_priority: list[str] | None = None,
        local_only_patterns: list[str] | None = None,
    ) -> FetchDecision:
        """Decide the fetch strategy order.

        Priority chain (highest to lowest):
        1. Explicit strategy (CLI flag) — single strategy, no fallback
        2. Local-only exemption — security: restrict to local strategies
        3. Private/local domain — hardcoded local detection
        4. Domain strategy_priority — per-domain full override
        5. Domain prefer_strategy — per-domain single promotion
        6. Global strategy_priority — global full override
        7. SPA/JS-required pattern — browser-first order
        8. Default order
        """
        # 1. Explicit strategy always wins
        if explicit_strategy and explicit_strategy != "auto":
            return FetchDecision(
                order=[explicit_strategy], reason=f"explicit_{explicit_strategy}"
            )

        is_fallback_domain = any(
            domain == p or domain.endswith(f".{p}") for p in fallback_patterns
        )

        # 2. Local-only exemption (security: skip external APIs)
        if local_only_patterns and match_local_only(domain, local_only_patterns):
            order = (
                ["playwright", "static"]
                if known_spa or is_fallback_domain
                else LOCAL_ONLY_STRATEGIES.copy()
            )
            return FetchDecision(order=order, reason="local_only_pattern")

        # 3. Private/local domain (hardcoded detection)
        if is_private_or_local_domain(domain):
            order = (
                ["playwright", "static"]
                if known_spa or is_fallback_domain
                else LOCAL_ONLY_STRATEGIES.copy()
            )
            return FetchDecision(order=order, reason="private_or_local")

        # 4. Per-domain strategy_priority (full override)
        if domain_strategy_priority:
            return FetchDecision(
                order=list(domain_strategy_priority), reason="domain_priority"
            )

        # 5. Per-domain prefer_strategy (single promotion)
        if domain_prefer_strategy:
            remaining = [s for s in ALL_STRATEGIES if s != domain_prefer_strategy]
            return FetchDecision(
                order=[domain_prefer_strategy] + remaining,
                reason=f"domain_prefer_{domain_prefer_strategy}",
            )

        # 6. Global strategy_priority (full override)
        if global_strategy_priority:
            return FetchDecision(
                order=list(global_strategy_priority), reason="global_priority"
            )

        if not policy_enabled:
            return FetchDecision(
                order=ALL_STRATEGIES.copy(),
                reason="disabled",
            )

        # 7. SPA/JS-heavy: playwright earlier
        if known_spa or is_fallback_domain:
            return FetchDecision(
                order=["defuddle", "jina", "playwright", "cloudflare", "static"],
                reason="spa_or_pattern",
            )

        # 8. Default
        return FetchDecision(
            order=ALL_STRATEGIES.copy(),
            reason="default",
        )
```

**Step 4: Run all fetch_policy tests**

Run: `uv run pytest packages/markitai/tests/unit/test_fetch_policy.py -v`
Expected: All PASS

---

## Task 5: Wire config through `fetch.py` call site

**Files:**
- Modify: `packages/markitai/src/markitai/fetch.py` (around line 2692-2704)
- Test: `packages/markitai/tests/unit/test_fetch.py` (existing integration — verify no regression)

**Step 1: Update the `_fetch_with_fallback()` call site**

Find the `engine.decide()` call in `fetch.py` (around line 2696) and update it:

```python
# Old:
domain = urlparse(url).netloc.lower()
engine = FetchPolicyEngine()
jina_key = config.jina.get_resolved_api_key()
profile = config.domain_profiles.get(domain)
domain_prefer = profile.prefer_strategy if profile else None
decision = engine.decide(
    domain=domain,
    known_spa=start_with_browser,
    explicit_strategy=config.strategy if config.strategy != "auto" else None,
    fallback_patterns=config.fallback_patterns,
    policy_enabled=config.policy.enabled,
    has_jina_key=bool(jina_key),
    domain_prefer_strategy=domain_prefer,
)

# New:
domain = urlparse(url).netloc.lower()
engine = FetchPolicyEngine()
jina_key = config.jina.get_resolved_api_key()
profile = config.domain_profiles.get(domain)
domain_prefer = profile.prefer_strategy if profile else None
domain_priority = profile.strategy_priority if profile else None

# Build effective local_only_patterns (config + optional NO_PROXY merge)
effective_local_only = _build_local_only_patterns(config.policy)

decision = engine.decide(
    domain=domain,
    known_spa=start_with_browser,
    explicit_strategy=config.strategy if config.strategy != "auto" else None,
    fallback_patterns=config.fallback_patterns,
    policy_enabled=config.policy.enabled,
    has_jina_key=bool(jina_key),
    domain_prefer_strategy=domain_prefer,
    global_strategy_priority=config.policy.strategy_priority,
    domain_strategy_priority=domain_priority,
    local_only_patterns=effective_local_only,
)
```

Add the helper function in `fetch.py` (near the top-level helpers, before `_fetch_with_fallback()`):

```python
def _build_local_only_patterns(policy: FetchPolicyConfig) -> list[str]:
    """Build effective local-only patterns from config + NO_PROXY env var."""
    import os

    from markitai.fetch_policy import parse_no_proxy

    patterns = list(policy.local_only_patterns)
    if policy.inherit_no_proxy:
        no_proxy = os.environ.get("NO_PROXY") or os.environ.get("no_proxy")
        if no_proxy:
            inherited = parse_no_proxy(no_proxy)
            seen = set(patterns)
            for p in inherited:
                if p not in seen:
                    patterns.append(p)
                    seen.add(p)
    return patterns
```

**Step 2: Run existing fetch tests**

Run: `uv run pytest packages/markitai/tests/unit/test_fetch.py -v`
Expected: All PASS (no behavioral change when new fields are at defaults)

---

## Task 6: Update JSON schema and schema sync tests

**Files:**
- Modify: `packages/markitai/src/markitai/config.schema.json`
- Modify: `packages/markitai/tests/unit/test_schema_sync.py`

**Step 1: Add new fields to config.schema.json**

In `FetchPolicyConfig` definition (after `max_strategy_hops`):

```json
"strategy_priority": {
  "anyOf": [
    {
      "items": {
        "enum": ["static", "defuddle", "playwright", "jina", "cloudflare"],
        "type": "string"
      },
      "type": "array"
    },
    {
      "type": "null"
    }
  ],
  "default": null,
  "description": "Custom global strategy order. Overrides the default priority.",
  "title": "Strategy Priority"
},
"local_only_patterns": {
  "default": [],
  "description": "Domain/IP patterns that must use local-only strategies (NO_PROXY syntax).",
  "items": {
    "type": "string"
  },
  "title": "Local Only Patterns",
  "type": "array"
},
"inherit_no_proxy": {
  "default": true,
  "description": "Merge NO_PROXY env var into local_only_patterns at runtime.",
  "title": "Inherit No Proxy",
  "type": "boolean"
}
```

In `DomainProfileConfig` definition (after `prefer_strategy`):

```json
"strategy_priority": {
  "anyOf": [
    {
      "items": {
        "enum": ["static", "defuddle", "playwright", "jina", "cloudflare"],
        "type": "string"
      },
      "type": "array"
    },
    {
      "type": "null"
    }
  ],
  "default": null,
  "description": "Custom strategy order for this domain. Overrides global and prefer_strategy.",
  "title": "Strategy Priority"
}
```

**Step 2: Add schema sync test**

Add to `test_schema_sync.py`:

```python
def test_fetch_policy_config_new_fields(self, schema: dict) -> None:
    """Verify FetchPolicyConfig new fields in schema."""
    props = schema["$defs"]["FetchPolicyConfig"]["properties"]
    assert "strategy_priority" in props
    assert props["strategy_priority"]["default"] is None
    assert "local_only_patterns" in props
    assert props["local_only_patterns"]["default"] == []
    assert "inherit_no_proxy" in props
    assert props["inherit_no_proxy"]["default"] is True

def test_domain_profile_strategy_priority_in_schema(self, schema: dict) -> None:
    """Verify DomainProfileConfig.strategy_priority in schema."""
    props = schema["$defs"]["DomainProfileConfig"]["properties"]
    assert "strategy_priority" in props
    assert props["strategy_priority"]["default"] is None
```

**Step 3: Run schema sync tests**

Run: `uv run pytest packages/markitai/tests/unit/test_schema_sync.py -v`
Expected: All PASS

---

## Task 7: Full CI verification

**Step 1: Run full test suite**

Run: `uv run pytest`
Expected: All tests PASS, no regressions

**Step 2: Run linting and type checking**

Run: `uv run ruff check packages/markitai/src packages/markitai/tests && uv run ruff format --check packages/markitai/src packages/markitai/tests && uv run pyright`
Expected: All checks PASS

**Step 3: Commit**

```bash
git add packages/markitai/src/markitai/constants.py \
       packages/markitai/src/markitai/config.py \
       packages/markitai/src/markitai/config.schema.json \
       packages/markitai/src/markitai/fetch_policy.py \
       packages/markitai/src/markitai/fetch.py \
       packages/markitai/tests/unit/test_fetch_policy.py \
       packages/markitai/tests/unit/test_config.py \
       packages/markitai/tests/unit/test_schema_sync.py \
       docs/plans/2026-03-09-fetch-strategy-priority-and-exemption.md
git commit -m "feat: configurable fetch strategy priority and local-only domain exemption

Add support for customizing URL fetch strategy priority order (globally
and per-domain) and domain/IP exemptions that restrict fetching to
local-only strategies (static, playwright) for information security.

New config fields:
- fetch.policy.strategy_priority: global strategy order override
- fetch.policy.local_only_patterns: NO_PROXY-style exemption patterns
- fetch.policy.inherit_no_proxy: auto-merge NO_PROXY env var (default: true)
- fetch.domain_profiles.*.strategy_priority: per-domain order override

Priority chain: explicit > local_only > private/local > domain_priority
> domain_prefer > global_priority > SPA detection > default

Supported exemption patterns: domain exact, suffix (.corp.com),
wildcard (*.corp.com), IP exact, CIDR (10.0.0.0/8), localhost, star (*)"
```
