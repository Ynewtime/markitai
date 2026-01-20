# Markit Performance Optimization Phase 3 - Implementation Spec

> **Note**: This document serves as a detailed implementation reference. When context is compacted, refer to this spec for implementation details.

## Overview

Based on `markit-perf-optimization-3.md` analysis, this spec defines three optimization phases:

| Phase | Optimization | Priority | Effort |
|-------|--------------|----------|--------|
| 1 | SQLite Dual-Layer Cache | P0 | 1.5 days |
| 2 | Windows COM Conversion | P1 | 0.5 days |
| 3 | Unified Model Group + Smart Vision Routing | P2 | 0.5 days |

---

## Phase 1: SQLite Dual-Layer Persistent Cache

### Goal
Eliminate redundant LLM API calls across sessions, enabling instant re-runs for debugging and incremental processing.

### Storage Strategy

```
Cache Lookup: Project Cache → Global Cache → LLM API
Cache Write:  Project Cache + Global Cache (write to both)

~/.markit/cache.db         ← Global cache (shared across projects)
.markit/cache.db           ← Project cache (project-specific)
```

- **Size Limit**: 1GB per cache file
- **Eviction**: LRU-based when size limit reached
- **Key Strategy**: `SHA256(prompt + content)[:32]`

### SQLite Schema

```sql
CREATE TABLE IF NOT EXISTS cache (
    key TEXT PRIMARY KEY,           -- SHA256 hash
    value TEXT NOT NULL,            -- JSON serialized response
    model TEXT,                     -- Model identifier for invalidation
    created_at INTEGER NOT NULL,    -- Unix timestamp
    accessed_at INTEGER NOT NULL,   -- LRU tracking
    size_bytes INTEGER NOT NULL     -- For capacity management
);

CREATE INDEX IF NOT EXISTS idx_accessed ON cache(accessed_at);
CREATE INDEX IF NOT EXISTS idx_created ON cache(created_at);
```

### File Changes

| File | Changes |
|------|---------|
| `constants.py` | Add `DEFAULT_CACHE_SIZE_LIMIT = 1GB`, `DEFAULT_GLOBAL_CACHE_DIR`, `DEFAULT_PROJECT_CACHE_DIR` |
| `llm.py` | Add `SQLiteCache` class, refactor `ContentCache` to support dual-layer |
| `cli.py` | Add `markit cache` subcommand (`clear`, `stats`) |
| `config.py` | Add `cache` configuration block |

### SQLiteCache Class Design

```python
class SQLiteCache:
    """SQLite-based persistent LRU cache with size limit."""
    
    def __init__(
        self,
        db_path: Path,
        max_size_bytes: int = 1 * 1024 * 1024 * 1024,  # 1GB
    ) -> None: ...
    
    def get(self, key: str) -> str | None:
        """Get cached value, update accessed_at for LRU."""
        ...
    
    def set(self, key: str, value: str, model: str = "") -> None:
        """Set cache value, evict LRU entries if size exceeded."""
        ...
    
    def clear(self) -> int:
        """Clear all entries, return count deleted."""
        ...
    
    def stats(self) -> dict:
        """Return cache statistics (count, size, hit rate)."""
        ...
    
    def _evict_lru(self, bytes_needed: int) -> None:
        """Evict least recently used entries to free space."""
        ...
```

### DualLayerCache Class Design

```python
class DualLayerCache:
    """Dual-layer cache: project-level + global-level."""
    
    def __init__(
        self,
        project_path: Path | None = None,
        global_path: Path | None = None,
        max_size_bytes: int = 1 * 1024 * 1024 * 1024,
    ) -> None:
        self._project_cache = SQLiteCache(project_path, max_size_bytes) if project_path else None
        self._global_cache = SQLiteCache(global_path, max_size_bytes) if global_path else None
    
    def get(self, prompt: str, content: str) -> Any | None:
        """Lookup: project → global → None"""
        key = self._compute_hash(prompt, content)
        # Try project cache first
        if self._project_cache:
            result = self._project_cache.get(key)
            if result:
                return json.loads(result)
        # Fallback to global cache
        if self._global_cache:
            result = self._global_cache.get(key)
            if result:
                return json.loads(result)
        return None
    
    def set(self, prompt: str, content: str, result: Any, model: str = "") -> None:
        """Write to both caches."""
        key = self._compute_hash(prompt, content)
        value = json.dumps(result)
        if self._project_cache:
            self._project_cache.set(key, value, model)
        if self._global_cache:
            self._global_cache.set(key, value, model)
```

### CLI Commands

```bash
# Show cache statistics
markit cache stats

# Clear project cache
markit cache clear

# Clear global cache
markit cache clear --global

# Clear all caches
markit cache clear --all
```

---

## Phase 2: Windows COM Conversion

### Goal
Leverage native MS Office on Windows for faster .doc/.ppt conversion with automatic fallback to LibreOffice.

### Source Code
Port from `docs/reference/convert_to_markdown.py`:
- `check_ms_office_available()` (lines 904-920)
- `check_ms_word_available()` (lines 923-938)
- `convert_with_ms_office()` (lines 941-983)
- `convert_doc_with_ms_word()` (lines 986-1028)

### Conversion Priority
```
MS Office COM (Windows) → LibreOffice CLI → Error with manual conversion hint
```

### Concurrency Model
- PowerShell subprocess provides process isolation
- No global mutex needed (unlike pywin32 COM)
- Safe for concurrent execution via ThreadPoolExecutor

### File Changes

| File | Changes |
|------|---------|
| `utils/office.py` | Add `check_ms_word_available()`, `check_ms_office_available()` |
| `legacy.py` | Add `_convert_with_ms_office()`, `_convert_doc_with_ms_word()`, update `convert()` |

### PowerShell Script Template (PPT)

```powershell
$ppt = New-Object -ComObject PowerPoint.Application
$ppt.Visible = [Microsoft.Office.Core.MsoTriState]::msoFalse
try {
    $presentation = $ppt.Presentations.Open('{input_path}', $true, $false, $false)
    $presentation.SaveAs('{output_path}', 24)  # 24 = ppSaveAsOpenXMLPresentation
    $presentation.Close()
    Write-Host "SUCCESS"
} catch {
    Write-Host "FAILED: $_"
} finally {
    $ppt.Quit()
    [System.Runtime.Interopservices.Marshal]::ReleaseComObject($ppt) | Out-Null
}
```

---

## Phase 3: Unified Model Group + Smart Vision Routing

### Goal
Simplify model configuration by using a single `default` group with automatic vision capability detection.

### Architecture

```
All models → single "default" group
├── Each model configured with model_info.supports_vision
├── Code auto-detects if message contains images
└── Auto-filters to vision-capable models when needed
```

### Key Insight
LiteLLM's `model_info.supports_vision` is **metadata only** - it doesn't affect routing. We implement smart routing in code.

### Code Changes (llm.py)

#### 1. Image Detection Helper

```python
def _message_contains_image(self, messages: list[dict]) -> bool:
    """Detect if messages contain image content."""
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    return True
    return False
```

#### 2. Vision Router Property

```python
@property
def vision_router(self) -> Router:
    """Get or create Router with only vision-capable models (lazy)."""
    if self._vision_router is None:
        vision_models = [
            m for m in self.config.model_list
            if m.model_info and m.model_info.supports_vision
        ]
        if not vision_models:
            raise ValueError("No vision-capable models configured")
        self._vision_router = self._create_router_from_models(vision_models)
    return self._vision_router
```

#### 3. Smart Router Selection

```python
async def _call_llm(self, model: str, messages: list[dict], context: str = "") -> LLMResponse:
    requires_vision = self._message_contains_image(messages)
    router = self.vision_router if requires_vision else self.router
    # Use selected router for completion
    ...
```

#### 4. Remove model="vision" References

Replace all `model="vision"` calls with `model="default"`:
- `analyze_image()` 
- `enhance_document_with_vision()`
- `enhance_document_complete()`
- `_analyze_image_with_fallback()`

### Configuration Changes (markit.json)

**Before**:
```json
{
  "model_list": [
    { "model_name": "default", ... },
    { "model_name": "vision", ..., "model_info": { "supports_vision": true } }
  ],
  "fallbacks": [{ "default": ["vision"] }]
}
```

**After**:
```json
{
  "model_list": [
    { 
      "model_name": "default", 
      "litellm_params": { "model": "gemini/gemini-2.5-flash-lite", "weight": 10 },
      "model_info": { "supports_vision": true }
    },
    { 
      "model_name": "default", 
      "litellm_params": { "model": "deepseek/deepseek-chat", "weight": 8 }
    }
  ],
  "router_settings": {
    "routing_strategy": "latency-based-routing",
    "enable_pre_call_checks": true
  }
}
```

---

## Task Checklist

### Phase 1: Cache
- [x] Add cache constants to `constants.py`
- [x] Implement `SQLiteCache` class in `llm.py`
- [x] Implement `DualLayerCache` class in `llm.py` (named `PersistentCache`)
- [x] Refactor `ContentCache` to use `DualLayerCache`
- [x] Add `cache` config block to `config.py`
- [x] Add `markit cache` CLI commands to `cli.py`
- [x] Write unit tests for cache
- [x] Write integration tests for cache

### Phase 2: COM
- [x] Add Office detection functions to `utils/office.py`
- [x] Port PowerShell COM conversion to `legacy.py`
- [x] Update `LegacyOfficeConverter.convert()` with fallback logic
- [ ] Test on Windows with MS Office installed

### Phase 3: Model
- [x] Implement `_message_contains_image()` in `llm.py`
- [x] Add `vision_router` property to `LLMProcessor`
- [x] Add `_create_router_from_models()` helper
- [x] Update `_call_llm()` with smart router selection
- [x] Remove `model="vision"` calls, use `model="default"`
- [x] Update `markit.json` with unified configuration
- [x] Update unit tests
- [x] Write integration tests for vision router

---

## Implementation Notes

### Cache Key Strategy
For different content types:
- **Image Analysis**: `SHA256(prompt + image_bytes)`
- **Document Cleaning**: `SHA256(prompt + content[:50000])` (truncate to avoid huge keys)
- **Vision Enhancement**: `SHA256(prompt + page_images_hash)`

### Backward Compatibility
- Keep `vision` model group working temporarily via fallback
- Deprecation warning when `model="vision"` is used directly

### Testing Strategy
1. Unit tests for `SQLiteCache` isolation
2. Integration tests for dual-layer lookup order
3. Manual testing on Windows for COM conversion
4. Regression tests for vision routing
