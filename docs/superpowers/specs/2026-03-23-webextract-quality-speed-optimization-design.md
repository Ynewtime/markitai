# WebExtract Quality & Speed Optimization Design

> Full-spectrum quality and performance optimization for markitai's web content extraction pipeline, porting key algorithms from defuddle (TypeScript) to native Python.

## Context

### Problem

markitai's `--playwright` mode produces significantly worse output than the defuddle-based strategy for sites like X/Twitter:

- **Quality**: Playwright output leaks page chrome (login prompts, sidebar, action buttons, error messages) while defuddle returns clean markdown with proper metadata
- **Speed**: Playwright takes 12+ seconds per page (3s blind wait + 5.6s auto-scroll + browser overhead) vs defuddle's ~1-2s

### Root Causes Identified

1. **Markdown conversion engine gap**: MarkItDown (wrapping markdownify) has near-zero custom rules vs defuddle's Turndown with 15+ custom rules (math, footnotes, code block language detection, complex tables, callouts, highlights, strikethrough, etc.)
2. **Missing content pattern removal**: markitai has 3 pattern types vs defuddle's 9
3. **No FxTwitter API integration**: defuddle has 3-layer Twitter extraction (DOM → Article → API fallback); markitai has only DOM parsing
4. **No Playwright domain profiles**: No site-specific wait strategies, no resource blocking, no auto-scroll skip
5. **Pipeline performance issues**: Up to 5x redundant HTML parsing on retries, 5-10+ DOM traversals per extraction, zero caching
6. **Missing modern web handling**: No CSS @media mobile style analysis, no React SSR streaming resolution

### Approach

Pure Python port (Approach B) — extend MarkItDown's `_CustomMarkdownify` with defuddle's conversion rules, port content extraction algorithms to Python, optimize pipeline performance. No Node.js runtime dependency.

### ADR: Markdown Converter Integration Strategy

MarkItDown internally instantiates `_CustomMarkdownify` inside `HtmlConverter.convert()`. The converter class is not configurable from outside — you cannot pass a custom class through `convert_stream()`.

**Chosen approach**: Register a custom `HtmlConverter` via MarkItDown's plugin entry-point system. This custom converter instantiates our `WebExtractMarkdownConverter` (subclass of `_CustomMarkdownify`) instead of the built-in one.

```
pyproject.toml entry point:
  [project.entry-points."markitdown.plugin"]
  markitai-webextract = "markitai.converter.webextract_plugin"

Module markitai/converter/webextract_plugin.py:
  def register_converters(markitdown_instance, **kwargs):
      markitdown_instance.register_converter(WebExtractHtmlConverter(), priority=-1)

Converter chain:
  MarkItDown(enable_plugins=True).convert_stream(.html)
    → WebExtractHtmlConverter.convert()  (priority -1, runs before built-in)
      → WebExtractMarkdownConverter(**kwargs).convert_soup(body)
```

The webextract pipeline creates `MarkItDown(enable_plugins=True)` to activate the custom converter. Non-webextract paths (PDF, DOCX, etc.) continue to use the built-in `HtmlConverter` unchanged.

**Image handling ownership**: Srcset resolution, `data-src` lazy loading, and base64 placeholder detection are handled in `standardize_content()` (Module 5.1) on the parsed DOM — NOT in the markdown converter's `convert_img()`. The converter only handles the final `<img>` → `![alt](src)` rendering. This avoids triple-processing.

## Module 1: Markdown Conversion Engine Enhancement

Extend `_CustomMarkdownify` (which inherits from `markdownify.MarkdownConverter`) with new converter methods, porting defuddle's 15+ Turndown rules.

### New converter

Create `WebExtractMarkdownConverter` extending `_CustomMarkdownify` in a new `markitai/converter/webextract_html_converter.py`. Create `WebExtractHtmlConverter` extending `HtmlConverter` to instantiate it.

### P0 — Format Fidelity Core

| Rule | Source | Implementation |
|------|--------|----------------|
| Code block language detection | `defuddle/src/elements/code.ts` | `convert_pre()` — Extract language from `data-lang`, `class="language-*"`, Prism/Highlight.js/SyntaxHighlighter patterns (120+ languages) |
| Table enhancement | `defuddle/src/markdown.ts` tables rule | `convert_table()` — Detect colspan/rowspan (preserve raw HTML), layout table extraction, ArXiv equation tables |
| Image srcset smart selection | `defuddle/src/elements/images.ts` | `convert_img()` — Handle CDN URLs with commas in srcset, `data-src` lazy loading, base64 placeholder replacement |
| List nesting & task lists | `defuddle/src/markdown.ts` lists rule | `convert_li()` — Tab indentation, `[x]`/`[ ]` checkbox, custom `start` numbering |

### P1 — Content Richness

| Rule | Source | Implementation |
|------|--------|----------------|
| Math formulas | `defuddle/src/elements/math.base.ts` | `convert_math()` — MathML alttext, KaTeX annotation, MathJax, WordPress LaTeX → `$...$`/`$$...$$` |
| Footnotes | `defuddle/src/elements/footnotes.ts` | `convert_sup()` + post-processing — 8 formats (Wikipedia, arXiv, Substack, Nature, Science.org, Tufte CSS sidenotes, etc.) → `[^N]` |
| Callouts/admonitions | `defuddle/src/markdown.ts` callouts rule | `convert_blockquote()` — GitHub alerts, Bootstrap alerts → `> [!note]` |
| Highlights/strikethrough | `defuddle/src/markdown.ts` | `convert_mark()` → `==text==`, `convert_del()` → `~~text~~` |

### P2 — Edge Cases

| Rule | Source | Implementation |
|------|--------|----------------|
| Embedded content | `defuddle/src/markdown.ts` embeds rule | YouTube/Twitter iframe → `![](url)` |
| Complex links | `defuddle/src/markdown.ts` complex links rule | `convert_a()` — Links containing heading children |
| ArXiv enumerations | `defuddle/src/markdown.ts` | Paper list special handling |

### Post-processing Enhancement

- Remove empty links `[](url)` (preserve `![](url)`)
- Fix `!` sticking to image syntax
- Collapse 3+ blank lines to 2
- Append collected footnote definitions at document end

## Module 2: Content Extraction & Noise Removal Enhancement

### 2.1 Content Pattern Removal (6 new patterns)

Port to `webextract/removals/content_patterns.py`:

| Pattern | Logic | Priority |
|---------|-------|----------|
| Hero header removal | H1/H2 + `<time>` within first 300 chars, block < 30 words → remove | P0 |
| Trailing thin section removal | Scan backward from end, each block < 25 words with heading, no substantive content → remove | P0 |
| Boundary date element removal | `<time>` within first/last 200 chars, ≤ 10 words, not inline nested → remove | P1 |
| Metadata list removal | ul/ol/dl within first/last 500 chars, 1-8 items, each < 8 words, total < 30 words → remove | P1 |
| Breadcrumb navigation removal | ≤ 10 words leaf element, link points to parent path or index.html → remove | P1 |
| Trailing external link list removal | "See Also" heading + all-external-link list at end → remove | P2 |

Also enhance existing boilerplate truncation: cascade-delete all following siblings after match (align with defuddle behavior).

**Note**: All numeric thresholds above (300 chars, 30 words, 25 words, 200 chars, 500 chars, 8 words, 10 words) are taken directly from defuddle's `content-patterns.ts` source code to ensure parity.

### 2.2 CSS @media Mobile Style Analysis

New module `webextract/removals/mobile_styles.py`.

**CSS parsing dependency**: Use `tinycss2` (maintained, standards-compliant, lightweight) to parse CSS from `<style>` tags. Pipeline:

1. Extract `<style>` tag text content
2. Parse with `tinycss2.parse_stylesheet()`
3. Filter `@media` rules where the `max-width` breakpoint value is ≤ 768px (i.e., rules targeting mobile/tablet viewports)
4. Within those rules, find selectors whose declarations include `display: none`
5. Apply those selectors via BeautifulSoup's `.select()` to remove elements

Purpose: Suppress sidebars/navbars hidden on mobile that inflate content scoring noise.

**New dependency**: Add `tinycss2` to `pyproject.toml` dependencies.

### 2.3 React SSR Streaming Resolution

Add to `webextract/preprocess.py` (raw HTML string-level operations, runs before BeautifulSoup parsing — consistent with existing `_flatten_declarative_shadow_dom()` and `_remove_wbr_tags()` in the same module):

- Detect `$RC("B:X","S:X")` inline script calls in HTML
- Replace `<!--$?-->...<template id="B:X">` placeholders with actual content from `<div hidden id="S:X">`
- Significant quality improvement for Next.js / Remix rendered pages

### 2.4 Playwright DOM Cleanup Enhancement

Extend `constants.py` with site-specific noise selectors:

```python
SITE_NOISE_SELECTORS: dict[str, tuple[str, ...]] = {
    "x.com": (
        '[data-testid="sidebarColumn"]',
        '[data-testid="DMDrawer"]',
        '[data-testid="sheetDialog"]',
        '[data-testid="bottomBar"]',
        '[data-testid="placementTracking"]',
        '[aria-label="Sign up"]',
        '[aria-label="Footer"]',
    ),
    "twitter.com": ...,  # same as x.com
}
```

Inject into `_build_dom_cleanup_script()` based on URL domain.

## Module 3: Site-Specific Extraction Enhancement

### 3.1 Twitter/X FxTwitter API Integration

New file `fetch_fxtwitter.py`:

**Strategy position**: Higher priority than playwright, lower than defuddle.

```
x.com/twitter.com + /status/\d+ strategy order:
defuddle → fxtwitter (new) → playwright → static
```

**Implementation**:
- `GET https://api.fxtwitter.com/{user}/status/{id}`
- Parse JSON: full tweet text, media URLs, quoted tweets, author info, timestamps
- For X Articles (`tweet.article`): parse Draft.js block structure (unstyled/header/atomic/list-item)
- For regular tweets: parse text + facets (mention/url/media rich text markers)
- Convert to markitai's `ConversationThread` semantic structure, reuse `render_semantic_content()`

**Fallback chain**:
- FxTwitter fails → Twitter oembed API (`publish.twitter.com/oembed?url=...&omit_script=true`)
- oembed returns blockquote HTML (may truncate long tweets, but still better than noisy playwright)

**Operational concerns**:
- FxTwitter is a community service (not official Twitter API) — treat as best-effort, with 10s timeout
- Rate limiting: undocumented, use conservative 20 RPM (same as defuddle strategy)
- Cache successful responses in fetch cache (strategy key = "fxtwitter")
- If FxTwitter is down, fall through to next strategy silently (no retry)

### 3.2 XTweetExtractor Quality Gate Fix

Fix `_assess_social_post()` in `quality.py`:

Current `_QUOTE_CARD_PATTERN` (`^Quote$`) false-positive rejects tweets that quote other tweets. Fix: only penalize when `Quote` appears independently followed by non-tweet content (recommendation lists), not when followed by `@handle` tweet content.

### 3.3 YouTube Transcript Enhancement

Enhance `webextract/extractors/youtube_page.py`:

- **InnerTube API caption fetching**: POST `youtube.com/youtubei/v1/player`, get `captionTracks`, download caption XML
- **Caption grouping algorithm**: Group by speaker (`>>`/`- ` markers) and sentence boundaries, merge short sentences (< 80 words, < 45s span)
- **Chapter extraction**: From player overlay markersMap or engagement panel
- **SSRF protection**: Validate caption URL hostname ends with `.youtube.com`

### 3.4 Other Site Extractor Alignment

| Extractor | Enhancement |
|-----------|-------------|
| Reddit | Add old.reddit.com fallback (new reddit lacks server-side comment rendering) |
| GitHub | Code block language extraction (`highlight-source-*` class), prefer `data-snippet-clipboard-copy-content` |
| HackerNews | Comment page mode (`.onstory` without `.titleline`) |

## Module 4: Playwright Speed Optimization

### 4.1 Built-in Domain Profiles

Extend the existing `DomainProfileConfig` class (at `config.py`) with new fields `skip_auto_scroll: bool = False` and `reject_resource_patterns: list[str] | None = None`. Add built-in profiles that serve as defaults; user-configured profiles in TOML take precedence and merge with built-ins.

**Twitter/X**:
```python
"x.com": DomainProfileConfig(
    wait_for_selector='[data-testid="tweet"]',
    wait_for="domcontentloaded",
    extra_wait_ms=500,           # down from 3000ms
    skip_auto_scroll=True,       # new field
    reject_resource_patterns=["**/analytics/**", "**/ads/**", "**/tracking/**", "**/*.mp4"],
)
```

**GitHub**:
```python
"github.com": DomainProfileConfig(
    wait_for_selector='.markdown-body',
    wait_for="domcontentloaded",
    extra_wait_ms=300,
    skip_auto_scroll=True,
)
```

### 4.2 skip_auto_scroll Support

Add `skip_auto_scroll` field to `DomainProfile` and `PlaywrightRenderer.fetch()`:
- When `True`, skip `_build_auto_scroll_script()` and `POST_SCROLL_DELAY_MS`
- Saves up to 5.6 seconds for single-content pages (tweets, issues, docs)

### 4.3 Smart Wait Strategy

Modify wait logic in `PlaywrightRenderer.fetch()`:
- When `wait_for_selector` is provided: wait for it, then use `extra_wait_ms` as short stabilization buffer (not either/or)
- When no `wait_for_selector`: fall back to `extra_wait_ms` as before

### 4.4 Resource Pattern Propagation

Propagate `reject_resource_patterns` from domain profiles through `_resolve_playwright_profile_overrides()`.

### 4.5 Expected Speed Improvement (Twitter)

| Phase | Before | After | Saved |
|-------|--------|-------|-------|
| extra_wait_ms | 3000ms | 500ms | 2500ms |
| auto-scroll | ~5600ms | skipped | 5600ms |
| resource loading | full | filtered | ~1000ms |
| **Total** | **~12s** | **~3-4s** | **~8-9s** |

## Module 5: Content Extraction Pipeline Enhancement

### 5.1 HTML Standardization Additions

Add to `webextract/standardize.py`:

| Step | Detail |
|------|--------|
| Code block standardization | Unify Prism/Highlight.js/SyntaxHighlighter structures to `<pre><code data-lang="x">` |
| Footnote standardization | 8 formats (Wikipedia `cite_note`, arXiv `ltx_biblist`, Substack `FootnoteToDOM`, Tufte CSS sidenotes, etc.) |
| Image standardization | `<picture>` → `<img>` merge, `data-src` lazy load promotion, base64 placeholder detection |
| Heading standardization | Remove permalink anchors (`<a class="anchor" href="#...">`) |
| Callout standardization | GitHub `[!NOTE]` blockquote, Bootstrap `.alert-*`, Obsidian `data-callout` → unified `<div data-callout="type">` |

### 5.2 Content Scoring Enhancement

Add to `webextract/scoring.py`:

**New positive factors**: comma count (+1 each), date detection (+10), author detection (+10), footnote indicators (+10 each), right-aligned bonus (+5)

**New negative factors**: social profile link detection (-15 for < 80 words + social links), enhanced card grid detection (-15)

**New penalty**: nested table penalty (-5 per nested table)

### 5.3 Retry Strategy Alignment

Port defuddle's exact retry acceptance heuristics (from `defuddle.ts` lines 74-139):

**New retry Level A — Hidden content subtree (when < 50 words)**:
- Find largest hidden element (by word count) in the original DOM
- Accept if: `hidden_words > current_words` OR (`hidden_words > max(20, current_words * 0.7)` AND `hidden_content.length < current_content.length`) — i.e., more focused content even with slightly fewer words
- Use that element's parent as content selector for re-extraction

**New retry Level B — Index page mode (when still < 50 words)**:
- Disable scoring removal, pattern removal, AND content pattern removal (not just scoring)
- This handles card-style index pages where content is distributed across many small blocks

### 5.4 DOM Parser Selection

- Playwright-rendered HTML: prefer `html.parser` (avoids lxml parsing differences on dynamic HTML)
- Static-fetched HTML: keep lxml priority (speed advantage)

## Module 6: TDD Strategy — Reuse defuddle Test Suite

### 6.1 defuddle Test Assets

defuddle has 83 HTML fixtures + 85 expected markdown files across 11 categories:

- general (25): Real-world pages (Wikipedia, GitHub, HN, Substack, X Article, etc.)
- issues (20): GitHub issue-specific edge cases
- elements (11): HTML element handling
- footnotes (8): Footnote/endnote extraction
- math (6): KaTeX, MathJax, MathML, Temml
- codeblocks (4): Various syntax highlighters
- table-layout (2), scoring (2), hidden (2), comments (2), extractor (1)

### 6.2 Test Structure

Use a **pinned-version copy script** (`scripts/sync_defuddle_fixtures.sh`) that copies fixtures from the local defuddle repo at a specific git tag/commit, recording the source version. This avoids git submodule complexity while keeping fixtures traceable. Re-run the script when defuddle updates.

```
tests/
├── defuddle_fixtures/           # Copied from defuddle at pinned version
│   ├── VERSION                  # Records defuddle commit hash
│   ├── fixtures/                # HTML input files
│   └── expected/                # Expected markdown files
└── test_defuddle_parity.py      # Parameterized parity tests
```

### 6.3 Parity Test Design

```python
@pytest.mark.parametrize("fixture", ALL_DEFUDDLE_FIXTURES)
def test_parity_with_defuddle(fixture):
    html = read_fixture(fixture.html_path)
    expected = read_expected(fixture.expected_path)

    result = extract_web_content(html, fixture.url)

    assert_metadata_match(result, expected)
    assert_content_coverage(result, expected)
    assert_no_noise(result, expected)
    assert_word_count_close(result, expected, tolerance=0.10)
```

### 6.4 Graduated Quality Levels

- **Level 1 (P0)**: Correct metadata + core content present + no severe noise leakage
- **Level 2 (P1)**: Markdown format fidelity (code block languages, footnote numbers, math formula syntax)
- **Level 3 (P2)**: Line-by-line match with defuddle output (long-term goal)

### 6.5 TDD Cadence Per Module

For each ported defuddle feature:

1. Select relevant fixtures from defuddle test suite
2. Write failing test asserting expected output
3. Implement feature to make test pass
4. Regression check: ensure other fixtures don't regress

### 6.6 Performance Benchmark Tests

```python
@pytest.mark.benchmark
@pytest.mark.parametrize("fixture", ALL_DEFUDDLE_FIXTURES)
def test_extraction_performance(fixture, benchmark):
    html = read_fixture(fixture.html_path)
    result = benchmark(extract_web_content, html, fixture.url)
    assert benchmark.stats["mean"] < 0.2  # < 200ms per page
```

## Module 7: Pipeline Performance Optimization

### 7.1 Problem: Why markitai is Slow

| Issue | Severity | Detail |
|-------|----------|--------|
| Redundant HTML parsing | Critical | Retry mechanism re-parses HTML from scratch — up to 5 times per extraction |
| Excessive DOM traversals | Critical | 5-10+ `find_all()` calls per extraction, multiple stages re-traverse |
| Zero caching | Severe | Metadata, scoring stats, selector results all recomputed every time |
| Markdown preprocessing extra parse | Severe | `_preprocess_for_markdown()` creates separate BeautifulSoup instance |
| Per-selector querying | Moderate | 109 exact selectors queried one-by-one instead of joined |

### 7.2 defuddle Performance Techniques to Port

| Technique | defuddle | markitai current |
|-----------|----------|------------------|
| Single clone | `doc.cloneNode(true)` at start, all ops on clone | Re-parse HTML on every retry |
| Cross-retry caching | Schema, metadata, mobile styles, small images all cached | No caching, full recompute |
| Selector pre-compilation | `EXACT_SELECTORS_JOINED` — one CSS query for all | Per-selector `root.select()` |
| Partial selector regex | `PARTIAL_SELECTORS_REGEX` — one combined regex | Per-pattern substring match |
| Batch collect-then-remove | Collect into Set/Map, single removal pass | Remove during traversal |
| Single-pass scoring | `scoreElement()` O(n) linear, no nested loops | `find_all("a")` called twice on same element |
| Selective computedStyle | Skip `getComputedStyle` in Node.js (slow), prefer inline style regex | No environment distinction |
| Early returns | Guard clauses for empty docs, low word count | Few guard clauses |

### 7.3 Optimization Plan

**P0: Eliminate redundant parsing (expected 3-5x speedup)**

```python
class _ExtractionContext:
    """Cache expensive computations across retry levels."""
    def __init__(self, html: str, url: str):
        self.original_soup = parse_html(html)          # parse ONCE
        self.metadata = extract_metadata(self.original_soup, url)  # cache
        self._small_images: set | None = None          # lazy, cached

    def fresh_root(self) -> Tag:
        return copy.deepcopy(self.original_soup)       # deepcopy >> re-parse
```

**P0: Join selectors into single query**

```python
# Before: 109 individual queries
for selector in EXACT_SELECTORS:
    for el in root.select(selector): el.decompose()

# After: 1 query
EXACT_SELECTORS_JOINED = ", ".join(EXACT_SELECTORS)
for el in root.select(EXACT_SELECTORS_JOINED): el.decompose()
```

**P1: Single-pass collection + batch removal**

```python
def _collect_removals(root, config) -> set[Tag]:
    removals = set()
    for el in root.find_all(True):  # ONE traversal
        if _is_small_image(el, cached_small_images): removals.add(el)
        if _is_hidden(el): removals.add(el)
        if _matches_partial_selector(el, PARTIAL_REGEX): removals.add(el)
    return removals

for el in removals: el.decompose()  # ONE removal pass
```

**P1: Merge markdown preprocessing into standardization**

Move srcset resolution, iframe canonicalization, figcaption handling from `_preprocess_for_markdown()` (creates extra BeautifulSoup) into `standardize_content()` (already has parsed DOM).

**P2: Candidate stats memoization**

Cache `_CandidateStats` per element ID to avoid duplicate `find_all()` calls on the same element.

### 7.4 Expected Combined Speedup

**Important**: The multipliers below are estimates. Phase 1 implementation MUST begin with profiling the current pipeline on representative fixtures to validate these estimates and prioritize accordingly. `copy.deepcopy` on large BeautifulSoup trees may be slower than expected due to recursive parent/sibling back-references — if profiling shows this, consider alternative cloning strategies (re-parse from cached HTML string, or selective tree reconstruction).

| Optimization | Estimated Speedup | Reason |
|-------------|---------|--------|
| Eliminate redundant parsing | 2-4x | 5 parses → 1 parse + deepcopy (pending profiling) |
| Join selectors | 1.5-3x | 109+ queries → 1 query |
| Single-pass collection | 1.3-2x | 5-10 find_all → 1-2 |
| Merge markdown preprocessing | 1.1-1.3x | Eliminate extra BeautifulSoup instance |
| **Combined** | **~3-8x** | Target: per-page extraction < 200ms |

## Implementation Priority

| Phase | Modules | Impact |
|-------|---------|--------|
| **Phase 1** | TDD setup (Module 6) + Performance (Module 7) | Foundation: tests + fast feedback loop |
| **Phase 2** | Markdown engine (Module 1 P0) + Playwright speed (Module 4) | Immediate visible improvement |
| **Phase 3** | Noise removal (Module 2) + FxTwitter API (Module 3.1) | Twitter/X quality leap |
| **Phase 4** | Markdown engine (Module 1 P1-P2) + Pipeline (Module 5) + remaining Module 3 | Full quality alignment |
