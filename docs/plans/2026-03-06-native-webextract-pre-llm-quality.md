# Native Web Extraction And Pre-LLM Markdown Quality Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace Defuddle-first Markdown cleanup with a native Python extraction and normalization pipeline that improves Markitai output before any LLM step, while keeping external fetch APIs as optional fallbacks.

**Architecture:** Add a new `markitai.webextract` package that parses raw HTML, extracts main content, standardizes structure, resolves metadata, and hands cleaned HTML to MarkItDown for Markdown rendering. Native webextract is a pre-MarkItDown HTML quality layer, not a replacement Markdown renderer. Reuse the fetch/router groundwork already present in the repo: `fetch.policy`, `fetch.domain_profiles`, Playwright session persistence, and `fetch_with_static_conditional()` as the main HTML/static ingress. Integrate native extraction inside the individual fetch strategies while they still hold raw HTML, then continue to use MarkItDown for HTML-to-Markdown rendering. Add a shared `utils.markdown_quality` pass used before writing base `.md` and reused by `DocumentMixin.format_llm_output()` so URL and file inputs share the same non-LLM cleanup rules.

**Tech Stack:** Python 3.11+, BeautifulSoup4 as the DOM API, `lxml` parser backend when available and `html.parser` fallback when not, pytest, markitdown, existing `security.py` atomic writes and frontmatter helpers.

**Execution Discipline:** `@test-driven-development` for every task, `@verification-before-completion` before claiming success. Do not call Defuddle at runtime or in tests; use copied fixtures only.

---

## Decision Summary

- Do not introduce `defuddle/node`.
- Do not port Defuddle 1:1.
- Borrow the high-value primitives:
  - adaptive retry when extraction is too short
  - content scoring and block pruning
  - schema.org text fallback
  - metadata extraction from JSON-LD, OG, Twitter, canonical/meta tags
  - exact selector removal plus conservative partial selector removal
  - heading/code/footnote/image standardization
  - site extractor registry pattern
- Defer the expensive/fragile primitives:
  - mobile CSS evaluation
  - full conversation extractor suite
  - exact fixture parity across the entire Defuddle corpus

## Borrowing Matrix

| Defuddle source | Markitai native target | What to borrow |
| --- | --- | --- |
| `defuddle/src/defuddle.ts` | `packages/markitai/src/markitai/webextract/pipeline.py` | Parse order, adaptive retry, schema fallback trigger, diagnostics |
| `defuddle/src/metadata.ts` | `packages/markitai/src/markitai/webextract/metadata.py` | Metadata merge order and title cleaning |
| `defuddle/src/scoring.ts` | `packages/markitai/src/markitai/webextract/scoring.py` | Candidate scoring heuristics and negative-score block pruning |
| `defuddle/src/standardize.ts` | `packages/markitai/src/markitai/webextract/standardize.py` | Wrapper flattening, heading cleanup, attr stripping, empty-node cleanup |
| `defuddle/src/elements/code.ts` | `packages/markitai/src/markitai/webextract/elements/code.py` | Pre/code normalization and language retention |
| `defuddle/src/elements/footnotes.ts` | `packages/markitai/src/markitai/webextract/elements/footnotes.py` | Canonical footnote list generation and backlink cleanup |
| `defuddle/src/elements/images.ts` | `packages/markitai/src/markitai/webextract/elements/images.py` | Lazy-image upgrade, figure/caption normalization |
| `defuddle/src/extractor-registry.ts` | `packages/markitai/src/markitai/webextract/extractors/registry.py` | Pattern-based site extractor selection |

## Planned Module Layout

```text
packages/markitai/src/markitai/webextract/
├── __init__.py
├── dom.py
├── types.py
├── constants.py
├── metadata.py
├── scoring.py
├── sanitize.py
├── schema.py
├── standardize.py
├── pipeline.py
└── extractors/
    ├── __init__.py
    ├── base.py
    ├── registry.py
    ├── github_issue.py
    └── x_article.py
```

## Result Contract

`ExtractedWebContent` should be the single native contract passed from HTML extraction to fetch/write stages.

```python
@dataclass(slots=True)
class WebMetadata:
    title: str | None = None
    author: str | None = None
    site: str | None = None
    published: str | None = None
    description: str | None = None
    canonical_url: str | None = None


@dataclass(slots=True)
class ExtractedWebContent:
    clean_html: str
    markdown: str
    metadata: WebMetadata
    word_count: int
    diagnostics: dict[str, Any] = field(default_factory=dict)
```

`clean_html` is the canonical native output. `markdown` must be derived from
`clean_html` through a single shared HTML-to-Markdown helper so tests, fetch
strategies, and future callers do not drift.

Diagnostics should include:

- `extractor`: `generic`, `github_issue`, `x_article`, etc.
- `schema_fallback_used`: `True/False`
- `adaptive_retry_used`: `True/False`
- `removed_partial_selectors`: `True/False`
- `candidate_count`: integer

## Non-Goals

- Do not add any Node.js runtime requirement.
- Do not remove external Defuddle/Jina/Cloudflare support in the same branch.
- Do not change LLM prompts or provider logic.
- Do not try to solve PDF/DOCX semantic reconstruction in the web extractor; file converters keep their current responsibilities.

## Success Metrics

- Base `.md` output for curated HTML fixtures keeps code blocks, headings, footnotes, and images materially better than current raw MarkItDown-on-full-page behavior.
- `write_base_markdown()` and URL base output both run the shared pre-LLM Markdown quality pass.
- `FetchPolicyEngine` no longer prefers external Defuddle for normal domains once native extraction is integrated and verified.
- No network tests are required for the new quality coverage.

## Quality Strategy

- Treat native webextract as an HTML main-content selector and normalizer whose job is to improve the HTML *before* Markdown rendering. Do not reimplement Markdown serialization logic that MarkItDown already owns.
- `extract_web_content()` should emit diagnostics that make fallback decisions explainable: extractor name, candidate count, schema fallback used, adaptive retry used, removed partial selectors, and a `fallback_reason` when native output is rejected.
- Add a quality gate before accepting native output. Reject or downgrade native output when it is clearly worse than the baseline, for example:
  - empty or near-empty Markdown
  - boilerplate-heavy output (very high link density / login or cookie text dominates)
  - title/body mismatch or no meaningful body text
  - code/image/footnote-bearing fixtures losing their key structure
- Prefer deterministic heuristics over subjective scoring. The goal is not "perfect extraction"; the goal is "better than whole-page MarkItDown often enough to become the default when quality gates pass."

## Fallback Policy

- Native extraction must fail closed. If `extract_web_content()` raises, returns empty output, or fails the quality gate, fall back to the current strategy-local Markdown path instead of propagating partial native output.
- Fallback should happen inside each fetch strategy while both inputs are still available:
  - Playwright: raw `page.content()` -> native webextract -> cleaned HTML -> MarkItDown -> fallback to current full-page `_html_to_markdown()` / `inner_text("body")` path if needed
  - Static HTML: raw HTTP HTML response -> native webextract -> cleaned HTML -> MarkItDown -> fallback to existing MarkItDown-on-full-response behavior if needed
- Record the outcome in metadata/diagnostics so golden tests and logs can tell whether native extraction or fallback won.
- Native metadata should be merged into the existing `metadata["source_frontmatter"]` contract. Do not introduce a second parallel metadata path for base `.md` writing.

## Evaluation Plan

- Measure native extraction against the current baseline, not against Defuddle alone.
- For each curated fixture, compare at least:
  - extracted title quality
  - body word count and duplicate-text rate
  - heading retention
  - code fence / language retention
  - footnote retention
  - image and caption retention
  - boilerplate leakage (nav, footer, cookie/login text)
- Keep two fixture views:
  - expected native golden output (`expected/*.md`)
  - optional notes recording where baseline MarkItDown is worse, so the test corpus explains *why* the fixture exists
- Do not flip fetch policy order until the native path wins on the curated corpus and fallback behavior is verified on failure scenarios.

## Why This Is Not Duplicate Work

- MarkItDown remains the Markdown renderer.
- Native webextract adds the missing capability MarkItDown does not provide here: choosing the right DOM subtree, normalizing noisy HTML, and extracting source metadata before rendering.
- A useful litmus test: if a function is mostly about DOM pruning, content scoring, schema fallback, metadata extraction, or HTML standardization, it belongs in `webextract`; if it is about HTML-to-Markdown rendering rules, it should stay in MarkItDown.

## Rollout Guardrails

- Copy a small curated fixture set from Defuddle into Markitai. Do not read `/home/y/dev/defuddle` during CI.
- Keep fallback behavior conservative: if native extraction fails, current MarkItDown fallback remains available.
- Only mock system boundaries: HTTP client, Playwright page object, filesystem temp file behavior.
- Preserve current CLI behavior and output paths.
- Reuse existing config surface. Do not add a second fetch policy layer or alternate domain-profile schema in this plan.
- **Rollback switch:** If native extraction causes quality regressions, the existing `FetchPolicyEngine` domain-prefer mechanism (`domain_profiles[domain].prefer_strategy`) can override per-domain. As a global escape hatch, setting `fetch.policy.enabled: false` reverts to the pre-native default strategy order (defuddle-first).

## Current Repo Baseline

- `FetchConfig.policy`, `FetchConfig.domain_profiles`, `DomainProfileConfig.prefer_strategy`, and `PlaywrightConfig.session_mode` already exist and should be reused rather than redesigned.
- `fetch_with_static_conditional()` is now the main static/HTML path for explicit `static` and many cacheable `auto` fetches. `fetch_with_static()` is still relevant, but it is no longer the only integration point.
- The URL router currently flows through `_fetch_with_fallback()` plus `FetchPolicyEngine`; there is no `_fetch_multi_source()` implementation to target.
- Base `.md` writing still happens in `workflow.helpers.add_basic_frontmatter()` and URL processor output code, while Markdown cleanup still lives inside `DocumentMixin.format_llm_output()`.
- `fetch_with_static()` and `fetch_with_static_conditional()` currently return Markdown, not raw HTML, by the time `_fetch_with_fallback()` sees their `FetchResult`. Native webextract therefore must be integrated inside the strategies that still hold raw HTML rather than as a late `_fetch_with_fallback()` post-processing hook.
- To avoid divergent behavior, `fetch_with_static()` should become a thin wrapper over the shared static HTML pipeline used by `fetch_with_static_conditional()` rather than continuing to call `MarkItDown.convert(url)` independently.

---

## Task Dependency Graph

```text
Task 1 (fixtures) ─┐
                    ├──→ Task 4 (scoring) ──→ Task 5 (schema/sanitize) ──┐
Task 2 (scaffold) ──→ Task 3 (metadata) ──┘                              │
                                                                          ├──→ Task 9 (playwright)  ──┐
Task 6 (standardize) ──┬──→ Task 7 (code blocks)  ─┐                     │                            │
                       └──→ Task 8 (footnotes/img) ─┴────────────────────┘                            │
                                                                                                       ├──→ Task 14 (golden tests + policy flip)
Task 10 (static/html strategy integration) ─────────────────────────────────────────────────────────────┤
Task 11 (markdown quality) ─────────────────────────────────────────────────────────────────────────────┤
Task 12 (frontmatter metadata) ────────────────────────────────────────────────────────────────────────┤
Task 13 (site extractors) ─────────────────────────────────────────────────────────────────────────────┘

Parallelizable pairs: 1∥2, 7∥8, 11∥12, 9∥10
```

---

### Task 1: Vendor Curated Defuddle Fixtures And Provenance

**Files:**
- Create: `packages/markitai/tests/fixtures/webextract/README.md`
- Create: `packages/markitai/tests/fixtures/webextract/manifest.json`
- Create: `packages/markitai/tests/fixtures/webextract/html/mdn-array.html`
- Create: `packages/markitai/tests/fixtures/webextract/html/github-issue-56.html`
- Create: `packages/markitai/tests/fixtures/webextract/html/x-article.html`
- Create: `packages/markitai/tests/fixtures/webextract/html/mintlify-codeblocks.html`
- Create: `packages/markitai/tests/fixtures/webextract/html/maggieappleton-footnotes.html`
- Create: `packages/markitai/tests/fixtures/webextract/html/lazy-image.html`
- Create: `packages/markitai/tests/fixtures/webextract/html/hidden-nodes.html`
- Create: `packages/markitai/tests/fixtures/webextract/html/javascript-links.html`
- Create: `packages/markitai/tests/fixtures/webextract/expected/mdn-array.md`
- Create: `packages/markitai/tests/fixtures/webextract/expected/github-issue-56.md`
- Create: `packages/markitai/tests/fixtures/webextract/expected/x-article.md`
- Create: `packages/markitai/tests/fixtures/webextract/expected/mintlify-codeblocks.md`
- Create: `packages/markitai/tests/fixtures/webextract/expected/maggieappleton-footnotes.md`
- Create: `packages/markitai/tests/fixtures/webextract/expected/lazy-image.md`
- Create: `packages/markitai/tests/fixtures/webextract/expected/hidden-nodes.md`
- Create: `packages/markitai/tests/fixtures/webextract/expected/javascript-links.md`
- Create: `packages/markitai/tests/unit/webextract/test_fixtures.py`

**Step 1: Write the failing test**

```python
from __future__ import annotations

import json
from pathlib import Path


def test_curated_webextract_fixture_manifest_is_complete() -> None:
    fixture_dir = Path(__file__).resolve().parents[2] / "fixtures" / "webextract"
    manifest = json.loads((fixture_dir / "manifest.json").read_text(encoding="utf-8"))

    assert [item["name"] for item in manifest] == [
        "mdn-array",
        "github-issue-56",
        "x-article",
        "mintlify-codeblocks",
        "maggieappleton-footnotes",
        "lazy-image",
        "hidden-nodes",
        "javascript-links",
    ]

    for item in manifest:
        assert (fixture_dir / "html" / item["html"]).exists()
        assert (fixture_dir / "expected" / item["expected_markdown"]).exists()
```

**Step 2: Run test to verify it fails**

Run: `cd packages/markitai && uv run pytest tests/unit/webextract/test_fixtures.py -v`
Expected: FAIL because the fixture corpus and manifest do not exist yet.

**Step 3: Write minimal implementation**

- Copy the curated fixture HTML and expected Markdown files from the local Defuddle checkout into the exact paths above.
- In `README.md`, record the original Defuddle filenames and note that these are seed baselines, not immutable output law.
- In `manifest.json`, store `name`, `html`, `expected_markdown`, and `url`.

```json
[
  {
    "name": "mdn-array",
    "html": "mdn-array.html",
    "expected_markdown": "mdn-array.md",
    "url": "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array"
  }
]
```

**Step 4: Run test to verify it passes**

Run: `cd packages/markitai && uv run pytest tests/unit/webextract/test_fixtures.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add packages/markitai/tests/fixtures/webextract packages/markitai/tests/unit/webextract/test_fixtures.py
git commit -m "test(webextract): vendor curated defuddle fixtures"
```

---

### Task 2: Add Native Webextract Scaffolding And Public API

**Files:**
- Modify: `packages/markitai/pyproject.toml`
- Modify: `uv.lock`
- Create: `packages/markitai/src/markitai/webextract/__init__.py`
- Create: `packages/markitai/src/markitai/webextract/types.py`
- Create: `packages/markitai/src/markitai/webextract/dom.py`
- Create: `packages/markitai/src/markitai/webextract/pipeline.py`
- Create: `packages/markitai/src/markitai/webextract/extractors/__init__.py`
- Create: `packages/markitai/src/markitai/webextract/extractors/base.py`
- Create: `packages/markitai/src/markitai/webextract/extractors/registry.py`
- Create: `packages/markitai/tests/unit/webextract/test_pipeline.py`

**Step 1: Write the failing test**

```python
from __future__ import annotations


def test_extract_web_content_returns_article_markdown() -> None:
    from markitai.webextract.pipeline import extract_web_content

    html = """
    <html>
      <head><title>Example Post</title></head>
      <body>
        <nav>menu</nav>
        <article><h1>Example Post</h1><p>Hello extraction.</p></article>
      </body>
    </html>
    """

    result = extract_web_content(html, "https://example.com/post")

    assert "Hello extraction." in result.markdown
    assert "<article" in result.clean_html
    assert result.metadata.title == "Example Post"
    assert result.word_count >= 2
```

**Step 2: Run test to verify it fails**

Run: `cd packages/markitai && uv run pytest tests/unit/webextract/test_pipeline.py::test_extract_web_content_returns_article_markdown -v`
Expected: FAIL because `markitai.webextract` does not exist.

**Step 3: Write minimal implementation**

- Add `beautifulsoup4>=4.12.3` as an explicit **runtime** (hard) dependency in `pyproject.toml`.
- In `dom.py`, centralize parser backend selection.
- In `pipeline.py`, add a first-pass implementation that chooses `<article>` or `<body>`, converts it to Markdown via `_html_fragment_to_markdown()` (uses existing `markitdown` dependency for HTML→Markdown rendering), and returns `ExtractedWebContent`.
- In `extractors/registry.py`, return `None` for now so later tasks can plug in site-specific extractors without changing the pipeline API.

```python
def parse_html(html: str) -> BeautifulSoup:
    parser = "lxml" if find_spec("lxml") is not None else "html.parser"
    return BeautifulSoup(html, parser)


def extract_web_content(html: str, url: str) -> ExtractedWebContent:
    soup = parse_html(html)
    root = soup.find("article") or soup.body or soup
    clean_html = str(root)
    markdown = _html_fragment_to_markdown(clean_html)
    return ExtractedWebContent(
        clean_html=clean_html,
        markdown=markdown,
        metadata=WebMetadata(title=_extract_title(soup)),
        word_count=len(markdown.split()),
    )
```

**Step 4: Run test to verify it passes**

Run: `cd packages/markitai && uv run pytest tests/unit/webextract/test_pipeline.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add packages/markitai/pyproject.toml uv.lock packages/markitai/src/markitai/webextract packages/markitai/tests/unit/webextract/test_pipeline.py
git commit -m "feat(webextract): add native extraction scaffolding"
```

---

### Task 3: Implement Metadata Extraction And Title Cleaning

**Files:**
- Create: `packages/markitai/src/markitai/webextract/metadata.py`
- Modify: `packages/markitai/src/markitai/webextract/types.py`
- Modify: `packages/markitai/src/markitai/webextract/pipeline.py`
- Create: `packages/markitai/tests/unit/webextract/test_metadata.py`

**Step 1: Write the failing tests**

```python
from __future__ import annotations


def test_extract_metadata_prefers_canonical_jsonld_and_meta_layers() -> None:
    from markitai.webextract.metadata import extract_metadata
    from markitai.webextract.dom import parse_html

    soup = parse_html(
        """
        <html>
          <head>
            <title>How Arrays Work | MDN</title>
            <link rel="canonical" href="https://example.com/canonical" />
            <meta property="og:site_name" content="MDN" />
            <meta name="author" content="Jane Doe" />
            <script type="application/ld+json">
              {"@type":"Article","headline":"How Arrays Work","datePublished":"2026-02-01"}
            </script>
          </head>
          <body></body>
        </html>
        """
    )

    metadata = extract_metadata(soup, "https://example.com/post")

    assert metadata.title == "How Arrays Work"
    assert metadata.author == "Jane Doe"
    assert metadata.site == "MDN"
    assert metadata.published == "2026-02-01"
    assert metadata.canonical_url == "https://example.com/canonical"


def test_clean_title_removes_site_suffixes() -> None:
    from markitai.webextract.metadata import clean_title

    assert clean_title("How Arrays Work | MDN", site="MDN") == "How Arrays Work"
```

**Step 2: Run test to verify it fails**

Run: `cd packages/markitai && uv run pytest tests/unit/webextract/test_metadata.py -v`
Expected: FAIL because `extract_metadata()` and `clean_title()` do not exist.

**Step 3: Write minimal implementation**

- Parse JSON-LD blocks for `headline`, `name`, `author`, `datePublished`, `text`, and `articleBody`.
- Merge metadata in this order: canonical/meta/OG/Twitter/JSON-LD/DOM fallback.
- Clean titles by stripping repeated site prefixes or suffixes.
- Update `pipeline.py` to use `extract_metadata()` rather than a local title-only helper.

```python
def clean_title(value: str | None, site: str | None = None) -> str | None:
    if not value:
        return None
    title = " ".join(value.split())
    if site:
        for sep in (" | ", " - ", " — ", " · "):
            if title.endswith(f"{sep}{site}"):
                return title[: -len(f"{sep}{site}")]
            if title.startswith(f"{site}{sep}"):
                return title[len(f"{site}{sep}") :]
    return title
```

**Step 4: Run test to verify it passes**

Run: `cd packages/markitai && uv run pytest tests/unit/webextract/test_metadata.py tests/unit/webextract/test_pipeline.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add packages/markitai/src/markitai/webextract/types.py packages/markitai/src/markitai/webextract/metadata.py packages/markitai/src/markitai/webextract/pipeline.py packages/markitai/tests/unit/webextract/test_metadata.py
git commit -m "feat(webextract): add metadata extraction and title cleaning"
```

---

### Task 4: Add Candidate Scoring, Block Pruning, And Adaptive Retry

**Files:**
- Create: `packages/markitai/src/markitai/webextract/constants.py`
- Create: `packages/markitai/src/markitai/webextract/scoring.py`
- Modify: `packages/markitai/src/markitai/webextract/pipeline.py`
- Create: `packages/markitai/tests/unit/webextract/test_scoring.py`
- Modify: `packages/markitai/tests/unit/webextract/test_pipeline.py`

**Step 1: Write the failing tests**

```python
from __future__ import annotations


def test_extract_web_content_prefers_article_over_nav_sidebar_and_footer() -> None:
    from markitai.webextract.pipeline import extract_web_content

    html = """
    <html>
      <body>
        <aside><p>signup signup signup</p></aside>
        <main>
          <article>
            <h1>Real Title</h1>
            <p>This is the real article body with enough text to win.</p>
            <p>It has paragraphs, low link density, and real content.</p>
          </article>
        </main>
        <footer><a href="/privacy">privacy</a></footer>
      </body>
    </html>
    """

    result = extract_web_content(html, "https://example.com/post")

    assert "Real Title" in result.markdown
    assert "signup signup signup" not in result.markdown


def test_extract_web_content_retries_without_partial_selectors_when_too_short() -> None:
    from markitai.webextract.pipeline import extract_web_content

    html = """
    <html>
      <body>
        <article class="post content story">
          <p>Short but valid statement that should survive fallback retry.</p>
        </article>
      </body>
    </html>
    """

    result = extract_web_content(html, "https://example.com/post")

    assert "should survive fallback retry" in result.markdown
    assert result.diagnostics["adaptive_retry_used"] is True
```

**Step 2: Run test to verify it fails**

Run: `cd packages/markitai && uv run pytest tests/unit/webextract/test_scoring.py tests/unit/webextract/test_pipeline.py -v`
Expected: FAIL because the naive extractor keeps the wrong blocks and has no retry logic.

**Step 3: Write minimal implementation**

- Add candidate scoring based on word count, paragraph count, link density, image density, heading bonus, date/author bonus, and content-class bonus.
- Remove likely noise blocks by negative score before selecting the best candidate.
- Add `remove_partial_selectors` support and an adaptive second pass when output word count is below `200`.

```python
def score_candidate(node: Tag) -> float:
    text = node.get_text(" ", strip=True)
    words = len(text.split())
    links = len(node.find_all("a"))
    paragraphs = len(node.find_all("p"))
    score = float(words)
    score += paragraphs * 20
    if links and words:
        score -= min(40.0, (links / max(words, 1)) * 200)
    classes = " ".join(node.get("class", []))
    if any(token in classes for token in ("article", "content", "post", "story")):
        score += 40
    return score
```

**Step 4: Run test to verify it passes**

Run: `cd packages/markitai && uv run pytest tests/unit/webextract/test_scoring.py tests/unit/webextract/test_pipeline.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add packages/markitai/src/markitai/webextract/constants.py packages/markitai/src/markitai/webextract/scoring.py packages/markitai/src/markitai/webextract/pipeline.py packages/markitai/tests/unit/webextract/test_scoring.py packages/markitai/tests/unit/webextract/test_pipeline.py
git commit -m "feat(webextract): add scoring and adaptive retry"
```

---

### Task 5: Add Schema Text Fallback And HTML Sanitization

**Files:**
- Create: `packages/markitai/src/markitai/webextract/schema.py`
- Create: `packages/markitai/src/markitai/webextract/sanitize.py`
- Modify: `packages/markitai/src/markitai/webextract/pipeline.py`
- Create: `packages/markitai/tests/unit/webextract/test_schema.py`
- Create: `packages/markitai/tests/unit/webextract/test_sanitize.py`

**Step 1: Write the failing tests**

```python
from __future__ import annotations


def test_schema_fallback_uses_smallest_matching_element_when_schema_text_is_longer() -> None:
    from markitai.webextract.pipeline import extract_web_content

    schema_text = (
        "This is the target post content with enough words to beat the short extracted "
        "result and trigger schema fallback."
    )

    html = f"""
    <html>
      <head>
        <script type="application/ld+json">
          {{"@type":"SocialMediaPosting","text":"{schema_text}"}}
        </script>
      </head>
      <body>
        <div id="feed">
          <div class="post"><p>Other post.</p></div>
          <div class="post" id="target"><p>{schema_text}</p></div>
        </div>
      </body>
    </html>
    """

    result = extract_web_content(html, "https://x.com/post/1")

    assert "target post content" in result.markdown
    assert "Other post." not in result.markdown
    assert result.diagnostics["schema_fallback_used"] is True


def test_sanitize_html_removes_event_handlers_and_javascript_links() -> None:
    from markitai.webextract.sanitize import sanitize_html_fragment

    sanitized = sanitize_html_fragment(
        '<div onclick="evil()"><a href="javascript:evil()">x</a><p>safe</p></div>'
    )

    assert "onclick" not in sanitized
    assert "javascript:" not in sanitized
    assert "safe" in sanitized
```

**Step 2: Run test to verify it fails**

Run: `cd packages/markitai && uv run pytest tests/unit/webextract/test_schema.py tests/unit/webextract/test_sanitize.py -v`
Expected: FAIL because there is no schema fallback or sanitizer.

**Step 3: Write minimal implementation**

- Parse JSON-LD `text` and `articleBody`.
- If schema word count exceeds extracted word count, look for the smallest DOM element whose normalized text contains the schema text.
- If no element matches, fall back to a sanitized `<p>` wrapper built from the raw schema text.
- Strip script/style/object/embed/form/button/input/textarea/select unless explicitly preserved.
- Remove `on*` attributes and reject `javascript:` / `data:text/html` URLs.

```python
def should_use_schema_fallback(schema_text: str | None, extracted_text: str) -> bool:
    return bool(schema_text) and len(schema_text.split()) > len(extracted_text.split())
```

**Step 4: Run test to verify it passes**

Run: `cd packages/markitai && uv run pytest tests/unit/webextract/test_schema.py tests/unit/webextract/test_sanitize.py tests/unit/webextract/test_pipeline.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add packages/markitai/src/markitai/webextract/schema.py packages/markitai/src/markitai/webextract/sanitize.py packages/markitai/src/markitai/webextract/pipeline.py packages/markitai/tests/unit/webextract/test_schema.py packages/markitai/tests/unit/webextract/test_sanitize.py
git commit -m "feat(webextract): add schema fallback and sanitization"
```

---

### Task 6: Add Core HTML Standardization

**Files:**
- Create: `packages/markitai/src/markitai/webextract/standardize.py`
- Modify: `packages/markitai/src/markitai/webextract/pipeline.py`
- Create: `packages/markitai/tests/unit/webextract/test_standardize.py`

**Step 1: Write the failing tests**

```python
from __future__ import annotations


def test_standardize_content_dedupes_title_heading_and_removes_comments() -> None:
    from markitai.webextract.dom import parse_html
    from markitai.webextract.standardize import standardize_content

    soup = parse_html(
        """
        <article>
          <!-- remove me -->
          <h1>How Arrays Work</h1>
          <h2>How Arrays Work</h2>
          <div><div><p>Body text.</p></div></div>
        </article>
        """
    )

    article = soup.article
    assert article is not None
    standardize_content(article, title="How Arrays Work", base_url="https://example.com")

    html = str(article)
    assert "<!--" not in html
    assert html.count("How Arrays Work") == 1
    assert "Body text." in html


def test_standardize_content_resolves_relative_urls_and_unwraps_javascript_links() -> None:
    from markitai.webextract.dom import parse_html
    from markitai.webextract.standardize import standardize_content

    soup = parse_html(
        '<article><a href="/docs">Docs</a><a href="javascript:void(0)">Click</a></article>'
    )

    article = soup.article
    assert article is not None
    standardize_content(article, title=None, base_url="https://example.com/base")

    html = str(article)
    assert 'href="https://example.com/docs"' in html
    assert "javascript:" not in html
```

**Step 2: Run test to verify it fails**

Run: `cd packages/markitai && uv run pytest tests/unit/webextract/test_standardize.py -v`
Expected: FAIL because `standardize_content()` does not exist.

**Step 3: Write minimal implementation**

- Remove HTML comments.
- Normalize spaces.
- Deduplicate the leading heading when it matches the extracted title.
- Flatten wrapper-only nodes with no semantic meaning.
- Strip unsafe or noisy attrs while preserving code language and footnote ids.
- Resolve relative links and image URLs using `urljoin()`.
- Remove orphaned trailing headings and extra `<br>` runs.

```python
def standardize_content(root: Tag, title: str | None, base_url: str) -> None:
    _remove_comments(root)
    _dedupe_title_heading(root, title)
    _flatten_wrappers(root)
    _strip_unwanted_attrs(root)
    _resolve_relative_urls(root, base_url)
```

**Step 4: Run test to verify it passes**

Run: `cd packages/markitai && uv run pytest tests/unit/webextract/test_standardize.py tests/unit/webextract/test_pipeline.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add packages/markitai/src/markitai/webextract/standardize.py packages/markitai/src/markitai/webextract/pipeline.py packages/markitai/tests/unit/webextract/test_standardize.py
git commit -m "feat(webextract): add core html standardization"
```

---

### Task 7: Normalize Code Blocks And Preserve Language Hints

**Files:**
- Create: `packages/markitai/src/markitai/webextract/elements/__init__.py`
- Create: `packages/markitai/src/markitai/webextract/elements/code.py`
- Modify: `packages/markitai/src/markitai/webextract/standardize.py`
- Create: `packages/markitai/tests/unit/webextract/test_code_elements.py`

**Step 1: Write the failing tests**

```python
from __future__ import annotations


def test_normalize_code_blocks_wraps_preformatted_code_and_keeps_language() -> None:
    from markitai.webextract.dom import parse_html
    from markitai.webextract.elements.code import normalize_code_blocks

    soup = parse_html(
        """
        <article>
          <code class="language-python" style="white-space: pre;">
          print("hello")
          </code>
        </article>
        """
    )

    article = soup.article
    assert article is not None
    normalize_code_blocks(article)

    html = str(article)
    assert "<pre>" in html
    assert "language-python" in html
    assert 'print("hello")' in html
```

**Step 2: Run test to verify it fails**

Run: `cd packages/markitai && uv run pytest tests/unit/webextract/test_code_elements.py -v`
Expected: FAIL because code normalization does not exist.

**Step 3: Write minimal implementation**

- Detect preformatted inline code and wrap it in `<pre><code>`.
- Preserve language classes from `data-lang`, `class`, and common highlighter wrappers.
- Remove line-number scaffolding and empty wrapper nodes around code samples.
- Call `normalize_code_blocks()` from `standardize_content()`.

```python
def normalize_code_blocks(root: Tag) -> None:
    for code in root.find_all("code"):
        if _looks_preformatted(code):
            _wrap_in_pre(code)
        _retain_language_hint(code)
        _drop_line_number_markup(code)
```

**Step 4: Run test to verify it passes**

Run: `cd packages/markitai && uv run pytest tests/unit/webextract/test_code_elements.py tests/unit/webextract/test_standardize.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add packages/markitai/src/markitai/webextract/elements/__init__.py packages/markitai/src/markitai/webextract/elements/code.py packages/markitai/src/markitai/webextract/standardize.py packages/markitai/tests/unit/webextract/test_code_elements.py
git commit -m "feat(webextract): normalize code blocks"
```

---

### Task 8: Normalize Footnotes And Images

**Files:**
- Create: `packages/markitai/src/markitai/webextract/elements/footnotes.py`
- Create: `packages/markitai/src/markitai/webextract/elements/images.py`
- Modify: `packages/markitai/src/markitai/webextract/standardize.py`
- Create: `packages/markitai/tests/unit/webextract/test_footnotes.py`
- Create: `packages/markitai/tests/unit/webextract/test_images.py`

**Step 1: Write the failing tests**

```python
from __future__ import annotations


def test_normalize_footnotes_creates_canonical_ordered_list() -> None:
    from markitai.webextract.dom import parse_html
    from markitai.webextract.elements.footnotes import normalize_footnotes

    soup = parse_html(
        """
        <article>
          <p>Text<a href="#fn1" id="fnref1">1</a></p>
          <section class="footnotes">
            <p id="fn1">1. Note body <a href="#fnref1">↩</a></p>
          </section>
        </article>
        """
    )

    article = soup.article
    assert article is not None
    normalize_footnotes(article)

    html = str(article)
    assert "<ol" in html
    assert "Note body" in html


def test_normalize_images_upgrades_lazy_sources_and_builds_figures() -> None:
    from markitai.webextract.dom import parse_html
    from markitai.webextract.elements.images import normalize_images

    soup = parse_html(
        """
        <article>
          <img data-src="/hero.png" alt="Hero" />
          <p class="caption">Hero caption</p>
        </article>
        """
    )

    article = soup.article
    assert article is not None
    normalize_images(article, base_url="https://example.com/post")

    html = str(article)
    assert 'src="https://example.com/hero.png"' in html
    assert "<figure" in html
    assert "Hero caption" in html
```

**Step 2: Run test to verify it fails**

Run: `cd packages/markitai && uv run pytest tests/unit/webextract/test_footnotes.py tests/unit/webextract/test_images.py -v`
Expected: FAIL because footnote/image normalization does not exist.

**Step 3: Write minimal implementation**

- Convert common footnote containers into a canonical `<section><ol><li>` shape.
- Remove backlink clutter while preserving reference ids.
- Upgrade `data-src`, `data-original`, `srcset`, and `<picture>` sources into final `src`.
- Turn image-plus-caption patterns into `<figure><img><figcaption>`.
- Skip tiny placeholder images.

```python
def normalize_images(root: Tag, base_url: str) -> None:
    for img in root.find_all("img"):
        src = img.get("src") or img.get("data-src") or img.get("data-original")
        if src:
            img["src"] = urljoin(base_url, src)
    _promote_captions_to_figure(root)
```

**Step 4: Run test to verify it passes**

Run: `cd packages/markitai && uv run pytest tests/unit/webextract/test_footnotes.py tests/unit/webextract/test_images.py tests/unit/webextract/test_standardize.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add packages/markitai/src/markitai/webextract/elements/footnotes.py packages/markitai/src/markitai/webextract/elements/images.py packages/markitai/src/markitai/webextract/standardize.py packages/markitai/tests/unit/webextract/test_footnotes.py packages/markitai/tests/unit/webextract/test_images.py
git commit -m "feat(webextract): normalize footnotes and images"
```

---

### Task 9: Integrate Native Extraction Into Playwright HTML Flow

**Files:**
- Modify: `packages/markitai/src/markitai/fetch_playwright.py`
- Modify: `packages/markitai/tests/unit/test_fetch_playwright.py`

**Step 1: Write the failing tests**

```python
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest


@pytest.mark.asyncio
async def test_playwright_fetch_runs_native_webextract_before_markdown_return() -> None:
    from markitai.fetch_playwright import PlaywrightRenderer
    from markitai.webextract.types import ExtractedWebContent, WebMetadata

    renderer = PlaywrightRenderer()
    renderer._browser = AsyncMock()
    renderer._playwright = AsyncMock()

    mock_page = AsyncMock()
    mock_page.content.return_value = "<html><body><article><p>Hello</p></article></body></html>"
    mock_page.title.return_value = "Fallback Title"
    mock_page.url = "https://example.com/post"
    mock_page.inner_text = AsyncMock(return_value="Hello")

    mock_context = AsyncMock()
    mock_context.new_page.return_value = mock_page

    renderer._browser.new_context.return_value = mock_context

    with patch("markitai.fetch_playwright.extract_web_content") as mock_extract:
        mock_extract.return_value = ExtractedWebContent(
            clean_html="<article><p>Hello</p></article>",
            markdown="# Hello\n\nBody",
            metadata=WebMetadata(title="Hello", author="Jane"),
            word_count=20,
            diagnostics={"extractor": "generic"},
        )

        result = await renderer.fetch("https://example.com/post")

    assert "# Hello" in result.content
    assert result.title == "Hello"
    assert result.metadata["source_frontmatter"]["author"] == "Jane"
```

**Step 2: Run test to verify it fails**

Run: `cd packages/markitai && uv run pytest tests/unit/test_fetch_playwright.py -k native_webextract -v`
Expected: FAIL because Playwright still calls `_html_to_markdown()` on the full page HTML.

**Step 3: Write minimal implementation**

- After `page.content()`, call `extract_web_content(html_content, final_url or url)`.
- Accept `extracted.markdown` only when it was derived from `extracted.clean_html` by the shared render helper and the native path passes quality gates.
- Keep the existing `_html_to_markdown()` and `inner_text("body")` fallback path when native extraction raises or fails the quality gate.
- Attach extracted metadata under `PlaywrightFetchResult.metadata["source_frontmatter"]` and store native diagnostics in a separate telemetry field (for example `metadata["webextract_diagnostics"]`).

```python
extracted = extract_web_content(html_content, final_url or url)
markdown_content = extracted.markdown
title = extracted.metadata.title or await page.title()
metadata = {
    "renderer": "playwright",
    "wait_for": wait_for,
    "source_frontmatter": dataclasses.asdict(extracted.metadata),
    "webextract_diagnostics": extracted.diagnostics,
}
```

**Step 4: Run test to verify it passes**

Run: `cd packages/markitai && uv run pytest tests/unit/test_fetch_playwright.py -k "native_webextract or converts_basic_html" -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add packages/markitai/src/markitai/fetch_playwright.py packages/markitai/tests/unit/test_fetch_playwright.py
git commit -m "feat(fetch): run native webextract in playwright flow"
```

---

### Task 10: Integrate Native Extraction Into Static HTML Strategies

**Files:**
- Modify: `packages/markitai/src/markitai/fetch.py`
- Modify: `packages/markitai/tests/unit/test_fetch.py`

**Step 1: Write the failing tests**

```python
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_fetch_with_static_conditional_prefers_native_webextract_for_html() -> None:
    from markitai.fetch import fetch_with_static_conditional

    html_content = b"<html><body><article><h1>T</h1><p>Body</p></article></body></html>"

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = html_content
    mock_response.text = html_content.decode("utf-8")
    mock_response.headers = {"content-type": "text/html"}
    mock_response.url = "https://example.com/post"

    with (
        patch("markitai.fetch.get_static_http_client") as mock_client_factory,
        patch("markitai.fetch.extract_web_content") as mock_extract,
    ):
        from markitai.webextract.types import ExtractedWebContent, WebMetadata

        mock_client = AsyncMock()
        mock_client.name = "httpx"
        mock_client.get.return_value = mock_response
        mock_client_factory.return_value = mock_client

        mock_extract.return_value = ExtractedWebContent(
            clean_html="<article><h1>T</h1><p>Body</p></article>",
            markdown="# T\n\nBody",
            metadata=WebMetadata(title="T"),
            word_count=2,
            diagnostics={"extractor": "generic"},
        )

        result = await fetch_with_static_conditional("https://example.com/post")

    assert result.result is not None
    assert "Body" in result.result.content
    assert result.result.metadata.get("converter") == "native-html"
    assert result.result.metadata["source_frontmatter"]["title"] == "T"


@pytest.mark.asyncio
async def test_fetch_with_static_uses_the_same_native_html_pipeline() -> None:
    from markitai.fetch import FetchResult, fetch_with_static

    expected = FetchResult(
        content="# T\n\nBody",
        strategy_used="static",
        title="T",
        url="https://example.com/post",
        final_url="https://example.com/post",
        metadata={
            "converter": "native-html",
            "source_frontmatter": {"title": "T"},
            "webextract_diagnostics": {"extractor": "generic"},
        },
    )

    with patch("markitai.fetch.fetch_with_static_conditional") as mock_conditional:
        mock_conditional.return_value = type(
            "Result",
            (),
            {"result": expected, "not_modified": False, "etag": None, "last_modified": None},
        )()

        result = await fetch_with_static("https://example.com/post")

    assert result.content == "# T\n\nBody"
    assert result.metadata["converter"] == "native-html"
```

**Step 2: Run test to verify it fails**

Run: `cd packages/markitai && uv run pytest tests/unit/test_fetch.py -k "native_webextract or static_conditional or same_native_html_pipeline" -v`
Expected: FAIL because static HTML fetching still sends the full page directly to MarkItDown, and `fetch_with_static()` still uses an independent MarkItDown-only path.

**Step 3: Write minimal implementation**

- Add a strategy-local helper in `fetch.py` that takes raw HTML plus URL, runs `extract_web_content()`, applies a quality gate, then renders `clean_html` via MarkItDown when native extraction is accepted.
- Use that helper from `fetch_with_static_conditional()` for `text/html` responses before the temp-file whole-page MarkItDown fallback path.
- If the native path fails closed, continue with the current full-response MarkItDown behavior so the static strategy remains conservative.
- Refactor `fetch_with_static()` into a thin wrapper over the same shared static HTML pipeline so explicit `static`, cached `static`, and `auto` strategy hops do not diverge.

```python
def _convert_html_with_native_webextract(
    html: str, url: str, final_url: str | None = None
) -> FetchResult:
    """Convert raw HTML through native webextract, failing closed to fallback."""
    from markitai.webextract.pipeline import extract_web_content

    extracted = extract_web_content(html, final_url or url)
    return FetchResult(
        content=extracted.markdown,
        strategy_used="static",
        title=extracted.metadata.title,
        url=url,
        final_url=final_url or url,
        metadata={
            "converter": "native-html",
            "source_frontmatter": asdict(extracted.metadata),
            "webextract_diagnostics": extracted.diagnostics,
        },
    )
```

**Step 4: Run test to verify it passes**

Run: `cd packages/markitai && uv run pytest tests/unit/test_fetch.py -k "native_webextract or static_conditional or same_native_html_pipeline" -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add packages/markitai/src/markitai/fetch.py packages/markitai/tests/unit/test_fetch.py
git commit -m "feat(fetch): apply native webextract in static html strategies"
```

---

### Task 11: Add Shared Pre-LLM Markdown Quality Pass

**Files:**
- Create: `packages/markitai/src/markitai/utils/markdown_quality.py`
- Modify: `packages/markitai/src/markitai/workflow/core.py`
- Modify: `packages/markitai/src/markitai/workflow/helpers.py`
- Modify: `packages/markitai/src/markitai/cli/processors/url.py`
- Modify: `packages/markitai/src/markitai/llm/document.py`
- Create: `packages/markitai/tests/unit/test_markdown_quality.py`
- Modify: `packages/markitai/tests/unit/test_workflow_core.py`
- Modify: `packages/markitai/tests/unit/test_workflow_helpers.py`
- Modify: `packages/markitai/tests/integration/test_output_format.py`

**Step 1: Write the failing tests**

```python
from __future__ import annotations


def test_normalize_markdown_reuses_existing_cleanup_primitives() -> None:
    from markitai.utils.markdown_quality import normalize_markdown

    markdown = "[Title\n\nDescription](/docs)\n\n__MARKITAI_FILE_ASSET__\n\n# Heading"
    cleaned = normalize_markdown(markdown)

    assert "[Title](/docs)" in cleaned
    assert "__MARKITAI" not in cleaned
    assert cleaned.endswith("\n")


def test_write_base_markdown_normalizes_content_before_frontmatter(sample_context) -> None:
    from markitai.converter.base import ConvertResult
    from markitai.workflow.core import write_base_markdown

    sample_context.input_path = Path("test.txt")
    sample_context.conversion_result = ConvertResult(
        markdown="[Title\n\nDescription](/docs)",
        images=[],
        metadata={},
    )
    sample_context.output_file = sample_context.output_dir / "test.md"

    result = write_base_markdown(sample_context)

    assert result.success is True
    written = sample_context.output_file.read_text(encoding="utf-8")
    assert "[Title](/docs)" in written
```

**Step 2: Run test to verify it fails**

Run: `cd packages/markitai && uv run pytest tests/unit/test_markdown_quality.py tests/unit/test_workflow_core.py -k normalize -v`
Expected: FAIL because the shared quality module does not exist and `write_base_markdown()` still writes raw Markdown.

**Step 3: Write minimal implementation**

- Move current cleanup sequence out of `DocumentMixin.format_llm_output()` into `utils/markdown_quality.py`.
- Reuse it in:
  - `workflow.core.write_base_markdown()`
  - URL processor base `.md` writing
  - `DocumentMixin.format_llm_output()`
- Start by reusing the existing cleanup primitives from `utils.text`: broken links, PPT headers/footers, residual placeholders, whitespace normalization.

```python
def normalize_markdown(content: str) -> str:
    from markitai.utils.text import (
        clean_ppt_headers_footers,
        clean_residual_placeholders,
        fix_broken_markdown_links,
        normalize_markdown_whitespace,
    )

    content = fix_broken_markdown_links(content)
    content = clean_ppt_headers_footers(content)
    content = clean_residual_placeholders(content)
    content = normalize_markdown_whitespace(content)
    return content
```

**Step 4: Run test to verify it passes**

Run: `cd packages/markitai && uv run pytest tests/unit/test_markdown_quality.py tests/unit/test_workflow_core.py tests/integration/test_output_format.py -k "normalize or broken_markdown_links or placeholders" -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add packages/markitai/src/markitai/utils/markdown_quality.py packages/markitai/src/markitai/workflow/core.py packages/markitai/src/markitai/workflow/helpers.py packages/markitai/src/markitai/cli/processors/url.py packages/markitai/src/markitai/llm/document.py packages/markitai/tests/unit/test_markdown_quality.py packages/markitai/tests/unit/test_workflow_core.py packages/markitai/tests/unit/test_workflow_helpers.py packages/markitai/tests/integration/test_output_format.py
git commit -m "feat(markdown): add shared pre-llm quality normalization"
```

---

### Task 12: Extend Existing Frontmatter Utilities With Native Metadata

**Files:**
- Modify: `packages/markitai/src/markitai/utils/frontmatter.py`
- Modify: `packages/markitai/src/markitai/workflow/helpers.py`
- Modify: `packages/markitai/src/markitai/cli/processors/url.py`
- Modify: `packages/markitai/tests/unit/test_frontmatter.py`
- Modify: `packages/markitai/tests/unit/test_workflow_helpers.py`

**Step 1: Write the failing tests**

```python
from __future__ import annotations


def test_add_basic_frontmatter_includes_native_web_metadata() -> None:
    from markitai.workflow.helpers import add_basic_frontmatter

    content = add_basic_frontmatter(
        "# Clean Title\n\nBody",
        "https://example.com/post",
        fetch_strategy="static",
        title="Clean Title",
        extra_meta={
            "author": "Jane Doe",
            "site": "Example",
            "published": "2026-02-01",
            "canonical_url": "https://example.com/post",
        },
    )

    assert "author: Jane Doe" in content
    assert "site: Example" in content
    assert "published: '2026-02-01'" in content
    assert "canonical_url: https://example.com/post" in content
```

**Step 2: Run test to verify it fails**

Run: `cd packages/markitai && uv run pytest tests/unit/test_frontmatter.py tests/unit/test_workflow_helpers.py -k native_web_metadata -v`
Expected: FAIL because `add_basic_frontmatter()` ignores extracted metadata.

**Step 3: Write minimal implementation**

- Do not add a second parallel frontmatter builder.
- Extend the existing utilities instead:
  - add a small `merge_source_metadata()` helper in `utils.frontmatter.py`
  - update `workflow.helpers.FRONTMATTER_FIELD_ORDER` so source metadata fields serialize in a stable order
  - keep `add_basic_frontmatter()` consuming `extra_meta`, and standardize native metadata so it is merged into `fetch_result.metadata["source_frontmatter"]` before this layer
- Preserve the current fields and add only stable, useful source metadata: `author`, `site`, `published`, `canonical_url`, `fetch_strategy`.

```python
def merge_source_metadata(
    frontmatter: dict[str, Any],
    metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    if metadata:
        for key in ("author", "site", "published", "canonical_url"):
            if metadata.get(key):
                frontmatter[key] = metadata[key]
    return frontmatter
```

Also update canonical field ordering to include:

```python
FRONTMATTER_FIELD_ORDER = [
    "title",
    "source",
    "author",
    "site",
    "published",
    "canonical_url",
    "description",
    "tags",
    "markitai_processed",
    "fetch_strategy",
]
```

**Step 4: Run test to verify it passes**

Run: `cd packages/markitai && uv run pytest tests/unit/test_frontmatter.py tests/unit/test_workflow_helpers.py -k native_web_metadata -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add packages/markitai/src/markitai/utils/frontmatter.py packages/markitai/src/markitai/workflow/helpers.py packages/markitai/src/markitai/cli/processors/url.py packages/markitai/tests/unit/test_frontmatter.py packages/markitai/tests/unit/test_workflow_helpers.py
git commit -m "feat(frontmatter): include native web metadata in base output"
```

---

### Task 13: Add Site Extractor Registry With GitHub And X Specializations

**Files:**
- Create: `packages/markitai/src/markitai/webextract/extractors/github_issue.py`
- Create: `packages/markitai/src/markitai/webextract/extractors/x_article.py`
- Modify: `packages/markitai/src/markitai/webextract/extractors/base.py`
- Modify: `packages/markitai/src/markitai/webextract/extractors/registry.py`
- Modify: `packages/markitai/src/markitai/webextract/pipeline.py`
- Create: `packages/markitai/tests/unit/webextract/test_extractors.py`

**Step 1: Write the failing tests**

```python
from __future__ import annotations


def test_registry_uses_github_issue_extractor_for_issue_urls() -> None:
    from markitai.webextract.extractors.registry import find_extractor

    extractor = find_extractor("https://github.com/org/repo/issues/56")

    assert extractor is not None
    assert extractor.name == "github_issue"


def test_registry_uses_x_article_extractor_for_x_article_urls() -> None:
    from markitai.webextract.extractors.registry import find_extractor

    extractor = find_extractor("https://x.com/i/articles/123")

    assert extractor is not None
    assert extractor.name == "x_article"
```

**Step 2: Run test to verify it fails**

Run: `cd packages/markitai && uv run pytest tests/unit/webextract/test_extractors.py -v`
Expected: FAIL because the registry has no concrete extractors.

**Step 3: Write minimal implementation**

- Define a tiny base protocol:
  - `name`
  - `matches_url(url: str) -> bool`
  - `extract_root(soup: BeautifulSoup) -> Tag | None`
- Implement only two initial extractors:
  - GitHub issue/PR conversation root
  - X article/social posting root
- Have `pipeline.py` ask the registry first, then fall back to the generic scorer.

```python
class BaseSiteExtractor(Protocol):
    name: str

    def matches_url(self, url: str) -> bool: ...
    def extract_root(self, soup: BeautifulSoup) -> Tag | None: ...
```

**Step 4: Run test to verify it passes**

Run: `cd packages/markitai && uv run pytest tests/unit/webextract/test_extractors.py tests/unit/webextract/test_pipeline.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add packages/markitai/src/markitai/webextract/extractors/base.py packages/markitai/src/markitai/webextract/extractors/registry.py packages/markitai/src/markitai/webextract/extractors/github_issue.py packages/markitai/src/markitai/webextract/extractors/x_article.py packages/markitai/src/markitai/webextract/pipeline.py packages/markitai/tests/unit/webextract/test_extractors.py
git commit -m "feat(webextract): add github and x site extractors"
```

---

### Task 14: Turn On Golden Quality Tests And Shift Fetch Policy To Local-First

**Files:**
- Create: `packages/markitai/tests/integration/test_webextract_golden.py`
- Modify: `packages/markitai/src/markitai/fetch_policy.py`
- Modify: `packages/markitai/src/markitai/fetch.py`
- Modify: `packages/markitai/tests/unit/test_fetch_policy.py`
- Modify: `packages/markitai/src/markitai/fetch_policy.py` (where `ALL_STRATEGIES` is defined)

**Step 1: Write the failing tests**

```python
from __future__ import annotations

import json
from pathlib import Path


def test_native_webextract_matches_curated_golden_outputs() -> None:
    from markitai.webextract.pipeline import extract_web_content

    fixture_dir = Path(__file__).resolve().parents[1] / "fixtures" / "webextract"
    manifest = json.loads((fixture_dir / "manifest.json").read_text(encoding="utf-8"))

    for item in manifest:
        html = (fixture_dir / "html" / item["html"]).read_text(encoding="utf-8")
        expected = (fixture_dir / "expected" / item["expected_markdown"]).read_text(
            encoding="utf-8"
        )
        result = extract_web_content(html, item["url"])

        # Use key-content assertions for early development stability.
        # Switch to exact matching (assert result.markdown.strip() == expected.strip())
        # once the pipeline is stable.
        for key_line in _extract_key_lines(expected):
            assert key_line in result.markdown, (
                f"{item['name']}: missing expected content: {key_line!r}"
            )


def _extract_key_lines(expected_md: str) -> list[str]:
    """Extract headings, code fences, and non-trivial content lines for assertion."""
    key = []
    for line in expected_md.strip().splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or stripped.startswith("```") or len(stripped) > 40:
            key.append(stripped)
    return key


def test_policy_prefers_local_native_paths_for_normal_domains() -> None:
    from markitai.fetch_policy import FetchPolicyEngine

    decision = FetchPolicyEngine().decide(
        domain="example.com",
        known_spa=False,
        explicit_strategy=None,
        fallback_patterns=["x.com"],
        policy_enabled=True,
    )

    assert decision.order == [
        "static",
        "playwright",
        "jina",
        "cloudflare",
        "defuddle",
    ]


def test_policy_flip_keeps_domain_preference_semantics() -> None:
    from markitai.fetch_policy import FetchPolicyEngine

    decision = FetchPolicyEngine().decide(
        domain="example.com",
        known_spa=False,
        explicit_strategy=None,
        fallback_patterns=["x.com"],
        policy_enabled=True,
        domain_prefer_strategy="jina",
    )

    assert decision.order[0] == "jina"
    assert decision.reason == "domain_prefer_jina"
```

**Step 2: Run test to verify it fails**

Run: `cd packages/markitai && uv run pytest tests/integration/test_webextract_golden.py tests/unit/test_fetch_policy.py -v`
Expected: FAIL because the native extractor is not yet locked to the curated baselines and the policy still prefers Defuddle first.

**Step 3: Write minimal implementation**

- Add the integration golden test harness.
- Add explicit fallback-path tests that prove native extraction can lose and MarkItDown still wins cleanly.
- Update `FetchPolicyEngine` end-state order:
  - default: `static -> playwright -> jina -> cloudflare -> defuddle`
  - SPA/fallback domains: `playwright -> jina -> cloudflare -> static -> defuddle`
- Update `_fetch_with_fallback()` so the router consumes the new policy order while preserving:
  - `domain_profiles[domain].prefer_strategy`
  - `policy.max_strategy_hops`
  - current invalid-content gating and SPA learning
- Update `ALL_STRATEGIES` and the Defuddle migration comment in `fetch_policy.py` to reflect the new native-first architecture.

```python
ALL_STRATEGIES = ["static", "playwright", "jina", "cloudflare", "defuddle"]
```

**Step 4: Run test to verify it passes**

Run: `cd packages/markitai && uv run pytest tests/integration/test_webextract_golden.py tests/unit/test_fetch_policy.py tests/unit/test_fetch.py tests/unit/test_fetch_playwright.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add packages/markitai/tests/integration/test_webextract_golden.py packages/markitai/src/markitai/fetch_policy.py packages/markitai/src/markitai/fetch.py packages/markitai/tests/unit/test_fetch_policy.py
git commit -m "feat(fetch): make native webextract the default html quality path"
```

---

## Verification Sequence Before Merge

Run these only after all tasks are green:

```bash
cd packages/markitai && uv run pytest tests/unit/webextract -v
cd packages/markitai && uv run pytest tests/unit/test_config.py tests/unit/test_fetch_policy.py -v
cd packages/markitai && uv run pytest tests/unit/test_fetch.py tests/unit/test_fetch_playwright.py tests/unit/test_fetch_policy.py -v
cd packages/markitai && uv run pytest tests/unit/test_workflow_core.py tests/unit/test_workflow_helpers.py tests/unit/test_frontmatter.py tests/unit/test_markdown_quality.py -v
cd packages/markitai && uv run pytest tests/integration/test_output_format.py tests/integration/test_webextract_golden.py -v
cd packages/markitai && uv run ruff check src tests
cd packages/markitai && uv run ruff format --check src tests
cd packages/markitai && uv run pyright
```

## Notes For The Implementer

- Treat Defuddle fixture outputs as a benchmark and starting point, not a forced byte-for-byte clone of every quirk.
- If a fixture mismatch is due to an intentional Markitai difference, update the copied expected Markdown in the same commit that explains the reason.
- Keep the HTML extractor deterministic and side-effect free. No network, no browser calls, no temp files.
- Do not duplicate cleanup logic between `utils.markdown_quality` and `DocumentMixin.format_llm_output()`. The latter should become a thin wrapper.
- Do not flip fetch policy order until the curated golden suite is passing.
