# Web Extraction Fixtures

This directory contains HTML fixtures and expected-output contracts used by the
native extraction parity and benchmark tests.

---

## Directory Contents

| File | Description |
|------|-------------|
| `x_status_2030105637204676808.playwright.html` | Minimized Playwright capture of an X/Twitter status page |
| `x_status_2030105637204676808.expected.json` | Semantic contract for the X status fixture |
| `generic_article.playwright.html` | Synthetic generic blog article with sidebar noise |
| `generic_article.expected.json` | Semantic contract for the generic article fixture |
| `github_issue_thread.playwright.html` | Minimized GitHub issue thread page |
| `github_issue_thread.expected.json` | Semantic contract for the GitHub issue fixture |
| `hackernews_thread.playwright.html` | Minimized Hacker News thread page |
| `reddit_post.playwright.html` | Minimized Reddit post page |
| `youtube_page.playwright.html` | Minimized YouTube video page |
| `shadow_dom_page.html` | Synthetic page with shadow DOM elements |

---

## Source Provenance

All `.playwright.html` files are **minimized snapshots** derived from real page
captures. Minimization means:

1. Personal data, session tokens, and ad tracking scripts removed.
2. Content sections shrunk to the minimum needed to exercise extraction logic.
3. Noise elements (trending, recommendations, signup walls) retained where
   necessary to verify they are correctly excluded from output.
4. No external assets — images use data URIs or are omitted entirely.

Synthetic fixtures (`generic_article`, `shadow_dom_page`) are hand-authored to
test specific extraction behaviours without any real user data.

---

## Expected JSON Schema

Each `*.expected.json` file conforms to the following structure:

```json
{
  "url": "https://example.com/...",       // optional; overrides og:url for extractor routing
  "metadata": {
    "title": "..."                         // expected page title
  },
  "info": {
    "content_profile": "social_post"       // expected ContentProfile enum value
  },
  "semantic": {                            // null if no semantic model expected
    "thread": {
      "main_item": {
        "author_handle": "@handle",        // expected @handle (substring match)
        "author_name": "Display Name",     // optional; substring match
        "text": "exact text",              // optional; substring match against item.text
        "text_contains": "substring",      // optional; substring match (preferred)
        "published_at": "ISO-8601"         // optional; exact match
      },
      "author_replies": [],                // reserved for future use
      "third_party_replies": []            // reserved for future use
    }
  },
  "markdown_must_contain": [              // phrases that must appear in extracted markdown
    "key phrase"
  ],
  "markdown_must_not_contain": [          // phrases that must NOT appear in extracted markdown
    "noise phrase"
  ]
}
```

All string comparisons in `semantic.thread.main_item` use substring matching
for robustness. Exact equality is intentionally avoided because rendering may
add punctuation or whitespace.

---

## Adding New Fixtures

1. Capture the page HTML using Playwright or a browser devtools save.
2. Minimise: remove personal data, reduce content to a representative sample,
   keep the noise elements you want the extractor to reject.
3. Save the HTML as `<slug>.playwright.html`.
4. Create `<slug>.expected.json` using the schema above. Start with just
   `markdown_must_contain` / `markdown_must_not_contain` before adding
   `semantic` assertions.
5. Run `uv run pytest packages/markitai/tests/integration/test_defuddle_parity.py`
   to confirm the fixture passes with the current extraction.
6. Add a parametrised entry to `test_parity_clean_html_is_non_empty` if needed.

---

## Minimization Expectations

A minimized fixture should:

- Be under 50 KB where possible.
- Contain only the primary content section plus the noise elements being tested.
- Have no `<script>` tags (extraction runs on static HTML only).
- Retain `<meta og:url>` so the extractor registry can route to the correct
  site-specific extractor.
- Use consistent indentation for readability in diffs.

---

## Provenance and Attribution

Every `.playwright.html` fixture must have a clear provenance record so
reviewers can verify the minimization was performed correctly and no personal
data was introduced accidentally.

When adding a new fixture, record its provenance in a comment at the top of
the HTML file:

```html
<!-- Fixture: <slug>.playwright.html
     Source:  <canonical URL>
     Captured: <YYYY-MM-DD>
     Minimized: removed <N> KB of sidebar / ad / script content
     PII check: no personal data retained
-->
```

For synthetic fixtures (e.g. `generic_article`, `shadow_dom_page`), note
`Source: synthetic` and the specific extraction behaviour being exercised.

---

## Resolver and Enricher Coverage

Fixture files are grouped by the extraction path they exercise:

| Fixture prefix | Resolver extractor | Enricher | Content profile |
|---------------|-------------------|----------|-----------------|
| `x_status_*` | `x_tweet` | `x_oembed` (optional) | `social_post` |
| `github_issue_thread` | `github_thread` | — | `discussion_issue` |
| `hackernews_thread` | `hackernews_thread` | — | `discussion_thread` |
| `reddit_post` | `reddit_post` | — | `discussion_thread` |
| `youtube_page` | `youtube_page` | — | `rich_media_page` |
| `generic_article` | — (generic pipeline) | — | `generic_article` |
| `shadow_dom_page` | — (generic pipeline) | — | `generic_article` |

Fixtures that cover **resolver extractors** (those implementing `resolve()`)
verify that the resolver path produces a `ResolvedPage` with the expected
`content_html` or `semantic.thread` before Markdown rendering.

Fixtures that cover **enrichers** should be tested both with and without the
enricher active (`EnrichmentPolicy(allow_network=False)`) to confirm the sync
resolver baseline remains acceptable on its own.

---

## Quality Profile Assertions

The `info.content_profile` field in `*.expected.json` must match the profile
used by the quality gate. If the profile is wrong, `assess_native_markdown()`
may apply the incorrect rejection heuristics and produce misleading test
failures.

Use the table above to determine which profile to set. For new site types not
yet in the table, default to `generic_article` until a dedicated profile is
added to `webextract/quality.py`.
