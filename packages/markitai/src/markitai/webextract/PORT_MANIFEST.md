# defuddle port manifest

`markitai.webextract` is a Python port of
[defuddle](https://github.com/kepano/defuddle) (TypeScript, MIT © kepano).
This manifest records which upstream sources the port tracks and the upstream
commit the parity corpus is pinned to, so corpus and algorithm resync from one
place.

Pinned upstream commit: `a332b4d5d539066ddfe19fc4ef6f1b6ffaf914b8`

The pin must match `tests/defuddle_fixtures/VERSION` (enforced by
`tests/unit/webextract/test_port_manifest.py`; both files are rewritten by
`scripts/sync_defuddle_fixtures.sh`). `.github/workflows/defuddle-watch.yml`
opens an issue when upstream cuts a release ahead of this pin.

## Tracked upstream sources

Upstream paths are relative to defuddle `src/`, port paths to
`markitai/webextract/`.

| upstream | port |
| --- | --- |
| `constants.ts` | `constants.py` |
| `content-boundary.ts` | `content_boundary.py` |
| `defuddle.ts` | `scoring.py` (findMainContent/ContentScorer), `mobile_styles.py` (mobile-style pruning), `pipeline.py` (orchestration) |
| `markdown.ts` | `markdown.py`, `html_to_markdown.py` |
| `metadata.ts` | `metadata.py` |
| `standardize.ts` | `standardize.py` |
| `utils.ts` | `utils.py` (normalize_text, count_words) |
| `elements/callouts.ts` | `elements/callouts.py` |
| `elements/code.ts` | `elements/code.py`, code rules in `html_to_markdown.py` |
| `elements/footnotes.ts` | `elements/footnotes.py` |
| `elements/headings.ts` | `elements/headings.py` |
| `elements/images.ts` | `elements/images.py` |
| `elements/math.base.ts`, `elements/math.core.ts` | `elements/math.py` (focused subset) |
| `removals/content-patterns.ts` | `removals/content_patterns.py` |
| `removals/hidden.ts` | `removals/hidden.py` |
| `removals/metadata-block.ts` | folded into `removals/content_patterns.py` |
| `removals/scoring.ts` | `removals/scoring.py` |
| `removals/selectors.ts` | `removals/selectors.py` |
| `removals/small-images.ts` | `removals/small_images.py` |
| `extractors/substack.ts` | `extractors/substack_note.py` (Notes, rendered/preload article bodies, and byline dates; custom-domain routing in `extractors/registry.py`) |
| `extractors/twitter.ts` | `extractors/x_tweet.py`, `extractors/x_common.py` (reference, reimplemented) |
| `extractors/x-article.ts` | `extractors/x_article.py` (reference) |
| `extractors/x-oembed.ts` | `enrichers/x_oembed.py` (reference) |

Not tracked (markitai-original, no upstream counterpart): `dom.py`,
`frontmatter.py`, `preprocess.py`, `quality.py`, `render.py`, `resolver.py`,
`sanitize.py`, `schema.py`, `semantics.py`, `thread_policy.py`, `types.py`,
`enrichers/base.py`, and the non-X extractors (`bilibili_opus`, `github_repo`,
`github_thread`, `hackernews_thread`, `reddit_post`, `steam_news`,
`youtube_page`, `registry`, `base`).

## Known divergences

Passing the parity quality tests does not mean byte-identical Markdown. Two
differences are intentional:

- Sites markitai has its own richer extractor for (Reddit, Hacker News) diverge
  from defuddle's expected output. Those fixtures score low in the benchmark
  and are held by its per-fixture guardrail floors, not by parity.
- Upstream accepts HTML on stdin; markitai takes files and URLs.

Dated audit records — which gaps were found when, and what was ported in
response — are in `process/defuddle-port-audits.md`.
