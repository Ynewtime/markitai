# defuddle port manifest

`markitai.webextract` is a Python port of
[defuddle](https://github.com/kepano/defuddle) (TypeScript, MIT © kepano).
This manifest records which upstream sources the port tracks and the upstream
commit the parity corpus is pinned to, so corpus and algorithm resync from one
place.

Pinned upstream commit: `a4dd0041376ff7c2a5a0614ddb28996979dd7e28`

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
| `extractors/twitter.ts` | `extractors/x_tweet.py`, `extractors/x_common.py` (reference, reimplemented) |
| `extractors/x-article.ts` | `extractors/x_article.py` (reference) |
| `extractors/x-oembed.ts` | `enrichers/x_oembed.py` (reference) |

Not tracked (markitai-original, no upstream counterpart): `dom.py`,
`frontmatter.py`, `preprocess.py`, `quality.py`, `render.py`, `resolver.py`,
`sanitize.py`, `schema.py`, `semantics.py`, `thread_policy.py`, `types.py`,
`enrichers/base.py`, and the non-X extractors (`bilibili_opus`, `github_repo`,
`github_thread`, `hackernews_thread`, `reddit_post`, `steam_news`,
`youtube_page`, `registry`, `base`).

## Known gaps vs upstream 0.19.3 (audited 2026-08-25)

A resync attempt against release 0.19.3 (`a332b4d5d539066ddfe19fc4ef6f1b6ffaf914b8`)
was reverted: 6 of the 125 new upstream fixtures fail because these upstream
behaviors are not ported yet. Port them before the next corpus resync:

- article content inside `aria-hidden` overlays with dismiss links is dropped
  (upstream issue 232, fixture `issues--232-dismiss-in-hidden-content`)
- CodeMirror-rendered code blocks lose content (`codeblocks--chatgpt-codemirror`)
- mid-article image rows are misclassified as related-post cards
  (`content-patterns--multi-image-row-midarticle`)
- Substack note permalinks keep extra chrome (`general--substack-note-permalink`)
- SVG-with-external-CSS fallback text is dropped (`general--svg-external-css-fallback`)
- inline related-stories blocks are over-removed (`related--inline-related-stories-block`)
