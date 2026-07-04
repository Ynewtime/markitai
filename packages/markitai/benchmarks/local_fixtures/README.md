# Local benchmark fixtures (markitai-owned)

These fixtures belong to markitai, NOT to the defuddle upstream corpus
(`tests/defuddle_fixtures/`, which is synced verbatim from defuddle — see its
`VERSION` file and sync script). Never add files there; add them here.

Layout mirrors the corpus: `fixtures/<stem>.html` + `expected/<stem>.md`.
Each HTML file starts with a `<!-- {"url": "..."} -->` comment that the
benchmark runner uses as the extraction URL.

## Self-baseline semantics

The `expected/*.md` files are **self-baselines**: they were generated from
`extract_web_content` output at the time the fixture was added (not from
defuddle ground truth). A fixture therefore scores ~100 by construction, and
any later score drop signals an extraction regression. When an intentional
improvement changes the output, regenerate the expected file deliberately
and review the diff.

The benchmark runner (`benchmarks/webextract_quality.py`) scores these under
a separate "local" aggregate (`aggregate.local_mean_score`); they are never
included in the defuddle-corpus mean, keeping that mean comparable across
baseline generations.

## Current fixtures

- `github-repo--panniantong-agent-reach` — GitHub repository home page
  (rendered README). Captured 2026-07-04 from
  https://github.com/Panniantong/Agent-Reach with non-JSON-LD `<script>`
  elements stripped (verified output-identical to the raw capture). Guards
  the `github_repo` extractor: README-only content, no file-tree/About/
  star-count chrome.
- `blog--jekyll.ynewtime.com-japanese-learning` — Jekyll blog post.
  Captured 2026-07-04. Guards bilibili iframe → link conversion,
  `og:site_name` via `name=` attribute, `<time datetime>` published date,
  and homepage-canonical suppression.
