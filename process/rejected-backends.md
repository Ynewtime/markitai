# Backends evaluated and not adopted

Alternatives that were tried and dropped, so the same ground is not covered
twice. Nothing here is part of the product; the backends markitai actually
ships are in the [CLI reference](https://markitai.dev/guide/cli).

## 2026-09 — Calamine spreadsheet reader, PDF layout model

Both were prototyped during the conversion-performance work that
`scripts/benchmarks/` was built for, as alternatives to the shipped
spreadsheet and PDF paths. Neither was adopted, and the harnesses were removed
from the maintained tool set in commit `a113218` rather than left behind as
dead scripts.

The measured outcome of that work — including what the shipped paths do
achieve — is
[`scripts/benchmarks/results/2026-09-14.json`](../scripts/benchmarks/results/2026-09-14.json).
