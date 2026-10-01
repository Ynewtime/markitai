# Validation records

最新交付：[R46](delivery-round46.md)（macOS arm64 与 Linux amd64 打包、三绑定），对应的改动与语料对照见
[R47](parallel-round47.md) 与 [R48](parallel-round48.md) 并行改进，冻结性能见[性能记录](performance-round36.md)（`all-r13`）。当前状态与剩余
事项见 [STATUS](../STATUS.md) 与 [剩余工作](../remaining-work.md)。

An implementation checkpoint is reproducible only when it records the command,
source revision, test inputs, build profile, and platform. Passing new unit
tests establishes specific contracts; it does not imply reference parity. Each
record below stays tied to its own source revision and artifacts.

Earlier checkpoints, oldest at the bottom of each series: delivery rounds
`delivery-round36.md` through `delivery-round44.md`, quality records
(`pdf-quality-round41.md`, `office-quality-round42.md`, `html-quality-round43.md`,
`html-quality-round44.md`, `ocr-quality-round45.md`, `epub-quality-round46.md`),
and, before them, [round twenty-four](structured-media-round24.md) and
[round twenty-three](vision-auth-cli-round23.md). Platform records: Linux
`platform-linux-round*`, Windows type checks `windows-check-round38.md`, and the
x86-64 macOS build under Rosetta 2 in [macos-x86_64-rosetta.md](macos-x86_64-rosetta.md).

[Round twenty-two](media-llm-round22.md) retains Office/Numbers/text processing
acceptance; [round twenty-one](service-ui-round21.md) retains actual web workspace
and browser validation. Historical evidence remains tied to its own source and
artifact. Remote CI, exact OCR quality and the complete migration remain open.

## Local checks

Run `scripts/check.sh` for Rust formatting, workspace tests, lint, and audit
runner tests. Run each native binding's integration tests against built
artifacts as described in [bindings](../bindings.md). These use local fixtures
and HTTP mock servers; real-provider tests are separate and opt-in.

## Differential audit

```sh
python3 scripts/audit_formats.py \
  --reference /path/to/reference-markitai \
  --library target/release/libmarkitai_ffi.dylib \
  --output .local/audits/first-run
```

On Linux select `.so`; on Windows select `.dll`. The reference must have its
own installed `.venv`, or pass `--reference-python`. The runner snapshots the
native library and compares Markdown, non-clock metadata, and extracted asset
hashes. It includes unimplemented source formats in the denominator. Token
recall is diagnostic, never a substitute for the exact-content checks.

## Recorded format gates

- [Initial baseline](formats-baseline.md): 0/24 exact passes.
- [Intermediate recovery](formats-recovery.md): 13/24 exact passes.
- [Second recovery](formats-recovery-r4.md): 15/24 exact passes, with all remaining differences listed.
- [MSG/HTML recovery](formats-recovery-r5.md): 17/24 exact passes; four differences and three expected image-only rejections.
- [Final round-three format check](formats-recovery-r6.md): the same 24 case outcomes after the CSS visibility fix.
- [Round-four format check](formats-recovery-r7.md): 17/24 exact passes; PPTX text recovered, image differences retained.
- [First round-five format check](formats-recovery-r8.md): historical 979d205 snapshot, unchanged 24 outcomes.
- [Final round-five format check](formats-recovery-r9.md): 17/24 exact passes; all r8 outcomes retained.
- [Full HTML corpus](html-corpus.md): 43/209 exact local API passes, 166 differences and zero conversion errors.

Each record identifies a frozen artifact; later fixes do not retroactively change
its measurements. Full JSON reports retain the complete fixture denominator.

## Round-fourteen evidence

- [HTML/PDF fidelity, release and measurements](native-quality-round14.md):
  567 distinct Rust tests, 209 HTML and 24 format cases; 14/16 exact curated code
  bodies, selected PDF table/page recovery, and explicitly slower ordinary-article
  medians. Source `e4c63f7`; CLI 22.58 MiB.
- [Hidden streamed-content gap](html-streaming-gap-round14.md) distinguishes a
  resolved timeout from the remaining missing article body.

## Round-fifteen evidence

- [Extraction cost and streamed-content recovery](native-performance-round15.md):
  585 distinct Rust tests; 207 unchanged HTML and 24 unchanged format responses;
  two recovered streamed articles; exact-output Rust and reference-Python timings.
- The largest tested code container is 43.17 times faster than round fourteen;
  broader speed, memory and size goals remain open. Source `2c53fe3`, CLI 22.61 MiB.
- [PDF page-prefix loss](pdf-page-prefix-round15.json) records a separately
  reproduced baseline defect rather than counting lost text as a performance gain.

## Round-sixteen evidence

- [Native OCR, browser and PDF correctness](native-backends-round16.md): source
  `f164978`, 617 distinct workspace tests, 55 vendored tests, 16 real backend cases,
  complete screenshot history, release size and scoped timing.
- CLI 22.94 MiB; simple exact Python comparisons are 17.67–27.26 times faster.
  Prior-Rust wall time increases 1.06–9.48%; OCR first-use latency remains open.

## Initial CLI benchmark

```sh
python3 scripts/benchmark_cli.py \
  --reference /path/to/reference-markitai \
  --binary target/release/markitai \
  --output .local/benchmarks/cli.json
```

Each sample starts a fresh process with explicit isolated configuration.
The filesystem cache is warm. Output differences are recorded; faster timing
with different output cannot establish equivalent-quality performance. Broad
format throughput, cold caches, peak RSS, repeated host calls, and platform
distribution matrices require additional measured records.

The updated CLI harness disables the reference's dotenv autoload, alternates
engine order, checks every stdout against its warmup and withholds time ratios
when conversion output differs. It preserves earlier measurements in separate
files. This isolates the exercised configuration paths; it is not an OS sandbox.

## Round-three evidence

- [Artifact identities](artifacts-round3.json) for the CLI, C library, installed
  Python wheel and npm archive at source `21bf8f5`.
- [CLI measurements](performance-round3.md), including exact-output gates and
  current isolation limitations.
- [Repeated API measurement method](api-benchmark-method.md) and
  [results](performance-api-round3.md). The native C ABI uses a pre-encoded
  request; these are not measurements of the new Python binding's full call cost.

## Round-four evidence

- [Source, rebuilt artifacts and performance](performance-round4.md): 189 Rust
  test executions, installed host packages, exact-output CLI/C-ABI measurements
  and explicit boundaries; source `5280fd3`.
- [Artifact identities](artifacts-round4.json), [CLI samples](cli-round4.json)
  and [C-ABI samples](api-round4.json) are unchanged copies of the local records.
- Persistent document caching is tested with local request counters. At this
  historical checkpoint, page caching was still planned.

## Round-five evidence

- [Source, artifacts and performance](performance-round5.md): 240 Rust executions,
  installed bindings, synthetic CLI/C-ABI measurements and explicit limitations;
  frozen source `0022e09`.
- [Artifact identities](artifacts-round5-final.json),
  [CLI samples](cli-round5-final.json) and [C-ABI samples](api-round5-final.json)
  are unchanged copies of the original records.
- [Static-page fetching](../fetch.md) documents conditional requests, TTL,
  bypass-refresh behavior and admission boundaries. Cache request avoidance is
  established with loopback counters, not a provider latency/cost benchmark.
- [Independent HTML review](html-review-round5-final.json) passes 11 cases;
  [full-corpus evidence](html-corpus.md) retains the shared-reference whitespace
  defect as an intentional exact-output difference after the native fix.
- The [first round-five checkpoint](round5-first.md), artifacts and measurements
  remain historical evidence. Its extra parity pass was not evidence of better
  extraction quality; later review found and corrected two structural defects.
- The [performance plan](../performance-plan.md) covers profile selection,
  complete binding costs, retained binary code/data and sustained memory checks.

## Round-six build-profile evidence

- [Profile tradeoff and reproduction](profile-round6.md): the existing dist CLI
  is 39.19% smaller, but measured PDF/PPTX C-ABI conversions take 6.58×/8.22× as
  long. Release remains the default. Builds use the same clean `e0cf110` source.
- [Format equivalence](profile-formats-round6.md) and
  [HTML/full-envelope equivalence](profile-html-round6.md) preserve the existing
  reference gaps while verifying 233 complete native response pairs.
- [Raw measurements](profile-measurements-round6.json) retain all CLI samples,
  worker medians, RSS peaks, input identities and isolation observations.
  Whole host-wrapper costs and long-lived memory behavior remain open.
- Eighteen Python harness tests pass, including 14 new regression tests against
  damaged or incomplete comparison evidence. These are harness checks, separate
  from the frozen artifact's Rust tests and native conversion gates.
- [Full-LTO speed-optimized follow-up](profile-fat-speed.md): dist-opt3 at clean
  `ab6b141` is a 6.14%-smaller CLI candidate with broadly similar measured timings
  on six inputs. New paired measurements and all equivalence gates are recorded;
  candidate package validation remains open and the default is unchanged.

## Round-seven report implementation

- [Report contracts and differential method](reports-round7.md): 296 Rust test
  executions, 23 harness checks and four matching release report pairs from
  clean `0ab59a0`. Rebuilt installed Python/Node packages and Go race checks pass;
  history/resume and complete runtime parity remain unfinished.
- [Build identities](reports-round7-build.json), [reference report comparison](reports-round7-r4.json)
  and [package/log records](reports-round7-artifacts.json) preserve exact evidence.

## Round-eight recovery storage

- [Storage validation](state-round8.md): 410 Rust executions /330 distinct,
  26 harness tests and 13 matching authored legacy state/replay pairs at clean
  `0b38e00`; [build](state-round8-build.json) and
  [differential](state-round8-r2.json) records retain exact identities.
  That historical store-only checkpoint predates CLI resume integration.

## Round-nine recovery scheduling

- [Scheduling and publication validation](recovery-round9.md): 515 Rust executions /
  395 distinct tests, 26 harness tests, 19 real CLI recovery cases, private
  member/receipt protocols and deterministic shared-asset process races. Clean
  `361c924` release passes 4/4 reference report pairs, two release recovery cases
  and rebuilt installed Python/Node/Go checks; all 16 prior archives are retained.
  [Build](recovery-round9-build.json), [gate](recovery-round9-gate.json),
  [reports](recovery-round9-reports.json), [release recovery](recovery-round9-release.json)
  and [packages](recovery-round9-packages.json) retain unmodified records. See the
  narrative for Unix scope, signal limits, artifact size and remaining boundaries.

## Round-ten optional history export

- [History validation](history-round10.md): 590 Rust executions /447 distinct,
  26 harness tests and 14 Unix history process cases. Clean `1686ed4` release
  passes 5/5 authored archive contract pairs (4/5 strict; one pinned parser-error
  wording difference), all 266 assertions and four ordinary report pairs.
  Rebuilt installed Python/Node packages and Go race checks pass; all 18 prior
  archives remain unchanged. The CLI is 17.75 MiB, +0.90% from round nine.
- [Build](history-round10-build.json), [gate](history-round10-gate.json),
  [history pairs](history-round10-release.json), [report pairs](history-round10-reports.json),
  [packages](history-round10-packages.json) and [final identity check](history-round10-final-check.json)
  preserve exact records. [Executed drivers](drivers/round10/README.md) retain the
  historical commands and private-state protections. Remaining asset syntax,
  consumer, fault-matrix and platform acceptance is explicit in the narrative.

## Round-eleven media destinations

- [Source validation](html-media-round11.md): 608 Rust executions /465 distinct,
  26 harness tests and 15 history process cases. Multiple media attributes and
  srcset candidates now share exact destination handling; image preparation and
  publication use original-path maps to prevent cascaded renames or filtering.
  The [gate record](html-media-round11-gate.json) retains unchanged source hashes.
  Clean `18680d9` [release](html-media-round11-build.json) passes its
  [media archive case](html-media-round11-release.json) and
  [installed packages](html-media-round11-packages.json).
- [Paired CLI samples](html-media-round11-benchmark-r2.json) retain 72 processes,
  exact output equality and 56 timed runs. The 250/1000-attachment fixtures have
  18.67%/39.54% lower median elapsed time than the previous Rust release on this
  host; text/50 sample ranges overlap. The [failed correctness attempt](html-media-round11-benchmark-r1.json),
  [final identities](html-media-round11-final-check.json) and
  [executed drivers](drivers/round11/README.md) preserve the experiment's boundaries.

## Round-twelve CSS resources

- [CSS acceptance](css-round12.md) records 625 Rust executions /482 distinct,
  26 harness tests and independent parser reviews. The [gate record](css-round12-gate.json)
  retains 96 unchanged source identities. Clean `8204ebe` has
  [release artifacts](css-round12-build.json), [installed packages](css-round12-packages.json)
  and an exact [CSS archive](css-round12-release.json).
- [Browser consumption](css-round12-browser.json) verifies two archived pages,
  renamed images/imports and encoded filenames after deleting live fixtures.
  [Final identities](css-round12-final-check.json) and
  [executed drivers](drivers/round12/README.md) retain the scope and provenance.

## Round-thirteen native services and SVG

- [Integrated acceptance](native-services-round13.md): 663 Rust executions /
  516 distinct tests, 26 harness checks, shared LLM concurrency, bounded SVG,
  eight REST and seven MCP process tests. The [final gate](native-services-round13-gate.json)
  preserves source identity and the [failed first gate](native-services-round13-gate-r1.json)
  retains its diagnostics boundary.
- Clean `2af556a` [release evidence](native-services-round13-release.json)
  records the 22.42 MiB CLI and actual REST/MCP smoke tests. The increase from
  round twelve is explicit; no new performance or installed-binding claim is
  made. [Executed drivers](drivers/round13/README.md) retain exact provenance.

## Round-nineteen evidence

- [Native images and persistent service operations](image-workflows-round19.md): complete TIFF pages, image analysis/shared budgets and metadata publication, REST retry/enhance/deletion, source gate and clean release workflows.
