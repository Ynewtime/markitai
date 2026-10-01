# Bun reference: practical implications for Markitai

> Historical design review. The profile described below has since changed:
> `release` now uses fat LTO with size-optimized dependencies and a
> speed-optimized conversion path ([architecture](architecture.md),
> [measurement](validation/binary-size-round39.md)). Current performance results
> are listed in the [control center](CONTROL.md).

Design review, 2026-09-28. This plan adds no new performance measurements.
Historical artifact figures below belong to `979d205`, not to a later build.
The subsequent [round-six experiment](validation/profile-round6.md) supplies the
release/dist evidence: dist reduces CLI bytes by 39.19% but takes 6.58×/8.22× as
long for the measured PDF/PPTX C-ABI calls. The separate
[dist-opt3 follow-up](validation/profile-fat-speed.md) reduces CLI bytes by
6.14% with broadly similar timings on six inputs. Release remains the default;
complete candidate package validation and full-wrapper measurements remain open.

## What the article actually reports

[Bun's official article](https://bun.com/blog/bun-in-rust), dated July 8, 2026,
centres the rewrite on stability and preserving behavior. It reports roughly
2–5% performance improvements on specific Linux workloads. Initial binary
savings were followed by identical-code folding and ICU data changes; the
combined Linux/Windows reduction was approximately 20%. It also demonstrates
memory growth levelling off during repeated in-process builds after fixing
leaks. These are Bun measurements, not predictions for Markitai. Its
language-independent tests, independent review and coordinated Git/build work
are reusable process ideas; its mechanical port is not the architecture
requested for this project.

## Recommendations inferred from the current project

1. **Measure the profile tradeoff before selecting a distribution default.**
   `Cargo.toml` currently uses release `opt-level=3` (Cargo default), thin LTO,
   one codegen unit and stripping; `dist` changes this to `z` and fat LTO.
   Compare both on the same source, feature set and passing corpus, measuring
   uncompressed/compressed size, startup, sustained conversion time and RSS.
   Include `s` as a candidate only in an explicitly scheduled experiment.
   Cargo warns that size optimization is not guaranteed to produce the
   smallest artifact and that `z` disables loop vectorization.
   [Cargo profiles](https://doc.rust-lang.org/cargo/reference/profiles.html).

2. **Locate retained code/data before trimming functionality.** The current
   core includes native document parsers, image codecs, TLS and bundled SQLite.
   A link map or binary-symbol analysis should establish which parts dominate
   before changing features. The inspected lockfile has one version each of
   image, zip, quick-xml, png, tiff and encoding_rs, so obvious duplicate versions
   of those crates are not an established source of savings. ICU Rust data
   crates in the lockfile are not evidence that Bun's C ICU treatment applies.
   Do not remove required formats or substitute browser/OCR downloads merely
   to improve the single-binary headline.

3. **Keep panic recovery at host boundaries.** The C, Python and Node adapters
   catch unwinding panics. A workspace-wide `panic=abort` size experiment would
   bypass this recovery and terminate the host process. Retain unwind for
   bindings; any CLI-specific alternative requires its own build and behavioral
   verification. The existing release/dist profiles do not enable abort.
   [Cargo panic behavior](https://doc.rust-lang.org/cargo/reference/profiles.html#panic).

4. **Measure complete bindings before optimizing their protocol.** The current
   benchmark pre-encodes C-ABI JSON; it excludes typed Python/Node/Go wrapper
   encoding, result construction and scheduling costs. Rust serializes an
   owned response, C callers free it explicitly, and Go copies it into host
   memory. Measure each public wrapper separately from core extraction. Only
   if profiles show a material cost should an additive batch/session or typed
   internal path be considered, retaining existing public interfaces.

5. **Add long-lived memory evidence independently of peak RSS.** Fresh workers
   and the C ABI's 64-call ownership test do not establish leak-free sustained
   use. A later controlled test should repeat successful and failing PDF,
   Office, HTML and FFI conversions thousands of times, track live allocation
   or a justified steady-state plateau, and account for allocator retention.
   Parser/ABI fuzzing and sanitizer runs complement output comparisons. Current
   small synthetic measurements cannot establish these properties.

6. **Do not label ordinary Cargo LTO as cross-language LTO.** Bun's technique
   needs compatible LLVM-produced objects and linker-plugin setup. The local
   manifest alone does not establish that bundled SQLite participates. It
   cannot inline across a dynamically loaded Python/Node/Go runtime boundary.
   Investigate C/Rust LTO only after a native hot-path profile justifies its
   build and distribution complexity.
   [rustc linker-plugin LTO](https://doc.rust-lang.org/rustc/linker-plugin-lto.html).

## Evidence limits

The historical round-five artifact record reports a 17,833,936-byte CLI
(8,545,407 gzip bytes), a 16,507,008-byte FFI library, and an 8,504,775-byte wheel.
These measure different deliverables. They do not prove equivalent-feature
installed-size savings against the Python runtime and dependencies. macOS arm64
is the measured platform; Linux/Windows packaging and comparative profile
results remain unestablished here. No new percentages are claimed by this review.

Local evidence: [workspace profiles](../Cargo.toml),
[core dependencies](../crates/markitai-core/Cargo.toml),
[C ABI](../crates/markitai-ffi/src/lib.rs),
[Python adapter](../crates/markitai-python/src/lib.rs),
[Node adapter](../crates/markitai-node/src/lib.rs),
[Go wrapper](../bindings/go/markitai.go),
[historical artifacts](validation/artifacts-round5-first.json) and
[API measurement contract](validation/api-benchmark-method.md).
