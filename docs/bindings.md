# Native language bindings

All bindings execute the Rust core in the host process. They share a UTF-8 JSON
request and response contract. Neither an installed CLI nor a Python worker is
used for conversion. Feature availability is therefore the same as the core;
an installed binding does not add missing format, OCR, or browser capabilities.

Native media backends (local OCR, PDF page rendering, HEIF/AVIF) are
macOS-only, exactly as in the CLI; see [quick start](quickstart.md#platform-support).

## Installation

No packages are published to PyPI, npm or a Go module proxy yet. Build them
from a checkout with Rust 1.92 or later; each build compiles the Rust core
once in the `release` profile and takes several minutes.

| Language | Minimum | Build | Result |
|---|---|---|---|
| Python | CPython 3.10 (ABI3 wheel) | `maturin build --release` in `bindings/python` | `markitai-1.3.0-cp310-abi3-<platform>.whl` (about 9.8 MB on macOS arm64) |
| Node.js | 18 (Node-API 8) | `npm --prefix bindings/node run build`, then `npm pack` | `markitai-1.3.0.tgz` (about 9.6 MB on macOS arm64) |
| Go | 1.23 with cgo and a C linker | `cargo build --release -p markitai-ffi` | `target/release/libmarkitai_ffi.{dylib,so}` |

A built wheel or npm archive contains the native library for the build
machine's operating system and architecture only. Using it needs no Rust
toolchain. Release automation for other platforms is unfinished.

### Python

```sh
python3 -m venv .local/py                  # Python 3.10 or later; Apple's /usr/bin/python3 may be older
.local/py/bin/python -m pip install 'maturin>=1.9,<2'
(cd bindings/python && ../../.local/py/bin/maturin build --release --out ../../dist/python)
.local/py/bin/python -m pip install dist/python/markitai-*.whl
.local/py/bin/python -c 'import markitai; print(markitai.__version__)'
```

During development `maturin develop --release` (inside an activated
environment, from `bindings/python`) installs the package in place.

### Node.js

```sh
npm --prefix bindings/node run build       # builds the addon as bindings/node/markitai.node
mkdir -p dist/node
(cd bindings/node && npm pack --pack-destination ../../dist/node)   # markitai-1.3.0.tgz
cd /path/to/your/project
npm install /path/to/markitai-rust/dist/node/markitai-1.3.0.tgz
node -e "console.log(require('markitai').version)"
```

### Go

```sh
cargo build --release -p markitai-ffi
```

In the consuming module, point the development module name at the checkout:

```text
require markitai.local/go v0.0.0
replace markitai.local/go => /path/to/markitai-rust/bindings/go
```

```go
import markitai "markitai.local/go"
```

The default build links `target/release/libmarkitai_ffi` dynamically and embeds
that directory as the runtime search path, so the checkout's `target/release`
must remain in place; see [Go and C ABI](#go-and-c-abi) for deployment and the
self-contained static package.

## Shared contract

```json
{"source":"report.md","options":{"llm":false,"config":{}}}
```

`source` is a local path or HTTP(S) URL. Options are `output_dir`, `config`,
`llm`, `ocr`, `screenshot`, `alt`, `desc`, and `profile`. Omitted booleans inherit
configuration; `false` explicitly disables the feature. An explicit empty
`config` uses built-in defaults and bypasses configuration files. An omitted
configuration follows the core's configuration loading rules.

A modern `.numbers` directory package is a single local document, just like its
ZIP form; the extension is case-insensitive. It uses the same bounded IWA table
reader and keeps the package path as `source`. Supplying `output_dir` writes one
`<package-name>.md` file. Ordinary directories still return `is_directory`
(`IsADirectoryError` in Python); the bindings do not recursively convert them.
Legacy XML packages, Numbers OCR and complete worksheet screenshots remain
explicitly unsupported. See [Numbers reader boundaries](numbers.md).

The binding tests unfold two pinned MIT ZIP fixtures into directory packages
and compare their bodies, warnings and output files through Python/Node sync
and async calls and Go typed/JSON calls. `MARKITAI_TEST_NUMBERS_FIXTURES` selects
the fixture directory for installed-package tests; repository tests also have a
local fallback. These are container-equivalence checks, not independently
exported Apple directory-package goldens or canvas-fidelity tests.

Success is `{"ok":true,"result":{...}}`; failure is
`{"ok":false,"error":{"code":"...","message":"..."}}`, with optional
`error.usage` when model responses have already been recorded. The typed adapters
unwrap this envelope into a result or host-language error. Result fields are
`source`, `markdown`, `llm_markdown`, `frontmatter`, `output_path`,
`llm_output_path`, `assets`, `screenshots`, `images`, `usage`, `skip_reason`,
`duration`, and `warnings`. Durations are seconds. In-memory conversions have
null output paths and empty asset/screenshot path lists, keep relative
`.markitai/...` image references and write no files; the CLI's stdout image
store (`image.stdout_persist`) is not used by the bindings.

Terminal usage uses the same `cost_usd`, `requests`, `input_tokens`,
`output_tokens` and `by_model` fields as successful conversions. Python errors
raised by the adapter expose `.usage` (`ConversionUsage` or `None`), Node
`ConversionError.usage` is optional, and Go `ConversionError.Usage` is a nullable
pointer. Existing constructors, exception categories, code and message stay
compatible. An absent value differs from a recorded response with zero tokens;
neither establishes zero provider cost. `cost_usd` covers only the models in
the bundled [price catalog](pricing.md); per-model rows state whether their
cost is complete, partial or unknown.

Rust callers can use `convert_detailed` and the detailed context/publication
entrypoints to receive `ConversionFailure { error, usage }`, including final
publication errors. Existing `convert` variants still return the original
`Error` variants. Scope accounting remains document-local even with a shared
runtime. Native panics retain their existing generic boundary handling and do
not promise detailed usage. CLI/report, REST and MCP terminal diagnostics need
separate propagation and are not covered by this binding contract.

JSON serialization adds copies at the boundary. Benchmark total host-call time
separately from native extraction when evaluating this cost. The protocol is
deliberately shared so compatibility can be checked across every host before
introducing format-specific zero-copy interfaces.

## Python

Python 3.10+ loads the PyO3 extension `markitai._native`. Its ABI3 build can be
packaged for multiple supported CPython versions on the same OS/architecture.
The extension releases the GIL during conversion. `aconvert` dispatches to
Python's thread pool; cancellation stops waiting, but an already running
conversion can finish and write its requested output.

```python
import asyncio
from pathlib import Path
import markitai

out = markitai.convert(Path("report.md"), config={}, llm=False)
print(out.markdown)

async def main():
    out = await markitai.aconvert("report.md", output_dir="out", config={})
    assert isinstance(out.output_path, Path)

asyncio.run(main())
```

`convert` and `aconvert` retain the existing keyword arguments. Results remain
dataclasses, and filesystem result fields are `pathlib.Path`. Calling
`convert` from a running event loop raises `RuntimeError` as in the reference.
`enable_worker_processes` remains an importable compatibility hook; Rust does
not require Python worker processes.

`MarkitaiConfig` and the 26 nested configuration model classes are available
from `markitai.config`. They use Rust's shared schema, defaults, coercions and
construction validation. Model lists contain typed objects; preset and domain
maps remain ordinary dictionaries with typed values:

```python
cfg = markitai.MarkitaiConfig(output={"on_conflict": "overwrite"})
cfg.llm.enabled = False
out = markitai.convert("report.md", config=cfg)
```

Declared field assignment remains unvalidated, as in the original models;
assigning an unknown model attribute raises `ValueError`. Configuration is
validated again on conversion. `model_copy` retains shallow/deep behavior,
and `model_validate` accepts an existing instance without replacing it.
`model_dump` supports JSON/Python modes and common include/exclude,
exclude-unset/defaults/none filters; `model_dump_json` and JSON validation
round-trip these values. `model_json_schema` exports structural types, required
fields, bounds and defaults without the reference project's prose. Environment
resolver methods preserve explicit-reference and fallback behavior.

These classes are not Pydantic `BaseModel` subclasses. Validation raises
`ValueError`, without Pydantic's aggregated `ValidationError` details. Custom
validators, custom schema generators, `extra="allow"`, serializer warning
behavior and Pydantic internals are not reproduced. Advanced combinations of
serialization selectors remain a compatibility testing target. JSON
dictionaries and external objects exposing
`model_dump(mode="json")` also work. The wrapper never mutates supplied config.
Native `fetch_error` maps to `FetchError`; input/configuration errors map to
`ValueError`; conversion/unsupported errors map to `ConversionError` with a
`code` attribute. Specific filesystem and missing-model codes map to the
existing named exception types when supplied by the core.

The wheel contains the extension and small typed Python wrapper; see
[installation](#python). Run the package tests against an installed wheel with
private state:

```sh
MARKITAI_HOME="$PWD/.local/test-home" .local/py/bin/python -m unittest discover -s bindings/python/tests -v
```

### Import cost

`import markitai` loads the native extension, the package and `markitai.api`
(exception classes, `convert`, `aconvert`) and, from the standard library, only
`__future__`. The rest loads on first use:

| First use | Loads |
|---|---|
| `markitai.convert(...)` | `json`, `pathlib` and `dataclasses` (with `inspect`, `re`, ...) through `markitai._records`, which holds the result dataclasses |
| `markitai.ConversionOutput`, `ConversionUsage`, `OutputProfileName`, `markitai.api.ConfigModel` | `markitai._records` |
| `await markitai.aconvert(...)` | `asyncio`; a caller running the coroutine has imported it already |
| `markitai.MarkitaiConfig`, `markitai.config` | `markitai.config`: the native schema, its model classes, `copy`, `json` and `pathlib` |

`convert` rejects a running event loop by looking at `sys.modules`: a loop can
only be running once `asyncio` is imported, so a synchronous caller never
imports it. Names, signatures, `from markitai import ...` and `import *`,
`dir()`, `markitai.config` as an attribute, `__all__`, the dataclass behavior of
the results (`dataclasses.is_dataclass`, module `markitai.api`, pickling,
`typing.get_type_hints(ConversionOutput)`) and the types mypy and pyright infer
are unchanged. Imports that only annotate sit under `if TYPE_CHECKING:` with
`TYPE_CHECKING = False` defined locally, so `typing` itself stays out of the
import. `bindings/python/tests/test_import.py` pins the modules each step loads.

Differences: `typing.get_type_hints(markitai.convert)` (and any tool that
evaluates the string annotations of `convert`, `aconvert` or
`ConversionError.__init__`) raises `NameError` until a lazy name has been used,
because `Path`, `Mapping`, `Any` and the record names are then not yet module
attributes of `markitai.api`; touching `markitai.ConversionOutput` first
resolves them, and `inspect.signature` is unaffected. `markitai.api` no longer
exposes the modules it used to import incidentally (`asyncio`, `json`, ...).
mypy prints the record types as `markitai._records.ConversionOutput`; pyright
shows the same names as before. The extension still loads at import, so a
broken installation fails there rather than at the first call.

Release wheel (`maturin build --release --locked`, `MACOSX_DEPLOYMENT_TARGET=11.0`,
byte-identical extension in both builds), macOS 27.0.1 on an Apple M5 Max,
CPython 3.13.15, `.pyc` compiled at install, one fresh interpreter per sample,
medians of 101 interleaved samples. "In process" is `time.perf_counter()`
around the statements; "whole process" runs from spawn to exit (`python -c
pass` takes 9.8 ms of it):

| Statements | In process, before | After | Whole process, before | After |
|---|---|---|---|---|
| `import markitai` | 20.83 ms | 1.89 ms | 34.44 ms | 12.37 ms |
| `import markitai` and `from markitai import MarkitaiConfig` | 21.08 ms | 6.96 ms | 35.22 ms | 18.29 ms |
| `import markitai` and one Markdown `convert(..., config={}, llm=False)` | 21.53 ms | 11.69 ms | 35.65 ms | 24.24 ms |
| `import markitai` and `convert(..., config=MarkitaiConfig())` | 21.90 ms | 12.30 ms | 36.22 ms | 24.89 ms |

CPython 3.12.14 (51 samples) gives 20.58 → 1.85 ms for the import and
21.12 → 12.36 ms for import plus conversion. The deferred imports are paid by
the first call, so a process that converts once saves about 10 ms rather than
20; a process that only imports the package saves about 19 ms. Most of the
remaining 1.89 ms is loading the extension (1.5-2.0 ms in `-X importtime`
runs); the package's own modules take about 0.3 ms. Check the import with
`python -X importtime -c 'import markitai'` (the `markitai` row is cumulative)
and the median with:

```sh
for i in $(seq 101); do .local/py/bin/python -c 'import time; t = time.perf_counter(); import markitai; print((time.perf_counter() - t) * 1000)'; done | sort -n | sed -n 51p
```

The 22.82 ms `import markitai` row of the table under
[macOS system frameworks](#macos-system-frameworks) was taken before this change
and counts the package's own modules.

## Node.js

The addon targets Node-API 8 and Node.js 18+. `convert` uses a native async
worker, leaving the JavaScript event loop available. `convertSync` blocks the
calling thread. Concurrency uses the host's libuv worker pool; hosts may set
`UV_THREADPOOL_SIZE` before startup after measuring their workload.

```javascript
// An installed package; inside the checkout use require('./bindings/node').
const { convert, convertSync, ConversionError } = require('markitai');

const out = await convert('report.md', { config: {}, llm: false });
console.log(out.markdown);
const sync = convertSync('report.md', { config: {}, llm: false });
```

Both functions expose the same snake_case result and option fields as the
shared protocol. Native conversion failures reject/throw `ConversionError`
with a stable `code`. TypeScript declarations ship with the package.

Build and pack as described under [installation](#nodejs); run the tests with
private state:

```sh
MARKITAI_HOME="$PWD/.local/test-home" npm --prefix bindings/node test
```

`MARKITAI_BUILD_PROFILE=debug` or `dist` selects another Cargo profile. The
build script copies the compiled dynamic library to `markitai.node`; it is a
build-time tool, never a runtime fallback. A packed package contains a native
addon for the build machine's OS and architecture. Publish platform-specific
artifacts before promising a universal npm install. Node-API compatibility
does not remove operating-system or architecture requirements.

## Go and C ABI

The Go package uses cgo and the C header in `bindings/c/markitai.h`. Build the
native library before compiling Go, then run the package tests with private
state:

```sh
cargo build --release -p markitai-ffi
cd bindings/go
MARKITAI_HOME="$PWD/../../.local/test-home" go test -race ./...
```

The development module name is `markitai.local/go`; use a local `replace`
directive while the release repository/module path is being decided. The
default cgo linker searches `target/release` and embeds its path as an rpath on
macOS/Linux. With this default dynamic mode, deployment must package
`libmarkitai_ffi` and configure the loader path for the destination.
`CGO_LDFLAGS` can supply additional library paths;
`DYLD_LIBRARY_PATH` on macOS or `LD_LIBRARY_PATH` on Linux can select a test
build. The optional static package described below is packaged for macOS arm64
and Linux x86-64 with glibc. Windows cgo distribution and other static targets
remain unvalidated. Go consumers need cgo enabled and a C linker at build time.

```go
out, err := markitai.Convert("report.md", &markitai.Options{
    Config: map[string]any{},
    LLM: markitai.Bool(false),
})
if err != nil { return err }
fmt.Println(out.Markdown)
```

Go result optional text/path fields are pointers so null and empty string stay
distinct. `ConvertJSON` returns a raw envelope for callers that need the
language-neutral protocol. Go calls can run concurrently. Cancellation is not
currently propagated into native work.

### Static Go package

The `markitai_static` build tag selects an explicit
`native/<goos>_<goarch>/libmarkitai_ffi.a` inside the Go module. Two targets are
packaged: macOS arm64 (`darwin_arm64`) and Linux x86-64 with glibc
(`linux_amd64`). The tag cannot silently select the adjacent dynamic library.
Without that tag, the original development linkage remains unchanged. Other
operating systems and architectures explicitly reject this static mode,
including iOS and Android, which Go also builds with the `darwin` and `linux`
tags; adding a target requires its own archive, system-link parameters and real
consumer validation. Go does not tell glibc from musl, so on a musl system the
Linux archive fails to link instead of being rejected. Each package carries only
its own target's archive.

`scripts/package_go_static.py` stages a self-contained module with the Go source,
tests, C header, native archive, license texts and a hash manifest. It does not
build Rust or download a toolchain. The coordinator supplies the frozen static
archive, full Cargo metadata, compiler `native-static-libs` output and a build
record containing the exact source and input hashes. The script independently
checks the current clean source revision and all tracked bytes before and after
execution. It runs only on the native host of the package's target
(`--expected-host` names its Rust triple and defaults to the detected host), and
it rejects an old output directory. The macOS archive comes from the
[R28 builder](validation/drivers/routing-domains-static-round28/build-static.py),
the Linux one from the [guest builder](validation/drivers/linux-static-go-round50/build-static.py),
which also requires rustc's host to be `x86_64-unknown-linux-gnu`.
[run-guest.py](validation/drivers/linux-static-go-round50/run-guest.py) runs both
Linux steps from the macOS host in the OrbStack Ubuntu guest: it clones a git
bundle of the clean HEAD into a new guest directory and brings back the records
and logs as a hash-checked tar.

The staged archive is unpacked into a separate module for the existing Go race
tests. A separate consumer then builds with the static tag and is copied to a
new directory for concurrent Unicode conversions and the JSON error contract.
The driver requires only system dynamic dependencies and no rpath in the final
consumer. On macOS, `otool -L` may name only `/System/Library/Frameworks` and
`/usr/lib`, and no `LC_RPATH` may remain. On Linux the archive must hold only
relocatable x86-64 ELF objects; the consumer's `readelf -d` may name only
glibc's libraries, its loader and `libgcc_s.so.1`, with no `RPATH` or `RUNPATH`;
`ldd` must resolve each from the system library directories; and the newest
glibc symbol version in `objdump -T` is recorded as the consumer's glibc
minimum. `HOME` is retained; Markitai state, temporary files and Go caches are
private. No dynamic-library search override is inherited. Fixture paths are
provided only to the package tests, not the independent consumer.

An unpacked module currently uses the development name `markitai.local/go`:

```go
require markitai.local/go v0.0.0
replace markitai.local/go => /absolute/path/to/unpacked/markitai-go
```

Build the consuming program with `go build -tags markitai_static`. Only the final
executable needs distribution; a Markitai dynamic library, CLI or Rust toolchain
is not required at runtime. macOS system libraries/frameworks, or glibc and
libgcc_s on Linux, remain dynamic, and optional Chromium/LibreOffice backends
still require their separate runtime. The macOS linkage names Vision,
Foundation, ImageIO, CoreGraphics, CoreFoundation, Objective-C, iconv and the
system C/math libraries; the Linux linkage names libgcc_s (the unwinder),
libutil, librt, libpthread, libm, libdl and libc. The packaging driver verifies
these against the actual Rust compiler dependency note. Since glibc 2.34 most of
the Linux ones are part of libc; with Ubuntu's default `--as-needed` linking the
consumer records only libgcc_s, libm, libc and the loader.

A Linux executable requires at least the glibc whose newest symbol versions it
binds, which depends on the glibc it was linked against. Linked on Ubuntu 24.04
(glibc 2.39) that is 2.39: Rust's standard library refers to `pidfd_spawnp` and
`pidfd_getpid` weakly, but the linker records their version as a hard
requirement (the newest otherwise is 2.35). Executables linked against an older
glibc have not been tested. In the R50 run (release profile, rustc 1.98.1, Go
1.27.1, gcc 13.3, Ubuntu 24.04 amd64 under OrbStack/Rosetta) the archive was
244,496,550 bytes in 619 objects, 139,629,492 of them embedded LLVM bitcode that
linkers discard (machine code and data: 31,846,070 bytes); the package was
71,858,526 bytes and the consumer 48,723,760 bytes (34,864,832 stripped). That
archive kept one object per crate: Cargo applies the release profile's LTO only
when all of a package's crate types allow it, and `markitai-ffi` also built an
`rlib`. It now builds only `cdylib` and `staticlib` (nothing depends on it as a
Rust library and it has no doctests), so both are link-time optimized: on macOS
arm64 the archive fell from 219,756,184 to 34,462,632 bytes and
`libmarkitai_ffi.dylib` from 18,058,736 to 16,942,784 bytes; R50's experiment
copy measured the Linux archive at 53.3 MB and both platforms' consumers at
about 25–28 MB. The release link of the library takes about 100 s longer.

Go's external link keeps every section of the archive members it pulls in, and
cgo rejects `-Wl,--gc-sections` (Linux) and `-Wl,-dead_strip` (macOS) in
`#cgo LDFLAGS`. A consumer can opt in when it builds:

```sh
CGO_LDFLAGS="$(go env CGO_LDFLAGS) -Wl,--gc-sections" go build -tags markitai_static  # Linux
CGO_LDFLAGS="$(go env CGO_LDFLAGS) -Wl,-dead_strip" go build -tags markitai_static    # macOS
```

Before the library was link-time optimized, this shrank the Linux consumer from 48,723,760 to 31,877,240 bytes
(24,012,912 with `-ldflags=-s -w`, against 34,864,832) and the macOS arm64 one
from 47,990,834 to 30,338,210 bytes. Both produced the same conversions and JSON
error, the installed packages' race tests passed with the flag, and the Linux
executable kept Go's build ID and build information. With the optimized archive the flag
adds little.

The package's `licenses.json` records original source paths and byte hashes for
collected texts, including separate Rust toolchain notices. Its Cargo closure
conservatively includes build/dev/other-target dependencies; it is not a precise
list of code reachable in the final binary. Missing texts are reported explicitly
as `unresolved`, and a successful technical consumer test does not complete the
redistribution review. This workflow's host results must be recorded separately;
the earlier dynamic-binding checkpoints do not establish static-link success or
minimum-macOS or minimum-glibc compatibility.

The C ABI is version 1:

1. Call `markitai_convert_json(request, len)` with valid readable bytes that
   remain alive through the call. The input is borrowed, never freed by Rust.
   Requests larger than 64 MiB, malformed JSON, or invalid UTF-8 receive error
   envelopes. Null with zero length is accepted as an empty, invalid request.
2. The returned `MarkitaiBuffer` owns exactly `len` UTF-8 bytes; there is no NUL
   terminator. Copy or consume them before releasing the buffer.
3. Pass the address of that same buffer to `markitai_buffer_free`. It clears
   its pointer and length, making a repeated free on that struct harmless.
   Never copy the ownership handle, change its fields, free it with C `free`,
   or retain a data pointer after release.

Go passes only byte storage, which contains no Go pointers, during the cgo
call. Rust never retains it. Go copies the response into Go-managed memory
and defers native freeing on every return path. Native panics during
conversion are caught at each language boundary. Invalid foreign pointers,
double frees of copied handles, allocation failure, and builds configured to
abort on panic remain outside this recoverable contract.

## macOS system frameworks

The Node addon, the Python extension and `libmarkitai_ffi.dylib` link
CoreFoundation, Foundation, CoreGraphics, ImageIO and Vision delay-initialized,
from the same build script and linker probe as the CLI. From macOS 15, dyld
maps them when the binding loads but initializes them on first use: OCR,
HEIF/AVIF decoding and PDF rasterization open their framework before they need
it. Earlier systems initialize them at load as before, and a linker without
`-delay_framework` keeps ordinary links. A conversion that uses none of these
backends (HTML, Office, text PDF, Markdown) never initializes them.

What a host saves depends on what it links itself. Node and Python link
CoreFoundation, which initializes Foundation, CoreGraphics and ImageIO at
launch, so the binding postpones Vision and the images that only Vision brings.
A Go program linking only the dynamic library postpones all five. Release
builds, macOS 27.0.1 on an Apple M5 Max, medians of 101 interleaved runs with
byte-identical copies of the old build as a noise check (within 0.1 ms); images
initialized are those `DYLD_PRINT_LIBRARIES` reports mapped and not postponed:

| Host | Measured | Before | After | Images initialized at load |
|---|---|---|---|---|
| Node.js 24.21 | `require()` of the package, in process | 4.88 ms | 3.58 ms | 510 → 390 (Node alone: 389) |
| CPython 3.13.15 | loading `markitai._native`, in process | 2.30 ms | 1.54 ms | 511 → 393 (Python alone: 392) |
| CPython 3.13.15 | `import markitai`, in process | 23.46 ms | 22.82 ms | as above; 1.89 ms after deferred imports (see [Import cost](#import-cost)) |
| Go 1.27.1 | process printing `Version()`, dynamic library | 7.99 ms | 5.68 ms | 510 → 85 |

The first OCR, HEIF/AVIF or PDF page in a process pays the postponed
initialization instead; conversions produce the same results. The static Go
package keeps ordinary links: cgo rejects `-Wl,-delay_framework` in `#cgo
LDFLAGS` without `CGO_LDFLAGS_ALLOW`, and requesting it from inside the archive
(an object's linker option) marks the frameworks without the call stubs that
initialize a framework on its first C call. A consumer can opt in when it
builds; flags from the environment are not checked against cgo's list:

```sh
CGO_LDFLAGS="$(go env CGO_LDFLAGS) -Wl,-delay_framework,Vision -Wl,-delay_framework,Foundation \
  -Wl,-delay_framework,ImageIO -Wl,-delay_framework,CoreGraphics -Wl,-delay_framework,CoreFoundation" \
  go build -tags markitai_static
```

Keep Go's default flags (`$(go env CGO_LDFLAGS)`, normally `-O2 -g`) in the
value, which otherwise replaces them. ld then notes that CoreGraphics has weak
definitions; dyld still postpones it. In the measurement above, such a static
program started in 4.91 ms instead of 7.19 ms (509 → 84 images) and its
executable grew by 19,952 bytes.

## Verification and maintenance

`cargo test -p markitai-ffi` exercises null pointers and oversized lengths,
invalid UTF-8, repeated Unicode conversions, and explicit release. Python,
Node, and Go integration suites load compiled native artifacts and cover
Unicode, concurrent repeated calls, output files, and structured failures.
Local HTTP-server tests also verify that Python's GIL and Node's event loop
remain available while native work runs. On macOS 15 and later each suite
loads the binding in a child process under `DYLD_PRINT_LIBRARIES`, checks that
loading postpones the media frameworks the host has not initialized itself and
initializes no postponed image, then renders a generated PDF page; the Go test
covers the dynamic library, not the static package.
Conversion tests use explicit empty config or test-owned config objects.
Terminal-usage tests use loopback HTTP fixtures with fake credentials, including
concurrent paid failures and zero-token responses. They do not contact real model
providers or read user configuration. Old-producer envelopes are tested separately
from actual compiled-native error paths. Run each suite against newly rebuilt
artifacts after core or ABI changes.

Implementation references: [PyO3 function and module interface](https://pyo3.rs/v0.27.2/module.html),
[PyO3 GIL release](https://pyo3.rs/v0.27.2/parallelism.html), and
[NAPI-RS native async tasks](https://napi.rs/docs/concepts/async-task).

## Verified evidence

The package driver in [native CI](ci.md) builds all three bindings from a clean
checkout, installs them into private consumers and runs their test suites. The
latest recorded run, [delivery R44](validation/delivery-round44.md) (source
`b961344`), passed on macOS arm64 and on Ubuntu amd64 under OrbStack/Rosetta:
installed Node 7/7, Python 20 and Go race tests, plus the macOS static Go
package (844 installed files, 24 relocated concurrent conversions). Toolchains
were Rust 1.98.1, Python 3.13, Node.js 24 and Go 1.27; the minimum versions in
the installation table come from the package manifests and are not separately
tested. The Linux x86-64 static Go package was first run in R50, outside a
delivery round (see [Static Go package](#static-go-package)). Windows, physical
Intel hosts and other static Go targets have not been run. Earlier rounds,
including the first binding checkpoints and their package sizes, are kept in
the [validation records](validation/README.md)
(for example [artifacts round 3](validation/artifacts-round3.json) and
[R28 static Go](validation/routing-domains-static-round28.md)).
