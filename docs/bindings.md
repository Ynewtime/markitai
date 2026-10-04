# Native language bindings

All bindings execute the Rust core in the host process. They share a UTF-8 JSON
request and response contract. Neither an installed CLI nor a Python worker is
used for conversion. Feature availability is therefore the same as the core;
an installed binding does not add missing format, OCR, or browser capabilities.

Local OCR uses Vision on macOS and Paddle on Windows/Linux; Paddle models
are prepared separately. PDF page rendering uses CoreGraphics on macOS by
default and built-in hayro on Windows/Linux. HEIF/AVIF decoding remains
macOS-only. Optional browser and Office backends still need their external
applications; see [quick start](quickstart.md#platform-support).

## Installation

This development version is distributed through platform artifacts or a local
source build, not a universal PyPI/npm/Go install. Build bindings
from a checkout with Rust 1.92 or later (development uses 1.99.0); each build
compiles the Rust core once in the `release` profile and takes several minutes.

| Language | Minimum | Build | Result |
|---|---|---|---|
| Python | CPython 3.10 (ABI3 wheel) | `maturin build --release --locked` in `bindings/python` | `markitai-1.3.0.dev0-cp310-abi3-<platform>.whl` |
| Node.js | 18 (Node-API 8) | `npm --prefix bindings/node run build`, then `npm pack` | `markitai-1.3.0-dev.0.tgz` |
| Go (macOS/Linux) | 1.23 with cgo and a C linker | `cargo build --release --locked -p markitai-ffi` | `target/release/libmarkitai_ffi.{dylib,so}` |

Building needs a native C/C++ toolchain/linker in addition to Rust; see
[source prerequisites](quickstart.md#build-from-source). The FFI also builds a
Windows `markitai_ffi.dll`, but that does not establish a usable Windows Go/cgo
package.

A built wheel or npm archive contains the native library for the build
machine's operating system and architecture only. Match the Python or Node
process architecture as well as the OS (for example, an x64 Python process on
ARM64 Windows needs an x64 extension). Using it needs no Rust
toolchain. Native package drivers build and exercise platform-specific artifacts;
see [CI](ci.md) for scope. Packages from different targets may have the same
internal npm name, so keep their external archive names or directories distinct.

### Python

```sh
python3 -m venv .local/py                  # Python 3.10 or later; Apple's /usr/bin/python3 may be older
.local/py/bin/python -m pip install 'maturin>=1.9,<2'
(cd bindings/python && ../../.local/py/bin/maturin build --release --locked --out ../../dist/python)
.local/py/bin/python -m pip install /path/to/the-matching-markitai.whl
.local/py/bin/python -c 'import markitai; print(markitai.__version__)'
```

Select the wheel produced above for your interpreter/target, rather than
installing every wheel left in a shared output directory. For an already supplied
Windows wheel, no Rust build is needed:

```powershell
py -3 -m venv .local\py
& .local\py\Scripts\python.exe -m pip install C:\path\to\the-matching-markitai.whl
& .local\py\Scripts\python.exe -c "import markitai; print(markitai.__version__)"
```

During development `maturin develop --release --locked` (inside an activated
environment, from `bindings/python`) installs the package in place.

### Node.js

```sh
npm --prefix bindings/node run build       # builds the addon as bindings/node/markitai.node
mkdir -p dist/node
(cd bindings/node && npm pack --pack-destination ../../dist/node)   # markitai-1.3.0-dev.0.tgz
cd /path/to/your/project
npm install /path/to/markitai-rust/dist/node/markitai-1.3.0-dev.0.tgz
node -e "console.log(require('markitai').version)"
```

### Go

```sh
cargo build --release --locked -p markitai-ffi
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

## Version identifiers and adapter differences

The development engine identifies itself as `1.3.0-dev`. Python's
`markitai.__version__`, Node's exported `version`, Go's `Version()` and the C
version function report that engine identifier. Distribution metadata uses the
package manager's spelling: Python `1.3.0.dev0` and npm `1.3.0-dev.0`. These
strings intentionally differ; compare installed distributions using
`importlib.metadata.version("markitai")` or npm's package metadata, and use the
engine identifier for diagnostics. Do not compare these strings for literal
equality or assume their prerelease ordering is interchangeable.

The conversion wire format is shared; the convenience adapters have these
language-specific contracts:

| Concern | Python | Node.js | Go |
|---|---|---|---|
| Configuration helpers | `markitai.config` exposes typed configuration models; the private native extension supplies JSON normalization/schema operations | Pass the shared `config` object; no configuration/schema helper export | Pass `Options.Config`; no configuration/schema helper export |
| Failure types | Native codes map to `ValueError`, `FileNotFoundError`, `IsADirectoryError`, `OSError`, `FetchError`, `NoModelConfiguredError` or `ConversionError` | Native failures use `ConversionError`; invalid JavaScript argument shapes use `TypeError` | Native failures use `*ConversionError`; JSON encoding and ABI failures use ordinary Go errors |
| Omitted options | Keyword options default to `None`; `config=None` loads configured defaults, `{}` selects built-in defaults | Omitted/`undefined` options become `{}`; explicit `null` is rejected | A nil `*Options` is accepted and uses defaults |
| Blocking and async use | `convert()` refuses an active asyncio loop; use `await aconvert()` | `convertSync()` blocks without an event-loop guard; use `await convert()` to leave the loop available | `Convert()` blocks its goroutine; concurrent calls are supported |
| Installation source | Build/install a platform wheel | Build/install a platform npm archive | `markitai.local/go` requires a local `replace`; `go get` alone cannot supply the untracked native library |

These are adapter differences, not different document-format implementations.
The Python `config_json` operation is an implementation interface of `_native`,
not an equally exposed Node/Go API. See each language section for deployment and
concurrency limitations.

## Shared contract

```json
{"source":"report.md","options":{"llm":false,"config":{}}}
```

`source` is a local path or HTTP(S) URL. Options are `output_dir`, `config`,
`llm`, `ocr`, `screenshot`, `alt`, `desc`, and `profile`. Omitted booleans inherit
configuration; `false` explicitly disables the feature. An explicit empty
`config` uses built-in defaults and bypasses configuration files. An omitted
configuration follows the core's configuration loading rules. `{}` does not
block networking: a URL input still downloads the source, and explicit LLM or
remote-backend settings still apply. For offline work, supply local files, use
`config={}` and `llm=False` (Python), or the corresponding false value in your
language, and prepare optional local models/applications first.

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
null output paths and empty asset/screenshot path lists and keep relative
`.markitai/...` image references. They do not publish document/image outputs;
configuration, caches and optional backends can still use local state or
temporary files. The CLI stdout image store (`image.stdout_persist`) is not
used by the bindings.

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
not promise detailed usage. CLI/report, REST and MCP also expose their own terminal diagnostics; their
public envelopes differ and are documented in the corresponding interface
guides. This section defines only the binding error contract.

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
private state (create those directories first; preserve toolchain/cache paths
explicitly when changing `HOME`):

```sh
HOME="$PWD/.local/test-user-home" MARKITAI_HOME="$PWD/.local/test-home" .local/py/bin/python -m unittest discover -s bindings/python/tests -v
```

### Import cost

The native extension loads on import, while result records, configuration
models and asyncio support load on first use. A broken native installation
therefore fails on import. Measure both startup and the first conversion for
your workload; deferred imports move work to that first call.

`inspect.signature` works normally. Code evaluating annotations with
`typing.get_type_hints(markitai.convert)` should first access
`markitai.ConversionOutput` to resolve lazy record/type names. These classes
remain dataclasses and keep their public names and pickling behavior.

## Node.js

The addon targets Node-API 8 and Node.js 18+. `convert` uses a native async
worker, leaving the JavaScript event loop available. `convertSync` blocks the
calling thread. Concurrency uses the host's libuv worker pool; hosts may set
`UV_THREADPOOL_SIZE` before startup after measuring their workload.

```javascript
// An installed package; inside the checkout use require('./bindings/node').
const { convert, convertSync, ConversionError } = require('markitai');

async function main() {
  const out = await convert('report.md', { config: {}, llm: false });
  console.log(out.markdown);
  const sync = convertSync('report.md', { config: {}, llm: false });
  console.log(sync.markdown);
}
main().catch(error => { console.error(error); process.exitCode = 1; });
```

Both functions expose the same snake_case result and option fields as the
shared protocol. Native conversion failures reject/throw `ConversionError`
with a stable `code`. TypeScript declarations ship with the package.

Build and pack as described under [installation](#nodejs); run the tests with
private state (create those directories first; preserve toolchain/cache paths
explicitly when changing `HOME`):

```sh
HOME="$PWD/.local/test-user-home" MARKITAI_HOME="$PWD/.local/test-home" npm --prefix bindings/node test
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
cargo build --release --locked -p markitai-ffi
cd bindings/go
HOME="$PWD/../../.local/test-user-home" MARKITAI_HOME="$PWD/../../.local/test-home" go test -race ./...
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

`scripts/package_go_static.py` packages a supplied native archive, compiler
linkage note, Cargo metadata and source build record. It runs on the matching
host, rejects dirty source and existing output directories, installs the module
into a separate consumer, and checks race tests, relocation, concurrent calls
and system-only dynamic dependencies. It does not build Rust or download a
toolchain. See `python scripts/package_go_static.py --help` for required inputs.
Use a verified archive and its original build records together; a renamed
archive from another target is not interchangeable.

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

The Linux package requires the glibc version recorded by its build; the
Ubuntu 24.04 packages have required glibc 2.39. Do not assume a package built on
a newer distribution runs on an older one. Use the package's `STATIC.md` and
consumer linkage record for its actual requirements.

The package's `licenses.json` records original source paths and byte hashes for
collected texts, including separate Rust toolchain notices. Its Cargo closure
conservatively includes build/dev/other-target dependencies; it is not a precise
list of code reachable in the final binary. Missing texts are reported explicitly
as `unresolved`, and a successful technical consumer test does not complete the
redistribution review. Check the actual consumer linkage and platform requirements
for the package you distribute.

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

The dynamic bindings use the same system media frameworks as the CLI. On
macOS 15 and later, supported linkers delay their initialization until needed;
earlier systems and static Go keep ordinary framework linkage. The first OCR,
HEIF/AVIF decode or PDF rasterization may therefore have a different startup
cost from plain text conversion. This does not change the platform's media
capabilities or remove framework dependencies.

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

Windows Python/Node package checks have run; Windows Go/cgo delivery,
physical Intel hardware and every minimum language runtime are not established
by those checks. A host test or a valid native-library header does not certify
another target. Historical evidence remains scoped to its recorded source and
platform.
