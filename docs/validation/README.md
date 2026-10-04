# Validation tools

Run the source checks described in [development](../development.md), and test
installed bindings using the [binding guide](../bindings.md). Native package
builds and installation checks are described in [CI](../ci.md).

The retained drivers support current checks:

- `drivers/windows-check-round38/check-windows.sh`: cross-target Windows type
  checking on macOS. This uses C header stubs and does not link or run Windows
  binaries; native Windows CI and installation tests are separate.
- `drivers/portable-raster-r1/`: PDF renderer comparison and scanned-page OCR
  probes referenced by [PDF rendering](../pdf-rendering.md).

## Differential checks

```sh
python3 scripts/audit_formats.py \
  --reference /path/to/reference-markitai \
  --library target/release/libmarkitai_ffi.dylib \
  --output .local/audits/formats

python3 scripts/benchmark_cli.py \
  --reference /path/to/reference-markitai \
  --binary target/release/markitai \
  --output .local/benchmarks/cli.json
```

On Linux use `.so`; on Windows use `.dll`. The reference needs its own installed
`.venv`, or an explicit `--reference-python`. Use isolated `HOME` and
`MARKITAI_HOME` directories, repository fixtures and loopback mock servers.

Record the source and artifact identities, commands, platform, fixture set and
raw results under ignored `.local/`. Compare output quality before interpreting
timing ratios. Source checks, installed-package tests and measured performance
establish different claims; none implies complete reference parity.

Historical reports, source snapshots and one-off drivers are available in
[Git history](https://github.com/Ynewtime/markitai-rust/tree/1749201edbaaa7198d3b9a5056ce4e87a1d08978/docs/validation).
Their conclusions apply only to their recorded source, artifacts and environment.
