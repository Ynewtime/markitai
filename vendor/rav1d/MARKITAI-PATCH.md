# Markitai's pinned AV1 decoder

This directory holds the Rust sources of rav1d, the memory-safety port of the
dav1d AV1 decoder (BSD-2-Clause, `COPYING`), at upstream commit
`d3d1cd67059f47803919be8276650e5870c9fd02` of
<https://github.com/memorysafety/rav1d> (main, 2026-08-14). markitai-core
uses it on Windows and Linux to decode AVIF images; macOS uses ImageIO.

That commit, not the 1.1.0 release on crates.io, is vendored because only
upstream main has the safe Rust API (`src/rust_api.rs`, upstream pull request
1439, adapted from dav1d-rs): with 1.1.0, markitai would have to call the C
ABI functions through `unsafe` code of its own. `UPSTREAM.json` records the
source archive checksum and every copied file's upstream checksum. Not
copied: the dav1d C sources and headers, the x86 and Arm assembly, the
command-line tool, tests, documentation, packaging, meson files, `Cargo.lock`,
`rust-toolchain.toml` and `rustfmt.toml` (nightly-only options; the
workspace's `cargo fmt --all` reaches this path dependency, and stable
rustfmt leaves the upstream sources unchanged without it).

Local changes:

- `Cargo.toml`: the `[workspace]` (the tool), the `[lib]` crate types
  (`staticlib` built a 26 MB archive nobody links), the `cc` and `nasm-rs`
  build dependencies and the profiles are removed, and the default features
  are `bitdepth_8` and `bitdepth_16` without assembly. The `asm` features
  remain declared because the sources test them, but their assembly is not
  here: enabling them does not build.
- `src/rust_api.rs`: `Settings::set_logging`, marked `markitai`, so that a
  damaged image does not print dav1d's diagnostics on the host process's
  standard error (libavif turns them off as well). The image's error carries
  the failure instead.
- `src/decode.rs`: the error path of `rav1d_submit_frame` no longer unwraps
  a frame header that `rav1d_decode_frame_exit` has already taken, marked
  `markitai`. With one frame context, which markitai uses, every damaged AV1
  frame panicked there (upstream issue 1497); the change is the one line that
  the open upstream pull requests 1364, 1496 and 1503 all make. Drop it when
  upstream merges one of them.
- `LICENSE-dav1d-rs` (new): the MIT notice of dav1d-rs that `src/rust_api.rs`
  reproduces in a comment, copied into a file so that binary distributions'
  license collection, which reads license files, carries it beside
  `COPYING`.

markitai-core decodes on the calling thread (one decoder thread, so no
worker threads), bounds frames to 32 million pixels and runs the whole
container decode under `catch_unwind`, so a panic fails one image.
To update, copy the same file set from a newer upstream commit, record it in
`UPSTREAM.json`, reapply the changes above, and rerun the AVIF tests
(`images::heif`) and a mutation run over the AVIF fixtures with a release
build, as was done when this directory was added.
