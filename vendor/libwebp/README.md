# libwebp lossy encoder

markitai writes `image.format: webp` as lossy WebP the way Python markitai 1.2
did with Pillow (`Image.save(format="WEBP", quality=q)`). This crate builds the
encoder half of libwebp 1.6.0, the version Pillow 12 and 13 bundle, from the C
sources here into one static library: no system libwebp, no runtime
dependency.

- `markitai_webp.c` is the only entry point. It repeats Pillow's
  `WebPEncode`: libwebp's default configuration at the given quality,
  `method` 4, `alpha_quality` 100, `exact` off, ARGB input, no ICC, EXIF or
  XMP chunk. With the same pixels the file is byte for byte Pillow's.
- `build.rs` compiles it with the sources the way libwebp's CMake build does:
  on x86 the `_sse41.c` and `_avx2.c` files get `-msse4.1` / `-mavx2` and run
  only on processors that have them (`src/dsp/cpu.c`); aarch64 always has
  NEON; MSVC enables the intrinsics without flags. Outside Windows it defines
  `WEBP_USE_THREAD`, as libwebp's own builds and Pillow's do, so the encoder's
  one-time table set-up is locked; libwebp never locks it on Windows, so
  `images/webp.rs` runs the first encode alone on every system before any
  other may run.
- `lib.rs` declares the function; `markitai-core`'s `images/webp.rs` wraps it.

The libwebp files are unchanged copies from the signed release archive; their
list, sizes and SHA-256 are in `UPSTREAM.json`. libwebp is BSD-3-Clause
(`COPYING`, with Google's patent grant in `PATENTS`); the release archives and
wheels carry both under `licenses/libwebp/`. markitai's own files here are
under the project's MIT license.

To update: verify the new release archive's signature, copy the same set of
files (add any new source a kept module needs; the link fails otherwise),
refresh `UPSTREAM.json` and `licenses/libwebp/`, and check that
`images::webp::tests::files_are_byte_for_byte_pillows` still matches the
Pillow release that bundles the same libwebp.
