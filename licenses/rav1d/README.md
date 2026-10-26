# rav1d

Windows and Linux builds of markitai decode AVIF pictures with rav1d
(BSD-2-Clause; VideoLAN, the dav1d authors and the Internet Security Research
Group), built statically from the Rust sources vendored in `vendor/rav1d` of
the markitai repository (upstream commit and file hashes in
`vendor/rav1d/UPSTREAM.json`, changes in `vendor/rav1d/MARKITAI-PATCH.md`).
macOS builds decode AVIF with the system's ImageIO; every package carries the
same notices so that the license set does not depend on the platform.

| File | Covers |
| --- | --- |
| `COPYING` | the rav1d license (BSD-2-Clause), copied unchanged from upstream |
| `LICENSE-dav1d-rs` | the MIT notice of dav1d-rs (Luca Barbato), from which rav1d's Rust API derives |
