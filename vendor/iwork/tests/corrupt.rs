//! What a broken document does to the reader.
//!
//! A library that opens files someone else wrote has one obligation before
//! any of the interesting ones: rubbish in has to come back as an `Err`, not
//! as a panic in the caller's process. Nothing here asserts what the error
//! *says* — a corrupted document has no correct reading — only that asking
//! for one returns rather than aborting.
//!
//! The three shapes are not interchangeable. A flipped bit or a spliced run
//! usually dies at the ZIP layer's checksum and never reaches the parser, so
//! those two tests mostly prove the outer layers hold. `corrupt_object_streams`
//! is the one that matters: it corrupts the *decompressed* stream and rebuilds
//! the package around it, which is the only way to hand the object parser
//! bytes it cannot make sense of.

use std::path::{Path, PathBuf};

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/generated")
}

/// The corpus is generated, never committed, so these skip where it is absent
/// — on CI, and on a clean checkout before `scripts/make-fixtures.sh` has run.
macro_rules! corpus {
    ($name:expr) => {{
        let path = fixtures().join($name);
        match std::fs::read(&path) {
            Ok(bytes) => bytes,
            Err(_) => {
                eprintln!("no {} — skipping (run scripts/make-fixtures.sh)", $name);
                return;
            }
        }
    }};
}

/// xorshift64, so a failing case can be reproduced from its seed.
fn next(seed: &mut u64) -> u64 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    *seed
}

fn scratch(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("iwork-corrupt-{}-{name}", std::process::id()))
}

#[test]
fn a_flipped_bit_does_not_panic() {
    let mut seed = 0x243f_6a88_85a3_08d3;
    let mut checked = 0usize;
    // The first one decides whether there is a corpus at all.
    let _ = corpus!("numbers-values.numbers");
    for name in [
        "numbers-values.numbers",
        "pages-plain.pages",
        "keynote-slides.key",
    ] {
        let Ok(original) = std::fs::read(fixtures().join(name)) else {
            continue;
        };
        for _ in 0..300 {
            let mut bytes = original.clone();
            let at = next(&mut seed) as usize % bytes.len();
            bytes[at] ^= 1 << (next(&mut seed) % 8);
            let out = scratch(name);
            std::fs::write(&out, &bytes).unwrap();
            let _ = iwork::Document::open(&out);
            let _ = std::fs::remove_file(&out);
            checked += 1;
        }
    }
    assert!(checked > 0, "no fixtures — run scripts/make-fixtures.sh");
}

#[test]
fn a_truncated_file_does_not_panic() {
    let original = corpus!("numbers-values.numbers");
    for n in 1..=400 {
        let out = scratch("truncated.numbers");
        std::fs::write(&out, &original[..original.len() * n / 400]).unwrap();
        let _ = iwork::Document::open(&out);
        let _ = std::fs::remove_file(&out);
    }
}

#[test]
fn corrupt_object_streams_do_not_panic() {
    let mut seed = 0xdead_beef_cafe_f00d;
    let mut reached = 0usize;
    let _ = corpus!("numbers-values.numbers");
    for name in [
        "numbers-values.numbers",
        "pages-plain.pages",
        "keynote-slides.key",
    ] {
        let Ok(package) = iwork::package::Package::read(fixtures().join(name)) else {
            continue;
        };
        for entry in package.iwa_names() {
            let Some(raw) = package.get(&entry) else {
                continue;
            };
            let Ok(stream) = iwork::iwa::decompress(raw) else {
                continue;
            };
            for _ in 0..12 {
                let mut bytes = stream.clone();
                let at = next(&mut seed) as usize % bytes.len();
                let len = 1 + (next(&mut seed) >> 40) as usize % 32;
                for byte in &mut bytes[at..(at + len).min(stream.len())] {
                    *byte = next(&mut seed) as u8;
                }
                let mut broken = package.clone();
                broken.set(&entry, iwork::iwa::compress(&bytes));
                let out = scratch(name);
                if broken.write(&out).is_ok() {
                    let _ = iwork::Document::open(&out);
                    reached += 1;
                }
                let _ = std::fs::remove_file(&out);
            }
        }
    }
    assert!(reached > 0, "nothing reached the object parser");
}
