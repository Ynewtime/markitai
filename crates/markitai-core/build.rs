// Links the macOS system frameworks delay-initialized in this package's
// executables and tests; `markitai-cli` includes this script for its own.
//
// Only OCR, HEIF/AVIF decoding and PDF rasterization use Vision, ImageIO and
// CoreGraphics, and Foundation through them. On current macOS, CoreGraphics or
// CoreFoundation alone initializes several hundred dependent images, and the
// conversion path calls CoreFoundation only from those backends and the
// time-zone fallback. From macOS 15, dyld maps delay-initialized frameworks at
// launch but runs their initializers on first use; earlier systems ignore the
// flag and initialize them at launch as before. A C call triggers that first
// use, an Objective-C class lookup by name does not, so the backends open their
// framework explicitly (`src/system_frameworks.rs`). Linkers that predate
// `-delay_framework` keep ordinary links.
//
// For a deployment target below macOS 15, ld notes on every such link that
// older systems ignore the flag, and it notes that CoreGraphics has weak
// definitions (its own template instances; dyld still delays it, as for
// Apple's own images that delay it). rustc reports linker output as warnings,
// so these links pass `-w`, which silences all ld warnings for them.

use std::{env, fs, path::PathBuf, process::Command};

const DELAYED: [&str; 5] = [
    "CoreFoundation",
    "Foundation",
    "CoreGraphics",
    "ImageIO",
    "Vision",
];

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=../markitai-core/build.rs");
    println!("cargo:rerun-if-env-changed=RUSTC_LINKER");
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") || !linker_delays() {
        return;
    }
    for framework in DELAYED {
        println!("cargo:rustc-link-arg=-Wl,-delay_framework,{framework}");
    }
    println!("cargo:rustc-link-arg=-Wl,-w");
}

fn linker_delays() -> bool {
    let arch = match env::var("CARGO_CFG_TARGET_ARCH").as_deref() {
        Ok("aarch64") => "arm64",
        Ok("x86_64") => "x86_64",
        _ => return false,
    };
    let Some(out) = env::var_os("OUT_DIR").map(PathBuf::from) else {
        return false;
    };
    let source = out.join("delay_probe.c");
    if fs::write(&source, "int main(void) { return 0; }\n").is_err() {
        return false;
    }
    let linker = env::var("RUSTC_LINKER").unwrap_or_else(|_| "cc".into());
    Command::new(linker)
        .args(["-arch", arch, "-Wl,-delay_framework,CoreFoundation", "-o"])
        .arg(out.join("delay_probe"))
        .arg(&source)
        .output()
        .is_ok_and(|output| output.status.success())
}
