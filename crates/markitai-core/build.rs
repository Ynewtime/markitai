// Links the macOS system frameworks delay-initialized in this package's
// executables and tests; `markitai-cli` includes this script for its own, and
// the language bindings' build scripts (the Node addon, the Python extension
// and the C library) call `delay_frameworks` through it as a module.
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
//
// On Linux with glibc the same executables pack their relative relocations
// (`-z pack-relative-relocs`, DT_RELR). Every pointer stored in a
// position-independent executable's data (string tables, vtables) is
// relocated at load time; as `.rela.dyn` entries that takes 24 bytes each,
// about 0.9 MB of the x86-64 CLI, and as DT_RELR bitmaps a few kilobytes.
// Mach-O already stores these in place. The loader must be glibc 2.36 or
// later, which the linker records as a `GLIBC_ABI_DT_RELR` requirement; an
// executable linked against glibc 2.39 already requires 2.39 (the standard
// library's pidfd functions). A probe linked by this rustc with the flag must
// carry DT_RELR and run correctly here, so a cross build, an older C library
// or a linker without the option keeps ordinary relocations.

use std::{
    env, fs,
    path::{Path, PathBuf},
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

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
    build_metadata();
    if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
        if env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("gnu") && packs_relocations() {
            println!("cargo:rustc-link-arg=-Wl,-z,pack-relative-relocs");
        }
        return;
    }
    delay_frameworks();
}

/// On macOS, links the media frameworks delay-initialized in every target of
/// the calling package that cargo links (executables, tests and cdylibs).
pub fn delay_frameworks() {
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

/// Whether this rustc links a native executable with packed relative
/// relocations that the C library here applies.
fn packs_relocations() -> bool {
    let (Ok(host), Ok(target)) = (env::var("HOST"), env::var("TARGET")) else {
        return false;
    };
    let Some(out) = env::var_os("OUT_DIR").map(PathBuf::from) else {
        return false;
    };
    if host != target {
        return false;
    }
    // Each string's address in the table is relocated at load time; read
    // through an unrelocated table, the program fails.
    let source = out.join("relr_probe.rs");
    let probe = out.join("relr_probe");
    let program = r#"
static WORDS: [&str; 3] = ["packed", "relative", "relocations"];
fn main() {
    let word = WORDS[std::hint::black_box(2)];
    std::process::exit(if word == "relocations" { 0 } else { 1 });
}
"#;
    if fs::write(&source, program).is_err() {
        return false;
    }
    let mut rustc = Command::new(env::var("RUSTC").unwrap_or_else(|_| "rustc".into()));
    rustc.args([
        "--edition",
        "2021",
        "--crate-type",
        "bin",
        "--target",
        &target,
    ]);
    if let Ok(flags) = env::var("CARGO_ENCODED_RUSTFLAGS") {
        rustc.args(flags.split('\x1f').filter(|flag| !flag.is_empty()));
    }
    if let Ok(linker) = env::var("RUSTC_LINKER") {
        rustc.arg(format!("-Clinker={linker}"));
    }
    rustc
        .arg("-Clink-arg=-Wl,-z,pack-relative-relocs")
        .arg("-o")
        .arg(&probe)
        .arg(&source);
    rustc.output().is_ok_and(|output| output.status.success())
        && fs::read(&probe).is_ok_and(|image| has_relr(&image))
        && Command::new(&probe)
            .output()
            .is_ok_and(|output| output.status.success())
}

/// Whether a 64-bit little-endian ELF image's dynamic section has DT_RELR.
fn has_relr(image: &[u8]) -> bool {
    const PT_DYNAMIC: u32 = 2;
    const DT_RELR: u64 = 36;
    let u16_at = |at: usize| {
        image
            .get(at..at + 2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
    };
    let u32_at = |at: usize| {
        image
            .get(at..at + 4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
    };
    let u64_at = |at: usize| {
        image
            .get(at..at + 8)
            .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
    };
    // ELF magic, 64-bit class, little-endian data.
    if image.get(..6) != Some(b"\x7fELF\x02\x01") {
        return false;
    }
    let (Some(phoff), Some(phentsize), Some(phnum)) = (u64_at(0x20), u16_at(0x36), u16_at(0x38))
    else {
        return false;
    };
    (0..usize::from(phnum)).any(|index| {
        let header = phoff as usize + index * usize::from(phentsize);
        if u32_at(header) != Some(PT_DYNAMIC) {
            return false;
        }
        let (Some(offset), Some(size)) = (u64_at(header + 0x08), u64_at(header + 0x20)) else {
            return false;
        };
        (offset as usize..(offset + size) as usize)
            .step_by(16)
            .map_while(u64_at)
            .take_while(|&tag| tag != 0)
            .any(|tag| tag == DT_RELR)
    })
}

/// The build identity a package may show in its help: the commit it was built
/// from, whether the worktree differed, the target and profile, and when the
/// build script ran. `SOURCE_DATE_EPOCH` fixes the timestamp, so a reproducible
/// build reports the identity of the build it reproduces; without a checkout the
/// commit is `unknown`.
///
/// Every package including this script publishes its own copy as `rustc-env`;
/// `markitai_core::build_info` reads this crate's, and the CLI prints that. A
/// package whose build script is this file through `include!` is not recompiled
/// when the file changes, so a copy taken from there could go stale.
fn build_metadata() {
    // The timestamp follows this variable, so a value that changes has to run
    // this script again.
    println!("cargo:rerun-if-env-changed=SOURCE_DATE_EPOCH");
    let checkout = git_checkout();
    let git = |args: &[&str]| {
        let output = Command::new("git")
            .arg("-C")
            .arg(checkout.as_ref()?.0.as_path())
            .args(args)
            .output()
            .ok()?;
        output
            .status
            .success()
            .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
    };
    let commit = git(&["rev-parse", "--short", "HEAD"])
        .filter(|hash| !hash.is_empty())
        .unwrap_or_else(|| "unknown".into());
    let dirty = git(&["status", "--porcelain"])
        .map(|changes| !changes.is_empty())
        .unwrap_or(false);
    let epoch = env::var("SOURCE_DATE_EPOCH")
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .or_else(|| {
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .ok()
                .map(|since| since.as_secs())
        })
        .unwrap_or(0);
    println!("cargo:rustc-env=MARKITAI_BUILD_COMMIT={commit}");
    println!("cargo:rustc-env=MARKITAI_BUILD_DIRTY={}", u8::from(dirty));
    println!("cargo:rustc-env=MARKITAI_BUILD_EPOCH={epoch}");
    println!(
        "cargo:rustc-env=MARKITAI_BUILD_TARGET={}",
        env::var("TARGET").unwrap_or_else(|_| "unknown".into())
    );
    println!(
        "cargo:rustc-env=MARKITAI_BUILD_PROFILE={}",
        env::var("PROFILE").unwrap_or_else(|_| "unknown".into())
    );
    // A new commit has to refresh the identity, so the branch pointers are
    // watched. A source archive has no `.git`, and a watched path that does not
    // exist would rerun this script, and rebuild the package, on every build.
    if let Some((_, git)) = checkout {
        for path in [git.join("HEAD"), git.join("refs")] {
            if path.exists() {
                println!("cargo:rerun-if-changed={}", path.display());
            }
        }
    }
}

/// The checkout above this package: its worktree (what `git status` needs to
/// see uncommitted changes) and its git directory (what holds `HEAD`). A linked
/// worktree has a `.git` file naming a git directory elsewhere.
fn git_checkout() -> Option<(PathBuf, PathBuf)> {
    let mut directory = PathBuf::from(env::var("CARGO_MANIFEST_DIR").ok()?);
    for _ in 0..4 {
        let dot = directory.join(".git");
        if dot.is_dir() {
            return Some((directory.clone(), dot));
        }
        if dot.is_file() {
            let text = fs::read_to_string(&dot).ok()?;
            let named = text
                .lines()
                .find_map(|line| line.strip_prefix("gitdir:"))?
                .trim();
            let git = Path::new(named);
            let path = if git.is_absolute() {
                git.to_path_buf()
            } else {
                directory.join(git)
            };
            return Some((directory, path));
        }
        directory = directory.parent()?.to_path_buf();
    }
    None
}
