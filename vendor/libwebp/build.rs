//! Builds libwebp's lossy encoder and `markitai_webp.c` into one static
//! library, the way libwebp's CMake build does: on x86 the SSE4.1 and AVX2
//! files are compiled for those instruction sets and chosen at run time
//! (`src/dsp/cpu.c`); NEON is part of every aarch64 target. MSVC's cl
//! enables these intrinsics without flags; clang-cl needs them as clang does.
use std::{env, fs, path::Path, path::PathBuf};

fn sources(directory: &Path) -> Vec<PathBuf> {
    let mut files: Vec<_> = fs::read_dir(directory)
        .unwrap_or_else(|error| panic!("{}: {error}", directory.display()))
        .map(|entry| entry.expect("vendored source entry").path())
        .filter(|path| path.extension().is_some_and(|extension| extension == "c"))
        .collect();
    files.sort();
    files
}

fn main() {
    let root = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let arch = env::var("CARGO_CFG_TARGET_ARCH").expect("CARGO_CFG_TARGET_ARCH");
    let x86 = matches!(arch.as_str(), "x86" | "x86_64");
    let mut files = vec![root.join("markitai_webp.c")];
    for directory in ["src/enc", "src/dsp", "src/utils", "sharpyuv"] {
        println!("cargo:rerun-if-changed={directory}");
        files.extend(sources(&root.join(directory)));
    }
    println!("cargo:rerun-if-changed=markitai_webp.c");

    let build = || {
        let mut build = cc::Build::new();
        build.include(&root).define("NDEBUG", None).warnings(false);
        if !build.get_compiler().is_like_msvc() {
            build.flag("-fvisibility=hidden");
            if x86 {
                build
                    .define("WEBP_HAVE_SSE41", None)
                    .define("WEBP_HAVE_AVX2", None);
            }
        }
        build
    };
    let mut main = build();
    let compiler = main.get_compiler();
    if x86 && (!compiler.is_like_msvc() || compiler.is_like_clang_cl()) {
        for (suffix, flag) in [("_sse41.c", "-msse4.1"), ("_avx2.c", "-mavx2")] {
            let (special, rest): (Vec<_>, Vec<_>) = files
                .into_iter()
                .partition(|path| path.to_string_lossy().ends_with(suffix));
            files = rest;
            for object in build().flag(flag).files(&special).compile_intermediates() {
                main.object(object);
            }
        }
    }
    main.files(&files).compile("markitai_webp");
}
