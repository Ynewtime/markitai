// As markitai-ffi: a dynamic library built from this package links the macOS
// media frameworks delay-initialized; the static library's consumer (the
// macOS app's Package.swift) names the frameworks itself.
#[allow(dead_code)]
#[path = "../markitai-core/build.rs"]
mod core_build;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=../markitai-core/build.rs");
    println!("cargo:rerun-if-env-changed=RUSTC_LINKER");
    core_build::delay_frameworks();
}
