// The dynamic library and this package's tests link the macOS media
// frameworks delay-initialized like the command-line executables, through the
// core's build script; its other functions (and its own `main`) serve the core
// and stay unused here. The static library is not linked here: its consumers
// name the frameworks themselves (`bindings/go/link_static_darwin_arm64.go`).
#[allow(dead_code)]
#[path = "../markitai-core/build.rs"]
mod core_build;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=../markitai-core/build.rs");
    println!("cargo:rerun-if-env-changed=RUSTC_LINKER");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-cdylib-link-arg=-Wl,-install_name,@rpath/libmarkitai_ffi.dylib");
    }
    core_build::delay_frameworks();
}
