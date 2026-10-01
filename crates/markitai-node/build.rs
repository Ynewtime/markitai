// The addon links the macOS media frameworks delay-initialized like the
// command-line executables, through the core's build script; its other
// functions (and its own `main`) serve the core and stay unused here.
#[allow(dead_code)]
#[path = "../markitai-core/build.rs"]
mod core_build;

fn main() {
    napi_build::setup();
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=../markitai-core/build.rs");
    println!("cargo:rerun-if-env-changed=RUSTC_LINKER");
    core_build::delay_frameworks();
}
