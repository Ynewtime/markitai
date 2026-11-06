//! macOS Vision in a process of its own, for a program that holds Markitai as a
//! library: the Python and Node packages carry this beside theirs. The `markitai`
//! executable is its own reader. See `ocr/reader.rs` in `markitai-core`.
#[cfg(target_os = "macos")]
fn main() {
    markitai_core::serve_vision_reader()
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("markitai-vision-reader is only for macOS");
    std::process::exit(1);
}
