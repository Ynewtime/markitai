// The program is the library's; this is its executable.
fn main() {
    #[cfg(target_os = "macos")]
    markitai_core::start_vision_reader();
    std::process::exit(markitai_cli::run(std::env::args_os().collect()));
}
