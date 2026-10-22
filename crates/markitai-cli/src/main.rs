// The program is the library's; this is its executable.
fn main() {
    std::process::exit(markitai_cli::run(std::env::args_os().collect()));
}
