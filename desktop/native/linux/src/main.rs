//! Markitai for GNOME: GTK 4, libadwaita and WebKitGTK 6 around the markitai
//! core, which runs in process (desktop/DESIGN.md).

mod model;
mod strings;
mod ui;

use std::path::Path;

/// The app is linked with the core it was built against, so a mismatch is a
/// build error here rather than the `fatal.abi.*` window.
const ABI_VERSION: u32 = 1;
const _: () = assert!(markitai_app::ABI_VERSION == ABI_VERSION);

fn main() -> gtk::glib::ExitCode {
    // `prepare_environment` makes the data folder the working directory, so
    // relative file arguments are resolved against the launch folder first.
    let launch_dir = std::env::current_dir().ok();
    let args: Vec<String> = std::env::args_os()
        .enumerate()
        .map(|(i, arg)| {
            let text = arg.to_string_lossy().into_owned();
            match &launch_dir {
                Some(dir) if i > 0 && !text.starts_with('-') && !text.contains("://") => {
                    dir.join(Path::new(&arg)).to_string_lossy().into_owned()
                }
                _ => text,
            }
        })
        .collect();

    let request = serde_json::json!({"data_dir": model::data_dir()}).to_string();
    // SAFETY: first in main, before GTK or anything else starts a thread.
    let _ = unsafe { markitai_app::prepare_environment(&request) };

    // After the login environment, which may carry the language (DESIGN.md
    // 7.5). The toolkit's own words follow the app's choice through gettext.
    let names = gtk::glib::language_names();
    let first = names.first().map(|s| s.as_str());
    let lang = strings::Lang::from_preferred(first.into_iter());
    if let Some(language) = lang.toolkit_language(first) {
        // SAFETY: GTK has not started; the only other threads are the login
        // import's pipe readers, which do not read the environment.
        unsafe { std::env::set_var("LANGUAGE", language) };
    }
    strings::set_lang(lang);

    ui::run(&args)
}
