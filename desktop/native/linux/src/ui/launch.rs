//! The programs the app opens: a file's or a link's default program and the
//! file manager. They start in the user's environment, without the variables
//! the app set for its own toolkit (DESIGN.md 7.5): a launch context gives
//! them the user's values back, and the app's environment stays as it is.

use std::ffi::OsString;
use std::path::Path;
use std::sync::Mutex;

use gtk::prelude::*;
use gtk::{gio, glib};

/// Each variable the app set for its toolkit, with the value the user's
/// environment gave it, None for none.
static USER_VALUES: Mutex<Vec<(&'static str, Option<OsString>)>> = Mutex::new(Vec::new());

/// Set `key` for the app's own toolkit, keeping the user's value for the
/// programs it opens.
///
/// # Safety
/// As [`std::env::set_var`]: no other thread reads or writes the environment.
pub unsafe fn set_for_toolkit(key: &'static str, value: &str) {
    let mut kept = USER_VALUES.lock().unwrap_or_else(|e| e.into_inner());
    if !kept.iter().any(|(k, _)| *k == key) {
        kept.push((key, std::env::var_os(key)));
    }
    // SAFETY: as the caller promises.
    unsafe { std::env::set_var(key, value) };
}

/// The launch context of `widget`'s display, with the user's values back.
fn user_context(widget: &impl IsA<gtk::Widget>) -> gio::AppLaunchContext {
    let context: gio::AppLaunchContext = widget.display().app_launch_context().upcast();
    for (key, value) in USER_VALUES.lock().unwrap_or_else(|e| e.into_inner()).iter() {
        match value {
            Some(value) => context.setenv(key, value),
            None => context.unsetenv(key),
        }
    }
    context
}

/// Open `uri` with its default program; `failed` runs when none opens it.
pub fn open_uri(widget: &impl IsA<gtk::Widget>, uri: &str, failed: impl FnOnce() + 'static) {
    let context = user_context(widget);
    gio::AppInfo::launch_default_for_uri_async(uri, Some(&context), gio::Cancellable::NONE, move |result| {
        if result.is_err() {
            failed();
        }
    });
}

/// Open the file or folder at `path` with its default program.
pub fn open_path(widget: &impl IsA<gtk::Widget>, path: &Path, failed: impl FnOnce() + 'static) {
    open_uri(widget, &gio::File::for_path(path).uri(), failed);
}

/// Ask the file manager to show `path` selected, else open its folder.
pub fn show_in_folder(widget: &impl IsA<gtk::Widget>, path: &Path) {
    let widget = widget.clone().upcast::<gtk::Widget>();
    let uri = gio::File::for_path(path).uri().to_string();
    let folder = path.parent().unwrap_or(path).to_path_buf();
    glib::spawn_future_local(async move {
        let shown = match gio::bus_get_future(gio::BusType::Session).await {
            Ok(bus) => bus
                .call_future(
                    Some("org.freedesktop.FileManager1"),
                    "/org/freedesktop/FileManager1",
                    "org.freedesktop.FileManager1",
                    "ShowItems",
                    Some(&(vec![uri], "").to_variant()),
                    None,
                    gio::DBusCallFlags::NONE,
                    -1,
                )
                .await
                .is_ok(),
            Err(_) => false,
        };
        if !shown {
            open_path(&widget, &folder, || {});
        }
    });
}
