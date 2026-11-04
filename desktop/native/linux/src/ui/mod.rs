//! The GTK side: the application, its window and dialogs. Behavior lives on
//! [`Win`]; [`crate::model`] holds the state it shows.

mod actions;
mod convert;
mod detail;
mod history;
mod prefs;
mod rows;
mod window;

use std::cell::RefCell;
use std::rc::Rc;

use adw::prelude::*;
use gtk::{gdk, gio, glib};

use crate::model::APP_ID;
use crate::strings::tr;

pub use window::Win;

const STYLE: &str = include_str!("style.css");
const ACCENT_LIGHT: &str =
    ":root { --accent-bg-color: #18181b; --accent-fg-color: #ffffff; --accent-color: #18181b; }";
const ACCENT_DARK: &str =
    ":root { --accent-bg-color: #71717a; --accent-fg-color: #ffffff; --accent-color: #d4d4d8; }";

pub fn run(args: &[String]) -> glib::ExitCode {
    let app = adw::Application::builder()
        .application_id(APP_ID)
        .flags(gio::ApplicationFlags::HANDLES_OPEN)
        .build();
    let window: Rc<RefCell<Option<Rc<Win>>>> = Rc::default();

    app.connect_startup(|app| {
        glib::set_application_name(&tr("app.name.native"));
        gtk::Window::set_default_icon_name(APP_ID);
        install_style();
        add_icon_search_path();
        // Showing a detail moves focus to its selectable title. Focus alone
        // should not select the whole title.
        if let Some(settings) = gtk::Settings::default() {
            settings.set_gtk_label_select_on_focus(false);
        }
        actions::install_app(app);
    });

    let slot = window.clone();
    app.connect_activate(move |app| {
        win(app, &slot).window.present();
    });
    let slot = window.clone();
    app.connect_open(move |app, files, _hint| {
        // Added before the window shows, so its first focus is not on the
        // empty page the files replace.
        let win = win(app, &slot);
        win.add_gfiles(files);
        win.window.present();
    });
    let code = app.run_with_args(args);
    // Only a confirmed Quit leaves items Converting here (DESIGN.md 7.3).
    if let Some(win) = window.borrow().as_ref() {
        win.discard_converting();
    }
    code
}

fn win(app: &adw::Application, slot: &Rc<RefCell<Option<Rc<Win>>>>) -> Rc<Win> {
    slot.borrow_mut()
        .get_or_insert_with(|| Win::new(app))
        .clone()
}

/// The ink accent (DESIGN.md 13.2), swapped with the appearance, and the
/// app's own classes.
fn install_style() {
    let Some(display) = gdk::Display::default() else {
        return;
    };
    let app_css = gtk::CssProvider::new();
    app_css.load_from_string(STYLE);
    gtk::style_context_add_provider_for_display(
        &display,
        &app_css,
        gtk::STYLE_PROVIDER_PRIORITY_APPLICATION,
    );
    let accent = gtk::CssProvider::new();
    gtk::style_context_add_provider_for_display(
        &display,
        &accent,
        gtk::STYLE_PROVIDER_PRIORITY_APPLICATION,
    );
    let manager = adw::StyleManager::default();
    let apply = move |manager: &adw::StyleManager| {
        accent.load_from_string(if manager.is_dark() {
            ACCENT_DARK
        } else {
            ACCENT_LIGHT
        });
    };
    apply(&manager);
    manager.connect_dark_notify(apply);
}

/// The icons staged next to the binary (`../share/icons`), so a build tree
/// shows the app icon without being installed.
fn add_icon_search_path() {
    let (Some(display), Ok(exe)) = (gdk::Display::default(), std::env::current_exe()) else {
        return;
    };
    if let Some(prefix) = exe.parent().and_then(|bin| bin.parent()) {
        let icons = prefix.join("share").join("icons");
        if icons.is_dir() {
            gtk::IconTheme::for_display(&display).add_search_path(icons);
        }
    }
}
