//! App and window actions with their shortcuts (DESIGN.md 7.2 and 11), and
//! the list context menus.

use std::path::{Path, PathBuf};
use std::rc::Rc;

use adw::prelude::*;
use gtk::{gdk, gio, glib};

use super::detail::View;
use super::rows::row_id;
use super::window::{Page, Win};
use crate::model::APP_ID;
use crate::model::queue::State;
use crate::model::settings::clamp_zoom;
use crate::strings::{tr, trf};

/// The shown Markdown, when it is in memory, and where it came from.
pub struct Current {
    pub markdown: Option<Rc<String>>,
    pub markdown_path: PathBuf,
    pub folder: PathBuf,
}

const ACCELS: &[(&str, &[&str])] = &[
    ("app.quit", &["<Control>q"]),
    ("app.preferences", &["<Control>comma"]),
    ("window.close", &["<Control>w"]),
    ("win.choose-files", &["<Control>o"]),
    ("win.add-url", &["<Control>l"]),
    ("win.go-convert", &["<Alt>1"]),
    ("win.go-history", &["<Alt>2"]),
    ("win.toggle-view", &["<Control><Shift>m"]),
    ("win.copy", &["<Control><Shift>c"]),
    ("win.save-as", &["<Control><Shift>s"]),
    ("win.show", &["<Control><Shift>r"]),
    ("win.retry", &["<Control>r"]),
    ("win.search", &["<Control>f"]),
    ("win.zoom-in", &["<Control>plus", "<Control>equal", "<Control>KP_Add"]),
    ("win.zoom-out", &["<Control>minus", "<Control>KP_Subtract"]),
    ("win.zoom-reset", &["<Control>0", "<Control>KP_0"]),
];

pub fn install_app(app: &adw::Application) {
    let quit = gio::SimpleAction::new("quit", None);
    let weak = app.downgrade();
    quit.connect_activate(move |_, _| {
        let Some(app) = weak.upgrade() else {
            return;
        };
        match app.active_window() {
            Some(window) => {
                let _ = window.activate_action("win.quit", None);
            }
            None => app.quit(),
        }
    });
    app.add_action(&quit);

    let preferences = gio::SimpleAction::new("preferences", None);
    let weak = app.downgrade();
    preferences.connect_activate(move |_, _| {
        if let Some(window) = weak.upgrade().and_then(|app| app.active_window()) {
            let _ = window.activate_action("win.preferences", None);
        }
    });
    app.add_action(&preferences);

    let about = gio::SimpleAction::new("about", None);
    let weak = app.downgrade();
    about.connect_activate(move |_, _| {
        let window = weak.upgrade().and_then(|app| app.active_window());
        if window
            .as_ref()
            .and_then(|window| window.downcast_ref::<adw::ApplicationWindow>())
            .is_some_and(|window| window.visible_dialog().is_some())
        {
            return;
        }
        let dialog = adw::AboutDialog::builder()
            .application_name(tr("app.name.native"))
            .application_icon(APP_ID)
            .version(env!("CARGO_PKG_VERSION"))
            .developer_name("Ynewtime")
            .copyright(tr("settings.about.copyright"))
            .license_type(gtk::License::MitX11)
            .build();
        dialog.present(window.as_ref());
    });
    app.add_action(&about);

    for (action, accels) in ACCELS {
        app.set_accels_for_action(action, accels);
    }
}

fn add(win: &Rc<Win>, name: &str, run: impl Fn(&Rc<Win>) + 'static) {
    let action = gio::SimpleAction::new(name, None);
    let weak = Rc::downgrade(win);
    action.connect_activate(move |_, _| {
        if let Some(win) = weak.upgrade() {
            run(&win);
        }
    });
    win.window.add_action(&action);
}

pub fn install_window(win: &Rc<Win>) {
    add(win, "quit", |win| win.quit());
    add(win, "preferences", super::prefs::present);
    add(win, "choose-files", |win| win.choose_files());
    add(win, "add-url", |win| win.focus_url());
    add(win, "paste", |win| win.paste());
    add(win, "go-convert", |win| {
        win.select_page(Page::Convert);
        win.outer.set_show_content(true);
    });
    add(win, "go-history", |win| {
        win.select_page(Page::History);
        win.outer.set_show_content(true);
    });
    add(win, "toggle-view", |win| {
        let view = match win.shared.view.get() {
            View::Preview => View::Markdown,
            View::Markdown => View::Preview,
        };
        win.convert_detail.set_view(view);
        win.history_detail.set_view(view);
    });
    add(win, "copy", |win| win.copy_markdown());
    add(win, "save-as", |win| win.save_as());
    add(win, "show", |win| {
        if let Some(current) = win.current() {
            win.show_in_files(&current.markdown_path);
        }
    });
    add(win, "retry", |win| win.retry());
    add(win, "remove", |win| win.remove_selected());
    add(win, "clear-finished", |win| {
        win.queue.borrow_mut().clear_finished();
        win.refresh_convert();
    });
    add(win, "clear-history", |win| win.ask_to_clear_history());
    add(win, "search", |win| {
        win.select_page(Page::History);
        win.outer.set_show_content(true);
        win.history_split.set_show_content(false);
        win.history_search.grab_focus();
    });
    add(win, "zoom-in", |win| win.zoom_by(0.1));
    add(win, "zoom-out", |win| win.zoom_by(-0.1));
    add(win, "zoom-reset", |win| win.set_zoom(1.0));
    add(win, "open-config", |win| win.open_config());

    // Keys a focused text field uses itself: these act only when it does
    // not (bubble phase, after the focus widget).
    win.window.add_controller(list_keys(gtk::PropagationPhase::Bubble));
}

/// Delete removes the selected item and Ctrl+V adds what the clipboard
/// holds (DESIGN.md 11), in `phase`.
pub fn list_keys(phase: gtk::PropagationPhase) -> gtk::ShortcutController {
    let shortcuts = gtk::ShortcutController::new();
    shortcuts.set_propagation_phase(phase);
    for (trigger, action) in [("Delete|KP_Delete", "win.remove"), ("<Control>v", "win.paste")] {
        shortcuts.add_shortcut(gtk::Shortcut::new(
            gtk::ShortcutTrigger::parse_string(trigger),
            Some(gtk::NamedAction::new(action)),
        ));
    }
    shortcuts
}

/// Right click, Shift+F10 or the Menu key on a row opens its menu.
pub fn connect_context_menu(win: &Rc<Win>, list: &gtk::ListBox, page: Page) {
    let click = gtk::GestureClick::builder().button(gdk::BUTTON_SECONDARY).build();
    let weak = Rc::downgrade(win);
    let target = list.clone();
    click.connect_pressed(move |gesture, _, x, y| {
        let Some(win) = weak.upgrade() else {
            return;
        };
        if let Some(row) = target.row_at_y(y as i32) {
            target.select_row(Some(&row));
            gesture.set_state(gtk::EventSequenceState::Claimed);
            win.popup_menu(page, &row, Some((x, y)));
        }
    });
    list.add_controller(click);

    let keys = gtk::ShortcutController::new();
    let weak = Rc::downgrade(win);
    let target = list.clone();
    keys.add_shortcut(gtk::Shortcut::new(
        gtk::ShortcutTrigger::parse_string("<Shift>F10|Menu"),
        Some(gtk::CallbackAction::new(move |_, _| {
            if let (Some(win), Some(row)) = (weak.upgrade(), target.selected_row()) {
                win.popup_menu(page, &row, None);
            }
            glib::Propagation::Stop
        })),
    ));
    list.add_controller(keys);
}

fn menu_section(items: &[(&str, &str)]) -> gio::Menu {
    let section = gio::Menu::new();
    for (key, action) in items {
        section.append(Some(&tr(key)), Some(action));
    }
    section
}

impl Win {
    fn popup_menu(&self, page: Page, row: &gtk::ListBoxRow, at: Option<(f64, f64)>) {
        let done_items = [
            ("detail.copy", "win.copy"),
            ("detail.save_as", "win.save-as"),
            ("detail.show.gnome", "win.show"),
        ];
        let menu = gio::Menu::new();
        let (popover, list) = match page {
            Page::Convert => {
                let state = row_id(row).and_then(|id| {
                    self.queue.borrow().get(id).map(|item| match item.state {
                        State::Done { .. } => 0,
                        State::Failed(_) => 1,
                        _ => 2,
                    })
                });
                match state {
                    Some(0) => menu.append_section(None, &menu_section(&done_items)),
                    Some(1) => menu.append_section(None, &menu_section(&[("detail.retry", "win.retry")])),
                    _ => {}
                }
                menu.append_section(None, &menu_section(&[("convert.remove", "win.remove")]));
                (&self.convert_menu, &self.convert_list)
            }
            Page::History => {
                menu.append_section(None, &menu_section(&done_items));
                menu.append_section(None, &menu_section(&[("history.remove", "win.remove")]));
                (&self.history_menu, &self.history_list)
            }
        };
        popover.set_menu_model(Some(&menu));
        let rect = match at {
            Some((x, y)) => gdk::Rectangle::new(x as i32, y as i32, 1, 1),
            None => match row.compute_bounds(list) {
                Some(bounds) => gdk::Rectangle::new(
                    bounds.x() as i32 + 24,
                    bounds.y() as i32 + bounds.height() as i32 / 2,
                    1,
                    1,
                ),
                None => gdk::Rectangle::new(0, 0, 1, 1),
            },
        };
        popover.set_pointing_to(Some(&rect));
        popover.popup();
    }

    /// The Done item or History entry on the current page.
    pub fn current(&self) -> Option<Current> {
        match self.page.get() {
            Page::Convert => {
                let queue = self.queue.borrow();
                let item = queue.get(queue.selected()?)?;
                match &item.state {
                    State::Done { entry, markdown } => Some(Current {
                        markdown: Some(markdown.clone()),
                        markdown_path: entry.markdown_path.clone(),
                        folder: entry.folder.clone(),
                    }),
                    _ => None,
                }
            }
            Page::History => {
                let id = self.history_selected.get()?;
                let entry = self.history.borrow().get(id)?.clone();
                if !entry.markdown_path.is_file() {
                    return None;
                }
                Some(Current {
                    markdown: self.history_text(id),
                    markdown_path: entry.markdown_path,
                    folder: entry.folder,
                })
            }
        }
    }

    pub fn refresh_actions(&self) {
        let page = self.page.get();
        let (done, retry, remove) = match page {
            Page::Convert => {
                let queue = self.queue.borrow();
                let state = queue.selected().and_then(|id| queue.get(id)).map(|i| &i.state);
                (
                    matches!(state, Some(State::Done { .. })),
                    matches!(state, Some(State::Failed(_))),
                    state.is_some_and(|s| !matches!(s, State::Converting { .. })),
                )
            }
            Page::History => {
                let history = self.history.borrow();
                let entry = self.history_selected.get().and_then(|id| history.get(id));
                (
                    entry.is_some_and(|e| e.markdown_path.is_file()),
                    false,
                    entry.is_some(),
                )
            }
        };
        let enable = |name: &str, on: bool| {
            if let Some(action) = self
                .window
                .lookup_action(name)
                .and_downcast::<gio::SimpleAction>()
            {
                action.set_enabled(on);
            }
        };
        for name in ["copy", "save-as", "show", "toggle-view"] {
            enable(name, done);
        }
        enable("retry", retry);
        enable("remove", remove);
        enable("clear-finished", self.queue.borrow().has_finished());
        enable("clear-history", !self.history.borrow().is_empty());
    }

    pub fn choose_files(self: &Rc<Self>) {
        self.select_page(Page::Convert);
        let dialog = gtk::FileDialog::builder()
            .title(tr("convert.choose"))
            .modal(true)
            .build();
        let win = self.clone();
        glib::spawn_future_local(async move {
            if let Ok(files) = dialog.open_multiple_future(Some(&win.window)).await {
                let files: Vec<gio::File> = files
                    .iter::<gio::File>()
                    .filter_map(Result::ok)
                    .collect();
                win.add_gfiles(&files);
            }
        });
    }

    /// Switch to Convert and focus the URL field.
    pub fn focus_url(&self) {
        self.select_page(Page::Convert);
        self.outer.set_show_content(true);
        if self.queue.borrow().is_empty() {
            self.empty_url.entry.grab_focus();
        } else {
            self.convert_split.set_show_content(false);
            self.list_url.entry.grab_focus();
        }
    }

    /// Files on the clipboard become items, else one web address does.
    pub fn paste(self: &Rc<Self>) {
        let clipboard = self.window.clipboard();
        let win = self.clone();
        glib::spawn_future_local(async move {
            let files = gdk::FileList::static_type();
            if clipboard.formats().contains_type(files)
                && let Ok(value) = clipboard.read_value_future(files, glib::Priority::DEFAULT).await
                && let Ok(list) = value.get::<gdk::FileList>()
            {
                win.add_gfiles(&list.files());
                return;
            }
            if let Ok(Some(text)) = clipboard.read_text_future().await
                && let Some(url) = crate::model::source::parse_web_address(&text)
            {
                win.add_sources(Vec::new(), vec![url]);
            }
        });
    }

    /// Run `then` with the current item's Markdown. History's is read off
    /// the main thread when its detail has not loaded it yet (DESIGN.md 6.4).
    fn with_markdown(self: &Rc<Self>, then: impl FnOnce(&Rc<Win>, Rc<String>, Current) + 'static) {
        let Some(current) = self.current() else {
            return;
        };
        if let Some(markdown) = current.markdown.clone() {
            then(self, markdown, current);
            return;
        }
        let win = self.clone();
        let path = current.markdown_path.clone();
        glib::spawn_future_local(async move {
            let text = gio::spawn_blocking(move || std::fs::read_to_string(path).ok())
                .await
                .ok()
                .flatten();
            match text {
                Some(text) => then(&win, Rc::new(text), current),
                None => {
                    let name = current.markdown_path.file_name().unwrap_or_default().to_string_lossy();
                    win.notice(&trf("error.open_failed", &[("name", &name)]));
                }
            }
        });
    }

    pub fn copy_markdown(self: &Rc<Self>) {
        self.with_markdown(|win, markdown, _| {
            win.window.clipboard().set_text(&markdown);
            win.toast(&tr("notice.copied"), 2);
        });
    }

    /// Save the shown Markdown, and its `.markitai` folder beside it when the
    /// text refers to it (DESIGN.md 7.2).
    pub fn save_as(self: &Rc<Self>) {
        self.with_markdown(|win, markdown, current| win.save_markdown_as(markdown, current));
    }

    fn save_markdown_as(self: &Rc<Self>, markdown: Rc<String>, current: Current) {
        let label = current
            .folder
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "markitai".to_owned());
        let dialog = gtk::FileDialog::builder()
            .title(tr("detail.save_as"))
            .modal(true)
            .initial_name(format!("{label}.md"))
            .build();
        let win = self.clone();
        glib::spawn_future_local(async move {
            let Ok(file) = dialog.save_future(Some(&win.window)).await else {
                return;
            };
            let Some(path) = file.path() else {
                return;
            };
            let markdown = markdown.as_str().to_owned();
            let folder = current.folder.clone();
            let saved = gio::spawn_blocking(move || save_copy(&path, &markdown, &folder))
                .await
                .unwrap_or_else(|_| Err(std::io::Error::other("save thread stopped")));
            if saved.is_err() {
                win.notice(&tr("error.save_failed"));
            }
        });
    }

    pub fn show_in_files(&self, path: &Path) {
        let launcher = gtk::FileLauncher::new(Some(&gio::File::for_path(path)));
        launcher.open_containing_folder(Some(&self.window), gio::Cancellable::NONE, |_| {});
    }

    fn retry(self: &Rc<Self>) {
        if self.page.get() != Page::Convert {
            return;
        }
        let Some(id) = self.queue.borrow().selected() else {
            return;
        };
        let options = self.settings.borrow().options();
        if self.queue.borrow_mut().retry(id, options) {
            self.pump();
        }
    }

    fn remove_selected(self: &Rc<Self>) {
        match self.page.get() {
            Page::Convert => {
                let Some(id) = self.queue.borrow().selected() else {
                    return;
                };
                if self.queue.borrow_mut().remove(id) {
                    self.refresh_convert();
                }
            }
            Page::History => {
                if let Some(id) = self.history_selected.get() {
                    self.remove_history_entry(id);
                }
            }
        }
    }

    fn zoom_by(&self, step: f64) {
        let zoom = (self.settings.borrow().zoom * 10.0).round() / 10.0 + step;
        self.set_zoom(zoom);
    }

    fn set_zoom(&self, zoom: f64) {
        self.settings.borrow_mut().zoom = clamp_zoom(zoom);
        self.save_settings();
        self.apply_zoom();
    }
}

/// Write `markdown` to `path`; when it refers to `.markitai/`, copy the
/// conversion's `.markitai` folder beside it, adding files that are not
/// there and never replacing one.
fn save_copy(path: &Path, markdown: &str, folder: &Path) -> std::io::Result<()> {
    std::fs::write(path, markdown)?;
    let assets = folder.join(".markitai");
    if markdown.contains(".markitai/") && assets.is_dir() {
        let target = path.parent().unwrap_or(Path::new(".")).join(".markitai");
        copy_missing(&assets, &target)?;
    }
    Ok(())
}

fn copy_missing(from: &Path, to: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(to)?;
    for entry in std::fs::read_dir(from)? {
        let entry = entry?;
        let target = to.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_missing(&entry.path(), &target)?;
        } else if !target.exists() {
            std::fs::copy(entry.path(), &target)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn save_as_copies_assets_beside_without_replacing() {
        let dir = tempfile::tempdir().unwrap();
        let folder = dir.path().join("Report");
        std::fs::create_dir_all(folder.join(".markitai/assets")).unwrap();
        std::fs::write(folder.join(".markitai/assets/a.png"), b"new").unwrap();
        std::fs::write(folder.join(".markitai/assets/b.png"), b"new").unwrap();
        let out = dir.path().join("out");
        std::fs::create_dir_all(out.join(".markitai/assets")).unwrap();
        std::fs::write(out.join(".markitai/assets/a.png"), b"mine").unwrap();
        let saved = out.join("Report.md");
        save_copy(&saved, "![a](.markitai/assets/a.png)", &folder).unwrap();
        assert_eq!(std::fs::read(&saved).unwrap(), b"![a](.markitai/assets/a.png)");
        assert_eq!(std::fs::read(out.join(".markitai/assets/a.png")).unwrap(), b"mine");
        assert_eq!(std::fs::read(out.join(".markitai/assets/b.png")).unwrap(), b"new");

        let plain = dir.path().join("plain");
        std::fs::create_dir(&plain).unwrap();
        save_copy(&plain.join("x.md"), "no assets", &folder).unwrap();
        assert!(!plain.join(".markitai").exists());
    }
}
