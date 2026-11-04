//! Preferences (DESIGN.md 6.6): Conversion, Output, Configuration, About in
//! one `AdwPreferencesDialog`.

use std::path::{Path, PathBuf};
use std::rc::Rc;

use adw::prelude::*;
use gtk::{gio, glib};
use serde_json::Value;

use super::window::Win;
use crate::model::APP_ID;
use crate::model::settings::{ConfigSource, Settings};
use crate::strings::tr;

/// `config_file()`: the markitai configuration file and whether it exists.
fn config_file() -> (PathBuf, bool) {
    let value: Value = serde_json::from_str(&markitai_app::config_file()).unwrap_or_default();
    let path = PathBuf::from(value["result"]["path"].as_str().unwrap_or_default());
    (path, value["result"]["exists"] == Value::Bool(true))
}

/// Whether a test file can be created and deleted in `folder`.
fn writable(folder: &Path) -> bool {
    let probe = folder.join(format!(".markitai-write-test-{}", uuid::Uuid::new_v4()));
    let ok = std::fs::write(&probe, b"").is_ok();
    ok && std::fs::remove_file(&probe).is_ok()
}

fn property_row(title: &str, value: &str) -> adw::ActionRow {
    let row = adw::ActionRow::builder()
        .title(title)
        .subtitle(value)
        .subtitle_selectable(true)
        .build();
    row.add_css_class("property");
    row
}

pub fn present(win: &Rc<Win>) {
    // One dialog at a time: a second Ctrl+, does not stack another.
    if win.window.visible_dialog().is_some() {
        return;
    }
    let dialog = adw::PreferencesDialog::builder()
        .title(tr("nav.preferences"))
        .search_enabled(false)
        .build();
    let page = adw::PreferencesPage::new();
    dialog.add(&page);

    let conversion = adw::PreferencesGroup::builder()
        .title(tr("settings.conversion"))
        .build();
    let settings = win.settings.borrow().clone();
    type Field = fn(&mut Settings) -> &mut bool;
    let switches: [(&str, &str, Field); 5] = [
        ("settings.llm", "settings.llm.note", |s| &mut s.llm),
        ("settings.ocr", "settings.ocr.note", |s| &mut s.ocr),
        ("settings.screenshot", "settings.screenshot.note", |s| &mut s.screenshot),
        ("settings.desc", "settings.desc.note", |s| &mut s.desc),
        ("settings.alt", "settings.alt.note", |s| &mut s.alt),
    ];
    let mut needs_llm = Vec::new();
    let mut llm_row = None;
    for (key, note, field) in switches {
        let mut current = settings.clone();
        let row = adw::SwitchRow::builder()
            .title(tr(key))
            .subtitle(tr(note))
            .active(*field(&mut current))
            .build();
        let weak = Rc::downgrade(win);
        row.connect_active_notify(move |row| {
            if let Some(win) = weak.upgrade() {
                *field(&mut win.settings.borrow_mut()) = row.is_active();
                win.save_settings();
            }
        });
        conversion.add(&row);
        match key {
            "settings.llm" => llm_row = Some(row),
            "settings.desc" | "settings.alt" => needs_llm.push(row),
            _ => {}
        }
    }
    if let Some(llm) = llm_row {
        let sync = move |llm: &adw::SwitchRow| {
            for row in &needs_llm {
                row.set_sensitive(llm.is_active());
            }
        };
        sync(&llm);
        llm.connect_active_notify(sync);
    }
    page.add(&conversion);

    let output = adw::PreferencesGroup::builder()
        .title(tr("settings.output"))
        .description(tr("settings.output.note"))
        .build();
    let folder_row = adw::ActionRow::builder()
        .title(tr("settings.output.folder"))
        .subtitle(win.display_path(&win.output_folder()))
        .subtitle_selectable(true)
        .build();
    let change = gtk::Button::builder()
        .label(tr("settings.output.change"))
        .valign(gtk::Align::Center)
        .build();
    let show = gtk::Button::builder()
        .icon_name("folder-open-symbolic")
        .tooltip_text(tr("detail.show.gnome"))
        .valign(gtk::Align::Center)
        .css_classes(["flat"])
        .build();
    show.update_property(&[gtk::accessible::Property::Label(&tr("detail.show.gnome"))]);
    folder_row.add_suffix(&change);
    folder_row.add_suffix(&show);
    output.add(&folder_row);
    page.add(&output);
    {
        let weak = Rc::downgrade(win);
        let dialog = dialog.clone();
        let folder_row = folder_row.clone();
        change.connect_clicked(move |_| {
            let Some(win) = weak.upgrade() else {
                return;
            };
            let chooser = gtk::FileDialog::builder()
                .title(tr("settings.output.folder"))
                .modal(true)
                .initial_folder(&gio::File::for_path(win.output_folder()))
                .build();
            let dialog = dialog.clone();
            let folder_row = folder_row.clone();
            glib::spawn_future_local(async move {
                let Ok(chosen) = chooser.select_folder_future(Some(&win.window)).await else {
                    return;
                };
                let Some(path) = chosen.path() else {
                    return;
                };
                if !writable(&path) {
                    dialog.add_toast(
                        adw::Toast::builder()
                            .title(tr("settings.output.unwritable"))
                            .use_markup(false)
                            .timeout(5)
                            .build(),
                    );
                    return;
                }
                win.settings.borrow_mut().output_folder = Some(path);
                win.save_settings();
                folder_row.set_subtitle(&win.display_path(&win.output_folder()));
                win.refresh_output_folder();
            });
        });
    }
    {
        let weak = Rc::downgrade(win);
        show.connect_clicked(move |_| {
            if let Some(win) = weak.upgrade() {
                let folder = win.output_folder();
                let _ = std::fs::create_dir_all(&folder);
                win.launch_file(&folder);
            }
        });
    }

    let configuration = adw::PreferencesGroup::builder()
        .title(tr("settings.config"))
        .build();
    let (config_path, exists) = config_file();
    let user_check = gtk::CheckButton::new();
    let builtin_check = gtk::CheckButton::new();
    builtin_check.set_group(Some(&user_check));
    let user_note = if exists {
        win.display_path(&config_path)
    } else {
        tr("settings.config.user.missing")
    };
    let user_row = adw::ActionRow::builder()
        .title(tr("settings.config.user"))
        .subtitle(user_note)
        .activatable_widget(&user_check)
        .build();
    user_row.add_prefix(&user_check);
    let builtin_row = adw::ActionRow::builder()
        .title(tr("settings.config.builtin"))
        .subtitle(tr("settings.config.builtin.note"))
        .activatable_widget(&builtin_check)
        .build();
    builtin_row.add_prefix(&builtin_check);
    match settings.config {
        ConfigSource::User => user_check.set_active(true),
        ConfigSource::Builtin => builtin_check.set_active(true),
    }
    let weak = Rc::downgrade(win);
    user_check.connect_toggled(move |check| {
        if let Some(win) = weak.upgrade() {
            win.settings.borrow_mut().config = if check.is_active() {
                ConfigSource::User
            } else {
                ConfigSource::Builtin
            };
            win.save_settings();
        }
    });
    let open_row = adw::ButtonRow::builder()
        .title(tr(if exists {
            "settings.config.open"
        } else {
            "settings.config.create"
        }))
        .build();
    let weak = Rc::downgrade(win);
    open_row.connect_activated(move |_| {
        if let Some(win) = weak.upgrade() {
            win.open_config();
        }
    });
    configuration.add(&user_row);
    configuration.add(&builtin_row);
    configuration.add(&open_row);
    page.add(&configuration);

    let about = adw::PreferencesGroup::builder()
        .title(tr("settings.about"))
        .build();
    let name_row = adw::ActionRow::builder().title(tr("app.name")).build();
    let icon = gtk::Image::builder().icon_name(APP_ID).pixel_size(32).build();
    name_row.add_prefix(&icon);
    about.add(&name_row);
    about.add(&property_row(&tr("settings.about.app"), env!("CARGO_PKG_VERSION")));
    about.add(&property_row(&tr("settings.about.core"), markitai_app::version()));
    about.add(&adw::ActionRow::builder().title(tr("settings.about.copyright")).build());
    page.add(&about);

    dialog.present(Some(&win.window));
}

impl Win {
    /// Open the markitai configuration file, creating it as `{}` first when
    /// it does not exist (DESIGN.md 6.6).
    pub fn open_config(&self) {
        let (path, exists) = config_file();
        if !exists {
            let created = path
                .parent()
                .map_or(Ok(()), std::fs::create_dir_all)
                .and_then(|()| std::fs::write(&path, b"{}\n"));
            if created.is_err() {
                let name = self.display_path(&path);
                self.notice(&crate::strings::trf("error.open_failed", &[("name", &name)]));
                return;
            }
        }
        self.launch_file(&path);
    }

    /// Open a file or folder in its default app; say so when that fails.
    pub fn launch_file(&self, path: &Path) {
        let launcher = gtk::FileLauncher::new(Some(&gio::File::for_path(path)));
        let name = self.display_path(path);
        let toasts = self.toasts.clone();
        launcher.launch(Some(&self.window), gio::Cancellable::NONE, move |result| {
            if result.is_err() {
                toasts.add_toast(
                    adw::Toast::builder()
                        .title(crate::strings::trf("error.open_failed", &[("name", &name)]))
                        .use_markup(false)
                        .timeout(5)
                        .build(),
                );
            }
        });
    }
}
