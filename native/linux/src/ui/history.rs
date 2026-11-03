//! History (DESIGN.md 6.5): search, rows, the detail read from disk, and
//! Clear History.

use std::rc::Rc;

use adw::prelude::*;
use gtk::{gio, glib};
use uuid::Uuid;

use super::convert::{done_status, footer_text};
use super::detail::{Body, Shown};
use super::rows::{Row, Trailing, row_id};
use super::window::{Page, Win};
use crate::model::history::Entry;
use crate::model::source::Kind;
use crate::strings::{Lang, lang, plural, tr, trf};

/// `Today, 10:42`, `Yesterday, 10:42`, or the date and time.
pub fn finished_text(finished_at: &str) -> String {
    let Ok(utc) = glib::DateTime::from_iso8601(finished_at, None) else {
        return finished_at.to_owned();
    };
    let Ok(local) = utc.to_local() else {
        return finished_at.to_owned();
    };
    let time = local.format("%R").map(|t| t.to_string()).unwrap_or_default();
    let today = glib::DateTime::now_local().ok();
    let same_day = |a: &glib::DateTime, b: &glib::DateTime| a.ymd() == b.ymd();
    if let Some(today) = today {
        if same_day(&local, &today) {
            return trf("date.today", &[("time", &time)]);
        }
        if today
            .add_days(-1)
            .is_ok_and(|yesterday| same_day(&local, &yesterday))
        {
            return trf("date.yesterday", &[("time", &time)]);
        }
    }
    let pattern = match lang() {
        Lang::En => "%b %-d, %Y, %R",
        Lang::ZhHans => "%Y/%-m/%-d %R",
    };
    local
        .format(pattern)
        .map(|t| t.to_string())
        .unwrap_or_else(|_| finished_at.to_owned())
}

fn entry_subtitle(entry: &Entry) -> String {
    let date = finished_text(&entry.finished_at);
    match entry.warnings.len() {
        0 => date,
        n => format!("{date} · {}", plural("state.warnings", n)),
    }
}

impl Win {
    pub(super) fn connect_history(self: &Rc<Self>) {
        let weak = Rc::downgrade(self);
        self.history_list.set_filter_func(move |row| {
            let Some(win) = weak.upgrade() else {
                return true;
            };
            let query = win.history_search.text();
            let history = win.history.borrow();
            row_id(row)
                .and_then(|id| history.get(id))
                .is_some_and(|entry| entry.matches(&query))
        });
        let weak = Rc::downgrade(self);
        self.history_search.connect_search_changed(move |search| {
            if let Some(win) = weak.upgrade() {
                win.history_placeholder
                    .set_label(&trf("history.no_results", &[("query", search.text().trim())]));
                win.history_list.invalidate_filter();
            }
        });
        let weak = Rc::downgrade(self);
        self.history_list.connect_row_selected(move |_, row| {
            let Some(win) = weak.upgrade() else {
                return;
            };
            if win.syncing.get() {
                return;
            }
            win.history_selected.set(row.and_then(row_id));
            win.refresh_history_detail();
            win.refresh_actions();
        });
        let weak = Rc::downgrade(self);
        self.history_list.connect_row_activated(move |_, _| {
            if let Some(win) = weak.upgrade() {
                win.history_split.set_show_content(true);
            }
        });
        super::actions::connect_context_menu(self, &self.history_list, Page::History);
    }

    pub fn refresh_history(self: &Rc<Self>) {
        self.sync_history_rows();
        self.refresh_pages();
        self.refresh_history_detail();
        self.refresh_actions();
    }

    fn sync_history_rows(&self) {
        let history = self.history.borrow();
        let mut rows = self.history_rows.borrow_mut();
        self.syncing.set(true);
        rows.retain(|id, row| {
            let keep = history.get(*id).is_some();
            if !keep {
                self.history_list.remove(&row.row);
            }
            keep
        });
        for (index, entry) in history.entries().iter().enumerate() {
            let row = rows.entry(entry.id).or_insert_with(|| {
                let row = Row::new(entry.id);
                self.history_list.insert(&row.row, index as i32);
                row
            });
            let trailing = if entry.warnings.is_empty() {
                Trailing::None
            } else {
                Trailing::Warning
            };
            row.set(entry.kind, &entry.title, &entry_subtitle(entry), trailing);
        }
        let selected = self.history_selected.get().filter(|id| history.get(*id).is_some());
        self.history_selected.set(selected);
        match selected.and_then(|id| rows.get(&id)) {
            Some(row) => {
                if self.history_list.selected_row().as_ref() != Some(&row.row) {
                    self.history_list.select_row(Some(&row.row));
                }
            }
            None => self.history_list.unselect_all(),
        }
        self.syncing.set(false);
    }

    pub fn refresh_history_detail(self: &Rc<Self>) {
        let entry = self
            .history_selected
            .get()
            .and_then(|id| self.history.borrow().get(id).cloned());
        let Some(entry) = entry else {
            self.history_split.set_show_content(false);
            self.history_detail.show(None);
            return;
        };
        let source = match entry.kind {
            Kind::File => self.display_path(std::path::Path::new(&entry.source)),
            Kind::Url => entry.source.clone(),
        };
        let body = if !entry.markdown_path.is_file() {
            Body::Missing {
                path: self.display_path(&entry.markdown_path),
            }
        } else {
            match self.history_markdown.borrow().as_ref() {
                Some((id, Some(markdown))) if *id == entry.id => Body::Done {
                    markdown: Some(markdown.clone()),
                    base_dir: entry.folder.clone(),
                },
                Some((id, None)) if *id == entry.id => Body::Missing {
                    path: self.display_path(&entry.markdown_path),
                },
                _ => {
                    self.load_markdown(&entry);
                    Body::Done {
                        markdown: None,
                        base_dir: entry.folder.clone(),
                    }
                }
            }
        };
        let missing = matches!(body, Body::Missing { .. });
        self.history_detail.show(Some(Shown {
            id: entry.id,
            title: entry.title.clone(),
            kind: entry.kind,
            source,
            status: format!("{} · {}", done_status(&entry), finished_text(&entry.finished_at)),
            warnings: entry.warnings.clone(),
            footer: (!missing).then(|| footer_text(&entry)),
            body,
        }));
    }

    /// Read the entry's Markdown off the main thread.
    fn load_markdown(self: &Rc<Self>, entry: &Entry) {
        if self.history_loading.get() == Some(entry.id) {
            return;
        }
        self.history_loading.set(Some(entry.id));
        let id = entry.id;
        let path = entry.markdown_path.clone();
        let win = self.clone();
        glib::spawn_future_local(async move {
            let text = gio::spawn_blocking(move || std::fs::read_to_string(path).ok())
                .await
                .ok()
                .flatten();
            win.history_loading.set(None);
            *win.history_markdown.borrow_mut() = Some((id, text.map(Rc::new)));
            if win.history_selected.get() == Some(id) {
                win.refresh_history_detail();
                win.refresh_actions();
            }
        });
    }

    /// The selected entry's Markdown when it has been read.
    pub fn history_text(&self, id: Uuid) -> Option<Rc<String>> {
        match self.history_markdown.borrow().as_ref() {
            Some((loaded, Some(text))) if *loaded == id => Some(text.clone()),
            _ => None,
        }
    }

    pub fn remove_history_entry(self: &Rc<Self>, id: Uuid) {
        let next = {
            let history = self.history.borrow();
            let entries = history.entries();
            let at = entries.iter().position(|e| e.id == id);
            at.and_then(|at| {
                entries
                    .get(at + 1)
                    .or_else(|| at.checked_sub(1).and_then(|i| entries.get(i)))
                    .map(|e| e.id)
            })
        };
        if self.history.borrow_mut().remove(id) {
            let _ = self.history.borrow().save();
            if self.history_selected.get() == Some(id) {
                self.history_selected.set(next);
            }
            self.refresh_history();
        }
    }

    /// Clear History after the confirmation (DESIGN.md 6.7).
    pub fn ask_to_clear_history(self: &Rc<Self>) {
        let count = self.history.borrow().entries().len();
        if count == 0 || self.window.visible_dialog().is_some() {
            return;
        }
        let dialog = adw::AlertDialog::new(
            Some(&plural("history.clear.title", count)),
            Some(&tr("history.clear.body")),
        );
        dialog.add_responses(&[
            ("cancel", &tr("common.cancel")),
            ("clear", &tr("history.clear.confirm")),
        ]);
        dialog.set_response_appearance("clear", adw::ResponseAppearance::Destructive);
        dialog.set_default_response(Some("cancel"));
        dialog.set_close_response("cancel");
        let win = self.clone();
        dialog.connect_response(None, move |_, response| {
            if response == "clear" {
                win.history.borrow_mut().clear();
                let _ = win.history.borrow().save();
                win.history_selected.set(None);
                win.refresh_history();
            }
        });
        dialog.present(Some(&self.window));
    }
}
