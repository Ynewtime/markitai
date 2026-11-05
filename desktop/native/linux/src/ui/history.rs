//! History (DESIGN.md 6.5): search, rows, the detail read from disk, and
//! Clear History.

use std::rc::Rc;

use adw::prelude::*;
use gtk::{gio, glib};
use uuid::Uuid;

use super::convert::{done_status, footer_text};
use super::detail::{Body, Shown};
use super::rows::{Row, Trailing, fade_in, fade_out, row_id};
use super::window::{Page, Win};
use crate::model::history::Entry;
use crate::model::source::Kind;
use crate::strings::{Lang, fill, lang, plural, text_in, tr, trf};

/// When an entry finished, in local time (DESIGN.md 6.5).
pub fn finished_text(finished_at: &str) -> String {
    let local = glib::DateTime::from_iso8601(finished_at, None).and_then(|utc| utc.to_local());
    match (local, glib::DateTime::now_local()) {
        (Ok(local), Ok(now)) => finished_in(lang(), &local, &now),
        _ => finished_at.to_owned(),
    }
}

const MONTHS: [&str; 12] = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

/// `date.today` or `date.yesterday` with the short time, else the medium
/// date and short time, all in `lang` and its clock, whatever the locale's:
/// `Today, 9:05 AM`, `Oct 2, 2026 at 2:05 PM`, `今天 09:05`,
/// `2026年10月2日 14:05`, as Native macOS writes them.
fn finished_in(lang: Lang, at: &glib::DateTime, now: &glib::DateTime) -> String {
    let (hour, minute) = (at.hour(), at.minute());
    let time = match lang {
        Lang::En => {
            let half = if hour < 12 { "AM" } else { "PM" };
            format!("{}:{minute:02}\u{202f}{half}", (hour + 11) % 12 + 1)
        }
        Lang::ZhHans => format!("{hour:02}:{minute:02}"),
    };
    let day = at.ymd();
    if day == now.ymd() {
        return fill(&text_in(lang, "date.today"), &[("time", &time)]);
    }
    if now.add_days(-1).is_ok_and(|yesterday| yesterday.ymd() == day) {
        return fill(&text_in(lang, "date.yesterday"), &[("time", &time)]);
    }
    let (year, month, date) = day;
    match lang {
        Lang::En => format!("{} {date}, {year} at {time}", MONTHS[(month as usize).clamp(1, 12) - 1]),
        Lang::ZhHans => format!("{year}年{month}月{date}日 {time}"),
    }
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
                fade_out(&self.history_list, &row.row);
            }
            keep
        });
        for (index, entry) in history.entries().iter().enumerate() {
            let row = rows.entry(entry.id).or_insert_with(|| {
                let row = Row::new(entry.id);
                self.history_list.insert(&row.row, index as i32);
                fade_in(&row.row);
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
            let text = gio::spawn_blocking(move || crate::model::history::read_markdown(&path))
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
        let query = self.history_search.text();
        let next = self.history.borrow().neighbor(id, |entry| entry.matches(&query));
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

#[cfg(test)]
mod tests {
    use super::*;

    fn at(day: i32, hour: i32, minute: i32) -> glib::DateTime {
        glib::DateTime::new(&glib::TimeZone::utc(), 2026, 10, day, hour, minute, 0.0).unwrap()
    }

    #[test]
    fn times_follow_the_app_language_and_its_clock() {
        let now = at(5, 18, 0);
        let en = |when: &glib::DateTime| finished_in(Lang::En, when, &now);
        let zh = |when: &glib::DateTime| finished_in(Lang::ZhHans, when, &now);
        assert_eq!(en(&at(5, 9, 5)), "Today, 9:05\u{202f}AM");
        assert_eq!(en(&at(4, 22, 42)), "Yesterday, 10:42\u{202f}PM");
        assert_eq!(en(&at(2, 14, 5)), "Oct 2, 2026 at 2:05\u{202f}PM");
        assert_eq!(en(&at(2, 0, 7)), "Oct 2, 2026 at 12:07\u{202f}AM");
        assert_eq!(en(&at(2, 12, 0)), "Oct 2, 2026 at 12:00\u{202f}PM");
        assert_eq!(zh(&at(5, 9, 5)), "今天 09:05");
        assert_eq!(zh(&at(4, 22, 42)), "昨天 22:42");
        assert_eq!(zh(&at(2, 14, 5)), "2026年10月2日 14:05");
    }
}
