//! Convert (DESIGN.md 6.2, 6.3 and 7.1): adding sources, running two at a
//! time, and the list column.

use std::path::PathBuf;
use std::rc::Rc;
use std::time::{Instant, SystemTime};

use adw::prelude::*;
use gtk::{gio, glib};
use serde_json::{Value, json};
use uuid::Uuid;

use super::detail::{Body, Shown, elapsed_status};
use super::rows::{Row, Trailing, fade_in, fade_out, row_id};
use super::window::{Page, UrlRow, Win};
use crate::model::errors::error_copy;
use crate::model::format;
use crate::model::history::Entry;
use crate::model::outcome::{CostStatus, Failure, Outcome, convert_request, map_envelope};
use crate::model::queue::{Item, State};
use crate::model::rfc3339_utc;
use crate::model::source::{Added, Source, classify, parse_web_address};
use crate::strings::{lang, plural, settings_name, tr, trf};

/// The sentence for a failure, as the error block and the row say it.
pub fn failure_sentence(failure: &Failure, output_folder: &str) -> String {
    let (key, _) = error_copy(&failure.code);
    match key {
        "error.output_unavailable" => trf(
            "error.output_unavailable",
            &[("folder", output_folder), ("settings", &settings_name())],
        ),
        "error.skipped" => trf(
            "error.skipped",
            &[("reason", failure.message.trim_start_matches("skip_reason: "))],
        ),
        _ => trf(key, &[("settings", &settings_name())]),
    }
}

pub fn done_status(entry: &Entry) -> String {
    trf("state.done", &[("duration", &format::duration(lang(), entry.duration))])
}

/// `Converted in 18.6 s · 24,310 tokens · $0.0412`.
pub fn footer_text(entry: &Entry) -> String {
    let mut parts = vec![trf(
        "footer.duration",
        &[("duration", &format::duration(lang(), entry.duration))],
    )];
    if let Some(usage) = &entry.usage {
        parts.push(trf("footer.tokens", &[("count", &format::grouped(usage.tokens()))]));
        parts.push(match usage.cost {
            CostStatus::Complete => format::usd(usage.cost_usd),
            CostStatus::Partial => trf("footer.cost.at_least", &[("cost", &format::usd(usage.cost_usd))]),
            CostStatus::Unknown => tr("footer.cost.unknown"),
        });
    }
    parts.join(" · ")
}

fn with_warnings(text: String, warnings: usize) -> String {
    if warnings == 0 {
        text
    } else {
        format!("{text} · {}", plural("state.warnings", warnings))
    }
}

impl Win {
    pub(super) fn connect_convert(self: &Rc<Self>) {
        for url in [&self.empty_url, &self.list_url] {
            self.connect_url_row(url);
        }
        let weak = Rc::downgrade(self);
        self.convert_list.connect_row_selected(move |_, row| {
            let Some(win) = weak.upgrade() else {
                return;
            };
            if win.syncing.get() {
                return;
            }
            win.queue.borrow_mut().select(row.and_then(row_id));
            win.refresh_convert_detail();
            win.refresh_actions();
        });
        let weak = Rc::downgrade(self);
        self.convert_list.connect_row_activated(move |_, _| {
            if let Some(win) = weak.upgrade() {
                win.convert_split.set_show_content(true);
            }
        });
        super::actions::connect_context_menu(self, &self.convert_list, Page::Convert);
    }

    fn connect_url_row(self: &Rc<Self>, url: &UrlRow) {
        let add = {
            let weak = Rc::downgrade(self);
            let entry = url.entry.clone();
            let error = url.error.clone();
            move || {
                let Some(win) = weak.upgrade() else {
                    return;
                };
                match parse_web_address(&entry.text()) {
                    Some(address) => {
                        entry.set_text("");
                        win.add_sources(Vec::new(), vec![address]);
                    }
                    None => error.set_visible(true),
                }
            }
        };
        let on_activate = add.clone();
        url.entry.connect_activate(move |_| on_activate());
        url.add.connect_clicked(move |_| add());
        let error = url.error.clone();
        url.entry.connect_changed(move |_| error.set_visible(false));
    }

    /// Files from the file chooser, a drop, the clipboard or `GApplication`
    /// open: local paths and, for dragged links, web addresses.
    pub fn add_gfiles(self: &Rc<Self>, files: &[gio::File]) {
        let mut paths = Vec::new();
        let mut urls = Vec::new();
        for file in files {
            let scheme = file.uri_scheme().map(|s| s.to_ascii_lowercase());
            match scheme.as_deref() {
                Some("http") | Some("https") => urls.push(file.uri().to_string()),
                _ => {
                    if let Some(path) = file.path() {
                        paths.push(path);
                    }
                }
            }
        }
        self.add_sources(paths, urls);
    }

    /// Refuse folders, add the rest newest first, and start what can start.
    pub fn add_sources(self: &Rc<Self>, paths: Vec<PathBuf>, urls: Vec<String>) {
        let mut sources = Vec::new();
        let mut folders = Vec::new();
        for path in paths {
            match classify(&path) {
                Added::File(path) => sources.push(Source::File(path)),
                Added::Folder(name) => folders.push(name),
            }
        }
        sources.extend(urls.into_iter().map(Source::Url));
        self.toasts.dismiss_all();
        match folders.as_slice() {
            [] => {}
            [name] => self.notice(&trf("notice.folder", &[("name", name)])),
            many => self.notice(&trf("notice.folders", &[("count", &many.len().to_string())])),
        }
        if sources.is_empty() {
            return;
        }
        let options = self.settings.borrow().options();
        let added = self.queue.borrow_mut().add(sources, options);
        for name in &added.duplicates {
            self.notice(&trf("notice.duplicate", &[("name", name)]));
        }
        if self.page.get() != Page::Convert {
            self.select_page(Page::Convert);
        }
        self.outer.set_show_content(true);
        self.pump();
    }

    /// Start Queued items while fewer than two convert (DESIGN.md 4.1).
    pub fn pump(self: &Rc<Self>) {
        loop {
            let Some(id) = self.queue.borrow().next_to_start() else {
                break;
            };
            let (source, options) = {
                let queue = self.queue.borrow();
                let item = queue.get(id).expect("queued item");
                (item.source.clone(), item.options)
            };
            let parent = self.output_folder();
            let claim = markitai_app::claim_output_folder(
                &json!({"parent": parent, "source": source.text()}).to_string(),
            );
            let claim: Value = serde_json::from_str(&claim).unwrap_or_default();
            let Some(folder) = claim["result"]["path"].as_str().map(PathBuf::from) else {
                let message = claim["error"]["message"].as_str().unwrap_or_default();
                self.queue
                    .borrow_mut()
                    .fail(id, Failure::new("output_unavailable", message));
                self.announce_finished(id);
                continue;
            };
            self.queue.borrow_mut().start(id, Instant::now(), folder.clone());
            let request = convert_request(&source, &folder, &options);
            let win = self.clone();
            glib::spawn_future_local(async move {
                let claimed = folder.clone();
                let outcome = gio::spawn_blocking(move || {
                    let outcome = map_envelope(&markitai_app::convert_json(&request));
                    if matches!(outcome, Outcome::Failed(_)) {
                        let _ = std::fs::remove_dir_all(&claimed);
                    }
                    outcome
                })
                .await
                .unwrap_or_else(|_| {
                    Outcome::Failed(Failure::new("internal_error", "The conversion thread stopped."))
                });
                win.finish(id, folder, outcome);
            });
        }
        self.refresh_convert();
        self.ensure_ticking();
    }

    fn finish(self: &Rc<Self>, id: Uuid, folder: PathBuf, outcome: Outcome) {
        match outcome {
            Outcome::Done(converted) => {
                let entry = {
                    let queue = self.queue.borrow();
                    let Some(item) = queue.get(id) else {
                        return;
                    };
                    item.entry(folder, &converted, rfc3339_utc(SystemTime::now()))
                };
                self.queue
                    .borrow_mut()
                    .finish(id, entry.clone(), Rc::new(converted.markdown));
                self.history.borrow_mut().add(entry);
                let _ = self.history.borrow().save();
                self.refresh_history();
            }
            Outcome::Failed(failure) => self.queue.borrow_mut().fail(id, failure),
        }
        self.announce_finished(id);
        self.pump();
    }

    fn announce_finished(&self, id: Uuid) {
        let queue = self.queue.borrow();
        let Some(item) = queue.get(id) else {
            return;
        };
        let key = match item.state {
            State::Done { .. } => "a11y.done",
            _ => "a11y.failed",
        };
        self.announce(&trf(key, &[("title", &item.title)]));
    }

    /// The elapsed time in the detail, once a second while anything converts.
    fn ensure_ticking(self: &Rc<Self>) {
        if self.ticking.get() || self.queue.borrow().converting() == 0 {
            return;
        }
        self.ticking.set(true);
        let weak = Rc::downgrade(self);
        glib::timeout_add_seconds_local(1, move || {
            let Some(win) = weak.upgrade() else {
                return glib::ControlFlow::Break;
            };
            win.convert_detail.tick(elapsed_status);
            if win.queue.borrow().converting() == 0 {
                win.ticking.set(false);
                return glib::ControlFlow::Break;
            }
            glib::ControlFlow::Continue
        });
    }

    fn item_row(&self, item: &Item) -> (String, Trailing) {
        match &item.state {
            State::Queued { .. } => (tr("state.queued"), Trailing::Queued),
            State::Converting { .. } => (tr("state.converting"), Trailing::Converting),
            State::Done { entry, .. } => {
                let n = entry.warnings.len();
                let trailing = if n > 0 { Trailing::Warning } else { Trailing::None };
                (with_warnings(done_status(entry), n), trailing)
            }
            State::Failed(failure) => {
                let folder = self.display_path(&self.output_folder());
                let reason = failure_sentence(failure, &folder);
                (trf("row.failed", &[("reason", &reason)]), Trailing::Failed)
            }
        }
    }

    pub fn refresh_convert(&self) {
        self.sync_convert_rows();
        self.refresh_pages();
        self.refresh_convert_detail();
        self.refresh_actions();
    }

    fn sync_convert_rows(&self) {
        let queue = self.queue.borrow();
        let mut rows = self.rows.borrow_mut();
        self.syncing.set(true);
        rows.retain(|id, row| {
            let keep = queue.get(*id).is_some();
            if !keep {
                fade_out(&self.convert_list, &row.row);
            }
            keep
        });
        for (index, item) in queue.items().iter().enumerate() {
            let row = rows.entry(item.id).or_insert_with(|| {
                let row = Row::new(item.id);
                self.convert_list.insert(&row.row, index as i32);
                fade_in(&row.row);
                row
            });
            let (subtitle, trailing) = self.item_row(item);
            row.set(item.source.kind(), &item.title, &subtitle, trailing);
            let busy = matches!(item.state, State::Converting { .. });
            row.row
                .set_tooltip_text(busy.then(|| tr("convert.remove.busy")).as_deref());
        }
        match queue.selected().and_then(|id| rows.get(&id)) {
            Some(row) => {
                if self.convert_list.selected_row().as_ref() != Some(&row.row) {
                    self.convert_list.select_row(Some(&row.row));
                }
            }
            None => self.convert_list.unselect_all(),
        }
        self.syncing.set(false);
    }

    fn refresh_convert_detail(&self) {
        let shown = {
            let queue = self.queue.borrow();
            queue
                .selected()
                .and_then(|id| queue.get(id))
                .map(|item| self.item_shown(item))
        };
        if shown.is_none() {
            self.convert_split.set_show_content(false);
        }
        self.convert_detail.show(shown);
    }

    fn item_shown(&self, item: &Item) -> Shown {
        let source = match &item.source {
            Source::File(path) => self.display_path(path),
            Source::Url(url) => url.clone(),
        };
        let mut shown = Shown {
            id: item.id,
            title: item.title.clone(),
            kind: item.source.kind(),
            source,
            status: String::new(),
            warnings: Vec::new(),
            footer: None,
            body: Body::Queued,
        };
        match &item.state {
            State::Queued { .. } => shown.status = tr("state.queued"),
            State::Converting { started_at, .. } => {
                shown.status = elapsed_status(*started_at);
                shown.body = Body::Converting {
                    started: *started_at,
                };
            }
            State::Done { entry, markdown } => {
                shown.status = done_status(entry);
                shown.warnings = entry.warnings.clone();
                shown.footer = Some(footer_text(entry));
                shown.body = Body::Done {
                    markdown: Some(markdown.clone()),
                    base_dir: entry.folder.clone(),
                };
            }
            State::Failed(failure) => {
                shown.status = tr("state.failed");
                shown.body = Body::Failed {
                    code: failure.code.clone(),
                    message: failure.message.clone(),
                    output_folder: self.display_path(&self.output_folder()),
                };
            }
        }
        shown
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::history::sample;
    use crate::model::outcome::Usage;

    #[test]
    fn the_footer_names_duration_tokens_and_cost() {
        crate::strings::set_lang(crate::strings::Lang::En);
        let mut entry = sample(1, std::path::Path::new("/o"));
        entry.duration = 18.6;
        assert_eq!(footer_text(&entry), "Converted in 18.6 s");
        let usage = |cost| Usage {
            requests: 6,
            input_tokens: 19870,
            output_tokens: 4440,
            cost_usd: 0.0412,
            cost,
        };
        entry.usage = Some(usage(CostStatus::Complete));
        assert_eq!(footer_text(&entry), "Converted in 18.6 s · 24,310 tokens · $0.0412");
        entry.usage = Some(usage(CostStatus::Partial));
        assert!(footer_text(&entry).ends_with("at least $0.0412"));
        entry.usage = Some(usage(CostStatus::Unknown));
        assert!(footer_text(&entry).ends_with("cost unknown"));
    }
}
