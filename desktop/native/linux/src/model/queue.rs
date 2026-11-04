//! The Convert list (DESIGN.md 4.1): items, their states, two conversions at a
//! time, and the single selection. It lives in memory only.

use std::path::PathBuf;
use std::rc::Rc;
use std::time::Instant;

use uuid::Uuid;

use super::history::Entry;
use super::outcome::{Converted, Failure};
use super::settings::Options;
use super::source::Source;

pub const SLOTS: usize = 2;

#[derive(Clone, Debug)]
pub enum State {
    /// `ticket` orders the queue: lower starts first.
    Queued { ticket: u64 },
    /// `folder` is the one claimed for it (DESIGN.md 4.1 step 2).
    Converting { started_at: Instant, folder: PathBuf },
    Done { entry: Box<Entry>, markdown: Rc<String> },
    Failed(Failure),
}

impl State {
    pub fn is_finished(&self) -> bool {
        matches!(self, State::Done { .. } | State::Failed(_))
    }

    pub fn is_active(&self) -> bool {
        matches!(self, State::Queued { .. } | State::Converting { .. })
    }
}

#[derive(Clone, Debug)]
pub struct Item {
    pub id: Uuid,
    pub source: Source,
    pub title: String,
    pub options: Options,
    pub state: State,
}

impl Item {
    /// The History entry for this item once its conversion is Done. A URL's
    /// title is the document's own, when it has one.
    pub fn entry(&self, folder: PathBuf, converted: &Converted, finished_at: String) -> Entry {
        let title = match (&self.source, &converted.title) {
            (Source::Url(_), Some(title)) => title.clone(),
            _ => self.title.clone(),
        };
        Entry {
            id: self.id,
            source: self.source.text(),
            kind: self.source.kind(),
            title,
            folder,
            markdown_path: converted.markdown_path.clone(),
            finished_at,
            duration: converted.duration,
            warnings: converted.warnings.clone(),
            usage: converted.usage.clone(),
            options: self.options,
            markitai_version: markitai_app::version().to_owned(),
        }
    }
}

/// What an add did: the new items, and the titles of sources already
/// waiting or converting.
#[derive(Debug, Default)]
pub struct Added {
    pub ids: Vec<Uuid>,
    pub duplicates: Vec<String>,
}

#[derive(Default)]
pub struct Queue {
    items: Vec<Item>,
    selected: Option<Uuid>,
    next_ticket: u64,
}

impl Queue {
    /// Newest first; sources added together keep their order. The first item
    /// added to an empty list is selected.
    pub fn add(&mut self, sources: Vec<Source>, options: Options) -> Added {
        let was_empty = self.items.is_empty();
        let mut added = Added::default();
        let mut batch: Vec<Item> = Vec::new();
        for source in sources {
            let busy = self
                .items
                .iter()
                .chain(batch.iter())
                .any(|item| item.source == source && item.state.is_active());
            if busy {
                added.duplicates.push(source.initial_title());
                continue;
            }
            let item = Item {
                id: Uuid::new_v4(),
                title: source.initial_title(),
                source,
                options,
                state: State::Queued {
                    ticket: self.ticket(),
                },
            };
            added.ids.push(item.id);
            batch.push(item);
        }
        self.items.splice(0..0, batch);
        if was_empty && self.selected.is_none() {
            self.selected = added.ids.first().copied();
        }
        added
    }

    fn ticket(&mut self) -> u64 {
        self.next_ticket += 1;
        self.next_ticket
    }

    pub fn items(&self) -> &[Item] {
        &self.items
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn get(&self, id: Uuid) -> Option<&Item> {
        self.items.iter().find(|i| i.id == id)
    }

    fn get_mut(&mut self, id: Uuid) -> Option<&mut Item> {
        self.items.iter_mut().find(|i| i.id == id)
    }

    pub fn position(&self, id: Uuid) -> Option<usize> {
        self.items.iter().position(|i| i.id == id)
    }

    pub fn selected(&self) -> Option<Uuid> {
        self.selected
    }

    pub fn select(&mut self, id: Option<Uuid>) {
        self.selected = id.filter(|id| self.get(*id).is_some());
    }

    pub fn converting(&self) -> usize {
        self.items
            .iter()
            .filter(|i| matches!(i.state, State::Converting { .. }))
            .count()
    }

    pub fn queued(&self) -> usize {
        self.items
            .iter()
            .filter(|i| matches!(i.state, State::Queued { .. }))
            .count()
    }

    pub fn has_finished(&self) -> bool {
        self.items.iter().any(|i| i.state.is_finished())
    }

    /// The Queued item that starts next, while fewer than two convert.
    pub fn next_to_start(&self) -> Option<Uuid> {
        if self.converting() >= SLOTS {
            return None;
        }
        self.items
            .iter()
            .filter_map(|i| match i.state {
                State::Queued { ticket } => Some((ticket, i.id)),
                _ => None,
            })
            .min()
            .map(|(_, id)| id)
    }

    pub fn start(&mut self, id: Uuid, now: Instant, folder: PathBuf) {
        if let Some(item) = self.get_mut(id)
            && matches!(item.state, State::Queued { .. })
        {
            item.state = State::Converting { started_at: now, folder };
        }
    }

    /// The folders claimed for the items still Converting, which a confirmed
    /// Quit deletes (DESIGN.md 7.3).
    pub fn converting_folders(&self) -> Vec<PathBuf> {
        self.items
            .iter()
            .filter_map(|i| match &i.state {
                State::Converting { folder, .. } => Some(folder.clone()),
                _ => None,
            })
            .collect()
    }

    /// Converting (or Queued, when the folder claim failed) to Failed.
    pub fn fail(&mut self, id: Uuid, failure: Failure) {
        if let Some(item) = self.get_mut(id)
            && item.state.is_active()
        {
            item.state = State::Failed(failure);
        }
    }

    /// Converting to Done. A URL item takes the document's title.
    pub fn finish(&mut self, id: Uuid, entry: Entry, markdown: Rc<String>) {
        if let Some(item) = self.get_mut(id)
            && matches!(item.state, State::Converting { .. })
        {
            item.title = entry.title.clone();
            item.state = State::Done {
                entry: Box::new(entry),
                markdown,
            };
        }
    }

    /// Failed to Queued with the current settings.
    pub fn retry(&mut self, id: Uuid, options: Options) -> bool {
        let ticket = self.next_ticket + 1;
        let Some(item) = self.get_mut(id) else {
            return false;
        };
        if !matches!(item.state, State::Failed(_)) {
            return false;
        }
        item.options = options;
        item.state = State::Queued { ticket };
        self.next_ticket = ticket;
        true
    }

    /// Queued, Done and Failed items can be removed; Converting ones cannot.
    pub fn remove(&mut self, id: Uuid) -> bool {
        let removable = self
            .get(id)
            .is_some_and(|i| !matches!(i.state, State::Converting { .. }));
        if removable {
            self.remove_where(|item| item.id == id);
        }
        removable
    }

    /// Remove every Done and Failed item; answers the removed ids.
    pub fn clear_finished(&mut self) -> Vec<Uuid> {
        self.remove_where(|item| item.state.is_finished())
    }

    /// When the selected item goes, the selection moves to the next row,
    /// else the previous one.
    fn remove_where(&mut self, gone: impl Fn(&Item) -> bool) -> Vec<Uuid> {
        let selected_at = self.selected.and_then(|id| self.position(id));
        let next = selected_at.and_then(|at| {
            if !gone(&self.items[at]) {
                return Some(self.items[at].id);
            }
            let after = self.items[at + 1..].iter().find(|i| !gone(i));
            let before = self.items[..at].iter().rev().find(|i| !gone(i));
            after.or(before).map(|i| i.id)
        });
        let removed = self
            .items
            .iter()
            .filter(|i| gone(i))
            .map(|i| i.id)
            .collect();
        self.items.retain(|i| !gone(i));
        self.selected = next;
        removed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::history::sample;

    fn file(name: &str) -> Source {
        Source::File(format!("/in/{name}").into())
    }

    fn titles(queue: &Queue) -> Vec<&str> {
        queue.items().iter().map(|i| i.title.as_str()).collect()
    }

    fn done(queue: &mut Queue, id: Uuid) {
        queue.start(id, Instant::now(), PathBuf::from("/out"));
        queue.finish(id, sample(1, std::path::Path::new("/out/x")), Rc::new("# x".to_owned()));
    }

    #[test]
    fn newest_first_but_batches_keep_their_order_and_oldest_starts_first() {
        let mut queue = Queue::default();
        let first = queue.add(vec![file("a.pdf"), file("b.pdf")], Options::default());
        queue.add(vec![file("c.pdf")], Options::default());
        assert_eq!(titles(&queue), ["c.pdf", "a.pdf", "b.pdf"]);
        assert_eq!(queue.selected(), Some(first.ids[0]));
        let a = queue.next_to_start().unwrap();
        assert_eq!(queue.get(a).unwrap().title, "a.pdf");
        queue.start(a, Instant::now(), PathBuf::from("/out/a"));
        let b = queue.next_to_start().unwrap();
        assert_eq!(queue.get(b).unwrap().title, "b.pdf");
        queue.start(b, Instant::now(), PathBuf::from("/out/b"));
        assert_eq!(queue.next_to_start(), None, "two at a time");
        assert_eq!(queue.converting_folders(), [PathBuf::from("/out/a"), PathBuf::from("/out/b")]);
        queue.fail(a, Failure::new("not_found", "gone"));
        assert_eq!(queue.converting_folders(), [PathBuf::from("/out/b")], "a quit deletes only these");
        assert_eq!(queue.get(queue.next_to_start().unwrap()).unwrap().title, "c.pdf");
    }

    #[test]
    fn waiting_or_converting_sources_are_duplicates_but_finished_ones_convert_again() {
        let mut queue = Queue::default();
        let added = queue.add(vec![file("a.pdf"), file("a.pdf")], Options::default());
        assert_eq!(added.ids.len(), 1);
        assert_eq!(added.duplicates, ["a.pdf"]);
        let id = added.ids[0];
        queue.start(id, Instant::now(), PathBuf::from("/out"));
        assert_eq!(queue.add(vec![file("a.pdf")], Options::default()).duplicates.len(), 1);
        queue.fail(id, Failure::new("conversion_error", "bad"));
        assert_eq!(queue.add(vec![file("a.pdf")], Options::default()).ids.len(), 1);
        assert_eq!(queue.items().len(), 2);
    }

    #[test]
    fn converting_items_cannot_be_removed_and_selection_moves_next_then_previous() {
        let mut queue = Queue::default();
        let ids = queue.add(vec![file("a"), file("b"), file("c")], Options::default()).ids;
        queue.start(ids[0], Instant::now(), PathBuf::from("/out"));
        assert!(!queue.remove(ids[0]));
        queue.select(Some(ids[1]));
        assert!(queue.remove(ids[1]));
        assert_eq!(queue.selected(), Some(ids[2]));
        assert!(queue.remove(ids[2]));
        assert_eq!(queue.selected(), Some(ids[0]));
    }

    #[test]
    fn clear_finished_keeps_active_items_and_retry_requeues_with_new_options() {
        let mut queue = Queue::default();
        let ids = queue.add(vec![file("a"), file("b"), file("c")], Options::default()).ids;
        done(&mut queue, ids[0]);
        queue.start(ids[1], Instant::now(), PathBuf::from("/out"));
        queue.fail(ids[1], Failure::new("io_error", "disk"));
        assert!(queue.has_finished());
        let llm = Options { llm: true, ..Options::default() };
        assert!(!queue.retry(ids[0], llm), "only failed items retry");
        assert!(queue.retry(ids[1], llm));
        assert!(queue.get(ids[1]).unwrap().options.llm);
        let ids_left_queued = queue.next_to_start();
        assert_eq!(ids_left_queued, Some(ids[2]), "the retry waits behind earlier items");
        queue.select(Some(ids[0]));
        assert_eq!(queue.clear_finished(), vec![ids[0]]);
        assert_eq!(queue.selected(), Some(ids[1]));
        assert_eq!(queue.items().len(), 2);
    }

    #[test]
    fn a_done_url_takes_the_document_title() {
        let mut queue = Queue::default();
        let id = queue.add(vec![Source::Url("https://example.com/post".into())], Options::default()).ids[0];
        assert_eq!(queue.get(id).unwrap().title, "example.com/post");
        done(&mut queue, id);
        assert_eq!(queue.get(id).unwrap().title, "Report 1.pdf");
    }
}
