//! Finished conversions, newest first, in
//! `$XDG_DATA_HOME/dev.markitai.Markitai/history.json` (DESIGN.md 4.4).

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::outcome::Usage;
use super::settings::Options;
use super::source::Kind;
use super::write_atomically;

pub const LIMIT: usize = 500;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Entry {
    pub id: Uuid,
    pub source: String,
    pub kind: Kind,
    pub title: String,
    pub folder: PathBuf,
    pub markdown_path: PathBuf,
    pub finished_at: String,
    pub duration: f64,
    pub warnings: Vec<String>,
    pub usage: Option<Usage>,
    pub options: Options,
    pub markitai_version: String,
}

impl Entry {
    /// Whether the title or source contains `query`, ignoring case.
    pub fn matches(&self, query: &str) -> bool {
        let query = query.trim().to_lowercase();
        query.is_empty()
            || self.title.to_lowercase().contains(&query)
            || self.source.to_lowercase().contains(&query)
    }
}

#[derive(Serialize, Deserialize)]
struct File {
    version: u32,
    entries: Vec<Entry>,
}

pub struct History {
    path: PathBuf,
    entries: Vec<Entry>,
}

impl History {
    /// Read `history.json` in `dir`. A file that is not valid JSON or not
    /// version 1 is renamed to `history-unreadable-<stamp>.json`, and its new
    /// name is returned so the app can say so once.
    pub fn load(dir: &Path, stamp: &str) -> (History, Option<String>) {
        let path = dir.join("history.json");
        let mut history = History {
            path: path.clone(),
            entries: Vec::new(),
        };
        let Ok(bytes) = std::fs::read(&path) else {
            return (history, None);
        };
        match serde_json::from_slice::<File>(&bytes) {
            Ok(file) if file.version == 1 => {
                history.entries = file.entries;
                history.entries.truncate(LIMIT);
                (history, None)
            }
            _ => {
                let name = format!("history-unreadable-{stamp}.json");
                match std::fs::rename(&path, dir.join(&name)) {
                    Ok(()) => (history, Some(name)),
                    Err(_) => (history, None),
                }
            }
        }
    }

    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }

    pub fn get(&self, id: Uuid) -> Option<&Entry> {
        self.entries.iter().find(|e| e.id == id)
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn add(&mut self, entry: Entry) {
        self.entries.insert(0, entry);
        self.entries.truncate(LIMIT);
    }

    /// The entry to select once `id` is removed: the next one `shown`, else
    /// the previous one.
    pub fn neighbor(&self, id: Uuid, shown: impl Fn(&Entry) -> bool) -> Option<Uuid> {
        let at = self.entries.iter().position(|e| e.id == id)?;
        self.entries[at + 1..]
            .iter()
            .find(|e| shown(e))
            .or_else(|| self.entries[..at].iter().rev().find(|e| shown(e)))
            .map(|e| e.id)
    }

    pub fn remove(&mut self, id: Uuid) -> bool {
        let before = self.entries.len();
        self.entries.retain(|e| e.id != id);
        self.entries.len() != before
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }

    pub fn save(&self) -> std::io::Result<()> {
        let file = File {
            version: 1,
            entries: self.entries.clone(),
        };
        let mut text = serde_json::to_string_pretty(&file).map_err(std::io::Error::other)?;
        text.push('\n');
        write_atomically(&self.path, text.as_bytes())
    }

    #[cfg(test)]
    fn matching(&self, query: &str) -> Vec<&Entry> {
        self.entries.iter().filter(|e| e.matches(query)).collect()
    }
}

#[cfg(test)]
pub(crate) fn sample(n: usize, folder: &Path) -> Entry {
    use super::settings::ConfigSource;
    Entry {
        id: Uuid::new_v4(),
        source: format!("/in/Report {n}.pdf"),
        kind: Kind::File,
        title: format!("Report {n}.pdf"),
        folder: folder.to_path_buf(),
        markdown_path: folder.join(format!("Report {n}.pdf.md")),
        finished_at: "2026-11-04T02:42:18Z".into(),
        duration: 1.5,
        warnings: Vec::new(),
        usage: None,
        options: Options {
            config: ConfigSource::User,
            ..Options::default()
        },
        markitai_version: "2.3.2".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn five_hundred_and_one_additions_keep_five_hundred_newest_first() {
        let dir = tempfile::tempdir().unwrap();
        let (mut history, reset) = History::load(dir.path(), "x");
        assert!(reset.is_none() && history.is_empty());
        for n in 0..501 {
            history.add(sample(n, dir.path()));
        }
        assert_eq!(history.entries().len(), LIMIT);
        assert_eq!(history.entries()[0].title, "Report 500.pdf");
        assert_eq!(history.entries()[LIMIT - 1].title, "Report 1.pdf");
        history.save().unwrap();
        let (again, _) = History::load(dir.path(), "x");
        assert_eq!(again.entries(), history.entries());
        let text: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.path().join("history.json")).unwrap()).unwrap();
        assert_eq!(text["version"], 1);
        assert_eq!(text["entries"][0]["kind"], "file");
        assert_eq!(text["entries"][0]["usage"], serde_json::Value::Null);
        assert_eq!(text["entries"][0]["options"]["config"], "user");
    }

    #[test]
    fn removing_selects_the_next_entry_the_search_shows() {
        let dir = tempfile::tempdir().unwrap();
        let (mut history, _) = History::load(dir.path(), "x");
        let mut add = |title: &str| {
            let mut entry = sample(0, dir.path());
            entry.title = title.into();
            entry.source = format!("/in/{title}");
            history.add(entry.clone());
            entry.id
        };
        let alpha = add("Alpha Note.md");
        let beta = add("Beta.md");
        let gamma = add("Gamma Note.md");
        let note = |e: &Entry| e.matches("note");
        assert_eq!(history.neighbor(gamma, note), Some(alpha), "Beta is filtered out");
        assert_eq!(history.neighbor(alpha, note), Some(gamma), "the previous shown one");
        assert_eq!(history.neighbor(gamma, |_| true), Some(beta));
        assert_eq!(history.neighbor(beta, note), Some(alpha));
        assert_eq!(history.neighbor(gamma, |e| e.id == gamma), None);
        assert_eq!(history.neighbor(Uuid::new_v4(), |_| true), None);
    }

    #[test]
    fn remove_and_clear_leave_the_files() {
        let dir = tempfile::tempdir().unwrap();
        let folder = dir.path().join("Report");
        std::fs::create_dir(&folder).unwrap();
        let (mut history, _) = History::load(dir.path(), "x");
        let entry = sample(1, &folder);
        std::fs::write(&entry.markdown_path, "# Report\n").unwrap();
        history.add(entry.clone());
        history.add(sample(2, &folder));
        assert!(history.remove(entry.id));
        assert!(!history.remove(entry.id));
        history.clear();
        history.save().unwrap();
        assert!(entry.markdown_path.exists());
        assert!(History::load(dir.path(), "x").0.is_empty());
    }

    #[test]
    fn a_corrupt_file_is_renamed_and_reported_once() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("history.json"), "{\"version\": 2, \"entries\": []}").unwrap();
        let (history, reset) = History::load(dir.path(), "20261104T024218Z");
        assert!(history.is_empty());
        assert_eq!(reset.as_deref(), Some("history-unreadable-20261104T024218Z.json"));
        assert!(dir.path().join("history-unreadable-20261104T024218Z.json").exists());
        let (_, again) = History::load(dir.path(), "20261104T024219Z");
        assert_eq!(again, None);
        std::fs::write(dir.path().join("history.json"), "not json").unwrap();
        assert!(History::load(dir.path(), "y").1.is_some());
    }

    #[test]
    fn search_matches_title_and_source_ignoring_case() {
        let dir = tempfile::tempdir().unwrap();
        let (mut history, _) = History::load(dir.path(), "x");
        history.add(sample(1, dir.path()));
        let mut url = sample(2, dir.path());
        url.source = "https://Example.com/Post".into();
        url.title = "Markdown for language models".into();
        history.add(url);
        assert_eq!(history.matching("REPORT 1").len(), 1);
        assert_eq!(history.matching("example.COM").len(), 1);
        assert_eq!(history.matching("").len(), 2);
        assert!(history.matching("nothing").is_empty());
    }
}
