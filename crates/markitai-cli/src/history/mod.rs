//! Optional, self-contained CLI archives. Conversion never depends on this store.
mod assets;
mod image_metadata;

use crate::output_claims::sync_group::SyncGroup;
use crate::report::{ItemKind, ItemStatus, RunItem, RunMode};
use chrono::{DateTime, Local, SecondsFormat};
use markitai_core::{config, platform};
use serde::Serialize;
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

const MAX_ITEMS: usize = 100_000;
const MAX_MARKDOWN: u64 = 64 * 1024 * 1024;
const MAX_METADATA: usize = 16 * 1024 * 1024;

pub(crate) struct Plan {
    jobs_root: PathBuf,
    output_root: PathBuf,
    mode: RunMode,
    options: Options,
    started_at: String,
    allow_symlinks: bool,
    show_path: bool,
}

#[derive(Serialize)]
struct Options {
    preset: Option<String>,
    llm: bool,
    ocr: bool,
    origin: &'static str,
}

impl Plan {
    pub(crate) fn new(
        cfg: &Value,
        output: Option<&Path>,
        mode: RunMode,
        preset: Option<&str>,
        started_at: &str,
        show_path: bool,
    ) -> Option<Self> {
        if !config::enabled(cfg, "/history/record") {
            return None;
        }
        Some(Self {
            jobs_root: config::home().join("serve/jobs"),
            output_root: output?.to_owned(),
            mode,
            options: Options {
                preset: preset.map(str::to_owned),
                llm: config::enabled(cfg, "/llm/enabled"),
                ocr: config::enabled(cfg, "/ocr/enabled"),
                origin: "cli",
            },
            started_at: DateTime::parse_from_rfc3339(started_at)
                .map(|time| time.to_rfc3339_opts(SecondsFormat::Millis, false))
                .unwrap_or_else(|_| now()),
            allow_symlinks: config::enabled(cfg, "/output/allow_symlinks"),
            show_path,
        })
    }

    pub(crate) fn record(&self, records: &[RunItem]) {
        match self.publish(records) {
            Ok(Some(path)) if self.show_path => {
                eprintln!("Recorded in history: {}", path.display());
            }
            Ok(_) => (),
            Err(error) => eprintln!("Warning: Could not record history: {error}"),
        }
    }

    fn publish(&self, records: &[RunItem]) -> io::Result<Option<PathBuf>> {
        let mut records: Vec<_> = records
            .iter()
            .filter(|item| self.mode != RunMode::SingleFile || item.history_eligible)
            .collect();
        if records.is_empty() || crate::signals::interrupted().is_some() {
            return Ok(None);
        }
        if records.len() > MAX_ITEMS {
            return Err(invalid("History exceeds the 100,000 item limit"));
        }
        if self.mode == RunMode::Directory {
            crate::sort::by_key(&mut records, |item| {
                (item.kind == ItemKind::Url, item.index)
            });
        }
        let finished_at = now();
        // History metadata is always private, independent of output symlink policy.
        private_directory(&self.jobs_root)?;
        let boundary = crate::report_store::resolve_path(&self.output_root)?;
        let mut stage_builder = tempfile::Builder::new();
        stage_builder.prefix(".tmp-");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            stage_builder.permissions(fs::Permissions::from_mode(0o700));
        }
        let stage = stage_builder.tempdir_in(&self.jobs_root)?;
        let out = stage.path().join("out");
        private_directory(&out)?;
        let mut budget = assets::Budget::default();
        let mut image_indexes = image_metadata::Indexes::default();
        let mut names = OutputNames::default();
        let mut items = Vec::with_capacity(records.len());
        // Insertion order matters: a later root receives collision suffixes.
        let mut roots = indexmap::IndexMap::<PathBuf, Vec<(PathBuf, usize)>>::new();
        for (index, item) in records.into_iter().enumerate() {
            if crate::signals::interrupted().is_some() {
                return Ok(None);
            }
            let source = item.output.as_ref().or(item.history_output.as_ref());
            let mut copied = None;
            if let Some(source) = source {
                if source.is_file() {
                    let filename = source
                        .file_name()
                        .and_then(|s| s.to_str())
                        .ok_or_else(|| invalid("History output filename is not UTF-8"))?;
                    let name = names.next(filename, &out)?;
                    let target = out.join(&name);
                    match assets::copy_file(source, &target, &mut budget, self.allow_symlinks) {
                        Ok(()) => copied = Some(name),
                        Err(error) => eprintln!(
                            "Warning: History output unavailable ({}): {error}",
                            source.display()
                        ),
                    }
                }
                let mut ascend = if item.kind == ItemKind::File {
                    Path::new(&item.report_key)
                        .parent()
                        .map_or(0, |path| path.components().count())
                } else {
                    0
                };
                if item.screenshots > 0
                    && source
                        .parent()
                        .is_some_and(|parent| parent.ends_with(".markitai/screenshots"))
                {
                    // A visual-only output lives two levels below its asset root.
                    // The root lookup still enforces the configured output boundary.
                    ascend += 2;
                }
                if let Some(root) =
                    assets::find_root(source, ascend, &boundary, self.allow_symlinks)?
                {
                    let documents = roots.entry(root.clone()).or_default();
                    if let Some(name) = &copied
                        && source
                            .extension()
                            .is_some_and(|extension| extension.eq_ignore_ascii_case("md"))
                    {
                        let parent = crate::report_store::resolve_path(
                            source.parent().unwrap_or(Path::new(".")),
                        )?;
                        let depth = parent
                            .strip_prefix(&root)
                            .map_err(|_| invalid("History asset root is outside output"))?
                            .components()
                            .count();
                        documents.push((out.join(name), depth));
                    }
                }
            }
            items.push(Item::from_record(
                index,
                item,
                copied,
                &finished_at,
                self.mode,
            ));
        }
        for (root, documents) in roots {
            if crate::signals::interrupted().is_some() {
                return Ok(None);
            }
            let mapping = assets::merge_root(&root, &out, &mut budget, self.allow_symlinks)?;
            image_indexes.merge(
                &root,
                &mapping,
                budget.take_image_indexes(),
                self.allow_symlinks,
            )?;
            if mapping.is_empty() {
                continue;
            }
            // Share each depth's lookup, then release it before the next depth.
            let mut by_depth = std::collections::BTreeMap::<usize, Vec<PathBuf>>::new();
            for (path, depth) in documents {
                by_depth.entry(depth).or_default().push(path);
            }
            for (depth, paths) in by_depth {
                let lookup = {
                    let mut lookup = HashMap::with_capacity(mapping.len() * 3);
                    let prefix = "../".repeat(depth);
                    for (old, new) in &mapping {
                        lookup.insert(old.clone(), new.clone());
                        lookup.insert(format!("./{old}"), new.clone());
                        if depth > 0 {
                            lookup.insert(format!("{prefix}{old}"), new.clone());
                        }
                    }
                    lookup
                };
                for path in paths {
                    let old_size = fs::metadata(&path)?.len();
                    if old_size > MAX_MARKDOWN {
                        return Err(invalid("History Markdown exceeds the 64 MiB rewrite limit"));
                    }
                    let source = fs::read_to_string(&path)?;
                    let rewritten =
                        markitai_core::output::rewrite_asset_references(&source, &lookup);
                    if source != rewritten {
                        budget.replace_bytes(old_size, rewritten.len() as u64)?;
                        let mut file = OpenOptions::new().write(true).truncate(true).open(&path)?;
                        file.write_all(rewritten.as_bytes())?;
                    }
                }
            }
        }
        // The stable lock only serializes the short publication window. Never
        // unlink its inode: another cooperating process may already hold it.
        let lock = publication_lock(&self.jobs_root)?;
        let waiting = std::time::Instant::now();
        loop {
            if crate::signals::interrupted().is_some() {
                return Ok(None);
            }
            match lock.try_lock() {
                Ok(()) => break,
                Err(std::fs::TryLockError::WouldBlock) if waiting.elapsed().as_secs() < 5 => {
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }
                Err(std::fs::TryLockError::WouldBlock) => {
                    return Err(io::Error::other("History publication is busy"));
                }
                Err(std::fs::TryLockError::Error(error)) => return Err(error),
            }
        }
        let _publication = PublicationLock(lock);
        let (job_id, target) = loop {
            let id = uuid::Uuid::new_v4().simple().to_string()[..12].to_owned();
            let target = self.jobs_root.join(&id);
            match fs::symlink_metadata(&target) {
                Err(error) if error.kind() == io::ErrorKind::NotFound => break (id, target),
                Err(error) => return Err(error),
                Ok(_) => continue,
            }
        };
        image_indexes.publish(&out, &target.join("out"), &mut budget, self.allow_symlinks)?;
        let metadata = Metadata {
            job_id: &job_id,
            created_at: &self.started_at,
            finished_at: &finished_at,
            status: "done",
            options: &self.options,
            dir_size_bytes: budget.bytes(),
            items,
        };
        let mut bytes = BoundedJson(Vec::new());
        serde_json::to_writer_pretty(&mut bytes, &metadata).map_err(io::Error::other)?;
        bytes.write_all(b"\n")?;
        let mut file = private_file(&stage.path().join("meta.json"), true)?;
        file.write_all(&bytes.0)?;
        drop(file);
        // Every file and directory of the stage is ordered before the job's
        // name: a job that survives a crash is complete. The parent's durable
        // fence on the same volume then persists all of it before returning.
        let mut contents = SyncGroup::new();
        stage_tree(stage.path(), &mut contents)?;
        contents.commit_ordered()?;
        if crate::signals::interrupted().is_some() {
            return Ok(None);
        }
        platform::rename(stage.path(), &target)?;
        let mut name = SyncGroup::new();
        // Windows has no directory flush: the job's metadata file, flushed
        // after the rename, commits it instead.
        name.stage_renamed_path(&target.join("meta.json"))?;
        name.stage_directory(&self.jobs_root)?;
        name.commit()?;
        Ok(Some(target))
    }
}

#[derive(Serialize)]
struct Metadata<'a> {
    job_id: &'a str,
    created_at: &'a str,
    finished_at: &'a str,
    status: &'static str,
    options: &'a Options,
    dir_size_bytes: u64,
    items: Vec<Item<'a>>,
}

#[derive(Serialize)]
struct Item<'a> {
    item_id: String,
    name: &'a str,
    kind: &'static str,
    status: &'static str,
    error: Option<&'a str>,
    output: Option<String>,
    output_name: Option<String>,
    duration_ms: Option<u64>,
    finished_at: &'a str,
    cost_usd: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pricing: Option<crate::pricing::Pricing>,
    llm_enhanced: bool,
    operation: &'static str,
    skipped: bool,
    skip_reason: Option<&'a str>,
    retryable: bool,
    warnings: Vec<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    diagnostics: Option<&'a crate::diagnostics::AttemptDiagnostics>,
}

impl<'a> Item<'a> {
    fn from_record(
        index: usize,
        item: &'a RunItem,
        output: Option<String>,
        finished_at: &'a str,
        mode: RunMode,
    ) -> Self {
        let pending = item.status == ItemStatus::Pending;
        let skipped = pending || item.status == ItemStatus::Skipped || item.skip_reason.is_some();
        let mut seen = HashSet::new();
        let duration = if item.status == ItemStatus::Completed && mode != RunMode::UrlList {
            item.conversion_duration_s
        } else {
            Some(item.elapsed_s)
        };
        Self {
            item_id: format!("i{}", index + 1),
            name: if item.kind == ItemKind::File {
                &item.report_key
            } else {
                &item.display
            },
            kind: if item.kind == ItemKind::File {
                "file"
            } else {
                "url"
            },
            status: if item.status == ItemStatus::Failed {
                "error"
            } else {
                "done"
            },
            error: if pending {
                Some(
                    "Provider batch enhancement is pending. This archive contains the base Markdown; use the printed --llm-batch-collect command before requesting enhancement again.",
                )
            } else {
                (item.status == ItemStatus::Failed)
                    .then_some(item.error.as_deref())
                    .flatten()
            },
            output_name: output.as_ref().map(|name| {
                name.strip_suffix(".llm.md")
                    .map_or_else(|| name.clone(), |stem| format!("{stem}.md"))
            }),
            llm_enhanced: !skipped
                && output
                    .as_ref()
                    .is_some_and(|name| name.ends_with(".llm.md")),
            output,
            duration_ms: duration
                .filter(|value| value.is_finite())
                .map(|seconds| (seconds * 1000.0).round_ties_even().max(0.0) as u64),
            finished_at,
            cost_usd: item.usage.cost_usd,
            pricing: crate::pricing::Pricing::from_usage(&item.usage),
            operation: "convert",
            diagnostics: item.diagnostics.as_ref(),
            skipped,
            skip_reason: if pending {
                Some("pending_batch")
            } else {
                item.skip_reason.as_deref()
            },
            retryable: item.kind == ItemKind::Url,
            warnings: item
                .warnings
                .iter()
                .map(String::as_str)
                .filter(|warning| seen.insert(*warning))
                .collect(),
        }
    }
}

#[derive(Default)]
struct OutputNames {
    used: HashSet<String>,
    counters: HashMap<String, u64>,
}

impl OutputNames {
    fn next(&mut self, name: &str, out: &Path) -> io::Result<String> {
        if Path::new(name).components().count() != 1 {
            return Err(invalid("History output name is not a filename"));
        }
        let folded = caseless::default_case_fold_str(name);
        if !self.used.contains(&folded) && !out.join(name).exists() {
            self.used.insert(folded);
            return Ok(name.to_owned());
        }
        let (stem, suffix) = split_suffix(name);
        let number = self.counters.entry(folded).or_insert(2);
        loop {
            let candidate = format!("{stem} ({number}){suffix}");
            *number = number
                .checked_add(1)
                .ok_or_else(|| invalid("History name counter exhausted"))?;
            let folded = caseless::default_case_fold_str(&candidate);
            if !self.used.contains(&folded) && !out.join(&candidate).exists() {
                self.used.insert(folded);
                return Ok(candidate);
            }
        }
    }
}

fn split_suffix(name: &str) -> (&str, &str) {
    if let Some(stem) = name.strip_suffix(".llm.md") {
        return (stem, ".llm.md");
    }
    match name
        .rfind('.')
        .filter(|index| name[..*index].chars().any(|character| character != '.'))
    {
        Some(index) => name.split_at(index),
        None => (name, ""),
    }
}

pub(crate) fn eligible(source: &str) -> bool {
    if source.starts_with("http://") || source.starts_with("https://") {
        return true;
    }
    let path = config::expand_home(Path::new(source));
    let extension = path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or("");
    if markitai_core::formats::is_numbers_package_path(&path) {
        // Package byte budgets belong to the reader; directory metadata length
        // does not describe its contents. Failed package conversions also record
        // their one original input, just like failed ordinary documents.
        return true;
    }
    (markitai_core::formats::supports_extension(extension)
        || markitai_core::is_image_extension(extension))
        && fs::metadata(path)
            .is_ok_and(|metadata| metadata.is_file() && metadata.len() <= 500 * 1024 * 1024)
}

fn now() -> String {
    Local::now().to_rfc3339_opts(SecondsFormat::Millis, false)
}
fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

fn private_directory(path: &Path) -> io::Result<()> {
    markitai_core::output::check_path(path, false).map_err(io::Error::other)?;
    platform::private_directory().recursive(true).create(path)?;
    let metadata = fs::symlink_metadata(path)?;
    if !metadata.is_dir() || metadata.file_type().is_symlink() {
        return Err(invalid("History directory is not a regular directory"));
    }
    Ok(())
}

fn private_file(path: &Path, exclusive: bool) -> io::Result<File> {
    let mut options = OpenOptions::new();
    options.read(true).write(true);
    if exclusive {
        options.create_new(true);
    } else {
        options.create(true).truncate(false);
    }
    platform::private_file(&mut options);
    let file = platform::open_no_follow(&options, path)?;
    let opened = platform::file_status(&file)?;
    if !opened.metadata().is_file() {
        return Err(invalid("History metadata is not a regular file"));
    }
    if opened.links() != 1 {
        return Err(invalid("History metadata has multiple hard links"));
    }
    Ok(file)
}

struct PublicationLock(File);
impl Drop for PublicationLock {
    fn drop(&mut self) {
        let _ = self.0.unlock();
    }
}

fn publication_lock(root: &Path) -> io::Result<File> {
    let path = root.join(".publish.lock");
    if fs::symlink_metadata(&path).is_ok_and(|meta| meta.file_type().is_symlink()) {
        return Err(invalid("History publication lock is a symbolic link"));
    }
    private_file(&path, false)
}

/// Synchronize every regular file and directory of the private stage, each
/// directory after its contents, without following links, for one fence.
/// Returns the number of entries staged.
fn stage_tree(root: &Path, fence: &mut SyncGroup) -> io::Result<usize> {
    let mut staged = 0;
    for entry in walkdir::WalkDir::new(root)
        .contents_first(true)
        .follow_links(false)
    {
        let entry = entry?;
        if entry.file_type().is_dir() {
            fence.stage_directory(entry.path())?;
            staged += 1;
        } else if entry.file_type().is_file() {
            let mut options = OpenOptions::new();
            // Flushing a Windows handle requires write access; nothing is written.
            if cfg!(windows) {
                options.write(true);
            } else {
                options.read(true);
            }
            fence.stage(&platform::open_no_follow(&options, entry.path())?)?;
            staged += 1;
        }
    }
    Ok(staged)
}

struct BoundedJson(Vec<u8>);
impl Write for BoundedJson {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if self.0.len().saturating_add(bytes.len()) > MAX_METADATA {
            return Err(invalid("History metadata exceeds the 16 MiB limit"));
        }
        self.0.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use markitai_core::ConversionUsage;
    use serde_json::json;

    fn record(index: usize, key: &str, output: Option<PathBuf>) -> RunItem {
        RunItem {
            index,
            kind: ItemKind::File,
            display: format!("/source/{key}"),
            report_key: key.into(),
            source_file: None,
            status: ItemStatus::Completed,
            output,
            history_output: None,
            history_eligible: true,
            error: None,
            warnings: vec!["first".into(), "second".into(), "first".into()],
            skip_reason: None,
            started_at: String::new(),
            completed_at: String::new(),
            elapsed_s: 0.123,
            conversion_duration_s: Some(0.0025),
            images: 0,
            screenshots: 0,
            usage: ConversionUsage::default(),
            diagnostics: None,
            llm_cache_hit: false,
            fetch_cache_hit: false,
            fetch_strategy: None,
        }
    }
    fn plan(root: &Path, mode: RunMode) -> Plan {
        Plan {
            jobs_root: root.join("home/serve/jobs"),
            output_root: root.join("output"),
            mode,
            options: Options {
                preset: None,
                llm: false,
                ocr: false,
                origin: "cli",
            },
            started_at: "2026-09-28T12:00:00.000+08:00".into(),
            allow_symlinks: false,
            show_path: false,
        }
    }
    fn put(path: &Path, bytes: &[u8]) {
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, bytes).unwrap();
    }
    fn metadata(job: &Path) -> Value {
        serde_json::from_slice(&fs::read(job.join("meta.json")).unwrap()).unwrap()
    }

    #[test]
    fn archive_survives_source_removal_and_relocates_only_each_roots_references() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let first = root.join("output/a/doc.llm.md");
        let second = root.join("output/b/doc.llm.md");
        put(
            &first,
            b"![first](.markitai/assets/p.png)\n`![literal](.markitai/assets/p.png)`\n",
        );
        put(
            &second,
            b"![second](.markitai/assets/p.png)\n`![literal](.markitai/assets/p.png)`\n",
        );
        put(
            &root.join("output/a/.markitai/assets/p.png"),
            b"first image",
        );
        put(
            &root.join("output/b/.markitai/assets/p.png"),
            b"second image",
        );
        put(
            &root.join("output/b/.markitai/states/private.json"),
            b"never copy",
        );
        let records = [
            record(1, "b/source.txt", Some(second)),
            record(0, "a/source.txt", Some(first)),
        ];
        let job = plan(root, RunMode::Directory)
            .publish(&records)
            .unwrap()
            .unwrap();
        fs::remove_dir_all(root.join("output")).unwrap();
        let meta = metadata(&job);
        assert_eq!(meta.as_object().unwrap().len(), 7);
        assert_eq!(meta["items"][0]["name"], "a/source.txt");
        assert_eq!(meta["items"][1]["output"], "doc (2).llm.md");
        assert_eq!(meta["items"][1]["output_name"], "doc (2).md");
        assert_eq!(meta["items"][0]["llm_enhanced"], true);
        let copied = fs::read_to_string(job.join("out/doc (2).llm.md")).unwrap();
        assert!(copied.contains("![second](.markitai/assets/p-2.png)"));
        assert!(copied.contains("`![literal](.markitai/assets/p.png)`"));
        assert_eq!(
            fs::read(job.join("out/.markitai/assets/p-2.png")).unwrap(),
            b"second image"
        );
        assert!(!job.join("out/.markitai/states").exists());
        let actual_size: u64 = walkdir::WalkDir::new(job.join("out"))
            .into_iter()
            .map(Result::unwrap)
            .filter(|entry| entry.file_type().is_file())
            .map(|entry| entry.metadata().unwrap().len())
            .sum();
        assert_eq!(meta["dir_size_bytes"], actual_size);
        assert_eq!(
            meta["job_id"].as_str().unwrap(),
            job.file_name().unwrap().to_str().unwrap()
        );
        assert_eq!(meta["job_id"].as_str().unwrap().len(), 12);
    }

    /// One ordering fence covers the whole stage before the job's name, and
    /// one durable fence follows the rename; every entry of the job is staged.
    #[test]
    fn whole_stage_is_ordered_before_the_job_name_and_then_made_durable() {
        use crate::output_claims::sync_group::take_commits;
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let first = root.join("output/a/doc.md");
        let second = root.join("output/b/doc.md");
        put(&first, b"![one](.markitai/assets/p.png)\n");
        put(&second, b"![two](.markitai/assets/p.png)\n");
        put(
            &root.join("output/a/.markitai/assets/p.png"),
            b"first image",
        );
        put(
            &root.join("output/b/.markitai/assets/p.png"),
            b"second image",
        );
        take_commits();
        let job = plan(root, RunMode::Directory)
            .publish(&[
                record(0, "a/doc.txt", Some(first)),
                record(1, "b/doc.txt", Some(second)),
            ])
            .unwrap()
            .unwrap();
        assert_eq!(take_commits(), ["ordered", "durable"]);
        assert!(
            fs::read_to_string(job.join("out/doc (2).md"))
                .unwrap()
                .contains("p-2.png")
        );
        let entries: Vec<_> = walkdir::WalkDir::new(&job)
            .into_iter()
            .map(Result::unwrap)
            .filter(|entry| entry.file_type().is_file() || entry.file_type().is_dir())
            .collect();
        let files = entries
            .iter()
            .filter(|entry| entry.file_type().is_file())
            .count();
        // Two documents, two assets and the metadata, plus their directories.
        assert_eq!(files, 5);
        let mut group = SyncGroup::new();
        assert_eq!(stage_tree(&job, &mut group).unwrap(), entries.len());
        group.commit_ordered().unwrap();
        // Nothing staged twice or left for a separate per-file flush.
        assert_eq!(take_commits(), ["ordered"]);
    }

    #[test]
    fn screenshot_only_history_preserves_all_tiles_as_binary_assets() {
        for mode in [RunMode::SingleUrl, RunMode::UrlList] {
            let tmp = tempfile::tempdir().unwrap();
            let root = tmp.path();
            let primary = root.join("output/.markitai/screenshots/page.full.jpg");
            let secondary = root.join("output/.markitai/screenshots/page.full--1.jpg");
            let first_bytes = b"\xff\xd8\x80first tile\xff\xd9";
            let second_bytes = b"\xff\xd8\x90second tile\xff\xd9";
            put(&primary, first_bytes);
            put(&secondary, second_bytes);
            put(
                &root.join("output/.markitai/states/private.json"),
                b"private",
            );
            let mut item = record(0, "https://example.test/page", Some(primary));
            item.kind = ItemKind::Url;
            item.screenshots = 2;
            let job = plan(root, mode).publish(&[item]).unwrap().unwrap();
            fs::remove_dir_all(root.join("output")).unwrap();
            let meta = metadata(&job);
            assert_eq!(meta["items"][0]["output"], "page.full.jpg");
            assert_eq!(meta["items"][0]["output_name"], "page.full.jpg");
            assert_eq!(meta["items"][0]["llm_enhanced"], false);
            assert_eq!(
                fs::read(job.join("out/page.full.jpg")).unwrap(),
                first_bytes
            );
            for (filename, expected) in [
                ("page.full.jpg", first_bytes.as_slice()),
                ("page.full--1.jpg", second_bytes.as_slice()),
            ] {
                assert_eq!(
                    fs::read(job.join("out/.markitai/screenshots").join(filename)).unwrap(),
                    expected
                );
            }
            assert!(!job.join("out/.markitai/states").exists());
            assert_eq!(
                meta["dir_size_bytes"],
                (first_bytes.len() * 2 + second_bytes.len()) as u64
            );
        }
    }

    #[test]
    fn metadata_retains_missing_failed_and_skipped_items_without_false_enhancement() {
        let tmp = tempfile::tempdir().unwrap();
        let p = plan(tmp.path(), RunMode::UrlList);
        let mut missing = record(9, "missing.txt", Some(tmp.path().join("output/missing.md")));
        missing.conversion_duration_s = Some(-2.0);
        let mut failure = record(0, "not-used", None);
        failure.kind = ItemKind::Url;
        failure.display = "https://example.test/failed".into();
        failure.status = ItemStatus::Failed;
        failure.error = Some("failed conversion".into());
        let mut skipped = record(2, "skip.txt", None);
        skipped.status = ItemStatus::Skipped;
        skipped.skip_reason = Some("exists".into());
        let existing = tmp.path().join("output/skip.llm.md");
        put(&existing, b"existing");
        skipped.history_output = Some(existing);
        let job = p.publish(&[missing, failure, skipped]).unwrap().unwrap();
        let meta = metadata(&job);
        assert_eq!(meta["status"], "done");
        assert_eq!(
            meta["options"],
            json!({"preset":null,"llm":false,"ocr":false,"origin":"cli"})
        );
        for item in meta["items"].as_array().unwrap() {
            assert_eq!(item.as_object().unwrap().len(), 16);
            assert_eq!(item["finished_at"], meta["finished_at"]);
            assert_eq!(item["warnings"], json!(["first", "second"]));
        }
        assert_eq!(meta["items"][0]["output"], Value::Null);
        assert_eq!(meta["items"][0]["duration_ms"], 123);
        assert_eq!(meta["items"][1]["name"], "https://example.test/failed");
        assert_eq!(meta["items"][1]["status"], "error");
        assert_eq!(meta["items"][1]["error"], "failed conversion");
        assert_eq!(meta["items"][1]["duration_ms"], 123);
        assert_eq!(meta["items"][1]["retryable"], true);
        assert_eq!(meta["items"][2]["skipped"], true);
        assert_eq!(meta["items"][2]["llm_enhanced"], false);
        assert_eq!(meta["items"][2]["output_name"], "skip.md");
    }

    #[test]
    fn half_millisecond_rounding_matches_python_and_preserves_unknown_duration() {
        let mut item = record(0, "source", None);
        for (seconds, expected) in [
            (-2.0, 0),
            (0.0005, 0),
            (0.0015, 2),
            (0.0025, 2),
            (0.0035, 4),
        ] {
            item.conversion_duration_s = Some(seconds);
            assert_eq!(
                Item::from_record(0, &item, None, "now", RunMode::SingleFile).duration_ms,
                Some(expected)
            );
        }
        item.conversion_duration_s = None;
        assert_eq!(
            Item::from_record(0, &item, None, "now", RunMode::SingleFile).duration_ms,
            None
        );
    }

    #[test]
    fn output_names_reserve_unicode_folds_and_full_enhanced_suffixes() {
        let tmp = tempfile::tempdir().unwrap();
        let mut names = OutputNames::default();
        assert_eq!(
            names.next("Straße.llm.md", tmp.path()).unwrap(),
            "Straße.llm.md"
        );
        assert_eq!(
            names.next("STRASSE.llm.md", tmp.path()).unwrap(),
            "STRASSE (2).llm.md"
        );
        assert_eq!(
            names.next("strasse (2).llm.md", tmp.path()).unwrap(),
            "strasse (2) (2).llm.md"
        );
        assert_eq!(
            names.next("Straße.llm.md", tmp.path()).unwrap(),
            "Straße (3).llm.md"
        );
        put(&tmp.path().join("occupied.md"), b"already present");
        assert_eq!(
            names.next("occupied.md", tmp.path()).unwrap(),
            "occupied (2).md"
        );
    }

    #[test]
    fn nested_document_relocates_parent_asset_paths_after_flattening() {
        let tmp = tempfile::tempdir().unwrap();
        let output = tmp.path().join("output/nested/doc.md");
        put(&output, b"![x](../assets/p.png)\n");
        put(&tmp.path().join("output/assets/p.png"), b"image");
        let job = plan(tmp.path(), RunMode::Directory)
            .publish(&[record(0, "nested/doc.txt", Some(output))])
            .unwrap()
            .unwrap();
        assert_eq!(
            fs::read_to_string(job.join("out/doc.md")).unwrap(),
            "![x](assets/p.png)\n"
        );
    }

    #[test]
    fn empty_or_ineligible_outcomes_leave_no_history_directories() {
        let tmp = tempfile::tempdir().unwrap();
        let plan = plan(tmp.path(), RunMode::SingleFile);
        assert!(plan.publish(&[]).unwrap().is_none());
        let mut item = record(0, "unknown.xyz", None);
        item.history_eligible = false;
        assert!(plan.publish(&[item]).unwrap().is_none());
        assert!(!plan.jobs_root.exists());
    }

    #[test]
    #[cfg(unix)]
    fn asset_failure_and_linked_metadata_never_publish_partial_archives() {
        use std::os::unix::fs::symlink;
        let tmp = tempfile::tempdir().unwrap();
        let output = tmp.path().join("output/doc.md");
        put(&output, b"text");
        let outside = tmp.path().join("outside");
        put(&outside, b"preserve");
        fs::create_dir_all(tmp.path().join("output/assets")).unwrap();
        symlink(&outside, tmp.path().join("output/assets/link")).unwrap();
        let p = plan(tmp.path(), RunMode::SingleFile);
        assert!(
            p.publish(&[record(0, "doc.txt", Some(output.clone()))])
                .is_err()
        );
        assert_eq!(fs::read_dir(&p.jobs_root).unwrap().count(), 0);
        fs::remove_file(tmp.path().join("output/assets/link")).unwrap();
        symlink(&outside, p.jobs_root.join(".publish.lock")).unwrap();
        assert!(p.publish(&[record(0, "doc.txt", Some(output))]).is_err());
        assert_eq!(fs::read(&outside).unwrap(), b"preserve");
        assert_eq!(fs::read_dir(&p.jobs_root).unwrap().count(), 1);
    }

    #[test]
    fn concurrent_archives_publish_distinct_complete_jobs() {
        let tmp = tempfile::tempdir().unwrap();
        let output = tmp.path().join("output/doc.md");
        put(&output, b"shared source");
        let jobs = std::thread::scope(|threads| {
            let handles: Vec<_> = (0..4)
                .map(|_| {
                    let p = plan(tmp.path(), RunMode::SingleFile);
                    let source = output.clone();
                    threads.spawn(move || {
                        p.publish(&[record(0, "doc.txt", Some(source))])
                            .unwrap()
                            .unwrap()
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect::<HashSet<_>>()
        });
        assert_eq!(jobs.len(), 4);
        for job in jobs {
            assert_eq!(fs::read(job.join("out/doc.md")).unwrap(), b"shared source");
            assert_eq!(metadata(&job)["items"][0]["status"], "done");
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                assert_eq!(
                    fs::metadata(&job).unwrap().permissions().mode() & 0o777,
                    0o700
                );
                assert_eq!(
                    fs::metadata(job.join("meta.json"))
                        .unwrap()
                        .permissions()
                        .mode()
                        & 0o777,
                    0o600
                );
            }
        }
    }
    #[test]
    fn image_indexes_merge_relocate_collisions_and_survive_source_removal() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let mut records = Vec::new();
        for (index, folder) in ["a", "b"].iter().enumerate() {
            let source = root.join("output").join(folder);
            let document = source.join("doc.md");
            put(&document, b"![image](.markitai/assets/p.png)\n");
            let picture = source.join(".markitai/assets/p.png");
            let shared = source.join(".markitai/assets/shared.png");
            put(&picture, folder.as_bytes());
            put(&shared, b"identical");
            put(&source.join(".markitai/assets/.images.lock"), b"");
            let images = serde_json::json!({
                "version":"1.0", "created":folder, "header_custom":{"kept":folder},
                "images":[
                    {"path":picture,"alt":format!("Picture {folder}"),"source":format!("source-{folder}"),"custom":{"nested":[1,true]},"expected":folder},
                    {"path":shared,"desc":format!("Shared {folder}"),"expected":"identical"},
                    {"path":root.join("not-in-this-archive.png"),"desc":"unrelated shared index record"},
                    {"path":"https://example.invalid/image.png","desc":"remote, not copied"}
                ]
            });
            put(
                &source.join(".markitai/assets/images.json"),
                images.to_string().as_bytes(),
            );
            records.push(record(
                index,
                &format!("{folder}/source.txt"),
                Some(document),
            ));
        }
        let job = plan(root, RunMode::Directory)
            .publish(&records)
            .unwrap()
            .unwrap();
        fs::remove_dir_all(root.join("output")).unwrap();
        let value: Value = serde_json::from_slice(
            &fs::read(job.join("out/.markitai/assets/images.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(value["created"], "a");
        assert_eq!(value["header_custom"]["kept"], "a");
        let images = value["images"].as_array().unwrap();
        assert_eq!(images.len(), 4);
        let physical_out = platform::canonicalize(&job.join("out")).unwrap();
        for item in images {
            let path = Path::new(item["path"].as_str().unwrap());
            assert!(path.is_absolute());
            assert!(path.starts_with(&physical_out));
            assert_eq!(
                fs::read(path).unwrap(),
                item["expected"].as_str().unwrap().as_bytes()
            );
        }
        let named = |index: usize| PathBuf::from(images[index]["path"].as_str().unwrap());
        assert!(named(0).ends_with(".markitai/assets/p.png"));
        assert!(named(2).ends_with(".markitai/assets/p-2.png"));
        // The index uses the platform's separator throughout.
        #[cfg(windows)]
        assert!(!images[0]["path"].as_str().unwrap().contains('/'));
        assert_eq!(images[1]["path"], images[3]["path"]);
        assert_eq!(images[0]["custom"], serde_json::json!({"nested":[1,true]}));
        assert_eq!(images[2]["source"], "source-b");
        assert!(!job.join("out/.markitai/assets/.images.lock").exists());
        assert!(!job.join("out/.markitai/assets/images-2.json").exists());
        let size: u64 = fs::read_dir(job.join("out/.markitai/assets"))
            .unwrap()
            .map(|entry| entry.unwrap().metadata().unwrap().len())
            .sum::<u64>()
            + fs::metadata(job.join("out/doc.md")).unwrap().len()
            + fs::metadata(job.join("out/doc (2).md")).unwrap().len();
        assert_eq!(metadata(&job)["dir_size_bytes"], size);
    }

    #[test]
    fn corrupt_or_linked_image_index_does_not_publish_partial_history() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let document = root.join("output/doc.md");
        let index = root.join("output/.markitai/assets/images.json");
        put(&document, b"body\n");
        for bad in [b"not JSON".as_slice(), br#"{"images":{}}"#] {
            put(&index, bad);
            assert!(
                plan(root, RunMode::SingleFile)
                    .publish(&[record(0, "doc.txt", Some(document.clone()))])
                    .is_err()
            );
            assert_eq!(fs::read(&index).unwrap(), bad);
            assert!(
                fs::read_dir(root.join("home/serve/jobs"))
                    .unwrap()
                    .next()
                    .is_none()
            );
        }
        #[cfg(unix)]
        {
            fs::remove_file(&index).unwrap();
            std::os::unix::fs::symlink(&document, &index).unwrap();
            assert!(
                plan(root, RunMode::SingleFile)
                    .publish(&[record(0, "doc.txt", Some(document.clone()))])
                    .is_err()
            );
            assert_eq!(fs::read(&document).unwrap(), b"body\n");
        }
    }

    #[test]
    fn rag_image_index_paths_are_relocated_without_uri_decoding() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let document = root.join("output/doc.md");
        let asset = root.join("output/assets/a%20b.png");
        put(&document, b"![literal percent](assets/a%2520b.png)\n");
        put(&asset, b"percent filename");
        put(&root.join("output/assets/.images.lock"), b"");
        put(
            &root.join("output/assets/images.json"),
            serde_json::json!({"created":"original","images":[{"path":asset,"text":"keep text"}]})
                .to_string()
                .as_bytes(),
        );
        let job = plan(root, RunMode::SingleFile)
            .publish(&[record(0, "doc.txt", Some(document))])
            .unwrap()
            .unwrap();
        let value: Value =
            serde_json::from_slice(&fs::read(job.join("out/assets/images.json")).unwrap()).unwrap();
        let path = Path::new(value["images"][0]["path"].as_str().unwrap());
        assert_eq!(path.file_name().unwrap(), "a%20b.png");
        assert_eq!(fs::read(path).unwrap(), b"percent filename");
        assert_eq!(value["images"][0]["text"], "keep text");
        assert!(!job.join("out/assets/.images.lock").exists());
    }

    #[test]
    fn provider_pending_archive_keeps_base_and_explains_the_collect_boundary() {
        let tmp = tempfile::tempdir().unwrap();
        let source = tmp.path().join("output/notes.md");
        put(&source, b"Original base pending cloud enhancement.\n");
        let mut pending = record(0, "notes.txt", Some(source.clone()));
        pending.status = ItemStatus::Pending;
        let job = plan(tmp.path(), RunMode::SingleFile)
            .publish(&[pending])
            .unwrap()
            .unwrap();
        let value = metadata(&job);
        let entry = &value["items"][0];
        assert_eq!(entry["status"], "done");
        assert_eq!(entry["skipped"], true);
        assert_eq!(entry["skip_reason"], "pending_batch");
        assert_eq!(entry["llm_enhanced"], false);
        assert!(
            entry["error"]
                .as_str()
                .unwrap()
                .contains("--llm-batch-collect")
        );
        assert_eq!(
            fs::read(job.join("out").join(entry["output"].as_str().unwrap())).unwrap(),
            fs::read(source).unwrap()
        );
        assert!(entry.get("pricing").is_none());
    }

    #[test]
    fn history_retains_pricing_coverage_and_preserves_legacy_unknown() {
        let mut current = record(0, "notes.txt", None);
        current.usage.cost_usd = 0.5;
        current.usage.by_model = json!({"known":{"requests":1,"cost_usd":0.5,"priced_requests":1,"unpriced_requests":0,"cost_status":"complete","pricing_snapshot":"catalog-v1"},"old":{"requests":1,"cost_usd":0.0}}).as_object().unwrap().clone();
        let entry = serde_json::to_value(Item::from_record(
            0,
            &current,
            Some("base.md".into()),
            "finished",
            RunMode::SingleFile,
        ))
        .unwrap();
        assert_eq!(entry["cost_usd"], 0.5);
        assert_eq!(
            entry["pricing"],
            json!({"priced_requests":1,"unpriced_requests":1,"cost_status":"partial","pricing_snapshots":["catalog-v1"]})
        );
    }
}
