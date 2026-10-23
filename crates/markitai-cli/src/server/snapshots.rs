//! What a URL item keeps of its page in the job's `snapshots/` folder, for
//! the workbench's Compare view (contract v1.14): the page it was converted
//! from (`<item>.html`), how usable that page and its screenshots were when
//! they were captured (`<item>.capture.json`), where the screenshots show the
//! page's blocks (`<item>.blocks.json`) and, when the conversion saved no
//! screenshots of its own, compare screenshots (`<item>.shots/<n>.jpg`).
//! A PDF item that local OCR read keeps where its recognized lines and cut
//! out pictures are (`<item>.ocr.json`), for the Compare view of its pages.
//! None of them is a result file: they are not listed in a result, not
//! archived and never sent to a model; they go with the item.
use super::store;
use markitai_core::page_capture::{CaptureQuality, MAX_BLOCKS_BYTES, PageCapture};
use serde_json::{Value, json};
use std::{
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
};

/// Compare screenshots an item keeps at most.
pub(super) const MAX_SHOTS: usize = 20;
/// The largest `capture.json` read.
const MAX_CAPTURE: u64 = 64 * 1024;

/// The names of an item's kept files inside `snapshots/`; none for an item
/// ID (from an older record) that is not a plain file name.
pub(super) struct Names {
    pub html: String,
    pub capture: String,
    pub blocks: String,
    pub shots: String,
    pub ocr: String,
}

pub(super) fn names(item_id: &str) -> Option<Names> {
    let plain = !item_id.is_empty()
        && item_id.len() <= 64
        && item_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'));
    plain.then(|| Names {
        html: format!("{item_id}.html"),
        capture: format!("{item_id}.capture.json"),
        blocks: format!("{item_id}.blocks.json"),
        shots: format!("{item_id}.shots"),
        ocr: format!("{item_id}.ocr.json"),
    })
}

/// Write `bytes` privately and atomically; a failure leaves nothing.
fn write(path: &Path, bytes: &[u8]) -> bool {
    let written = path.parent().ok_or(()).and_then(|parent| {
        let mut file = tempfile::NamedTempFile::new_in(parent).map_err(|_| ())?;
        file.write_all(bytes).map_err(|_| ())?;
        file.persist(path).map(drop).map_err(|_| ())
    });
    if written.is_err() {
        let _ = fs::remove_file(path);
    }
    written.is_ok()
}

/// Keep what a conversion's page load left in `folder` (a job's or a staged
/// retry's `snapshots/`, already created privately). Each file is written on
/// its own; one that cannot be written is only missing. Block boxes are kept
/// only beside screenshots: the conversion's own (`screenshots`) or the
/// compare screenshots kept here.
pub(super) fn keep(folder: &Path, item_id: &str, capture: &PageCapture, screenshots: bool) {
    let Some(names) = names(item_id) else {
        return;
    };
    if let Some(html) = &capture.html {
        write(&folder.join(&names.html), html.as_bytes());
    }
    let mut shots = 0;
    if !capture.compare_shots.is_empty() && !screenshots {
        let directory = folder.join(&names.shots);
        if markitai_core::output::create_private_dir(&directory).is_ok() {
            for (index, bytes) in capture.compare_shots.iter().take(MAX_SHOTS).enumerate() {
                if !write(&directory.join(format!("{}.jpg", index + 1)), bytes) {
                    break;
                }
                shots = index + 1;
            }
            if shots < capture.compare_shots.len().min(MAX_SHOTS) {
                // Tiles cut the page from its top: a gap would misplace the rest.
                let _ = fs::remove_dir_all(&directory);
                shots = 0;
            }
        }
    }
    let shown = screenshots || shots > 0;
    if let Some(blocks) = capture
        .blocks
        .as_ref()
        .filter(|blocks| shown && blocks.len() <= MAX_BLOCKS_BYTES)
    {
        write(&folder.join(&names.blocks), blocks);
    }
    if capture.page.is_some() || capture.screenshot.is_some() {
        let record = json!({
            "version": 1,
            "page": capture.page,
            "screenshot": capture.screenshot.as_ref().filter(|_| shown),
        });
        write(
            &folder.join(&names.capture),
            &serde_json::to_vec(&record).unwrap_or_default(),
        );
    }
}

/// The item's compare screenshots in order (`1.jpg`, `2.jpg`, … without a
/// gap), with their names inside the shots folder.
pub(super) fn shots(job: &Path, item_id: &str) -> Vec<(PathBuf, String)> {
    let Some(names) = names(item_id) else {
        return Vec::new();
    };
    let directory = job.join("snapshots").join(&names.shots);
    (1..=MAX_SHOTS)
        .map_while(|number| {
            let name = format!("{number}.jpg");
            store::safe_file(&directory, &name)
                .ok()
                .map(|path| (path, name))
        })
        .collect()
}

/// An item's kept file named by `pick`, if it is one.
pub(super) fn kept(job: &Path, item_id: &str, pick: fn(&Names) -> &String) -> Option<PathBuf> {
    let names = names(item_id)?;
    store::safe_file(&job.join("snapshots"), pick(&names)).ok()
}

/// Every file an item keeps in `job`'s `snapshots/`, as job-relative names
/// (`snapshots/…`), for deleting or replacing them with the item.
pub(super) fn files(job: &Path, item_id: &str) -> Vec<String> {
    let Some(names) = names(item_id) else {
        return Vec::new();
    };
    let snapshots = job.join("snapshots");
    let mut found: Vec<String> = [&names.html, &names.capture, &names.blocks, &names.ocr]
        .into_iter()
        .filter(|name| store::safe_file(&snapshots, name).is_ok())
        .map(|name| format!("snapshots/{name}"))
        .collect();
    let directory = snapshots.join(&names.shots);
    if markitai_core::output::check_path(&directory, false).is_ok()
        && fs::symlink_metadata(&directory).is_ok_and(|meta| meta.is_dir())
        && let Ok(listed) = store::files(&directory)
    {
        found.extend(
            listed
                .into_iter()
                .map(|(name, _)| format!("snapshots/{}/{name}", names.shots)),
        );
    }
    found.sort();
    found
}

/// The largest `ocr.json` kept and read.
const MAX_OCR_LAYOUT: u64 = 32 * 1024 * 1024;

/// Keep where an OCR'd PDF's recognized lines and pictures are
/// ([`markitai_core::ConversionOutput::ocr_layout`]) in `job`'s
/// `snapshots/`, replacing what an earlier attempt kept; a result without
/// one removes it. A layout that cannot be written is only missing.
pub(super) fn keep_ocr(job: &Path, item_id: &str, layout: Option<&Value>) {
    let Some(names) = names(item_id) else {
        return;
    };
    let folder = job.join("snapshots");
    let path = folder.join(&names.ocr);
    let bytes = layout
        .and_then(|layout| serde_json::to_vec(layout).ok())
        .filter(|bytes| bytes.len() as u64 <= MAX_OCR_LAYOUT);
    match bytes {
        Some(bytes) if markitai_core::output::create_private_dir(&folder).is_ok() => {
            write(&path, &bytes);
        }
        _ => {
            if store::safe_file(&folder, &names.ocr).is_ok() {
                let _ = fs::remove_file(&path);
            }
        }
    }
}

/// A version of an item's kept OCR layout (its size and modification time),
/// for caching what is read from it; `None` without one.
pub(super) fn ocr_version(job: &Path, item_id: &str) -> Option<String> {
    let metadata = fs::metadata(kept(job, item_id, |names| &names.ocr)?).ok()?;
    let modified = metadata
        .modified()
        .ok()
        .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
        .map_or(0, |time| time.as_nanos());
    Some(format!("{}-{modified}", metadata.len()))
}

/// The kept OCR layout of an item's page `number`: `{"page", "width",
/// "height", "lines", "images"}`; `None` when the page was not OCR'd.
pub(super) fn ocr_page(job: &Path, item_id: &str, number: usize) -> Option<Value> {
    let file = fs::File::open(kept(job, item_id, |names| &names.ocr)?).ok()?;
    let mut bytes = Vec::new();
    file.take(MAX_OCR_LAYOUT + 1).read_to_end(&mut bytes).ok()?;
    if bytes.len() as u64 > MAX_OCR_LAYOUT {
        return None;
    }
    let mut layout: Value = serde_json::from_slice(&bytes).ok()?;
    if layout["version"] != 1 {
        return None;
    }
    let pages = layout.get_mut("pages")?.as_array_mut()?;
    let index = pages
        .iter()
        .position(|page| page["page"].as_u64() == Some(number as u64))?;
    Some(pages.swap_remove(index))
}

/// Remove an item's shots folder once its files are gone (an empty folder
/// only; anything left in it stays).
pub(super) fn prune(job: &Path, item_id: &str) {
    if let Some(names) = names(item_id) {
        let _ = fs::remove_dir(job.join("snapshots").join(names.shots));
    }
}

/// The recorded qualities of an item's views: `capture.json`'s `page` and
/// `screenshot`, or `None` when there is no usable record.
pub(super) fn recorded(path: &Path) -> Option<(Option<CaptureQuality>, Option<CaptureQuality>)> {
    let mut bytes = Vec::new();
    fs::File::open(path)
        .ok()?
        .take(MAX_CAPTURE + 1)
        .read_to_end(&mut bytes)
        .ok()?;
    if bytes.len() as u64 > MAX_CAPTURE {
        return None;
    }
    let value: Value = serde_json::from_slice(&bytes).ok()?;
    if value["version"] != 1 {
        return None;
    }
    let quality = |key: &str| serde_json::from_value::<Option<CaptureQuality>>(value[key].clone());
    Some((quality("page").ok()?, quality("screenshot").ok()?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use markitai_core::page_capture::CaptureState;

    fn quality(state: CaptureState) -> CaptureQuality {
        CaptureQuality {
            state,
            text_length: 120,
            title: Some("Invented".into()),
            reason: None,
        }
    }

    #[test]
    fn an_ocr_layout_is_kept_replaced_listed_and_read_by_page() {
        let job = tempfile::tempdir().unwrap();
        let layout = json!({"version":1,"pages":[
            {"page":2,"width":595.0,"height":842.0,"lines":[{"text":"Read","bbox":[1.0,2.0,3.0,4.0]}],"images":[]},
            {"page":3,"width":595.0,"height":842.0,"lines":[],"images":[{"bbox":[5.0,6.0,7.0,8.0]}]}]});
        keep_ocr(job.path(), "i1", Some(&layout));
        assert_eq!(files(job.path(), "i1"), ["snapshots/i1.ocr.json"]);
        assert!(ocr_version(job.path(), "i1").is_some());
        assert_eq!(ocr_page(job.path(), "i1", 3).unwrap(), layout["pages"][1]);
        assert!(ocr_page(job.path(), "i1", 1).is_none());
        // A retry that read no page by OCR leaves none behind.
        keep_ocr(job.path(), "i1", None);
        assert!(files(job.path(), "i1").is_empty() && ocr_version(job.path(), "i1").is_none());
        // Another version is not read.
        keep_ocr(
            job.path(),
            "i1",
            Some(&json!({"version":2,"pages":layout["pages"]})),
        );
        assert!(ocr_page(job.path(), "i1", 2).is_none());
    }

    #[test]
    fn only_plain_item_ids_name_kept_files() {
        assert_eq!(names("i12").unwrap().shots, "i12.shots");
        for unsafe_id in ["", "..", "../meta", "a/b", "a\\b", ".hidden", "i1.html"] {
            assert!(names(unsafe_id).is_none(), "{unsafe_id}");
        }
    }

    #[test]
    fn a_capture_is_kept_privately_listed_and_read_back() {
        let job = tempfile::tempdir().unwrap();
        let folder = job.path().join("snapshots");
        markitai_core::output::create_private_dir(&folder).unwrap();
        let capture = PageCapture {
            html: Some("<p>page</p>".into()),
            page: Some(quality(CaptureState::Ok)),
            screenshot: Some(quality(CaptureState::Challenge)),
            compare_shots: vec![b"one".to_vec(), b"two".to_vec()],
            blocks: Some(br#"{"version":1,"width":1,"height":1,"blocks":[]}"#.to_vec()),
        };
        keep(&folder, "i1", &capture, false);
        assert_eq!(
            files(job.path(), "i1"),
            [
                "snapshots/i1.blocks.json",
                "snapshots/i1.capture.json",
                "snapshots/i1.html",
                "snapshots/i1.shots/1.jpg",
                "snapshots/i1.shots/2.jpg",
            ]
        );
        let kept_shots = shots(job.path(), "i1");
        assert_eq!(
            kept_shots
                .iter()
                .map(|(_, name)| name.as_str())
                .collect::<Vec<_>>(),
            ["1.jpg", "2.jpg"]
        );
        assert_eq!(fs::read(&kept_shots[1].0).unwrap(), b"two");
        #[cfg(unix)]
        for name in files(job.path(), "i1") {
            use std::os::unix::fs::PermissionsExt;
            let mode = fs::metadata(job.path().join(&name))
                .unwrap()
                .permissions()
                .mode();
            assert_eq!(mode & 0o077, 0, "{name}");
        }
        let (page, screenshot) =
            recorded(&kept(job.path(), "i1", |names| &names.capture).unwrap()).unwrap();
        assert_eq!(page, Some(quality(CaptureState::Ok)));
        assert_eq!(screenshot, Some(quality(CaptureState::Challenge)));
        // With screenshots of its own, no compare copy is kept; without any
        // screenshot neither blocks nor a screenshot quality are.
        keep(&folder, "i2", &capture, true);
        assert!(shots(job.path(), "i2").is_empty());
        assert!(kept(job.path(), "i2", |names| &names.blocks).is_some());
        let bare = PageCapture {
            compare_shots: Vec::new(),
            ..capture
        };
        keep(&folder, "i3", &bare, false);
        assert!(kept(job.path(), "i3", |names| &names.blocks).is_none());
        let (_, screenshot) =
            recorded(&kept(job.path(), "i3", |names| &names.capture).unwrap()).unwrap();
        assert!(screenshot.is_none());
        // An unreadable record is no record.
        fs::write(folder.join("i4.capture.json"), r#"{"version":2}"#).unwrap();
        assert!(recorded(&folder.join("i4.capture.json")).is_none());
        // Pruning leaves a folder that still holds files.
        prune(job.path(), "i1");
        assert_eq!(shots(job.path(), "i1").len(), 2);
    }
}
