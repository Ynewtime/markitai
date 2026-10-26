//! The `rag` and `obsidian` profiles publish a document's pictures from the
//! bytes the conversion holds, under content-addressed names inside the
//! output folder. An asset reference in the document's own text, however it
//! is spelled (percent-encoded `..`, absolute, through a planted link), only
//! has its text rewritten: no file outside the output folder is read, moved
//! or replaced.
use crate::console;

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// An 80×80 PNG, large enough to pass the default picture filter.
const PNG: &str = "iVBORw0KGgoAAAANSUhEUgAAAFAAAABQCAIAAAABc2X6AAAAZ0lEQVR42u3PAQ0AAAgDoGd6/wDGsoeyUYBM+0qEhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFL1l6G020DcUTKQAAAABJRU5ErkJggg==";

/// Every file and link below `root` but outside `skip`, with its bytes or target.
fn tree(root: &Path, skip: &Path) -> BTreeMap<PathBuf, String> {
    let mut found = BTreeMap::new();
    let mut pending = vec![root.to_path_buf()];
    while let Some(directory) = pending.pop() {
        for entry in fs::read_dir(&directory).unwrap() {
            let path = entry.unwrap().path();
            if path.starts_with(skip) {
                continue;
            }
            let kind = fs::symlink_metadata(&path).unwrap().file_type();
            if kind.is_symlink() {
                let target = fs::read_link(&path).unwrap();
                found.insert(path, format!("-> {}", target.display()));
            } else if kind.is_dir() {
                pending.push(path);
            } else {
                found.insert(
                    path.clone(),
                    String::from_utf8_lossy(&fs::read(&path).unwrap()).into(),
                );
            }
        }
    }
    found
}

#[test]
fn asset_references_in_document_text_never_reach_outside_the_output_folder() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    for directory in ["home", "state", "in", "victim", "out/.markitai/assets"] {
        fs::create_dir_all(root.join(directory)).unwrap();
    }
    fs::write(root.join("state/config.json"), br#"{"history":{"record":false},"log":{"dir":null},"cache":{"enabled":false},"llm":{"enabled":false}}"#).unwrap();
    fs::write(root.join("victim/keep.txt"), "SECRET-VICTIM").unwrap();
    fs::write(root.join("keep.txt"), "OLD-DST").unwrap();
    fs::write(root.join("out/.markitai/assets/inside.txt"), "INSIDE").unwrap();
    #[cfg(unix)]
    std::os::unix::fs::symlink(
        "../../../victim/keep.txt",
        root.join("out/.markitai/assets/link.txt"),
    )
    .unwrap();
    let absolute = root.join("victim/keep.txt");
    let absolute = absolute.to_string_lossy().replace('\\', "/");
    fs::write(
        root.join("in/notes.md"),
        format!(
            "# Notes\n\n![real](data:image/png;base64,{PNG})\n\n\
             ![x](.markitai/assets/..%2F..%2F..%2Fvictim%2Fkeep.txt)\n\
             ![y](.markitai/assets/../../../victim/keep.txt)\n\
             ![z](.markitai/assets/%2F{absolute})\n\
             ![w](.markitai/assets/..%5C..%5C..%5Cvictim%5Ckeep.txt)\n\
             ![s](.markitai/assets/link.txt)\n\
             ![i](.markitai/assets/inside.txt)\n\
             [[.markitai/assets/..%2F..%2Fkeep.txt]]\n\
             <img src=\".markitai/assets/..%2F..%2Fkeep.txt\">\n"
        ),
    )
    .unwrap();
    let output = root.join("out");
    let before = tree(root, &output);
    let inside = tree(&output, &root.join("none"));
    for profile in ["rag", "obsidian"] {
        let mut command = Command::new(env!("CARGO_BIN_EXE_markitai"));
        console::english(command.env_clear());
        for name in ["PATH", "SYSTEMROOT", "TMPDIR", "TEMP", "TMP"] {
            if let Some(value) = std::env::var_os(name) {
                command.env(name, value);
            }
        }
        let result = command
            .current_dir(root)
            .env("HOME", root.join("home"))
            .env("USERPROFILE", root.join("home"))
            .env("MARKITAI_HOME", root.join("state"))
            .args(["in/notes.md", "-o", "out", "--profile", profile])
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{profile}: {}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert_eq!(tree(root, &output), before, "{profile}");
    }
    // What was in the output folder before is still there, unchanged.
    let after = tree(&output, &root.join("none"));
    for (path, content) in &inside {
        assert_eq!(after.get(path), Some(content), "{}", path.display());
    }
    // The one real picture was published inside the visible assets folder.
    let published: Vec<_> = after
        .keys()
        .filter(|path| path.starts_with(output.join("assets")))
        .collect();
    assert_eq!(published.len(), 1, "{after:?}");
}
