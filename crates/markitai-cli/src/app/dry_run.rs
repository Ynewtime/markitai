//! What `--dry-run` says each item would do: the file it would write, with the
//! conflict policy applied, or why it would be skipped. The plan only reads the
//! file system; it never creates a directory or a lock.

use super::{Task, config, is_url, publishes_nothing};
use serde_json::Value;
use std::collections::HashSet;
use std::path::{Path, PathBuf};

pub(super) enum Preview {
    /// The document is printed on standard output.
    Stdout,
    /// No document would be written, and why.
    Skip(String),
    /// The document goes to `path`; `note` says what the conflict policy did.
    Write { path: PathBuf, note: Option<String> },
}

impl Preview {
    pub(super) fn converts(&self) -> bool {
        !matches!(self, Self::Skip(_))
    }

    /// The right-hand side of the item's line on stdout (always English: the
    /// listing is data, like the rest of stdout).
    pub(super) fn line(&self) -> String {
        match self {
            Self::Stdout => "stdout".into(),
            Self::Skip(why) => format!("skip ({why})"),
            Self::Write { path, note: None } => path.display().to_string(),
            Self::Write {
                path,
                note: Some(note),
            } => format!("{} ({note})", path.display()),
        }
    }
}

fn source_name(task: &Task) -> String {
    if is_url(&task.source) {
        markitai_core::output::url_name(&task.source, &serde_json::Map::new())
    } else {
        Path::new(&task.source)
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned()
    }
}

/// One preview per task, in order. Names are decided as a real run decides
/// them: an item whose name is taken, on disk or by an earlier item of the
/// same run, gets the next `.vN` under `rename`, is skipped under `skip` and
/// replaces the file under `overwrite`.
pub(super) fn plan(tasks: &[Task], cfg: &Value) -> Vec<Preview> {
    let policy = cfg["output"]["on_conflict"].as_str().unwrap_or("rename");
    // Whether two spellings that differ only in case name one file is a
    // property of the volume; a preview cannot probe it without writing.
    let folds = cfg!(any(target_os = "macos", target_os = "windows"));
    let key = |directory: &Path, name: &str| {
        (
            directory.to_path_buf(),
            if folds {
                name.to_lowercase()
            } else {
                name.to_owned()
            },
        )
    };
    let enhanced = config::enabled(cfg, "/llm/enabled");
    let keep_base = config::enabled(cfg, "/llm/keep_base");
    let explicit = cfg["output"]["filename"].as_str().map(str::to_owned);
    let mut claimed = HashSet::new();
    tasks
        .iter()
        .map(|task| {
            if publishes_nothing(task, cfg) {
                return Preview::Skip("an image needs --ocr or --llm".into());
            }
            let Some(directory) = task.output.as_deref() else {
                return Preview::Stdout;
            };
            let stem = explicit
                .as_deref()
                .or(task.filename.as_deref())
                .map(|name| name.strip_suffix(".md").unwrap_or(name).to_owned())
                .unwrap_or_else(|| source_name(task));
            // The text a model enhanced lands beside the base document, except
            // under an explicit name, which it takes over unless the base stays.
            let suffix = if enhanced && (explicit.is_none() || keep_base) {
                ".llm.md"
            } else {
                ".md"
            };
            let members = |stem: &str| {
                [
                    key(directory, &format!("{stem}.md")),
                    key(directory, &format!("{stem}.llm.md")),
                ]
            };
            let occupied = |stem: &str| {
                [".md", ".llm.md"].iter().any(|suffix| {
                    std::fs::symlink_metadata(directory.join(format!("{stem}{suffix}"))).is_ok()
                })
            };
            let target = |stem: &str| directory.join(format!("{stem}{suffix}"));
            let duplicate = members(&stem).iter().any(|member| claimed.contains(member));
            if !duplicate && policy == "skip" && occupied(&stem) {
                return Preview::Skip(format!(
                    "{} already exists and output.on_conflict is skip",
                    target(&stem).display()
                ));
            }
            let mut resolved = stem.clone();
            let mut note = None;
            if duplicate || (policy == "rename" && occupied(&stem)) {
                let mut version = 2_u64;
                loop {
                    resolved = format!("{stem}.v{version}");
                    if !members(&resolved)
                        .iter()
                        .any(|member| claimed.contains(member))
                        && !occupied(&resolved)
                    {
                        break;
                    }
                    version += 1;
                }
                let existing = format!("{stem}{suffix}");
                note = Some(if duplicate {
                    format!("{existing} is taken by an earlier item")
                } else {
                    format!("{existing} already exists")
                });
            } else if policy == "overwrite" && occupied(&stem) {
                note = Some("replaces the existing file".into());
            }
            claimed.extend(members(&resolved));
            Preview::Write {
                path: target(&resolved),
                note,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn task(source: &str, output: &Path) -> Task {
        Task {
            source: source.into(),
            display: source.into(),
            report_key: source.into(),
            output: Some(output.to_owned()),
            filename: None,
            reserved_stem: None,
            source_file: None,
        }
    }

    fn lines(tasks: &[Task], cfg: &Value) -> Vec<String> {
        plan(tasks, cfg).iter().map(Preview::line).collect()
    }

    #[test]
    fn each_item_names_its_final_file_and_what_the_policy_decided() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("out");
        std::fs::create_dir(&out).unwrap();
        std::fs::write(out.join("taken.docx.md"), "x").unwrap();
        let tasks = [
            task("a.docx", &out),
            task("taken.docx", &out),
            task("taken.docx", &out),
            task("photo.png", &out),
            task("new.docx", &out),
            task("new.docx", &out),
        ];
        let mut cfg = json!({"output": {"on_conflict": "rename"}, "llm": {}, "ocr": {}});
        let shown = |path: &str| out.join(path).display().to_string();
        assert_eq!(
            lines(&tasks, &cfg),
            [
                shown("a.docx.md"),
                format!(
                    "{} (taken.docx.md already exists)",
                    shown("taken.docx.v2.md")
                ),
                format!(
                    "{} (taken.docx.md already exists)",
                    shown("taken.docx.v3.md")
                ),
                "skip (an image needs --ocr or --llm)".into(),
                shown("new.docx.md"),
                format!(
                    "{} (new.docx.md is taken by an earlier item)",
                    shown("new.docx.v2.md")
                ),
            ]
        );
        cfg["output"]["on_conflict"] = json!("skip");
        assert_eq!(
            lines(&tasks[1..2], &cfg),
            [format!(
                "skip ({} already exists and output.on_conflict is skip)",
                shown("taken.docx.md")
            )]
        );
        cfg["output"]["on_conflict"] = json!("overwrite");
        assert_eq!(
            lines(&tasks[1..2], &cfg),
            [format!(
                "{} (replaces the existing file)",
                shown("taken.docx.md")
            )]
        );
        // OCR makes the image convertible; a model puts the text in `.llm.md`.
        cfg["ocr"]["enabled"] = json!(true);
        cfg["llm"]["enabled"] = json!(true);
        assert_eq!(lines(&tasks[3..4], &cfg), [shown("photo.png.llm.md")]);
        // Nothing is written by planning.
        assert_eq!(std::fs::read_dir(&out).unwrap().count(), 1);
    }

    #[test]
    fn an_explicit_name_and_standard_output_are_previewed_too() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("not-created");
        let cfg = json!({"output": {"filename": "chosen.md"}, "llm": {"enabled": true}});
        assert_eq!(
            lines(&[task("a.pdf", &out)], &cfg),
            [out.join("chosen.md").display().to_string()]
        );
        let mut stdout = task("a.pdf", &out);
        stdout.output = None;
        assert_eq!(lines(&[stdout], &json!({})), ["stdout"]);
        assert!(!out.exists());
    }
}
