//! Each conversion's own folder under the output folder (desktop/DESIGN.md,
//! section 3.3).

use std::io::ErrorKind;
use std::path::{Path, PathBuf};

use crate::preview::percent_decode;

/// The label a source's folder is named after: a file's name without its last
/// extension, or a URL's host and last path segment.
pub fn label(source: &str) -> String {
    let lower = source.to_ascii_lowercase();
    let label = if lower.starts_with("http://") || lower.starts_with("https://") {
        url_label(source)
    } else {
        let name = source
            .rsplit(['/', '\\'])
            .find(|part| !part.is_empty())
            .unwrap_or(source);
        without_extension(name).to_owned()
    };
    markitai_core::output_name::sanitize(&label, "untitled")
}

fn url_label(url: &str) -> String {
    let rest = &url[url.find("://").map_or(0, |i| i + 3)..];
    let (authority, path) = rest.split_at(rest.find(['/', '?', '#']).unwrap_or(rest.len()));
    let host = authority.rsplit('@').next().unwrap_or(authority);
    let host = host.to_ascii_lowercase();
    let host = host.strip_prefix("www.").unwrap_or(&host);
    let path = path.split(['?', '#']).next().unwrap_or_default();
    match path.split('/').rev().find(|segment| !segment.is_empty()) {
        Some(segment) => {
            let segment = percent_decode(segment);
            format!("{host} - {}", without_extension(&segment))
        }
        None => host.to_owned(),
    }
}

/// `name` without its last extension; a name that is only an extension
/// (`.bashrc`) stays whole.
fn without_extension(name: &str) -> &str {
    match name.rfind('.') {
        Some(0) | None => name,
        Some(dot) => &name[..dot],
    }
}

/// Create `parent/label`, or `label (2)` up to `label (999)` when taken.
/// `create_dir` is atomic, so two conversions never share a folder.
pub fn claim(parent: &Path, source: &str) -> std::io::Result<(PathBuf, String)> {
    std::fs::create_dir_all(parent)?;
    let label = label(source);
    for n in 1..=999 {
        let name = if n == 1 {
            label.clone()
        } else {
            format!("{label} ({n})")
        };
        let path = parent.join(&name);
        match std::fs::create_dir(&path) {
            Ok(()) => return Ok((path, name)),
            Err(e) if e.kind() == ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e),
        }
    }
    Err(std::io::Error::new(
        ErrorKind::AlreadyExists,
        format!(
            "{} already holds {label} up to {label} (999)",
            parent.display()
        ),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels() {
        assert_eq!(
            label("/Users/example/Downloads/Quarterly Report Q3 2026.pdf"),
            "Quarterly Report Q3 2026"
        );
        assert_eq!(
            label(r"C:\Users\example\Downloads\notes.tar.gz"),
            "notes.tar"
        );
        assert_eq!(label("/home/example/.bashrc"), "bashrc");
        assert_eq!(
            label("https://www.example.com/blog/markdown-for-language-models"),
            "example.com - markdown-for-language-models"
        );
        assert_eq!(
            label("https://example.com/docs/%E6%8A%A5%E5%91%8A.html?x=1#y"),
            "example.com - 报告"
        );
        assert_eq!(label("HTTPS://Example.COM/"), "example.com");
        assert_eq!(label("https://example.com/a/b:c"), "example.com - b_c");
    }

    #[test]
    fn a_taken_name_gets_the_next_number() {
        let dir = tempfile::tempdir().unwrap();
        let parent = dir.path().join("Markitai");
        let (first, name) = claim(&parent, "/x/report.pdf").unwrap();
        assert_eq!(name, "report");
        assert!(first.is_dir());
        let (_, second) = claim(&parent, "/y/report.docx").unwrap();
        assert_eq!(second, "report (2)");
        std::fs::write(parent.join("report (3)"), "a file").unwrap();
        assert_eq!(claim(&parent, "/z/report.md").unwrap().1, "report (4)");
    }
}
