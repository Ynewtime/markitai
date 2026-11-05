//! What the user adds: local files and web addresses (DESIGN.md 7.1).

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Source {
    File(PathBuf),
    Url(String),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    File,
    Url,
}

impl Source {
    pub fn kind(&self) -> Kind {
        match self {
            Source::File(_) => Kind::File,
            Source::Url(_) => Kind::Url,
        }
    }

    /// The `source` the core and History receive.
    pub fn text(&self) -> String {
        match self {
            Source::File(path) => path.to_string_lossy().into_owned(),
            Source::Url(url) => url.clone(),
        }
    }

    /// A file's name, or the URL without its scheme.
    pub fn initial_title(&self) -> String {
        match self {
            Source::File(path) => path
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_else(|| path.to_string_lossy().into_owned()),
            Source::Url(url) => without_scheme(url).to_owned(),
        }
    }
}

pub fn without_scheme(url: &str) -> &str {
    url.find("://").map_or(url, |i| &url[i + 3..])
}

/// The URL field's text as a web address: trimmed, `https://` added when
/// there is no scheme and the text has a dot, and only http or https with a
/// host accepted.
pub fn parse_web_address(text: &str) -> Option<String> {
    let text = text.trim();
    if text.is_empty() || text.chars().any(char::is_whitespace) {
        return None;
    }
    let url = if text.contains("://") {
        text.to_owned()
    } else if text.contains('.') {
        format!("https://{text}")
    } else {
        return None;
    };
    let lower = url.to_ascii_lowercase();
    if !(lower.starts_with("http://") || lower.starts_with("https://")) {
        return None;
    }
    let rest = without_scheme(&url);
    let authority = &rest[..rest.find(['/', '?', '#']).unwrap_or(rest.len())];
    let host_port = authority.rsplit('@').next().unwrap_or_default();
    let host = if host_port.starts_with('[') {
        &host_port[..host_port.find(']').map_or(0, |i| i + 1)]
    } else {
        host_port.split(':').next().unwrap_or_default()
    };
    (!host.is_empty() && host != "[]").then_some(url)
}

/// Pasted or dropped text that is one `http` or `https` URL, scheme
/// included (DESIGN.md 7.1). `example.com` or a sentence with a dot adds
/// nothing: the URL field is where a scheme is added.
pub fn link_text(text: &str) -> Option<String> {
    let lower = text.trim().to_ascii_lowercase();
    if lower.starts_with("http://") || lower.starts_with("https://") {
        parse_web_address(text)
    } else {
        None
    }
}

/// A local path the user added, before any core call.
#[derive(Debug, PartialEq, Eq)]
pub enum Added {
    File(PathBuf),
    Folder(String),
}

/// Folders are refused, except a `.numbers` package (any case), which is one
/// document. A path that does not exist is a file: the core reports it.
pub fn classify(path: &Path) -> Added {
    let is_numbers = path
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("numbers"));
    if path.is_dir() && !is_numbers {
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.to_string_lossy().into_owned());
        Added::Folder(name)
    } else {
        Added::File(path.to_path_buf())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn web_addresses_follow_the_url_field_rule() {
        let ok = |text: &str| parse_web_address(text);
        assert_eq!(ok("  example.com/a "), Some("https://example.com/a".into()));
        assert_eq!(ok("http://example.com"), Some("http://example.com".into()));
        assert_eq!(ok("HTTPS://Example.com/x?y#z"), Some("HTTPS://Example.com/x?y#z".into()));
        assert_eq!(ok("https://user@host.example:8080/p"), Some("https://user@host.example:8080/p".into()));
        assert_eq!(ok("https://[::1]:8000/"), Some("https://[::1]:8000/".into()));
        assert_eq!(ok("localhost"), None);
        assert_eq!(ok("ftp://example.com/file"), None);
        assert_eq!(ok("https://"), None);
        assert_eq!(ok("https:///path"), None);
        assert_eq!(ok("https://exa mple.com"), None);
        assert_eq!(ok(""), None);
    }

    #[test]
    fn pasted_or_dropped_text_adds_only_one_url_with_its_scheme() {
        assert_eq!(link_text(" https://example.com/a\n"), Some("https://example.com/a".into()));
        assert_eq!(link_text("HTTP://example.com"), Some("HTTP://example.com".into()));
        assert_eq!(link_text("example.com"), None);
        assert_eq!(link_text("See the notes.txt file"), None);
        assert_eq!(link_text("https://a.example\nhttps://b.example"), None);
        assert_eq!(link_text("ftp://example.com"), None);
        assert_eq!(link_text("https://"), None);
    }

    #[test]
    fn folders_are_refused_but_numbers_packages_are_documents() {
        let dir = tempfile::tempdir().unwrap();
        let folder = dir.path().join("Reports");
        let numbers = dir.path().join("Budget.NUMBERS");
        std::fs::create_dir(&folder).unwrap();
        std::fs::create_dir(&numbers).unwrap();
        assert_eq!(classify(&folder), Added::Folder("Reports".into()));
        assert_eq!(classify(&numbers), Added::File(numbers.clone()));
        let missing = dir.path().join("gone.pdf");
        assert_eq!(classify(&missing), Added::File(missing.clone()));
    }

    #[test]
    fn titles_are_the_file_name_or_the_url_without_scheme() {
        let file = Source::File("/home/example/Quarterly Report Q3 2026.pdf".into());
        assert_eq!(file.initial_title(), "Quarterly Report Q3 2026.pdf");
        let url = Source::Url("https://example.com/blog/post".into());
        assert_eq!(url.initial_title(), "example.com/blog/post");
        assert_eq!(url.kind(), Kind::Url);
    }
}
