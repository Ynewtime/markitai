//! Local, deterministic format adapters. No adapter performs network requests.

mod html;
mod mail;
mod markup;
mod msg;
mod native;
mod numbers;
mod sniff;
mod text;

pub(crate) use html::count_words as word_count;
pub(crate) use html::decode_fetched as decode_fetched_html;
pub use html::extract_html;
pub(crate) use html::parse::{document as parse_html, fragment as parse_html_fragment};
pub(crate) use html::{canonical_status_url, status_address, video_data_scripts};
pub(crate) use native::extract_presentation_count;
#[cfg(test)]
pub(crate) use native::pdf::extract_pages as extract_pdf_pages;
pub(crate) use native::pdf::{
    PdfPages, UNREAD_PAGES,
    extract_pages_bounded_with_config as extract_pdf_pages_bounded_with_config,
    is_unrecovered_warning, screenshot_reference as pdf_screenshot_reference,
};
pub(crate) use sniff::{
    check_image, check_not_empty, explain_damage, not_a_package, real_extension, retyped_warning,
};

use crate::{Document, Error, Result};
use std::collections::HashSet;
use std::path::Path;

/// Page numbers as a person writes them: ascending, each run of consecutive
/// pages as a range with an en dash (`1–12, 15, 20–31`).
pub(crate) fn page_ranges(pages: &[usize]) -> String {
    let mut sorted = pages.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    let mut runs: Vec<(usize, usize)> = Vec::new();
    for page in sorted {
        match runs.last_mut() {
            Some((_, end)) if *end + 1 == page => *end = page,
            _ => runs.push((page, page)),
        }
    }
    runs.iter()
        .map(|&(start, end)| {
            if start == end {
                start.to_string()
            } else {
                format!("{start}–{end}")
            }
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// An address written as an inline Markdown link or image destination,
/// `[text](here)`: one that every CommonMark reader takes whole. Tabs and line
/// breaks are dropped, as a browser drops them from an address; spaces,
/// parentheses, angle brackets, double quotes and other control characters
/// are percent-encoded; a backslash is escaped, so it is not read as escaping
/// the character after it.
pub(crate) fn link_destination(value: &str) -> String {
    destination(value, true)
}

/// An address written as a Markdown autolink, `<here>`: as
/// [`link_destination`], except that a backslash stays as it is, since an
/// autolink reads no escapes.
pub(crate) fn autolink_destination(value: &str) -> String {
    destination(value, false)
}

fn destination(value: &str, escapes: bool) -> String {
    let mut out = String::with_capacity(value.len());
    for ch in value.chars() {
        match ch {
            '\t' | '\r' | '\n' => {}
            '\\' if escapes => out.push_str("\\\\"),
            ' ' | '(' | ')' | '<' | '>' | '"' => out.push_str(&format!("%{:02X}", ch as u32)),
            _ if ch.is_control() => {
                for byte in ch.encode_utf8(&mut [0; 4]).bytes() {
                    out.push_str(&format!("%{byte:02X}"));
                }
            }
            _ => out.push(ch),
        }
    }
    out
}

pub(crate) fn extract_pdf_with_config(bytes: &[u8], cfg: &serde_json::Value) -> Result<Document> {
    native::pdf::extract_with_config(bytes, cfg)
}

/// Extensions with an implemented local reader (leading dots are accepted).
/// Document extensions (without the dot) that `extract` reads, including
/// those anydoc recognizes; images are listed in `images::IMAGE_EXTENSIONS`.
/// Templates (`.dotx`, `.xltx`, `.potx`, `.ott`, `.ots`, `.otp` and the
/// macro-enabled and legacy ones) are the containers of the documents they
/// make and read as those.
pub const DOCUMENT_EXTENSIONS: &[&str] = &[
    "txt", "md", "markdown", "html", "htm", "xhtml", "csv", "tsv", "ipynb", "json", "xml", "eml",
    "msg", "rst", "org", "tex", "latex", "numbers", "doc", "dot", "docx", "docm", "dotx", "dotm",
    "odt", "ott", "pdf", "pptx", "pptm", "ppsx", "ppsm", "potx", "potm", "ppt", "pps", "pot",
    "rtf", "epub", "xlsx", "xlsm", "xltx", "xltm", "xlsb", "xls", "xlt", "ods", "ots", "odp",
    "otp",
];

/// The document extension a template's reader goes by: a template is read,
/// and its Markdown written, as the document it makes.
pub(crate) fn document_extension(extension: &str) -> &str {
    match extension {
        "dot" => "doc",
        "dotx" => "docx",
        "dotm" => "docm",
        "ott" => "odt",
        "potx" => "pptx",
        "potm" => "pptm",
        "xltx" => "xlsx",
        "xltm" => "xlsm",
        "xlt" => "xls",
        "ots" => "ods",
        "otp" => "odp",
        other => other,
    }
}

pub fn supports_extension(extension: &str) -> bool {
    let extension = extension.trim_start_matches('.').to_ascii_lowercase();
    DOCUMENT_EXTENSIONS.contains(&extension.as_str())
}

/// One actionable line for a file whose extension cannot be converted: the
/// extension and every supported one, documents and images, as the reference
/// words it.
pub fn unsupported_format_message(extension: &str) -> String {
    let extension = extension.trim_start_matches('.').to_ascii_lowercase();
    let shown = if extension.is_empty() {
        "(no extension)".to_owned()
    } else {
        format!("'.{extension}'")
    };
    let mut supported: Vec<&str> = DOCUMENT_EXTENSIONS
        .iter()
        .chain(crate::images::IMAGE_EXTENSIONS)
        .copied()
        .collect();
    supported.sort_unstable();
    supported.dedup();
    let supported = supported
        .iter()
        .map(|extension| format!(".{extension}"))
        .collect::<Vec<_>>()
        .join(" ");
    format!("Unsupported file format: {shown}. Supported extensions: {supported}.")
}

#[cfg(test)]
mod extension_tests {
    #[test]
    fn page_ranges_are_ascending_runs_with_an_en_dash() {
        use super::page_ranges;
        assert_eq!(page_ranges(&[1, 2, 3, 5, 7, 8, 9]), "1–3, 5, 7–9");
        assert_eq!(page_ranges(&[20, 4, 3, 5, 4, 1]), "1, 3–5, 20");
        assert_eq!(page_ranges(&[4]), "4");
        assert_eq!(page_ranges(&[7, 8]), "7–8");
        assert_eq!(page_ranges(&[]), "");
    }

    use super::*;

    #[test]
    fn document_extensions_cover_anydoc_and_the_message_lists_all_supported() {
        // Every extension anydoc recognizes is a document extension here.
        for candidate in DOCUMENT_EXTENSIONS
            .iter()
            .chain(["dotx", "ots", "key", "pages", "odg", "wps", "docb", "xlt"].iter())
        {
            if anydoc::Format::from_extension(candidate).is_some() {
                assert!(supports_extension(candidate), "{candidate}");
            }
        }
        for native in ["txt", "md", "html", "eml", "msg", "numbers", "tex", "ipynb"] {
            assert!(
                supports_extension(native)
                    && supports_extension(&format!(".{}", native.to_uppercase()))
            );
        }
        assert!(
            !supports_extension("png") && !supports_extension("xyz") && !supports_extension("")
        );
        let message = unsupported_format_message("XYZ");
        assert!(
            message.starts_with(
                "Unsupported file format: '.xyz'. Supported extensions: .avif .bmp .csv"
            ),
            "{message}"
        );
        for extension in DOCUMENT_EXTENSIONS
            .iter()
            .chain(crate::images::IMAGE_EXTENSIONS)
        {
            assert!(
                message.contains(&format!(" .{extension} "))
                    || message.ends_with(&format!(" .{extension}.")),
                "{extension}"
            );
        }
        assert!(
            unsupported_format_message("").starts_with("Unsupported file format: (no extension).")
        );
    }
}

/// Whether a local directory is an atomic Numbers document candidate.
///
/// Invalid or unsupported packages still belong to this class: callers must
/// report one document error rather than traverse their implementation files.
pub fn is_numbers_package_path(path: &Path) -> bool {
    path.extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("numbers"))
        && path.is_dir()
}

/// Extract a local document without modifying the input or writing assets.
pub fn extract(path: &Path) -> Result<Document> {
    let extension = path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    extract_as(path, &extension)
}

/// `extract`, reading the file as the format `extension` names (lowercase,
/// without the dot) rather than the one its own name gives.
pub(crate) fn extract_as(path: &Path, extension: &str) -> Result<Document> {
    extract_as_inner(path, extension, None)
}

/// The full conversion keeps mail download provenance outside the public
/// Document/Asset schema; extracting text alone does not prepare its assets.
pub(crate) fn extract_as_with_attachments(
    path: &Path,
    extension: &str,
) -> Result<(Document, HashSet<String>)> {
    let mut originals = HashSet::new();
    let document = extract_as_inner(path, extension, Some(&mut originals))?;
    Ok((document, originals))
}

fn extract_as_inner(
    path: &Path,
    extension: &str,
    mut originals: Option<&mut HashSet<String>>,
) -> Result<Document> {
    let extension = extension.to_owned();
    if !supports_extension(&extension) {
        return Err(Error::Unsupported(unsupported_format_message(&extension)));
    }
    let mut result = if is_numbers_package_path(path) {
        numbers::extract_directory(path)?
    } else {
        let bytes = if extension == "numbers" {
            const LIMIT: u64 = 128 * 1024 * 1024;
            let file = std::fs::File::open(path)?;
            if file.metadata()?.len() > LIMIT {
                return Err(Error::Conversion(
                    "Numbers package exceeds the 128 MiB limit".into(),
                ));
            }
            crate::platform::read_limited(file, LIMIT)?.ok_or_else(|| {
                Error::Conversion("Numbers package exceeds the 128 MiB limit".into())
            })?
        } else {
            std::fs::read(path)?
        };
        match extension.as_str() {
            "txt" | "md" | "markdown" => text::plain(&bytes)?,
            "html" | "htm" | "xhtml" => html::extract_html_bytes(&bytes)?,
            "csv" | "tsv" => {
                text::delimited_bytes(&bytes, if extension == "tsv" { b'\t' } else { b',' })?
            }
            "ipynb" => text::notebook(&text::decode(&bytes)?)?,
            "json" => text::json(&text::decode(&bytes)?)?,
            "xml" => text::xml(&text::decode(&bytes)?)?,
            "eml" => match originals.as_mut() {
                Some(originals) => {
                    let (document, names) = text::email_with_attachments(&bytes)?;
                    **originals = names;
                    document
                }
                None => text::email(&bytes)?,
            },
            "msg" => match originals.as_mut() {
                Some(originals) => {
                    let (document, names) = msg::extract_with_attachments(&bytes)?;
                    **originals = names;
                    document
                }
                None => msg::extract(&bytes)?,
            },
            "numbers" => numbers::extract(&bytes)?,
            "rst" | "org" | "tex" | "latex" => markup::extract(&text::decode(&bytes)?, &extension)?,
            _ => native::extract(&bytes, &extension)?,
        }
    };
    result
        .metadata
        .insert("source".into(), path.to_string_lossy().as_ref().into());
    result
        .metadata
        .insert("format".into(), extension.to_uppercase().into());
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_destination_is_read_whole_by_a_markdown_reader() {
        let address = "https://example.org/a b(c)<d>\"e\\_f\u{7}\r\n\tg";
        let written = link_destination(address);
        assert_eq!(
            written,
            "https://example.org/a%20b%28c%29%3Cd%3E%22e\\\\_f%07g"
        );
        let markdown = format!("[x]({written})");
        let destinations: Vec<String> = pulldown_cmark::Parser::new(&markdown)
            .filter_map(|event| match event {
                pulldown_cmark::Event::Start(pulldown_cmark::Tag::Link { dest_url, .. }) => {
                    Some(dest_url.to_string())
                }
                _ => None,
            })
            .collect();
        assert_eq!(
            destinations,
            ["https://example.org/a%20b%28c%29%3Cd%3E%22e\\_f%07g"]
        );
        // An autolink reads no escapes, so its backslash stays single.
        let autolink = format!("<{}>", autolink_destination("https://example.org/a\\_b c"));
        assert_eq!(autolink, "<https://example.org/a\\_b%20c>");
        let read: Vec<String> = pulldown_cmark::Parser::new(&autolink)
            .filter_map(|event| match event {
                pulldown_cmark::Event::Start(pulldown_cmark::Tag::Link { dest_url, .. }) => {
                    Some(dest_url.to_string())
                }
                _ => None,
            })
            .collect();
        assert_eq!(read, ["https://example.org/a\\_b%20c"]);
    }

    #[test]
    fn format_support_does_not_promise_unimplemented_readers() {
        for extension in [
            ".DOCX", "pdf", "pptx", "xls", "ods", "eml", "markdown", "tsv", "org", "rst", "tex",
            "latex", "msg", "numbers",
        ] {
            assert!(supports_extension(extension), "{extension}");
        }
        for extension in ["", "exe", "png", "heic"] {
            assert!(!supports_extension(extension), "{extension}");
        }
        // Templates are read by their documents' readers.
        for (template, document) in [
            ("dot", "doc"),
            ("dotx", "docx"),
            ("dotm", "docm"),
            ("ott", "odt"),
            ("potx", "pptx"),
            ("potm", "pptm"),
            ("xltx", "xlsx"),
            ("xltm", "xlsm"),
            ("xlt", "xls"),
            ("ots", "ods"),
            ("otp", "odp"),
        ] {
            assert!(supports_extension(template), "{template}");
            assert_eq!(document_extension(template), document);
            assert_eq!(
                anydoc::Format::from_extension(template),
                anydoc::Format::from_extension(document)
            );
        }
        assert_eq!(document_extension("docx"), "docx");
    }

    #[test]
    fn text_preserves_frontmatter_and_non_ascii() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("hello.MD");
        std::fs::write(&path, "---\ntitle: 原题\n---\n\n# 原题\n\n正文\n").unwrap();
        let result = extract(&path).unwrap();
        assert_eq!(result.markdown, std::fs::read_to_string(&path).unwrap());
        assert!(!result.metadata.contains_key("title"));
    }

    #[test]
    fn unsupported_and_corrupt_files_are_errors() {
        let dir = tempfile::tempdir().unwrap();
        let unknown = dir.path().join("file.unknown");
        assert!(matches!(extract(&unknown), Err(Error::Unsupported(_))));
        let broken = dir.path().join("file.docx");
        std::fs::write(&broken, "not a document").unwrap();
        assert!(matches!(extract(&broken), Err(Error::Conversion(_))));
    }
}
