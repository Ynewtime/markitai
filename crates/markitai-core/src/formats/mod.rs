//! Local, deterministic format adapters. No adapter performs network requests.

mod html;
mod markup;
mod msg;
mod native;
mod numbers;
mod text;

pub(crate) use html::canonical_status_url;
pub(crate) use html::count_words as word_count;
pub(crate) use html::decode_fetched as decode_fetched_html;
pub use html::extract_html;
pub(crate) use native::extract_presentation_count;
#[cfg(test)]
pub(crate) use native::pdf::extract_pages as extract_pdf_pages;
pub(crate) use native::pdf::{
    PdfPages, extract_pages_bounded as extract_pdf_pages_bounded,
    screenshot_reference as pdf_screenshot_reference,
};

use crate::{Document, Error, Result};
use std::path::Path;

pub(crate) fn extract_pdf(bytes: &[u8]) -> Result<Document> {
    native::extract(bytes, "pdf")
}

/// Extensions with an implemented local reader (leading dots are accepted).
/// Document extensions (without the dot) that `extract` reads, including
/// those anydoc recognizes; images are listed in `images::IMAGE_EXTENSIONS`.
pub const DOCUMENT_EXTENSIONS: &[&str] = &[
    "txt", "md", "markdown", "html", "htm", "xhtml", "csv", "tsv", "ipynb", "json", "xml", "eml",
    "msg", "rst", "org", "tex", "latex", "numbers", "doc", "docx", "docm", "odt", "pdf", "pptx",
    "pptm", "ppsx", "ppsm", "ppt", "pps", "pot", "rtf", "epub", "xlsx", "xlsm", "xlsb", "xls",
    "ods", "odp",
];

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
    if !supports_extension(&extension) {
        return Err(Error::Unsupported(unsupported_format_message(&extension)));
    }
    let mut result = if is_numbers_package_path(path) {
        numbers::extract_directory(path)?
    } else {
        let bytes = if extension == "numbers" {
            use std::io::Read;
            const LIMIT: u64 = 128 * 1024 * 1024;
            let file = std::fs::File::open(path)?;
            if file.metadata()?.len() > LIMIT {
                return Err(Error::Conversion(
                    "Numbers package exceeds the 128 MiB limit".into(),
                ));
            }
            let mut bytes = Vec::new();
            file.take(LIMIT + 1).read_to_end(&mut bytes)?;
            if bytes.len() as u64 > LIMIT {
                return Err(Error::Conversion(
                    "Numbers package exceeds the 128 MiB limit".into(),
                ));
            }
            bytes
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
            "eml" => text::email(&bytes)?,
            "msg" => msg::extract(&bytes)?,
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
