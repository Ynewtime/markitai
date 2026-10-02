//! What a local file is, checked before any format parser reads it.
//!
//! Parsers report what they could not parse, in their own terms (an EOCD
//! record, a cross reference table). This module answers the questions a user
//! has first: is the file empty, is it really the type its name says, and is it
//! damaged. It reads at most a few kilobytes, apart from a bounded decode of an
//! image that would otherwise be skipped.

use crate::{Error, Result};
use std::io::Read;
use std::path::Path;

/// Extensions whose empty file stays an (empty) document, as it always was.
const EMPTY_IS_A_DOCUMENT: &[&str] = &[
    "txt", "md", "markdown", "csv", "tsv", "json", "ipynb", "xml", "eml", "rst", "org", "tex",
    "latex",
];

/// Office, OpenDocument and e-book extensions that a PDF is often saved under.
const PACKAGED_DOCUMENTS: &[&str] = &[
    "doc", "docx", "docm", "xls", "xlsx", "xlsm", "xlsb", "ppt", "pps", "pot", "pptx", "pptm",
    "ppsx", "ppsm", "odt", "ods", "odp", "epub",
];

const PDF: &[u8] = b"%PDF-";
const ZIP: &[u8] = b"PK\x03\x04";
const OLE: &[u8] = &[0xd0, 0xcf, 0x11, 0xe0, 0xa1, 0xb1, 0x1a, 0xe1];

/// A file with no content at all cannot be a document of a binary format.
pub(crate) fn check_not_empty(path: &Path, extension: &str) -> Result<()> {
    if EMPTY_IS_A_DOCUMENT.contains(&extension) {
        return Ok(());
    }
    match std::fs::metadata(path) {
        Ok(meta) if meta.is_file() && meta.len() == 0 => {
            Err(Error::InvalidInput("File is empty (0 bytes)".into()))
        }
        _ => Ok(()),
    }
}

fn head(path: &Path, limit: u64) -> Option<Vec<u8>> {
    let mut bytes = Vec::new();
    std::fs::File::open(path)
        .ok()?
        .take(limit)
        .read_to_end(&mut bytes)
        .ok()?;
    Some(bytes)
}

/// The extension the content says the file has, when that is another document
/// type than its name claims. `None` means the name can be believed, or the
/// content is not recognized well enough to overrule it.
pub(crate) fn real_extension(path: &Path, extension: &str) -> Option<&'static str> {
    let packaged = PACKAGED_DOCUMENTS.contains(&extension);
    if !packaged && extension != "pdf" {
        return None;
    }
    let head = head(path, 1024)?;
    // A PDF may be preceded by a little junk, but never by another container.
    if extension != "pdf"
        && !head.starts_with(ZIP)
        && !head.starts_with(OLE)
        && head.windows(PDF.len()).any(|window| window == PDF)
    {
        return Some("pdf");
    }
    if head.starts_with(ZIP) && (extension == "pdf" || is_legacy(extension)) {
        return zip_kind(path);
    }
    if head.starts_with(OLE) {
        // An old binary document saved under the name of its modern successor.
        return match extension {
            "docx" | "docm" => Some("doc"),
            "xlsx" | "xlsm" | "xlsb" => Some("xls"),
            "pptx" | "pptm" | "ppsx" | "ppsm" => Some("ppt"),
            _ => None,
        };
    }
    None
}

fn is_legacy(extension: &str) -> bool {
    matches!(extension, "doc" | "xls" | "ppt" | "pps" | "pot")
}

/// The document type of a ZIP package, from the parts that define each type.
fn zip_kind(path: &Path) -> Option<&'static str> {
    let mut archive = zip::ZipArchive::new(std::fs::File::open(path).ok()?).ok()?;
    let has =
        |archive: &mut zip::ZipArchive<std::fs::File>, name: &str| archive.by_name(name).is_ok();
    if has(&mut archive, "word/document.xml") {
        return Some("docx");
    }
    if has(&mut archive, "xl/workbook.xml") {
        return Some("xlsx");
    }
    if has(&mut archive, "ppt/presentation.xml") {
        return Some("pptx");
    }
    let mut mimetype = String::new();
    archive
        .by_name("mimetype")
        .ok()?
        .take(128)
        .read_to_string(&mut mimetype)
        .ok()?;
    match mimetype.trim() {
        "application/vnd.oasis.opendocument.text" => Some("odt"),
        "application/vnd.oasis.opendocument.spreadsheet" => Some("ods"),
        "application/vnd.oasis.opendocument.presentation" => Some("odp"),
        "application/epub+zip" => Some("epub"),
        _ => None,
    }
}

/// A one-line name for what a recognized extension holds.
pub(crate) fn kind_name(extension: &str) -> &'static str {
    match extension {
        "pdf" => "PDF document",
        "docx" | "docm" | "doc" => "Word document",
        "xlsx" | "xlsm" | "xlsb" | "xls" => "Excel workbook",
        "pptx" | "pptm" | "ppsx" | "ppsm" | "ppt" => "PowerPoint presentation",
        "odt" => "OpenDocument text",
        "ods" => "OpenDocument spreadsheet",
        "odp" => "OpenDocument presentation",
        "epub" => "EPUB e-book",
        _ => "document",
    }
}

/// The warning for a file converted by what it is rather than what it is called.
pub(crate) fn retyped_warning(extension: &str, real: &str) -> String {
    format!(
        "The content is a {} although the file name ends in .{extension}; it was converted as .{real}. Rename the file to say so.",
        kind_name(real)
    )
}

/// A parse failure that points at the file rather than at the program, with
/// the parser's own wording kept after it.
pub(crate) fn explain_damage(error: Error) -> Error {
    let Error::Conversion(message) = &error else {
        return error;
    };
    const SIGNS: &[&str] = &[
        "malformed document",
        "zip archive",
        "Zip archive",
        "cross reference",
        "couldn't parse input",
        "not a readable",
    ];
    if SIGNS.iter().any(|sign| message.contains(sign)) {
        Error::Conversion(format!(
            "The file appears to be damaged or truncated ({message})"
        ))
    } else {
        error
    }
}

/// The prefix read to learn an image's format and size. A JPEG's frame header
/// can follow long metadata; a prefix that ends first is trusted.
const HEADER_LIMIT: u64 = 1024 * 1024;

/// An image that would be skipped for want of OCR or a model is still checked
/// for being an image at all, so a damaged one is an error and not a "skip".
/// The check reads the header and, where the format has one, its declared
/// length or trailer; it never decodes pixels, so skipping a folder of photos
/// stays instant.
pub(crate) fn check_image(path: &Path, extension: &str) -> Result<()> {
    let invalid =
        |reason: String| Error::InvalidInput(format!("File is not a valid image: {reason}"));
    let damaged = |reason: &str| {
        invalid(format!(
            "the file appears to be damaged or truncated ({reason})"
        ))
    };
    let meta = std::fs::metadata(path)?;
    if meta.len() == 0 {
        return Err(Error::InvalidInput("File is empty (0 bytes)".into()));
    }
    let start = head(path, 1024).unwrap_or_default();
    match extension {
        "svg" => {
            // A long prolog or DOCTYPE may come before the root element.
            let text =
                String::from_utf8_lossy(&head(path, 64 * 1024).unwrap_or_default()).into_owned();
            if !text.contains("<svg") {
                return Err(invalid("it does not look like SVG markup".into()));
            }
            return Ok(());
        }
        "heic" | "heif" | "avif" => {
            if start.get(4..8) != Some(b"ftyp") {
                return Err(invalid("it is not a HEIF or AVIF container".into()));
            }
            return Ok(());
        }
        "tif" | "tiff" => {
            if !(start.starts_with(b"II*\0") || start.starts_with(b"MM\0*")) {
                return Err(invalid("it is not a TIFF file".into()));
            }
            return Ok(());
        }
        _ => {}
    }
    let bytes = head(path, HEADER_LIMIT).ok_or_else(|| invalid("it cannot be read".into()))?;
    let reader = image::ImageReader::new(crate::images::ImageBytes::new(&bytes))
        .with_guessed_format()
        .map_err(|error| invalid(error.to_string()))?;
    let Some(format) = reader.format() else {
        return Err(invalid(
            "its content is not a recognized image format".into(),
        ));
    };
    let whole = bytes.len() as u64 == meta.len();
    if let Err(error) = reader.into_dimensions()
        && whole
    {
        return Err(damaged(error.to_string().trim()));
    }
    let tail = tail(path, 16).unwrap_or_default();
    let declared = |offset: usize| {
        start
            .get(offset..offset + 4)
            .map(|field| u64::from(u32::from_le_bytes(field.try_into().unwrap())))
    };
    let intact = match format {
        image::ImageFormat::Png => {
            tail.len() >= 8 && &tail[tail.len() - 8..tail.len() - 4] == b"IEND"
        }
        image::ImageFormat::Gif => tail.last() == Some(&0x3B),
        image::ImageFormat::WebP => declared(4).is_none_or(|size| size + 8 <= meta.len()),
        image::ImageFormat::Bmp => declared(2).is_none_or(|size| size <= meta.len()),
        // A JPEG may carry data after its end marker (motion photos), so only
        // its header is read.
        _ => true,
    };
    if intact {
        Ok(())
    } else {
        Err(damaged("its end is missing"))
    }
}

/// The last `count` bytes of a file.
fn tail(path: &Path, count: u64) -> std::io::Result<Vec<u8>> {
    use std::io::{Seek, SeekFrom};
    let mut file = std::fs::File::open(path)?;
    let length = file.metadata()?.len();
    file.seek(SeekFrom::Start(length.saturating_sub(count)))?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn file(dir: &tempfile::TempDir, name: &str, bytes: &[u8]) -> std::path::PathBuf {
        let path = dir.path().join(name);
        std::fs::write(&path, bytes).unwrap();
        path
    }

    fn zip_with(entries: &[(&str, &str)]) -> Vec<u8> {
        let mut writer = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
        for (name, body) in entries {
            writer
                .start_file(*name, zip::write::SimpleFileOptions::default())
                .unwrap();
            writer.write_all(body.as_bytes()).unwrap();
        }
        writer.finish().unwrap().into_inner()
    }

    #[test]
    fn an_empty_file_is_named_empty_unless_empty_text_is_a_document() {
        let dir = tempfile::tempdir().unwrap();
        for extension in ["docx", "pdf", "xlsx", "html", "png"] {
            let path = file(&dir, &format!("empty.{extension}"), b"");
            let error = check_not_empty(&path, extension).unwrap_err();
            assert_eq!(error.to_string(), "File is empty (0 bytes)", "{extension}");
        }
        for extension in ["txt", "md", "csv", "tsv"] {
            let path = file(&dir, &format!("empty.{extension}"), b"");
            assert!(check_not_empty(&path, extension).is_ok(), "{extension}");
        }
        let path = file(&dir, "one.docx", b"x");
        assert!(check_not_empty(&path, "docx").is_ok());
    }

    #[test]
    fn content_overrules_a_wrong_extension_only_between_document_types() {
        let dir = tempfile::tempdir().unwrap();
        let pdf = file(&dir, "a.docx", b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n1 0 obj");
        assert_eq!(real_extension(&pdf, "docx"), Some("pdf"));
        let genuine = file(&dir, "a.pdf", b"%PDF-1.7\n");
        assert_eq!(real_extension(&genuine, "pdf"), None);
        let word = file(
            &dir,
            "b.pdf",
            &zip_with(&[("word/document.xml", "<w:document/>")]),
        );
        assert_eq!(real_extension(&word, "pdf"), Some("docx"));
        let sheet = file(
            &dir,
            "c.pdf",
            &zip_with(&[("xl/workbook.xml", "<workbook/>")]),
        );
        assert_eq!(real_extension(&sheet, "pdf"), Some("xlsx"));
        let odt = file(
            &dir,
            "d.pdf",
            &zip_with(&[("mimetype", "application/vnd.oasis.opendocument.text")]),
        );
        assert_eq!(real_extension(&odt, "pdf"), Some("odt"));
        // A modern Word file named .doc is read as the format it is.
        let modern = file(
            &dir,
            "e.doc",
            &zip_with(&[("word/document.xml", "<w:document/>")]),
        );
        assert_eq!(real_extension(&modern, "doc"), Some("docx"));
        // An old binary document named .docx is read as .doc.
        let mut ole = OLE.to_vec();
        ole.extend([0; 64]);
        let old = file(&dir, "f.docx", &ole);
        assert_eq!(real_extension(&old, "docx"), Some("doc"));
        // An unrecognized archive under .pdf leaves the name in charge.
        let other = file(&dir, "g.pdf", &zip_with(&[("data.bin", "x")]));
        assert_eq!(real_extension(&other, "pdf"), None);
        // Text formats are never second-guessed, nor a genuine package.
        let text = file(&dir, "h.txt", b"%PDF-1.7 is the header of a PDF");
        assert_eq!(real_extension(&text, "txt"), None);
        let genuine_docx = file(
            &dir,
            "i.docx",
            &zip_with(&[("word/document.xml", "<w:document/>")]),
        );
        assert_eq!(real_extension(&genuine_docx, "docx"), None);
        assert!(retyped_warning("docx", "pdf").contains("content is a PDF document"));
    }

    #[test]
    fn a_parser_failure_about_the_container_reads_as_a_damaged_file() {
        for message in [
            "Native PDF conversion failed: couldn't parse input",
            "Native PDF conversion failed: failed parsing cross reference table",
            "Native document conversion failed: malformed document: not a readable zip archive: invalid Zip archive: Could not find EOCD",
            "Presentation conversion failed: invalid Zip archive: Could not find EOCD",
            "Native document conversion failed: malformed document: not a readable workbook container",
        ] {
            let explained = explain_damage(Error::Conversion(message.into())).to_string();
            assert_eq!(
                explained,
                format!("The file appears to be damaged or truncated ({message})")
            );
        }
        // Everything else keeps its own words.
        let other = Error::Conversion("Native PDF conversion failed: password required".into());
        assert_eq!(
            explain_damage(other).to_string(),
            "Native PDF conversion failed: password required"
        );
        let kept = Error::InvalidInput("malformed document".into());
        assert_eq!(explain_damage(kept).to_string(), "malformed document");
    }

    #[test]
    fn an_image_that_cannot_be_decoded_is_not_a_valid_image() {
        let dir = tempfile::tempdir().unwrap();
        let mut png = Vec::new();
        image::DynamicImage::new_rgb8(40, 40)
            .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
            .unwrap();
        assert!(check_image(&file(&dir, "ok.png", &png), "png").is_ok());
        let empty = check_image(&file(&dir, "empty.png", b""), "png").unwrap_err();
        assert_eq!(empty.to_string(), "File is empty (0 bytes)");
        let junk = check_image(&file(&dir, "junk.png", b"not an image at all"), "png")
            .unwrap_err()
            .to_string();
        assert!(junk.starts_with("File is not a valid image: "), "{junk}");
        let cut = check_image(&file(&dir, "cut.png", &png[..png.len() - 30]), "png")
            .unwrap_err()
            .to_string();
        assert!(
            cut.starts_with(
                "File is not a valid image: the file appears to be damaged or truncated"
            ),
            "{cut}"
        );
        assert!(check_image(&file(&dir, "x.svg", b"hello"), "svg").is_err());
        assert!(check_image(&file(&dir, "y.svg", b"<svg xmlns=\"\"/>"), "svg").is_ok());
        // A long prolog before the root element is still SVG.
        let prolog = format!("<?xml version=\"1.0\"?><!-- {} --><svg/>", "x".repeat(3000));
        assert!(check_image(&file(&dir, "z.svg", prolog.as_bytes()), "svg").is_ok());
    }
}
