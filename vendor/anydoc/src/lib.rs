//! anydoc converts documents to GitHub-Flavored Markdown.
//!
//! Recovery and skipped-content events are reported through the [`log`]
//! facade (debug/warn level); logging never changes conversion behavior and
//! its messages are not a stable API.

#![warn(missing_docs)]

pub mod model;

mod error;
mod formats;
mod package;
mod render;
mod shared;
mod sort; // markitai: shared sort instantiations

pub use error::ConvertError;

use render::markdown::document_to_markdown;

use std::path::Path;

/// Input format. Selects the parser; container variants that share a parser
/// (docm, xlsm, ...) map onto these via [`Format::from_bytes`] or
/// [`Format::from_extension`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Format {
    /// Binary Word 97-2003 (`.doc`, template `.dot`).
    Doc,
    /// WordprocessingML (`.docx`, `.docm`, templates `.dotx`, `.dotm`), both
    /// Transitional and Strict.
    Docx,
    /// OpenDocument Text (`.odt`, template `.ott`).
    Odt,
    /// Converted with [pdf-inspector], which emits Markdown directly:
    /// [`to_document`] is unsupported for PDFs. Scanned or image-only pages
    /// need OCR, which anydoc does not do: the document errors with
    /// [`ConvertError::NeedsOcr`] naming them.
    ///
    /// [pdf-inspector]: https://github.com/firecrawl/pdf-inspector
    Pdf,
    /// Binary PowerPoint 97-2003 (`.ppt`, `.pps`, `.pot`).
    Ppt,
    /// PresentationML (`.pptx`, `.pptm`, `.ppsx`, `.ppsm`, templates `.potx`,
    /// `.potm`).
    Pptx,
    /// Rich Text Format (`.rtf`).
    Rtf,
    /// EPUB 2 and 3 (`.epub`).
    Epub,
    /// Excel workbooks: `.xlsx`, `.xlsm`, binary `.xlsb`, and legacy
    /// OLE-based `.xls`, with their templates `.xltx`, `.xltm` and `.xlt`.
    Excel,
    /// OpenDocument Spreadsheet (`.ods`, template `.ots`).
    Ods,
    /// OpenDocument Presentation (`.odp`, template `.otp`).
    Odp,
    /// Delimiter-separated text (`.csv`). Carries no signature, so it has to
    /// be named rather than detected.
    Csv,
}

impl Format {
    /// Detect the format from the content itself: the signature and identity
    /// each container specification designates (PDF header, RTF open group,
    /// OLE stream names, ZIP package mimetype/content types). Plain-text
    /// formats (CSV) carry no signature and return `None`; so does anything
    /// unrecognized.
    pub fn from_bytes(bytes: &[u8]) -> Option<Format> {
        formats::detect::from_bytes(bytes)
    }

    /// The format a bare extension names (no leading dot), matched
    /// case-insensitively. `None` for anything unrecognized.
    pub fn from_extension(ext: &str) -> Option<Format> {
        // markitai: templates are the containers of the documents they make
        // (`.dotx` a WordprocessingML package, `.ott` an OpenDocument text).
        Some(match ext.to_ascii_lowercase().as_str() {
            "doc" | "dot" => Format::Doc,
            "docx" | "docm" | "dotx" | "dotm" => Format::Docx,
            "odt" | "ott" => Format::Odt,
            "pdf" => Format::Pdf,
            "pptx" | "pptm" | "ppsx" | "ppsm" | "potx" | "potm" => Format::Pptx,
            "ppt" | "pps" | "pot" => Format::Ppt,
            "rtf" => Format::Rtf,
            "epub" => Format::Epub,
            "xlsx" | "xlsm" | "xlsb" | "xls" | "xltx" | "xltm" | "xlt" => Format::Excel,
            "ods" | "ots" => Format::Ods,
            "odp" | "otp" => Format::Odp,
            "csv" => Format::Csv,
            _ => return None,
        })
    }

    /// The format a path's extension names. `None` when the path has no
    /// extension or names nothing recognized.
    pub fn from_path(path: &Path) -> Option<Format> {
        path.extension().and_then(|e| e.to_str()).and_then(Format::from_extension)
    }
}

/// Convert a document file to Markdown. The format is detected from the
/// file content ([`Format::from_bytes`]); the extension is the fallback for
/// signature-less formats (CSV) and unrecognizable containers.
pub fn to_markdown(path: impl AsRef<Path>) -> Result<String, ConvertError> {
    let path = path.as_ref();
    let bytes = std::fs::read(path)?;
    let Some(format) = Format::from_bytes(&bytes).or_else(|| Format::from_path(path)) else {
        return Err(ConvertError::Unsupported(format!(
            "unrecognized file content and extension: {}",
            path.display()
        )));
    };
    to_markdown_bytes(&bytes, format)
}

/// Convert an in-memory document to Markdown. Pass a [`Format`] to select the
/// parser, or `None` to detect it from the content ([`Format::from_bytes`]),
/// which signature-less formats (CSV) have to name explicitly.
pub fn to_markdown_bytes(
    bytes: &[u8],
    format: impl Into<Option<Format>>,
) -> Result<String, ConvertError> {
    let format = resolve_format(bytes, format.into())?;
    // PDFs convert to Markdown directly (pdf-inspector) without passing
    // through the document model.
    if format == Format::Pdf {
        return formats::pdf::to_markdown(bytes);
    }
    Ok(document_to_markdown(&to_document(bytes, format)?))
}

/// A number as a spreadsheet cell with the number format `code` displays it:
/// dates and times as ISO-like dates and clock times, percentages, grouping,
/// currency and the other implemented SpreadsheetML codes; anything else as
/// General. `date1904` selects the 1904 date system.
///
/// markitai: exposed so a chart's cached values read as the chart shows
/// them (a date category is otherwise a bare serial number).
pub fn format_number(code: &str, value: f64, date1904: bool) -> String {
    formats::format_number(code, value, date1904)
}

/// The data an embedded OLE object's own file holds, as blocks: an Excel
/// worksheet or chart, an MS Graph chart, an OpenDocument chart or
/// spreadsheet (from a compound file), or a zipped OOXML workbook or
/// OpenDocument object. Empty for any other object (an equation, a
/// document, a picture), or when the object's content is unreadable; only a
/// resource limit is an error.
///
/// markitai: exposed so an OOXML presentation's embedded objects (its
/// `ppt/embeddings` parts) read as a legacy deck's do.
pub fn embedded_object(bytes: &[u8]) -> Result<Vec<model::Block>, ConvertError> {
    formats::embedded_object(bytes)
}

/// The text points of a SmartArt diagram's data part (`dgm:dataModel`), as a
/// bullet list in the order the part lists them; the DOCX reader reads a
/// diagram the same way. Empty when the part holds no text or is not
/// readable XML; only a resource limit is an error.
///
/// markitai: exposed so an OOXML presentation's diagrams (`ppt/diagrams`)
/// read as a Word document's do.
pub fn diagram_data(bytes: &[u8]) -> Result<Vec<model::Block>, ConvertError> {
    match package::xml::parse_xml(bytes) {
        Ok(root) => Ok(shared::drawingml::diagram_blocks(&root)),
        Err(e) if e.is_fatal() => Err(e),
        Err(e) => {
            log::warn!("skipping corrupt diagram part: {e}");
            Ok(Vec::new())
        }
    }
}

/// Parse an in-memory document into the document model. Pass a [`Format`] to
/// select the parser, or `None` to detect it from the content.
///
/// Unsupported for [`Format::Pdf`]: PDF conversion produces Markdown
/// directly and has no document-model form; use [`to_markdown_bytes`].
pub fn to_document(
    bytes: &[u8],
    format: impl Into<Option<Format>>,
) -> Result<model::Document, ConvertError> {
    formats::parse(bytes, resolve_format(bytes, format.into())?)
}

fn resolve_format(bytes: &[u8], format: Option<Format>) -> Result<Format, ConvertError> {
    format.or_else(|| Format::from_bytes(bytes)).ok_or_else(|| {
        ConvertError::Unsupported("unrecognized file content: name the format explicitly".into())
    })
}
