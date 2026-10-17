//! Stable codes for conversion warnings.
//!
//! Converters report what happened as sentences: that is what a terminal, a
//! report and the workbench show. Scripts reading the [document
//! JSON](crate::document_json) need something they can match on instead, so
//! each known family of sentences has a code that does not change when its
//! wording does, a level and the subjects (pages, images, sheets) it names.
//! A sentence no family claims is `other`. Codes are only ever added.

use regex::Regex;
use std::sync::LazyLock;

/// How much a notice asks of the reader.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Level {
    /// Something was lost, failed or needs an action.
    Warning,
    /// An explanation of how the output was produced; nothing is missing.
    Info,
}

/// A warning read as a family: its code, level and subjects.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Classified {
    pub code: &'static str,
    pub level: Level,
    /// The pages, images or sheets the sentence names, as written in it.
    pub items: Vec<String>,
}

/// Where in the sentence a family's subject is.
#[derive(Clone, Copy)]
enum Subject {
    None,
    /// `PDF page 3:` / `Office page 2:` / `TIFF page 4`: the page number.
    Page,
    /// `Image analysis failed for asset-1.emf:`: the image file name.
    ImageAfterFor,
    /// `PDF embedded image name:`: the image name.
    EmbeddedImage,
    /// `Hidden worksheet "Notes"`: the quoted sheet name.
    Quoted,
}

/// `(code, level, starts with, contains, subject)`; the first match wins, so
/// specific families come before the general ones of the same prefix.
const FAMILIES: &[(&str, Level, &str, &str, Subject)] = &[
    // PDF pages.
    (
        "pdf.page_without_text",
        Level::Warning,
        "PDF page ",
        ": native text was not recovered",
        Subject::Page,
    ),
    (
        "pdf.page_plain_text",
        Level::Info,
        "PDF page ",
        ": read as plain text because ",
        Subject::Page,
    ),
    ("pdf.page", Level::Warning, "PDF page ", "", Subject::Page),
    (
        "pdf.images_after_text",
        Level::Info,
        "PDF images are placed after their page's text",
        "",
        Subject::None,
    ),
    (
        "pdf.images_after_text",
        Level::Info,
        "Embedded PDF images are placed after their page's text",
        "",
        Subject::None,
    ),
    (
        "pdf.embedded_image",
        Level::Warning,
        "PDF embedded image ",
        "",
        Subject::EmbeddedImage,
    ),
    (
        "pdf.layout",
        Level::Warning,
        "PDF layout refinement",
        "",
        Subject::None,
    ),
    (
        "pdf.hidden_text",
        Level::Warning,
        "PDF hidden-text removal",
        "",
        Subject::None,
    ),
    // Local OCR.
    (
        "ocr.no_text",
        Level::Warning,
        "Local OCR found no readable text on TIFF page ",
        "",
        Subject::Page,
    ),
    (
        "ocr.no_text",
        Level::Warning,
        "Local OCR found no readable text",
        "",
        Subject::None,
    ),
    (
        "ocr.capped",
        Level::Warning,
        "Local OCR read only the first ",
        "",
        Subject::Page,
    ),
    (
        "ocr.unread",
        Level::Warning,
        "Local OCR could not read ",
        "",
        Subject::Page,
    ),
    (
        "ocr.unavailable",
        Level::Warning,
        "Local OCR",
        "",
        Subject::None,
    ),
    // Models.
    (
        "llm.no_model_succeeded",
        Level::Warning,
        "No model succeeded for this document",
        "",
        Subject::None,
    ),
    (
        "llm.enhancement_failed",
        Level::Warning,
        "LLM enhancement failed",
        "",
        Subject::None,
    ),
    (
        "llm.answer_retried",
        Level::Warning,
        "LLM ",
        " answer was rejected",
        Subject::None,
    ),
    (
        "llm.answer_retried",
        Level::Warning,
        "A visual batch answer was rejected",
        "",
        Subject::None,
    ),
    (
        "llm.cache_unavailable",
        Level::Warning,
        "Persistent LLM cache is unavailable",
        "",
        Subject::None,
    ),
    (
        "llm.cache_entry_ignored",
        Level::Warning,
        "A malformed ",
        " cache entry was ignored",
        Subject::None,
    ),
    (
        "llm.text_fallback",
        Level::Warning,
        "Rendered-page enhancement failed",
        "",
        Subject::None,
    ),
    (
        "llm.cost_incomplete",
        Level::Warning,
        "Cost is incomplete",
        "",
        Subject::None,
    ),
    (
        "text_only.no_text",
        Level::Warning,
        "No text was recognised",
        "",
        Subject::None,
    ),
    // Images.
    (
        "image.analysis_failed",
        Level::Warning,
        "Image analysis failed for ",
        "",
        Subject::ImageAfterFor,
    ),
    (
        "image.analysis_failed",
        Level::Warning,
        "Image analysis failed",
        "",
        Subject::None,
    ),
    (
        "image.not_localized",
        Level::Warning,
        "Image resource ",
        "",
        Subject::None,
    ),
    // Office documents and workbooks.
    (
        "office.renderer_missing",
        Level::Warning,
        "Page screenshots of Office documents need",
        "",
        Subject::None,
    ),
    (
        "office.renderer_missing",
        Level::Warning,
        "Page screenshots and OCR of Office pages need",
        "",
        Subject::None,
    ),
    (
        "office.renderer_missing",
        Level::Warning,
        "OCR of Office pages needs",
        "",
        Subject::None,
    ),
    (
        "office.page",
        Level::Warning,
        "Office page",
        "",
        Subject::Page,
    ),
    (
        "office.screenshot_format",
        Level::Info,
        "Office screenshots use lossless WebP",
        "",
        Subject::None,
    ),
    (
        "office.count_verified",
        Level::Info,
        "",
        " count was verified against LibreOffice",
        Subject::None,
    ),
    (
        "spreadsheet.hidden_sheet",
        Level::Info,
        "Hidden worksheet ",
        "",
        Subject::Quoted,
    ),
    (
        "spreadsheet.screenshot_scope",
        Level::Info,
        "Workbook screenshots use complete-sheet export",
        "",
        Subject::None,
    ),
    (
        "spreadsheet.font_color",
        Level::Info,
        "Workbook fonts with no declared color",
        "",
        Subject::None,
    ),
    (
        "spreadsheet.rendering",
        Level::Warning,
        "Workbook ",
        "",
        Subject::None,
    ),
    (
        "spreadsheet.numbers_formatting",
        Level::Info,
        "Numbers display formatting is normalized",
        "",
        Subject::None,
    ),
    (
        "spreadsheet.numbers_objects",
        Level::Warning,
        "Numbers non-table canvas objects",
        "",
        Subject::None,
    ),
    (
        "word.picture_records",
        Level::Warning,
        "Word picture metadata",
        "",
        Subject::None,
    ),
    (
        "table.merged_cells",
        Level::Info,
        "Merged table cells are represented",
        "",
        Subject::None,
    ),
    (
        "table.malformed",
        Level::Warning,
        "pipe table at line ",
        "",
        Subject::None,
    ),
    // Mail.
    ("mail.eml", Level::Warning, "EML ", "", Subject::None),
    ("mail.msg", Level::Warning, "MSG ", "", Subject::None),
    // Pages fetched from the web.
    (
        "browser.screenshot_failed",
        Level::Warning,
        "Browser screenshot failed",
        "",
        Subject::None,
    ),
    (
        "cloudflare.not_used",
        Level::Warning,
        "The Cloudflare backend was not used",
        "",
        Subject::None,
    ),
];

static PAGE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\bpages? (\d+)(?:\s*[-\u{2013}]\s*(\d+))?").expect("page pattern compiles")
});
static QUOTED: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#""((?:[^"\\]|\\.)*)""#).expect("quoted pattern compiles"));

/// The family of one warning sentence.
pub fn classify(message: &str) -> Classified {
    let family = FAMILIES.iter().find(|(_, _, prefix, needle, _)| {
        message.starts_with(prefix) && (needle.is_empty() || message.contains(needle))
    });
    let Some(&(code, level, prefix, _, subject)) = family else {
        return Classified {
            code: "other",
            level: Level::Warning,
            items: Vec::new(),
        };
    };
    let rest = &message[prefix.len()..];
    let items = match subject {
        Subject::None => Vec::new(),
        Subject::Page => PAGE
            .captures(message)
            .map(|found| {
                let first: usize = found[1].parse().unwrap_or(0);
                let last = found
                    .get(2)
                    .and_then(|last| last.as_str().parse().ok())
                    .filter(|last| *last >= first && last - first < 10_000)
                    .unwrap_or(first);
                (first..=last).map(|page| page.to_string()).collect()
            })
            .unwrap_or_default(),
        Subject::ImageAfterFor => rest
            .split_once(": ")
            .map(|(name, _)| vec![name.to_owned()])
            .unwrap_or_default(),
        Subject::EmbeddedImage => rest
            .split_once(':')
            .map(|(name, _)| vec![name.to_owned()])
            .unwrap_or_default(),
        Subject::Quoted => QUOTED
            .captures(rest)
            .map(|found| vec![found[1].replace("\\\"", "\"").replace("\\\\", "\\")])
            .unwrap_or_default(),
    };
    Classified { code, level, items }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn code(message: &str) -> (&'static str, Level, Vec<String>) {
        let found = classify(message);
        (found.code, found.level, found.items)
    }

    #[test]
    fn known_families_get_their_code_level_and_subjects() {
        assert_eq!(
            code(
                "PDF page 3: native text was not recovered (no text layer); OCR is required for this page."
            ),
            (
                "pdf.page_without_text",
                Level::Warning,
                vec!["3".to_owned()]
            )
        );
        assert_eq!(
            code("PDF page 2: read as plain text because the layout is ambiguous. Read more."),
            ("pdf.page_plain_text", Level::Info, vec!["2".to_owned()])
        );
        assert_eq!(
            code("Hidden worksheet \"Hidden notes\" is omitted by the native spreadsheet reader."),
            (
                "spreadsheet.hidden_sheet",
                Level::Info,
                vec!["Hidden notes".to_owned()]
            )
        );
        assert_eq!(
            code(
                "Image analysis failed for asset-1.emf: Cannot decode; existing alt text was retained and no analysis entry was produced."
            ),
            (
                "image.analysis_failed",
                Level::Warning,
                vec!["asset-1.emf".to_owned()]
            )
        );
        assert_eq!(
            code(
                "Local OCR found no readable text on TIFF page 4; the output retains its image reference."
            ),
            ("ocr.no_text", Level::Warning, vec!["4".to_owned()])
        );
        assert_eq!(
            code("Office pages 2\u{2013}3 (2 of 3): OCR found no text."),
            (
                "office.page",
                Level::Warning,
                vec!["2".to_owned(), "3".to_owned()]
            )
        );
        assert_eq!(
            code("LLM enhancement failed: LLM output was truncated by its token limit; the converted Markdown is kept as it was.").0,
            "llm.enhancement_failed"
        );
        assert_eq!(
            code(&crate::ocr::capped_warning("PDF page 7")),
            ("ocr.capped", Level::Warning, vec!["7".to_owned()])
        );
        assert_eq!(
            code("No model succeeded for this document; only the base Markdown was kept.").0,
            "llm.no_model_succeeded"
        );
    }

    #[test]
    fn an_unknown_sentence_is_other_and_a_warning() {
        assert_eq!(
            code("Something new happened."),
            ("other", Level::Warning, Vec::new())
        );
    }

    #[test]
    fn codes_are_dotted_lower_case_names() {
        for (code, ..) in FAMILIES {
            assert!(
                code.split('.').count() == 2
                    && code
                        .chars()
                        .all(|c| c.is_ascii_lowercase() || c == '.' || c == '_'),
                "{code}"
            );
        }
    }
}
