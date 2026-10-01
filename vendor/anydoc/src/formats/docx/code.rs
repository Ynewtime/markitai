//! markitai: code set in a monospaced font (see [`crate::shared::code`]).
//!
//! Word has no notion of code: a writer, or an exporter such as macOS
//! `textutil`, sets a snippet in Courier, Consolas or Menlo. A paragraph set
//! entirely in such a font is read as a line of a code block and a run of it
//! inside prose as inline code - unless monospace is what the document is
//! set in (a typewriter-style manuscript), where it says nothing.

use super::styles::{Styles, run_font};
use crate::error::ConvertError;
use crate::package::xml::{Element, ns};
use crate::shared::code::MonoShare;
pub use crate::shared::code::{is_monospace, without_line_gutter};

/// Whether a monospaced font marks code in this body: it sets at most
/// [`crate::shared::code::BODY_SHARE`] of the text's visible characters (by
/// each run's own font, else its character style's, else its paragraph's).
pub fn sets_code_apart(body: &Element, styles: &Styles) -> Result<bool, ConvertError> {
    let mut share = MonoShare::default();
    for paragraph in body.descendants(ns::W, "p") {
        let style = paragraph
            .find(ns::W, "pPr")
            .and_then(|pr| pr.find(ns::W, "pStyle"))
            .and_then(|e| e.attr(ns::W, "val"))
            .or(styles.default_paragraph);
        let paragraph_font = match style {
            Some(id) => styles.style_font(id)?,
            None => None,
        }
        .or(styles.default_font);
        for run in paragraph.child_elems().filter(|c| c.is(ns::W, "r")) {
            let rpr = run.find(ns::W, "rPr");
            let font = match rpr.and_then(run_font) {
                Some(font) => Some(font),
                None => match rpr
                    .and_then(|rpr| rpr.find(ns::W, "rStyle"))
                    .and_then(|e| e.attr(ns::W, "val"))
                {
                    Some(id) => styles.style_font(id)?,
                    None => None,
                }
                .or(paragraph_font),
            };
            let mono = font.is_some_and(is_monospace);
            for text in run.child_elems().filter(|c| c.is(ns::W, "t")) {
                share.text(&text.text(), mono);
            }
        }
    }
    Ok(share.sets_code_apart())
}
