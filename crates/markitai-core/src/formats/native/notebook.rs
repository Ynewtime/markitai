//! Jupyter notebooks (nbformat 4): markdown, code and raw cells, and what the
//! code cells printed or drew.
//!
//! Outputs are bounded: each is cut to a few dozen lines with a note, the
//! whole notebook to a fixed amount of output text and of decoded images, and
//! nothing a notebook holds is executed or fetched.

use super::{fence, source_text};
use crate::{Asset, Document, Error, Result};
use base64::Engine as _;
use serde_json::Value;
use std::collections::BTreeSet;

/// Lines of one stream or result kept, the first `OUTPUT_HEAD` and the rest
/// from the end, with a note between them.
const OUTPUT_LINES: usize = 80;
const OUTPUT_HEAD: usize = 40;
/// A traceback keeps its first lines (the call) and its last (the error).
const TRACEBACK_LINES: usize = 30;
const TRACEBACK_HEAD: usize = 6;
/// Characters of one line kept (a printed array, a base64 blob).
const LINE_CHARS: usize = 500;
/// Output text and decoded image bytes kept per notebook.
const TEXT_BUDGET: usize = 4 * 1024 * 1024;
const IMAGE_BUDGET: usize = 64 * 1024 * 1024;

/// Image types a notebook stores as base64, with the extension of their file.
const IMAGE_TYPES: [(&str, &str); 5] = [
    ("image/png", "png"),
    ("image/jpeg", "jpg"),
    ("image/gif", "gif"),
    ("image/webp", "webp"),
    ("image/bmp", "bmp"),
];

/// A notebook's text as a string, or `None` for anything else: a malformed
/// output is left out, it does not fail the notebook.
fn lenient_text(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::Array(parts) => parts
            .iter()
            .map(|part| part.as_str())
            .collect::<Option<Vec<_>>>()
            .map(|parts| parts.concat()),
        _ => None,
    }
}

/// Text without the terminal control a program printed: color and cursor
/// sequences, and the overwriting a carriage return does (a progress bar
/// shows its last state).
fn terminal_text(text: &str) -> String {
    let mut stripped = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '\u{1b}' {
            stripped.push(c);
            continue;
        }
        match chars.peek() {
            // CSI: parameters and intermediates, then a final byte.
            Some('[') => {
                chars.next();
                for next in chars.by_ref() {
                    if ('\u{40}'..='\u{7e}').contains(&next) {
                        break;
                    }
                }
            }
            // OSC: a string up to BEL or ST.
            Some(']') => {
                chars.next();
                while let Some(next) = chars.next() {
                    if next == '\u{7}' {
                        break;
                    }
                    if next == '\u{1b}' {
                        if chars.peek() == Some(&'\\') {
                            chars.next();
                        }
                        break;
                    }
                }
            }
            // A character-set selection (`ESC ( B`) takes one more byte.
            Some('(' | ')' | '*' | '+') => {
                chars.next();
                chars.next();
            }
            Some(_) => {
                chars.next();
            }
            None => {}
        }
    }
    stripped
        .replace("\r\n", "\n")
        .split('\n')
        .map(|line| {
            line.rsplit('\r')
                .find(|part| !part.is_empty())
                .unwrap_or("")
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// `text` cut to `limit` lines (the first `head` and the last ones) with a
/// note between, each line to `LINE_CHARS` characters.
fn clip(text: &str, limit: usize, head: usize) -> String {
    let lines: Vec<String> = text
        .trim_end()
        .lines()
        .map(|line| {
            let line = line.trim_end();
            match line.char_indices().nth(LINE_CHARS) {
                Some((at, _)) => format!("{}… [line cut]", &line[..at]),
                None => line.to_owned(),
            }
        })
        .collect();
    if lines.len() <= limit {
        return lines.join("\n");
    }
    let tail = limit - head;
    format!(
        "{}\n… [{} lines omitted]\n{}",
        lines[..head].join("\n"),
        lines.len() - limit,
        lines[lines.len() - tail..].join("\n")
    )
}

#[derive(Default)]
struct Notebook {
    assets: Vec<Asset>,
    text_left: usize,
    image_left: usize,
    text_exhausted: bool,
    image_exhausted: bool,
    /// Data types of outputs that had no text or image form.
    skipped: BTreeSet<String>,
    undecodable_images: usize,
}

impl Notebook {
    /// A fenced block of output text, or nothing when there is none or the
    /// notebook's budget is spent (a note says so once).
    fn text_block(&mut self, text: &str, limit: usize, head: usize) -> Option<String> {
        let clipped = clip(&terminal_text(text), limit, head);
        if clipped.trim().is_empty() {
            return None;
        }
        self.spend_text(clipped.len())
            .then(|| fence(&clipped, "text"))
    }

    fn spend_text(&mut self, bytes: usize) -> bool {
        if bytes > self.text_left {
            self.text_exhausted = true;
            return false;
        }
        self.text_left -= bytes;
        true
    }

    /// A decoded image as an asset, and the Markdown that shows it.
    fn image(&mut self, encoded: &str, extension: &str, alt: &str) -> Option<String> {
        let encoded: String = encoded.chars().filter(|c| !c.is_whitespace()).collect();
        // The decoded size is known before decoding.
        if encoded.len() / 4 * 3 > self.image_left {
            self.image_exhausted = true;
            return None;
        }
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(&encoded)
            .ok()
            .filter(|bytes| !bytes.is_empty());
        let Some(bytes) = bytes else {
            self.undecodable_images += 1;
            return None;
        };
        self.image_left -= bytes.len();
        let name = format!("notebook-image-{}.{extension}", self.assets.len() + 1);
        let reference = format!("![{alt}](.markitai/assets/{name})");
        self.assets.push(Asset { name, bytes });
        Some(reference)
    }

    /// The blocks one code cell's outputs become.
    fn outputs(&mut self, outputs: Option<&Value>) -> Vec<String> {
        let Some(outputs) = outputs.and_then(Value::as_array) else {
            return Vec::new();
        };
        let mut blocks = Vec::new();
        // Consecutive writes to one stream are one block, as the notebook shows them.
        let mut stream: Option<(String, String)> = None;
        for output in outputs {
            let kind = output.get("output_type").and_then(Value::as_str);
            if kind == Some("stream") {
                let name = output
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or("stdout");
                let text = output
                    .get("text")
                    .and_then(lenient_text)
                    .unwrap_or_default();
                match &mut stream {
                    Some((current, pending)) if current == name => pending.push_str(&text),
                    _ => {
                        self.flush_stream(&mut stream, &mut blocks);
                        stream = Some((name.to_owned(), text));
                    }
                }
                continue;
            }
            self.flush_stream(&mut stream, &mut blocks);
            match kind {
                Some("execute_result" | "display_data") => {
                    if let Some(data) = output.get("data") {
                        self.rich(data, &mut blocks);
                    }
                }
                Some("error") => {
                    let name = output.get("ename").and_then(Value::as_str).unwrap_or("");
                    let value = output.get("evalue").and_then(Value::as_str).unwrap_or("");
                    let traceback = output
                        .get("traceback")
                        .and_then(Value::as_array)
                        .map(|lines| {
                            lines
                                .iter()
                                .filter_map(Value::as_str)
                                .collect::<Vec<_>>()
                                .join("\n")
                        })
                        .unwrap_or_default();
                    let text = if terminal_text(&traceback).trim().is_empty() {
                        match (name, value) {
                            ("", value) => value.to_owned(),
                            (name, "") => name.to_owned(),
                            (name, value) => format!("{name}: {value}"),
                        }
                    } else {
                        traceback
                    };
                    blocks.extend(self.text_block(&text, TRACEBACK_LINES, TRACEBACK_HEAD));
                }
                _ => {}
            }
        }
        self.flush_stream(&mut stream, &mut blocks);
        blocks
    }

    fn flush_stream(&mut self, stream: &mut Option<(String, String)>, blocks: &mut Vec<String>) {
        if let Some((_, text)) = stream.take() {
            blocks.extend(self.text_block(&text, OUTPUT_LINES, OUTPUT_HEAD));
        }
    }

    /// A result or a display: Markdown when it has some, else its image,
    /// else its plain text (a figure's `<Figure size ...>` is only the
    /// object's name, so it gives way to the figure).
    fn rich(&mut self, data: &Value, blocks: &mut Vec<String>) {
        let Some(data) = data.as_object() else {
            return;
        };
        if let Some(markdown) = data.get("text/markdown").and_then(lenient_text) {
            let markdown = markdown.trim_matches('\n');
            if !markdown.trim().is_empty() {
                if self.spend_text(markdown.len()) {
                    blocks.push(markdown.to_owned());
                }
                return;
            }
        }
        let mut shown = false;
        for (mime, extension) in IMAGE_TYPES {
            if let Some(encoded) = data.get(mime).and_then(lenient_text) {
                if let Some(reference) = self.image(&encoded, extension, "output") {
                    blocks.push(reference);
                    shown = true;
                } else {
                    shown = true;
                    blocks.push("*[Image output omitted]*".to_owned());
                }
            }
        }
        if shown {
            return;
        }
        match data.get("text/plain").and_then(lenient_text) {
            Some(plain) => blocks.extend(self.text_block(&plain, OUTPUT_LINES, OUTPUT_HEAD)),
            None => self.skipped.extend(data.keys().cloned()),
        }
    }

    /// A markdown cell with its attached images (`![](attachment:name.png)`)
    /// pointing at assets.
    fn markdown_cell(&mut self, source: &str, attachments: Option<&Value>) -> String {
        let mut text = source.trim_end().to_owned();
        let Some(attachments) = attachments.and_then(Value::as_object) else {
            return text;
        };
        for (name, bundle) in attachments {
            // Spaces in a name are percent-encoded in a Markdown destination.
            let spellings = [name.clone(), name.replace(' ', "%20")];
            let used = |text: &str| {
                spellings.iter().any(|spelled| {
                    ["](", "=\"", "='"]
                        .iter()
                        .any(|before| text.contains(&format!("{before}attachment:{spelled}")))
                })
            };
            if !used(&text) {
                continue;
            }
            for (mime, extension) in IMAGE_TYPES {
                let Some(encoded) = bundle.get(mime).and_then(lenient_text) else {
                    continue;
                };
                let Some(reference) = self.image(&encoded, extension, "") else {
                    break;
                };
                let target = reference
                    .strip_prefix("![](")
                    .and_then(|rest| rest.strip_suffix(')'))
                    .unwrap_or_default()
                    .to_owned();
                for spelled in &spellings {
                    for before in ["](", "=\"", "='"] {
                        text = text.replace(
                            &format!("{before}attachment:{spelled}"),
                            &format!("{before}{target}"),
                        );
                    }
                }
                break;
            }
        }
        text
    }
}

pub(super) fn read(source: &str) -> Result<Document> {
    read_within(source, TEXT_BUDGET, IMAGE_BUDGET)
}

fn read_within(source: &str, text_budget: usize, image_budget: usize) -> Result<Document> {
    let value: Value = serde_json::from_str(source)?;
    let cells = value
        .get("cells")
        .and_then(Value::as_array)
        .ok_or_else(|| Error::Conversion("Notebook cells must be an array".into()))?;
    let language = value
        .pointer("/metadata/language_info/name")
        .and_then(Value::as_str)
        .unwrap_or("python");
    let language: String = language
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '+' | '-'))
        .collect();
    let mut notebook = Notebook {
        text_left: text_budget,
        image_left: image_budget,
        ..Notebook::default()
    };
    let mut blocks = Vec::new();
    for item in cells {
        let content = source_text(
            item.get("source")
                .ok_or_else(|| Error::Conversion("Notebook cell has no source".into()))?,
        )?;
        match item.get("cell_type").and_then(Value::as_str) {
            Some("code") => {
                blocks.push(fence(&content, &language));
                blocks.extend(notebook.outputs(item.get("outputs")));
            }
            Some("markdown") => {
                blocks.push(notebook.markdown_cell(&content, item.get("attachments")));
            }
            Some("raw") => blocks.push(fence(&content, "")),
            _ => return Err(Error::Conversion("Unknown notebook cell_type".into())),
        }
    }
    let mut warnings = Vec::new();
    if notebook.text_exhausted {
        blocks.push("*[Further notebook output omitted: the output limit was reached.]*".into());
        warnings.push(format!(
            "Notebook output beyond {} KiB was omitted.",
            text_budget / 1024
        ));
    }
    if notebook.image_exhausted {
        warnings.push(format!(
            "Notebook images beyond {} KiB were omitted.",
            image_budget / 1024
        ));
    }
    if notebook.undecodable_images > 0 {
        warnings.push(format!(
            "{} notebook image(s) were not valid base64 and were omitted.",
            notebook.undecodable_images
        ));
    }
    if !notebook.skipped.is_empty() {
        warnings.push(format!(
            "Notebook outputs with no text or image form ({}) were not converted.",
            notebook
                .skipped
                .iter()
                .cloned()
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    let mut result = Document {
        markdown: blocks.join("\n\n"),
        assets: notebook.assets,
        warnings,
        ..Document::default()
    };
    if let Some(title) = value.pointer("/metadata/title").and_then(Value::as_str) {
        result.metadata.insert("title".into(), title.into());
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// A 1x1 PNG.
    const PIXEL: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/q842iQAAAABJRU5ErkJggg==";

    fn notebook(cells: Value) -> String {
        json!({"nbformat": 4, "metadata": {"language_info": {"name": "python"}}, "cells": cells})
            .to_string()
    }

    fn code(source: &str, outputs: Value) -> Value {
        json!({"cell_type": "code", "source": source, "outputs": outputs})
    }

    #[test]
    fn streams_results_and_errors_follow_their_code_cell() {
        let doc = read(&notebook(json!([
            {"cell_type": "markdown", "source": "# Title"},
            code("print('hi')\ndf", json!([
                {"output_type": "stream", "name": "stdout", "text": ["hi\n"]},
                {"output_type": "stream", "name": "stdout", "text": ["again\n"]},
                {"output_type": "execute_result", "execution_count": 1,
                 "data": {"text/plain": ["   a  b\n", "0  1  2"],
                          "text/html": "<table><tr><td>1</td></tr></table>"}},
                {"output_type": "error", "ename": "ValueError", "evalue": "bad",
                 "traceback": ["\u{1b}[0;31m---------------\u{1b}[0m", "\u{1b}[0;31mValueError\u{1b}[0m: bad"]}
            ])),
            {"cell_type": "raw", "source": "raw text"}
        ])))
        .unwrap();
        assert_eq!(
            doc.markdown,
            "# Title\n\n```python\nprint('hi')\ndf\n```\n\n```text\nhi\nagain\n```\n\n\
             ```text\n   a  b\n0  1  2\n```\n\n```text\n---------------\nValueError: bad\n```\n\n\
             ```\nraw text\n```"
        );
        assert!(doc.assets.is_empty());
        assert!(doc.warnings.is_empty(), "{:?}", doc.warnings);
    }

    #[test]
    fn markdown_results_stay_markdown_and_the_other_stream_is_its_own_block() {
        let doc = read(&notebook(json!([code(
            "x",
            json!([
                {"output_type": "stream", "name": "stdout", "text": "out\n"},
                {"output_type": "stream", "name": "stderr", "text": "warning\n"},
                {"output_type": "display_data", "data": {
                    "text/markdown": ["**Result**\n", "\n", "| a |\n", "| - |\n", "| 1 |\n"],
                    "text/plain": "Result"}}
            ])
        )])))
        .unwrap();
        assert_eq!(
            doc.markdown,
            "```python\nx\n```\n\n```text\nout\n```\n\n```text\nwarning\n```\n\n\
             **Result**\n\n| a |\n| - |\n| 1 |"
        );
    }

    #[test]
    fn an_image_output_is_an_asset_and_replaces_the_figure_s_text_form() {
        let doc = read(&notebook(json!([code(
            "plot()",
            json!([
                {"output_type": "display_data", "data": {
                    "image/png": PIXEL, "text/plain": "<Figure size 640x480 with 1 Axes>"}},
                {"output_type": "display_data", "data": {
                    "image/jpeg": [&PIXEL[..20], "\n", &PIXEL[20..]]}}
            ])
        )])))
        .unwrap();
        assert_eq!(
            doc.markdown,
            "```python\nplot()\n```\n\n![output](.markitai/assets/notebook-image-1.png)\n\n\
             ![output](.markitai/assets/notebook-image-2.jpg)"
        );
        assert_eq!(doc.assets.len(), 2);
        assert_eq!(doc.assets[0].name, "notebook-image-1.png");
        assert_eq!(&doc.assets[0].bytes[1..4], b"PNG");
        assert!(!doc.markdown.contains("Figure"));
    }

    #[test]
    fn markdown_cell_attachments_become_assets() {
        let doc = read(&notebook(json!([
            {"cell_type": "markdown", "source": ["![a](attachment:my image.png) and ", "<img src=\"attachment:other.png\"> and ![missing](attachment:none.png)"],
             "attachments": {
                "my image.png": {"image/png": PIXEL},
                "other.png": {"image/png": PIXEL},
                "unused.png": {"image/png": PIXEL}}}
        ])))
        .unwrap();
        assert_eq!(
            doc.markdown,
            "![a](.markitai/assets/notebook-image-1.png) and \
             <img src=\".markitai/assets/notebook-image-2.png\"> and \
             ![missing](attachment:none.png)"
        );
        // Only the attachments the text uses are read.
        assert_eq!(doc.assets.len(), 2);
    }

    #[test]
    fn long_output_is_cut_with_a_note_and_a_traceback_keeps_its_ends() {
        let lines: Vec<String> = (1..=500).map(|n| format!("line {n}\n")).collect();
        let traceback: Vec<String> = (1..=100).map(|n| format!("frame {n}")).collect();
        let wide = "x".repeat(5000);
        let doc = read(&notebook(json!([code(
            "go()",
            json!([
                {"output_type": "stream", "name": "stdout", "text": lines},
                {"output_type": "stream", "name": "stderr", "text": wide},
                {"output_type": "error", "ename": "E", "evalue": "v", "traceback": traceback}
            ])
        )])))
        .unwrap();
        let text = &doc.markdown;
        assert!(
            text.contains("line 40\n… [420 lines omitted]\nline 461\n"),
            "{text}"
        );
        assert!(text.contains("line 500\n```"));
        assert!(!text.contains("line 41\n"));
        assert!(text.contains(&format!("{}… [line cut]", "x".repeat(500))));
        assert!(!text.contains(&"x".repeat(501)));
        assert!(
            text.contains("frame 6\n… [70 lines omitted]\nframe 77\n"),
            "{text}"
        );
        assert!(text.ends_with("frame 100\n```"));
    }

    #[test]
    fn terminal_control_is_removed_and_a_carriage_return_keeps_the_last_state() {
        assert_eq!(
            terminal_text("\u{1b}[1;32mok\u{1b}[0m \u{1b}]0;title\u{7}done \u{1b}[2K\u{1b}(Bx"),
            "ok done x"
        );
        assert_eq!(
            terminal_text("10%\r50%\r100%\nnext\r\nlast\r"),
            "100%\nnext\nlast"
        );
    }

    #[test]
    fn output_beyond_the_budgets_is_omitted_with_a_note_and_a_warning() {
        let big = "y".repeat(400);
        let outputs: Vec<Value> = (0..10)
            .map(|n| json!({"output_type": "stream", "name": if n % 2 == 0 { "stdout" } else { "stderr" }, "text": big}))
            .collect();
        let doc =
            read_within(&notebook(json!([code("a", json!(outputs))])), 1000, 1 << 20).unwrap();
        assert_eq!(doc.markdown.matches(&big).count(), 2);
        assert!(
            doc.markdown.ends_with("limit was reached.]*"),
            "{}",
            doc.markdown
        );
        assert!(doc.warnings.iter().any(|w| w.contains("omitted")));
        // Images count their decoded bytes.
        let image = json!({"output_type": "display_data", "data": {"image/png": PIXEL}});
        let doc = read_within(
            &notebook(json!([code("a", json!([image.clone(), image]))])),
            1000,
            80,
        )
        .unwrap();
        assert_eq!(doc.assets.len(), 1);
        assert!(doc.markdown.contains("*[Image output omitted]*"));
        assert!(doc.warnings.iter().any(|w| w.contains("images beyond")));
    }

    #[test]
    fn malformed_outputs_do_not_fail_the_notebook() {
        let doc = read(&notebook(json!([
            code("a", json!("not a list")),
            code("b", json!([
                7,
                {"output_type": "stream", "text": 5},
                {"output_type": "execute_result", "data": "text"},
                {"output_type": "display_data", "data": {"image/png": "***not base64***"}},
                {"output_type": "display_data", "data": {"text/html": "<b>x</b>", "application/json": {}}},
                {"output_type": "error"},
                {"output_type": "future"}
            ]))
        ])))
        .unwrap();
        assert_eq!(
            doc.markdown,
            "```python\na\n```\n\n```python\nb\n```\n\n*[Image output omitted]*"
        );
        assert!(doc.assets.is_empty());
        assert!(doc.warnings.iter().any(|w| w.contains("not valid base64")));
        assert!(
            doc.warnings
                .iter()
                .any(|w| w.contains("application/json, text/html")),
            "{:?}",
            doc.warnings
        );
        // A notebook without outputs reads as it did.
        let plain = read(&notebook(json!([{"cell_type": "code", "source": "1"}]))).unwrap();
        assert_eq!(plain.markdown, "```python\n1\n```");
    }

    #[test]
    fn output_text_cannot_close_its_fence() {
        let doc = read(&notebook(json!([code(
            "a",
            json!([
                {"output_type": "stream", "name": "stdout", "text": "```\nnot code\n```\n"}
            ])
        )])))
        .unwrap();
        assert!(
            doc.markdown.ends_with("````text\n```\nnot code\n```\n````"),
            "{}",
            doc.markdown
        );
    }
}
