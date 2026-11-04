//! A conversion's envelope mapped to Done or Failed (DESIGN.md 4.1 step 4
//! and 4.2), and the request that starts it (step 2).

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::settings::{ConfigSource, Options};
use super::source::Source;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CostStatus {
    Complete,
    Partial,
    Unknown,
}

/// AI usage, present only when the model was asked at least once.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Usage {
    pub requests: u64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cost_usd: f64,
    pub cost: CostStatus,
}

impl Usage {
    pub fn tokens(&self) -> u64 {
        self.input_tokens + self.output_tokens
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Converted {
    pub markdown: String,
    pub markdown_path: PathBuf,
    pub duration: f64,
    pub warnings: Vec<String>,
    pub usage: Option<Usage>,
    pub title: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Failure {
    pub code: String,
    pub message: String,
    /// The output folder no folder could be claimed in, for
    /// `output_unavailable`: the one at that moment, not the current one.
    pub folder: Option<PathBuf>,
}

impl Failure {
    pub fn new(code: &str, message: impl Into<String>) -> Failure {
        Failure {
            code: code.to_owned(),
            message: message.into(),
            folder: None,
        }
    }

    pub fn output_unavailable(folder: PathBuf, message: impl Into<String>) -> Failure {
        Failure {
            folder: Some(folder),
            ..Failure::new("output_unavailable", message)
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Outcome {
    Done(Box<Converted>),
    Failed(Failure),
}

/// `convert_json`'s request for one item: `alt` and `desc` only with `llm`,
/// `config` omitted for the user's configuration and `{}` for built-in
/// defaults.
pub fn convert_request(source: &Source, output_dir: &Path, options: &Options) -> String {
    let mut opts = json!({
        "output_dir": output_dir,
        "llm": options.llm,
        "ocr": options.ocr,
        "screenshot": options.screenshot,
        "alt": options.llm && options.alt,
        "desc": options.llm && options.desc,
    });
    if options.config == ConfigSource::Builtin {
        opts["config"] = json!({});
    }
    json!({"source": source.text(), "options": opts}).to_string()
}

/// Done when the envelope is `ok` with no `skip_reason`; any skip is Failed
/// with the app's code `skipped`, and an error keeps the core's code
/// (`image_only` included).
pub fn map_envelope(response: &str) -> Outcome {
    let envelope: Value = match serde_json::from_str(response) {
        Ok(value) => value,
        Err(e) => return Outcome::Failed(Failure::new("invalid_json", e.to_string())),
    };
    if envelope["ok"] != Value::Bool(true) {
        let error = &envelope["error"];
        let code = error["code"].as_str().unwrap_or("internal_error");
        let message = error["message"].as_str().unwrap_or_default();
        return Outcome::Failed(Failure::new(code, message));
    }
    let result = &envelope["result"];
    if let Some(reason) = result["skip_reason"].as_str() {
        return Outcome::Failed(Failure::new("skipped", format!("skip_reason: {reason}")));
    }
    let text = |key: &str| result[key].as_str().filter(|s| !s.is_empty());
    let Some(markdown_path) = text("llm_output_path").or_else(|| text("output_path")) else {
        return Outcome::Failed(Failure::new(
            "internal_error",
            "The conversion finished without an output file.",
        ));
    };
    let markdown = result["llm_markdown"]
        .as_str()
        .or_else(|| result["markdown"].as_str())
        .unwrap_or_default();
    let warnings = result["warnings"]
        .as_array()
        .map(|list| {
            list.iter()
                .filter_map(|w| w.as_str().map(str::to_owned))
                .collect()
        })
        .unwrap_or_default();
    Outcome::Done(Box::new(Converted {
        markdown: markdown.to_owned(),
        markdown_path: PathBuf::from(markdown_path),
        duration: result["duration"].as_f64().unwrap_or(0.0),
        warnings,
        usage: usage(&result["usage"]),
        title: frontmatter_title(&result["frontmatter"]),
    }))
}

fn usage(usage: &Value) -> Option<Usage> {
    let count = |key: &str| usage[key].as_u64().unwrap_or(0);
    let requests = count("requests");
    if requests == 0 {
        return None;
    }
    let cost_usd = usage["cost_usd"].as_f64().unwrap_or(0.0);
    let rows: Vec<&Value> = match &usage["by_model"] {
        Value::Object(map) => map.values().collect(),
        Value::Array(list) => list.iter().collect(),
        _ => Vec::new(),
    };
    let complete = rows
        .iter()
        .all(|row| row["cost_status"].as_str() == Some("complete"));
    let cost = if complete {
        CostStatus::Complete
    } else if cost_usd > 0.0 {
        CostStatus::Partial
    } else {
        CostStatus::Unknown
    };
    Some(Usage {
        requests,
        input_tokens: count("input_tokens"),
        output_tokens: count("output_tokens"),
        cost_usd,
        cost,
    })
}

/// A non-empty `title`, trimmed, at most 200 characters.
fn frontmatter_title(frontmatter: &Value) -> Option<String> {
    let title = frontmatter["title"].as_str()?.trim();
    (!title.is_empty()).then(|| title.chars().take(200).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn done(response: Value) -> Converted {
        match map_envelope(&response.to_string()) {
            Outcome::Done(converted) => *converted,
            other => panic!("not done: {other:?}"),
        }
    }

    fn failed(response: Value) -> Failure {
        match map_envelope(&response.to_string()) {
            Outcome::Failed(failure) => failure,
            other => panic!("not failed: {other:?}"),
        }
    }

    #[test]
    fn the_llm_text_and_file_win_when_ai_ran() {
        let converted = done(json!({"ok": true, "result": {
            "markdown": "plain", "llm_markdown": "tidy",
            "output_path": "/o/a.md", "llm_output_path": "/o/a.llm.md",
            "duration": 18.6, "warnings": ["w1", "w2"], "skip_reason": null,
            "frontmatter": {"title": "  Quarterly  "},
            "usage": {"requests": 6, "input_tokens": 19870, "output_tokens": 4440, "cost_usd": 0.0412,
                      "by_model": {"m": {"cost_status": "complete"}}}
        }}));
        assert_eq!(converted.markdown, "tidy");
        assert_eq!(converted.markdown_path, PathBuf::from("/o/a.llm.md"));
        assert_eq!(converted.warnings, ["w1", "w2"]);
        assert_eq!(converted.title.as_deref(), Some("Quarterly"));
        let usage = converted.usage.unwrap();
        assert_eq!(usage.tokens(), 24310);
        assert_eq!(usage.cost, CostStatus::Complete);
    }

    #[test]
    fn without_ai_the_plain_text_and_no_usage() {
        let converted = done(json!({"ok": true, "result": {
            "markdown": "plain", "llm_markdown": null, "output_path": "/o/a.md",
            "llm_output_path": null, "duration": 0.2, "warnings": [],
            "usage": {"requests": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "by_model": {}},
            "frontmatter": {"title": ""}
        }}));
        assert_eq!(converted.markdown, "plain");
        assert_eq!(converted.markdown_path, PathBuf::from("/o/a.md"));
        assert_eq!(converted.usage, None);
        assert_eq!(converted.title, None);
    }

    #[test]
    fn cost_is_partial_or_unknown_when_a_row_is_not_complete() {
        let cost = |cost_usd: f64| {
            done(json!({"ok": true, "result": {"markdown": "", "output_path": "/o/a.md",
                "usage": {"requests": 2, "cost_usd": cost_usd,
                          "by_model": {"a": {"cost_status": "complete"}, "b": {"cost_status": "unknown"}}}}}))
            .usage
            .unwrap()
            .cost
        };
        assert_eq!(cost(0.5), CostStatus::Partial);
        assert_eq!(cost(0.0), CostStatus::Unknown);
    }

    #[test]
    fn skips_and_errors_are_failures_with_their_codes() {
        let skip = |reason: &str| {
            failed(json!({"ok": true, "result": {"markdown": "", "output_path": null, "skip_reason": reason}}))
                .code
        };
        assert_eq!(skip("image_only"), "skipped");
        assert_eq!(skip("exists"), "skipped");
        let image = failed(json!({"ok": false, "error": {"code": "image_only", "message": "no text"}}));
        assert_eq!(image.code, "image_only");
        let error = failed(json!({"ok": false, "error": {"code": "not_found", "message": "No such file"}}));
        assert_eq!(error, Failure::new("not_found", "No such file"));
        assert_eq!(
            map_envelope("garbage"),
            Outcome::Failed(Failure::new("invalid_json", "expected value at line 1 column 1"))
        );
    }

    #[test]
    fn long_titles_are_cut_at_200_characters() {
        let long = "标".repeat(250);
        let converted = done(json!({"ok": true, "result": {"markdown": "", "output_path": "/o/a.md",
            "frontmatter": {"title": long}}}));
        assert_eq!(converted.title.unwrap().chars().count(), 200);
    }

    #[test]
    fn the_request_sends_alt_and_desc_only_with_llm() {
        let source = Source::File("/in/a.pdf".into());
        let options = Options { alt: true, desc: true, ..Options::default() };
        let request: Value =
            serde_json::from_str(&convert_request(&source, Path::new("/out/a"), &options)).unwrap();
        assert_eq!(request["source"], "/in/a.pdf");
        assert_eq!(request["options"]["output_dir"], "/out/a");
        assert_eq!(request["options"]["alt"], false);
        assert_eq!(request["options"]["desc"], false);
        assert!(request["options"].get("config").is_none());
        let options = Options { llm: true, alt: true, config: ConfigSource::Builtin, ..Options::default() };
        let request: Value =
            serde_json::from_str(&convert_request(&source, Path::new("/out/a"), &options)).unwrap();
        assert_eq!(request["options"]["alt"], true);
        assert_eq!(request["options"]["desc"], false);
        assert_eq!(request["options"]["config"], json!({}));
    }
}
