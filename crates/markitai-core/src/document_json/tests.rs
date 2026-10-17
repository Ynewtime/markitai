use super::*;
use serde_json::json;
use std::path::PathBuf;

const SCHEMA: &str = include_str!("../../../../docs/schema/document-v1.schema.json");
/// The field set of schema version 1.0, frozen: later versions keep every
/// line (a key is never removed, renamed or retyped) and may only add lines.
const FIELDS_V1_0: &str = include_str!("fields-v1.0.txt");
const SNAPSHOT: &str = include_str!("snapshot-paged.json");

/// A scanned two-page PDF read by a model: page markers, a page screenshot,
/// an analysed picture, a table, notices and model usage.
fn paged() -> ConversionOutput {
    let dir = Path::new("/out");
    let paged = "<!-- Page number: 1 -->\n\n# Report\n\nIntro with a [link](https://example.test) and `code`.\n\n![A chart](.markitai/assets/chart.png)\n\n<!-- ![Page 1](.markitai/screenshots/report.pdf.page0001.jpg) -->\n\n<!-- Page number: 2 -->\n\n| Name | Value |\n|---|---|\n| a | 1 |\n| b | 2 |\n\n```text\n<!-- Page number: 9 -->\n```\n\n<!-- Page images for reference -->\n<!-- ![Page 2](.markitai/screenshots/report.pdf.page0002.jpg) -->\n";
    let mut frontmatter = Map::new();
    frontmatter.insert("tags".into(), json!(["report"]));
    frontmatter.insert("title".into(), json!("Report"));
    frontmatter.insert("source".into(), json!("report.pdf"));
    frontmatter.insert("description".into(), json!("A short report."));
    frontmatter.insert(
        "markitai_processed".into(),
        json!("2026-10-17T12:00:00.000+08:00"),
    );
    let mut by_model = Map::new();
    by_model.insert(
        "openai/gpt-test".into(),
        json!({"requests":2,"input_tokens":100,"output_tokens":40,"cost_usd":0.0012,"priced_requests":2,"unpriced_requests":0,"cost_status":"complete"}),
    );
    ConversionOutput {
        source: "https://user:secret@example.test/report.pdf?token=abc".into(),
        markdown: "# Report\n\nBase text.\n".into(),
        llm_markdown: Some(crate::output_profiles::remove_page_markers(paged)),
        paged_markdown: Some(paged.into()),
        frontmatter,
        output_path: None,
        llm_output_path: Some(dir.join("report.pdf.llm.md")),
        assets: vec![dir.join(".markitai/assets/chart.png")],
        screenshots: vec![
            dir.join(".markitai/screenshots/report.pdf.page0001.jpg"),
            dir.join(".markitai/screenshots/report.pdf.page0002.jpg"),
        ],
        images: vec![json!({
            "asset": "/out/.markitai/assets/chart.png",
            "alt": "A chart",
            "desc": "Bars rising to the right.",
            "text": "",
        })],
        usage: crate::ConversionUsage {
            cost_usd: 0.0012,
            requests: 2,
            input_tokens: 100,
            output_tokens: 40,
            by_model,
        },
        duration: 1.23456,
        warnings: vec![
            "PDF page 2: native text was not recovered (no text layer); OCR is required for this page.".into(),
            "Something nobody classified yet.".into(),
        ],
        ..Default::default()
    }
}

fn render(result: &ConversionOutput) -> Value {
    let text = document_json(
        result,
        &DocumentJsonContext {
            base_dir: Some(Path::new("/out")),
            ..Default::default()
        },
    )
    .render();
    serde_json::from_str(&text).unwrap()
}

/// The document as text with the build commit fixed, as the snapshot has it.
fn snapshot_text(result: &ConversionOutput) -> String {
    let text = document_json(
        result,
        &DocumentJsonContext {
            base_dir: Some(Path::new("/out")),
            ..Default::default()
        },
    )
    .render();
    text.replace(
        &format!("\"commit\": \"{}\"", crate::build_info::COMMIT),
        "\"commit\": \"COMMIT\"",
    )
    .replace(
        &format!("\"version\": \"{}\"", crate::VERSION),
        "\"version\": \"VERSION\"",
    )
}

#[test]
fn a_paged_document_matches_its_snapshot_byte_for_byte() {
    let text = snapshot_text(&paged());
    if std::env::var_os("MARKITAI_UPDATE_SNAPSHOTS").is_some() {
        let path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/document_json/snapshot-paged.json");
        std::fs::write(path, &text).unwrap();
        return;
    }
    assert_eq!(
        text, SNAPSHOT,
        "the document JSON changed; review it, then rerun with MARKITAI_UPDATE_SNAPSHOTS=1"
    );
}

/// `path: type` of every key: list items under `[]`, the free-form front
/// matter object as one entry.
fn fields(value: &Value, path: &str, out: &mut std::collections::BTreeSet<String>) {
    let kind = match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(number) if number.is_f64() => "number",
        Value::Number(_) => "integer",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    };
    if !path.is_empty() {
        out.insert(format!("{path}: {kind}"));
    }
    match value {
        Value::Object(map) if path != "properties" => {
            for (key, item) in map {
                let child = if path.is_empty() {
                    key.clone()
                } else {
                    format!("{path}.{key}")
                };
                fields(item, &child, out);
            }
        }
        Value::Array(items) => {
            for item in items {
                fields(item, &format!("{path}[]"), out);
            }
        }
        _ => {}
    }
}

/// Every key of the documents that exercise every field, with the types it
/// takes across them.
fn current_fields() -> std::collections::BTreeSet<String> {
    let mut found = std::collections::BTreeSet::new();
    let mut empty = ConversionOutput {
        source: "notes.txt".into(),
        markdown: "Plain.\n".into(),
        skip_reason: Some("image_only".into()),
        ..Default::default()
    };
    empty.frontmatter.insert("title".into(), json!("notes"));
    // A local file: size and hash, a base output, no title, a picture and a
    // table before the first page marker, a page without a screenshot and
    // an image with text but no description.
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("scan.pdf");
    std::fs::write(&source, b"%PDF-1.4").unwrap();
    let local = ConversionOutput {
        source: source.to_string_lossy().into_owned(),
        markdown:
            "![x](.markitai/assets/a.png)\n\n|h|\n|-|\n|v|\n\n<!-- Page number: 1 -->\n\nText\n"
                .into(),
        output_path: Some(PathBuf::from("/out/scan.pdf.md")),
        images: vec![json!({"asset": "/out/.markitai/assets/a.png", "alt": "x", "text": "Words"})],
        ..Default::default()
    };
    fields(&render(&paged()), "", &mut found);
    fields(&render(&empty), "", &mut found);
    fields(&render(&local), "", &mut found);
    found
}

#[test]
fn the_v1_field_set_is_only_ever_extended() {
    let frozen: std::collections::BTreeSet<String> = FIELDS_V1_0
        .lines()
        .filter(|line| !line.trim().is_empty() && !line.starts_with('#'))
        .map(str::to_owned)
        .collect();
    let current = current_fields();
    let missing: Vec<_> = frozen.difference(&current).collect();
    assert!(
        missing.is_empty(),
        "schema 1 keys may not be removed, renamed or retyped: {missing:?}"
    );
    let added: Vec<_> = current.difference(&frozen).collect();
    assert!(
        added.is_empty() || SCHEMA_VERSION != "1.0",
        "new keys {added:?} need a minor schema version (1.1) and docs/json-output.md"
    );
}

#[test]
fn a_consumer_of_version_1_ignores_keys_a_later_version_adds() {
    #[derive(serde::Deserialize)]
    #[allow(dead_code)]
    struct V1Reader {
        schema_version: String,
        title: Option<String>,
        text: String,
        pages: Vec<Value>,
    }
    let mut document = render(&paged());
    document["future_field"] = json!({"anything": [1, 2]});
    document["pages"][0]["future_field"] = json!(true);
    document["schema_version"] = json!("1.7");
    let reader: V1Reader = serde_json::from_value(document.clone()).unwrap();
    assert_eq!(reader.title.as_deref(), Some("Report"));
    // The published schema accepts it too: it never forbids unknown keys.
    validate(&schema(), &document, "", &schema(), false).unwrap();
}

#[test]
fn every_key_is_present_and_null_stands_for_not_applicable() {
    let document = render(&ConversionOutput {
        source: "notes.txt".into(),
        markdown: "Plain.\n".into(),
        ..Default::default()
    });
    assert_eq!(document["title"], Value::Null);
    assert_eq!(document["markdown"]["llm"], Value::Null);
    assert_eq!(document["source"]["url"], Value::Null);
    assert_eq!(document["source"]["kind"], "file");
    // The file does not exist here: size and hash are unknown, not missing.
    assert_eq!(document["source"]["size"], Value::Null);
    assert_eq!(document["source"]["sha256"], Value::Null);
    assert_eq!(document["pages"], json!([]));
    assert_eq!(document["outputs"]["markdown"], Value::Null);
    validate(&schema(), &document, "", &schema(), true).unwrap();
}

fn schema() -> Value {
    serde_json::from_str(SCHEMA).unwrap()
}

/// The subset of JSON Schema 2020-12 the document schema uses. `strict`
/// also rejects keys the schema does not list (our own output must be
/// fully described), except inside the free-form front matter.
fn validate(
    schema: &Value,
    value: &Value,
    path: &str,
    root: &Value,
    strict: bool,
) -> Result<(), String> {
    if let Some(reference) = schema.get("$ref").and_then(Value::as_str) {
        let name = reference.strip_prefix("#/$defs/").unwrap();
        return validate(&root["$defs"][name], value, path, root, strict);
    }
    let kind_of = |value: &Value| match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(number) if number.is_f64() => "number",
        Value::Number(_) => "integer",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    };
    if let Some(types) = schema.get("type") {
        let allowed: Vec<&str> = match types {
            Value::String(one) => vec![one.as_str()],
            Value::Array(many) => many.iter().filter_map(Value::as_str).collect(),
            _ => unreachable!(),
        };
        let kind = kind_of(value);
        if !allowed.contains(&kind) && !(kind == "integer" && allowed.contains(&"number")) {
            return Err(format!("{path}: {kind} is not {allowed:?}"));
        }
    }
    if let Some(options) = schema.get("enum").and_then(Value::as_array)
        && !options.contains(value)
    {
        return Err(format!("{path}: {value} is not one of {options:?}"));
    }
    if let Some(expected) = schema.get("const")
        && expected != value
    {
        return Err(format!("{path}: {value} is not {expected}"));
    }
    if let (Some(pattern), Some(text)) = (
        schema.get("pattern").and_then(Value::as_str),
        value.as_str(),
    ) && !Regex::new(pattern).unwrap().is_match(text)
    {
        return Err(format!("{path}: {text:?} does not match {pattern}"));
    }
    if let (Some(minimum), Some(number)) = (
        schema.get("minimum").and_then(Value::as_f64),
        value.as_f64(),
    ) && number < minimum
    {
        return Err(format!("{path}: {number} < {minimum}"));
    }
    if let Value::Object(map) = value {
        for key in schema
            .get("required")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let key = key.as_str().unwrap();
            if !map.contains_key(key) {
                return Err(format!("{path}: missing {key}"));
            }
        }
        if let Some(properties) = schema.get("properties").and_then(Value::as_object) {
            for (key, item) in map {
                match properties.get(key) {
                    Some(child) => validate(child, item, &format!("{path}.{key}"), root, strict)?,
                    None if strict => return Err(format!("{path}: {key} is not in the schema")),
                    None => {}
                }
            }
        }
    }
    if let (Value::Array(items), Some(item_schema)) = (value, schema.get("items")) {
        for (index, item) in items.iter().enumerate() {
            validate(item_schema, item, &format!("{path}[{index}]"), root, strict)?;
        }
    }
    Ok(())
}

#[test]
fn the_published_schema_describes_every_key_of_the_output() {
    let schema = schema();
    validate(&schema, &render(&paged()), "", &schema, true).unwrap();
    // Every object the schema describes requires all of its keys.
    fn all_required(node: &Value, path: &str) {
        if let Some(properties) = node.get("properties").and_then(Value::as_object) {
            let required: HashSet<&str> = node["required"]
                .as_array()
                .unwrap_or_else(|| panic!("{path} lists no required keys"))
                .iter()
                .filter_map(Value::as_str)
                .collect();
            for (key, child) in properties {
                assert!(required.contains(key.as_str()), "{path}.{key} is optional");
                all_required(child, &format!("{path}.{key}"));
            }
        }
        if let Some(items) = node.get("items") {
            all_required(items, &format!("{path}[]"));
        }
    }
    all_required(&schema, "");
    for (name, definition) in schema["$defs"].as_object().unwrap() {
        all_required(definition, name);
    }
}

#[test]
fn pages_tables_and_pictures_follow_the_page_markers() {
    let document = render(&paged());
    let pages = document["pages"].as_array().unwrap();
    assert_eq!(pages.len(), 2, "a marker inside a code block is text");
    assert_eq!(pages[0]["number"], 1);
    assert_eq!(pages[0]["images"], json!([".markitai/assets/chart.png"]));
    assert_eq!(
        pages[0]["screenshot"],
        ".markitai/screenshots/report.pdf.page0001.jpg"
    );
    // The reference block at the end names page 2's screenshot.
    assert_eq!(
        pages[1]["screenshot"],
        ".markitai/screenshots/report.pdf.page0002.jpg"
    );
    assert!(
        !pages[1]["markdown"]
            .as_str()
            .unwrap()
            .contains("Page images")
    );
    assert_eq!(document["tables"][0]["page"], 2);
    assert_eq!(document["tables"][0]["header"], json!(["Name", "Value"]));
    assert_eq!(
        document["tables"][0]["rows"],
        json!([["a", "1"], ["b", "2"]])
    );
    let image = &document["images"][0];
    assert_eq!(image["alt"], "A chart");
    assert_eq!(image["description"], "Bars rising to the right.");
    // An empty transcription is no text.
    assert_eq!(image["text"], Value::Null);
    assert_eq!(image["page"], 1);
    assert_eq!(document["notices"][0]["code"], "pdf.page_without_text");
    assert_eq!(document["notices"][0]["items"], json!(["2"]));
    assert_eq!(document["notices"][1]["code"], "other");
    assert_eq!(
        document["source"]["url"],
        "https://example.test/report.pdf?token=REDACTED"
    );
    assert_eq!(document["outputs"]["llm_markdown"], "report.pdf.llm.md");
    assert_eq!(document["timings"]["duration_s"], 1.235);
    // Front matter keeps the Markdown file's order, not the name order.
    let keys: Vec<_> = document["properties"]
        .as_object()
        .unwrap()
        .keys()
        .cloned()
        .collect();
    let text = snapshot_text(&paged());
    let positions: Vec<usize> = [
        "\"title\": \"Report\"",
        "\"source\": \"report.pdf\"",
        "\"description\"",
        "\"tags\"",
        "\"markitai_processed\"",
    ]
    .iter()
    .map(|key| text.find(key).unwrap())
    .collect();
    assert!(
        positions.windows(2).all(|pair| pair[0] < pair[1]),
        "{keys:?}"
    );
}

#[test]
fn plain_text_drops_markup_pictures_and_comments() {
    assert_eq!(
        plain_text(
            "# Title\n\nA **bold** [link](x) and ![pic](p.png) `code`.\n\n<!-- note -->\n\n- one\n- two\n\n|a|b|\n|-|-|\n|1|2|\n"
        ),
        "Title\n\nA bold link and  code.\n\none\ntwo\n\na\tb\n1\t2"
    );
    assert_eq!(plain_text(""), "");
}

#[test]
fn a_local_file_has_its_size_and_hash() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("note.txt");
    std::fs::write(&path, b"hello").unwrap();
    let document = render(&ConversionOutput {
        source: path.to_string_lossy().into_owned(),
        ..Default::default()
    });
    assert_eq!(document["source"]["size"], 5);
    assert_eq!(
        document["source"]["sha256"],
        "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    );
    assert_eq!(document["source"]["type"], "txt");
    assert_eq!(document["source"]["name"], "note.txt");
}
