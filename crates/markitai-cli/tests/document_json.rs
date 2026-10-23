//! `--format json`: the document on stdout, beside the Markdown, under an
//! exact name and in a batch with its index (docs/json-output.md).
#[path = "support/console.rs"]
mod console;

use serde_json::Value;
use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};

struct Fixture(tempfile::TempDir);

impl Fixture {
    fn new() -> Self {
        let fixture = Self(tempfile::tempdir().unwrap());
        fs::create_dir(fixture.path("home")).unwrap();
        fs::create_dir(fixture.path("state")).unwrap();
        fs::write(fixture.path("state/config.json"), br#"{"history":{"record":false},"log":{"dir":null},"cache":{"enabled":false},"llm":{"enabled":false},"ocr":{"enabled":false},"screenshot":{"enabled":false},"image":{"alt_enabled":false,"desc_enabled":false}}"#).unwrap();
        fixture
    }
    fn path(&self, name: &str) -> PathBuf {
        self.0.path().join(name)
    }
    fn run(&self, args: &[&str]) -> Output {
        let mut command = Command::new(env!("CARGO_BIN_EXE_markitai"));
        console::english(command.env_clear());
        for name in ["PATH", "SYSTEMROOT", "TMPDIR", "TEMP", "TMP"] {
            if let Some(value) = std::env::var_os(name) {
                command.env(name, value);
            }
        }
        command
            .current_dir(self.0.path())
            .env("HOME", self.path("home"))
            .env("MARKITAI_HOME", self.path("state"))
            .env("USERPROFILE", self.path("home"))
            .args(args)
            .output()
            .unwrap()
    }
    fn json(&self, name: &str) -> Value {
        serde_json::from_slice(&fs::read(self.path(name)).unwrap()).unwrap()
    }
}

fn good(output: &Output) {
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn stdout_carries_the_document_with_every_key() {
    let fixture = Fixture::new();
    fs::write(fixture.path("note.txt"), b"# Note\n\nHello world.\n").unwrap();
    let output = fixture.run(&["note.txt", "--format", "JSON"]);
    good(&output);
    let document: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(document["schema_version"], "1.0");
    assert_eq!(document["tool"]["name"], "markitai");
    assert_eq!(document["source"]["kind"], "file");
    assert_eq!(document["source"]["path"], "note.txt");
    assert_eq!(document["source"]["size"], 21);
    assert_eq!(
        document["source"]["sha256"].as_str().unwrap().len(),
        64,
        "{document}"
    );
    assert_eq!(document["title"], "Note");
    assert_eq!(document["markdown"]["llm"], Value::Null);
    assert!(
        document["markdown"]["base"]
            .as_str()
            .unwrap()
            .contains("Hello world.")
    );
    assert_eq!(document["text"], "Note\n\nHello world.");
    assert_eq!(document["outputs"]["markdown"], Value::Null);
    // Top-level keys in their documented order (the parsed map sorts them).
    let text = String::from_utf8_lossy(&output.stdout);
    let order = [
        "schema_version",
        "tool",
        "source",
        "status",
        "skip_reason",
        "title",
        "properties",
        "markdown",
        "text",
        "pages",
        "images",
        "tables",
        "notices",
        "usage",
        "timings",
        "outputs",
    ];
    let positions: Vec<usize> = order
        .iter()
        .map(|key| text.find(&format!("\n  \"{key}\":")).unwrap())
        .collect();
    assert!(positions.windows(2).all(|pair| pair[0] < pair[1]), "{text}");
    assert_eq!(document.as_object().unwrap().len(), order.len());
    // Markdown stays the default.
    let output = fixture.run(&["note.txt", "-f", "md"]);
    good(&output);
    assert!(String::from_utf8_lossy(&output.stdout).starts_with("---\n"));
}

#[test]
fn with_an_output_folder_the_json_sits_beside_the_markdown_and_follows_its_renames() {
    let fixture = Fixture::new();
    fs::write(fixture.path("note.txt"), b"Hello.\n").unwrap();
    let output = fixture.run(&["note.txt", "-f", "json", "-o", "out"]);
    good(&output);
    // Output locations are built with `Path::join`: Windows shows `\` there.
    let wrote = format!("Wrote out{}note.txt.json", std::path::MAIN_SEPARATOR);
    assert!(
        String::from_utf8_lossy(&output.stderr).contains(&wrote),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let document = fixture.json("out/note.txt.json");
    assert_eq!(document["outputs"]["markdown"], "note.txt.md");
    assert!(fixture.path("out/note.txt.md").is_file());
    good(&fixture.run(&["note.txt", "-f", "json", "-o", "out"]));
    assert_eq!(
        fixture.json("out/note.txt.v2.json")["outputs"]["markdown"],
        "note.txt.v2.md"
    );
    // A JSON left by an earlier run is not overwritten under `rename`.
    fs::remove_file(fixture.path("out/note.txt.v2.md")).unwrap();
    good(&fixture.run(&["note.txt", "-f", "json", "-o", "out"]));
    assert!(fixture.path("out/note.txt.v3.json").is_file());
}

#[test]
fn an_exact_json_name_names_the_markdown_beside_it() {
    let fixture = Fixture::new();
    fs::write(fixture.path("note.txt"), b"Hello.\n").unwrap();
    good(&fixture.run(&["note.txt", "-f", "json", "-o", "out/result.json"]));
    assert_eq!(
        fixture.json("out/result.json")["outputs"]["markdown"],
        "result.md"
    );
    assert!(fixture.path("out/result.md").is_file());
}

#[test]
fn a_batch_writes_one_document_per_file_and_an_index() {
    let fixture = Fixture::new();
    fs::create_dir(fixture.path("in")).unwrap();
    fs::write(fixture.path("in/a.txt"), b"A.\n").unwrap();
    fs::write(fixture.path("in/b.md"), b"# B\n").unwrap();
    let output = fixture.run(&["in", "-f", "json", "-o", "out"]);
    good(&output);
    let index = fixture.json("out/markitai.index.json");
    assert_eq!(index["schema_version"], "1.0");
    let items = index["items"].as_array().unwrap();
    assert_eq!(items.len(), 2);
    assert_eq!(items[0]["source"], "a.txt");
    assert_eq!(items[0]["status"], "completed");
    assert_eq!(items[0]["document"], "a.txt.json");
    assert_eq!(items[0]["markdown"], "a.txt.md");
    assert_eq!(items[0]["error"], Value::Null);
    assert_eq!(items[1]["document"], "b.md.json");
    assert!(
        fixture.json("out/b.md.json")["markdown"]["base"]
            .as_str()
            .unwrap()
            .contains("# B")
    );
    // A resumed or repeated run keeps the items it did not touch.
    fs::remove_file(fixture.path("in/a.txt")).unwrap();
    good(&fixture.run(&["in", "-f", "json", "-o", "out"]));
    let index = fixture.json("out/markitai.index.json");
    let sources: Vec<&str> = index["items"]
        .as_array()
        .unwrap()
        .iter()
        .map(|item| item["source"].as_str().unwrap())
        .collect();
    assert_eq!(sources, ["a.txt", "b.md"]);
}
