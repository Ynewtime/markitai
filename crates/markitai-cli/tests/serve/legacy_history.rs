use super::*;
use std::collections::BTreeMap;

const ID: &str = "012345abcdef";
type Inventory = BTreeMap<String, (u32, Option<Vec<u8>>)>;

fn item(id: &str, name: &str, kind: &str, output: &str, output_name: &str) -> Value {
    json!({"item_id":id,"name":name,"kind":kind,"status":"done","error":null,
        "output":output,"output_name":output_name,"duration_ms":17,"cost_usd":0.25,
        "operation":"convert","skipped":false,"skip_reason":null,"warnings":["Saved warning"]})
}

fn history(root: &Path, items: Vec<Value>, bases: Value) -> PathBuf {
    configure(root);
    let folder = root.join("home/serve/jobs").join(ID);
    std::fs::create_dir_all(folder.join("out/assets")).unwrap();
    std::fs::create_dir_all(folder.join("uploads")).unwrap();
    let mut metadata = json!({"job_id":ID,"created_at":"2025-03-04T05:06:07+08:00",
        "finished_at":"2025-03-04T05:06:08+08:00","status":"done","version":2,
        "options":{"origin":"cli","llm":false,"ocr":false},"items":items});
    if !bases.is_null() {
        metadata["native_bases"] = bases;
    }
    std::fs::write(
        folder.join("meta.json"),
        serde_json::to_vec_pretty(&metadata).unwrap(),
    )
    .unwrap();
    folder
}

// Include directory membership, exact bytes and Unix permissions. Comparing only
// parsed metadata would miss an unsolicited migration during a read-only request.
fn inventory(root: &Path) -> Inventory {
    use std::os::unix::fs::PermissionsExt;
    fn visit(root: &Path, current: &Path, result: &mut Inventory) {
        for entry in std::fs::read_dir(current).unwrap() {
            let path = entry.unwrap().path();
            let metadata = std::fs::symlink_metadata(&path).unwrap();
            assert!(!metadata.file_type().is_symlink());
            result.insert(
                path.strip_prefix(root)
                    .unwrap()
                    .to_string_lossy()
                    .into_owned(),
                (
                    metadata.permissions().mode() & 0o7777,
                    metadata.is_file().then(|| std::fs::read(&path).unwrap()),
                ),
            );
            if metadata.is_dir() {
                visit(root, &path, result);
            }
        }
    }
    let mut result = BTreeMap::new();
    visit(root, root, &mut result);
    result
}

fn retry(server: &Server, item: &str) -> Reply {
    server.request(
        "POST",
        &format!("/api/jobs/{ID}/items/{item}/retry"),
        &[("Content-Type", "application/json")],
        br#"{"options":{"llm":false}}"#,
    )
}

fn result(server: &Server, item: &str) -> Value {
    server.json(&format!("/api/jobs/{ID}/items/{item}/result"))
}

#[test]
fn legacy_enhanced_history_reads_in_place_and_survives_restart_without_rewriting_bytes() {
    let temp = tempfile::tempdir().unwrap();
    let mut first = item("i1", "old.txt", "file", "page.llm.md", "page.llm.md");
    first["llm_enhanced"] = json!(true);
    let second = item("i2", "second.txt", "file", "only.llm.md", "only.llm.md");
    let mut third = item(
        "i3",
        "third.txt",
        "file",
        "explicit.llm.md",
        "explicit.llm.md",
    );
    third["llm_enhanced"] = json!(false);
    let folder = history(temp.path(), vec![first, second, third], Value::Null);
    for (name, body) in [
        ("page.md", "Original base. ![shared](assets/shared.png)\n"),
        (
            "page.llm.md",
            "Saved enhanced. ![shared](assets/shared.png)\n",
        ),
        ("only.llm.md", "Enhanced without a base.\n"),
        ("explicit.md", "Explicit false base.\n"),
        ("explicit.llm.md", "Unselected enhanced.\n"),
        ("assets/shared.png", "exact saved asset"),
    ] {
        std::fs::write(folder.join("out").join(name), body).unwrap();
    }
    let before = inventory(&folder);
    for _ in 0..2 {
        let server = Server::start(temp.path());
        assert_eq!(server.json("/api/history")[0]["job_id"], ID);
        let snapshot = server.json(&format!("/api/jobs/{ID}"));
        assert_eq!(snapshot["items"][0]["output_name"], "page.md");
        assert_eq!(snapshot["items"][1]["llm_enhanced"], true);
        assert_eq!(snapshot["items"][2]["llm_enhanced"], false);
        assert_eq!(snapshot["items"][0]["duration_ms"], 17);
        assert_eq!(snapshot["items"][0]["cost_usd"], 0.25);
        assert_eq!(snapshot["items"][0]["warnings"], json!(["Saved warning"]));
        assert_eq!(snapshot["items"][0]["retryable"], false);
        let first = result(&server, "i1");
        assert_eq!(first["variant"], "llm");
        assert!(
            first["markdown"]
                .as_str()
                .unwrap()
                .starts_with("Saved enhanced.")
        );
        let names = first["artifacts"]
            .as_array()
            .unwrap()
            .iter()
            .map(|entry| entry["relpath"].as_str().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(names, ["page.md", "page.llm.md", "assets/shared.png"]);
        assert_eq!(result(&server, "i2")["variant"], "llm");
        assert_eq!(result(&server, "i3")["markdown"], "Explicit false base.\n");
        assert_eq!(retry(&server, "i1").status, 409);
        for name in names {
            let response = server.request("GET", &format!("/api/jobs/{ID}/files/{name}"), &[], &[]);
            assert_eq!(response.status, 200);
            assert_eq!(
                response.body,
                std::fs::read(folder.join("out").join(name)).unwrap()
            );
        }
        let archive = server.request("GET", &format!("/api/jobs/{ID}/archive"), &[], &[]);
        assert_eq!(archive.status, 200);
        let expected = before
            .iter()
            .filter_map(|(name, (_, bytes))| {
                Some((
                    name.strip_prefix("out/")?.to_owned(),
                    bytes.as_ref()?.clone(),
                ))
            })
            .collect::<HashMap<_, _>>();
        assert_eq!(zip_contents(&archive.body), expected);
        server.stop();
        assert_eq!(inventory(&folder), before);
    }
}

#[test]
fn legacy_url_retry_uses_the_base_pair_and_persists_it_across_restart() {
    let origin = Origin::start();
    let temp = tempfile::tempdir().unwrap();
    let folder = history(
        temp.path(),
        vec![item(
            "i1",
            &origin.url("/legacy"),
            "url",
            "page.html.llm.md",
            "page.html.llm.md",
        )],
        Value::Null,
    );
    std::fs::write(folder.join("out/page.html.md"), "Saved base.").unwrap();
    std::fs::write(folder.join("out/page.html.llm.md"), "Saved enhancement.").unwrap();
    let server = Server::start(temp.path());
    assert_eq!(retry(&server, "i1").status, 202);
    let done = server.done(ID);
    assert_eq!(done["items"][0]["output"], "page.html.md");
    assert_eq!(done["items"][0]["output_name"], "page.html.md");
    assert_eq!(done["items"][0]["llm_enhanced"], false);
    let output = result(&server, "i1");
    assert_eq!(output["variant"], "base");
    assert!(
        output["markdown"]
            .as_str()
            .unwrap()
            .contains("Origin response for /legacy.")
    );
    assert_eq!(origin.entered.load(Ordering::SeqCst), 1);
    assert!(!folder.join("out/page.html.llm.md").exists());
    assert!(!folder.join("out/page.html.llm.llm.md").exists());
    server.stop();
    let saved: Value =
        serde_json::from_slice(&std::fs::read(folder.join("meta.json")).unwrap()).unwrap();
    assert_eq!(saved["native_bases"]["i1"], "page.html");
    let server = Server::start(temp.path());
    assert_eq!(result(&server, "i1"), output);
    assert_eq!(origin.entered.load(Ordering::SeqCst), 1);
    server.stop();
}

#[test]
fn native_literal_llm_name_is_authoritative_even_when_enhanced_flag_is_absent() {
    let temp = tempfile::tempdir().unwrap();
    let folder = history(
        temp.path(),
        vec![item(
            "i1",
            "notes.llm",
            "file",
            "notes.llm.md",
            "notes.llm.md",
        )],
        json!({"i1":"notes.llm"}),
    );
    std::fs::write(folder.join("out/notes.llm.md"), "Literal source base.\n").unwrap();
    std::fs::write(folder.join("out/notes.md"), "Unrelated shorter name.\n").unwrap();
    let before = inventory(&folder);
    let server = Server::start(temp.path());
    let snapshot = server.json(&format!("/api/jobs/{ID}"));
    assert_eq!(snapshot["items"][0]["output_name"], "notes.llm.md");
    assert_eq!(snapshot["items"][0]["llm_enhanced"], false);
    let output = result(&server, "i1");
    assert_eq!(output["variant"], "base");
    assert_eq!(output["markdown"], "Literal source base.\n");
    assert_eq!(output["artifacts"].as_array().unwrap().len(), 1);
    server.stop();
    assert_eq!(inventory(&folder), before);
}

#[test]
fn contradictory_and_overlapping_legacy_families_cannot_retry_or_delete() {
    for overlap in [false, true] {
        let temp = tempfile::tempdir().unwrap();
        let mut first = item(
            "i1",
            "source.txt",
            "file",
            "page.llm.md",
            if overlap {
                "page.llm.md"
            } else {
                "different.md"
            },
        );
        first["retryable"] = json!(true);
        let (items, bases) = if overlap {
            (
                vec![
                    first,
                    item("i2", "literal.llm", "file", "page.llm.md", "page.llm.md"),
                ],
                json!({"i2":"page.llm"}),
            )
        } else {
            (vec![first], Value::Null)
        };
        let folder = history(temp.path(), items, bases);
        std::fs::write(folder.join("uploads/source.txt"), "Original source.").unwrap();
        std::fs::write(folder.join("out/page.md"), "Base.").unwrap();
        std::fs::write(folder.join("out/page.llm.md"), "Enhanced.").unwrap();
        let before = inventory(&folder);
        let server = Server::start(temp.path());
        assert_eq!(retry(&server, "i1").status, 409);
        assert_eq!(
            server
                .request("DELETE", &format!("/api/jobs/{ID}/items/i1"), &[], &[])
                .status,
            409
        );
        if !overlap {
            assert_eq!(
                server
                    .request("GET", &format!("/api/jobs/{ID}/items/i1/result"), &[], &[])
                    .status,
                409
            );
        }
        let download = server.request(
            "GET",
            &format!("/api/jobs/{ID}/files/page.llm.md"),
            &[],
            &[],
        );
        assert_eq!(download.status, 200);
        assert_eq!(download.body, b"Enhanced.");
        server.stop();
        assert_eq!(inventory(&folder), before);
    }
}

#[test]
fn legacy_failed_retry_preserves_previous_result_bytes_and_item_fields() {
    let temp = tempfile::tempdir().unwrap();
    let mut saved = item(
        "i1",
        "bad.ipynb",
        "file",
        "notebook.llm.md",
        "notebook.llm.md",
    );
    saved["retryable"] = json!(true);
    let folder = history(temp.path(), vec![saved], Value::Null);
    std::fs::write(folder.join("uploads/bad.ipynb"), "{malformed notebook").unwrap();
    std::fs::write(folder.join("out/notebook.md"), "Previous base.\n").unwrap();
    std::fs::write(
        folder.join("out/notebook.llm.md"),
        "Previous enhanced. ![image](assets/shared.png)\n",
    )
    .unwrap();
    std::fs::write(folder.join("out/assets/shared.png"), b"asset bytes").unwrap();
    let bytes = inventory(&folder.join("out"));
    let server = Server::start(temp.path());
    let previous = result(&server, "i1");
    let item_before = server.json(&format!("/api/jobs/{ID}"))["items"][0].clone();
    assert_eq!(retry(&server, "i1").status, 202);
    let done = server.done(ID);
    let retained = done["items"][0]["rerun_failure"].clone();
    assert_eq!(retained["operation"], "retry");
    assert_eq!(retained["error_code"], "invalid_json");
    assert!(
        retained["error"]
            .as_str()
            .unwrap()
            .contains("key must be a string")
    );
    chrono::DateTime::parse_from_rfc3339(retained["failed_at"].as_str().unwrap()).unwrap();
    let mut restored = done["items"][0].clone();
    restored.as_object_mut().unwrap().remove("rerun_failure");
    assert_eq!(restored, item_before);
    assert_eq!(result(&server, "i1"), previous);
    assert_eq!(inventory(&folder.join("out")), bytes);
    server.stop();
    let server = Server::start(temp.path());
    assert_eq!(
        server.json(&format!("/api/jobs/{ID}"))["items"][0],
        done["items"][0]
    );
    assert_eq!(result(&server, "i1"), previous);
    assert_eq!(inventory(&folder.join("out")), bytes);
    server.stop();
}

#[test]
fn legacy_delete_removes_the_correct_pair_and_retains_sibling_shared_assets() {
    let temp = tempfile::tempdir().unwrap();
    let mut second = item("i2", "other.txt", "file", "other.md", "other.md");
    second["llm_enhanced"] = json!(false);
    let folder = history(
        temp.path(),
        vec![
            item("i1", "old.txt", "file", "page.llm.md", "page.llm.md"),
            second,
        ],
        Value::Null,
    );
    for (name, body) in [
        ("page.md", "Old base. ![shared](assets/shared.png)\n"),
        ("page.llm.md", "Old enhanced. ![private](assets/only.png)\n"),
        ("other.md", "Sibling. ![shared](assets/shared.png)\n"),
        ("assets/shared.png", "shared"),
        ("assets/only.png", "only old item"),
    ] {
        std::fs::write(folder.join("out").join(name), body).unwrap();
    }
    let server = Server::start(temp.path());
    assert_eq!(
        server
            .request("DELETE", &format!("/api/jobs/{ID}/items/i1"), &[], &[])
            .status,
        204
    );
    for name in ["page.md", "page.llm.md", "assets/only.png"] {
        assert!(!folder.join("out").join(name).exists(), "{name}");
    }
    assert_eq!(
        std::fs::read(folder.join("out/assets/shared.png")).unwrap(),
        b"shared"
    );
    assert_eq!(
        result(&server, "i2")["markdown"],
        "Sibling. ![shared](assets/shared.png)\n"
    );
    server.stop();
    let server = Server::start(temp.path());
    assert_eq!(
        server.json(&format!("/api/jobs/{ID}"))["items"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    assert_eq!(
        server
            .request("DELETE", &format!("/api/jobs/{ID}/items/i2"), &[], &[])
            .status,
        204
    );
    assert!(!folder.exists());
    server.stop();
}
