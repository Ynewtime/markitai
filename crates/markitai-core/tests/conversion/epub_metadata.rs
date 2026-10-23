//! An EPUB's own metadata goes to the front matter, not into the body, with
//! the keys and values Python markitai 1.3.0 writes. The real books are in
//! `tests/fixtures/epub` (see its `provenance.json`).

use super::*;

const GUTENBERG: &[u8] = include_bytes!("../fixtures/epub/zh-gutenberg-23825.epub");
const MARKITDOWN: &[u8] = include_bytes!("../fixtures/epub/markitdown-test.epub");

fn written(name: &str, bytes: &[u8]) -> String {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join(name);
    std::fs::write(&source, bytes).unwrap();
    let out = dir.path().join("out");
    let result = convert(
        source.to_str().unwrap(),
        ConvertOptions {
            output_dir: Some(out.clone()),
            ..options()
        },
    )
    .unwrap();
    std::fs::read_to_string(result.output_path.unwrap()).unwrap()
}

/// The front matter without its `markitai_processed` time, and the body.
fn parts(text: &str) -> (String, String) {
    let rest = text.strip_prefix("---\n").unwrap();
    let (yaml, body) = rest.split_once("\n---\n\n").unwrap();
    let yaml = yaml
        .lines()
        .filter(|line| !line.starts_with("markitai_processed:"))
        .collect::<Vec<_>>()
        .join("\n");
    (yaml, body.to_owned())
}

#[test]
fn a_chinese_books_metadata_is_front_matter() {
    let (yaml, body) = parts(&written("zh-gutenberg-23825.epub", GUTENBERG));
    assert_eq!(
        yaml,
        "title: 施公案\nsource: zh-gutenberg-23825.epub\nauthors:\n- Anonymous\nlanguage: zh\ndate: '2007-12-11'"
    );
    // The body starts with the book; its identifier is nowhere.
    assert!(
        body.starts_with("## The Project Gutenberg eBook of 施公案\n"),
        "{body}"
    );
    assert!(!body.contains("**Title:**") && !body.contains("**Authors:**"));
    assert!(!body.contains("**Identifier:**"));
    assert!(body.contains("話說江都縣有一秀才"));
}

#[test]
fn a_books_description_is_front_matter_too() {
    let (yaml, body) = parts(&written("markitdown-test.epub", MARKITDOWN));
    assert_eq!(
        yaml,
        "title: Test EPUB Document\nsource: markitdown-test.epub\nauthors:\n- Test Author\nlanguage: en\n\
         description: A test EPUB document for MarkItDown testing"
    );
    assert!(!body.contains("**Description:**") && !body.contains("**Language:**"));
}

#[test]
fn the_enhanced_output_keeps_the_books_facts_with_the_models_summary() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("zh-gutenberg-23825.epub");
    std::fs::write(&source, GUTENBERG).unwrap();
    let (base, server) = llm_server(
        200,
        r#"{"choices":[{"message":{"content":"{protected_input}"}}],"usage":{"prompt_tokens":5,"completion_tokens":5}}"#,
    );
    let cfg = json!({"cache":{"enabled":false},"llm":{"enabled":true,"keep_base":true,"model_list":[
        {"model_name":"test","litellm_params":{"model":"openai/test-model","api_base":base,"api_key":"test-key"}}]}});
    let result = convert(
        source.to_str().unwrap(),
        ConvertOptions {
            config: Some(cfg),
            output_dir: Some(dir.path().join("out")),
            ..Default::default()
        },
    )
    .unwrap();
    server.join().unwrap();
    let enhanced = std::fs::read_to_string(result.llm_output_path.unwrap()).unwrap();
    let (yaml, body) = parts(&enhanced);
    assert_eq!(
        yaml,
        "title: 施公案\nsource: zh-gutenberg-23825.epub\nauthors:\n- Anonymous\nlanguage: zh\ndate: '2007-12-11'\n\
         description: Mock document description\ntags:\n- mock"
    );
    assert!(!body.contains("**Title:**"));
}
