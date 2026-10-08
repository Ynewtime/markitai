use markitai_core::output::Publication;
use markitai_core::{ConvertContext, ConvertOptions, Error, convert_with_publication};
use serde_json::json;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

struct Skip {
    calls: AtomicUsize,
}
impl Publication for Skip {
    fn skip_existing(&self) -> bool {
        true
    }
    fn publish(&self, _path: &Path, _bytes: &[u8]) -> markitai_core::Result<()> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Err(Error::Conversion(
            "a skipped publication must not be attempted".into(),
        ))
    }
}

#[test]
fn callback_skip_avoids_document_parsing_model_work_and_output_side_effects() {
    let root = tempfile::tempdir().unwrap();
    let source = root.path().join("broken.ipynb");
    std::fs::write(&source, "{invalid notebook JSON").unwrap();
    let destination = root.path().join("output");
    let publication = Skip {
        calls: AtomicUsize::new(0),
    };
    // The ordinary conflict policy would convert this nonexistent destination.
    // Parsing this source or reaching the unconfigured model would return an error.
    let result = convert_with_publication(
        source.to_str().unwrap(),
        ConvertOptions {
            output_dir: Some(destination.clone()),
            config: Some(json!({
                "output":{"on_conflict":"overwrite"},
                "llm":{"enabled":true}, "cache":{"enabled":false},
                "prompts":{"dir":root.path().join("private-prompts")}
            })),
            ..Default::default()
        },
        ConvertContext::default(),
        Some(&publication),
    )
    .unwrap();
    assert_eq!(result.skip_reason.as_deref(), Some("exists"));
    assert_eq!(result.source, source.to_str().unwrap());
    assert!(result.markdown.is_empty());
    assert!(result.llm_markdown.is_none());
    assert!(result.output_path.is_none() && result.llm_output_path.is_none());
    assert!(result.assets.is_empty() && result.screenshots.is_empty());
    assert_eq!(result.usage.requests, 0);
    assert_eq!(publication.calls.load(Ordering::SeqCst), 0);
    assert!(!destination.exists());
    assert!(!root.path().join(".markitai").exists());
}
