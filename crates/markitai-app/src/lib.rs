//! The native apps' private core (desktop/DESIGN.md, section 3): conversion
//! through the bindings' JSON contract, the Markdown preview, and what the
//! apps need around them. Not a public binding. Every call takes and answers
//! JSON, `{"ok": true, "result"}` or `{"ok": false, "error": {"code",
//! "message"}}`, the same from Rust and over the C ABI
//! (`include/markitai_app.h`).

mod body;
mod environment;
mod folder;
mod preview;

use std::ffi::c_char;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::ptr;

use serde_json::{Value, json};

pub use preview::{PREVIEW_CSS, Theme, file_url, render as render_preview};

pub const ABI_VERSION: u32 = 2;
const MAX_REQUEST_BYTES: usize = 64 * 1024 * 1024;

/// The markitai version.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// `{"data_dir"}`: create it, make it the working directory and, on macOS and
/// Linux, import the login shell's environment. Answers
/// `{"login_environment": "imported" | "inherited" | "failed", "detail"}`.
///
/// # Safety
/// Sets environment variables: call it first in `main`, before the process
/// starts any other thread.
pub unsafe fn prepare_environment(request: &str) -> String {
    answer(|| {
        let request = parse(request)?;
        let data_dir = string(&request, "data_dir")?;
        let login = unsafe { environment::prepare(Path::new(data_dir)) }
            .map_err(|e| failure("output_unavailable", &format!("{data_dir}: {e}")))?;
        Ok(match login {
            environment::Login::Imported => {
                json!({"login_environment": "imported", "detail": null})
            }
            environment::Login::Inherited => {
                json!({"login_environment": "inherited", "detail": null})
            }
            environment::Login::Failed(why) => {
                json!({"login_environment": "failed", "detail": why})
            }
        })
    })
}

/// Convert as `markitai_convert_json` does (docs/bindings.md, "Shared
/// contract"), except for an image with no text to extract (neither AI nor
/// OCR on): it fails with the code `image_only`, which the CLI reports as that
/// skip reason, rather than `conversion_error`, and a message without the
/// Python binding's argument names. Blocks until the conversion ends.
pub fn convert_json(request: &str) -> String {
    let image_only = std::cell::Cell::new(false);
    let response = catch_unwind(AssertUnwindSafe(|| {
        markitai_core::convert_json_as(request, |failure| match failure.error {
            markitai_core::Error::ImageOnly(_) => {
                image_only.set(true);
                "image_only"
            }
            _ => failure.code(),
        })
    }))
    .unwrap_or_else(|_| markitai_core::internal_error_json());
    if !image_only.get() {
        return response;
    }
    let name = serde_json::from_str::<Value>(request)
        .ok()
        .and_then(|r| r.get("source")?.as_str().map(str::to_owned))
        .map(|source| {
            Path::new(&source)
                .file_name()
                .map_or(source.clone(), |n| n.to_string_lossy().into_owned())
        })
        .unwrap_or_default();
    let message = format!("{name} is an image file with no text to extract.");
    json!({"ok": false, "error": {"code": "image_only", "message": message.trim_start()}})
        .to_string()
}

/// `{"markdown", "theme": "light" | "dark", "base_dir", "files_url"}`:
/// `{"html"}`. `files_url`, optional, is where the page reads `base_dir`'s
/// files, for an app that serves them itself.
pub fn preview_html(request: &str) -> String {
    answer(|| {
        let request = parse(request)?;
        let markdown = string(&request, "markdown")?;
        let theme = match request.get("theme").and_then(Value::as_str) {
            Some("light") => Theme::Light,
            Some("dark") => Theme::Dark,
            _ => return Err(invalid("\"theme\" must be \"light\" or \"dark\"")),
        };
        let base_dir = match request.get("base_dir") {
            None | Some(Value::Null) => None,
            Some(Value::String(dir)) => Some(PathBuf::from(dir)),
            Some(_) => return Err(invalid("\"base_dir\" must be a string or null")),
        };
        let files_url = match request.get("files_url") {
            None | Some(Value::Null) => None,
            Some(Value::String(url)) if preview::url_scheme(url).is_some() => Some(url.as_str()),
            Some(_) => return Err(invalid("\"files_url\" must be a URL or null")),
        };
        Ok(json!({
            "html": preview::render(markdown, theme, base_dir.as_deref(), files_url)
        }))
    })
}

/// `{"markdown"}`: the text of a Markdown file the core wrote, `{"markdown"}`
/// without its front matter, as the conversion's `markdown` field had it.
pub fn markdown_body(request: &str) -> String {
    answer(|| {
        let request = parse(request)?;
        let markdown = string(&request, "markdown")?;
        Ok(json!({"markdown": body::body(markdown)}))
    })
}

/// `{"parent", "source"}`: create the conversion's own folder, `{"path", "name"}`.
pub fn claim_output_folder(request: &str) -> String {
    answer(|| {
        let request = parse(request)?;
        let parent = string(&request, "parent")?;
        let source = string(&request, "source")?;
        let (path, name) = folder::claim(Path::new(parent), source)
            .map_err(|e| failure("output_unavailable", &e.to_string()))?;
        Ok(json!({"path": path, "name": name}))
    })
}

/// The markitai configuration file the core reads when `config` is omitted,
/// or where the user's own would be: `{"path", "exists"}`.
pub fn config_file() -> String {
    answer(|| {
        let path = markitai_core::config::selected_path(None)
            .unwrap_or_else(|| markitai_core::config::home().join("config.json"));
        Ok(json!({"exists": path.is_file(), "path": path}))
    })
}

type Failure = Value;

fn answer(call: impl FnOnce() -> Result<Value, Failure>) -> String {
    match catch_unwind(AssertUnwindSafe(call)) {
        Ok(Ok(result)) => json!({"ok": true, "result": result}).to_string(),
        Ok(Err(error)) => json!({"ok": false, "error": error}).to_string(),
        Err(_) => markitai_core::internal_error_json(),
    }
}

fn failure(code: &str, message: &str) -> Failure {
    json!({"code": code, "message": message})
}

fn invalid(message: &str) -> Failure {
    failure("invalid_input", message)
}

fn parse(request: &str) -> Result<Value, Failure> {
    let value: Value =
        serde_json::from_str(request).map_err(|e| invalid(&format!("Request is not JSON: {e}")))?;
    if value.is_object() {
        Ok(value)
    } else {
        Err(invalid("Request must be a JSON object"))
    }
}

fn string<'a>(request: &'a Value, key: &str) -> Result<&'a str, Failure> {
    request
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| invalid(&format!("Request needs a string \"{key}\"")))
}

#[repr(C)]
pub struct MarkitaiAppBuffer {
    pub data: *mut u8,
    pub len: usize,
}

impl MarkitaiAppBuffer {
    fn from_string(value: String) -> Self {
        let bytes = value.into_bytes().into_boxed_slice();
        let len = bytes.len();
        Self {
            data: Box::into_raw(bytes).cast(),
            len,
        }
    }
}

/// Answer `call` with the request's text, or the envelope saying why there is
/// none.
///
/// # Safety
/// `request` points to `len` readable bytes for the duration of the call.
unsafe fn with_request(
    request: *const u8,
    len: usize,
    call: impl FnOnce(&str) -> String,
) -> MarkitaiAppBuffer {
    let refused = |message: &str| json!({"ok": false, "error": invalid(message)}).to_string();
    let response = if len > MAX_REQUEST_BYTES {
        refused("JSON request exceeds the 64 MiB ABI limit")
    } else if request.is_null() && len != 0 {
        refused("Request pointer is null with nonzero length")
    } else {
        let bytes = if len == 0 {
            &[][..]
        } else {
            unsafe { std::slice::from_raw_parts(request, len) }
        };
        match std::str::from_utf8(bytes) {
            Ok(text) => call(text),
            Err(_) => refused("JSON request is not valid UTF-8"),
        }
    };
    MarkitaiAppBuffer::from_string(response)
}

#[unsafe(no_mangle)]
pub extern "C" fn markitai_app_abi_version() -> u32 {
    ABI_VERSION
}

#[unsafe(no_mangle)]
pub extern "C" fn markitai_app_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr().cast()
}

/// # Safety
/// As [`prepare_environment`], and `request` points to `len` readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn markitai_app_prepare_environment(
    request: *const u8,
    len: usize,
) -> MarkitaiAppBuffer {
    unsafe { with_request(request, len, |r| prepare_environment(r)) }
}

/// # Safety
/// `request` points to `len` readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn markitai_app_convert_json(
    request: *const u8,
    len: usize,
) -> MarkitaiAppBuffer {
    unsafe { with_request(request, len, convert_json) }
}

/// # Safety
/// `request` points to `len` readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn markitai_app_preview_html(
    request: *const u8,
    len: usize,
) -> MarkitaiAppBuffer {
    unsafe { with_request(request, len, preview_html) }
}

/// # Safety
/// `request` points to `len` readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn markitai_app_markdown_body(
    request: *const u8,
    len: usize,
) -> MarkitaiAppBuffer {
    unsafe { with_request(request, len, markdown_body) }
}

/// # Safety
/// `request` points to `len` readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn markitai_app_claim_output_folder(
    request: *const u8,
    len: usize,
) -> MarkitaiAppBuffer {
    unsafe { with_request(request, len, claim_output_folder) }
}

#[unsafe(no_mangle)]
pub extern "C" fn markitai_app_config_file() -> MarkitaiAppBuffer {
    MarkitaiAppBuffer::from_string(config_file())
}

/// Release a response and clear it; null or a cleared buffer is harmless.
///
/// # Safety
/// `buffer` is null or points to a buffer one of these functions returned,
/// its pointer and length unchanged.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn markitai_app_buffer_free(buffer: *mut MarkitaiAppBuffer) {
    if buffer.is_null() {
        return;
    }
    let buffer = unsafe { &mut *buffer };
    if !buffer.data.is_null() {
        let allocation = ptr::slice_from_raw_parts_mut(buffer.data, buffer.len);
        buffer.data = ptr::null_mut();
        buffer.len = 0;
        unsafe { drop(Box::from_raw(allocation)) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    unsafe fn take(mut buffer: MarkitaiAppBuffer) -> Value {
        let bytes = unsafe { std::slice::from_raw_parts(buffer.data, buffer.len) };
        let value = serde_json::from_slice(bytes).unwrap();
        unsafe { markitai_app_buffer_free(&mut buffer) };
        assert!(buffer.data.is_null());
        unsafe { markitai_app_buffer_free(&mut buffer) };
        value
    }

    fn call(
        f: unsafe extern "C" fn(*const u8, usize) -> MarkitaiAppBuffer,
        request: &str,
    ) -> Value {
        unsafe { take(f(request.as_ptr(), request.len())) }
    }

    #[test]
    fn every_call_answers_an_envelope() {
        let preview = call(
            markitai_app_preview_html,
            r##"{"markdown":"# Hi","theme":"dark","base_dir":null}"##,
        );
        assert_eq!(preview["ok"], true, "{preview}");
        assert!(
            preview["result"]["html"]
                .as_str()
                .unwrap()
                .contains("<h1>Hi</h1>")
        );
        for bad in [
            r#"{"theme":"dark"}"#,
            r#"{"markdown":"x","theme":"sepia"}"#,
            r#"{"markdown":"x"}"#,
            "[]",
            "nope",
        ] {
            assert_eq!(
                call(markitai_app_preview_html, bad)["error"]["code"],
                "invalid_input",
                "{bad}"
            );
        }
        assert_eq!(
            call(markitai_app_claim_output_folder, "{}")["error"]["code"],
            "invalid_input"
        );
        for bad in [r#"{}"#, r#"{"markdown":1}"#, "[]"] {
            assert_eq!(
                call(markitai_app_markdown_body, bad)["error"]["code"],
                "invalid_input",
                "{bad}"
            );
        }
        let refused = unsafe { take(markitai_app_convert_json(ptr::null(), 1)) };
        assert_eq!(refused["error"]["code"], "invalid_input");
        let config = unsafe { take(markitai_app_config_file()) };
        assert!(
            config["result"]["path"]
                .as_str()
                .unwrap()
                .ends_with(".json"),
            "{config}"
        );
        unsafe { markitai_app_buffer_free(ptr::null_mut()) };
        assert_eq!(markitai_app_abi_version(), ABI_VERSION);
        let printed = unsafe { std::ffi::CStr::from_ptr(markitai_app_version()) };
        assert_eq!(printed.to_str().unwrap(), version());
    }

    #[test]
    fn a_conversion_writes_into_its_claimed_folder() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("报告.md");
        std::fs::write(&source, "# Native\n\n你好\n").unwrap();
        let parent = dir.path().join("Markitai");
        let claimed = call(
            markitai_app_claim_output_folder,
            &json!({"parent": parent, "source": source}).to_string(),
        );
        assert_eq!(claimed["result"]["name"], "报告", "{claimed}");
        let folder = claimed["result"]["path"].as_str().unwrap();
        let request = json!({"source": source, "options": {"config": {}, "llm": false, "output_dir": folder}});
        let response = call(markitai_app_convert_json, &request.to_string());
        assert_eq!(response["ok"], true, "{response}");
        assert!(
            response["result"]["markdown"]
                .as_str()
                .unwrap()
                .contains("你好")
        );
        let written = response["result"]["output_path"].as_str().unwrap();
        assert!(Path::new(written).starts_with(folder), "{written}");
    }

    fn body_of(text: &str) -> String {
        let answer = call(
            markitai_app_markdown_body,
            &json!({ "markdown": text }).to_string(),
        );
        assert_eq!(answer["ok"], true, "{answer}");
        answer["result"]["markdown"].as_str().unwrap().to_owned()
    }

    #[test]
    fn a_written_file_without_its_front_matter_is_the_conversions_markdown() {
        let dir = tempfile::tempdir().unwrap();
        for (name, text) in [
            ("plain.md", "# Plain\n\nBody.\n"),
            (
                "own.md",
                "---\ntitle: Existing\ncustom: kept\n---\n\n# Heading\n",
            ),
            (
                "rules.md",
                "---\n\nIntro after a rule.\n\n---\n\nSecond part.\n",
            ),
        ] {
            let source = dir.path().join(name);
            std::fs::write(&source, text).unwrap();
            let request = json!({"source": source, "options": {"config": {}, "llm": false, "output_dir": dir.path().join(name.replace(".md", ""))}});
            let response = call(markitai_app_convert_json, &request.to_string());
            assert_eq!(response["ok"], true, "{response}");
            let written =
                std::fs::read_to_string(response["result"]["output_path"].as_str().unwrap())
                    .unwrap();
            assert!(written.starts_with("---\ntitle: "), "{name}: {written}");
            assert_eq!(
                body_of(&written),
                response["result"]["markdown"].as_str().unwrap(),
                "{name}"
            );
        }
    }

    #[test]
    fn only_a_leading_yaml_mapping_is_front_matter() {
        for (text, body) in [
            ("", ""),
            (
                "# No front matter\n\nBody.\n",
                "# No front matter\n\nBody.\n",
            ),
            ("---\ntitle: Only\n---\n", ""),
            ("---\ntitle: x\n---", ""),
            ("---\ntitle: x\n---\n\n\n# Body\n", "# Body\n"),
            (
                "---\n\nIntro after a rule.\n\n---\n\nSecond part.\n",
                "---\n\nIntro after a rule.\n\n---\n\nSecond part.\n",
            ),
            ("---\n- a\n- b\n---\nbody\n", "---\n- a\n- b\n---\nbody\n"),
            (
                "---\ninvalid: [\n---\nbody\n",
                "---\ninvalid: [\n---\nbody\n",
            ),
            ("---\n---\nbody\n", "---\n---\nbody\n"),
            (
                "---\ntitle: never closed\n\n# Body\n",
                "---\ntitle: never closed\n\n# Body\n",
            ),
            ("# Title\n\n---\na: 1\n---\n", "# Title\n\n---\na: 1\n---\n"),
            (
                "---\r\ntitle: CRLF\r\n---\r\n\r\n# CRLF body\r\n",
                "# CRLF body\r\n",
            ),
            ("---\r\ntitle: 中文\r\n---", ""),
            (
                "---\r\n\r\nA rule.\r\n\r\n---\r\n\r\nNext.\r\n",
                "---\r\n\r\nA rule.\r\n\r\n---\r\n\r\nNext.\r\n",
            ),
            (
                "\u{feff}---\ntitle: BOM\n---\n\n# BOM body\n",
                "# BOM body\n",
            ),
            ("\u{feff}# Plain\n", "# Plain\n"),
            (
                "\u{feff}---\r\ntitle: Both\r\n---\r\n\r\nBody\r\n",
                "Body\r\n",
            ),
        ] {
            assert_eq!(body_of(text), body, "{text:?}");
        }
    }

    #[test]
    fn an_image_without_text_is_image_only() {
        let dir = tempfile::tempdir().unwrap();
        let image = dir.path().join("diagram.png");
        std::fs::copy(
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../markitai-core/src/formats/native/fixtures/office-images/diagram.png"
            ),
            &image,
        )
        .unwrap();
        let request =
            json!({"source": image, "options": {"config": {}, "llm": false, "ocr": false}});
        let response = call(markitai_app_convert_json, &request.to_string());
        assert_eq!(response["error"]["code"], "image_only", "{response}");
        assert_eq!(
            response["error"]["message"],
            "diagram.png is an image file with no text to extract."
        );
        let missing = json!({"source": dir.path().join("gone.png"), "options": {"config": {}}});
        let response = call(markitai_app_convert_json, &missing.to_string());
        assert_eq!(response["error"]["code"], "not_found", "{response}");
    }

    #[test]
    fn an_unwritable_parent_is_output_unavailable() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("file");
        std::fs::write(&file, "x").unwrap();
        let claimed = call(
            markitai_app_claim_output_folder,
            &json!({"parent": file.join("below"), "source": "/a/b.pdf"}).to_string(),
        );
        assert_eq!(claimed["error"]["code"], "output_unavailable", "{claimed}");
    }
}
