use serde_json::{Value, json};
use std::path::Path;
use std::process::{Command, Output};

fn command(root: &Path, cfg: Value) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_markitai"));
    command.env_clear();
    for key in ["HOME", "SYSTEMROOT", "USERPROFILE", "TMPDIR"] {
        if let Some(value) = std::env::var_os(key) {
            command.env(key, value);
        }
    }
    command
        .current_dir(root)
        .env("MARKITAI_HOME", root.join("home"))
        .env("MARKITAI_BROWSER_EXECUTABLE", root.join("absent-browser"))
        .env("PLAYWRIGHT_BROWSERS_PATH", root.join("browser-cache"));
    let bin = root.join("bin");
    std::fs::create_dir_all(&bin).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let path = bin.join("soffice");
        std::fs::write(&path, "#!/bin/sh\nprintf 'LibreOffice 26.2.0.1\\n'\n").unwrap();
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700)).unwrap();
    }
    command
        .env("PATH", bin)
        .arg("--config-json")
        .arg(cfg.to_string())
        .arg("doctor");
    command
}
fn decode(output: &Output) -> Value {
    serde_json::from_slice(&output.stdout).unwrap_or_else(|error| {
        panic!(
            "{error}: {} / {}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )
    })
}

#[test]
fn optional_missing_diagnostics_use_reference_checks_without_a_capabilities_envelope() {
    let dir = tempfile::tempdir().unwrap();
    let output = command(dir.path(), json!({}))
        .arg("--json")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value = decode(&output);
    assert_eq!(value["playwright"]["status"], "missing");
    assert_eq!(value["llm-api"]["status"], "missing");
    assert_eq!(value["rapidocr"]["optional"], true);
    assert_eq!(value["anydoc"]["optional"], true);
    assert!(value.get("capabilities").is_none());
    assert!(value.get("version").is_none());
    let text = String::from_utf8(output.stdout).unwrap();
    let mut previous = 0;
    for key in [
        "playwright",
        "libreoffice",
        "rapidocr",
        "anydoc",
        "serve",
        "llm-api",
        "vision-model",
        "vlm-ocr",
    ] {
        let index = text.find(&format!("\"{key}\":")).unwrap();
        assert!(index >= previous);
        previous = index;
        for field in ["name", "description", "status", "message", "install_hint"] {
            assert!(value[key][field].is_string(), "{key}.{field}");
        }
    }
    assert!(!dir.path().join("home").exists());
}

#[test]
fn missing_active_inline_and_linked_env_references_fail_but_disabled_models_do_not() {
    let dir = tempfile::tempdir().unwrap();
    for linked in [false, true] {
        let mut cfg = json!({"llm":{"model_list":[{"model_name":"default","litellm_params":{"model":"openai/test"}}]}});
        if linked {
            cfg["llm"]["providers"] = json!([{"id":"p","provider":"openai","api_key":"env:ROUND23_MISSING_KEY","api_base":"env:ROUND23_MISSING_BASE"}]);
            cfg["llm"]["model_list"][0]["model_info"] = json!({"provider_id":"p"});
        } else {
            cfg["llm"]["model_list"][0]["litellm_params"]["api_key"] =
                json!("env:ROUND23_MISSING_KEY");
        }
        let output = command(dir.path(), cfg.clone())
            .env("IRRELEVANT_SECRET", "must-not-print-secret")
            .arg("--json")
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(1));
        assert_eq!(decode(&output)["llm-api"]["status"], "error");
        let text = String::from_utf8_lossy(&output.stdout);
        assert!(text.contains("ROUND23_MISSING_KEY"));
        assert!(!text.contains("must-not-print-secret"));
        cfg["llm"]["model_list"][0]["litellm_params"]["weight"] = json!(0);
        let output = command(dir.path(), cfg).arg("--json").output().unwrap();
        assert!(output.status.success());
        assert_eq!(decode(&output)["llm-api"]["status"], "warning");
    }
}

#[test]
fn configured_browser_and_subscription_provider_fail_without_attempting_repairs() {
    let dir = tempfile::tempdir().unwrap();
    let output = command(dir.path(), json!({"fetch":{"strategy":"playwright"}}))
        .arg("--json")
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(decode(&output)["playwright"]["status"], "missing");
    let output = command(dir.path(), json!({"llm":{"model_list":[{"model_name":"default","litellm_params":{"model":"copilot/example"}}]}})).arg("--json").output().unwrap();
    assert_eq!(output.status.code(), Some(1));
    let value = decode(&output);
    assert_eq!(value["copilot-sdk"]["status"], "error");
    assert_eq!(value["copilot-auth"]["status"], "error");
    let output = command(dir.path(), json!({}))
        .arg("--fix")
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&output.stderr).contains("no installer was started"));
    let output = command(dir.path(), json!({}))
        .args(["--fix", "--json"])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    assert!(output.stdout.is_empty());
    let output = command(dir.path(), json!({}))
        .arg("--suggest-extras")
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&output.stderr).contains("Python extras"));
}

#[cfg(unix)]
#[test]
fn executable_presence_is_not_browser_readiness_and_child_secrets_stay_private() {
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().unwrap();
    let browser = dir.path().join("fake-browser");
    std::fs::write(
        &browser,
        "#!/bin/sh\nprintf 'child-sensitive-token' >&2\nexit 9\n",
    )
    .unwrap();
    std::fs::set_permissions(&browser, std::fs::Permissions::from_mode(0o700)).unwrap();
    let output = command(dir.path(), json!({"screenshot":{"enabled":true}}))
        .env("MARKITAI_BROWSER_EXECUTABLE", browser)
        .arg("--json")
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(decode(&output)["playwright"]["status"], "warning");
    assert!(!String::from_utf8_lossy(&output.stdout).contains("child-sensitive-token"));
    assert!(!String::from_utf8_lossy(&output.stderr).contains("child-sensitive-token"));
}

#[test]
fn vlm_optout_is_reported_without_contacting_the_configured_endpoint() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = json!({"llm":{"model_list":[{"model_name":"default","litellm_params":{"model":"openai/test","api_key":"private-test-key","api_base":"http://127.0.0.1:1"},"model_info":{"supports_vision":true}}]}});
    let output = command(dir.path(), cfg)
        .env("MARKITAI_NO_VLM_OCR", "yes")
        .arg("--json")
        .output()
        .unwrap();
    assert!(output.status.success());
    let value = decode(&output);
    assert_eq!(value["vision-model"]["status"], "ok");
    assert_eq!(value["vision-model"]["models"], json!(["openai/test"]));
    assert_eq!(value["vlm-ocr"]["status"], "warning");
    assert!(
        value["vlm-ocr"]["message"]
            .as_str()
            .unwrap()
            .contains("MARKITAI_NO_VLM_OCR")
    );
    assert!(!String::from_utf8_lossy(&output.stdout).contains("private-test-key"));
}

#[test]
fn environment_models_and_text_summary_match_what_a_conversion_uses() {
    let dir = tempfile::tempdir().unwrap();
    // Without llm.model_list, --llm uses MODEL or detected provider keys.
    let output = command(dir.path(), json!({}))
        .env("MODEL", "openai/env-model")
        .env("OPENAI_API_KEY", "doctor-fixture-key")
        .arg("--json")
        .output()
        .unwrap();
    assert!(output.status.success());
    let value = decode(&output);
    assert_eq!(value["llm-api"]["status"], "ok");
    let message = value["llm-api"]["message"].as_str().unwrap();
    assert!(message.contains("openai/env-model"), "{message}");
    assert!(!String::from_utf8_lossy(&output.stdout).contains("doctor-fixture-key"));
    let output = command(dir.path(), json!({}))
        .env("MODEL", "openai/env-model")
        .arg("--json")
        .output()
        .unwrap();
    assert!(output.status.success());
    assert_eq!(decode(&output)["llm-api"]["status"], "warning");

    // An explicit executable is what failed, and --fix will not replace it.
    let value = decode(
        &command(dir.path(), json!({}))
            .arg("--json")
            .output()
            .unwrap(),
    );
    assert_eq!(value["playwright"]["status"], "missing");
    assert!(
        value["playwright"]["message"]
            .as_str()
            .unwrap()
            .contains("MARKITAI_BROWSER_EXECUTABLE does not name an executable file")
    );
    let hint = value["playwright"]["install_hint"].as_str().unwrap();
    assert!(hint.contains("doctor --fix will not replace it"), "{hint}");

    let output = command(dir.path(), json!({})).output().unwrap();
    assert!(output.status.success());
    let text = String::from_utf8(output.stdout).unwrap();
    assert!(
        text.lines()
            .nth(1)
            .is_some_and(|line| line.starts_with("Configuration: built-in defaults")),
        "{text}"
    );
    assert!(
        text.trim_end().lines().last().is_some_and(
            |line| line.starts_with("Summary: nothing the configuration requires is blocked")
        ),
        "{text}"
    );
    let output = command(dir.path(), json!({"fetch":{"strategy":"playwright"}}))
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1));
    let text = String::from_utf8(output.stdout).unwrap();
    assert!(
        text.contains(
            "Summary: 1 check required by the configuration is not ready: Chromium (native CDP)."
        ),
        "{text}"
    );
}
