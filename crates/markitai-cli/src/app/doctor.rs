//! Configuration-aware diagnostics; explicit browser repair uses the native installer.
use super::{CliResult, runtime};
use indexmap::IndexMap;
use markitai_core::config;
use serde::Serialize;
use serde_json::Value;
use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};

#[derive(Serialize)]
struct Check {
    name: &'static str,
    description: &'static str,
    status: &'static str,
    message: String,
    install_hint: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    optional: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    path: Option<PathBuf>,
    #[serde(skip_serializing_if = "Option::is_none")]
    models: Option<Vec<String>>,
    #[serde(skip)]
    required: bool,
}
impl Check {
    fn new(
        name: &'static str,
        description: &'static str,
        status: &'static str,
        message: impl Into<String>,
        hint: impl Into<String>,
    ) -> Self {
        Self {
            name,
            description,
            status,
            message: message.into(),
            install_hint: hint.into(),
            optional: None,
            path: None,
            models: None,
            required: false,
        }
    }
    fn failed(&self) -> bool {
        self.required && self.status != "ok"
    }
}

pub(super) fn run(cfg: &Value, path: Option<&Path>, json: bool, fix: bool) -> CliResult<i32> {
    let (browser, office) = std::thread::scope(|scope| {
        let browser = scope.spawn(markitai_core::browser_diagnostic);
        let office = scope.spawn(markitai_core::office_diagnostic);
        (browser.join(), office.join())
    });
    let mut browser = browser.map_err(|_| runtime("Browser diagnostic did not complete"))?;
    let office = office.map_err(|_| runtime("Office diagnostic did not complete"))?;
    let repair_missing = fix && !matches!(&browser, Ok(Some(_)));
    let mut repair_failed = false;
    if repair_missing {
        eprintln!("Installing the official Chrome headless shell in the private Markitai home...");
        match markitai_core::install_browser() {
            Ok(path) => {
                eprintln!("Installed and verified {}", path.display());
                browser = Ok(Some(path));
            }
            Err(error) => {
                repair_failed = true;
                eprintln!("Browser repair failed: {error}");
            }
        }
    }
    let env = config::environment();
    let checks = checks(
        cfg,
        path,
        &env,
        browser,
        office,
        markitai_core::llm_vision_models(cfg),
    );
    let failed = checks.values().any(Check::failed);
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&checks).map_err(runtime)?
        );
    } else {
        println!("Markitai {} — native diagnostics", markitai_core::VERSION);
        match path {
            Some(path) => println!("Configuration: {}", path.display()),
            None => println!(
                "Configuration: built-in defaults (no configuration file; create one with `markitai init`)"
            ),
        }
        for check in checks.values() {
            println!(
                "{}: {}{} — {}",
                check.name,
                check.status,
                if check.required {
                    " (required by configuration)"
                } else {
                    ""
                },
                check.message
            );
            if !check.install_hint.is_empty() {
                println!("  {}", check.install_hint);
            }
        }
        println!("{}", summary(&checks));
    }
    if fix && !repair_missing {
        eprintln!(
            "Chromium launches successfully; no browser repair is needed. Other diagnostic hints require manual action."
        );
    }
    Ok(i32::from(failed || repair_failed))
}

/// One closing line that states the verdict the exit status encodes.
fn summary(checks: &IndexMap<&'static str, Check>) -> String {
    let failed: Vec<_> = checks
        .values()
        .filter(|check| check.failed())
        .map(|check| check.name)
        .collect();
    let ok = checks.values().filter(|check| check.status == "ok").count();
    let attention = checks.len() - ok;
    if failed.is_empty() && attention == 0 {
        format!("Summary: all {ok} checks ok.")
    } else if failed.is_empty() {
        format!(
            "Summary: nothing the configuration requires is blocked; {ok} ok, {attention} optional {} not ready (see the hints above).",
            if attention == 1 { "check" } else { "checks" }
        )
    } else {
        let (noun, verb) = if failed.len() == 1 {
            ("check", "is")
        } else {
            ("checks", "are")
        };
        format!(
            "Summary: {} {noun} required by the configuration {verb} not ready: {}. Fix the hints above, then rerun `markitai doctor`.",
            failed.len(),
            failed.join(", ")
        )
    }
}

fn checks(
    cfg: &Value,
    path: Option<&Path>,
    env: &HashMap<String, String>,
    browser: markitai_core::Result<Option<PathBuf>>,
    office: markitai_core::Result<Option<PathBuf>>,
    models: Vec<String>,
) -> IndexMap<&'static str, Check> {
    let mut result = IndexMap::new();
    let mut browser = backend(
        "Chromium (native CDP)",
        "Browser automation for dynamic URLs",
        browser,
        "Launch and CDP connection succeeded in a private profile; no remote page was opened",
        "Run markitai doctor --fix to install the official headless shell, or set MARKITAI_BROWSER_EXECUTABLE to an installed Chrome/Chromium executable",
    );
    let strategy = cfg
        .pointer("/fetch/strategy")
        .and_then(Value::as_str)
        .unwrap_or("auto");
    browser.required = strategy == "playwright"
        || (strategy == "auto"
            && (cfg
                .pointer("/fetch/playwright/http_credentials")
                .is_some_and(|value| !value.is_null())
                || cfg
                    .pointer("/fetch/playwright/cookies")
                    .and_then(Value::as_array)
                    .is_some_and(|value| !value.is_empty())
                || cfg
                    .pointer("/fetch/playwright/extra_http_headers")
                    .and_then(Value::as_object)
                    .is_some_and(|value| !value.is_empty())))
        || config::enabled(cfg, "/screenshot/enabled")
        || config::enabled(cfg, "/screenshot/screenshot_only");
    // An explicit executable replaces discovery, and --fix will not replace it,
    // so a failed check must point at the variable rather than suggest --fix.
    if browser.status != "ok"
        && let Some(explicit) = std::env::var_os("MARKITAI_BROWSER_EXECUTABLE")
    {
        if browser.status == "missing" {
            browser.message = format!(
                "MARKITAI_BROWSER_EXECUTABLE does not name an executable file: {}",
                Path::new(&explicit).display()
            );
        }
        browser.install_hint = "MARKITAI_BROWSER_EXECUTABLE is set, so only that path is tried and doctor --fix will not replace it: point it at a working Chrome/Chromium executable, or unset it to use an installed browser or markitai doctor --fix".into();
    }
    result.insert("playwright", browser);
    let mut office = backend(
        "LibreOffice",
        "Office page and slide rendering",
        office,
        "Installed executable responds to --version in an isolated profile; document export fidelity is not tested",
        "Install LibreOffice; Office screenshots also require the native macOS PDF renderer",
    );
    if office.status == "ok" && !markitai_core::pdf_raster_available() {
        office.status = "warning";
        office
            .message
            .push_str("; native PDF rendering is unavailable on this platform");
    }
    result.insert("libreoffice", office);
    let ocr = markitai_core::local_ocr_available();
    let mut check = Check::new(
        "Local OCR",
        "Local image and rendered-page text recognition",
        if ocr { "ok" } else { "missing" },
        if ocr {
            "macOS Vision API is available; this check does not warm up OCR or establish recognition accuracy"
        } else {
            "No native local OCR backend is available on this platform"
        },
        if ocr {
            ""
        } else {
            "Use a supported macOS host or configure a vision model for supported VLM workflows"
        },
    );
    check.optional = Some(true);
    result.insert("rapidocr", check);
    let mut legacy = Check::new(
        "Native Office readers",
        "Legacy Office text extraction (.doc/.ppt)",
        "ok",
        "Rust readers are included; no Python anydoc package is required",
        "",
    );
    legacy.optional = Some(true);
    result.insert("anydoc", legacy);
    result.insert(
        "serve",
        Check::new(
            "Serve",
            "Web UI and REST conversion service",
            "ok",
            "Native server and embedded web interface are included",
            "",
        ),
    );
    let (llm, local) = model_check(cfg, path, env);
    result.insert("llm-api", llm);
    for provider in local {
        if provider == "copilot" {
            let auth = super::auth::copilot_status(env);
            let verified = auth.details.get("protocol_version").is_some();
            let mut adapter = Check::new(
                "Copilot official runtime",
                "Configured local provider",
                if verified { "ok" } else { "error" },
                if verified {
                    "Installed official runtime matches the native adapter protocol"
                } else {
                    "The installed Copilot runtime could not verify its supported protocol"
                },
                "Install the supported official Copilot CLI; no Python SDK is required",
            );
            adapter.required = true;
            let mut identity = Check::new(
                "Copilot authentication", "Configured local provider",
                if auth.authenticated { "ok" } else { "error" },
                auth.error.unwrap_or_else(|| "Official runtime reports an authenticated account; model access was not probed".into()),
                "Run markitai auth copilot login",
            );
            identity.required = true;
            result.insert("copilot-sdk", adapter);
            result.insert("copilot-auth", identity);
            continue;
        }
        if provider == "claude-agent" {
            let auth = super::auth::claude_status(env);
            let verified = auth.details.get("cli_version").is_some();
            let mut adapter = Check::new(
                "Claude official runtime",
                "Configured local provider",
                if verified { "ok" } else { "error" },
                if verified {
                    "Installed official runtime matches the native adapter version"
                } else {
                    "The installed Claude runtime could not verify its supported version"
                },
                "Install the supported official Claude CLI; no Python SDK is required",
            );
            adapter.required = true;
            let mut identity=Check::new("Claude subscription authentication", "Configured local provider", if auth.authenticated {"ok"} else {"error"},
                auth.error.unwrap_or_else(|| "Official runtime reports an authenticated subscription; model access was not probed".into()), "Run markitai auth claude login");
            identity.required = true;
            result.insert("claude-agent-sdk", adapter);
            result.insert("claude-agent-auth", identity);
            continue;
        }
        if provider == "chatgpt" {
            let auth = super::auth::chatgpt_status(env);
            let verified = auth.details.get("cli_version").is_some();
            let mut adapter = Check::new(
                "Codex official runtime",
                "Configured local provider",
                if verified { "ok" } else { "error" },
                if verified {
                    "Installed Codex runtime matches the pinned adapter version"
                } else {
                    "Installed Codex runtime could not verify its supported version"
                },
                "Install official Codex 0.159.0; this adapter currently supports chatgpt/gpt-5.5",
            );
            adapter.required = true;
            let mut identity = Check::new(
                "ChatGPT subscription authentication", "Configured local provider",
                if auth.authenticated { "ok" } else { "error" },
                auth.error.unwrap_or_else(|| "Official runtime reports a ChatGPT login; model entitlement and inference were not probed".into()),
                "Run markitai auth chatgpt login",
            );
            identity.required = true;
            result.insert("chatgpt-runtime", adapter);
            result.insert("chatgpt-auth", identity);
        }
    }
    let has_vision = !models.is_empty();
    let mut vision = Check::new(
        "Vision Model",
        "Image analysis (alt text, descriptions)",
        if has_vision { "ok" } else { "warning" },
        if has_vision {
            "Vision routing is configured; provider availability and image capability have not been probed"
        } else {
            "No routable vision model is configured"
        },
        if has_vision {
            ""
        } else {
            "Configure a supported model and its model_info.supports_vision capability"
        },
    );
    vision.models = Some(models.clone());
    result.insert("vision-model", vision);
    let opt_out = env
        .get("MARKITAI_NO_VLM_OCR")
        .is_some_and(|v| !["", "0", "false", "no"].contains(&v.trim().to_lowercase().as_str()));
    let mut vlm = Check::new(
        "VLM OCR",
        "Vision OCR for scanned documents (--ocr --llm)",
        if has_vision && !opt_out {
            "ok"
        } else {
            "warning"
        },
        if opt_out {
            "Disabled by MARKITAI_NO_VLM_OCR; OCR uses the local backend"
        } else if has_vision {
            "Vision requests are configured; no model request was made"
        } else {
            "Unavailable without a routable vision model"
        },
        if opt_out {
            "Unset MARKITAI_NO_VLM_OCR to allow VLM OCR"
        } else if !has_vision {
            "Configure a supported vision model"
        } else {
            ""
        },
    );
    vlm.optional = Some(true);
    vlm.models = Some(models);
    result.insert("vlm-ocr", vlm);
    result
}

fn backend(
    name: &'static str,
    description: &'static str,
    result: markitai_core::Result<Option<PathBuf>>,
    success: &'static str,
    hint: &'static str,
) -> Check {
    match result {
        Ok(Some(path)) => {
            let mut check = Check::new(name, description, "ok", success, "");
            check.path = Some(path);
            check
        }
        Ok(None) => Check::new(
            name,
            description,
            "missing",
            "No installed executable was found",
            hint,
        ),
        // Child stderr and arbitrary exception strings may contain credentials.
        Err(_) => Check::new(
            name,
            description,
            "warning",
            "The installed backend could not complete its bounded startup check",
            hint,
        ),
    }
}

fn nonempty(value: Option<&Value>) -> Option<&str> {
    value.and_then(Value::as_str).filter(|s| !s.is_empty())
}

fn model_check(
    cfg: &Value,
    path: Option<&Path>,
    env: &HashMap<String, String>,
) -> (Check, BTreeSet<&'static str>) {
    let models = cfg
        .pointer("/llm/model_list")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    let active: Vec<_> = models
        .iter()
        .filter(|model| {
            model
                .pointer("/litellm_params/weight")
                .and_then(Value::as_u64)
                .unwrap_or(1)
                > 0
        })
        .collect();
    let mut local = BTreeSet::new();
    let mut providers = BTreeSet::new();
    let mut missing = BTreeSet::new();
    let mut unsupported = false;
    for model in &active {
        let params = &model["litellm_params"];
        let name = params["model"].as_str().unwrap_or_default();
        let provider = name
            .split_once('/')
            .map_or("openai", |(provider, _)| provider);
        match provider {
            "claude-agent" => {
                local.insert("claude-agent");
                continue;
            }
            "copilot" => {
                local.insert("copilot");
                continue;
            }
            "chatgpt" => {
                local.insert("chatgpt");
                continue;
            }
            "openai" | "anthropic" | "gemini" | "deepseek" | "openrouter" | "azure" | "ollama"
            | "ollama_chat" => {
                providers.insert(provider);
            }
            _ => unsupported = true,
        }
        let linked = model
            .pointer("/model_info/provider_id")
            .and_then(Value::as_str)
            .and_then(|id| {
                cfg.pointer("/llm/providers")
                    .and_then(Value::as_array)
                    .and_then(|items| {
                        items
                            .iter()
                            .find(|provider| provider["id"].as_str() == Some(id))
                    })
            });
        for field in ["api_key", "api_base"] {
            if let Some(variable) = nonempty(params.get(field))
                .or_else(|| nonempty(linked.and_then(|p| p.get(field))))
                .and_then(|value| value.strip_prefix("env:"))
                && !env.contains_key(variable)
            {
                // A malformed env name can contain terminal control characters;
                // only expose its safe spelling, never a resolved value.
                missing.insert(
                    variable
                        .chars()
                        .take(128)
                        .map(|c| if c.is_control() { ' ' } else { c })
                        .collect::<String>(),
                );
            }
        }
    }
    let mut check = Check::new(
        "LLM API",
        "Content enhancement and image analysis",
        "ok",
        format!(
            "{} active model(s); {} supported API provider(s)",
            active.len(),
            providers.len()
        ),
        "",
    );
    check.required = !providers.is_empty() || unsupported;
    // Without llm.model_list, conversions fall back to MODEL or provider API
    // keys in the environment; report what --llm would actually use.
    let detected = models
        .is_empty()
        .then(|| markitai_core::llm_capabilities(cfg))
        .filter(|detected| !detected.models.is_empty());
    if let Some(detected) = detected {
        let names = detected.models.join(", ");
        if detected.routable {
            check.message = format!(
                "No llm.model_list; --llm uses {names} detected from MODEL or provider API keys in the environment"
            );
            check.install_hint =
                "Optional: run markitai init to save the detected model in a configuration file"
                    .into();
        } else {
            check.status = "warning";
            check.message = format!(
                "No llm.model_list; the environment names {names}, but its provider API key is not set"
            );
            check.install_hint =
                "Set the provider's API key (for example OPENAI_API_KEY) or configure llm.model_list"
                    .into();
        }
    } else if models.is_empty() {
        check.status = "missing";
        check.message = "No models configured in llm.model_list".into();
        check.install_hint = path.map(|path| format!("Configure llm.model_list in {}", path.display())).unwrap_or_else(|| "Set a provider API key such as OPENAI_API_KEY (optionally with MODEL), or run markitai init or configure llm.model_list".into());
    } else if active.is_empty() {
        check.status = "warning";
        check.message = "Models are configured, but all have weight 0 (disabled)".into();
        check.install_hint = "Set weight > 0 on at least one model to enable LLM use".into();
    } else if !missing.is_empty() {
        check.status = "error";
        check.message = format!(
            "Missing environment variable(s) used by active models: {}",
            missing.into_iter().collect::<Vec<_>>().join(", ")
        );
        check.install_hint =
            "Set the named environment variables or update the selected model/provider connection"
                .into();
    } else if unsupported {
        check.status = "error";
        check.message = "An active API provider is not implemented in this native runtime".into();
        check.install_hint = "Choose a supported API provider".into();
    } else if !providers.is_empty() && !markitai_core::llm_capabilities(cfg).routable {
        check.status = "error";
        check.message = "No active API model has usable credentials and an endpoint".into();
        check.install_hint =
            "Check model/provider credentials, API base and routing configuration".into();
    }
    (check, local)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn required_status_and_linked_credentials_do_not_leak_values() {
        let cfg = config::normalize(&json!({"fetch":{"strategy":"playwright"},"llm":{"providers":[{"id":"connection","provider":"openai","api_key":"env:ABSENT_ROUND23"}],"model_list":[{"model_name":"default","litellm_params":{"model":"openai/test"},"model_info":{"provider_id":"connection"}}]}})).unwrap();
        let checks = checks(
            &cfg,
            None,
            &HashMap::new(),
            Err(markitai_core::Error::Fetch("secret-child-error".into())),
            Ok(None),
            Vec::new(),
        );
        assert!(checks["playwright"].failed());
        assert!(checks["llm-api"].failed());
        let text = serde_json::to_string(&checks).unwrap();
        assert!(text.contains("ABSENT_ROUND23"));
        assert!(!text.contains("secret-child-error"));
        let value: Value = serde_json::from_str(&text).unwrap();
        assert!(
            value
                .as_object()
                .unwrap()
                .values()
                .all(|check| check.get("required").is_none())
        );
    }

    #[test]
    fn optional_missing_and_disabled_models_are_nonblocking() {
        let cfg = config::normalize(&json!({"llm":{"model_list":[{"model_name":"off","litellm_params":{"model":"openai/test","weight":0,"api_key":"env:ABSENT_ROUND23"}}]}})).unwrap();
        let checks = checks(&cfg, None, &HashMap::new(), Ok(None), Ok(None), Vec::new());
        assert!(!checks.values().any(Check::failed));
        assert_eq!(checks["llm-api"].status, "warning");
        assert_eq!(
            checks.keys().copied().collect::<Vec<_>>(),
            [
                "playwright",
                "libreoffice",
                "rapidocr",
                "anydoc",
                "serve",
                "llm-api",
                "vision-model",
                "vlm-ocr"
            ]
        );
    }
    #[test]
    fn credentialed_auto_requires_browser_but_explicit_static_does_not() {
        for (field, identity) in [
            (
                "http_credentials",
                json!({"username":"test-user","password":"test-password"}),
            ),
            (
                "cookies",
                json!([{"name":"session","value":"fixture","url":"https://example.test"}]),
            ),
            (
                "extra_http_headers",
                json!({"Authorization":"fixture-only"}),
            ),
        ] {
            let mut cfg = config::defaults();
            cfg["fetch"]["playwright"][field] = identity;
            let auto = checks(&cfg, None, &HashMap::new(), Ok(None), Ok(None), Vec::new());
            assert!(auto["playwright"].failed(), "{field}");
            cfg["fetch"]["strategy"] = json!("static");
            let explicit = checks(&cfg, None, &HashMap::new(), Ok(None), Ok(None), Vec::new());
            assert!(!explicit["playwright"].failed(), "{field}");
        }
    }
}
