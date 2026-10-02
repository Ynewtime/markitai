//! Gather one conversion without changing configuration or re-entering the CLI.
use super::{Cli, CliResult, interactive, runtime};
use markitai_core::config;
use serde_json::{Value, json};
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};

pub(super) struct GuidedRun {
    pub input: String,
    pub output: PathBuf,
    pub config: Value,
}

pub(super) fn collect(cli: &Cli, cfg: Value) -> CliResult<Option<GuidedRun>> {
    interactive::terminal()?;
    let _cancel = interactive::CancelGuard::new().map_err(runtime)?;
    collect_with(cli, cfg, &mut io::stdin().lock(), &mut io::stderr().lock())
}

fn collect_with(
    cli: &Cli,
    mut cfg: Value,
    input: &mut impl BufRead,
    out: &mut impl Write,
) -> CliResult<Option<GuidedRun>> {
    writeln!(
        out,
        "Markitai conversion wizard\nChoices affect this run only. Enter q to cancel."
    )
    .map_err(runtime)?;
    let default = match cli.input.as_deref() {
        Some(source) if super::is_url(source) => "3",
        Some(source)
            if {
                let path = config::expand_home(Path::new(source));
                path.is_dir() && !markitai_core::formats::is_numbers_package_path(&path)
            } =>
        {
            "2"
        }
        _ => "1",
    };
    let Some(kind) = interactive::choice(
        input,
        out,
        &format!("Input: 1 file, 2 directory, 3 URL [{default}]: "),
        &["1", "2", "3"],
        default,
    )?
    else {
        return Ok(None);
    };
    let source = loop {
        let current = cli.input.as_deref().unwrap_or_default();
        let label = if current.is_empty() {
            "Input path or URL: ".into()
        } else {
            format!("Input path or URL [{}]: ", visible(current))
        };
        let Some(answer) = interactive::prompt(input, out, &label)? else {
            return Ok(None);
        };
        if cancelled(&answer) {
            return Ok(None);
        }
        let source = if answer.is_empty() { current } else { &answer };
        if kind == "3" {
            if let Ok(url) = url::Url::parse(source)
                && matches!(url.scheme(), "http" | "https")
                && url.host_str().is_some()
                && url.username().is_empty()
                && url.password().is_none()
            {
                break source.to_owned();
            }
            writeln!(
                out,
                "Enter an absolute HTTP(S) URL without embedded credentials."
            )
            .map_err(runtime)?;
        } else {
            let path = config::expand_home(Path::new(source));
            if !source.is_empty()
                && if kind == "2" {
                    path.is_dir() && !markitai_core::formats::is_numbers_package_path(&path)
                } else {
                    path.is_file() || markitai_core::formats::is_numbers_package_path(&path)
                }
            {
                break path.to_string_lossy().into_owned();
            }
            writeln!(
                out,
                "The selected {} does not exist.",
                if kind == "2" { "directory" } else { "file" }
            )
            .map_err(runtime)?;
        }
    };
    let default_output = cli
        .output
        .clone()
        .or_else(|| cfg["output"]["dir"].as_str().map(PathBuf::from))
        .unwrap_or_else(|| "./output".into());
    let output = loop {
        let Some(answer) = interactive::prompt(
            input,
            out,
            &format!(
                "Output directory [{}]: ",
                visible(&default_output.to_string_lossy())
            ),
        )?
        else {
            return Ok(None);
        };
        if cancelled(&answer) {
            return Ok(None);
        }
        let path = config::expand_home(if answer.is_empty() {
            &default_output
        } else {
            Path::new(&answer)
        });
        if path.is_file()
            || (!path.is_dir()
                && path
                    .extension()
                    .is_some_and(|extension| extension.eq_ignore_ascii_case("md")))
        {
            writeln!(out, "Choose an output directory, not a Markdown filename.")
                .map_err(runtime)?;
            continue;
        }
        break path;
    };
    let Some(llm) = toggle(
        input,
        out,
        "LLM enhancement",
        config::enabled(&cfg, "/llm/enabled"),
    )?
    else {
        return Ok(None);
    };
    cfg["llm"]["enabled"] = json!(llm);
    while config::enabled(&cfg, "/llm/enabled") && !markitai_core::llm_capabilities(&cfg).routable {
        writeln!(
            out,
            "No usable API model is configured. Subscription/CLI providers are not supported here."
        )
        .map_err(runtime)?;
        let Some(action) = interactive::choice(
            input,
            out,
            "1 configure a model for this run, 2 retry detection, 3 disable LLM, q cancel [3]: ",
            &["1", "2", "3"],
            "3",
        )?
        else {
            return Ok(None);
        };
        match action.as_str() {
            "1" if !configure_model(&mut cfg, input, out)? => return Ok(None),
            "3" => cfg["llm"]["enabled"] = json!(false),
            _ => {}
        }
    }
    for (label, section, field) in [
        ("Image alt text", "image", "alt_enabled"),
        ("Image descriptions", "image", "desc_enabled"),
        ("Pure output", "llm", "pure"),
        ("Local / vision OCR", "ocr", "enabled"),
        ("Page screenshots", "screenshot", "enabled"),
    ] {
        if section == "image" && !config::enabled(&cfg, "/llm/enabled") {
            continue;
        }
        let current = cfg[section][field].as_bool().unwrap_or(false);
        let Some(value) = toggle(input, out, label, current)? else {
            return Ok(None);
        };
        cfg[section][field] = json!(value);
    }
    // screenshot-only is an explicit input flag/configuration choice; the wizard
    // must show the same effective implication that the conversion core applies.
    if config::enabled(&cfg, "/screenshot/screenshot_only") {
        cfg["screenshot"]["enabled"] = json!(true);
    }
    config::validate(&cfg).map_err(runtime)?;
    writeln!(
        out,
        "\nConfiguration summary\nInput: {}\nOutput: {}",
        visible(&source),
        visible(&output.to_string_lossy())
    )
    .map_err(runtime)?;
    for (label, pointer) in [
        ("LLM", "/llm/enabled"),
        ("Alt text", "/image/alt_enabled"),
        ("Descriptions", "/image/desc_enabled"),
        ("Pure", "/llm/pure"),
        ("OCR", "/ocr/enabled"),
        ("Screenshots", "/screenshot/enabled"),
        ("Screenshot only", "/screenshot/screenshot_only"),
    ] {
        let enabled = config::enabled(&cfg, pointer);
        let inactive = pointer.starts_with("/image/") && !config::enabled(&cfg, "/llm/enabled");
        writeln!(
            out,
            "{label}: {}{}",
            if enabled { "enabled" } else { "disabled" },
            if inactive {
                " (inactive without LLM)"
            } else {
                ""
            }
        )
        .map_err(runtime)?;
    }
    if config::enabled(&cfg, "/llm/enabled") {
        let models = markitai_core::llm_capabilities(&cfg).models;
        writeln!(
            out,
            "Models: {}",
            models
                .iter()
                .map(|name| visible(name))
                .collect::<Vec<_>>()
                .join(", ")
        )
        .map_err(runtime)?;
    }
    if let Some(profile) = cfg["output"]["profile"].as_str() {
        writeln!(out, "Output profile: {}", visible(profile)).map_err(runtime)?;
    }
    let Some(answer) = interactive::choice(
        input,
        out,
        "Execute conversion? y/n [y]: ",
        &["y", "n"],
        "y",
    )?
    else {
        return Ok(None);
    };
    Ok((answer == "y").then_some(GuidedRun {
        input: source,
        output,
        config: cfg,
    }))
}

fn cancelled(value: &str) -> bool {
    matches!(value.trim(), "q" | "\u{1b}")
}

fn toggle(
    input: &mut impl BufRead,
    out: &mut impl Write,
    label: &str,
    current: bool,
) -> CliResult<Option<bool>> {
    let Some(answer) = interactive::choice(
        input,
        out,
        &format!(
            "{label}: y enable, n disable, Enter keep {}: ",
            if current { "enabled" } else { "disabled" }
        ),
        &["y", "n"],
        if current { "y" } else { "n" },
    )?
    else {
        return Ok(None);
    };
    Ok(Some(answer == "y"))
}

fn configure_model(
    cfg: &mut Value,
    input: &mut impl BufRead,
    out: &mut impl Write,
) -> CliResult<bool> {
    let model = loop {
        let Some(model) = interactive::prompt(
            input,
            out,
            "Model (for example openai/gpt-5.6-luna or ollama/model): ",
        )?
        else {
            return Ok(false);
        };
        if cancelled(&model) {
            return Ok(false);
        }
        let model = model.trim();
        let provider = model
            .split_once('/')
            .map_or("openai", |(provider, _)| provider);
        if !model.is_empty()
            && !model.chars().any(char::is_control)
            && interactive::api_provider(provider)
        {
            break model.to_owned();
        }
        writeln!(out, "Enter a model for a supported API provider.").map_err(runtime)?;
    };
    let Some(base) =
        interactive::prompt(input, out, "API base URL (Enter uses provider default): ")?
    else {
        return Ok(false);
    };
    if cancelled(&base) {
        return Ok(false);
    }
    let Some(key_kind) = interactive::choice(
        input,
        out,
        "Credential: 1 environment variable, 2 secret for this run, 3 provider default [3]: ",
        &["1", "2", "3"],
        "3",
    )?
    else {
        return Ok(false);
    };
    let mut params = json!({"model":model});
    if !base.is_empty() {
        params["api_base"] = json!(base);
    }
    match key_kind.as_str() {
        "1" => {
            let name = loop {
                let Some(name) = interactive::prompt(input, out, "Environment variable name: ")?
                else {
                    return Ok(false);
                };
                if cancelled(&name) {
                    return Ok(false);
                }
                if !name.is_empty()
                    && name
                        .bytes()
                        .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
                {
                    break name;
                }
                writeln!(
                    out,
                    "Use a nonempty variable name containing letters, digits or underscores."
                )
                .map_err(runtime)?;
            };
            params["api_key"] = json!(format!("env:{name}"));
        }
        "2" => {
            let Some(key) = interactive::secret(input, out, "API key (hidden; not saved): ")?
            else {
                return Ok(false);
            };
            if cancelled(&key) {
                return Ok(false);
            }
            params["api_key"] = json!(key);
        }
        _ => {}
    }
    cfg["llm"]["model_list"] = json!([{"model_name":"default","litellm_params":params}]);
    Ok(true)
}

fn visible(value: &str) -> String {
    let mut value = value.to_owned();
    if let Ok(mut url) = url::Url::parse(&value)
        && matches!(url.scheme(), "http" | "https")
    {
        let _ = url.set_username("");
        let _ = url.set_password(None);
        url.set_query(None);
        url.set_fragment(None);
        value = url.to_string();
    }
    value
        .chars()
        .take(512)
        .map(|ch| if ch.is_control() { ' ' } else { ch })
        .collect()
}
