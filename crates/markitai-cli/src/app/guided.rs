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
        "{}",
        text!(
            "Markitai conversion wizard\nChoices affect this run only. Enter q to cancel.",
            "Markitai 转换向导\n选择只对本次运行生效。输入 q 取消。"
        )
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
        &text!(
            "Input: 1 file, 2 directory, 3 URL [{default}]: ",
            "输入：1 文件，2 目录，3 URL [{default}]："
        ),
        &["1", "2", "3"],
        default,
    )?
    else {
        return Ok(None);
    };
    let source = loop {
        let current = cli.input.as_deref().unwrap_or_default();
        let label = if current.is_empty() {
            text!("Input path or URL: ", "输入路径或 URL：")
        } else {
            format!(
                "{} [{}]: ",
                text!("Input path or URL", "输入路径或 URL"),
                visible(current)
            )
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
                "{}",
                text!(
                    "Enter an absolute HTTP(S) URL without embedded credentials.",
                    "请输入不含内嵌凭据的完整 HTTP(S) URL。"
                )
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
            let kind_label = if kind == "2" {
                text!("directory", "目录")
            } else {
                text!("file", "文件")
            };
            writeln!(
                out,
                "{}",
                text!(
                    "The selected {kind_label} does not exist.",
                    "所选{kind_label}不存在。"
                )
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
                "{} [{}]: ",
                text!("Output directory", "输出目录"),
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
            writeln!(
                out,
                "{}",
                text!(
                    "Choose an output directory, not a Markdown filename.",
                    "请选择输出目录，不要指定 Markdown 文件名。"
                )
            )
            .map_err(runtime)?;
            continue;
        }
        break path;
    };
    let Some(llm) = toggle(
        input,
        out,
        &text!("LLM enhancement", "LLM 增强"),
        config::enabled(&cfg, "/llm/enabled"),
    )?
    else {
        return Ok(None);
    };
    cfg["llm"]["enabled"] = json!(llm);
    while config::enabled(&cfg, "/llm/enabled") && !markitai_core::llm_capabilities(&cfg).routable {
        writeln!(
            out,
            "{}",
            text!("No usable API model is configured. Subscription/CLI providers are not supported here.", "尚未配置可用的 API 模型；此处不支持订阅或 CLI 提供商。")
        )
        .map_err(runtime)?;
        let Some(action) = interactive::choice(
            input,
            out,
            &text!(
                "1 configure a model for this run, 2 retry detection, 3 disable LLM, q cancel [3]: ",
                "1 为本次运行配置模型，2 重新检测，3 关闭 LLM，q 取消 [3]："
            ),
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
        (
            text!("Image alt text", "图片 alt 文本"),
            "image",
            "alt_enabled",
        ),
        (
            text!("Image descriptions", "图片描述"),
            "image",
            "desc_enabled",
        ),
        (text!("Pure output", "纯净输出"), "llm", "pure"),
        (
            text!("Local / vision OCR", "本地 / 视觉 OCR"),
            "ocr",
            "enabled",
        ),
        (
            text!("Page screenshots", "页面截图"),
            "screenshot",
            "enabled",
        ),
    ] {
        if section == "image" && !config::enabled(&cfg, "/llm/enabled") {
            continue;
        }
        let current = cfg[section][field].as_bool().unwrap_or(false);
        let Some(value) = toggle(input, out, &label, current)? else {
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
        "{}\n{}: {}\n{}: {}",
        text!("\nConfiguration summary", "\n配置摘要"),
        text!("Input", "输入"),
        visible(&source),
        text!("Output", "输出"),
        visible(&output.to_string_lossy())
    )
    .map_err(runtime)?;
    for (label, pointer) in [
        (text!("LLM", "LLM"), "/llm/enabled"),
        (text!("Alt text", "alt 文本"), "/image/alt_enabled"),
        (text!("Descriptions", "图片描述"), "/image/desc_enabled"),
        (text!("Pure", "纯净输出"), "/llm/pure"),
        (text!("OCR", "OCR"), "/ocr/enabled"),
        (text!("Screenshots", "页面截图"), "/screenshot/enabled"),
        (
            text!("Screenshot only", "仅截图"),
            "/screenshot/screenshot_only",
        ),
    ] {
        let enabled = config::enabled(&cfg, pointer);
        let inactive = pointer.starts_with("/image/") && !config::enabled(&cfg, "/llm/enabled");
        writeln!(
            out,
            "{label}: {}{}",
            if enabled {
                text!("enabled", "开启")
            } else {
                text!("disabled", "关闭")
            },
            if inactive {
                text!(" (inactive without LLM)", "（LLM 未开启，不生效）")
            } else {
                String::new()
            }
        )
        .map_err(runtime)?;
    }
    if config::enabled(&cfg, "/llm/enabled") {
        let models = markitai_core::llm_capabilities(&cfg).models;
        writeln!(
            out,
            "{}: {}",
            text!("Models", "模型"),
            models
                .iter()
                .map(|name| visible(name))
                .collect::<Vec<_>>()
                .join(", ")
        )
        .map_err(runtime)?;
    }
    if let Some(profile) = cfg["output"]["profile"].as_str() {
        writeln!(
            out,
            "{}: {}",
            text!("Output profile", "输出配置"),
            visible(profile)
        )
        .map_err(runtime)?;
    }
    let Some(answer) = interactive::choice(
        input,
        out,
        &text!("Execute conversion? y/n [y]: ", "开始转换？y/n [y]："),
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
    let state = if current {
        text!("enabled", "开启")
    } else {
        text!("disabled", "关闭")
    };
    let Some(answer) = interactive::choice(
        input,
        out,
        &text!(
            "{label}: y enable, n disable, Enter keep {state}: ",
            "{label}：y 开启，n 关闭，Enter 保持{state}："
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
            &text!(
                "Model (for example openai/gpt-5.6-luna or ollama/model): ",
                "模型（例如 openai/gpt-5.6-luna 或 ollama/model）："
            ),
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
        writeln!(
            out,
            "{}",
            text!(
                "Enter a model for a supported API provider.",
                "请输入受支持 API 提供商的模型。"
            )
        )
        .map_err(runtime)?;
    };
    let Some(base) = interactive::prompt(
        input,
        out,
        &text!(
            "API base URL (Enter uses provider default): ",
            "API 地址（Enter 使用提供商默认值）："
        ),
    )?
    else {
        return Ok(false);
    };
    if cancelled(&base) {
        return Ok(false);
    }
    let Some(key_kind) = interactive::choice(
        input,
        out,
        &text!(
            "Credential: 1 environment variable, 2 secret for this run, 3 provider default [3]: ",
            "凭据：1 环境变量，2 本次运行的密钥，3 提供商默认值 [3]："
        ),
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
                let Some(name) = interactive::prompt(
                    input,
                    out,
                    &text!("Environment variable name: ", "环境变量名称："),
                )?
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
                    "{}",
                    text!(
                        "Use a nonempty variable name containing letters, digits or underscores.",
                        "变量名不能为空，只能包含字母、数字或下划线。"
                    )
                )
                .map_err(runtime)?;
            };
            params["api_key"] = json!(format!("env:{name}"));
        }
        "2" => {
            let Some(key) = interactive::secret(
                input,
                out,
                &text!(
                    "API key (hidden; not saved): ",
                    "API 密钥（隐藏输入，不保存）："
                ),
            )?
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
