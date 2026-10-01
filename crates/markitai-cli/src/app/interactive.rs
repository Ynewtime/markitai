//! Line-oriented navigation works without a terminal UI runtime or subprocess.
use super::i18n::{Lang, lang};
use super::{CliResult, runtime, write_config};
use markitai_core::config;
use serde_json::{Value, json};
use std::fs;
use std::io::{self, BufRead, IsTerminal, Read, Write};
use std::path::{Path, PathBuf};

const CONFIG_LIMIT: u64 = 8 * 1024 * 1024;
const PAGE: usize = 15;

struct Setting {
    key: String,
    value: Value,
    node: Value,
}

pub(super) fn terminal() -> CliResult<()> {
    if !io::stdin().is_terminal() || !io::stderr().is_terminal() {
        return Err((
            2,
            text!(
                "Interactive configuration requires a terminal; use config set or init --yes for automation",
                "交互式配置需要终端；自动化场景请使用 config set 或 init --yes"
            ),
        ));
    }
    Ok(())
}

pub(super) fn prompt(
    input: &mut impl BufRead,
    output: &mut impl Write,
    label: &str,
) -> CliResult<Option<String>> {
    write!(output, "{label}").map_err(runtime)?;
    output.flush().map_err(runtime)?;
    let mut bytes = Vec::new();
    let read = input
        .take(65537)
        .read_until(b'\n', &mut bytes)
        .map_err(runtime)?;
    if read == 0 {
        return Ok(None);
    }
    if read > 65536 {
        return Err(runtime(text!(
            "Interactive input exceeds 64 KiB",
            "交互输入超过 64 KiB"
        )));
    }
    let line = String::from_utf8(bytes).map_err(|_| {
        runtime(text!(
            "Interactive input must be UTF-8",
            "交互输入必须是 UTF-8"
        ))
    })?;
    Ok(Some(line.trim_end_matches(['\r', '\n']).to_owned()))
}

fn read_bytes(path: &Path) -> CliResult<Option<Vec<u8>>> {
    match fs::metadata(path) {
        Ok(metadata) if !metadata.is_file() => {
            return Err(runtime(text!(
                "Configuration must be a regular file",
                "配置必须是普通文件"
            )));
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(runtime(error)),
        _ => {}
    }
    let mut options = fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NONBLOCK);
    }
    let file = match options.open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(runtime(error)),
    };
    if !file.metadata().map_err(runtime)?.is_file() {
        return Err(runtime(text!(
            "Configuration must be a regular file",
            "配置必须是普通文件"
        )));
    }
    let mut bytes = Vec::new();
    file.take(CONFIG_LIMIT + 1)
        .read_to_end(&mut bytes)
        .map_err(runtime)?;
    if bytes.len() as u64 > CONFIG_LIMIT {
        return Err(runtime(text!(
            "Configuration exceeds 8 MiB",
            "配置文件超过 8 MiB"
        )));
    }
    Ok(Some(bytes))
}
fn parse_config(bytes: Option<&[u8]>) -> CliResult<Value> {
    let raw: Value = bytes
        .map(serde_json::from_slice)
        .transpose()
        .map_err(runtime)?
        .unwrap_or_else(|| json!({}));
    if !raw.is_object() {
        return Err(runtime(text!(
            "Configuration must be a JSON object",
            "配置必须是 JSON 对象"
        )));
    }
    config::validate(&raw).map_err(runtime)?;
    Ok(raw)
}
fn read_config(path: &Path) -> CliResult<(Value, Option<Vec<u8>>)> {
    let bytes = read_bytes(path)?;
    Ok((parse_config(bytes.as_deref())?, bytes))
}

fn unchanged(path: &Path, expected: &Option<Vec<u8>>) -> CliResult<()> {
    let actual = read_bytes(path)?;
    if &actual != expected {
        return Err(runtime(text!(
            "Configuration changed outside this session; reopen it before saving",
            "配置文件在本次会话之外被修改；请重新打开后再保存"
        )));
    }
    Ok(())
}

fn node_type(node: &Value) -> &Value {
    if let Some(reference) = node["$ref"].as_str() {
        return node_type(config::schema().pointer(&reference[1..]).unwrap_or(node));
    }
    node
}
fn scalar_node(node: &Value) -> &Value {
    let node = node_type(node);
    node["anyOf"]
        .as_array()
        .and_then(|nodes| nodes.iter().find(|n| n["type"] != "null"))
        .map(node_type)
        .unwrap_or(node)
}
fn settings(cfg: &Value) -> Vec<Setting> {
    fn walk(value: &Value, node: &Value, prefix: &str, out: &mut Vec<Setting>) {
        let node = node_type(node);
        if let Some(properties) = node["properties"].as_object() {
            for (name, child) in properties {
                if ["presets", "prompts", "domain_profiles"].contains(&name.as_str()) {
                    continue;
                }
                let key = if prefix.is_empty() {
                    name.clone()
                } else {
                    format!("{prefix}.{name}")
                };
                walk(&value[name], child, &key, out);
            }
        } else if matches!(
            scalar_node(node)["type"].as_str(),
            Some("string" | "boolean" | "number" | "integer")
        ) {
            out.push(Setting {
                key: prefix.into(),
                value: value.clone(),
                node: node.clone(),
            });
        }
    }
    let mut result = Vec::new();
    walk(cfg, config::schema(), "", &mut result);
    result
}

fn shown(key: &str, value: &Value) -> String {
    let visible = config::redact_for_key(key, value);
    let value = match visible {
        Value::String(s) => s,
        Value::Null => "null".into(),
        value => value.to_string(),
    };
    let mut text: String = value
        .chars()
        .take(100)
        .map(|c| if c.is_control() { ' ' } else { c })
        .collect();
    if value.chars().count() > 100 {
        text.push('…');
    }
    text
}

fn fuzzy(query: &str, key: &str) -> bool {
    let query = query.to_lowercase();
    let mut chars = query.chars();
    let mut wanted = chars.next();
    for ch in key.to_lowercase().chars() {
        if wanted == Some(ch) {
            wanted = chars.next();
        }
    }
    wanted.is_none()
}

pub(super) fn edit(path: &Path) -> CliResult<()> {
    terminal()?;
    edit_with(
        path,
        &mut io::stdin().lock(),
        &mut io::stderr().lock(),
        true,
    )
}

fn edit_with(
    path: &Path,
    input: &mut impl BufRead,
    output: &mut impl Write,
    terminal_input: bool,
) -> CliResult<()> {
    let (mut raw, mut original) = read_config(path)?;
    let mut query = String::new();
    let mut page = 0;
    loop {
        let normalized = config::normalize(&raw).map_err(runtime)?;
        let all = settings(&normalized);
        let filtered: Vec<_> = all
            .iter()
            .filter(|setting| fuzzy(&query, &setting.key))
            .collect();
        let pages = filtered.len().div_ceil(PAGE).max(1);
        page = page.min(pages - 1);
        writeln!(
            output,
            "\nConfiguration: {}\nPage {}/{}; search: {}",
            path.display(),
            page + 1,
            pages,
            query
        )
        .map_err(runtime)?;
        for (index, setting) in filtered.iter().skip(page * PAGE).take(PAGE).enumerate() {
            writeln!(
                output,
                "  {}. {} = {}",
                index + 1,
                setting.key,
                shown(&setting.key, &setting.value)
            )
            .map_err(runtime)?;
        }
        let Some(command) = prompt(
            input,
            output,
            "Select number or full key; /search, n next, p previous, q quit: ",
        )?
        else {
            break;
        };
        let command = command.trim();
        match command {
            "q" | "quit" | "\u{1b}" => break,
            "n" => {
                page = (page + 1).min(pages - 1);
                continue;
            }
            "p" => {
                page = page.saturating_sub(1);
                continue;
            }
            _ if command.starts_with('/') => {
                query = command[1..].to_owned();
                page = 0;
                continue;
            }
            _ => {}
        }
        let selected = command
            .parse::<usize>()
            .ok()
            .filter(|i| (1..=PAGE).contains(i))
            .and_then(|i| filtered.get(page * PAGE + i - 1).copied())
            .or_else(|| all.iter().find(|setting| setting.key == command));
        let Some(setting) = selected else {
            writeln!(output, "No editable setting selected.").map_err(runtime)?;
            continue;
        };
        let shape = scalar_node(&setting.node);
        let hint = shape["enum"]
            .as_array()
            .map(|values| {
                format!(
                    "; choices: {}",
                    values
                        .iter()
                        .map(Value::to_string)
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            })
            .unwrap_or_default();
        writeln!(
            output,
            "{} ({}){hint}. Current: {}",
            setting.key,
            shape["type"].as_str().unwrap_or("value"),
            shown(&setting.key, &setting.value)
        )
        .map_err(runtime)?;
        let secret =
            config::redact_for_key(&setting.key, &json!("test-value")) != json!("test-value");
        let value = {
            let _echo = if secret && terminal_input {
                Some(Echo::hide().map_err(runtime)?)
            } else {
                None
            };
            prompt(
                input,
                output,
                "New value (empty keeps current, :empty sets an empty string, :cancel returns): ",
            )
        };
        if secret && terminal_input {
            writeln!(output).map_err(runtime)?;
        }
        let Some(value) = value? else { break };
        if value.is_empty() || matches!(value.as_str(), ":cancel" | "\u{1b}") {
            continue;
        }
        let parsed = if value == ":empty" {
            Ok(json!(""))
        } else {
            config::parse_cli_value(&raw, &setting.key, &value)
        };
        let mut candidate = raw.clone();
        let changed =
            parsed.and_then(|value| config::set_value(&mut candidate, &setting.key, value));
        match changed {
            Ok(value) => {
                unchanged(path, &original)?;
                write_config(path, &candidate)?;
                let (saved, bytes) = read_config(path)?;
                raw = saved;
                original = bytes;
                writeln!(
                    output,
                    "Saved {} = {}",
                    setting.key,
                    shown(&setting.key, &value)
                )
                .map_err(runtime)?;
            }
            Err(error) => {
                writeln!(output, "Invalid value: {error}").map_err(runtime)?;
            }
        }
    }
    Ok(())
}

/// A wizard owns cancellation only while prompting, before conversion begins.
#[cfg(unix)]
static GUIDED_CANCEL: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
#[cfg(unix)]
extern "C" fn cancelled(_: libc::c_int) {
    const MESSAGE: &[u8] = b"\nCancelled.\n";
    unsafe {
        libc::write(libc::STDERR_FILENO, MESSAGE.as_ptr().cast(), MESSAGE.len());
        libc::_exit(0);
    }
}
#[cfg(unix)]
pub(super) struct CancelGuard(libc::sigaction);
#[cfg(unix)]
impl CancelGuard {
    pub(super) fn new() -> io::Result<Self> {
        let mut previous = unsafe { std::mem::zeroed() };
        let mut action: libc::sigaction = unsafe { std::mem::zeroed() };
        action.sa_sigaction = cancelled as *const () as libc::sighandler_t;
        unsafe {
            libc::sigemptyset(&mut action.sa_mask);
        }
        if unsafe { libc::sigaction(libc::SIGINT, &action, &mut previous) } != 0 {
            return Err(io::Error::last_os_error());
        }
        GUIDED_CANCEL.store(true, std::sync::atomic::Ordering::Relaxed);
        Ok(Self(previous))
    }
}
#[cfg(unix)]
impl Drop for CancelGuard {
    fn drop(&mut self) {
        GUIDED_CANCEL.store(false, std::sync::atomic::Ordering::Relaxed);
        unsafe {
            libc::sigaction(libc::SIGINT, &self.0, std::ptr::null_mut());
        }
    }
}
#[cfg(not(unix))]
pub(super) struct CancelGuard;
#[cfg(not(unix))]
impl CancelGuard {
    pub(super) fn new() -> io::Result<Self> {
        Ok(Self)
    }
}

pub(super) fn secret(
    input: &mut impl BufRead,
    output: &mut impl Write,
    label: &str,
) -> CliResult<Option<String>> {
    let answer = {
        let _echo = Echo::hide().map_err(runtime)?;
        prompt(input, output, label)
    };
    writeln!(output).map_err(runtime)?;
    answer
}

#[cfg(unix)]
static PREVIOUS_ECHO: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
#[cfg(unix)]
extern "C" fn interrupted_secret(signal: libc::c_int) {
    // POSIX lists tcgetattr/tcsetattr and _exit as async-signal-safe. Avoid Rust
    // allocation, locks and unwinding while restoring the sole flag we changed.
    let mut terminal = std::mem::MaybeUninit::<libc::termios>::uninit();
    unsafe {
        if libc::tcgetattr(libc::STDIN_FILENO, terminal.as_mut_ptr()) == 0 {
            let mut terminal = terminal.assume_init();
            if PREVIOUS_ECHO.load(std::sync::atomic::Ordering::Relaxed) {
                terminal.c_lflag |= libc::ECHO;
            } else {
                terminal.c_lflag &= !libc::ECHO;
            }
            libc::tcsetattr(libc::STDIN_FILENO, libc::TCSANOW, &terminal);
        }
        if signal == libc::SIGINT && GUIDED_CANCEL.load(std::sync::atomic::Ordering::Relaxed) {
            cancelled(signal);
        }
        libc::_exit(128 + signal);
    }
}
#[cfg(unix)]
struct Echo {
    previous: libc::termios,
    handlers: [(libc::c_int, libc::sigaction); 2],
}
#[cfg(unix)]
impl Echo {
    fn hide() -> io::Result<Self> {
        let mut previous = std::mem::MaybeUninit::<libc::termios>::uninit();
        // The terminal is borrowed only for this input; Drop restores its flags.
        if unsafe { libc::tcgetattr(libc::STDIN_FILENO, previous.as_mut_ptr()) } != 0 {
            return Err(io::Error::last_os_error());
        }
        let previous = unsafe { previous.assume_init() };
        PREVIOUS_ECHO.store(
            previous.c_lflag & libc::ECHO != 0,
            std::sync::atomic::Ordering::Relaxed,
        );
        let mut action: libc::sigaction = unsafe { std::mem::zeroed() };
        action.sa_sigaction = interrupted_secret as *const () as libc::sighandler_t;
        unsafe {
            libc::sigemptyset(&mut action.sa_mask);
            libc::sigaddset(&mut action.sa_mask, libc::SIGINT);
            libc::sigaddset(&mut action.sa_mask, libc::SIGTERM);
        }
        let mut handlers = [
            (libc::SIGINT, unsafe { std::mem::zeroed() }),
            (libc::SIGTERM, unsafe { std::mem::zeroed() }),
        ];
        for index in 0..handlers.len() {
            let (signal, old) = &mut handlers[index];
            if unsafe { libc::sigaction(*signal, &action, old) } != 0 {
                let error = io::Error::last_os_error();
                for (signal, old) in &handlers[..index] {
                    unsafe {
                        libc::sigaction(*signal, old, std::ptr::null_mut());
                    }
                }
                return Err(error);
            }
        }
        let guard = Self { previous, handlers };
        let mut current = previous;
        current.c_lflag &= !libc::ECHO;
        if unsafe { libc::tcsetattr(libc::STDIN_FILENO, libc::TCSANOW, &current) } != 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(guard)
    }
}
#[cfg(unix)]
impl Drop for Echo {
    fn drop(&mut self) {
        unsafe {
            libc::tcsetattr(libc::STDIN_FILENO, libc::TCSANOW, &self.previous);
            for (signal, previous) in &self.handlers {
                libc::sigaction(*signal, previous, std::ptr::null_mut());
            }
        }
    }
}
#[cfg(not(unix))]
struct Echo;
#[cfg(not(unix))]
impl Echo {
    fn hide() -> io::Result<Self> {
        Err(io::Error::other(
            "Secret entry is unsupported in this terminal; use an env: reference with config set",
        ))
    }
}

fn detected() -> Value {
    let models = markitai_core::llm_capabilities(&config::defaults())
        .models
        .into_iter()
        .filter(|model| {
            model.split_once('/').is_none_or(|(provider, _)| {
                [
                    "openai",
                    "anthropic",
                    "gemini",
                    "deepseek",
                    "openrouter",
                    "azure",
                    "ollama",
                    "ollama_chat",
                ]
                .contains(&provider)
            })
        })
        .map(|model| json!({"model_name":"default", "litellm_params":{"model":model}}))
        .collect::<Vec<_>>();
    let mut result = json!({"output":{"dir":"./output"}, "llm":{"enabled":false}});
    if !models.is_empty() {
        result["llm"]["model_list"] = json!(models);
    }
    result
}

fn merge_models(existing: &mut Value, detected: &Value) -> CliResult<usize> {
    let incoming = detected
        .pointer("/llm/model_list")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    if incoming.is_empty() {
        return Ok(0);
    }
    if existing.get("llm").is_none() {
        existing["llm"] = json!({"enabled":false});
    }
    let section = existing["llm"].as_object_mut().ok_or_else(|| {
        runtime(text!(
            "Existing llm section must be an object",
            "已有的 llm 配置段必须是对象"
        ))
    })?;
    let list = section
        .entry("model_list")
        .or_insert_with(|| json!([]))
        .as_array_mut()
        .ok_or_else(|| {
            runtime(text!(
                "Existing model_list must be an array",
                "已有的 model_list 必须是数组"
            ))
        })?;
    let mut count = 0;
    for model in incoming {
        let name = &model["litellm_params"]["model"];
        if !list
            .iter()
            .any(|entry| &entry["litellm_params"]["model"] == name)
        {
            list.push(model.clone());
            count += 1;
        }
    }
    Ok(count)
}

pub(super) fn init(yes: bool, output: Option<&Path>, local: bool) -> CliResult<()> {
    if !yes {
        terminal()?;
    }
    let mut path = output.map(config::expand_home).unwrap_or_else(|| {
        if local {
            PathBuf::from("markitai.json")
        } else {
            config::home().join("config.json")
        }
    });
    if path.is_dir() {
        path = path.join("markitai.json");
    }
    let fresh = detected();
    let mut input = io::stdin().lock();
    let mut console = io::stderr().lock();
    if !yes {
        writeln!(
            console,
            "{}",
            text!(
                "Native configuration setup; no provider requests are made.",
                "原生配置初始化；不会向任何供应商发出请求。"
            )
        )
        .map_err(runtime)?;
        if let Some(models) = fresh.pointer("/llm/model_list").and_then(Value::as_array) {
            for model in models {
                let name = model["litellm_params"]["model"]
                    .as_str()
                    .unwrap_or_default();
                writeln!(
                    console,
                    "{}",
                    text!("Detected model: {name}", "检测到模型：{name}")
                )
                .map_err(runtime)?;
            }
        } else {
            writeln!(
                console,
                "{}",
                text!(
                    "No API model detected. LLM remains disabled.",
                    "未检测到 API 模型，LLM 保持关闭。"
                )
            )
            .map_err(runtime)?;
        }
        if output.is_none() && !local {
            let Some(choice) = choice_in(
                lang(),
                &mut input,
                &mut console,
                &text!(
                    "Save to: 1 user configuration, 2 ./markitai.json, q cancel [1]: ",
                    "保存到：1 用户配置，2 ./markitai.json，q 取消 [1]："
                ),
                &["1", "2"],
                "1",
            )?
            else {
                return Ok(());
            };
            if choice == "2" {
                path = PathBuf::from("markitai.json");
            }
        }
    }
    let original = read_bytes(&path)?;
    let action = if original.is_some() {
        if yes {
            "update".to_owned()
        } else {
            let shown = path.display();
            writeln!(
                console,
                "{}",
                text!("Configuration exists: {shown}", "配置文件已存在：{shown}")
            )
            .map_err(runtime)?;
            let Some(choice) = choice_in(
                lang(),
                &mut input,
                &mut console,
                &text!(
                    "1 update detected models, 2 overwrite, 3 keep [3]: ",
                    "1 追加检测到的模型，2 覆盖，3 保留 [3]："
                ),
                &["1", "2", "3"],
                "3",
            )?
            else {
                return Ok(());
            };
            match choice.as_str() {
                "1" => "update",
                "2" => "overwrite",
                _ => "keep",
            }
            .into()
        }
    } else {
        "create".into()
    };
    let shown = path.display();
    if action == "keep" {
        writeln!(console, "{}", text!("Kept {shown}", "已保留 {shown}")).map_err(runtime)?;
        return Ok(());
    }
    let mut existing;
    if action == "update" {
        existing = parse_config(original.as_deref())?;
        if merge_models(&mut existing, &fresh)? == 0 {
            println!(
                "{}",
                text!(
                    "Configuration already up to date: {shown}",
                    "配置已是最新：{shown}"
                )
            );
            return Ok(());
        }
    } else {
        existing = fresh;
    }
    config::validate(&existing).map_err(runtime)?;
    unchanged(&path, &original)?;
    write_config(&path, &existing)?;
    println!(
        "{}",
        if action == "update" {
            text!("Configuration updated: {shown}", "配置已更新：{shown}")
        } else {
            text!("Configuration created: {shown}", "配置已创建：{shown}")
        }
    );
    if action != "update" {
        // Guidance stays on stderr; stdout keeps the one-line result for scripts.
        let models = existing
            .pointer("/llm/model_list")
            .and_then(Value::as_array)
            .map_or(0, Vec::len);
        let llm = if models > 0 {
            text!(
                "Detected models are saved, but LLM processing stays off: add --llm to a conversion or run `markitai config set llm.enabled true`.",
                "已保存检测到的模型，但 LLM 处理仍处于关闭状态：转换时加上 --llm，或运行 `markitai config set llm.enabled true`。"
            )
        } else {
            text!(
                "No API model was detected: set a provider API key such as OPENAI_API_KEY, then run `markitai init -y` again to add it.",
                "未检测到 API 模型：请设置供应商的 API key（如 OPENAI_API_KEY），然后再次运行 `markitai init -y` 添加。"
            )
        };
        writeln!(
            console,
            "{}",
            text!(
                "Next steps:\n  markitai FILE -o DIR    Convert a document\n  markitai doctor         Check models and optional backends\n{llm}",
                "后续步骤：\n  markitai FILE -o DIR    转换文档\n  markitai doctor         检查模型和可选后端\n{llm}"
            )
        )
        .map_err(runtime)?;
    }
    Ok(())
}

/// A choice among `choices` (Enter takes `default`); the retry hint is in
/// English, as the other wizard text is.
pub(super) fn choice(
    input: &mut impl BufRead,
    output: &mut impl Write,
    label: &str,
    choices: &[&str],
    default: &str,
) -> CliResult<Option<String>> {
    choice_in(Lang::En, input, output, label, choices, default)
}

/// [`choice`] with the retry hint in the given language.
fn choice_in(
    lang: Lang,
    input: &mut impl BufRead,
    output: &mut impl Write,
    label: &str,
    choices: &[&str],
    default: &str,
) -> CliResult<Option<String>> {
    loop {
        let Some(value) = prompt(input, output, label)? else {
            return Ok(None);
        };
        let value = value.trim();
        if matches!(value, "q" | "\u{1b}") {
            return Ok(None);
        }
        if value.is_empty() {
            return Ok(Some(default.into()));
        }
        if choices.contains(&value) {
            return Ok(Some(value.into()));
        }
        let choices = choices.join(", ");
        writeln!(
            output,
            "{}",
            text!(lang => "Choose one of: {choices}", "请选择其中之一：{choices}")
        )
        .map_err(runtime)?;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn navigation_validates_before_saving_and_preserves_unknown_and_secret_fields() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.json");
        let raw = json!({"custom":{"keep":42},"image":{"quality":75},"fetch":{"jina":{"api_key":"never-print-me"}}});
        write_config(&path, &raw).unwrap();
        let mut input = io::Cursor::new(b"/imgql\n1\n500\n1\n85\nq\n");
        let mut output = Vec::new();
        edit_with(&path, &mut input, &mut output, false).unwrap();
        let saved: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        assert_eq!(saved["image"]["quality"], 85);
        assert_eq!(saved["custom"], raw["custom"]);
        assert_eq!(saved["fetch"], raw["fetch"]);
        let output = String::from_utf8(output).unwrap();
        assert!(output.contains("Invalid value:"));
        assert!(output.contains("Saved image.quality = 85"));
        assert!(!output.contains("never-print-me"));
    }
    #[test]
    fn cancelled_edits_and_null_or_empty_values_are_distinct() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.json");
        let mut output = Vec::new();
        edit_with(
            &path,
            &mut io::Cursor::new(b"output.dir\n:cancel\nq\n"),
            &mut output,
            false,
        )
        .unwrap();
        assert!(!path.exists());
        edit_with(
            &path,
            &mut io::Cursor::new(b"output.dir\n:empty\noutput.profile\nnull\nq\n"),
            &mut output,
            false,
        )
        .unwrap();
        let saved: Value = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
        assert_eq!(saved["output"]["dir"], "");
        assert!(saved["output"]["profile"].is_null());
    }
    #[test]
    fn update_models_is_additive_and_idempotent() {
        let mut existing = json!({"custom":1,"llm":{"enabled":true,"model_list":[{"model_name":"mine","litellm_params":{"model":"openai/existing","api_key":"env:PRIVATE"}}]}});
        let fresh = json!({"llm":{"model_list":[{"model_name":"default","litellm_params":{"model":"openai/existing"}},{"model_name":"default","litellm_params":{"model":"anthropic/new"}}]}});
        assert_eq!(merge_models(&mut existing, &fresh).unwrap(), 1);
        assert_eq!(merge_models(&mut existing, &fresh).unwrap(), 0);
        assert_eq!(existing["llm"]["enabled"], true);
        assert_eq!(existing["llm"]["model_list"][0]["model_name"], "mine");
        assert_eq!(
            existing["llm"]["model_list"][0]["litellm_params"]["api_key"],
            "env:PRIVATE"
        );
        assert_eq!(existing["custom"], 1);
    }
    #[test]
    fn scalar_catalog_omits_complex_sections_and_external_edits_block_save() {
        let list = settings(&config::defaults());
        assert!(list.iter().any(|s| s.key == "fetch.playwright.timeout"));
        assert!(!list.iter().any(|s| s.key.starts_with("prompts.")
            || s.key.contains("model_list")
            || s.key.ends_with("cookies")));
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.json");
        write_config(&path, &json!({})).unwrap();
        let (_, bytes) = read_config(&path).unwrap();
        write_config(&path, &json!({"custom":true})).unwrap();
        assert!(unchanged(&path, &bytes).is_err());
        assert_eq!(read_config(&path).unwrap().0["custom"], true);
    }
}
