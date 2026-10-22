//! Configuration-aware diagnostics; explicit browser repair uses the native installer.
use super::i18n::{Lang, lang};
use super::logging::{self, Shape};
use super::{CliResult, runtime};
use crate::term::{Mark, Style};
use indexmap::IndexMap;
use markitai_core::config;
use serde::Serialize;
use serde_json::Value;
#[path = "doctor_zh.rs"]
mod doctor_zh;
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
    /// Provider key variables set in the environment that no active model
    /// uses (`markitai init` leaves out a key whose default model another
    /// configured route already serves).
    #[serde(skip_serializing_if = "Option::is_none")]
    unused_api_keys: Option<Vec<String>>,
    #[serde(skip)]
    required: bool,
    /// Not ready yet, but it sets itself up the first time it is used (the OCR
    /// models download then), so the verdict does not call it unavailable.
    #[serde(skip)]
    first_use: bool,
    /// The remedy as the terminal shows it, with each command on its own line;
    /// empty when `install_hint` reads well as it is.
    #[serde(skip)]
    fix: Vec<Fix>,
}

/// One line of a remedy on the terminal.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Fix {
    Text(String),
    /// A command to copy, never wrapped.
    Command(String),
}

impl Fix {
    fn text(text: impl Into<String>) -> Self {
        Self::Text(text.into())
    }
    fn command(command: impl Into<String>) -> Self {
        Self::Command(command.into())
    }
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
            unused_api_keys: None,
            required: false,
            first_use: false,
            fix: Vec::new(),
        }
    }
    fn failed(&self) -> bool {
        self.required && self.status != "ok"
    }
    /// The remedy on the terminal: [`Check::fix`], or the hint as written.
    fn remedy(&self) -> Vec<Fix> {
        if self.fix.is_empty() {
            vec![Fix::text(&self.install_hint)]
        } else {
            self.fix.clone()
        }
    }
}

/// The report's sections in the order Python markitai shows them; the JSON
/// keeps the order of the checks.
const SECTIONS: [(&str, &str, &[&str]); 3] = [
    (
        "Optional Capabilities",
        "可选能力",
        &[
            "rapidocr",
            "vlm-ocr",
            "playwright",
            "libreoffice",
            "anydoc",
            "serve",
        ],
    ),
    (
        "LLM",
        "LLM",
        &[
            "llm-api",
            "vision-model",
            "claude-agent-sdk",
            "copilot-sdk",
            "chatgpt-runtime",
        ],
    ),
    (
        "Authentication",
        "认证状态",
        &["claude-agent-auth", "copilot-auth", "chatgpt-auth"],
    ),
];

pub(super) fn run(
    cfg: &Value,
    path: Option<&Path>,
    project: bool,
    json: bool,
    fix: bool,
) -> CliResult<i32> {
    let lang = lang();
    let style = Style::stdout();
    if !json {
        for line in heading(path, project, lang, style) {
            println!("{line}");
        }
    }
    let (browser, office) = std::thread::scope(|scope| {
        let browser = scope.spawn(markitai_core::browser_diagnostic);
        let office = scope.spawn(markitai_core::office_diagnostic);
        (browser.join(), office.join())
    });
    let mut browser = browser.map_err(|_| runtime("Browser diagnostic did not complete"))?;
    let office = office.map_err(|_| runtime("Office diagnostic did not complete"))?;
    let repair_missing = fix && !matches!(&browser, Ok(Some(_)));
    let mut repair_failed = Vec::new();
    if fix {
        logging::blank();
        say!(Shape::SECTION; "Repairing", "自动修复");
    }
    if repair_missing {
        say!(
            Shape::item(Mark::Action);
            "Installing the official Chrome headless shell in the private Markitai home...",
            "正在 Markitai 私有目录中安装官方 Chrome headless shell……"
        );
        match markitai_core::install_browser() {
            Ok(path) => {
                let shown = path.display();
                say!(
                    Shape::item(Mark::Success);
                    "Installed and verified {shown}",
                    "已安装并验证 {shown}"
                );
                browser = Ok(Some(path));
            }
            Err(error) => {
                repair_failed.push("Chromium (native CDP)");
                say!(
                    Shape::item(Mark::Error);
                    "Error: Browser repair failed: {error}",
                    "浏览器修复失败：{error}"
                );
            }
        }
    } else if fix {
        say!(
            Shape::item(Mark::Success);
            "Chromium launches successfully; no browser repair is needed.",
            "Chromium 可以正常启动，无需修复浏览器。"
        );
    }
    if fix && !repair_ocr_models(cfg) {
        repair_failed.push("Local OCR");
    }
    if fix {
        say!(
            Shape::item(Mark::Info);
            "Other diagnostic hints require manual action.",
            "其他诊断提示需要手动处理。"
        );
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
        for line in report(&checks, &repair_failed, lang, style) {
            println!("{line}");
        }
    }
    Ok(i32::from(failed || !repair_failed.is_empty()))
}

/// The title, then the version and configuration lines.
fn heading(path: Option<&Path>, project: bool, lang: Lang, style: Style) -> [String; 3] {
    let version = markitai_core::VERSION;
    let configured = path.map(|path| super::config_line(lang, path, project));
    let source = match path {
        Some(path) => path.display().to_string(),
        None => {
            let home = config::home().join("config.json");
            let home = home.display();
            text!(lang =>
                "defaults (no config file found; create one with 'markitai init' at {home} or ./markitai.json)",
                "默认配置（未找到配置文件；可运行 'markitai init' 在 {home} 或 ./markitai.json 创建）"
            )
        }
    };
    [
        style.title(&text!(lang => "System Check", "系统检查")),
        style.item(
            Mark::Info,
            &text!(lang => "Version: markitai {version} (native)", "版本：markitai {version}（原生）"),
        ),
        style.item(
            Mark::Info,
            &configured.unwrap_or_else(|| text!(lang => "Config: {source}", "配置文件：{source}")),
        ),
    ]
}

/// The checks by section, the remedies and the verdict, one string per line
/// ("" between blocks). Names, messages and hints are the JSON values and stay
/// in English; the frame follows the terminal language.
fn report(
    checks: &IndexMap<&'static str, Check>,
    repair_failed: &[&str],
    lang: Lang,
    style: Style,
) -> Vec<String> {
    let mut lines = Vec::new();
    // A check no section names is shown in the first one rather than lost.
    let listed = |key: &str| SECTIONS.iter().any(|(_, _, keys)| keys.contains(&key));
    let mut shown: Vec<&Check> = Vec::new();
    for (index, (english, chinese, keys)) in SECTIONS.iter().enumerate() {
        let members: Vec<&Check> = keys
            .iter()
            .filter_map(|key| checks.get(key))
            .chain(
                checks
                    .iter()
                    .filter(|(key, _)| index == 0 && !listed(key))
                    .map(|(_, check)| check),
            )
            .collect();
        if members.is_empty() {
            continue;
        }
        lines.push(String::new());
        lines.push(style.section(match lang {
            Lang::En => english,
            Lang::Zh => chinese,
        }));
        for check in members {
            lines.push(style.item(mark(check), &check_line(check, lang)));
            if let Some(unused) = &check.unused_api_keys {
                lines.push(style.detail(&match lang {
                    Lang::En => format!(
                        "Not used: {} (set, but no model in llm.model_list uses {})",
                        unused.join(", "),
                        if unused.len() == 1 { "it" } else { "them" }
                    ),
                    Lang::Zh => format!(
                        "未使用：{}（已设置，但 llm.model_list 中没有模型用到{}）",
                        unused.join("、"),
                        if unused.len() == 1 { "它" } else { "它们" }
                    ),
                }));
            }
            // An optional next step for a check that is ready.
            if check.status == "ok" && !check.install_hint.is_empty() {
                lines.push(style.detail(&localized(&check.install_hint, lang)));
            }
            shown.push(check);
        }
    }
    // One entry per distinct remedy: checks that share one are named together.
    let mut remedies: Vec<(String, Vec<Fix>)> = Vec::new();
    for check in shown
        .iter()
        .filter(|check| check.status != "ok" && !check.install_hint.is_empty())
    {
        let remedy = check.remedy();
        match remedies.iter_mut().find(|(_, fix)| *fix == remedy) {
            Some((names, _)) => {
                names.push_str(", ");
                names.push_str(check.name);
            }
            None => remedies.push((check.name.to_owned(), remedy)),
        }
    }
    let failed: Vec<&str> = shown
        .iter()
        .filter(|check| check.failed())
        .map(|check| check.name)
        .collect();
    if !remedies.is_empty() {
        lines.push(String::new());
        lines.push(style.section(&if failed.is_empty() {
            text!(lang => "To enable what is missing", "启用缺失的功能")
        } else {
            text!(lang => "To fix", "修复方法")
        }));
        let colon = match lang {
            Lang::En => ": ",
            Lang::Zh => "：",
        };
        for (names, remedy) in remedies {
            let mut rest = remedy.iter();
            match rest.next() {
                Some(Fix::Text(text)) => lines.push(style.item(
                    Mark::Info,
                    &format!("{names}{colon}{}", localized(text, lang)),
                )),
                Some(Fix::Command(command)) => {
                    lines.push(style.item(Mark::Info, &format!("{names}{}", colon.trim_end())));
                    lines.push(style.command(4, command));
                }
                None => lines.push(style.item(Mark::Info, &names)),
            }
            for line in rest {
                lines.push(match line {
                    Fix::Text(text) => style.line(4, None, &localized(text, lang), None),
                    Fix::Command(command) => style.command(4, command),
                });
            }
        }
    }
    lines.push(String::new());
    let (verdict, text) = verdict(&shown, &failed, repair_failed, lang);
    lines.push(style.summary(verdict, &text));
    lines
}

fn mark(check: &Check) -> Mark {
    if check.status == "ok" {
        Mark::Success
    } else if check.first_use && !check.failed() {
        // It works and sets itself up when used: information, not a warning.
        Mark::Info
    } else if check.failed() {
        Mark::Error
    } else {
        Mark::Warning
    }
}

/// "Name: message", noting a check the configuration requires. The name,
/// message and hint are the JSON values and stay in English; the note follows
/// the terminal language.
fn check_line(check: &Check, lang: Lang) -> String {
    let name = check.name;
    let message = &match lang {
        Lang::En => check.message.clone(),
        Lang::Zh => doctor_zh::zh(&check.message),
    };
    match (lang, check.required) {
        (Lang::En, false) => format!("{name}: {message}"),
        (Lang::En, true) => format!("{name}: {message} (required by the configuration)"),
        (Lang::Zh, false) => format!("{name}：{message}"),
        (Lang::Zh, true) => format!("{name}：{message}（配置要求）"),
    }
}

/// A message or remedy in the terminal language; the JSON values stay English.
fn localized(text: &str, lang: Lang) -> String {
    match lang {
        Lang::En => text.to_owned(),
        Lang::Zh => doctor_zh::zh(text),
    }
}

/// The closing line: the verdict the exit status encodes, with its mark.
fn verdict(
    shown: &[&Check],
    failed: &[&str],
    repair_failed: &[&str],
    lang: Lang,
) -> (Mark, String) {
    let separator = match lang {
        Lang::En => ", ",
        Lang::Zh => "、",
    };
    let pending = |check: &&&Check| check.status != "ok" && !check.failed();
    let missing: Vec<&str> = shown
        .iter()
        .filter(pending)
        .filter(|check| !check.first_use)
        .map(|check| check.name)
        .collect();
    let on_first_use: Vec<&str> = shown
        .iter()
        .filter(pending)
        .filter(|check| check.first_use)
        .map(|check| check.name)
        .collect();
    if !failed.is_empty() {
        let names = failed.join(separator);
        (
            Mark::Error,
            text!(lang => "Not ready: fix {names} first (see above)", "未就绪：请先修复 {names}（见上文）"),
        )
    } else if !repair_failed.is_empty() {
        let names = repair_failed.join(separator);
        (
            Mark::Error,
            text!(lang => "Repair failed: {names}", "修复失败：{names}"),
        )
    } else if !missing.is_empty() || !on_first_use.is_empty() {
        // Ready is a success, as in Python markitai: what is missing is
        // optional and already listed with its remedy above. What only needs
        // a download on first use is not missing.
        let mut segments = Vec::new();
        if !missing.is_empty() {
            let names = missing.join(separator);
            segments.push(
                text!(lang => "Not available (optional): {names}", "以下可选功能不可用：{names}"),
            );
        }
        if !on_first_use.is_empty() {
            let names = on_first_use.join(separator);
            segments
                .push(text!(lang => "Set up on first use: {names}", "首次使用时自动准备：{names}"));
        }
        let (lead, joint) = match lang {
            Lang::En => ("Ready. ", ". "),
            Lang::Zh => ("已就绪。", "；"),
        };
        (Mark::Success, format!("{lead}{}", segments.join(joint)))
    } else {
        (
            Mark::Success,
            text!(lang => "Ready. Everything is available.", "已就绪，所有功能均可用。"),
        )
    }
}

/// The optional local OCR check. An x86_64 build translated by Rosetta finds
/// the Vision API, but Vision text recognition fails there on every system
/// measured, so it is reported as a warning with the remedy.
fn local_ocr_check(available: bool, translated: bool) -> Check {
    let (status, message, hint) = match (available, translated) {
        (false, _) => (
            "missing",
            "No native local OCR backend is available on this platform",
            "Use a supported macOS host or configure a vision model for supported VLM workflows",
        ),
        (true, true) => (
            "warning",
            "macOS Vision API is present, but this x86_64 build runs under Rosetta on Apple silicon, where Vision text recognition fails",
            "Use the native arm64 build of markitai for local OCR",
        ),
        (true, false) => ("ok", "macOS Vision API is available", ""),
    };
    let mut check = Check::new(
        "Local OCR",
        "Local image and rendered-page text recognition",
        status,
        message,
        hint,
    );
    check.optional = Some(true);
    check
}

/// The optional local OCR check of the portable engine: whether the model
/// files the configured `ocr.lang` needs are installed. It never loads them.
fn portable_ocr_check(
    engine: &str,
    models: markitai_core::Result<Vec<markitai_core::LocalOcrModel>>,
) -> Check {
    let mut fix = Vec::new();
    let mut first_use = false;
    let (status, message, hint, path) = match models {
        Err(error) => (
            "warning",
            format!("Portable PaddleOCR engine ({engine}): {error}"),
            "Set ocr.lang to a language the portable OCR models read".to_owned(),
            None,
        ),
        Ok(models) => {
            use markitai_core::LocalOcrModelState as State;
            let folder = models
                .first()
                .and_then(|model| model.path.parent()?.parent())
                .map(Path::to_path_buf);
            let unsafe_paths: Vec<_> = models
                .iter()
                .filter(|model| model.state == State::Unsafe)
                .collect();
            let corrupt: Vec<_> = models
                .iter()
                .filter(|model| model.state == State::Corrupt)
                .collect();
            let missing: Vec<_> = models
                .iter()
                .filter(|model| model.state == State::Missing)
                .collect();
            if !unsafe_paths.is_empty() {
                let detail = unsafe_paths
                    .iter()
                    .map(|model| model.detail.as_deref().unwrap_or("unsafe model path"))
                    .collect::<Vec<_>>()
                    .join("; ");
                ("warning", format!("Portable PaddleOCR engine ({engine}): unsafe model path: {detail}"),
                    "Restore an ordinary owned private models directory/file; doctor --fix refuses unsafe paths before downloading".to_owned(), folder)
            } else if !corrupt.is_empty() {
                let detail = corrupt
                    .iter()
                    .map(|model| model.detail.as_deref().unwrap_or("damaged model file"))
                    .collect::<Vec<_>>()
                    .join("; ");
                fix = vec![
                    Fix::text(
                        "Replace only the damaged managed model files, after verifying each new size and SHA-256:",
                    ),
                    Fix::command("markitai doctor --fix"),
                ];
                ("warning", format!("Portable PaddleOCR engine ({engine}): {} damaged model file(s); {detail}; ordinary OCR will not overwrite them", corrupt.len()),
                    "Run markitai doctor --fix to replace only damaged managed model files, after verifying each new size and SHA-256".to_owned(), folder)
            } else if !missing.is_empty() {
                first_use = true;
                let bytes: u64 = missing.iter().map(|model| model.bytes).sum();
                fix = vec![
                    Fix::text(
                        "Download them now from the official PaddleOCR mirror (each verified by SHA-256):",
                    ),
                    Fix::command("markitai doctor --fix"),
                ];
                ("warning", format!("Portable PaddleOCR engine ({engine}): {} of the {} model files the configured ocr.lang needs ({:.1} MB) are not installed; the first OCR downloads them", missing.len(), models.len(), bytes as f64 / 1e6),
                    "Run markitai doctor --fix to download them now from the official PaddleOCR mirror (each verified by SHA-256)".to_owned(), folder)
            } else {
                (
                    "ok",
                    format!(
                        "Portable PaddleOCR engine ({engine}): the {} model files the configured ocr.lang needs are installed and match their SHA-256",
                        models.len()
                    ),
                    String::new(),
                    folder,
                )
            }
        }
    };
    let mut check = Check::new(
        "Local OCR",
        "Local image and rendered-page text recognition",
        status,
        message,
        hint,
    );
    check.optional = Some(true);
    check.first_use = first_use;
    check.path = path;
    check.fix = fix;
    check
}

/// The portable engine's name in the check: qppocr says that tract takes over
/// where it cannot read.
fn portable_engine() -> String {
    match markitai_core::local_ocr_engine() {
        "qppocr" => "qppocr, tract fallback".to_owned(),
        other => other.to_owned(),
    }
}

/// The local OCR check of the engine this process reads with.
fn local_ocr(cfg: &Value) -> Check {
    match markitai_core::local_ocr_models(cfg) {
        Ok(None) => local_ocr_check(
            markitai_core::local_ocr_available(),
            markitai_core::rosetta_translated(),
        ),
        Ok(Some(models)) => portable_ocr_check(&portable_engine(), Ok(models)),
        Err(error) => portable_ocr_check(&portable_engine(), Err(error)),
    }
}

/// `doctor --fix` for the portable OCR engine: download the model files the
/// configured `ocr.lang` needs that are missing or safely damaged. Returns
/// false for unsafe paths or failed repair; an engine that needs nothing returns true.
fn repair_ocr_models(cfg: &Value) -> bool {
    let models = match markitai_core::local_ocr_models(cfg) {
        Ok(None) => return true,
        Ok(Some(models)) => models,
        Err(error) => {
            say!(
                Shape::item(Mark::Error);
                "Error: Local OCR model check failed: {error}",
                "本地 OCR 模型检查失败：{error}"
            );
            return false;
        }
    };
    use markitai_core::LocalOcrModelState as State;
    let missing: Vec<_> = models
        .iter()
        .filter(|model| model.state != State::Ready)
        .collect();
    if missing.is_empty() {
        return true;
    }
    if let Some(unsafe_path) = models.iter().find(|model| model.state == State::Unsafe) {
        let detail = unsafe_path.detail.as_deref().unwrap_or("unsafe model path");
        say!(
            Shape::item(Mark::Error);
            "Error: Local OCR model repair refused: {detail}",
            "已拒绝修复本地 OCR 模型：{detail}"
        );
        return false;
    }
    let count = missing.len();
    let megabytes = format!(
        "{:.1}",
        missing.iter().map(|model| model.bytes).sum::<u64>() as f64 / 1e6
    );
    say!(
        Shape::item(Mark::Action);
        "Downloading {count} local OCR model file(s) ({megabytes} MB) from the official PaddleOCR mirror into the private Markitai home...",
        "正在从 PaddleOCR 官方镜像下载 {count} 个本地 OCR 模型文件（{megabytes} MB）到 Markitai 私有目录……"
    );
    match markitai_core::install_local_ocr_models(cfg) {
        Ok(paths) => {
            for path in paths {
                let shown = path.display();
                say!(
                    Shape::item(Mark::Success);
                    "Installed and verified {shown}",
                    "已安装并验证 {shown}"
                );
            }
            true
        }
        Err(error) => {
            say!(
                Shape::item(Mark::Error);
                "Error: Local OCR model download failed: {error}",
                "本地 OCR 模型下载失败：{error}"
            );
            false
        }
    }
}

/// The LibreOffice remedy on the terminal: the install command (one per
/// distribution family on Linux), each on its own line.
fn office_fix(platform: &str) -> Vec<Fix> {
    const NOTE: &str = "Office screenshots also require the native PDF page renderer";
    match platform {
        "windows" => vec![
            Fix::text(format!("Install LibreOffice ({NOTE}):")),
            Fix::command("winget install --id TheDocumentFoundation.LibreOffice --exact"),
        ],
        "macos" => vec![
            Fix::text(format!("Install LibreOffice ({NOTE}):")),
            Fix::command("brew install --cask libreoffice"),
        ],
        _ => vec![
            Fix::text(format!(
                "Install LibreOffice with the distribution's package manager and put soffice on PATH ({NOTE}):"
            )),
            Fix::command("sudo apt install libreoffice  # Debian, Ubuntu"),
            Fix::command("sudo dnf install libreoffice  # Fedora"),
            Fix::command("sudo pacman -S libreoffice-fresh  # Arch"),
        ],
    }
}

fn office_install_hint(platform: &str) -> &'static str {
    match platform {
        "windows" => {
            "Install LibreOffice: winget install --id TheDocumentFoundation.LibreOffice --exact; Office screenshots also require the native PDF page renderer"
        }
        "macos" => {
            "Install LibreOffice: brew install --cask libreoffice; Office screenshots also require the native PDF page renderer"
        }
        _ => {
            "Install LibreOffice using your distribution package manager and put soffice on PATH; Office screenshots also require the native PDF page renderer"
        }
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
    if browser.status == "ok"
        && browser
            .path
            .as_deref()
            .is_some_and(markitai_core::browser_diagnostic_leaves_app_untouched)
    {
        browser.message = "Installed browser app found (version check only)".into();
    }
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
    } else if browser.status != "ok" {
        browser.fix = vec![
            Fix::text("Install the official headless shell:"),
            Fix::command("markitai doctor --fix"),
            Fix::text(
                "or set MARKITAI_BROWSER_EXECUTABLE to an installed Chrome/Chromium executable.",
            ),
        ];
    }
    result.insert("playwright", browser);
    let mut office = backend(
        "LibreOffice",
        "Office page and slide rendering",
        office,
        "Installed, and it answers --version",
        office_install_hint(std::env::consts::OS),
    );
    if office.status != "ok" {
        office.fix = office_fix(std::env::consts::OS);
    }
    result.insert("libreoffice", office);
    result.insert("rapidocr", local_ocr(cfg));
    let mut legacy = Check::new(
        "Native Office readers",
        "Legacy Office text extraction (.doc/.ppt)",
        "ok",
        "Old .doc and .ppt files are read by built-in readers; nothing to install",
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
            newer_release(&mut adapter, "Copilot CLI", &auth);
            let mut identity = Check::new(
                "Copilot authentication", "Configured local provider",
                if auth.authenticated { "ok" } else { "error" },
                auth.error.unwrap_or_else(|| "Official runtime reports an authenticated account; model access was not probed".into()),
                "Run markitai auth copilot login",
            );
            identity.required = true;
            identity.fix = vec![Fix::command("markitai auth copilot login")];
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
            newer_release(&mut adapter, "Claude Code", &auth);
            let mut identity=Check::new("Claude subscription authentication", "Configured local provider", if auth.authenticated {"ok"} else {"error"},
                auth.error.unwrap_or_else(|| "Official runtime reports an authenticated subscription; model access was not probed".into()), "Run markitai auth claude login");
            identity.required = true;
            identity.fix = vec![Fix::command("markitai auth claude login")];
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
                "Install official Codex 0.159.0 or later; chatgpt/<model> takes any model it offers",
            );
            adapter.required = true;
            newer_release(&mut adapter, "Codex", &auth);
            let mut identity = Check::new(
                "ChatGPT subscription authentication", "Configured local provider",
                if auth.authenticated { "ok" } else { "error" },
                auth.error.unwrap_or_else(|| "Official runtime reports a ChatGPT login; model entitlement and inference were not probed".into()),
                "Run markitai auth chatgpt login",
            );
            identity.required = true;
            identity.fix = vec![Fix::command("markitai auth chatgpt login")];
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
        .is_some_and(|value| config::env_opt_out(value));
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
        // Child stderr and arbitrary exception strings may contain credentials;
        // only the fixed text of a known cause is shown.
        Err(error)
            if error
                .to_string()
                .contains(markitai_core::BROWSER_ROOT_SANDBOX_REFUSED) =>
        {
            Check::new(
                name,
                description,
                "warning",
                markitai_core::BROWSER_ROOT_SANDBOX_REFUSED,
                hint,
            )
        }
        Err(_) => Check::new(
            name,
            description,
            "warning",
            "The installed backend could not complete its bounded startup check",
            hint,
        ),
    }
}

/// A runtime newer than the release its adapter was verified with is used:
/// the check stays ready and says so.
fn newer_release(check: &mut Check, runtime: &str, auth: &markitai_core::subscription::AuthStatus) {
    let details = &auth.details;
    if details["newer_than_verified"] == true
        && let (Some(installed), Some(verified)) = (
            details["cli_version"].as_str(),
            details["supported_from"].as_str(),
        )
    {
        check.message =
            markitai_core::subscription::newer_release_message(runtime, installed, verified);
    }
}

/// How to set a key variable, in the user's shell or the `.env` file.
fn set_hint(variable: &str) -> String {
    format!(
        "Set it with: {} (or add {variable}=your_key to {})",
        markitai_core::llm_keys::set_command(variable),
        markitai_core::llm_keys::dotenv_file()
    )
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
            _ if config::llm_provider_supported(provider) => {
                providers.insert(provider);
            }
            _ => unsupported = true,
        }
    }
    // The rule a conversion applies before sending anything.
    let (missing, usable) = markitai_core::llm_keys::check_with(cfg, env);
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
    // Keys of the built-in providers that are set but used by no active model.
    let unused: Vec<String> = ["anthropic", "openai", "gemini", "deepseek", "openrouter"]
        .into_iter()
        .filter(|provider| !active.is_empty() && !providers.contains(provider))
        .filter_map(markitai_core::provider_management::provider_entry)
        .flat_map(|entry| entry.key_variables)
        .filter(|name| {
            env.get(**name)
                .is_some_and(|value| !value.trim().is_empty())
        })
        .map(|name| (*name).to_owned())
        .collect();
    check.unused_api_keys = (!unused.is_empty()).then_some(unused);
    // Without llm.model_list, conversions fall back to MODEL or provider API
    // keys in the environment; report what --llm would actually use.
    let detected = models
        .is_empty()
        .then(|| markitai_core::llm_capabilities(cfg))
        .filter(|detected| !detected.models.is_empty());
    if let Some(detected) = detected {
        let names = detected.models.join(", ");
        let unset = missing.first().and_then(|item| item.variable.as_deref());
        if detected.routable {
            check.message = format!(
                "No llm.model_list; --llm uses {names} detected from MODEL or provider API keys in the environment"
            );
            check.install_hint =
                "Optional: run markitai init to save the detected model in a configuration file"
                    .into();
        } else if let Some(variable) = unset {
            check.status = "warning";
            check.message = format!(
                "No llm.model_list; the environment names {names}, but {variable} is not set"
            );
            check.install_hint = set_hint(variable);
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
        let mut variables: Vec<&str> = Vec::new();
        for variable in missing.iter().filter_map(|item| item.variable.as_deref()) {
            if !variables.contains(&variable) {
                variables.push(variable);
            }
        }
        let named = if variables.is_empty() {
            "an API key".to_owned()
        } else {
            variables.join(", ")
        };
        if usable {
            check.status = "warning";
            check.message = format!(
                "Not set: {named}; {} model(s) left out, {} still used",
                missing.len(),
                active.len().saturating_sub(missing.len())
            );
        } else {
            check.status = "error";
            check.message = format!(
                "Not set: {named}; no model can be used without {}",
                if variables.len() > 1 { "them" } else { "it" }
            );
        }
        check.install_hint = variables.first().map_or_else(
            || "Set litellm_params.api_key on the model or its provider connection".into(),
            |variable| set_hint(variable),
        );
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

    /// Chromium run as root exits at once; doctor names that cause (a fixed
    /// text) instead of a startup check that failed for no stated reason,
    /// and still shows nothing else of the error.
    #[test]
    fn a_browser_refused_as_root_is_named() {
        let cfg = config::normalize(&json!({})).unwrap();
        let refused = format!(
            "{} (secret-child-error)",
            markitai_core::BROWSER_ROOT_SANDBOX_REFUSED
        );
        let named = checks(
            &cfg,
            None,
            &HashMap::new(),
            Err(markitai_core::Error::Fetch(refused)),
            Ok(None),
            Vec::new(),
        );
        let browser = &named["playwright"];
        assert_eq!(browser.message, markitai_core::BROWSER_ROOT_SANDBOX_REFUSED);
        assert!(
            !serde_json::to_string(browser)
                .unwrap()
                .contains("secret-child-error")
        );
        let other = checks(
            &cfg,
            None,
            &HashMap::new(),
            Err(markitai_core::Error::Fetch("exited early".into())),
            Ok(None),
            Vec::new(),
        );
        assert_eq!(
            other["playwright"].message,
            "The installed backend could not complete its bounded startup check"
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
    fn local_ocr_under_rosetta_is_an_optional_warning_with_the_remedy() {
        let native = local_ocr_check(true, false);
        assert_eq!((native.status, native.install_hint.as_str()), ("ok", ""));
        let translated = local_ocr_check(true, true);
        assert_eq!(translated.status, "warning");
        assert!(translated.message.contains("Rosetta"));
        assert!(translated.install_hint.contains("native arm64 build"));
        assert_eq!(local_ocr_check(false, true).status, "missing");
        for check in [native, translated, local_ocr_check(false, false)] {
            assert_eq!(check.optional, Some(true));
            assert!(!check.failed());
        }
    }

    #[test]
    fn portable_ocr_reports_missing_models_with_the_fix_and_stays_optional() {
        let model = |installed| {
            markitai_core::LocalOcrModel::new(
                "model".into(),
                PathBuf::from("/home/models/ocr/model-0123abcd/model.onnx"),
                2_000_000,
                if installed {
                    markitai_core::LocalOcrModelState::Ready
                } else {
                    markitai_core::LocalOcrModelState::Missing
                },
                None,
            )
        };
        let ready = portable_ocr_check("tract", Ok(vec![model(true), model(true)]));
        assert_eq!(ready.status, "ok");
        assert!(ready.message.contains("PaddleOCR") && ready.install_hint.is_empty());
        assert_eq!(ready.path, Some(PathBuf::from("/home/models/ocr")));
        let missing = portable_ocr_check("tract", Ok(vec![model(true), model(false)]));
        assert_eq!(missing.status, "warning");
        assert!(
            missing.message.contains("1 of the 2 model files")
                && missing.message.contains("(2.0 MB)"),
            "{}",
            missing.message
        );
        assert!(missing.install_hint.contains("markitai doctor --fix"));
        // Models that only download on first use are not an unavailable
        // capability; damaged files and a wrong language are.
        assert!(missing.first_use && !ready.first_use);
        let wrong = portable_ocr_check(
            "tract",
            Err(markitai_core::Error::Unsupported(
                "ocr.lang is not a language the portable OCR models read (xx)".into(),
            )),
        );
        assert_eq!(wrong.status, "warning");
        assert!(wrong.message.contains("(xx)"));
        for check in [ready, missing, wrong] {
            assert_eq!(check.optional, Some(true));
            assert!(!check.failed());
        }
    }

    #[test]
    fn portable_ocr_names_damaged_and_unsafe_files_without_a_false_language_hint() {
        use markitai_core::{LocalOcrModel, LocalOcrModelState as State};
        let model = |state| {
            LocalOcrModel::new(
                "fixture".into(),
                PathBuf::from("/home/models/ocr/fixture/model.onnx"),
                10,
                state,
                Some("/home/models/ocr/fixture/model.onnx: exact path reason".into()),
            )
        };
        let corrupt = portable_ocr_check("tract", Ok(vec![model(State::Corrupt)]));
        assert_eq!(corrupt.status, "warning");
        assert!(
            corrupt.message.contains("damaged") && corrupt.message.contains("exact path reason")
        );
        assert!(corrupt.install_hint.contains("doctor --fix"));
        let unsafe_path = portable_ocr_check("tract", Ok(vec![model(State::Unsafe)]));
        assert!(
            unsafe_path.message.contains("unsafe model path")
                && unsafe_path.message.contains("exact path reason")
        );
        assert!(!unsafe_path.install_hint.contains("ocr.lang"));
        assert!(unsafe_path.install_hint.contains("refuses unsafe paths"));
    }

    fn checks_of(states: &[(&'static str, &'static str, bool)]) -> IndexMap<&'static str, Check> {
        states
            .iter()
            .map(|&(name, status, required)| {
                let mut check = Check::new(name, "description", status, "message", "");
                check.required = required;
                (name, check)
            })
            .collect()
    }

    /// Every message and remedy of every check the doctor builds without a
    /// subscription runtime or a network has a Chinese wording: a changed
    /// English string that falls out of the table fails here, instead of
    /// showing as English in the Chinese terminal.
    #[test]
    fn every_check_message_and_remedy_has_a_chinese_wording() {
        use markitai_core::{LocalOcrModel, LocalOcrModelState as State};
        let cjk = |text: &str| text.chars().any(|c| ('\u{4e00}'..='\u{9fff}').contains(&c));
        let mut seen: Vec<Check> = Vec::new();
        // The configuration-dependent checks, with every backend outcome.
        let off = config::normalize(&json!({"llm":{"model_list":[{"model_name":"off","litellm_params":{"model":"openai/test","weight":0,"api_key":"env:ABSENT_ROUND8"}}]}})).unwrap();
        let absent = config::normalize(&json!({"llm":{"model_list":[{"model_name":"m","litellm_params":{"model":"openai/test","api_key":"env:ABSENT_ROUND8"}}]}})).unwrap();
        let plain = config::normalize(&json!({})).unwrap();
        let named = HashMap::from([("MODEL".to_owned(), "openai/gpt-x".to_owned())]);
        let named_without_key = HashMap::from([("MODEL".to_owned(), "openai/gpt-x".to_owned())]);
        // Not found, found, or a probe that failed.
        let outcome = |kind: u8| -> markitai_core::Result<Option<PathBuf>> {
            match kind {
                0 => Ok(None),
                1 => Ok(Some(PathBuf::from("/usr/bin/tool"))),
                _ => Err(markitai_core::Error::Unsupported("x".into())),
            }
        };
        for (cfg, env) in [
            (&plain, HashMap::new()),
            (&off, HashMap::new()),
            (&absent, HashMap::new()),
            (&plain, named),
            (&plain, named_without_key),
        ] {
            for browser in 0..3 {
                for office in 0..3 {
                    seen.extend(
                        checks(
                            cfg,
                            None,
                            &env,
                            outcome(browser),
                            outcome(office),
                            Vec::new(),
                        )
                        .into_values(),
                    );
                }
            }
        }
        // The local OCR states.
        for (available, translated) in [(false, false), (true, true), (true, false)] {
            seen.push(local_ocr_check(available, translated));
        }
        let model = |state, detail: Option<&str>| {
            LocalOcrModel::new(
                "m".into(),
                PathBuf::from("/home/models/ocr/m-0123/model.onnx"),
                2_000_000,
                state,
                detail.map(str::to_owned),
            )
        };
        for models in [
            vec![model(State::Ready, None)],
            vec![model(State::Missing, None), model(State::Ready, None)],
            vec![model(State::Corrupt, Some("size differs"))],
            vec![model(State::Unsafe, Some("a link"))],
        ] {
            seen.push(portable_ocr_check("tract", Ok(models)));
        }
        seen.push(portable_ocr_check(
            "tract",
            Err(markitai_core::Error::Unsupported(
                "ocr.lang is not a language the portable OCR models read (xx)".into(),
            )),
        ));
        for platform in ["windows", "macos", "linux"] {
            let mut office = Check::new(
                "LibreOffice",
                "d",
                "missing",
                "No installed executable was found",
                office_install_hint(platform),
            );
            office.fix = office_fix(platform);
            seen.push(office);
        }
        assert!(seen.len() > 100);
        // Texts that are a path, a command or a name the table need not translate.
        let exempt = |text: &str| {
            text.is_empty()
                || text.starts_with("markitai ")
                || text.starts_with("sudo ")
                || text.starts_with("brew ")
                || text.starts_with("winget ")
        };
        let mut missing = std::collections::BTreeSet::new();
        for check in &seen {
            let mut texts = vec![check.message.clone(), check.install_hint.clone()];
            texts.extend(check.fix.iter().filter_map(|fix| match fix {
                Fix::Text(text) => Some(text.clone()),
                Fix::Command(_) => None,
            }));
            for text in texts {
                if !exempt(&text) && !cjk(&doctor_zh::zh(&text)) {
                    missing.insert(text);
                }
            }
        }
        assert!(missing.is_empty(), "no Chinese wording for: {missing:#?}");
    }

    #[test]
    fn verdicts_state_the_same_outcome_in_both_languages() {
        let verdict_of = |states: &[(&'static str, &'static str, bool)], repair: &[&str], lang| {
            let checks = checks_of(states);
            let shown: Vec<&Check> = checks.values().collect();
            let failed: Vec<&str> = shown
                .iter()
                .filter(|check| check.failed())
                .map(|check| check.name)
                .collect();
            verdict(&shown, &failed, repair, lang)
        };
        let all = [("A", "ok", true), ("B", "ok", false)];
        assert_eq!(
            verdict_of(&all, &[], Lang::En),
            (Mark::Success, "Ready. Everything is available.".into())
        );
        assert_eq!(
            verdict_of(&all, &[], Lang::Zh),
            (Mark::Success, "已就绪，所有功能均可用。".into())
        );
        // Optional checks that are not ready leave the verdict a success
        // that names them.
        let optional = [
            ("A", "ok", false),
            ("B", "warning", false),
            ("C", "missing", false),
        ];
        assert_eq!(
            verdict_of(&optional, &[], Lang::En),
            (
                Mark::Success,
                "Ready. Not available (optional): B, C".into()
            )
        );
        assert_eq!(
            verdict_of(&optional, &[], Lang::Zh),
            (Mark::Success, "已就绪。以下可选功能不可用：B、C".into())
        );
        // A check that sets itself up on first use is named apart from the
        // unavailable ones, and alone it makes no "not available" claim.
        let verdict_with_first_use = |lang| {
            let mut checks = checks_of(&[("A", "ok", false), ("L", "missing", false)]);
            let mut ocr = Check::new("Local OCR", "description", "warning", "message", "");
            ocr.first_use = true;
            checks.insert("Local OCR", ocr);
            let shown: Vec<&Check> = checks.values().collect();
            verdict(&shown, &[], &[], lang)
        };
        assert_eq!(
            verdict_with_first_use(Lang::En),
            (
                Mark::Success,
                "Ready. Not available (optional): L. Set up on first use: Local OCR".into()
            )
        );
        assert_eq!(
            verdict_with_first_use(Lang::Zh),
            (
                Mark::Success,
                "已就绪。以下可选功能不可用：L；首次使用时自动准备：Local OCR".into()
            )
        );
        let mut only = checks_of(&[("A", "ok", false)]);
        let mut ocr = Check::new("Local OCR", "description", "warning", "message", "");
        ocr.first_use = true;
        only.insert("Local OCR", ocr);
        let shown: Vec<&Check> = only.values().collect();
        assert_eq!(
            verdict(&shown, &[], &[], Lang::En).1,
            "Ready. Set up on first use: Local OCR"
        );
        let blocked = [("A", "missing", true), ("B", "error", true)];
        assert_eq!(
            verdict_of(&blocked, &[], Lang::En),
            (Mark::Error, "Not ready: fix A, B first (see above)".into())
        );
        assert_eq!(
            verdict_of(&blocked, &[], Lang::Zh),
            (Mark::Error, "未就绪：请先修复 A、B（见上文）".into())
        );
        assert_eq!(
            verdict_of(&all, &["Local OCR"], Lang::En),
            (Mark::Error, "Repair failed: Local OCR".into())
        );
    }

    /// A report with every kind of line: ready, optional and required checks,
    /// a detail, a shared remedy, commands and a verdict.
    fn sample() -> IndexMap<&'static str, Check> {
        let mut checks = IndexMap::new();
        let mut browser = Check::new(
            "Chromium (native CDP)",
            "d",
            "missing",
            "No installed executable was found",
            "Run markitai doctor --fix to install the official headless shell",
        );
        browser.fix = vec![
            Fix::text("Install the official headless shell:"),
            Fix::command("markitai doctor --fix"),
        ];
        checks.insert("playwright", browser);
        let mut office = Check::new(
            "LibreOffice",
            "d",
            "missing",
            "No installed executable was found",
            "x",
        );
        office.fix = office_fix("linux");
        checks.insert("libreoffice", office);
        checks.insert(
            "rapidocr",
            Check::new(
                "Local OCR",
                "d",
                "ok",
                "Portable PaddleOCR engine (tract): the 4 model files are verified",
                "",
            ),
        );
        checks.insert(
            "llm-api",
            Check::new(
                "LLM API",
                "d",
                "ok",
                "No llm.model_list; --llm uses openai/gpt-6-luna detected from provider API keys",
                "Optional: run markitai init to save the detected model in a configuration file",
            ),
        );
        let hint = "Configure a supported vision model";
        checks.insert(
            "vision-model",
            Check::new(
                "Vision Model",
                "d",
                "warning",
                "No routable vision model is configured",
                hint,
            ),
        );
        checks.insert(
            "vlm-ocr",
            Check::new(
                "VLM OCR",
                "d",
                "warning",
                "Unavailable without a routable vision model",
                hint,
            ),
        );
        let mut auth = Check::new(
            "Claude subscription authentication",
            "d",
            "error",
            "Claude Code is not signed in",
            "Run markitai auth claude login",
        );
        auth.required = true;
        auth.fix = vec![Fix::command("markitai auth claude login")];
        checks.insert("claude-agent-auth", auth);
        checks
    }

    #[test]
    fn keys_no_active_model_uses_are_listed() {
        let cfg = config::normalize(&json!({"llm":{"model_list":[
            {"model_name":"default","litellm_params":{"model":"deepseek/deepseek-flash"}}]}}))
        .unwrap();
        let env = HashMap::from([
            ("DEEPSEEK_API_KEY".to_owned(), "fake-test-key".to_owned()),
            ("OPENAI_API_KEY".to_owned(), "fake-test-key".to_owned()),
            ("GOOGLE_API_KEY".to_owned(), "fake-test-key".to_owned()),
            ("OR_API_KEY".to_owned(), " ".to_owned()),
        ]);
        let checks = checks(&cfg, None, &env, Ok(None), Ok(None), Vec::new());
        let llm = &checks["llm-api"];
        assert_eq!(
            llm.unused_api_keys.as_deref(),
            Some(&["OPENAI_API_KEY".to_owned(), "GOOGLE_API_KEY".to_owned()][..])
        );
        let plain = Style {
            color: false,
            unicode: true,
            width: None,
            bright: false,
        };
        let english = report(&checks, &[], Lang::En, plain).join("\n");
        assert!(
            english.contains("Not used: OPENAI_API_KEY, GOOGLE_API_KEY (set, but no model in llm.model_list uses them)"),
            "{english}"
        );
        let chinese = report(&checks, &[], Lang::Zh, plain).join("\n");
        assert!(
            chinese.contains("未使用：OPENAI_API_KEY、GOOGLE_API_KEY（已设置，但 llm.model_list 中没有模型用到它们）"),
            "{chinese}"
        );
        // Without a model list, the keys choose the models: none is unused.
        let plain_cfg = config::normalize(&json!({})).unwrap();
        let checks = super::checks(&plain_cfg, None, &env, Ok(None), Ok(None), Vec::new());
        assert!(checks["llm-api"].unused_api_keys.is_none());
    }

    #[test]
    fn the_report_renders_sections_remedies_and_the_verdict_in_one_style() {
        let plain = Style {
            color: false,
            unicode: true,
            width: None,
            bright: false,
        };
        let lines = |style| {
            let mut lines =
                heading(Some(Path::new("/cfg/config.json")), false, Lang::En, style).to_vec();
            lines.extend(report(&sample(), &[], Lang::En, style));
            lines.join("\n")
        };
        let version = markitai_core::VERSION;
        assert_eq!(
            lines(plain),
            format!(
                "◆ System Check
  • Version: markitai {version} (native)
  • Config: /cfg/config.json

Optional Capabilities
  ✓ Local OCR: Portable PaddleOCR engine (tract): the 4 model files are verified
  ! VLM OCR: Unavailable without a routable vision model
  ! Chromium (native CDP): No installed executable was found
  ! LibreOffice: No installed executable was found

LLM
  ✓ LLM API: No llm.model_list; --llm uses openai/gpt-6-luna detected from provider API keys
    Optional: run markitai init to save the detected model in a configuration file
  ! Vision Model: No routable vision model is configured

Authentication
  ✗ Claude subscription authentication: Claude Code is not signed in (required by the configuration)

To fix
  • VLM OCR, Vision Model: Configure a supported vision model
  • Chromium (native CDP): Install the official headless shell:
    markitai doctor --fix
  • LibreOffice: Install LibreOffice with the distribution's package manager and put soffice on PATH (Office screenshots also require the native PDF page renderer):
    sudo apt install libreoffice  # Debian, Ubuntu
    sudo dnf install libreoffice  # Fedora
    sudo pacman -S libreoffice-fresh  # Arch
  • Claude subscription authentication:
    markitai auth claude login

✗ Not ready: fix Claude subscription authentication first (see above)"
            )
        );
        // With colour: the marks and commands in their colours, structure bold,
        // details gray; the text itself stays uncoloured.
        let colored = lines(Style {
            color: true,
            ..plain
        });
        for expected in [
            "\x1b[36m◆\x1b[0m \x1b[1mSystem Check\x1b[0m\n",
            "\n\x1b[1mOptional Capabilities\x1b[0m\n",
            "\n  \x1b[32m✓\x1b[0m Local OCR: ",
            "\n  \x1b[33m!\x1b[0m VLM OCR: ",
            "\n  \x1b[31m✗\x1b[0m Claude subscription authentication: ",
            "\n    \x1b[90mOptional: run markitai init",
            "\n  \x1b[90m•\x1b[0m Chromium (native CDP): Install the official headless shell:\n    \x1b[36mmarkitai doctor --fix\x1b[0m\n",
            "\n\x1b[1mTo fix\x1b[0m\n",
            "\n\x1b[31m✗\x1b[0m Not ready: fix ",
        ] {
            assert!(colored.contains(expected), "{expected:?} in\n{colored}");
        }
        assert_eq!(
            colored
                .replace("\x1b[0m", "")
                .split("\x1b[")
                .map(|part| part.split_once('m').map_or(part, |(_, rest)| rest))
                .collect::<String>(),
            lines(plain)
        );
        // ASCII marks at 40 columns: wrapped under the text, commands whole.
        let narrow = lines(Style {
            unicode: false,
            width: Some(40 + crate::term::EDGE),
            ..plain
        });
        assert!(narrow.starts_with("* System Check\n"), "{narrow}");
        assert!(
            narrow.contains("\n  + Local OCR: Portable PaddleOCR engine\n    (tract): the 4 model files are\n    verified\n"),
            "{narrow}"
        );
        assert!(
            narrow.contains("\n    sudo pacman -S libreoffice-fresh  # Arch\n"),
            "{narrow}"
        );
        assert!(
            narrow.ends_with(
                "\nx Not ready: fix Claude subscription\n  authentication first (see above)"
            ),
            "{narrow}"
        );
        assert!(
            narrow
                .lines()
                .filter(|line| !line.trim_start().starts_with("sudo "))
                .all(|line| line.chars().count() <= 40),
            "{narrow}"
        );
    }

    #[test]
    fn office_install_hints_name_the_platform_install_method() {
        assert!(
            office_install_hint("windows")
                .contains("winget install --id TheDocumentFoundation.LibreOffice --exact")
        );
        assert!(!office_install_hint("windows").contains("macOS"));
        assert!(office_install_hint("macos").contains("brew install --cask libreoffice"));
        assert!(office_install_hint("linux").contains("package manager"));
    }

    #[test]
    fn missing_libreoffice_has_the_platform_install_hint_and_is_nonblocking() {
        let checks = checks(
            &config::defaults(),
            None,
            &HashMap::new(),
            Ok(None),
            Ok(None),
            Vec::new(),
        );
        let office = &checks["libreoffice"];
        assert_eq!(office.status, "missing");
        assert!(!office.required);
        assert!(!office.failed());
        assert_eq!(office.path, None);
        assert_eq!(office.optional, None);
        let expected = match std::env::consts::OS {
            "windows" => {
                "Install LibreOffice: winget install --id TheDocumentFoundation.LibreOffice --exact; Office screenshots also require the native PDF page renderer"
            }
            "macos" => {
                "Install LibreOffice: brew install --cask libreoffice; Office screenshots also require the native PDF page renderer"
            }
            _ => {
                "Install LibreOffice using your distribution package manager and put soffice on PATH; Office screenshots also require the native PDF page renderer"
            }
        };
        assert_eq!(office.install_hint, expected);
        let value = serde_json::to_value(office).unwrap();
        assert_eq!(value["status"], "missing");
        assert_eq!(value["install_hint"], expected);
        for field in ["optional", "path", "required", "available"] {
            assert!(value.get(field).is_none(), "{field}: {value}");
        }
    }

    #[test]
    fn check_lines_translate_only_the_requirement_note() {
        let states = checks_of(&[("A", "ok", false), ("C", "missing", true)]);
        let lines = |lang| {
            states
                .values()
                .map(|check| check_line(check, lang))
                .collect::<Vec<_>>()
        };
        assert_eq!(lines(Lang::Zh), ["A：message", "C：message（配置要求）"]);
        assert_eq!(
            lines(Lang::En),
            ["A: message", "C: message (required by the configuration)"]
        );
        let marks: Vec<_> = checks_of(&[
            ("A", "ok", true),
            ("B", "warning", false),
            ("C", "error", true),
        ])
        .values()
        .map(mark)
        .collect();
        assert_eq!(marks, [Mark::Success, Mark::Warning, Mark::Error]);
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
