//! The Chinese of the doctor's check messages and remedies. The messages are
//! the English JSON values; the terminal shows these in `zh`, and any text
//! this table does not know (a path, an error from a library) stays as it is.

/// Static texts and their Chinese.
const KNOWN: &[(&str, &str)] = &[
    ("No native local OCR backend is available on this platform", "此平台没有可用的本机 OCR 后端"),
    ("Use a supported macOS host or configure a vision model for supported VLM workflows", "请使用受支持的 macOS 主机，或为支持的 VLM 流程配置视觉模型"),
    ("macOS Vision API is present, but this x86_64 build runs under Rosetta on Apple silicon, where Vision text recognition fails", "已有 macOS Vision API，但此 x86_64 版本在 Apple 芯片上通过 Rosetta 运行，Vision 文字识别在这种情况下会失败"),
    ("Use the native arm64 build of markitai for local OCR", "请使用 markitai 的原生 arm64 版本进行本地 OCR"),
    ("macOS Vision API is available", "macOS Vision API 可用"),
    ("Set ocr.lang to a language the portable OCR models read", "请把 ocr.lang 设为便携 OCR 模型能识别的语言"),
    ("Restore an ordinary owned private models directory/file; doctor --fix refuses unsafe paths before downloading", "请恢复为普通的、归你所有的私有模型目录或文件；doctor --fix 在下载前会拒绝不安全的路径"),
    ("Replace only the damaged managed model files, after verifying each new size and SHA-256:", "只替换已损坏的受管模型文件，并先校验每个新文件的大小和 SHA-256："),
    ("Run markitai doctor --fix to replace only damaged managed model files, after verifying each new size and SHA-256", "运行 markitai doctor --fix，只替换已损坏的受管模型文件，并先校验每个新文件的大小和 SHA-256"),
    ("Download them now from the official PaddleOCR mirror (each verified by SHA-256):", "现在从 PaddleOCR 官方镜像下载（每个文件都校验 SHA-256）："),
    ("Run markitai doctor --fix to download them now from the official PaddleOCR mirror (each verified by SHA-256)", "运行 markitai doctor --fix，现在从 PaddleOCR 官方镜像下载（每个文件都校验 SHA-256）"),
    ("Launch and CDP connection succeeded in a private profile; no remote page was opened", "已在私有配置中启动并连上 CDP；没有打开任何远程页面"),
    ("Run markitai doctor --fix to install the official headless shell, or set MARKITAI_BROWSER_EXECUTABLE to an installed Chrome/Chromium executable", "运行 markitai doctor --fix 安装官方无头浏览器，或把 MARKITAI_BROWSER_EXECUTABLE 设为已安装的 Chrome/Chromium 可执行文件"),
    ("Installed browser app found (version check only)", "已找到已安装的浏览器应用（只检查了版本）"),
    ("MARKITAI_BROWSER_EXECUTABLE is set, so only that path is tried and doctor --fix will not replace it: point it at a working Chrome/Chromium executable, or unset it to use an installed browser or markitai doctor --fix", "已设置 MARKITAI_BROWSER_EXECUTABLE，因此只尝试该路径，doctor --fix 也不会替换它：请指向可用的 Chrome/Chromium 可执行文件，或取消设置以使用已安装的浏览器或 markitai doctor --fix"),
    ("Install the official headless shell:", "安装官方无头浏览器："),
    ("or set MARKITAI_BROWSER_EXECUTABLE to an installed Chrome/Chromium executable.", "或把 MARKITAI_BROWSER_EXECUTABLE 设为已安装的 Chrome/Chromium 可执行文件。"),
    ("No installed executable was found", "没有找到已安装的可执行文件"),
    ("The installed backend could not complete its bounded startup check", "已安装的后端没能完成有时限的启动检查"),
    ("Installed, and it answers --version", "已安装，且能响应 --version"),
    ("Install LibreOffice: winget install --id TheDocumentFoundation.LibreOffice --exact; Office screenshots also require the native PDF page renderer", "安装 LibreOffice：winget install --id TheDocumentFoundation.LibreOffice --exact；Office 截图还需要原生 PDF 页面渲染器"),
    ("Install LibreOffice: brew install --cask libreoffice; Office screenshots also require the native PDF page renderer", "安装 LibreOffice：brew install --cask libreoffice；Office 截图还需要原生 PDF 页面渲染器"),
    ("Install LibreOffice using your distribution package manager and put soffice on PATH; Office screenshots also require the native PDF page renderer", "用发行版的包管理器安装 LibreOffice，并让 soffice 在 PATH 中；Office 截图还需要原生 PDF 页面渲染器"),
    ("Install LibreOffice (Office screenshots also require the native PDF page renderer):", "安装 LibreOffice（Office 截图还需要原生 PDF 页面渲染器）："),
    ("Install LibreOffice with the distribution's package manager and put soffice on PATH (Office screenshots also require the native PDF page renderer):", "用发行版的包管理器安装 LibreOffice，并让 soffice 在 PATH 中（Office 截图还需要原生 PDF 页面渲染器）："),
    ("Old .doc and .ppt files are read by built-in readers; nothing to install", "旧版 .doc 和 .ppt 由内置读取器读取，无需安装"),
    ("Native server and embedded web interface are included", "已包含原生服务和内嵌的网页界面"),
    ("Installed official runtime matches the native adapter protocol", "已安装的官方运行时与原生适配器的协议一致"),
    ("The installed Copilot runtime could not verify its supported protocol", "已安装的 Copilot 运行时无法确认它支持的协议"),
    ("Install the supported official Copilot CLI; no Python SDK is required", "请安装受支持的官方 Copilot CLI；不需要 Python SDK"),
    ("Official runtime reports an authenticated account; model access was not probed", "官方运行时报告已登录账号；没有探测模型访问权限"),
    ("Run markitai auth copilot login", "运行 markitai auth copilot login"),
    ("Installed official runtime matches the native adapter version", "已安装的官方运行时与原生适配器的版本一致"),
    ("The installed Claude runtime could not verify its supported version", "已安装的 Claude 运行时无法确认它支持的版本"),
    ("Install the supported official Claude CLI; no Python SDK is required", "请安装受支持的官方 Claude CLI；不需要 Python SDK"),
    ("Official runtime reports an authenticated subscription; model access was not probed", "官方运行时报告已登录订阅；没有探测模型访问权限"),
    ("Run markitai auth claude login", "运行 markitai auth claude login"),
    ("Installed Codex runtime matches the pinned adapter version", "已安装的 Codex 运行时与固定的适配器版本一致"),
    ("Installed Codex runtime could not verify its supported version", "已安装的 Codex 运行时无法确认它支持的版本"),
    ("Install official Codex 0.159.0; this adapter currently supports chatgpt/gpt-5.5", "请安装官方 Codex 0.159.0；此适配器目前支持 chatgpt/gpt-5.5"),
    ("Official runtime reports a ChatGPT login; model entitlement and inference were not probed", "官方运行时报告已登录 ChatGPT；没有探测模型权限和推理"),
    ("Run markitai auth chatgpt login", "运行 markitai auth chatgpt login"),
    ("Vision routing is configured; provider availability and image capability have not been probed", "已配置视觉路由；没有探测提供商是否可用以及图片能力"),
    ("No routable vision model is configured", "没有配置可路由的视觉模型"),
    ("Configure a supported model and its model_info.supports_vision capability", "请配置受支持的模型，并设置其 model_info.supports_vision"),
    ("Disabled by MARKITAI_NO_VLM_OCR; OCR uses the local backend", "已被 MARKITAI_NO_VLM_OCR 关闭；OCR 使用本地后端"),
    ("Vision requests are configured; no model request was made", "已配置视觉请求；没有发出模型请求"),
    ("Unavailable without a routable vision model", "没有可路由的视觉模型时不可用"),
    ("Unset MARKITAI_NO_VLM_OCR to allow VLM OCR", "取消设置 MARKITAI_NO_VLM_OCR 即可使用 VLM OCR"),
    ("Configure a supported vision model", "请配置受支持的视觉模型"),
    ("Optional: run markitai init to save the detected model in a configuration file", "可选：运行 markitai init，把检测到的模型保存到配置文件"),
    ("Set the provider's API key (for example OPENAI_API_KEY) or configure llm.model_list", "请设置提供商的 API 密钥（例如 OPENAI_API_KEY），或配置 llm.model_list"),
    ("No models configured in llm.model_list", "llm.model_list 中没有配置模型"),
    ("Set a provider API key such as OPENAI_API_KEY (optionally with MODEL), or run markitai init or configure llm.model_list", "请设置提供商的 API 密钥（如 OPENAI_API_KEY，可再设 MODEL），或运行 markitai init，或配置 llm.model_list"),
    ("Models are configured, but all have weight 0 (disabled)", "已配置模型，但权重全为 0（已停用）"),
    ("Set weight > 0 on at least one model to enable LLM use", "请至少给一个模型设置大于 0 的权重，才能使用 LLM"),
    ("Set the named environment variables or update the selected model/provider connection", "请设置所列的环境变量，或更新所选模型/提供商的连接"),
    ("An active API provider is not implemented in this native runtime", "此原生运行时尚未实现某个已启用的 API 提供商"),
    ("Choose a supported API provider", "请选择受支持的 API 提供商"),
    ("No active API model has usable credentials and an endpoint", "没有任何已启用的 API 模型同时具备可用的凭据和端点"),
    ("Check model/provider credentials, API base and routing configuration", "请检查模型/提供商的凭据、API 地址和路由配置"),
];

/// The Chinese of `text` when it is a message or remedy of the doctor,
/// otherwise `text` as it is.
pub(super) fn zh(text: &str) -> String {
    if let Some((_, chinese)) = KNOWN.iter().find(|(english, _)| *english == text) {
        return (*chinese).to_owned();
    }
    portable_ocr(text)
        .or_else(|| with_tail(text, "Missing environment variable(s) used by active models: ", "活跃模型用到的环境变量未设置："))
        .or_else(|| with_tail(text, "MARKITAI_BROWSER_EXECUTABLE does not name an executable file: ", "MARKITAI_BROWSER_EXECUTABLE 指向的不是可执行文件："))
        .or_else(|| with_tail(text, "Configure llm.model_list in ", "请在此配置 llm.model_list："))
        .or_else(|| detected_models(text))
        .unwrap_or_else(|| text.to_owned())
}

fn with_tail(text: &str, english: &str, chinese: &str) -> Option<String> {
    text.strip_prefix(english).map(|rest| format!("{chinese}{rest}"))
}

/// `No llm.model_list; …` with the models the environment names.
fn detected_models(text: &str) -> Option<String> {
    let rest = text.strip_prefix("No llm.model_list; ")?;
    if let Some(names) = rest
        .strip_prefix("--llm uses ")
        .and_then(|rest| rest.strip_suffix(" detected from MODEL or provider API keys in the environment"))
    {
        return Some(format!("未配置 llm.model_list；--llm 会使用从环境变量 MODEL 或提供商 API 密钥检测到的 {names}"));
    }
    let names = rest
        .strip_prefix("the environment names ")
        .and_then(|rest| rest.strip_suffix(", but its provider API key is not set"))?;
    Some(format!("未配置 llm.model_list；环境变量指向 {names}，但没有设置它的提供商 API 密钥"))
}

/// `Portable PaddleOCR engine (tract): …` and its four states.
fn portable_ocr(text: &str) -> Option<String> {
    let rest = text.strip_prefix("Portable PaddleOCR engine (")?;
    let (engine, rest) = rest.split_once("): ")?;
    let head = format!("便携 PaddleOCR 引擎（{engine}）：");
    // `N of the M model files the configured ocr.lang needs (S MB) are not installed; the first OCR downloads them`
    if let Some((missing, rest)) = rest.split_once(" of the ")
        && let Some((total, rest)) = rest.split_once(" model files the configured ocr.lang needs (")
        && let Some((size, tail)) = rest.split_once(" MB) ")
        && tail == "are not installed; the first OCR downloads them"
    {
        return Some(format!("{head}配置的 ocr.lang 需要 {total} 个模型文件，其中 {missing} 个（{size} MB）尚未安装；首次 OCR 时会自动下载"));
    }
    // `the M model files the configured ocr.lang needs are installed and match their SHA-256`
    if let Some(total) = rest
        .strip_prefix("the ")
        .and_then(|rest| rest.strip_suffix(" model files the configured ocr.lang needs are installed and match their SHA-256"))
    {
        return Some(format!("{head}配置的 ocr.lang 需要的 {total} 个模型文件均已安装，且与 SHA-256 一致"));
    }
    if let Some(detail) = rest.strip_prefix("unsafe model path: ") {
        return Some(format!("{head}模型路径不安全：{detail}"));
    }
    // `N damaged model file(s); DETAIL; ordinary OCR will not overwrite them`
    if let Some((count, rest)) = rest.split_once(" damaged model file(s); ")
        && let Some(detail) = rest.strip_suffix("; ordinary OCR will not overwrite them")
    {
        return Some(format!("{head}{count} 个模型文件已损坏；{detail}；普通 OCR 不会覆盖它们"));
    }
    Some(format!("{head}{rest}"))
}

#[cfg(test)]
mod tests {
    use super::zh;

    #[test]
    fn known_messages_and_the_portable_ocr_states_read_in_chinese() {
        assert_eq!(zh("No installed executable was found"), "没有找到已安装的可执行文件");
        assert_eq!(
            zh("Portable PaddleOCR engine (tract): 4 of the 4 model files the configured ocr.lang needs (45.2 MB) are not installed; the first OCR downloads them"),
            "便携 PaddleOCR 引擎（tract）：配置的 ocr.lang 需要 4 个模型文件，其中 4 个（45.2 MB）尚未安装；首次 OCR 时会自动下载"
        );
        assert_eq!(
            zh("Portable PaddleOCR engine (qppocr, tract fallback): the 4 model files the configured ocr.lang needs are installed and match their SHA-256"),
            "便携 PaddleOCR 引擎（qppocr, tract fallback）：配置的 ocr.lang 需要的 4 个模型文件均已安装，且与 SHA-256 一致"
        );
        assert_eq!(zh("Configure llm.model_list in /x/config.json"), "请在此配置 llm.model_list：/x/config.json");
        assert_eq!(
            zh("No llm.model_list; --llm uses openai/gpt-x detected from MODEL or provider API keys in the environment"),
            "未配置 llm.model_list；--llm 会使用从环境变量 MODEL 或提供商 API 密钥检测到的 openai/gpt-x"
        );
        // What the table does not know stays as it is.
        assert_eq!(zh("something else"), "something else");
    }
}
