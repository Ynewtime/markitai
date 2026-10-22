//! Chinese wording for the diagnostics the CLI prints in English only because
//! they are built where no terminal language is known: run-level errors, and
//! the reasons and warnings that converters return.
//!
//! The English text is the one `--json`, reports and the file log carry, so it
//! is not changed at its source. The terminal translates the lines it knows
//! when it prints them in Chinese, which keeps a run in one language. A
//! message that is not in the table (a provider's reply, an error from the
//! operating system, a parser's detail inside parentheses) is printed as it
//! came; those are technical details, not wording of this program.

/// English templates with `{}` for the variable parts, and their Chinese.
/// The first match wins.
const RULES: &[(&str, &str)] = &[
    ("Input file not found: {}", "找不到输入文件：{}"),
    ("Input is a directory: {}", "输入是目录：{}"),
    ("Input does not exist: {}", "输入不存在：{}"),
    (
        "No model configured; set MODEL and a provider API key, or llm.model_list",
        "未配置 LLM 模型；请设置 MODEL 和供应商的 API key，或配置 llm.model_list",
    ),
    (
        "The installed Codex could not list its models, so chatgpt/{} is sent as named, unvalidated",
        "已安装的 Codex 无法列出它的模型，因此按名称发送 chatgpt/{}，未经验证",
    ),
    (
        "The installed Codex does not offer chatgpt/{}; it offers {}",
        "已安装的 Codex 不提供 chatgpt/{}；它提供 {}",
    ),
    (
        "{} is newer than {}, the release this adapter was verified with; it is used, and every message it sends is still checked",
        "{} 比本适配器验证过的 {} 更新；仍会使用它，并检查它发来的每条消息",
    ),
    (
        "{} is left out of the model pool: {} is not set; set it with: {} (or add {}=your_key to {})",
        "{} 已移出模型池：未设置 {}；设置方法：{}（或把 {}=your_key 写入 {}）",
    ),
    (
        "{} is left out of the model pool: it has no API key (set litellm_params.api_key)",
        "{} 已移出模型池：没有 API key（请设置 litellm_params.api_key）",
    ),
    (
        "{} is not set, and {} needs it; set it with: {} (or add {}=your_key to {})",
        "未设置 {}，{} 需要它；设置方法：{}（或把 {}=your_key 写入 {}）",
    ),
    (
        "{} is not set, and {} need it; set it with: {} (or add {}=your_key to {})",
        "未设置 {}，{} 需要它；设置方法：{}（或把 {}=your_key 写入 {}）",
    ),
    (
        "No LLM model can be used: {} need an API key; set litellm_params.api_key",
        "没有可用的 LLM 模型：{} 需要 API key；请设置 litellm_params.api_key",
    ),
    (
        "No LLM model can be used: {} needs an API key; set litellm_params.api_key",
        "没有可用的 LLM 模型：{} 需要 API key；请设置 litellm_params.api_key",
    ),
    (
        "No LLM model can be used: {}. Set one of them, for example {} (or in {})",
        "没有可用的 LLM 模型：{}。设置其中一个即可，例如 {}（或写入 {}）",
    ),
    (
        "Unsupported file format: {}. Supported extensions: {}.",
        "不支持的文件格式：{}。支持的扩展名：{}。",
    ),
    (
        "The file appears to be damaged or truncated ({})",
        "文件似乎已损坏或被截断（{}）",
    ),
    ("No valid URLs found in {}.", "{} 中没有找到有效的 URL。"),
    (
        "PDF page {}: native text was not recovered ({}); OCR is required to read it. Run again with --ocr.",
        "PDF 第 {} 页：未能提取文字（{}）；需要 OCR 才能读取。请加上 --ocr 重新运行。",
    ),
    (
        "PDF pages {}: native text was not recovered ({}); OCR is required to read them. Run again with --ocr.",
        "PDF 第 {} 页：未能提取文字（{}）；需要 OCR 才能读取。请加上 --ocr 重新运行。",
    ),
    (
        "--llm-batch requires enabled LLM processing",
        "--llm-batch 需要启用 LLM 处理",
    ),
    (
        "--llm-batch requires a directory input",
        "--llm-batch 需要目录作为输入",
    ),
    (
        "--llm-batch-timeout requires --llm-batch",
        "--llm-batch-timeout 需要同时使用 --llm-batch",
    ),
    (
        "--glob and --max-depth require a directory input",
        "--glob 和 --max-depth 需要目录作为输入",
    ),
    (
        "Unknown preset '{}'. Available: {}",
        "未知预设 '{}'。可用：{}",
    ),
    (
        "A directory or .urls batch needs an output directory: pass -o DIR or set output.dir in the configuration",
        "目录或 .urls 批量转换需要输出目录：请用 -o DIR 指定，或在配置中设置 output.dir",
    ),
    (
        "A batch requires an output directory, not a .md file",
        "批量转换需要输出目录，不能是 .md 文件",
    ),
    (
        "--json requires -o: stdout carries the JSON result, so the Markdown needs an output directory or .md file",
        "--json 需要同时使用 -o：标准输出用于输出 JSON 结果，Markdown 需要写入输出目录或 .md 文件",
    ),
    (
        "--interactive and --json cannot be used together",
        "--interactive 和 --json 不能同时使用",
    ),
    (
        "--json and --fix cannot be used together",
        "--json 和 --fix 不能同时使用",
    ),
    (
        "--config-json must be a JSON object",
        "--config-json 必须是 JSON 对象",
    ),
    (
        "--config-json overrides cannot be saved; drop --config-json to write a config file",
        "无法保存 --config-json 的覆盖设置；要写入配置文件，请去掉 --config-json",
    ),
    (
        "--config-json overrides cannot be saved; drop --config-json to edit a config file",
        "无法保存 --config-json 的覆盖设置；要编辑配置文件，请去掉 --config-json",
    ),
    (
        "Configuration must be a JSON object",
        "配置必须是 JSON 对象",
    ),
    (
        "Configuration file does not exist: {}",
        "配置文件不存在：{}",
    ),
    (
        "Cannot mix INPUT, --interactive or Provider Batch options with a subcommand",
        "INPUT、--interactive 或 Provider Batch 选项不能与子命令混用",
    ),
    (
        "Batch collection cannot be mixed with input, submission, resume, preview or interactive mode",
        "批量收集不能与输入、提交、续跑、预览或交互模式混用",
    ),
    (
        "Frozen Batch resume cannot use preview or interactive mode",
        "续跑已冻结的批量任务时不能使用预览或交互模式",
    ),
    (
        "--resume for a single file or URL is not implemented: only directory and .urls batches save progress. Run it again without --resume.",
        "单个文件或 URL 不支持 --resume：只有目录和 .urls 批量转换会保存进度。请去掉 --resume 重新运行。",
    ),
    (
        "Output {} is the input file itself, and overwriting it would destroy the source. Choose another output name or directory.",
        "输出 {} 就是输入文件本身，覆盖会毁掉源文件。请换一个输出名称或目录。",
    ),
    (
        "Output path {} exists and is not a directory; pass a directory with -o (a single input may also name a .md file)",
        "输出路径 {} 已存在且不是目录；请用 -o 指定目录（单个输入也可以指定 .md 文件）",
    ),
    (
        "Cannot create output directory {}: {} is not a directory",
        "无法创建输出目录 {}：{} 不是目录",
    ),
    (
        "Batch exceeds scan_max_files ({}); narrow --glob or increase the configuration limit",
        "批量文件数超过 scan_max_files（{}）；请缩小 --glob 范围或提高配置中的上限",
    ),
    (
        "URL output_name must be a filename without directory components",
        "URL 的 output_name 必须是不含目录的文件名",
    ),
    (
        "Cannot expand ~ in {}: HOME is not set. Spell the directory out, or use ./~ for a directory literally named ~",
        "无法展开 {} 中的 ~：未设置 HOME。请写出完整目录；若目录名就是 ~，请写成 ./~",
    ),
];

/// The pieces of `english` between its `{}` marks matched against `text`;
/// the variable parts in order. The first piece must start the text and the
/// last must end it.
fn captures<'a>(template: &str, text: &'a str) -> Option<Vec<&'a str>> {
    let pieces: Vec<&str> = template.split("{}").collect();
    let (first, rest) = pieces.split_first()?;
    let mut remaining = text.strip_prefix(first)?;
    let mut found = Vec::new();
    for (index, piece) in rest.iter().enumerate() {
        let last = index + 1 == rest.len();
        let at = if last {
            remaining.strip_suffix(piece).map(|head| head.len())?
        } else {
            remaining.find(piece)?
        };
        found.push(&remaining[..at]);
        remaining = &remaining[at + piece.len()..];
    }
    remaining.is_empty().then_some(found)
}

fn fill(template: &str, parts: &[&str]) -> String {
    let mut out = String::new();
    let mut parts = parts.iter();
    for (index, piece) in template.split("{}").enumerate() {
        if index > 0 {
            out.push_str(parts.next().copied().unwrap_or_default());
        }
        out.push_str(piece);
    }
    out
}

fn known(text: &str) -> Option<String> {
    RULES.iter().find_map(|(english, chinese)| {
        let parts = captures(english, text)?;
        Some(fill(chinese, &parts))
    })
}

/// The Chinese for a diagnostic's text (without its `Error:`/`Warning:`
/// label), or the text itself when it is not a known message. An item's line
/// is `name: reason`; the reason is translated and the name kept.
pub(super) fn localize(text: &str) -> String {
    if let Some(chinese) = known(text) {
        return chinese;
    }
    let mut from = 0;
    while let Some(at) = text[from..].find(": ") {
        let split = from + at;
        if let Some(chinese) = known(&text[split + 2..]) {
            return format!("{}: {chinese}", &text[..split]);
        }
        from = split + 2;
    }
    text.to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_messages_keep_their_variable_parts() {
        assert_eq!(
            localize("Input file not found: a b.pdf"),
            "找不到输入文件：a b.pdf"
        );
        assert_eq!(
            localize("Unsupported file format: '.xyz'. Supported extensions: .md .pdf."),
            "不支持的文件格式：'.xyz'。支持的扩展名：.md .pdf。"
        );
        assert_eq!(
            localize("Cannot create output directory a/b: a is not a directory"),
            "无法创建输出目录 a/b：a 不是目录"
        );
    }

    #[test]
    fn an_item_name_is_kept_and_its_reason_translated() {
        assert_eq!(
            localize(
                "broken.docx: The file appears to be damaged or truncated (Native: zip (os error 2))"
            ),
            "broken.docx: 文件似乎已损坏或被截断（Native: zip (os error 2)）"
        );
        assert_eq!(
            localize("a: b: Input file not found: c: d"),
            "a: b: 找不到输入文件：c: d"
        );
    }

    #[test]
    fn the_merged_pdf_page_line_is_translated_for_one_page_and_for_ranges() {
        assert_eq!(
            localize(
                "scan.pdf: PDF pages 1-3, 5: native text was not recovered (no text layer); OCR is required to read them. Run again with --ocr."
            ),
            "scan.pdf: PDF 第 1-3, 5 页：未能提取文字（no text layer）；需要 OCR 才能读取。请加上 --ocr 重新运行。"
        );
        assert!(
            localize(
                "PDF page 2: native text was not recovered (x); OCR is required to read it. Run again with --ocr."
            )
            .starts_with("PDF 第 2 页")
        );
    }

    #[test]
    fn unknown_text_is_not_touched() {
        for text in [
            "connection refused (os error 111)",
            "Input file not found",
            "x: provider said no",
            "",
        ] {
            assert_eq!(localize(text), text);
        }
    }

    #[test]
    fn every_template_has_the_same_number_of_variable_parts_in_both_languages() {
        for (english, chinese) in RULES {
            assert_eq!(
                english.matches("{}").count(),
                chinese.matches("{}").count(),
                "{english}"
            );
        }
    }
}
