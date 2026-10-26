//! Chinese for the usage errors the argument parser prints (`error:
//! unexpected argument …`, its tips, the usage line and the `--help` hint).
//!
//! Clap has no translations, so its rendered error is read line by line: each
//! line must match one of the templates below, and only then is the Chinese
//! printed. A line of a kind not listed keeps the whole error in English, so
//! an error is never half translated. Names of options, values and commands
//! stay as typed.
use super::reasons::{captures, fill};
use crate::term::{Style, Tone};

/// Clap's sentences after `error: `.
const ERRORS: &[(&str, &str)] = &[
    ("unexpected argument '{}' found", "无法识别的参数 '{}'"),
    ("unrecognized subcommand '{}'", "无法识别的子命令 '{}'"),
    (
        "a value is required for '{}' but none was supplied",
        "'{}' 需要一个值，但没有提供",
    ),
    (
        "invalid value '{}' for '{}': {}",
        "'{1}' 的值 '{0}' 无效：{2}",
    ),
    ("invalid value '{}' for '{}'", "'{1}' 的值 '{0}' 无效"),
    (
        "the following required arguments were not provided:",
        "缺少以下必需的参数：",
    ),
    (
        "the argument '{}' cannot be used multiple times",
        "参数 '{}' 不能重复使用",
    ),
    (
        "the argument '{}' cannot be used with one or more of the other specified arguments",
        "参数 '{}' 不能与其他已指定的参数一起使用",
    ),
    (
        "the argument '{}' cannot be used with '{}'",
        "参数 '{}' 不能与 '{}' 一起使用",
    ),
    (
        "the argument '{}' cannot be used with:",
        "参数 '{}' 不能与以下参数一起使用：",
    ),
    (
        "the subcommand '{}' cannot be used with '{}'",
        "子命令 '{}' 不能与 '{}' 一起使用",
    ),
    (
        "'{}' requires a subcommand but one was not provided",
        "'{}' 需要一个子命令，但没有提供",
    ),
    (
        "unexpected value '{}' for '{}' found; no more were expected",
        "'{1}' 不需要更多的值，多出了 '{0}'",
    ),
    (
        "equal sign is needed when assigning values to '{}'",
        "给 '{}' 赋值时需要使用等号",
    ),
    // Markitai's own refusals of removed options.
    (
        "--{} has been removed, use '{}' instead.",
        "--{} 已移除，请改用 '{}'。",
    ),
    (
        "--{} has been removed, RTF converts natively now.",
        "--{} 已移除，RTF 现在原生读取。",
    ),
];

/// Lines under the error: tips, value lists and the closing hint.
const OTHERS: &[(&str, &str)] = &[
    (
        "  tip: a similar argument exists: {}",
        "  提示：有相似的参数：{}",
    ),
    (
        "  tip: some similar arguments exist: {}",
        "  提示：有相似的参数：{}",
    ),
    (
        "  tip: a similar subcommand exists: {}",
        "  提示：有相似的子命令：{}",
    ),
    (
        "  tip: some similar subcommands exist: {}",
        "  提示：有相似的子命令：{}",
    ),
    (
        "  tip: a similar value exists: {}",
        "  提示：有相似的值：{}",
    ),
    (
        "  tip: some similar values exist: {}",
        "  提示：有相似的值：{}",
    ),
    (
        "  tip: to pass '{}' as a value, use '{}'",
        "  提示：要把 '{}' 作为值传入，请使用 '{}'",
    ),
    ("  [possible values: {}]", "  [可选值：{}]"),
    ("  [subcommands: {}]", "  [子命令：{}]"),
    ("Usage: {}", "用法：{}"),
    ("For more information, try '{}'.", "更多信息请运行 '{}'。"),
];

fn translated(templates: &[(&str, &str)], line: &str) -> Option<String> {
    templates.iter().find_map(|(english, chinese)| {
        let parts = captures(english, line)?;
        Some(fill(chinese, &parts))
    })
}

/// The Chinese of clap's rendered `error`, in `style`'s colours, or `None`
/// when a line is not one this module knows.
pub(super) fn render(error: &str, style: Style) -> Option<String> {
    let mut out = Vec::new();
    for line in error.trim_end().lines() {
        let shown = if let Some(sentence) = line.strip_prefix("error: ") {
            let sentence = translated(ERRORS, sentence)?;
            format!("{}{sentence}", style.paint(Tone::Error, "错误："))
        } else if line.is_empty() {
            String::new()
        } else if let Some(text) = translated(OTHERS, line) {
            if let Some(usage) = text.strip_prefix("用法：") {
                format!("{}{usage}", style.paint(Tone::Structure, "用法："))
            } else {
                text
            }
        } else if line.starts_with("  ") && !line.trim_start().contains(' ') {
            // An argument of a list under the error (`  <INPUT>`).
            line.to_owned()
        } else {
            return None;
        };
        out.push(shown);
    }
    Some(out.join("\n") + "\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_unknown_option_reads_in_chinese_with_its_tip_usage_and_hint() {
        let english = "error: unexpected argument '--nope' found\n\n  tip: a similar argument exists: '--no-page-markers'\n\nUsage: markitai [OPTIONS] [INPUT] [COMMAND]\n\nFor more information, try '--help'.\n";
        assert_eq!(
            render(english, Style::PLAIN).unwrap(),
            "错误：无法识别的参数 '--nope'\n\n  提示：有相似的参数：'--no-page-markers'\n\n用法：markitai [OPTIONS] [INPUT] [COMMAND]\n\n更多信息请运行 '--help'。\n"
        );
    }

    #[test]
    fn values_and_lists_keep_their_names_and_order() {
        assert_eq!(
            render(
                "error: invalid value 'x' for '--preset <NAME>': must be at least 1\n",
                Style::PLAIN
            )
            .unwrap(),
            "错误：'--preset <NAME>' 的值 'x' 无效：must be at least 1\n"
        );
        assert_eq!(
            render(
                "error: the following required arguments were not provided:\n  <KEY>\n\nUsage: markitai config get <KEY>\n",
                Style::PLAIN
            )
            .unwrap(),
            "错误：缺少以下必需的参数：\n  <KEY>\n\n用法：markitai config get <KEY>\n"
        );
    }

    #[test]
    fn an_error_with_an_unknown_line_stays_english() {
        assert_eq!(
            render("error: something new happened\n", Style::PLAIN),
            None
        );
        assert_eq!(
            render(
                "error: unrecognized subcommand 'x'\n\nA sentence clap added later.\n",
                Style::PLAIN
            ),
            None
        );
    }
}
