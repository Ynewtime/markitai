//! Whether a terminal line printed in Chinese still carries English prose.
//!
//! A line with Chinese in it is in Chinese: it may quote what is not this
//! program's wording (a parser's or the operating system's detail, a
//! provider's reply), as the terminal-language rules allow. A line without
//! any is English prose when one of its words is an ordinary English
//! word, once paths, URLs, options, variables, keys, quoted code and the
//! names of products, formats and checks are set aside.
#![allow(dead_code)]

/// Names and words that read the same in both languages.
const NAMES: &[&str] = &[
    "markitai",
    "mkai",
    "llm",
    "ocr",
    "vlm",
    "pdf",
    "api",
    "url",
    "urls",
    "json",
    "yaml",
    "mcp",
    "claude",
    "code",
    "codex",
    "copilot",
    "chatgpt",
    "chrome",
    "chromium",
    "cdp",
    "libreoffice",
    "gemini",
    "openai",
    "anthropic",
    "deepseek",
    "openrouter",
    "jina",
    "cloudflare",
    "defuddle",
    "playwright",
    "office",
    "word",
    "excel",
    "powerpoint",
    "epub",
    "html",
    "markdown",
    "svg",
    "png",
    "jpeg",
    "webp",
    "heic",
    "tiff",
    "rtf",
    "csv",
    "windows",
    "macos",
    "linux",
    "python",
    "rust",
    "vision",
    "paddleocr",
    "bedrock",
    "vertex",
    "ai",
    "foundry",
    "azure",
    "github",
    "spa",
    "id",
    "mib",
    "kib",
    "gib",
    "mb",
    "kb",
    "gb",
    "ms",
    "ok",
    "true",
    "false",
    "null",
    "on",
    "off",
    "yes",
    "no",
    "native",
    "readers",
    "local",
    "model",
    "serve",
    "utf-8",
    "stdout",
    "stderr",
    "tract",
    // Preset and profile names.
    "minimal",
    "standard",
    "rich",
    "rag",
    "obsidian",
    "okf",
];

fn has_chinese(text: &str) -> bool {
    text.chars().any(|c| ('\u{4e00}'..='\u{9fff}').contains(&c))
}

/// A word a sentence is made of: letters only (an inner apostrophe or hyphen
/// allowed), not a name, not all capitals (a variable or an acronym).
fn prose_word(token: &str) -> bool {
    let word = token.trim_matches(|c: char| !c.is_ascii_alphanumeric());
    word.len() >= 2
        && word
            .chars()
            .all(|c| c.is_ascii_alphabetic() || c == '\'' || c == '-')
        && word.chars().any(|c| c.is_ascii_lowercase())
        && !word.starts_with('-')
        && !NAMES.contains(&word.to_ascii_lowercase().as_str())
}

/// Code, a path, an address, an option or a key rather than a word.
fn technical(token: &str) -> bool {
    let token = token.trim_matches(|c: char| matches!(c, ',' | ';' | ':' | '(' | ')' | '.'));
    token.starts_with('-')
        || token.starts_with('<')
        || token.contains(['/', '\\', '=', '_', '@', '{', '[', '#'])
        || token.contains("://")
        || (token.contains('.') && !token.ends_with('.'))
        || token.chars().any(|c| c.is_ascii_digit())
}

pub fn english_prose(line: &str) -> bool {
    if has_chinese(line) {
        return false;
    }
    let trimmed = line.trim_start();
    // A command to copy, on a line of its own.
    if trimmed.starts_with("markitai ") || trimmed.starts_with("mkai ") {
        return false;
    }
    let mut quoted = false;
    let mut words = 0;
    for token in line.split_whitespace() {
        // `code` and 'values' are quoted as typed.
        let opens = token.starts_with(['`', '\'', '"']);
        let closes = token.ends_with(['`', '\'', '"'])
            || token
                .trim_end_matches(['.', ',', ':', ';', ')'])
                .ends_with(['`', '\'', '"']);
        if quoted || opens {
            quoted = !(closes && (quoted || token.len() > 1));
            continue;
        }
        if !technical(token) && prose_word(token) {
            words += 1;
        }
    }
    words >= 1
}

#[test]
fn the_lines_that_stayed_english_are_caught() {
    for line in [
        "Markitai server listening on http://127.0.0.1:3600",
        "Remote access token: 0123456789abcdef0123456789abcdef",
        "  → Server stopping: draining active conversions and saving history.",
        "◆ Provider Authentication",
        "  ! chatgpt: unavailable",
        "    Codex executable is unavailable",
        "error: unexpected argument '--nope' found",
        "For more information, try '--help'.",
        "✗ llm.enabled must be a boolean",
        "✗ Cannot establish MCP stdio session",
        "  ✗ error sending request: client error (Connect): tcp connect error",
        "  ! Ignored COPILOT_CLI_PATH from ./.env: a folder's .env may not set programs",
    ] {
        assert!(english_prose(line), "{line}");
    }
}

#[test]
fn chinese_lines_and_technical_lines_pass() {
    for line in [
        "◆ Markitai 服务正在监听 http://127.0.0.1:3600",
        "  • 访问令牌：0123456789abcdef",
        "    http://127.0.0.1:3600/#token=0123456789abcdef",
        "  ✗ 文件似乎已损坏或被截断（Native PDF conversion failed: failed parsing cross reference table）",
        "    markitai doctor --fix",
        "  <KEY>",
        "/home/user/.markitai/config.json",
        "  ✓ Native Office readers：旧版 .doc 和 .ppt 由内置读取器读取",
        "output.on_conflict = \"skip\"",
        "",
    ] {
        assert!(!english_prose(line), "{line}");
    }
}
