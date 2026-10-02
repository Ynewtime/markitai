//! Stateful readers for lightweight markup and non-executing TeX extraction.

use crate::{Document, Error, Result};
use std::collections::HashMap;

const MAX_DEPTH: usize = 64;

pub(super) fn extract(source: &str, extension: &str) -> Result<Document> {
    let source = source.replace("\r\n", "\n").replace('\r', "\n");
    let mut doc = Document::default();
    doc.markdown = match extension {
        "rst" => rst(&source, &mut doc.warnings, 0),
        "org" => org(&source, &mut doc.metadata, &mut doc.warnings, 0),
        "tex" | "latex" => tex(&source, &mut doc.metadata, &mut doc.warnings),
        _ => {
            return Err(Error::Unsupported(format!(
                "Unsupported markup format: {extension}"
            )));
        }
    };
    doc.metadata.insert(
        "converter".into(),
        if extension == "tex" || extension == "latex" {
            "latex"
        } else {
            "markup"
        }
        .into(),
    );
    Ok(doc)
}

fn warn(warnings: &mut Vec<String>, message: impl Into<String>) {
    let message = message.into();
    if !warnings.contains(&message) {
        warnings.push(message);
    }
}

fn finish(value: String) -> String {
    if value.trim().is_empty() {
        String::new()
    } else {
        format!("{}\n", value.trim_matches('\n'))
    }
}

/// Joins a reader's lines and blocks; blank lines never run on (a block ends
/// with one, and so may the source line before the next).
fn join_blocks(mut out: Vec<String>) -> String {
    out.dedup_by(|current, previous| current.is_empty() && previous.is_empty());
    out.join("\n")
}

/// A block of its own: a blank line before it and after, so the line above
/// cannot make it a lazy continuation (a quote and a table row), the line
/// below cannot become one of its rows or its text (a table, a footnote
/// definition), and a fence or a displayed formula never shares a paragraph.
fn push_block(out: &mut Vec<String>, block: String) {
    if block.trim().is_empty() {
        return;
    }
    if out.last().is_some_and(|last| !last.is_empty()) {
        out.push(String::new());
    }
    out.push(block);
    out.push(String::new());
}

fn tick_run(text: &str) -> usize {
    text.split(|c| c != '`').map(str::len).max().unwrap_or(0)
}

fn fence(text: &str, language: &str) -> String {
    let marker = "`".repeat((tick_run(text) + 1).max(3));
    let language: String = language
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || "_+-.".contains(*c))
        .collect();
    format!("{marker}{language}\n{}\n{marker}", text.trim_matches('\n'))
}

fn fence_lines(lines: &[&str], language: &str) -> String {
    let marker =
        "`".repeat((lines.iter().map(|line| tick_run(line)).max().unwrap_or(0) + 1).max(3));
    let language: String = language
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || "_+-.".contains(*c))
        .collect();
    let mut output = format!("{marker}{language}\n");
    for line in lines {
        output.push_str(line);
        output.push('\n');
    }
    output.push_str(&marker);
    output
}

fn code(text: &str) -> String {
    let marker = "`".repeat(tick_run(text) + 1);
    let padded = text.starts_with('`')
        || text.ends_with('`')
        || (text.starts_with(' ') && text.ends_with(' '));
    if padded {
        format!("{marker} {text} {marker}")
    } else {
        format!("{marker}{text}{marker}")
    }
}

fn destination(value: &str) -> String {
    value
        .replace(' ', "%20")
        .replace('(', "%28")
        .replace(')', "%29")
        .replace('<', "%3C")
        .replace('>', "%3E")
        .replace('\n', "%0A")
}

fn label(value: &str) -> String {
    value.replace('[', "\\[").replace(']', "\\]")
}

fn link(text: &str, target: &str) -> String {
    format!("[{}]({})", label(text), destination(target))
}

fn image(text: &str, target: &str) -> String {
    format!("!{}", link(text, target))
}

fn slug(text: &str) -> String {
    text.to_lowercase()
        .chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace() || *c == '-')
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join("-")
}

fn table(rows: &[Vec<String>], header: bool) -> String {
    let columns = rows.iter().map(Vec::len).max().unwrap_or(0);
    if columns == 0 {
        return String::new();
    }
    let row = |cells: &[String]| {
        format!(
            "| {} |",
            (0..columns)
                .map(|i| cells
                    .get(i)
                    .map(|s| s.replace('|', "\\|").replace('\n', "<br>"))
                    .unwrap_or_default())
                .collect::<Vec<_>>()
                .join(" | ")
        )
    };
    let mut out = vec![
        if header { row(&rows[0]) } else { row(&[]) },
        format!("| {} |", vec!["---"; columns].join(" | ")),
    ];
    out.extend(
        rows.iter()
            .skip(usize::from(header))
            .map(|cells| row(cells)),
    );
    out.join("\n")
}

fn indent(line: &str) -> usize {
    line.len() - line.trim_start_matches([' ', '\t']).len()
}

fn dedent(lines: &[&str]) -> String {
    let width = lines
        .iter()
        .filter(|line| !line.trim().is_empty())
        .map(|line| indent(line))
        .min()
        .unwrap_or(0);
    lines
        .iter()
        .map(|line| {
            if line.trim().is_empty() {
                ""
            } else {
                &line[width..]
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
        .trim_matches('\n')
        .to_owned()
}

fn indented_end(lines: &[&str], start: usize, base: usize) -> usize {
    let mut end = start;
    while end < lines.len() && (lines[end].trim().is_empty() || indent(lines[end]) > base) {
        end += 1;
    }
    end
}

fn adornment(line: &str) -> Option<char> {
    if line.starts_with(char::is_whitespace) {
        return None;
    }
    let value = line.trim_end();
    let ch = value.chars().next()?;
    (value.len() >= 3 && ch.is_ascii_punctuation() && value.chars().all(|c| c == ch)).then_some(ch)
}

#[derive(Default)]
struct RstRefs {
    targets: HashMap<String, String>,
    substitutions: HashMap<String, String>,
}

fn reference_name(text: &str) -> String {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

fn rst_inline(text: &str, refs: &RstRefs) -> String {
    let mut out = String::new();
    let mut pos = 0;
    while pos < text.len() {
        let tail = &text[pos..];
        if let Some(rest) = tail.strip_prefix("``")
            && let Some(end) = rest.find("``")
        {
            out.push_str(&code(&rest[..end]));
            pos += end + 4;
            continue;
        }
        if tail.starts_with('`')
            && let Some(end) = tail[1..].find('`')
        {
            let body = &tail[1..end + 1];
            let after = &tail[end + 2..];
            if after.starts_with('_') {
                if let Some((text, target)) = body
                    .rsplit_once(" <")
                    .and_then(|(a, b)| b.strip_suffix('>').map(|b| (a, b)))
                {
                    out.push_str(&link(text, target));
                } else if let Some(target) = refs.targets.get(&reference_name(body)) {
                    out.push_str(&link(body, target));
                } else {
                    out.push_str(&tail[..end + 3]);
                }
                pos += end + 3 + usize::from(after.starts_with("__"));
                continue;
            }
        }
        if let Some(rest) = tail.strip_prefix(":math:`")
            && let Some(end) = rest.find('`')
        {
            out.push('$');
            out.push_str(&rest[..end]);
            out.push('$');
            pos += end + 8;
            continue;
        }
        if tail.starts_with('|')
            && let Some(end) = tail[1..].find('|')
            && let Some(value) = refs.substitutions.get(&reference_name(&tail[1..end + 1]))
        {
            out.push_str(value);
            pos += end + 2;
            continue;
        }
        if tail.starts_with('[')
            && let Some(end) = tail.find("]_")
        {
            let name = &tail[1..end];
            if !name.contains(char::is_whitespace) {
                out.push_str(&format!("[^{name}]"));
                pos += end + 2;
                continue;
            }
        }
        let ch = tail.chars().next().unwrap();
        if ch.is_alphanumeric()
            && (pos == 0 || !text[..pos].chars().last().unwrap().is_alphanumeric())
        {
            let end = tail
                .char_indices()
                .find(|(_, c)| !c.is_alphanumeric() && !"_-".contains(*c))
                .map(|(i, _)| i)
                .unwrap_or(tail.len());
            let word = &tail[..end];
            if let Some(name) = word.strip_suffix('_')
                && let Some(target) = refs.targets.get(&reference_name(name))
            {
                out.push_str(&link(name, target));
                pos += end;
                continue;
            }
        }
        out.push(ch);
        pos += ch.len_utf8();
    }
    out
}

fn rst(source: &str, warnings: &mut Vec<String>, depth: usize) -> String {
    if depth > MAX_DEPTH {
        warn(
            warnings,
            "RST nesting limit reached; remaining source preserved.",
        );
        return fence(source, "rst");
    }
    let lines: Vec<_> = source.lines().collect();
    let mut refs = RstRefs::default();
    for line in &lines {
        if let Some(def) = line.strip_prefix(".. _")
            && let Some((name, target)) = def
                .split_once(": ")
                .or_else(|| def.strip_suffix(':').map(|name| (name, "")))
        {
            refs.targets.insert(
                reference_name(name.trim_matches('`')),
                if target.trim().is_empty() {
                    format!("#{}", slug(name))
                } else {
                    target.trim().into()
                },
            );
        }
        if let Some(def) = line.strip_prefix(".. |")
            && let Some((name, body)) = def.split_once("| replace::")
        {
            refs.substitutions
                .insert(reference_name(name), body.trim().to_owned());
        }
    }
    let mut levels: HashMap<(char, bool), usize> = HashMap::new();
    let mut out = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i];
        if let Some(ch) = adornment(line)
            && i + 2 < lines.len()
            && !lines[i + 1].trim().is_empty()
            && adornment(lines[i + 2]) == Some(ch)
            && lines[i + 2].trim().chars().count() >= lines[i + 1].trim().chars().count()
        {
            let next = levels.len() + 1;
            let level = *levels.entry((ch, true)).or_insert(next);
            push_block(
                &mut out,
                format!(
                    "{} {}",
                    "#".repeat(level.min(6)),
                    rst_inline(lines[i + 1].trim(), &refs)
                ),
            );
            i += 3;
            continue;
        }
        if !line.trim().is_empty()
            && i + 1 < lines.len()
            && let Some(ch) = adornment(lines[i + 1])
            && lines[i + 1].trim().chars().count() >= line.trim().chars().count()
        {
            let next = levels.len() + 1;
            let level = *levels.entry((ch, false)).or_insert(next);
            push_block(
                &mut out,
                format!(
                    "{} {}",
                    "#".repeat(level.min(6)),
                    rst_inline(line.trim(), &refs)
                ),
            );
            i += 2;
            continue;
        }
        if grid_border(line) {
            let start = i;
            while i < lines.len() && (grid_border(lines[i]) || lines[i].starts_with('|')) {
                i += 1;
            }
            match rst_grid(&lines[start..i], &refs) {
                Some(value) => push_block(&mut out, value),
                None => {
                    warn(
                        warnings,
                        "RST table with spans or irregular boundaries preserved as source.",
                    );
                    push_block(&mut out, fence(&lines[start..i].join("\n"), "rst"));
                }
            }
            continue;
        }
        if let Some(ranges) = simple_ranges(line) {
            let start = i;
            i += 1;
            let mut rows = Vec::new();
            let mut header = false;
            let mut closed = false;
            while i < lines.len() && !lines[i].trim().is_empty() {
                if simple_ranges(lines[i]).is_some() {
                    if i + 1 < lines.len()
                        && !lines[i + 1].trim().is_empty()
                        && simple_ranges(lines[i + 1]).is_none()
                        && rows.len() == 1
                    {
                        header = true;
                        i += 1;
                        continue;
                    }
                    closed = true;
                    i += 1;
                    break;
                }
                let chars: Vec<_> = lines[i].chars().collect();
                rows.push(
                    ranges
                        .iter()
                        .enumerate()
                        .map(|(column, (from, to))| {
                            let end = if column + 1 == ranges.len() {
                                chars.len()
                            } else {
                                (*to).min(chars.len())
                            };
                            if *from >= end {
                                String::new()
                            } else {
                                rst_inline(
                                    chars[*from..end].iter().collect::<String>().trim(),
                                    &refs,
                                )
                            }
                        })
                        .collect(),
                );
                i += 1;
            }
            if closed {
                push_block(&mut out, table(&rows, header));
            } else {
                warn(warnings, "Unclosed RST simple table preserved as source.");
                push_block(&mut out, fence(&lines[start..i].join("\n"), "rst"));
            }
            continue;
        }
        let trimmed = line.trim_start();
        if trimmed.starts_with(".. ") || trimmed == ".." {
            let end = indented_end(&lines, i + 1, indent(line));
            if trimmed.starts_with(".. _")
                || trimmed.starts_with(".. |") && trimmed.contains("| replace::")
            {
                i = end;
                continue;
            }
            if let Some(note) = trimmed.strip_prefix(".. [")
                && let Some((name, body)) = note.split_once(']')
            {
                let body = format!("{}\n{}", body.trim(), dedent(&lines[i + 1..end]));
                push_block(
                    &mut out,
                    format!(
                        "[^{name}]: {}",
                        rst_inline(body.trim(), &refs).replace('\n', "\n    ")
                    ),
                );
                i = end;
                continue;
            }
            if let Some((name, argument)) =
                trimmed.strip_prefix(".. ").and_then(|v| v.split_once("::"))
            {
                let mut options = HashMap::new();
                let mut body_start = i + 1;
                while body_start < end {
                    let option = lines[body_start].trim();
                    if option.is_empty() {
                        body_start += 1;
                        continue;
                    }
                    if let Some((key, value)) =
                        option.strip_prefix(':').and_then(|v| v.split_once(':'))
                    {
                        options.insert(key, value.trim());
                        body_start += 1;
                    } else {
                        break;
                    }
                }
                let body = dedent(&lines[body_start..end]);
                match name.trim() {
                    "code" | "code-block" | "sourcecode" => {
                        push_block(&mut out, fence(&body, argument.trim()))
                    }
                    "math" => push_block(
                        &mut out,
                        format!("$$\n{}\n$$", format!("{}\n{body}", argument.trim()).trim()),
                    ),
                    "image" | "figure" => {
                        let rendered =
                            image(options.get("alt").copied().unwrap_or(""), argument.trim());
                        push_block(
                            &mut out,
                            if let Some(target) = options.get("target") {
                                format!("[{rendered}]({})", destination(target))
                            } else {
                                rendered
                            },
                        );
                        if !body.is_empty() {
                            let caption = rst(&body, warnings, depth + 1);
                            push_block(&mut out, caption.trim_end().to_owned());
                        }
                    }
                    "note" | "warning" | "tip" | "important" | "caution" | "admonition" => {
                        let title = if name.trim() == "admonition" {
                            argument.trim()
                        } else {
                            name.trim()
                        };
                        let content = if name.trim() == "admonition" {
                            body
                        } else {
                            format!("{}\n{body}", argument.trim()).trim().to_owned()
                        };
                        push_block(
                            &mut out,
                            format!(
                                "> **{}**\n>\n{}",
                                label(title),
                                rst(&content, warnings, depth + 1)
                                    .lines()
                                    .map(|s| format!("> {s}"))
                                    .collect::<Vec<_>>()
                                    .join("\n")
                            ),
                        );
                    }
                    _ => {
                        warn(
                            warnings,
                            format!(
                                "RST directive '{}' preserved without execution.",
                                name.trim()
                            ),
                        );
                        push_block(&mut out, fence(&lines[i..end].join("\n"), "rst"));
                    }
                }
            }
            i = end;
            continue;
        }
        if line.trim_end().ends_with("::") {
            let end = indented_end(&lines, i + 1, indent(line));
            let body = dedent(&lines[i + 1..end]);
            if !body.is_empty() {
                let intro = line.trim_end().strip_suffix("::").unwrap();
                if !intro.trim().is_empty() {
                    out.push(rst_inline(&format!("{intro}:"), &refs));
                }
                push_block(&mut out, fence(&body, ""));
                i = end;
                continue;
            }
        }
        // A term with its definition indented under it, no blank line
        // between: Markdown has no definition list, so the term is a bold
        // line and the definition (itself RST) the block after it.
        if let Some(end) = definition_end(&lines, i) {
            push_block(&mut out, format!("**{}**", rst_inline(line.trim(), &refs)));
            let definition = rst(&dedent(&lines[i + 1..end]), warnings, depth + 1);
            push_block(&mut out, definition.trim_end().to_owned());
            i = end;
            continue;
        }
        out.push(rst_inline(line, &refs));
        i += 1;
    }
    finish(join_blocks(out))
}

/// Whether `text` opens a list item, a field, an option, a line block or a
/// quotation: an indented line below it continues it and is not a definition.
fn opens_other_construct(text: &str) -> bool {
    let mut chars = text.chars();
    let first = chars.next().unwrap_or(' ');
    let second = chars.next();
    if matches!(first, '*' | '-' | '+' | '•' | '‣' | '⁃') && second.is_none_or(char::is_whitespace)
    {
        return true;
    }
    // `-a`, `--long` and `-a FILE  description` open an option list.
    if first == '-' && second.is_some_and(|c| c == '-' || c.is_alphanumeric()) {
        return true;
    }
    if matches!(first, ':' | '|' | '>' | '.') {
        return true;
    }
    // `1.`, `a)`, `(iv)` and `#.` open an enumerated list.
    let rest = text.strip_prefix('(').unwrap_or(text);
    let token: String = rest
        .chars()
        .take_while(|c| c.is_alphanumeric() || *c == '#')
        .collect();
    !token.is_empty()
        && token.chars().count() <= 9
        && rest[token.len()..].starts_with(['.', ')'])
        && rest[token.len() + 1..].starts_with(char::is_whitespace)
}

/// Where the definition of a term at line `i` ends: a line at the margin that
/// starts a block and is followed at once by a more indented line.
fn definition_end(lines: &[&str], i: usize) -> Option<usize> {
    let line = lines[i];
    let next = lines.get(i + 1)?;
    (!line.trim().is_empty()
        && indent(line) == 0
        && (i == 0 || lines[i - 1].trim().is_empty())
        && !next.trim().is_empty()
        && indent(next) > 0
        && !opens_other_construct(line.trim_end()))
    .then(|| indented_end(lines, i + 1, 0))
}

fn grid_border(line: &str) -> bool {
    let text = line.trim_end();
    text.starts_with('+')
        && text.ends_with('+')
        && text.contains(['-', '='])
        && text.chars().all(|c| "+-=".contains(c))
}

fn rst_grid(lines: &[&str], refs: &RstRefs) -> Option<String> {
    let widths: Vec<_> = lines
        .first()?
        .split('+')
        .skip(1)
        .filter(|s| !s.is_empty())
        .map(str::len)
        .collect();
    let columns = widths.len();
    let mut rows = Vec::new();
    let mut current: Vec<String> = Vec::new();
    let mut header = false;
    for line in &lines[1..] {
        if grid_border(line) {
            let parts: Vec<_> = line
                .split('+')
                .skip(1)
                .filter(|s| !s.is_empty())
                .map(str::len)
                .collect();
            if parts != widths {
                return None;
            }
            if !current.is_empty() {
                rows.push(std::mem::take(&mut current));
            }
            if line.contains('=') {
                if rows.len() != 1 {
                    return None;
                }
                header = true;
            }
        } else {
            let cells: Vec<_> = line
                .strip_prefix('|')?
                .trim_end()
                .strip_suffix('|')?
                .split('|')
                .map(|s| rst_inline(s.trim(), refs))
                .collect();
            if cells.len() != columns {
                return None;
            }
            if current.is_empty() {
                current = cells;
            } else {
                for (old, new) in current.iter_mut().zip(cells) {
                    if !new.is_empty() {
                        if !old.is_empty() {
                            old.push('\n');
                        }
                        old.push_str(&new);
                    }
                }
            }
        }
    }
    if !current.is_empty() || rows.is_empty() {
        return None;
    }
    Some(table(&rows, header))
}

fn simple_ranges(line: &str) -> Option<Vec<(usize, usize)>> {
    if !line.chars().all(|c| c == '=' || c == ' ') {
        return None;
    }
    let mut ranges = Vec::new();
    let mut start = None;
    for (i, ch) in line.chars().chain(std::iter::once(' ')).enumerate() {
        if ch == '=' {
            start.get_or_insert(i);
        } else if let Some(begin) = start.take() {
            if i - begin < 2 {
                return None;
            }
            ranges.push((begin, i));
        }
    }
    (ranges.len() >= 2).then_some(ranges)
}

fn org_inline(text: &str, depth: usize) -> String {
    if depth > MAX_DEPTH {
        return text.to_owned();
    }
    let mut out = String::new();
    let mut pos = 0;
    while pos < text.len() {
        let tail = &text[pos..];
        let math = if tail.starts_with("\\(") {
            Some(("\\(", "\\)", "$"))
        } else if tail.starts_with("\\[") {
            Some(("\\[", "\\]", "$$"))
        } else {
            None
        };
        if let Some((open, close, marker)) = math
            && let Some(end) = tail[open.len()..].find(close)
        {
            out.push_str(marker);
            out.push_str(&tail[open.len()..open.len() + end]);
            out.push_str(marker);
            pos += open.len() + end + close.len();
            continue;
        }
        if let Some(rest) = tail.strip_prefix("[[")
            && let Some(end) = rest.find("]]")
        {
            let raw = &rest[..end];
            let (target, description) = raw
                .split_once("][")
                .map(|(a, b)| (a, Some(b)))
                .unwrap_or((raw, None));
            let target = target.strip_prefix("file:").unwrap_or(target);
            let target = if let Some(title) = target.strip_prefix('*') {
                format!("#{}", slug(title))
            } else if let Some(id) = target.strip_prefix("id:") {
                format!("#{id}")
            } else {
                target.to_owned()
            };
            let extension = target
                .split(['?', '#'])
                .next()
                .unwrap_or("")
                .rsplit('.')
                .next()
                .unwrap_or("")
                .to_lowercase();
            let is_image = matches!(
                extension.as_str(),
                "png" | "jpg" | "jpeg" | "gif" | "svg" | "webp" | "bmp" | "tiff" | "avif"
            );
            if let Some(description) = description {
                out.push_str(&link(&org_inline(description, depth + 1), &target));
            } else if is_image {
                out.push_str(&image("", &target));
            } else if target.starts_with("http://")
                || target.starts_with("https://")
                || target.starts_with("mailto:")
            {
                out.push_str(&format!("<{}>", destination(&target)));
            } else {
                out.push_str(&link(&target, &target));
            }
            pos += end + 4;
            continue;
        }
        if let Some(rest) = tail.strip_prefix("[fn:")
            && let Some(end) = rest.find(']')
        {
            out.push_str(&format!("[^{}]", &rest[..end]));
            pos += end + 5;
            continue;
        }
        let ch = tail.chars().next().unwrap();
        if "~=*/_+".contains(ch) {
            let previous = text[..pos].chars().last();
            let opens = previous.is_none_or(|c| c.is_whitespace() || "-({'\"[".contains(c));
            if opens && let Some(end) = tail[ch.len_utf8()..].find(ch) {
                let body = &tail[ch.len_utf8()..ch.len_utf8() + end];
                let after = &tail[ch.len_utf8() + end + ch.len_utf8()..];
                let closes = after
                    .chars()
                    .next()
                    .is_none_or(|c| c.is_whitespace() || "-.,:!?;')\"}]".contains(c));
                if !body.is_empty()
                    && !body.starts_with(char::is_whitespace)
                    && !body.ends_with(char::is_whitespace)
                    && closes
                {
                    let value = match ch {
                        '~' | '=' => code(body),
                        '*' => format!("**{}**", org_inline(body, depth + 1)),
                        '/' => format!("*{}*", org_inline(body, depth + 1)),
                        '_' => format!("<u>{}</u>", org_inline(body, depth + 1)),
                        '+' => format!("~~{}~~", org_inline(body, depth + 1)),
                        _ => unreachable!(),
                    };
                    out.push_str(&value);
                    pos += end + 2 * ch.len_utf8();
                    continue;
                }
            }
        }
        out.push(ch);
        pos += ch.len_utf8();
    }
    out
}

fn org(
    source: &str,
    metadata: &mut serde_json::Map<String, serde_json::Value>,
    warnings: &mut Vec<String>,
    depth: usize,
) -> String {
    if depth > MAX_DEPTH {
        warn(
            warnings,
            "Org nesting limit reached; remaining source preserved.",
        );
        return fence(source, "org");
    }
    let lines: Vec<_> = source.lines().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i];
        let trimmed = line.trim_start();
        let upper = trimmed.to_ascii_uppercase();
        if let Some(rest) = upper.strip_prefix("#+BEGIN_") {
            let name = rest.split_whitespace().next().unwrap_or("");
            let args = trimmed.get(8 + name.len()..).unwrap_or("").trim();
            let start = i;
            let body_start = i + 1;
            i += 1;
            let mut nesting = 1;
            while i < lines.len() {
                let current = lines[i].trim().to_ascii_uppercase();
                if current == format!("#+END_{name}") {
                    nesting -= 1;
                    if nesting == 0 {
                        break;
                    }
                }
                if !matches!(name, "SRC" | "EXAMPLE" | "EXPORT" | "COMMENT")
                    && current
                        .strip_prefix(&format!("#+BEGIN_{name}"))
                        .is_some_and(|s| s.is_empty() || s.starts_with(char::is_whitespace))
                {
                    nesting += 1;
                }
                i += 1;
            }
            let body = lines[body_start..i].join("\n");
            let closed = i < lines.len();
            if closed {
                i += 1;
            } else {
                warn(
                    warnings,
                    format!("Unclosed Org {name} block; content preserved."),
                );
            }
            match name {
                "SRC" => push_block(
                    &mut out,
                    fence_lines(
                        &lines[body_start..if closed { i - 1 } else { i }],
                        args.split_whitespace().next().unwrap_or(""),
                    ),
                ),
                "EXAMPLE" => push_block(
                    &mut out,
                    fence_lines(&lines[body_start..if closed { i - 1 } else { i }], ""),
                ),
                "QUOTE" => push_block(
                    &mut out,
                    org(&body, metadata, warnings, depth + 1)
                        .lines()
                        .map(|line| format!("> {line}"))
                        .collect::<Vec<_>>()
                        .join("\n"),
                ),
                "VERSE" => push_block(
                    &mut out,
                    body.lines()
                        .map(|line| format!("{}  ", org_inline(line, 0)))
                        .collect::<Vec<_>>()
                        .join("\n"),
                ),
                "COMMENT" => {}
                "EXPORT"
                    if args.eq_ignore_ascii_case("markdown")
                        || args.eq_ignore_ascii_case("md")
                        || args.eq_ignore_ascii_case("html") =>
                {
                    push_block(&mut out, body)
                }
                _ => {
                    warn(
                        warnings,
                        format!("Org {name} block preserved as source without execution."),
                    );
                    push_block(&mut out, fence(&lines[start..i].join("\n"), "org"));
                }
            }
            continue;
        }
        if trimmed.starts_with("#+")
            && let Some((key, value)) = trimmed[2..].split_once(':')
        {
            match key.to_ascii_uppercase().as_str() {
                "TITLE" => {
                    metadata.insert("title".into(), org_inline(value.trim(), 0).into());
                }
                // A local file's frontmatter holds only the title, so the
                // author and the date are also kept where the source has them.
                "AUTHOR" => {
                    metadata.insert("author".into(), value.trim().into());
                    if !value.trim().is_empty() {
                        push_block(
                            &mut out,
                            format!("**Author:** {}", org_inline(value.trim(), 0)),
                        );
                    }
                }
                "DATE" => {
                    metadata.insert("published".into(), value.trim().into());
                    if !value.trim().is_empty() {
                        push_block(
                            &mut out,
                            format!("**Date:** {}", org_inline(value.trim(), 0)),
                        );
                    }
                }
                "CAPTION" => push_block(&mut out, format!("*{}*", org_inline(value.trim(), 0))),
                "NAME" => push_block(&mut out, format!("<a id=\"{}\"></a>", slug(value.trim()))),
                _ => {
                    warn(
                        warnings,
                        format!("Org keyword '{}' preserved without evaluation.", key),
                    );
                    push_block(&mut out, code(line));
                }
            }
            i += 1;
            continue;
        }
        if trimmed.starts_with("# ") || trimmed == "#" {
            i += 1;
            continue;
        }
        if line.starts_with('*') {
            let count = line.bytes().take_while(|b| *b == b'*').count();
            if line[count..].starts_with(char::is_whitespace) {
                out.push(format!(
                    "{} {}",
                    "#".repeat(count.min(6)),
                    org_inline(line[count..].trim(), 0)
                ));
                i += 1;
                continue;
            }
        }
        if trimmed.starts_with('|') && trimmed.ends_with('|') {
            let start = i;
            let mut rows = Vec::new();
            let mut header = false;
            let mut invalid = false;
            while i < lines.len()
                && lines[i].trim().starts_with('|')
                && lines[i].trim().ends_with('|')
            {
                let row = lines[i].trim().trim_matches('|');
                if row.chars().all(|c| "-+: ".contains(c)) && row.contains('-') {
                    if rows.len() == 1 {
                        header = true;
                    } else if !rows.is_empty() {
                        invalid = true;
                    }
                } else {
                    rows.push(
                        row.split('|')
                            .map(|cell| org_inline(cell.trim(), 0))
                            .collect::<Vec<_>>(),
                    );
                }
                i += 1;
            }
            if invalid
                || rows
                    .iter()
                    .any(|row| row.len() != rows.first().map(Vec::len).unwrap_or(0))
            {
                warn(
                    warnings,
                    "Org table with irregular boundaries preserved as source.",
                );
                push_block(&mut out, fence(&lines[start..i].join("\n"), "org"));
            } else {
                push_block(&mut out, table(&rows, header));
            }
            continue;
        }
        if let Some(note) = trimmed.strip_prefix("[fn:")
            && let Some((name, body)) = note.split_once(']')
        {
            push_block(
                &mut out,
                format!("[^{name}]: {}", org_inline(body.trim_start(), 0)),
            );
            i += 1;
            continue;
        }
        if trimmed.starts_with(": ") || trimmed == ":" {
            // Consecutive example lines are one block.
            let mut example = Vec::new();
            while i < lines.len() {
                let line = lines[i].trim_start();
                match line
                    .strip_prefix(": ")
                    .or_else(|| (line == ":").then_some(""))
                {
                    Some(body) => example.push(body),
                    None => break,
                }
                i += 1;
            }
            push_block(&mut out, fence_lines(&example, ""));
            continue;
        }
        out.push(org_inline(line, 0));
        i += 1;
    }
    finish(join_blocks(out))
}

fn skip_space(text: &str, mut pos: usize) -> usize {
    while let Some(ch) = text[pos..].chars().next() {
        if !ch.is_whitespace() {
            break;
        }
        pos += ch.len_utf8();
    }
    pos
}

fn command(text: &str, pos: usize) -> (&str, usize) {
    let start = pos + 1;
    let Some(first) = text[start..].chars().next() else {
        return ("", start);
    };
    let mut end = start + first.len_utf8();
    if first.is_ascii_alphabetic() {
        while let Some(ch) = text[end..].chars().next() {
            if !ch.is_ascii_alphabetic() {
                break;
            }
            end += 1;
        }
        if text[end..].starts_with('*') {
            end += 1;
        }
    }
    (&text[start..end], end)
}

fn group(
    text: &str,
    pos: usize,
    open: char,
    close: char,
    comments: bool,
) -> Option<(&str, usize, bool)> {
    let start = skip_space(text, pos);
    if !text[start..].starts_with(open) {
        return None;
    }
    let content = start + open.len_utf8();
    let mut at = content;
    let mut depth = 1;
    let mut braces = 0;
    while at < text.len() {
        let ch = text[at..].chars().next().unwrap();
        if ch == '\\' {
            at += 1;
            if let Some(next) = text[at..].chars().next() {
                at += next.len_utf8();
            }
            continue;
        }
        if ch == '%' && comments {
            at = text[at..]
                .find('\n')
                .map(|end| at + end + 1)
                .unwrap_or(text.len());
            continue;
        }
        if open == '[' {
            if ch == '{' {
                braces += 1;
            }
            if ch == '}' && braces > 0 {
                braces -= 1;
            }
        }
        if braces == 0 {
            if ch == open {
                depth += 1;
            }
            if ch == close {
                depth -= 1;
                if depth == 0 {
                    return Some((&text[content..at], at + ch.len_utf8(), true));
                }
            }
        }
        at += ch.len_utf8();
    }
    Some((&text[content..], text.len(), false))
}

fn argument(text: &str, pos: usize) -> Option<(&str, usize, bool)> {
    group(text, pos, '{', '}', true)
}

fn environment_end<'a>(text: &'a str, start: usize, name: &str) -> (&'a str, usize, bool) {
    if matches!(name, "verbatim" | "Verbatim" | "lstlisting" | "minted") {
        let closing = format!("\\end{{{name}}}");
        return text[start..]
            .find(&closing)
            .map(|end| (&text[start..start + end], start + end + closing.len(), true))
            .unwrap_or((&text[start..], text.len(), false));
    }
    let mut at = start;
    let mut level = 1;
    while at < text.len() {
        let ch = text[at..].chars().next().unwrap();
        if ch == '%' {
            at = text[at..]
                .find('\n')
                .map(|n| at + n + 1)
                .unwrap_or(text.len());
            continue;
        }
        if ch == '\\' {
            let command_start = at;
            let (cmd, next) = command(text, at);
            at = next;
            if let Some((value, next, _)) = argument(text, at) {
                at = next;
                if cmd == "begin" && value == name {
                    level += 1;
                }
                if cmd == "end" && value == name {
                    level -= 1;
                    if level == 0 {
                        return (&text[start..command_start], at, true);
                    }
                }
                if cmd == "begin"
                    && matches!(value, "verbatim" | "Verbatim" | "lstlisting" | "minted")
                {
                    at = environment_end(text, at, value).1;
                }
            }
            continue;
        }
        at += ch.len_utf8();
    }
    (&text[start..], text.len(), false)
}

fn tex(
    source: &str,
    metadata: &mut serde_json::Map<String, serde_json::Value>,
    warnings: &mut Vec<String>,
) -> String {
    let mut at = 0;
    let mut body = source;
    while at < source.len() {
        let ch = source[at..].chars().next().unwrap();
        if ch == '%' {
            at = source[at..]
                .find('\n')
                .map(|n| at + n + 1)
                .unwrap_or(source.len());
            continue;
        }
        if ch == '\\' {
            let (name, next) = command(source, at);
            at = next;
            if let Some((_, next, _)) = group(source, at, '[', ']', true) {
                at = next;
            }
            if let Some((value, next, _)) = argument(source, at) {
                at = next;
                if name == "title" || name == "author" {
                    let value = tex_render(value, warnings, 0, None, 0).trim().to_owned();
                    if !value.is_empty() {
                        metadata.insert(name.into(), value.into());
                    }
                }
                if name == "begin" && value == "document" {
                    let (contents, _, closed) = environment_end(source, at, "document");
                    body = contents;
                    if !closed {
                        warn(
                            warnings,
                            "Unclosed TeX document environment; available body preserved.",
                        );
                    }
                    break;
                }
                if name == "begin"
                    && matches!(value, "verbatim" | "Verbatim" | "lstlisting" | "minted")
                {
                    at = environment_end(source, at, value).1;
                }
            }
            continue;
        }
        at += ch.len_utf8();
    }
    finish(
        tex_render(body, warnings, 0, None, 0)
            .trim_matches('\n')
            .to_owned(),
    )
}

fn tex_render(
    source: &str,
    warnings: &mut Vec<String>,
    depth: usize,
    list: Option<&str>,
    list_depth: usize,
) -> String {
    if depth > MAX_DEPTH {
        warn(
            warnings,
            "TeX nesting limit reached; remaining source preserved.",
        );
        return fence(source, "tex");
    }
    let mut out = String::new();
    let mut pos = 0;
    while pos < source.len() {
        let tail = &source[pos..];
        let ch = tail.chars().next().unwrap();
        if ch == '%' {
            while out.ends_with([' ', '\t']) {
                out.pop();
            }
            pos = tail.find('\n').map(|n| pos + n + 1).unwrap_or(source.len());
            if !out.ends_with('\n') {
                out.push('\n');
            }
            continue;
        }
        if ch == '$' {
            let marker = if tail.starts_with("$$") { "$$" } else { "$" };
            let mut end = marker.len();
            let mut found = None;
            while end < tail.len() {
                if tail[end..].starts_with('\\') {
                    end += 1;
                    if let Some(next) = tail[end..].chars().next() {
                        end += next.len_utf8();
                    }
                    continue;
                }
                if tail[end..].starts_with(marker) {
                    found = Some(end + marker.len());
                    break;
                }
                end += tail[end..].chars().next().unwrap().len_utf8();
            }
            if let Some(end) = found {
                out.push_str(&tail[..end]);
                pos += end;
                continue;
            }
            warn(warnings, "Unclosed TeX math delimiter; source preserved.");
            out.push_str(tail);
            break;
        }
        if ch == '{' {
            let (value, next, closed) = argument(source, pos).unwrap();
            if !closed {
                warn(
                    warnings,
                    "Unbalanced TeX braces; available content preserved.",
                );
            }
            out.push_str(&tex_render(value, warnings, depth + 1, list, list_depth));
            pos = next;
            continue;
        }
        if ch != '\\' {
            if ch == '~' {
                out.push(' ');
            } else {
                out.push(ch);
            }
            pos += ch.len_utf8();
            continue;
        }
        let begin = pos;
        let (raw_name, next) = command(source, pos);
        let name = raw_name.trim_end_matches('*');
        pos = next;
        if matches!(name, "(" | "[") {
            let close = if name == "(" { "\\)" } else { "\\]" };
            if let Some(end) = source[pos..].find(close) {
                let marker = if name == "(" { "$" } else { "$$" };
                out.push_str(marker);
                out.push_str(&source[pos..pos + end]);
                out.push_str(marker);
                pos += end + close.len();
            } else {
                warn(warnings, "Unclosed TeX math delimiter; source preserved.");
                out.push_str(&source[begin..]);
                break;
            }
            continue;
        }
        if matches!(name, "%" | "&" | "_" | "$" | "#" | "{" | "}") {
            out.push_str(name);
            continue;
        }
        if name == "\\" {
            out.push_str("  \n");
            continue;
        }
        if name == "verb" {
            if let Some(delimiter) = source[pos..].chars().next() {
                let start = pos + delimiter.len_utf8();
                if let Some(end) = source[start..].find(delimiter) {
                    out.push_str(&code(&source[start..start + end]));
                    pos = start + end + delimiter.len_utf8();
                    continue;
                }
            }
            warn(warnings, "Unclosed TeX verbatim command; source preserved.");
            out.push_str(&source[begin..]);
            break;
        }
        if name == "begin" {
            let Some((environment, next, closed)) = argument(source, pos) else {
                out.push_str("\\begin");
                continue;
            };
            pos = next;
            if !closed {
                warn(
                    warnings,
                    "Unbalanced TeX environment name; source preserved.",
                );
                out.push_str(&source[begin..]);
                break;
            }
            let mut options = "";
            if let Some((value, next, _)) = group(source, pos, '[', ']', true) {
                options = value;
                pos = next;
            }
            let mut language = "";
            if environment == "minted"
                && let Some((value, next, _)) = argument(source, pos)
            {
                language = value;
                pos = next;
            }
            if environment == "lstlisting" {
                language = options
                    .split(',')
                    .find_map(|part| part.trim().strip_prefix("language="))
                    .unwrap_or("");
            }
            if matches!(environment, "tabular" | "tabular*" | "tabularx") {
                if environment != "tabular"
                    && let Some((_, next, _)) = argument(source, pos)
                {
                    pos = next;
                }
                if let Some((_, next, _)) = argument(source, pos) {
                    pos = next;
                }
            }
            let (body, next, closed) = environment_end(source, pos, environment);
            pos = next;
            if !closed {
                warn(
                    warnings,
                    format!("Unclosed TeX {environment} environment; available content preserved."),
                );
            }
            let rendered = match environment {
                "verbatim" | "Verbatim" | "lstlisting" | "minted" => fence(body, language),
                "equation" | "equation*" | "displaymath" | "math" => {
                    format!("$$\n{}\n$$", body.trim())
                }
                "align" | "align*" | "alignat" | "alignat*" | "eqnarray" | "eqnarray*"
                | "gather" | "gather*" | "multline" | "multline*" => format!(
                    "$$\n\\begin{{aligned}}\n{}\n\\end{{aligned}}\n$$",
                    body.trim()
                ),
                "itemize" | "enumerate" | "description" => {
                    tex_render(body, warnings, depth + 1, Some(environment), list_depth + 1)
                        .trim_matches('\n')
                        .to_owned()
                }
                "quote" | "quotation" => tex_render(body, warnings, depth + 1, list, list_depth)
                    .lines()
                    .map(|line| format!("> {line}"))
                    .collect::<Vec<_>>()
                    .join("\n"),
                "tabular" | "tabular*" | "tabularx" => tex_table(body, warnings, depth + 1)
                    .unwrap_or_else(|| {
                        warn(
                            warnings,
                            "TeX table with spans or unsupported structure preserved as source.",
                        );
                        fence(&source[begin..pos], "tex")
                    }),
                "document" | "center" | "flushleft" | "flushright" | "figure" | "figure*"
                | "table" | "table*" | "minipage" => {
                    tex_render(body, warnings, depth + 1, list, list_depth)
                        .trim_matches('\n')
                        .to_owned()
                }
                "abstract" => format!(
                    "## Abstract\n\n{}",
                    tex_render(body, warnings, depth + 1, list, list_depth).trim()
                ),
                "comment" => String::new(),
                _ => {
                    warn(
                        warnings,
                        format!("TeX environment '{environment}' preserved without execution."),
                    );
                    fence(&source[begin..pos], "tex")
                }
            };
            if !rendered.is_empty() {
                // A list inside an item follows it closely; everything else
                // is a block with a blank line on each side, so the text
                // before it cannot swallow it (a list, a displayed formula,
                // a fence, a table) and the text after it cannot join it.
                let nested_list = list.is_some()
                    && matches!(environment, "itemize" | "enumerate" | "description");
                if nested_list {
                    if !out.is_empty() && !out.ends_with('\n') {
                        out.push('\n');
                    }
                    out.push_str(&rendered);
                    out.push('\n');
                } else {
                    if !out.is_empty() && !out.ends_with("\n\n") {
                        out.push_str(if out.ends_with('\n') { "\n" } else { "\n\n" });
                    }
                    out.push_str(&rendered);
                    out.push_str("\n\n");
                    // The blank line is written; the source's own line break
                    // after the block adds nothing.
                    pos += source[pos..].len() - source[pos..].trim_start().len();
                }
            }
            continue;
        }
        let mut optional = None;
        while let Some((value, next, _)) = group(source, pos, '[', ']', true) {
            optional = Some(value);
            pos = next;
        }
        if name == "item" {
            if !out.ends_with('\n') {
                out.push('\n');
            }
            out.push_str(&"  ".repeat(list_depth.saturating_sub(1)));
            out.push_str(if list == Some("enumerate") {
                "1. "
            } else {
                "- "
            });
            if let Some(title) = optional {
                out.push_str(&format!(
                    "**{}** ",
                    tex_render(title, warnings, depth + 1, None, 0)
                ));
            }
            pos = skip_space(source, pos);
            continue;
        }
        if matches!(name, "href" | "url" | "path" | "includegraphics")
            && let Some((target, next, _)) = group(source, pos, '{', '}', false)
        {
            pos = next;
            if name == "href" {
                if let Some((description, next, _)) = argument(source, pos) {
                    out.push_str(&link(
                        &tex_render(description, warnings, depth + 1, None, 0),
                        target,
                    ));
                    pos = next;
                } else {
                    out.push_str(&link(target, target));
                }
            } else if name == "includegraphics" {
                out.push_str(&image("", target));
            } else if name == "path" {
                out.push_str(&code(target));
            } else {
                out.push_str(&format!("<{}>", destination(target)));
            }
            continue;
        }
        let arg = argument(source, pos);
        if let Some((_, next, closed)) = arg {
            pos = next;
            if !closed {
                warn(
                    warnings,
                    "Unbalanced TeX braces; available content preserved.",
                );
            }
        }
        let raw = arg.map(|(value, _, _)| value);
        match name {
            "documentclass" | "usepackage" | "title" | "author" | "date" | "maketitle"
            | "label" | "index" | "centering" | "newpage" | "clearpage" | "pagebreak"
            | "tableofcontents" | "hline" | "toprule" | "midrule" | "bottomrule" | "cline" => {}
            "chapter" | "section" | "subsection" | "subsubsection" | "paragraph"
            | "subparagraph" => {
                let level = match name {
                    "subsection" => 2,
                    "subsubsection" => 3,
                    "paragraph" => 4,
                    "subparagraph" => 5,
                    _ => 1,
                };
                let body = tex_render(raw.unwrap_or(""), warnings, depth + 1, None, 0);
                if !out.is_empty() && !out.ends_with("\n\n") {
                    out.push_str("\n\n");
                }
                out.push_str(&format!("{} {}\n\n", "#".repeat(level), body.trim()));
            }
            "textbf" | "emph" | "textit" | "texttt" | "underline" | "sout" | "textsc"
            | "textrm" | "textnormal" | "textsf" | "mbox" => {
                let body = tex_render(raw.unwrap_or(""), warnings, depth + 1, None, 0);
                out.push_str(&match name {
                    "textbf" => format!("**{body}**"),
                    "emph" | "textit" => format!("*{body}*"),
                    "texttt" => code(&body),
                    "underline" => format!("<u>{body}</u>"),
                    "sout" => format!("~~{body}~~"),
                    _ => body,
                });
            }
            "caption" => out.push_str(&format!(
                "\n\n*{}*\n\n",
                tex_render(raw.unwrap_or(""), warnings, depth + 1, None, 0).trim()
            )),
            "footnote" => out.push_str(&format!(
                " ({})",
                tex_render(raw.unwrap_or(""), warnings, depth + 1, None, 0).trim()
            )),
            "ref" | "autoref" | "eqref" | "pageref" => {
                let key = raw.unwrap_or("");
                out.push_str(&link(key, &format!("#{}", slug(key))));
                warn(
                    warnings,
                    "TeX cross-references preserve labels; numbering is not evaluated.",
                );
            }
            "cite" | "citep" | "citet" => {
                out.push_str(&format!("[{}]", raw.unwrap_or("")));
                warn(
                    warnings,
                    "TeX citations preserve keys; bibliography resolution is not executed.",
                );
            }
            "input" | "include" | "includeonly" | "bibliography" | "bibliographystyle"
            | "addbibresource" | "newcommand" | "renewcommand" | "def" => {
                out.push_str(&code(&source[begin..pos]));
                warn(
                    warnings,
                    format!(
                        "TeX command '\\{name}' preserved without execution or external file access."
                    ),
                );
            }
            "TeX" => out.push_str("TeX"),
            "LaTeX" => out.push_str("LaTeX"),
            "ldots" | "dots" => out.push('…'),
            "textbackslash" => out.push('\\'),
            "textasciitilde" => out.push('~'),
            "textasciicircum" => out.push('^'),
            "," | ";" | ":" | "quad" | "qquad" | " " => out.push(' '),
            "!" | "/" => {}
            "par" => out.push_str("\n\n"),
            "'" | "`" | "^" | "\"" | "~" | "c" | "v" | "H" => {
                let body = if let Some(raw) = raw {
                    tex_render(raw, warnings, depth + 1, None, 0)
                } else if let Some(ch) = source[pos..].chars().next() {
                    pos += ch.len_utf8();
                    ch.to_string()
                } else {
                    String::new()
                };
                out.push_str(&body);
                out.push(match name {
                    "'" => '\u{301}',
                    "`" => '\u{300}',
                    "^" => '\u{302}',
                    "\"" => '\u{308}',
                    "~" => '\u{303}',
                    "c" => '\u{327}',
                    "v" => '\u{30c}',
                    _ => '\u{30b}',
                });
            }
            _ => {
                warn(
                    warnings,
                    format!("TeX command '\\{name}' is not evaluated; its content is preserved."),
                );
                if let Some(raw) = raw {
                    out.push_str(&tex_render(raw, warnings, depth + 1, None, 0));
                } else {
                    out.push_str(&source[begin..pos]);
                }
            }
        }
    }
    out
}

fn tex_table(source: &str, warnings: &mut Vec<String>, depth: usize) -> Option<String> {
    if source.contains("\\multicolumn")
        || source.contains("\\multirow")
        || source.contains("\\begin")
    {
        return None;
    }
    let mut rows = Vec::new();
    let mut row = Vec::new();
    let mut cell = String::new();
    let mut pos = 0;
    let mut braces = 0;
    let mut math = false;
    let mut header = false;
    while pos < source.len() {
        let tail = &source[pos..];
        let ch = tail.chars().next().unwrap();
        if ch == '%' {
            pos = tail.find('\n').map(|n| pos + n + 1).unwrap_or(source.len());
            continue;
        }
        if ch == '\\' {
            let (name, next) = command(source, pos);
            if braces == 0
                && !math
                && matches!(
                    name,
                    "hline" | "toprule" | "midrule" | "bottomrule" | "cline"
                )
            {
                if name == "midrule" || name == "hline" && rows.len() == 1 {
                    header = true;
                }
                pos = next;
                if name == "cline"
                    && let Some((_, next, _)) = argument(source, pos)
                {
                    pos = next;
                }
                continue;
            }
            if name == "\\" && braces == 0 && !math {
                row.push(tex_render(cell.trim(), warnings, depth + 1, None, 0));
                cell.clear();
                if row.iter().any(|v| !v.is_empty()) {
                    rows.push(std::mem::take(&mut row));
                } else {
                    row.clear();
                }
                pos = next;
                if let Some((_, next, _)) = group(source, pos, '[', ']', true) {
                    pos = next;
                }
                continue;
            }
            cell.push_str(&source[pos..next]);
            pos = next;
            continue;
        }
        if ch == '$' {
            math = !math;
        }
        if ch == '{' {
            braces += 1;
        }
        if ch == '}' {
            braces -= 1;
        }
        if ch == '&' && braces == 0 && !math {
            row.push(tex_render(cell.trim(), warnings, depth + 1, None, 0));
            cell.clear();
        } else {
            cell.push(ch);
        }
        pos += ch.len_utf8();
    }
    if !cell.trim().is_empty() || !row.is_empty() {
        row.push(tex_render(cell.trim(), warnings, depth + 1, None, 0));
        rows.push(row);
    }
    if rows.is_empty() || rows.iter().any(|row| row.len() != rows[0].len()) {
        return None;
    }
    Some(table(&rows, header))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read(source: &str, extension: &str) -> Document {
        extract(source, extension).unwrap()
    }

    #[test]
    fn org_blocks_never_share_a_line_run_with_their_neighbors() {
        // A table row after a quote line is a lazy continuation of the quote,
        // and a line after a table is one more row: blocks get blank lines.
        let doc = read(
            "#+TITLE: Probe\n#+AUTHOR: Me\n#+DATE: 2026\n* Heading\nParagraph.\n#+BEGIN_SRC python\nx = 1\n#+END_SRC\n#+BEGIN_QUOTE\nA quote.\n#+END_QUOTE\n| Name | Qty |\n|------+-----|\n| Apple | 3 |\nFootnote[fn:1].\n[fn:1] The footnote text.\n: first\n:\n: third\n#+CAPTION: Fruit\n| a | b |\nlast\n",
            "org",
        );
        assert_eq!(
            doc.markdown,
            "**Author:** Me\n\n**Date:** 2026\n\n# Heading\nParagraph.\n\n```python\nx = 1\n```\n\n\
             > A quote.\n\n| Name | Qty |\n| --- | --- |\n| Apple | 3 |\n\nFootnote[^1].\n\n\
             [^1]: The footnote text.\n\n```\nfirst\n\nthird\n```\n\n*Fruit*\n\n\
             |  |  |\n| --- | --- |\n| a | b |\n\nlast\n"
        );
        assert_eq!(doc.metadata["author"], "Me");
        assert_eq!(doc.metadata["published"], "2026");
        // Several blank lines in the source stay one.
        assert_eq!(
            read("a\n\n\n\n#+BEGIN_EXAMPLE\nx\n#+END_EXAMPLE\n\n\nb\n", "org").markdown,
            "a\n\n```\nx\n```\n\nb\n"
        );
    }

    #[test]
    fn rst_blocks_never_share_a_line_run_with_their_neighbors() {
        let doc = read(
            "Text before.\n.. note::\n   A note.\n.. code-block:: python\n\n   x = 1\n.. image:: pic.png\n   :alt: A picture\n.. [1] The footnote text.\n:math:`E=mc^2` and\n\n.. math::\n\n   y\n",
            "rst",
        );
        assert_eq!(
            doc.markdown,
            "Text before.\n\n> **note**\n>\n> A note.\n\n```python\nx = 1\n```\n\n![A picture](pic.png)\n\n\
             [^1]: The footnote text.\n\n$E=mc^2$ and\n\n$$\ny\n$$\n"
        );
    }

    #[test]
    fn rst_definition_lists_become_a_bold_term_and_its_definition() {
        let doc = read(
            "Term\n   Definition of term.\n\nSecond *term*\n   First paragraph.\n\n   * a bullet\n   * another\n\nAfter.\n",
            "rst",
        );
        assert_eq!(
            doc.markdown,
            "**Term**\n\nDefinition of term.\n\n**Second *term***\n\nFirst paragraph.\n\n* a bullet\n* another\n\nAfter.\n"
        );
        // What is not a definition list stays: items and fields continued on
        // an indented line, a paragraph, a quotation after a blank line.
        for source in [
            "* item\n  continued\n",
            "1. item\n   continued\n",
            "(a) item\n    continued\n",
            ":field: value\n   continued\n",
            "-v  option text\n    continued\n",
            "Paragraph line\nTerm\n   not at a block start\n",
            "Intro.\n\n   A quotation.\n",
        ] {
            assert!(!read(source, "rst").markdown.contains("**"), "{source:?}");
        }
    }

    #[test]
    fn tex_blocks_never_share_a_paragraph_with_the_text_around_them() {
        let doc = read(
            r"\begin{document}
Text before.
\begin{itemize}\item First \item Second\begin{itemize}\item Nested\end{itemize}\end{itemize}
Text between.
\begin{equation}x=1\end{equation}
\begin{verbatim}
raw
\end{verbatim}
Text after.
\end{document}",
            "tex",
        );
        assert_eq!(
            doc.markdown,
            "Text before.\n\n- First \n- Second\n  - Nested\n\nText between.\n\n$$\nx=1\n$$\n\n```\nraw\n```\n\nText after.\n"
        );
    }

    #[test]
    fn rst_heading_levels_literals_and_fields() {
        let doc = read(
            "Top\n===\n\nMiddle\n------\n\nDeep\n~~~~\n\nAlso top\n========\n\n- ``文档.html``\n:author: Ada\n",
            "rst",
        );
        assert!(doc.markdown.starts_with("# Top\n"));
        assert!(doc.markdown.contains("## Middle\n"));
        assert!(doc.markdown.contains("### Deep\n"));
        assert!(doc.markdown.contains("# Also top\n"));
        assert!(doc.markdown.contains("- `文档.html`\n:author: Ada"));
        assert_eq!(
            read("======\nTitle\n======\n\nbody\n", "rst").markdown,
            "# Title\n\nbody\n"
        );
        assert!(
            !read("intro\n\n- one\n- two\n", "rst")
                .markdown
                .contains('#')
        );
    }

    #[test]
    fn rst_code_and_literal_blocks_preserve_syntax() {
        let doc = read(
            ".. code-block:: bash\n   :linenos:\n\n   curl -O \\\n     https://example.invalid/file\n   echo '```'\n\nThen::\n\n   # not a heading\n   50% done\n\nEnd\n",
            "rst",
        );
        assert!(
            doc.markdown
                .contains("````bash\ncurl -O \\\n  https://example.invalid/file\necho '```'\n````")
        );
        assert!(
            doc.markdown
                .contains("Then:\n\n```\n# not a heading\n50% done\n```")
        );
        assert!(doc.markdown.ends_with("End\n"));
    }

    #[test]
    fn rst_links_notes_math_and_image_directives() {
        let doc = read(
            "See `the site <https://example.invalid/a(b)>`_, Python_ and |project| [1]_.\n\n.. _Python: https://python.invalid/\n.. |project| replace:: Markitai\n.. [1] Footnote content.\n\n.. image:: image (1).png\n   :alt: 图片\n\n.. math:: E = mc^2\n\nUse :math:`\\alpha + x`.\n",
            "rst",
        );
        assert!(
            doc.markdown
                .contains("[the site](https://example.invalid/a%28b%29)")
        );
        assert!(
            doc.markdown
                .contains("[Python](https://python.invalid/) and Markitai [^1]")
        );
        assert!(doc.markdown.contains("[^1]: Footnote content."));
        assert!(doc.markdown.contains("![图片](image%20%281%29.png)"));
        assert!(doc.markdown.contains("$$\nE = mc^2\n$$"));
        assert!(doc.markdown.contains("Use $\\alpha + x$."));
    }

    #[test]
    fn rst_tables_and_unsupported_directives_do_not_discard_content() {
        let doc = read(
            "+------+-----+\n| Name | Age |\n+======+=====+\n| Ada  |  36 |\n+------+-----+\n\n.. include:: secret.txt\n",
            "rst",
        );
        assert!(
            doc.markdown
                .contains("| Name | Age |\n| --- | --- |\n| Ada | 36 |")
        );
        assert!(doc.markdown.contains(".. include:: secret.txt"));
        assert!(doc.warnings.iter().any(|w| w.contains("include")));
        let simple = read(
            "=====  ====\nName   Age\n=====  ====\nAda    36\n=====  ====\n",
            "rst",
        );
        assert!(
            simple
                .markdown
                .contains("| Name | Age |\n| --- | --- |\n| Ada | 36 |")
        );
        let spans = read(
            "+-----+-----+\n| a   | b   |\n+=====+=====+\n| spanning  |\n+-----------+\n",
            "rst",
        );
        assert!(spans.markdown.contains("| spanning  |"));
        assert!(!spans.warnings.is_empty());
    }

    #[test]
    fn org_metadata_headings_links_and_inline_boundaries() {
        let doc = read(
            "#+TITLE: My Notes\n#+AUTHOR: Ada\n\n* One\n** Two\nSee [[https://example.invalid][the site]] and [[https://plain.invalid]].\nrun ~ls -l~ or =pwd= now\na = b + c = d\n*bold* /italic/ _underlined_ +deleted+\n[[file:图片.png]]\n",
            "org",
        );
        assert_eq!(doc.metadata["title"], "My Notes");
        assert_eq!(doc.metadata["author"], "Ada");
        // The frontmatter of a local file holds only the title: the author
        // stays readable in the text.
        assert!(
            doc.markdown
                .starts_with("**Author:** Ada\n\n# One\n## Two\n"),
            "{}",
            doc.markdown
        );
        assert!(
            doc.markdown
                .contains("[the site](https://example.invalid) and <https://plain.invalid>")
        );
        assert!(
            doc.markdown
                .contains("run `ls -l` or `pwd` now\na = b + c = d")
        );
        assert!(
            doc.markdown
                .contains("**bold** *italic* <u>underlined</u> ~~deleted~~")
        );
        assert!(doc.markdown.contains("![](图片.png)"));
        assert_eq!(
            read("a/b/c file_name word+word\n", "org").markdown,
            "a/b/c file_name word+word\n"
        );
    }

    #[test]
    fn org_blocks_math_and_tables() {
        let doc = read(
            "#+BEGIN_SRC rust\nprintln!(\"*literal* ``` 50%\");\n#+END_SRC\n\n| Name | Age |\n|------+-----|\n| Ada  | 36  |\n\n#+BEGIN_QUOTE\n*strong* and \\(\\frac{a}{b}\\)\n#+END_QUOTE\n",
            "org",
        );
        assert!(
            doc.markdown
                .contains("````rust\nprintln!(\"*literal* ``` 50%\");\n````")
        );
        assert!(
            doc.markdown
                .contains("| Name | Age |\n| --- | --- |\n| Ada | 36 |")
        );
        assert!(doc.markdown.contains("> **strong** and $\\frac{a}{b}$"));
        let unclosed = read("#+BEGIN_SRC python\nprint(1)\n", "org");
        assert_eq!(unclosed.markdown, "```python\nprint(1)\n```\n");
        assert_eq!(unclosed.warnings.len(), 1);
    }

    #[test]
    fn org_source_blocks_preserve_leading_and_trailing_blank_lines() {
        let doc = read("#+BEGIN_SRC bash\n\n  echo kept\n\n#+END_SRC\n", "org");
        assert_eq!(doc.markdown, "```bash\n\n  echo kept\n\n```\n");
        assert_eq!(
            read("#+BEGIN_SRC bash\n#+END_SRC\n", "org").markdown,
            "```bash\n```\n"
        );
    }

    #[test]
    fn tex_document_metadata_and_nested_commands() {
        let doc = read(
            r"\documentclass{article}
\usepackage{amsmath}
\title{On \emph{Conversion}}
\begin{document}
\maketitle
\section{One}text
\subsection{Two}\subsubsection{Three}
\textbf{bold \emph{nested}} \textit{it} \texttt{code}
\label{sec:a}\textsc{Kept} words
\end{document}",
            "tex",
        );
        assert_eq!(doc.metadata["title"], "On *Conversion*");
        assert!(!doc.markdown.contains("documentclass"));
        assert!(doc.markdown.contains("# One"));
        assert!(doc.markdown.contains("## Two"));
        assert!(doc.markdown.contains("### Three"));
        assert!(doc.markdown.contains("**bold *nested*** *it* `code`"));
        assert!(doc.markdown.contains("Kept words"));
    }

    #[test]
    fn tex_comments_escapes_verbatim_and_math_are_contextual() {
        let doc = read(
            r"\begin{document}
kept 50\% here % discarded
\verb|a%{b} \textbf{x}| and $\frac{a_b}{c} + \alpha$
\begin{verbatim}
50% done
\textbf{not bold} ```
\end{verbatim}
\[
\sum_{i=1}^{n} x_i
\]
\begin{equation}E=mc^2\end{equation}
\end{document}",
            "tex",
        );
        assert!(doc.markdown.contains("kept 50% here\n"));
        assert!(!doc.markdown.contains("discarded"));
        assert!(doc.markdown.contains(r"`a%{b} \textbf{x}`"));
        assert!(doc.markdown.contains(r"$\frac{a_b}{c} + \alpha$"));
        assert!(
            doc.markdown
                .contains("````\n50% done\n\\textbf{not bold} ```\n````")
        );
        assert!(doc.markdown.contains("$$\n\\sum_{i=1}^{n} x_i\n$$"));
        assert!(doc.markdown.contains("$$\nE=mc^2\n$$"));
    }

    #[test]
    fn tex_nested_lists_links_images_and_tables() {
        let doc = read(
            r"\begin{document}
\begin{itemize}\item outer
\begin{enumerate}\item inner\end{enumerate}
\item[Label] final\end{itemize}
\href{https://example.invalid/a%20b}{the \emph{site}}
\includegraphics[width=2cm]{image (1).png}
\begin{tabular}{lc}
\hline Name & Amount \\ \hline
Ada \& Grace & $x_1$ \\ \hline
\end{tabular}
\end{document}",
            "tex",
        );
        assert!(doc.markdown.contains("- outer"));
        assert!(doc.markdown.contains("  1. inner"));
        assert!(doc.markdown.contains("- **Label** final"));
        assert!(
            doc.markdown
                .contains("[the *site*](https://example.invalid/a%20b)")
        );
        assert!(doc.markdown.contains("![](image%20%281%29.png)"));
        assert!(
            doc.markdown
                .contains("| Name | Amount |\n| --- | --- |\n| Ada & Grace | $x_1$ |")
        );
    }

    #[test]
    fn tex_malformed_and_unsupported_input_is_preserved_with_warnings() {
        let doc = read("\\begin{document}\n\\textbf{oops\n", "tex");
        assert!(doc.markdown.contains("oops"));
        assert!(!doc.warnings.is_empty());
        let doc = read(
            r"\begin{custom}Preserve \special{content}.\end{custom}
\input{outside.tex}
\begin{tabular}{cc}\multicolumn{2}{c}{Combined}\\\end{tabular}",
            "tex",
        );
        assert!(doc.markdown.contains("Preserve \\special{content}."));
        assert!(doc.markdown.contains("\\input{outside.tex}"));
        assert!(doc.markdown.contains("Combined"));
        assert!(doc.warnings.len() >= 3);
    }

    #[test]
    fn empty_unicode_and_crlf_inputs_are_supported() {
        for extension in ["rst", "org", "tex"] {
            assert_eq!(read("", extension).markdown, "");
            assert!(
                read("你好 🌍\r\n", extension)
                    .markdown
                    .contains("你好 🌍\n")
            );
        }
        assert!(extract("hello", "unknown").is_err());
    }
}
