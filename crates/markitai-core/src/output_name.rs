//! Portable labels for generated files. Source paths are never rewritten.
//!
//! Callers selecting a leaf from a supplied path do so before sanitizing it;
//! a decoded URL segment can contain separators that are part of its label.

/// Maximum UTF-8 bytes of a generated label, leaving room for Markdown and
/// collision suffixes on ordinary filesystems. Unicode is kept on boundaries.
const MAX_BYTES: usize = 180;

/// Sanitize one generated filename component without changing its identity
/// into the leaf of a path. Invalid characters become underscores, leading
/// and trailing spaces/dots disappear, and Windows devices receive a prefix.
/// `fallback` is a caller-owned label (for example `upload` or `unnamed`).
#[doc(hidden)]
pub fn sanitize(name: &str, fallback: &str) -> String {
    let clean: String = name
        .chars()
        .map(|c| {
            if c.is_control() || "<>:\"/\\|?*".contains(c) {
                '_'
            } else {
                c
            }
        })
        .collect();
    let mut clean = clean.trim_matches([' ', '.']).to_owned();
    if clean.is_empty() {
        clean = fallback.to_owned();
    }
    let base = clean.split('.').next().unwrap_or("").trim_end_matches(' ');
    let upper = base.to_ascii_uppercase();
    if matches!(
        upper.as_str(),
        "CON" | "PRN" | "AUX" | "NUL" | "CONIN$" | "CONOUT$"
    ) || ["COM", "LPT"].iter().any(|family| {
        upper.strip_prefix(family).is_some_and(|number| {
            matches!(
                number,
                "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" | "¹" | "²" | "³"
            )
        })
    }) {
        clean.insert(0, '_');
    }
    if clean.len() > MAX_BYTES {
        // Preserve a modest extension, without copying it out of a label
        // that has no stem. The byte budget works on Unix and Windows alike.
        let extension = clean
            .rsplit_once('.')
            .filter(|(stem, ext)| !stem.is_empty() && !ext.is_empty() && ext.len() < MAX_BYTES / 2)
            .map(|(_, ext)| format!(".{ext}"))
            .unwrap_or_default();
        let mut end = MAX_BYTES - extension.len();
        while !clean.is_char_boundary(end) {
            end -= 1;
        }
        clean = format!("{}{extension}", &clean[..end]);
        clean = clean.trim_end_matches([' ', '.']).to_owned();
    }
    clean
}

/// Attachment labels also occur in unquoted Markdown/HTML destinations.
/// Keep their previous punctuation spelling while preserving readable Unicode.
pub(crate) fn attachment(name: &str) -> String {
    let leaf = name.rsplit(['/', '\\']).next().unwrap_or("");
    sanitize(leaf, "attachment.bin")
        .chars()
        .map(|c| {
            if c.is_whitespace() || matches!(c, '[' | ']' | '(' | ')' | '#' | '%' | '&' | '`') {
                '_'
            } else {
                c
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels_keep_unicode_and_refuse_devices_and_path_characters() {
        assert_eq!(sanitize("报告与予定.txt", "unnamed"), "报告与予定.txt");
        assert_eq!(sanitize("a<>:\"/\\|?*b\n.md", "unnamed"), "a_________b_.md");
        assert_eq!(sanitize(" .. ", "upload"), "upload");
        for name in [
            "con",
            "NUL.txt",
            "COM1",
            "LPT9.log",
            "CON .md",
            "CONIN$",
            "CONOUT$.txt",
            "COM¹",
            "LPT².pdf",
        ] {
            assert!(sanitize(name, "unnamed").starts_with('_'), "{name}");
        }
        assert_eq!(sanitize("COM10.md", "unnamed"), "COM10.md");
        assert_eq!(sanitize("word. ", "unnamed"), "word");
        assert_eq!(sanitize("two/names", "unnamed"), "two_names");
    }

    #[test]
    fn byte_limit_preserves_extension_unicode_boundaries_and_idempotence() {
        for name in [
            "报".repeat(300) + ".pdf",
            "x".repeat(400) + ".docx",
            "界".repeat(100),
            "x".repeat(179) + "...tail",
        ] {
            let clean = sanitize(&name, "unnamed");
            assert!(clean.len() <= MAX_BYTES);
            assert!(!clean.ends_with([' ', '.']));
            assert_eq!(sanitize(&clean, "unnamed"), clean);
            if name.ends_with(".pdf") {
                assert!(clean.ends_with(".pdf"));
            }
        }
        assert_eq!(attachment("../../chart é.png"), "chart_é.png");
        assert_eq!(attachment(r"C:\docs\CON.txt"), "_CON.txt");
        assert_eq!(
            attachment("chart [v2] (final).png"),
            "chart__v2___final_.png"
        );
    }
}
