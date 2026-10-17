use super::*;
use std::path::Path;

/// The parity corpus's scanned page of Chinese, Japanese, Korean and a table
/// (`corpus/ocr/cjk-scanned.pdf`) must come out as the text PDF it was
/// printed from reads: its two headings, each sentence a paragraph of its
/// own (a Markdown viewer joins the lines of one), the Korean line, the
/// table as a Markdown table, and its letter-spaced Latin closed up. It needs
/// `MARKITAI_TEST_CJK_SCANNED_PDF` to name the file and passes without it.
#[test]
fn the_corpus_cjk_scan_keeps_its_lines_its_korean_and_its_table_rows() {
    let Some(path) = std::env::var_os("MARKITAI_TEST_CJK_SCANNED_PDF") else {
        return;
    };
    let config = json!({"ocr":{"enabled":true},"cache":{"enabled":false},"history":{"record":false},"llm":{"enabled":false},"image":{"alt_enabled":false,"desc_enabled":false}});
    let result = convert(
        Path::new(&path).to_str().unwrap(),
        ConvertOptions {
            config: Some(config),
            ..Default::default()
        },
    );
    #[cfg(target_os = "macos")]
    if super::vision_unavailable_under_rosetta(&result) {
        return;
    }
    let markdown = result.unwrap().markdown;
    let lines: Vec<&str> = markdown.lines().map(str::trim).collect();
    let squeezed = |line: &&str| line.split_whitespace().collect::<String>();
    let has = |phrase: &str| lines.iter().any(|line| squeezed(line).contains(phrase));
    // Each sentence on a line of its own.
    assert!(has("本文档用于测试中文文本提取"), "{markdown}");
    assert!(has("日本語のテキストも含まれています"), "{markdown}");
    assert!(has("한국어문장도포함합니다"), "{markdown}");
    assert!(has("서울은대한민국의수도입니다"), "{markdown}");
    assert!(
        !lines
            .iter()
            .any(|line| squeezed(line).contains("日本語") && squeezed(line).contains("한국어")),
        "{markdown}"
    );
    // The table keeps a row per line, with its first character of each word.
    for row in ["项目", "苹果", "香蕉"] {
        assert!(
            lines.iter().any(|line| {
                let cells = squeezed(line);
                let cells = cells.trim_start_matches('|');
                cells.starts_with(row) && cells.len() > row.len()
            }),
            "{row}: {markdown}"
        );
    }
    assert!(has("结论：所有字符都应该被正确保留"), "{markdown}");
    // Headings, paragraphs and a table, as a viewer shows them.
    for heading in ["#第一章概述", "#第二章数据表"] {
        assert!(
            lines.iter().any(|line| squeezed(line) == heading),
            "{heading}: {markdown}"
        );
    }
    let paragraphs: Vec<String> = markdown
        .split("\n\n")
        .map(|paragraph| paragraph.split_whitespace().collect())
        .collect();
    for (a, b) in [
        ("本文档", "Markitai"),
        ("Markitai", "日本語"),
        ("日本語", "한국어"),
    ] {
        assert!(
            !paragraphs.iter().any(|p| p.contains(a) && p.contains(b)),
            "{a} and {b} in one paragraph: {markdown}"
        );
    }
    for row in [
        "|项目|数量|单价|",
        "|---|---|---|",
        "|苹果|12|3.50|",
        "|香蕉|30|1.20|",
    ] {
        assert!(
            lines.iter().any(|line| squeezed(line) == row),
            "{row}: {markdown}"
        );
    }
    assert!(
        markdown.contains("Markitai") && markdown.contains("Markdown"),
        "{markdown}"
    );
}
