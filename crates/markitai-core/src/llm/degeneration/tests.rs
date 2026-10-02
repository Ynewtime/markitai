use super::*;

fn repeat(unit: &str, copies: usize) -> String {
    unit.repeat(copies)
}

#[track_caller]
fn untouched(answer: &str, source: &str) {
    assert_eq!(salvage(answer, source), None, "{answer:?}");
}

#[test]
fn a_repeated_line_at_the_end_keeps_one_copy() {
    let line = "The quick brown fox jumps over the lazy dog.\n";
    let answer = format!("# Notes\n\nFirst paragraph.\n\n{}", repeat(line, 30));
    let found = salvage(&answer, "").unwrap();
    assert_eq!(
        found.text,
        "# Notes\n\nFirst paragraph.\n\nThe quick brown fox jumps over the lazy dog."
    );
    assert_eq!((found.copies, found.kept), (30, 1));
    assert_eq!(found.removed, answer.trim_end().len() - found.text.len());
    // Blank lines between copies and trailing spaces do not hide the loop.
    let spaced = format!(
        "Intro\n\n{}",
        repeat("Same sentence over and over again.  \n\n", 12)
    );
    assert_eq!(
        salvage(&spaced, "").unwrap().text,
        "Intro\n\nSame sentence over and over again."
    );
}

#[test]
fn a_loop_inside_one_line_is_cut_after_a_whole_sentence() {
    let answer = format!(
        "Summary: {}This is the end of the pa",
        repeat("This is the end of the page. ", 20)
    );
    let found = salvage(&answer, "").unwrap();
    assert_eq!(found.text, "Summary: This is the end of the page.");
    // Twenty whole copies and the one the answer stopped in.
    assert_eq!(found.copies, 21);
    // The run begins inside the line before it (both end in ".\n"): the cut
    // still keeps a whole copy rather than a rotation of it.
    let answer = format!(
        "Intro.\n{}",
        repeat("Alpha line here.\nBeta line here.\n", 10)
    );
    let found = salvage(&answer, "").unwrap();
    assert_eq!(found.text, "Intro.\nAlpha line here.\nBeta line here.");
    assert_eq!(found.copies, 10);
}

#[test]
fn multi_line_and_cjk_loops_are_found() {
    let answer = format!(
        "Header\n\n{}",
        repeat("Row one with words\nRow two with words\n", 10)
    );
    assert_eq!(
        salvage(&answer, "").unwrap().text,
        "Header\n\nRow one with words\nRow two with words"
    );
    let answer = format!("前言。{}", repeat("这是一个重复出现的句子。", 30));
    let found = salvage(&answer, "").unwrap();
    assert_eq!(found.text, "前言。这是一个重复出现的句子。");
    assert_eq!(found.copies, 30);
    // A long paragraph repeated six times counts too.
    let paragraph: String = (1..=40)
        .map(|n| format!("Sentence number {n} of a long paragraph. "))
        .chain(["\n\n".to_owned()])
        .collect();
    let answer = format!("Start\n\n{}", repeat(&paragraph, 6));
    assert_eq!(
        salvage(&answer, "").unwrap().text,
        format!("Start\n\n{}", paragraph.trim_end())
    );
    // A repeated line that itself repeats one sentence is reduced twice.
    let line = format!("{}\n", repeat("Thanks for reading this page. ", 20));
    let found = salvage(&format!("Intro\n{}", repeat(&line, 8)), "").unwrap();
    assert_eq!(found.text, "Intro\nThanks for reading this page.");
    assert_eq!(found.copies, 8);
}

#[test]
fn tables_lists_logs_and_code_keep_their_rows() {
    // Rows that differ are not repetition.
    let mut table = String::from("| Month | Value |\n|---|---|\n");
    for month in 1..=200 {
        table.push_str(&format!("| 2024-{month:03} | 0 |\n"));
    }
    untouched(&table, "");
    let list: String = (1..=80)
        .map(|n| format!("- Item {n}: same text\n"))
        .collect();
    untouched(&list, "");
    let log: String = (0..60)
        .map(|n| format!("2026-10-02T10:{:02}:00 ERROR connection refused\n", n % 60))
        .collect();
    untouched(&log, "");
    // A blank form keeps its blank rows, and a data table its identical rows.
    let form = format!(
        "| Name | Signature | Date |\n|---|---|---|\n{}",
        repeat("|      |           |      |\n", 40)
    );
    untouched(&form, "");
    let filled = format!(
        "| A | B | C |\n|---|---|---|\n{}",
        repeat("| N/A | N/A | N/A |\n", 40)
    );
    untouched(&filled, "");
    // Fill-in lines, rules and closing braces are structure, not a loop.
    untouched(&repeat("Name: ______________________\n", 40), "");
    untouched(&format!("Text\n\n{}", repeat("* * *\n", 30)), "");
    untouched(&format!("fn main() {{\n{}", repeat("}\n", 50)), "");
    let nested: String = (0..12)
        .rev()
        .map(|depth| format!("{}}}\n", "    ".repeat(depth)))
        .collect();
    untouched(&nested, "");
    // Code whose repetition ends before the block does is not a tail.
    let array = format!(
        "```c\nstatic const unsigned char zeros[] = {{\n{}}};\n```\n",
        repeat("    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,\n", 30)
    );
    untouched(&array, "");
    // Dot leaders, underscores and digit runs have too short a period.
    untouched(&format!("Signature: {}", "_".repeat(400)), "");
    untouched(&format!("Contents {} 5", ".".repeat(400)), "");
    untouched(&format!("Zeros: {}", "0 ".repeat(400)), "");
}

#[test]
fn short_or_sparse_repetition_is_left_alone() {
    // Five copies are below the threshold; a refrain sung three times too.
    untouched(&repeat("This sentence appears a few times.\n", 5), "");
    untouched(
        &format!("Verse\n\n{}", repeat("We will, we will rock you\n", 3)),
        "",
    );
    // A short line needs many copies before it adds up to a loop.
    untouched(&format!("Quantities\n{}", repeat("1\n", 40)), "");
    untouched(&format!("Answers\n{}", repeat("Yes\n", 30)), "");
    assert!(salvage(&format!("Answers\n{}", repeat("Yes\n", 50)), "").is_some());
    untouched("", "");
    untouched("   \n\n", "");
}

#[test]
fn repetition_the_source_has_is_the_sources_own() {
    let line = "Repeated disclaimer line from the source.\n";
    let source = format!("Body\n{}", repeat(line, 10));
    // The answer repeats it exactly as often, with other whitespace.
    let answer = format!(
        "Body\n{}",
        repeat("Repeated   disclaimer line from the source.\n\n", 10)
    );
    untouched(&answer, &source);
    // A loop beyond the source's own copies keeps those copies.
    let answer = format!("Body\n{}", repeat(line, 40));
    let found = salvage(&answer, &source).unwrap();
    assert_eq!((found.copies, found.kept), (40, 10));
    assert_eq!(found.text, format!("Body\n{}", repeat(line, 10)).trim_end());
    assert!(
        found
            .warning()
            .contains("the 10 copies the source has were kept")
    );
    // The same holds for a loop inside one line.
    let sentence = "Terms apply to every order placed. ";
    let source = repeat(sentence, 8);
    untouched(&repeat(sentence, 8), &source);
    let found = salvage(&repeat(sentence, 30), &source).unwrap();
    assert_eq!(found.kept, 8);
}

#[test]
fn a_loop_longer_than_the_scan_window_is_removed_whole() {
    let unit = "Looping text that never ends here. ";
    let answer = format!("Opening words. {}", repeat(unit, 20_000));
    assert!(answer.len() > 2 * SCAN_BYTES);
    let started = std::time::Instant::now();
    let found = salvage(&answer, "").unwrap();
    assert_eq!(
        found.text,
        "Opening words. Looping text that never ends here."
    );
    assert_eq!(found.copies, 20_000);
    // Large ordinary text is also cheap: one pass, nothing found.
    let mut plain = String::new();
    let mut state = 7u64;
    while plain.len() < 2 * 1024 * 1024 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        plain.push_str(&format!("word{} ", state >> 40));
        if state.is_multiple_of(13) {
            plain.push('\n');
        }
    }
    untouched(&plain, "");
    assert!(started.elapsed() < std::time::Duration::from_secs(10));
}

#[test]
fn multibyte_text_is_never_cut_inside_a_character() {
    for unit in [
        "é€😀 mixed ünïcödé text, ",
        "日本語のテキストが繰り返される。",
        "Ωmega😀\n",
        "a😀b😀c😀d😀e😀f😀g😀",
    ] {
        for prefix in ["", "x", "😀", "前"] {
            let half: String = unit.chars().take(unit.chars().count() / 2).collect();
            let answer = format!("{prefix}{}{half}", repeat(unit, 70));
            if let Some(found) = salvage(&answer, "") {
                assert!(answer.starts_with(&found.text));
                assert!(found.text.len() >= unit.trim_end().len(), "{found:?}");
            }
        }
    }
}

#[test]
fn the_warning_says_what_was_removed_and_that_nothing_is_cached() {
    let found = salvage(&repeat("Same words again and again.\n", 9), "").unwrap();
    let warning = found.warning();
    assert!(warning.contains("9 times"), "{warning}");
    assert!(warning.contains("one copy was kept"), "{warning}");
    assert!(warning.contains("not cached"), "{warning}");
}
