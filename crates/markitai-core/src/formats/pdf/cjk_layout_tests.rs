//! A Chinese form printed by iText: the shapes of its first pages, each once
//! misread by the page reader, set as an authored course enrolment form with
//! invented text (`fixtures/cjk-course`).
use super::*;

fn markdown(bytes: &[u8]) -> String {
    extract(bytes).unwrap().markdown
}

#[test]
fn content_operators_separated_by_nul_are_read() {
    // NUL is white space in a content stream; the page was read as empty.
    let markdown = markdown(include_bytes!("fixtures/cjk-course/nul-separated.pdf"));
    for text in [
        "## 报名资料",
        "报名人：示例学员",
        "报名日期：2026年01月02日",
        "本报名单由课程简章、报名表、缴费凭证、课程安排和其他书面约定构成，各方应当遵守约定。",
    ] {
        assert!(markdown.contains(text), "{text}: {markdown}");
    }
}

#[test]
fn a_header_between_segmented_rules_makes_a_table_of_the_rows_below() {
    let markdown = markdown(include_bytes!(
        "fixtures/cjk-course/segmented-header-table.pdf"
    ));
    assert!(
        markdown.contains(
            "|序号|课程名称|累计课时分钟|课程周期|上课周数|上课频次|材料费|\n\
             |---|---|---|---|---|---|---|\n\
             |1|社区书法艺术与传统绘画研修进阶课程（周末）|1,440.00|1年|1年|每周|640.00|\n\
             |2|社区合唱团基础声乐入门课程（晚间班）|2,880.00|1年|1年|每周|320.50|\n"
        ),
        "{markdown}"
    );
    assert!(
        markdown.contains("**首期材料费合计：960.50**"),
        "{markdown}"
    );
}

#[test]
fn fields_on_rows_are_no_headings() {
    let markdown = markdown(include_bytes!("fixtures/cjk-course/form-grid.pdf"));
    assert!(markdown.contains("## 报名资料"), "{markdown}");
    assert!(markdown.contains("## 学员信息"), "{markdown}");
    for field in [
        "报名人：示例学员 证件类型：学生证 学员编号：202601020001",
        "报名日期：2026年01月02日 开课日期：2026年01月03日上午",
        "上课学员：示例学员乙 与报名人关系：子女",
        "班级类型：周末班",
    ] {
        let line = markdown
            .lines()
            .find(|line| line.contains(field))
            .unwrap_or_else(|| panic!("{field}: {markdown}"));
        assert!(!line.starts_with('#') && !line.contains('`'), "{line}");
    }
    // A row of fields is a paragraph of its own.
    assert!(
        markdown.contains("\n\n报名人：示例学员 证件类型：学生证 学员编号：202601020001\n\n"),
        "{markdown}"
    );
}

#[test]
fn sentences_in_a_larger_bold_face_are_no_headings() {
    let markdown = markdown(include_bytes!("fixtures/cjk-course/long-sentences.pdf"));
    assert!(
        markdown.contains("## 学习中心办学资质告知和服务评价"),
        "{markdown}"
    );
    for sentence in [
        "报名人已确认学员符合课程简章所述的报名条件。",
        "如有隐瞒或填写不实",
        "本中心办学资质与服务评价符合要求",
    ] {
        let line = markdown
            .lines()
            .find(|line| line.contains(sentence))
            .unwrap_or_else(|| panic!("{sentence}: {markdown}"));
        assert!(line.starts_with("**"), "{line}");
    }
    // The address ends before the bold run's closing marks.
    assert!(
        markdown.contains(
            "[http://www.example.com/services/](http://www.example.com/services/) Default.aspx"
        ),
        "{markdown}"
    );
}

#[test]
fn cjk_faces_are_not_set_as_code() {
    // One face flagged FixedPitch with varied widths, one listing only its
    // half-width glyphs: neither is monospaced, and no line is code.
    let markdown = markdown(include_bytes!("fixtures/cjk-course/fixed-pitch-cjk.pdf"));
    assert!(!markdown.contains('`'), "{markdown}");
    assert_eq!(
        markdown
            .matches("第2条 学习期间：本课程的学习期间为1年，材料费为640.00元，可以按年交纳。")
            .count(),
        2,
        "{markdown}"
    );
}

#[test]
fn notdef_glyphs_mapped_to_a_noncharacter_read_as_nothing() {
    let markdown = markdown(include_bytes!("fixtures/cjk-course/notdef-ffff.pdf"));
    assert!(!markdown.contains('\u{ffff}'), "{markdown:?}");
    assert!(
        markdown.contains(
            "本课程的学习期间为一年，自课程开始之日零时起至结业之日二十四时止，续报另行约定。"
        ),
        "{markdown}"
    );
}

#[test]
fn cjk_enumerators_are_list_items() {
    let markdown = markdown(include_bytes!("fixtures/cjk-course/cjk-enumerators.pdf"));
    assert!(
        markdown.contains(
            "- 1．学员曾经或目前参加过如下2类课程：\n\
             - 1）书法课程：楷书基础（4级及以上）、行书进阶（3课时以上书写/临摹/创作练习）。\n\
             - 2）绘画进阶课程：由素描或水彩基础课程延伸的进阶内容（详见五大类素描延伸课程、七大类水彩延伸课程清 单）。\n\
             - 2．学员有以下上课情况：\n"
        ),
        "{markdown}"
    );
    // A line after an item that ended short is a paragraph of its own.
    assert!(
        markdown.contains("\n\n下述情况，仍可正常报名：\n\n"),
        "{markdown}"
    );
    assert!(markdown.contains("\n\n说明\n\n"), "{markdown}");
}

#[test]
fn double_spaced_cjk_lines_join_their_paragraph() {
    // 10.4 pt at 20 pt leading: the lines of a clause are farther apart
    // than the reader's paragraph threshold, and a clause's text hangs five
    // ems past its number.
    let markdown = markdown(include_bytes!("fixtures/cjk-course/double-spaced.pdf"));
    for paragraph in [
        "\n\n第一条 本附加服务约定（以下简称“本附加约定”）需附加于各类线下课程报名协议、 各类线上课程报名协议（以下简称“主协议”）上方可生效。\n\n",
        "\n\n第二条 本附加约定与主协议相悖之处，以本附加约定为准；本附加约定未尽事宜，以 主协议为准。\n\n",
        "1.1 报名学员范围 首次报名或非连续报名本课程时年满十八周岁至七十周岁的个人可作 为报名学员参加本课程，且报名时需符合本中心当时的报名规定。\n\n",
    ] {
        assert!(markdown.contains(paragraph), "{paragraph}: {markdown}");
    }
    // A centred line stopping short of the text under it ends its block.
    assert!(markdown.contains("\n\n总则\n\n"), "{markdown}");
}
