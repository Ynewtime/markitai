//! Ruled tables whose cells wrap (`fixtures/cell-wrap`): a line the column's
//! width ended joins the next as one line of text, while a cell's own line
//! breaks stay `<br>`. Python markitai 1.3.0 breaks every wrapped line.
use super::*;

fn markdown(bytes: &[u8]) -> String {
    extract(bytes).unwrap().markdown
}

#[test]
fn lines_the_column_wrapped_are_one_line() {
    let markdown = markdown(include_bytes!("fixtures/cell-wrap/cell-wrap.pdf"));
    for table in [
        // The last word alone wrapped, then two lines of one sentence.
        "|Duty|Notes|Crew|\n|---|---|---|\n\
         |Hull|Tar the hull of the old pilot boat before the rains go.|3|\n\
         |Lamps|Check each lamp on the breakwater after dusk and log the faults in the book.|2|\n\
         |Flags|Raise the flags|1|\n",
        // A centred header wraps as its column does; one broken into two
        // short lines keeps its break.
        "|Step|Hands needed on each of the working days of the week|Unit<br>(kg)|\n|---|---|---|\n",
        // Chinese text wraps between characters, without a space, also
        // after Latin words.
        "|项目|说明|\n|---|---|\n\
         |甲|码头北侧的缆绳每天中午之前需要全部整理好并逐一登记在值班簿上。|\n\
         |乙|第一行<br>第二行|\n\
         |丙|混合 English 与中文的单元格内容在窄列里也会自然换行显示。|",
    ] {
        assert!(markdown.contains(table), "{table}: {markdown}");
    }
}

#[test]
fn a_cells_own_line_breaks_stay() {
    let markdown = markdown(include_bytes!("fixtures/cell-wrap/cell-wrap.pdf"));
    for row in [
        // Two short lines.
        "|1|Bring gloves<br>Wear boots|4|",
        // A label, numbered lines, and a numbered line that wraps.
        "|2|Ingredients:<br>1. Flour<br>2. Sugar<br>3. Butter softened at room temperature for an hour|2|",
        // Bulleted lines, the second wrapping.
        "|3|- first point<br>- second point that is long enough to wrap in this column|1|",
        // An empty line between two lines.
        "|4|Opening line<br>After an empty line|5|",
        // A sentence ending well short of the edge.
        "|5|Paint the shed doors.<br>Then oil the hinges.|3|",
    ] {
        assert!(markdown.contains(row), "{row}: {markdown}");
    }
}

#[test]
fn a_word_split_after_its_hyphen_joins_without_a_space() {
    let markdown = markdown(include_bytes!("fixtures/cell-wrap/cell-hyphens.pdf"));
    assert!(
        markdown.contains(
            "|Case|Text|N|\n|---|---|---|\n\
             |split|The committee reviewed the extra-ordinary proposal in detail.|1|\n\
             |compound|The town council signed a long-term lease for the park.|2|\n\
             |suspended|The harbour plan sets out its short- and long-term goals.|3|\n\
             |break|Flour-<br>Sugar<br>Butter|4|"
        ),
        "{markdown}"
    );
}
