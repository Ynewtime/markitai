//! Superscripts, subscripts and footnote marks set off the baseline by
//! reportlab (a text rise) and LibreOffice (footnote anchors), as invented
//! notes (`fixtures/raised-marks`). The reader once dropped such a mark or
//! moved it to the start of the line below (`70 m²` read `70 m`).
use super::*;

fn markdown(bytes: &[u8]) -> String {
    extract(bytes).unwrap().markdown
}

/// No line is left holding only marks, and no paragraph opens with a mark
/// taken from the line above.
fn no_stray_marks(markdown: &str) {
    for line in markdown.lines() {
        let line = line.trim();
        assert!(
            line.is_empty() || line.chars().any(|c| c.is_alphabetic() && c.is_lowercase()),
            "a line of marks: {line:?} in {markdown}"
        );
        assert!(
            !line.starts_with(|c: char| c.is_ascii_digit()),
            "a mark heads {line:?} in {markdown}"
        );
    }
}

#[test]
fn reportlab_raised_marks_stay_on_their_words() {
    let markdown = markdown(include_bytes!("fixtures/raised-marks/reportlab.pdf"));
    for text in [
        "The spring tide reached 5.6 m¹ on the twelfth, the highest reading this year.",
        "Water covers 70 m² of the car park at high tide, and the new tank holds 450 cm³ of sand.",
        "Harbour dues rose³ in April and again⁴ in June",
        "The keeper logged CO₂ readings twice a day; see the appendix<sup>a</sup> and the errata<sup>\\*</sup> for corrections",
        "which the reader must keep on it⁶",
    ] {
        assert!(markdown.contains(text), "{text}: {markdown}");
    }
    no_stray_marks(&markdown);
}

#[test]
fn reportlab_marks_on_twelve_point_text_stay_on_their_words() {
    let markdown = markdown(include_bytes!("fixtures/raised-marks/reportlab-12pt.pdf"));
    for text in [
        "the ferry deck measures 210 m² and the hold 95 m³, as the surveyor wrote⁵ in the report",
        "The ramp takes 3.5 t⁷ per axle and the water tank carries 8 m³ for the crossing.",
    ] {
        assert!(markdown.contains(text), "{text}: {markdown}");
    }
    no_stray_marks(&markdown);
}

#[test]
fn libreoffice_footnote_anchors_and_labels_are_kept() {
    let markdown = markdown(include_bytes!("fixtures/raised-marks/libreoffice.pdf"));
    for text in [
        "The ferry runs hourly¹ between the east pier and the island. The deck holds 120 m² of cars.",
        "Harbour dues rose² in April. The tank holds 450 cm³ of water and the air has little CO₂ in it.",
        "with a mark after its last word.³",
        // The footnotes' numbers, the last one in the folio band.
        "<sup>1</sup> Except on public holidays.",
        "<sup>2</sup> By five per cent.",
        "<sup>3</sup> Written in March.",
    ] {
        assert!(markdown.contains(text), "{text}: {markdown}");
    }
}
