//! The text-layer check (`text_trust`): each signal on text, then on
//! authored PDFs whose text layer is right, or present but wrong, through
//! extraction, OCR routing and assembly.
use super::text_trust::{self, FAILED, NO_TEXT_LAYER, TRUSTED, char_script, dominant_script};
use super::*;
use lopdf::dictionary;
use serde_json::json;

const LATIN: &str =
    "The committee met on Tuesday to review the annual budget and the plans for the new library.";
const LATIN_LINES: [&str; 6] = [
    LATIN,
    "Members asked how the opening hours would change over the summer.",
    "A short report on reading rooms was approved without any changes.",
    "Volunteers will help move the archive boxes during the first week.",
    "Questions about parking should be sent to the facilities office.",
    "The next meeting is planned for the second Monday of the month.",
];
const CHINESE: &str = "本报告介绍了城市公共图书馆的建设计划，以及明年预算的主要内容。";
const JAPANESE: &str = "この資料は新しい図書館の計画と来年度の予算について説明します。";
// Hebrew and Arabic in reading order (made-up sentences of common words).
const HEBREW: &str = "הספר החדש נמצא בבית הספר שלום לכל הילדים והמורים בעולם הגדול";
const ARABIC: &str = "الكتاب الجديد في المكتبة على الطاولة الكبيرة مع الطلاب في المدرسة";
const MOJIBAKE: &str = "The cafÃ© served crÃ¨me brÃ»lÃ©e and a â€œspecialâ€ rÃ©sumÃ©.";

fn reversed(text: &str) -> String {
    text.chars().rev().collect()
}

fn fails(text: &str) -> text_trust::Failure {
    text_trust::judge(text).unwrap_or_else(|| panic!("trusted: {text}"))
}

// --- scripts ----------------------------------------------------------------

#[test]
fn letters_belong_to_their_writing_systems() {
    for (ch, script) in [
        ('a', Some("Latin")),
        ('é', Some("Latin")),
        ('中', Some("CJK")),
        ('の', Some("CJK")),
        ('한', Some("Hangul")),
        ('ש', Some("Hebrew")),
        ('ب', Some("Arabic")),
        ('ж', Some("Cyrillic")),
        ('क', Some("Devanagari")),
        ('த', Some("Tamil")),
        ('1', None),
        ('∑', None),
        ('𝑥', None),
    ] {
        assert_eq!(char_script(ch), script, "{ch}");
    }
    assert_eq!(dominant_script("使用 pymupdf 提取文本的方法"), Some("CJK"));
    assert_eq!(dominant_script("Run uv sync 然后"), Some("Latin"));
    assert_eq!(dominant_script("12 + 3"), None);
}

#[test]
fn each_ocr_language_reads_its_writing_systems() {
    let reads = text_trust::readable_scripts;
    assert_eq!(reads("en"), ["Latin", "CJK", "Hangul"]);
    assert_eq!(reads("zh"), ["Latin", "CJK"]);
    assert_eq!(reads("ar"), ["Latin", "Arabic"]);
    assert_eq!(reads("ar-sa"), ["Latin", "Arabic"]);
    assert_eq!(reads("ru"), ["Latin", "Cyrillic"]);
    assert_eq!(reads("latin"), ["Latin"]);
    assert_eq!(reads("no-such-language"), ["Latin", "CJK"]);
    assert_eq!(text_trust::lang_for_script("Arabic"), Some("ar"));
    assert_eq!(text_trust::lang_for_script("Hebrew"), None);
}

// --- the signals on text ------------------------------------------------------

#[test]
fn a_broken_font_map_is_private_use_replacement_or_control_codes() {
    let private: String = (0..200)
        .map(|i| char::from_u32(0xE000 + i % 40).unwrap())
        .collect();
    let failure = fails(&private);
    assert_eq!(failure.kind, "a broken font map");
    assert_eq!(
        failure.detail,
        "a broken font map (100% of the characters are private-use, replacement or control codes)"
    );
    assert_eq!(failure.script, None);
    assert_eq!(
        fails(&"Report \u{FFFD}\u{FFFD}\u{FFFD} ".repeat(20)).kind,
        "a broken font map"
    );
    // Unassigned code points and C0 controls count; tabs and line breaks do not.
    assert!(text_trust::judge(&"Report\u{0378}\u{0001}\u{0379} ".repeat(20)).is_some());
    assert!(text_trust::judge(&format!("{LATIN}\t\r\n").repeat(10)).is_none());
}

#[test]
fn symbol_font_bullets_and_a_few_icons_are_no_broken_map() {
    // Wingdings and Symbol bytes (U+F0xx) are bullets, not letters.
    let bullets = format!("\u{F0B7} {LATIN}\n").repeat(10);
    assert!(text_trust::judge(&bullets).is_none());
    let icons = format!("\u{E001} {LATIN} \u{E002} {LATIN} \u{FFFD}");
    assert!(text_trust::judge(&icons).is_none());
}

#[test]
fn raw_glyph_codes_are_cid_codes_covering_a_fifth_of_the_text() {
    let failure = fails(&format!(
        "Summary {}",
        "(cid:12)(cid:45)(cid:3) ".repeat(10)
    ));
    assert_eq!(
        failure.detail,
        "raw glyph codes instead of text (30 (cid:NN) codes)"
    );
    // A document about the codes is not made of them.
    let about = format!("{LATIN} pdfminer writes (cid:12) for an unmapped glyph. ").repeat(3);
    assert!(text_trust::judge(&about).is_none());
}

#[test]
fn mojibake_is_utf8_read_as_windows_1252() {
    let failure = fails(&format!("{MOJIBAKE} ").repeat(3));
    assert_eq!(failure.kind, "mojibake, UTF-8 read as Windows-1252");
    assert_eq!(
        failure.detail,
        "mojibake, UTF-8 read as Windows-1252 (12 words)"
    );
    assert_eq!(failure.script, Some("Latin"));
    // Chinese through Windows-1252: the script it reads as is CJK.
    let (decoded, _, _) = encoding_rs::WINDOWS_1252.decode("中文内容说明".as_bytes());
    let failure = fails(&format!(
        "{decoded} {decoded} {decoded} report and summary of the year"
    ));
    assert_eq!(failure.script, Some("CJK"));
}

#[test]
fn accented_and_other_non_ascii_text_is_no_mojibake() {
    for text in [
        "Le café servait une crème brûlée et un résumé très soigné, à côté. ",
        "Größere Änderungen an der Straße wurden für Übermorgen geplant. ",
        "Ünlü şarkıcı İstanbul'da büyük bir konser verdi, çok güzeldi. ",
    ] {
        assert!(text_trust::judge(&text.repeat(3)).is_none(), "{text}");
    }
    // A page about mojibake quotes an example or two, not three.
    let about =
        format!("{LATIN} {LATIN} Mojibake such as cafÃ© or crÃ¨me is UTF-8 read as Windows-1252.");
    assert!(text_trust::judge(&about).is_none());
}

#[test]
fn right_to_left_text_in_reading_order_is_trusted() {
    for text in [HEBREW, ARABIC] {
        assert!(text_trust::judge(text).is_none(), "{text}");
    }
    // Pointed Hebrew and Arabic with harakat read as their letters.
    let pointed = "בְּרֵאשִׁית בָּרָא אֱלֹהִים אֵת הַשָּׁמַיִם וְאֵת הָאָרֶץ וְהָאָרֶץ הָיְתָה תֹהוּ";
    assert!(text_trust::judge(pointed).is_none());
    let voweled = "ذَهَبَ الطَّالِبُ إِلَى الْمَدْرَسَةِ الْجَدِيدَةِ فِي الصَّبَاحِ مَعَ أَصْدِقَائِهِ";
    assert!(text_trust::judge(voweled).is_none());
}

#[test]
fn right_to_left_text_in_reverse_order_fails_naming_its_script() {
    for (text, script) in [(HEBREW, "Hebrew"), (ARABIC, "Arabic")] {
        let failure = fails(&reversed(text));
        assert_eq!(
            failure.reason(),
            format!("{FAILED}right-to-left text in reverse order")
        );
        assert_eq!(failure.script, Some(script));
    }
    // Arabic presentation forms read as their letters: reversed, they fail.
    let forms = "\u{FEB3}\u{FEE0}\u{FE8E}\u{FEE1} \u{FE8D}\u{FEDF}\u{FEE4}\u{FEAA}\u{FEAD}\u{FEB3}\u{FE94} ";
    assert!(text_trust::judge(&forms.repeat(4)).is_none());
    let reversed_forms = reversed(&forms.repeat(4));
    assert_eq!(fails(&reversed_forms).script, Some("Arabic"));
}

#[test]
fn a_reversed_quote_in_another_script_is_left_alone() {
    let text = format!("{} {}", LATIN.repeat(4), reversed(&HEBREW[..20]));
    assert!(text_trust::judge(&text).is_none());
}

#[test]
fn clean_pages_of_every_kind_pass() {
    let pages = [
        LATIN_LINES.join("\n"),
        CHINESE.repeat(8),
        JAPANESE.repeat(8),
        "使用以下命令安装依赖，然后运行测试：\n```\nuv sync --all-extras && uv run pytest -q\n\
         def parse(path: str) -> dict[str, int]:\n    return {k: len(v) for k, v in load(path).items()}\n```\n\
         返回值是一个字典，键为文件名，值为行数。"
            .into(),
        "|Region|Q1|Q2|Q3|Q4|Total|\n|---|---|---|---|---|---|\n|North|1,204|1,388|1,502|1,611|5,705|\n\
         |South|987|1,021|1,150|1,233|4,391|\n|Total|4,947|5,161|5,560|5,972|21,640|"
            .into(),
        "Theorem 2. For every x > 0 and n >= 1 we have\n∑ᵢ αᵢ xᵢ ≤ ∫₀¹ f(x) dx + ε, where ‖x‖₂ ≤ 1 and ∀ε > 0\n\
         f(x) = a₀ + a₁x + a₂x² + … + aₙxⁿ, ∂f/∂x = ∇f · e₁\nP(A ∩ B) = P(A) · P(B | A), Σ λₖ = tr(M)"
            .into(),
        "한국어 문서는 새 도서관 계획과 내년 예산에 대해 설명합니다. 회의는 화요일에 열렸습니다.".repeat(2),
        "Новая библиотека откроется летом; комитет утвердил бюджет и план работ на следующий год.".into(),
    ];
    for page in pages {
        assert!(text_trust::judge(&page).is_none(), "{page}");
    }
}

#[test]
fn advice_names_the_ocr_language_that_reads_the_script() {
    let advice = text_trust::unread_script_advice;
    assert_eq!(
        advice(Some("Arabic"), "en", false),
        "consider re-running with --ocr and ocr.lang set to ar"
    );
    assert_eq!(
        advice(Some("Hebrew"), "en", false),
        "no OCR language reads Hebrew, so it is kept as extracted"
    );
    assert_eq!(
        advice(Some("Arabic"), "en", true),
        "OCR with ocr.lang en does not read Arabic, so the text layer is kept; set ocr.lang to ar to OCR it"
    );
    assert_eq!(
        advice(Some("Hebrew"), "zh", true),
        "OCR with ocr.lang zh does not read Hebrew, so the text layer is kept"
    );
}

// --- authored PDFs ------------------------------------------------------------

/// A PDF string literal of raw bytes.
fn literal(bytes: &[u8]) -> String {
    let mut out = String::from("(");
    for &byte in bytes {
        match byte {
            b'(' | b')' | b'\\' => {
                out.push('\\');
                out.push(byte as char);
            }
            0x20..=0x7E => out.push(byte as char),
            _ => out.push_str(&format!("\\{byte:03o}")),
        }
    }
    out.push(')');
    out
}

/// The pages of lines (each a byte string) shown in Helvetica with
/// WinAnsiEncoding; with `map`, a ToUnicode CMap reads each printable byte
/// as itself or as `map` names it, as a font whose map lies does.
fn helvetica_pdf(pages: &[Vec<Vec<u8>>], map: Option<&[(u8, String)]>) -> Vec<u8> {
    let mut pdf = lopdf::Document::with_version("1.4");
    let mut font = dictionary! {
        "Type" => "Font", "Subtype" => "Type1", "BaseFont" => "Helvetica",
        "Encoding" => "WinAnsiEncoding",
    };
    if let Some(map) = map {
        let mut entries: Vec<(u8, String)> = (0x20u8..0x7F)
            .map(|b| (b, (b as char).to_string()))
            .collect();
        for (byte, text) in map {
            entries.retain(|(own, _)| own != byte);
            entries.push((*byte, text.clone()));
        }
        entries.sort();
        let mut cmap = String::from(
            "/CIDInit /ProcSet findresource begin 12 dict begin begincmap\n\
             /CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def\n\
             /CMapName /Synthetic def /CMapType 2 def\n\
             1 begincodespacerange <00> <FF> endcodespacerange\n",
        );
        for chunk in entries.chunks(100) {
            cmap.push_str(&format!("{} beginbfchar\n", chunk.len()));
            for (byte, text) in chunk {
                let units: String = text
                    .encode_utf16()
                    .map(|unit| format!("{unit:04X}"))
                    .collect();
                cmap.push_str(&format!("<{byte:02X}> <{units}>\n"));
            }
            cmap.push_str("endbfchar\n");
        }
        cmap.push_str("endcmap CMapName currentdict /CMap defineresource pop end end\n");
        let to_unicode = pdf.add_object(Stream::new(dictionary! {}, cmap.into_bytes()));
        font.set("ToUnicode", to_unicode);
    }
    let font = pdf.add_object(font);
    let content = |lines: &Vec<Vec<u8>>| {
        lines
            .iter()
            .enumerate()
            .map(|(index, line)| {
                format!(
                    "BT /F1 12 Tf 72 {} Td {} Tj ET\n",
                    720 - 20 * index,
                    literal(line)
                )
            })
            .collect::<String>()
    };
    finish_pdf(pdf, font, pages.iter().map(content).collect())
}

/// The pages of lines in an Identity-H font whose ToUnicode CMap names each
/// glyph's character: any text, as a browser prints it.
fn unicode_pdf(pages: &[Vec<String>]) -> Vec<u8> {
    let mut glyphs: Vec<char> = pages
        .iter()
        .flatten()
        .flat_map(|line| line.chars())
        .collect();
    glyphs.sort_unstable();
    glyphs.dedup();
    let cid = |c: char| glyphs.binary_search(&c).unwrap() + 1;
    let mut cmap = String::from(
        "/CIDInit /ProcSet findresource begin 12 dict begin begincmap\n\
         /CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def\n\
         /CMapName /Adobe-Identity-UCS def /CMapType 2 def\n\
         1 begincodespacerange <0000> <FFFF> endcodespacerange\n",
    );
    for chunk in glyphs.chunks(100) {
        cmap.push_str(&format!("{} beginbfchar\n", chunk.len()));
        for &c in chunk {
            let units: String = c
                .encode_utf16(&mut [0; 2])
                .iter()
                .map(|u| format!("{u:04X}"))
                .collect();
            cmap.push_str(&format!("<{:04X}> <{units}>\n", cid(c)));
        }
        cmap.push_str("endbfchar\n");
    }
    cmap.push_str("endcmap CMapName currentdict /CMap defineresource pop end end\n");
    let mut pdf = lopdf::Document::with_version("1.4");
    let to_unicode = pdf.add_object(Stream::new(dictionary! {}, cmap.into_bytes()));
    let descendant = pdf.add_object(dictionary! {
        "Type" => "Font", "Subtype" => "CIDFontType2", "BaseFont" => "NotoSansSC",
        "CIDSystemInfo" => dictionary! {
            "Registry" => Object::string_literal("Adobe"),
            "Ordering" => Object::string_literal("Identity"),
            "Supplement" => 0,
        },
        "DW" => 1000,
    });
    let font = pdf.add_object(dictionary! {
        "Type" => "Font", "Subtype" => "Type0", "BaseFont" => "NotoSansSC",
        "Encoding" => "Identity-H",
        "DescendantFonts" => vec![Object::Reference(descendant)],
        "ToUnicode" => to_unicode,
    });
    let contents = pages
        .iter()
        .map(|lines| {
            lines
                .iter()
                .enumerate()
                .map(|(index, line)| {
                    let hex: String = line.chars().map(|c| format!("{:04X}", cid(c))).collect();
                    format!("BT /F1 11 Tf 60 {} Td <{hex}> Tj ET\n", 720 - 18 * index)
                })
                .collect::<String>()
        })
        .collect();
    finish_pdf(pdf, font, contents)
}

fn finish_pdf(mut pdf: lopdf::Document, font: ObjectId, contents: Vec<String>) -> Vec<u8> {
    let tree = pdf.new_object_id();
    let mut kids = Vec::new();
    for content in contents {
        let content = pdf.add_object(Stream::new(dictionary! {}, content.into_bytes()));
        kids.push(Object::Reference(pdf.add_object(dictionary! {
            "Type" => "Page", "Parent" => tree, "Contents" => content,
            "Resources" => dictionary! { "Font" => dictionary! { "F1" => font } },
        })));
    }
    pdf.objects.insert(
        tree,
        dictionary! {
            "Type" => "Pages", "Count" => kids.len() as i64, "Kids" => kids,
            "MediaBox" => vec![0.into(), 0.into(), 612.into(), 792.into()],
        }
        .into(),
    );
    let catalog = pdf.add_object(dictionary! { "Type" => "Catalog", "Pages" => tree });
    pdf.trailer.set("Root", catalog);
    let mut bytes = Vec::new();
    pdf.save_to(&mut bytes).unwrap();
    bytes
}

/// A page of right-to-left `text`, its letters drawn by Helvetica's ASCII
/// glyphs and read through the map: in display order (the reader turns it
/// into reading order), or in reading order, as a producer that does not
/// order right-to-left text writes it.
fn rtl_pdf(text: &str, reading_order_stored: bool) -> Vec<u8> {
    let mut letters: Vec<char> = text.chars().filter(|c| *c != ' ').collect();
    letters.sort_unstable();
    letters.dedup();
    let codes = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    assert!(letters.len() <= codes.len());
    let code = |c: char| {
        letters
            .binary_search(&c)
            .map_or(c as u8, |index| codes[index])
    };
    let drawn: Vec<char> = if reading_order_stored {
        text.chars().collect()
    } else {
        text.chars().rev().collect()
    };
    let line: Vec<u8> = drawn.into_iter().map(code).collect();
    let map: Vec<(u8, String)> = letters
        .iter()
        .enumerate()
        .map(|(index, letter)| (codes[index], letter.to_string()))
        .collect();
    helvetica_pdf(&[vec![line; 6]], Some(&map))
}

fn ascii_lines(lines: &[&str]) -> Vec<Vec<u8>> {
    lines.iter().map(|line| line.as_bytes().to_vec()).collect()
}

/// Windows-1252 bytes of a line.
fn ansi(line: &str) -> Vec<u8> {
    encoding_rs::WINDOWS_1252.encode(line).0.into_owned()
}

fn sources(document: &Document) -> serde_json::Value {
    document.metadata[PAGE_SOURCES].clone()
}

#[test]
fn clean_pdf_pages_keep_their_text_and_say_it_was_trusted() {
    let english = helvetica_pdf(&[ascii_lines(&LATIN_LINES)], None);
    let table = helvetica_pdf(
        &[ascii_lines(&[
            "Region      Q1      Q2      Q3      Q4     Total",
            "North     1,204   1,388   1,502   1,611    5,705",
            "South       987   1,021   1,150   1,233    4,391",
            "East      2,301   2,250   2,410   2,598    9,559",
            "West        455     502     498     530    1,985",
            "Total     4,947   5,161   5,560   5,972   21,640",
        ])],
        None,
    );
    let lines = |lines: &[&str]| {
        lines
            .iter()
            .map(|line| line.to_string())
            .collect::<Vec<_>>()
    };
    let unicode = unicode_pdf(&[
        vec![CHINESE.to_string(); 8],
        vec![JAPANESE.to_string(); 8],
        lines(&[
            "使用以下命令安装依赖，然后运行测试：",
            "uv sync --all-extras && uv run pytest -q",
            "def parse(path: str) -> dict[str, int]:",
            "    return {k: len(v) for k, v in load(path).items()}",
            "返回值是一个字典，键为文件名，值为行数。",
            "See the README for the full list of options.",
        ]),
        lines(&[
            "Theorem 2. For every x > 0 and n >= 1 we have",
            "∑ᵢ αᵢ xᵢ ≤ ∫₀¹ f(x) dx + ε, where ‖x‖₂ ≤ 1 and ∀ε > 0",
            "f(x) = a₀ + a₁x + a₂x² + … + aₙxⁿ, ∂f/∂x = ∇f · e₁",
            "Proof. Let y = 2x + 3 and z = (y - 1) / 2; then z = x + 1.",
            "P(A ∩ B) = P(A) · P(B | A), Σ λₖ = tr(M), det(M) = Π λₖ",
            "This completes the proof of the inequality stated above.",
        ]),
        vec![HEBREW.chars().rev().collect::<String>(); 6],
    ]);
    for bytes in [english, table, unicode] {
        let pages = extract_pages(&bytes).unwrap();
        for page in &pages.pages {
            assert!(
                page.layer_failure.is_none() && !page.needs_ocr,
                "page {}: {:?}\n{}",
                page.number,
                page.layer_failure,
                page.markdown
            );
        }
        let count = pages.pages.len();
        let document = pages.finish().unwrap();
        assert!(
            !document.warnings.iter().any(|w| w.contains("text layer")),
            "{:?}",
            document.warnings
        );
        let trusted: Vec<_> = (1..=count)
            .map(|page| json!({"page": page, "source": "text", "reason": TRUSTED}))
            .collect();
        assert_eq!(sources(&document), json!(trusted));
    }
}

#[test]
fn mojibake_and_cid_code_pages_go_to_ocr_like_a_scan() {
    let mojibake = vec![ansi(MOJIBAKE); 4];
    let cids =
        vec![b"(cid:41)(cid:72)(cid:79)(cid:79)(cid:82) (cid:90)(cid:82)(cid:85)".to_vec(); 4];
    let bytes = helvetica_pdf(&[ascii_lines(&LATIN_LINES), mojibake, cids], None);
    let pages = extract_pages(&bytes).unwrap();
    assert!(pages.pages[0].layer_failure.is_none());
    for (page, kind) in pages.pages[1..].iter().zip([
        "mojibake, UTF-8 read as Windows-1252",
        "raw glyph codes instead of text",
    ]) {
        assert!(page.needs_ocr, "{}", page.markdown);
        assert_eq!(page.layer_failure.as_ref().map(|f| f.kind), Some(kind));
        assert_eq!(page.ocr_reason, Some(format!("{FAILED}{kind}")));
    }
    let document = pages.finish().unwrap();
    assert!(
        document
            .markdown
            .contains("Members asked how the opening hours")
    );
    assert!(!document.markdown.contains("cafÃ©") && !document.markdown.contains("(cid:"));
    assert!(
        document.warnings.contains(&format!(
            "PDF page 2 (1 of 3): native text was not recovered ({FAILED}mojibake, UTF-8 read as Windows-1252); OCR is required to read it. Run again with --ocr."
        )),
        "{:?}",
        document.warnings
    );
    assert_eq!(document.metadata[UNREAD_PAGES], json!([2, 3]));
    let sources = sources(&document);
    assert_eq!(sources[0]["reason"], TRUSTED);
    assert_eq!(sources[1]["source"], "text");
    assert_eq!(
        sources[1]["reason"],
        format!("{FAILED}mojibake, UTF-8 read as Windows-1252 (16 words)")
    );
    assert_eq!(
        sources[2]["reason"],
        format!("{FAILED}raw glyph codes instead of text (32 (cid:NN) codes)")
    );
}

#[test]
fn a_font_map_into_unassigned_code_points_goes_to_ocr() {
    // The map sends the commonest letters where Unicode has nothing.
    let map: Vec<(u8, String)> = b"etaoinsr"
        .iter()
        .zip(0x0378u32..)
        .map(|(&byte, code)| (byte, char::from_u32(code).unwrap().to_string()))
        .collect();
    let bytes = helvetica_pdf(&[ascii_lines(&LATIN_LINES)], Some(&map));
    let pages = extract_pages(&bytes).unwrap();
    let page = &pages.pages[0];
    assert!(page.needs_ocr, "{}", page.markdown);
    assert_eq!(
        page.layer_failure.as_ref().map(|f| f.kind),
        Some("a broken font map")
    );
}

#[test]
fn right_to_left_pages_in_reverse_order_fail_and_keep_their_layer_when_ocr_cannot_read_them() {
    for (text, script) in [(HEBREW, "Hebrew"), (ARABIC, "Arabic")] {
        let good = extract_pages(&rtl_pdf(text, false)).unwrap();
        assert!(
            good.pages[0].layer_failure.is_none(),
            "{}",
            good.pages[0].markdown
        );
        let bad = extract_pages(&rtl_pdf(text, true)).unwrap();
        let page = &bad.pages[0];
        let failure = page
            .layer_failure
            .as_ref()
            .unwrap_or_else(|| panic!("{}", page.markdown));
        assert_eq!(failure.script, Some(script));
        // The default OCR language reads neither: the layer stays.
        assert!(!page.needs_ocr);
        let markdown = page.markdown.clone();
        let document = bad.finish().unwrap();
        assert!(document.markdown.contains(markdown.trim()));
        let advice = if script == "Hebrew" {
            "no OCR language reads Hebrew, so it is kept as extracted".to_string()
        } else {
            "consider re-running with --ocr and ocr.lang set to ar".to_string()
        };
        assert!(
            document.warnings.contains(&format!(
                "PDF page 1 (1 of 1): the text layer reads wrong (right-to-left text in reverse order); {advice}."
            )),
            "{:?}",
            document.warnings
        );
        assert_eq!(
            sources(&document),
            json!([{"page": 1, "source": "text", "reason": format!("{FAILED}right-to-left text in reverse order")}])
        );
    }
}

#[test]
fn an_ocr_language_that_reads_the_script_sends_the_page_to_ocr() {
    let bytes = rtl_pdf(ARABIC, true);
    let cfg = json!({"ocr": {"lang": "ar"}});
    let pages = extract_pages_with_config(&bytes, &cfg).unwrap();
    assert!(pages.pages[0].needs_ocr);
    assert_eq!(
        pages.pages[0].ocr_reason.as_deref(),
        Some(format!("{FAILED}right-to-left text in reverse order").as_str())
    );
    // Its only page unread, the document asks for --ocr as a scan does.
    let error = pages.finish().unwrap_err().to_string();
    assert!(error.contains("run again with --ocr"), "{error}");
    // With --ocr and a language that does not read it, the note says so.
    let cfg = json!({"ocr": {"enabled": true, "lang": "zh"}});
    let document = extract_pages_with_config(&bytes, &cfg)
        .unwrap()
        .finish()
        .unwrap();
    assert_eq!(
        sources(&document)[0]["reason"],
        format!(
            "{FAILED}right-to-left text in reverse order; kept, since OCR with ocr.lang zh does not read Arabic"
        )
    );
    assert!(
        document.warnings.iter().any(|w| w.ends_with(
            "OCR with ocr.lang zh does not read Arabic, so the text layer is kept; set ocr.lang to ar to OCR it."
        )),
        "{:?}",
        document.warnings
    );
}

#[test]
fn page_sources_say_where_the_text_came_from_after_ocr() {
    let bytes = helvetica_pdf(
        &[
            ascii_lines(&LATIN_LINES),
            vec![ansi(MOJIBAKE); 4],
            vec![],
            ascii_lines(&LATIN_LINES),
        ],
        None,
    );
    let mut pages = extract_pages(&bytes).unwrap();
    assert_eq!(pages.pages[2].layer_reason(), NO_TEXT_LAYER);
    // OCR reads the failed page and the empty one, as `pdf_media` does: it
    // notes why first, then the recognition replaces the reader's verdict.
    for (index, text) in [(1, "The café served crème brûlée."), (2, "")] {
        let page = &mut pages.pages[index];
        page.ocr_note = Some(page.layer_reason());
        page.markdown = text.into();
        page.needs_ocr = false;
        page.ocr_reason = None;
        page.ocr_completed = true;
    }
    pages.pages[3]
        .asset_ocr
        .insert("chart.png".into(), "Revenue by quarter".into());
    pages.pages[0].vision_read = false;
    let document = pages.finish_with_media().unwrap();
    assert_eq!(
        sources(&document),
        json!([
            {"page": 1, "source": "text", "reason": TRUSTED},
            {"page": 2, "source": "ocr", "reason": format!("{FAILED}mojibake, UTF-8 read as Windows-1252 (16 words)")},
            {"page": 3, "source": "ocr", "reason": format!("{NO_TEXT_LAYER}; OCR found no text")},
            {"page": 4, "source": "mixed", "reason": format!("{TRUSTED}; text in its pictures OCR'd")},
        ])
    );
    assert!(
        !document
            .warnings
            .iter()
            .any(|w| w.contains("not recovered")),
        "{:?}",
        document.warnings
    );
}

#[test]
fn a_page_read_again_by_ocr_drops_the_kept_layer_warning() {
    let mut pages = extract_pages(&rtl_pdf(HEBREW, true)).unwrap();
    let page = &mut pages.pages[0];
    page.ocr_note = Some(OCR_FORCED.into());
    page.ocr_completed = true;
    let document = pages.finish_with_media().unwrap();
    assert!(
        !document.warnings.iter().any(|w| w.contains("reads wrong")),
        "{:?}",
        document.warnings
    );
    assert_eq!(
        sources(&document),
        json!([{"page": 1, "source": "ocr", "reason": OCR_FORCED}])
    );
}
