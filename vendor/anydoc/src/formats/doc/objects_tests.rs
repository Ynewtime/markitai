//! Regression inputs use real CFB streams and fields, with independently
//! controlled references and object facts. No Office process is executed.

use super::*;
use crate::model::CellSlot;
use std::io::Write;

const CHART: &str = r#"<office:document-content
 xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
 xmlns:chart="urn:oasis:names:tc:opendocument:xmlns:chart:1.0"
 xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"
 xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0">
 <office:body><office:chart><chart:chart>
 <table:table><table:table-row><table:table-cell/><table:table-cell><text:p>Revenue</text:p></table:table-cell></table:table-row>
 <table:table-row><table:table-cell><text:p>North</text:p></table:table-cell><table:table-cell><text:p>1250.50</text:p></table:table-cell></table:table-row>
 </table:table></chart:chart></office:chart></office:body></office:document-content>"#;

fn chart() -> Vec<u8> {
    let mut zip = zip::ZipWriter::new(Cursor::new(Vec::new()));
    zip.start_file("content.xml", zip::write::SimpleFileOptions::default()).unwrap();
    zip.write_all(CHART.as_bytes()).unwrap();
    zip.finish().unwrap().into_inner()
}

fn pool(objects: &[(i32, &[u8])]) -> Vec<u8> {
    let mut ole = cfb::CompoundFile::create(Cursor::new(Vec::new())).unwrap();
    ole.create_storage("/ObjectPool").unwrap();
    for (id, data) in objects {
        let path = format!("/ObjectPool/_{id}");
        ole.create_storage(&path).unwrap();
        ole.create_stream(format!("{path}/package_stream")).unwrap().write_all(data).unwrap();
    }
    ole.into_inner().into_inner()
}

fn props(id: i32) -> Vec<u8> {
    let mut props = vec![0x55, 0x08, 1, 0x0A, 0x08, 1, 0x56, 0x08, 1, 0x03, 0x6A];
    props.extend(id.to_le_bytes());
    props
}

fn assembler<'a>(text: &str, bytes: &'a [u8], metadata: objects::Field) -> Assembler<'a> {
    let chars: Vec<_> = text.chars().collect();
    let count = chars.len();
    let fields = chars
        .iter()
        .enumerate()
        .filter(|(_, c)| **c == '\u{14}')
        .map(|(index, _)| (index, metadata))
        .collect();
    Assembler {
        text: TextStream {
            chars,
            fcs: (0..count as u32).collect(),
            cps: (0..count as u32).collect(),
            piece_of: vec![0; count],
        },
        chpx: Runs::new(vec![Run {
            fc_start: 0,
            fc_end: count as u32,
            props: RunProps { chpx: props(7), ..RunProps::default() },
        }]),
        papx: Runs::new(Vec::new()),
        stylesheet: Stylesheet::default(),
        lists: Lists::default(),
        prcs: Vec::new(),
        piece_prcs: vec![None],
        note_refs: HashMap::new(),
        counters: std::cell::RefCell::new(Counters::default()),
        data: Vec::new(),
        assets: std::cell::RefCell::new(AssetSink::new()),
        fields: objects::Fields::from_separators(fields),
        textboxes: objects::Textboxes::default(),
        objects: std::cell::RefCell::new(Some(objects::Objects::new(
            cfb::CompoundFile::open(Cursor::new(bytes)).unwrap(),
        ))),
    }
}

fn known() -> objects::Field {
    objects::Field { kind: Some(0x3A), end_flags: Some(0x80), ..Default::default() }
}

fn md(blocks: Vec<Block>) -> String {
    crate::render::markdown::document_to_markdown(&Document { blocks, ..Document::default() })
}

#[test]
fn references_keep_chart_in_paragraph_order_and_never_read_an_orphan() {
    let a = chart();
    let orphan = CHART.replace("Revenue", "ORPHAN DATA");
    let mut zip = zip::ZipWriter::new(Cursor::new(Vec::new()));
    zip.start_file("content.xml", zip::write::SimpleFileOptions::default()).unwrap();
    zip.write_all(orphan.as_bytes()).unwrap();
    let b = zip.finish().unwrap().into_inner();
    let bytes = pool(&[(7, &a), (99, &b)]);
    let text = "Before\rChart here\u{13} EMBED LibreOffice.ChartDocument.1 \u{14}\u{15}\rAfter\r";
    let reader = assembler(text, &bytes, known());
    let output = md(reader.build_blocks(0, text.len(), None, None).unwrap());
    assert!(output.find("Before").unwrap() < output.find("Chart here").unwrap());
    assert!(output.find("Chart here").unwrap() < output.find("Revenue").unwrap());
    assert!(output.find("Revenue").unwrap() < output.find("After").unwrap());
    assert!(output.contains("| North | 1250.50 |"));
    assert_eq!(output.matches("Revenue").count(), 1);
    assert!(!output.contains("ORPHAN DATA"));
}

#[test]
fn chart_stays_inside_its_list_item_and_its_table_cell() {
    let a = chart();
    let bytes = pool(&[(7, &a)]);
    let text = "Item\u{13} EMBED chart \u{14}\u{15}\rSecond\r";
    let mut reader = assembler(text, &bytes, known());
    reader.papx = Runs::new(vec![Run {
        fc_start: 0,
        fc_end: text.len() as u32,
        props: RunProps {
            pap: PapDelta { ilfo: Some(1), ..PapDelta::default() },
            ..RunProps::default()
        },
    }]);
    let blocks = reader.build_blocks(0, text.len(), None, None).unwrap();
    let [Block::List(list)] = blocks.as_slice() else { panic!("{blocks:?}") };
    assert_eq!(list.items.len(), 2);
    assert!(list.items[0].blocks.iter().any(|block| matches!(block, Block::Table(_))));
    assert!(!list.items[1].blocks.iter().any(|block| matches!(block, Block::Table(_))));

    let text = "Cell\u{13} EMBED chart \u{14}\u{15}\u{7}\u{7}";
    let mut reader = assembler(text, &bytes, known());
    let mut pap = PapDelta { in_table: Some(true), ..PapDelta::default() };
    let end = text.chars().count() as u32;
    reader.papx = Runs::new(vec![
        Run {
            fc_start: 0,
            fc_end: end - 1,
            props: RunProps { pap: pap.clone(), ..RunProps::default() },
        },
        {
            pap.ttp = Some(true);
            pap.tap = Some(Tap {
                boundaries: vec![0, 4000],
                cells: vec![sprm::TapCell::default()],
                ..Tap::default()
            });
            Run { fc_start: end - 1, fc_end: end, props: RunProps { pap, ..RunProps::default() } }
        },
    ]);
    let blocks = reader.build_blocks(0, end as usize, None, None).unwrap();
    let [Block::Table(table)] = blocks.as_slice() else { panic!("{blocks:?}") };
    let CellSlot::Origin(cell) = &table.grid[0][0] else { panic!("expected source cell") };
    assert!(cell.blocks.iter().any(|block| matches!(block, Block::Table(_))));
}

#[test]
fn nested_objects_in_an_instruction_never_leak_and_two_visible_references_repeat_at_their_positions()
 {
    let a = chart();
    let bytes = pool(&[(7, &a)]);
    let text = "\u{13} IF \u{13} EMBED chart \u{14}\u{15}\u{14}Result\u{15}\r";
    let output =
        md(assembler(text, &bytes, known()).build_blocks(0, text.len(), None, None).unwrap());
    assert_eq!(output.trim(), "Result");
    let text = "\u{13} EMBED chart \u{14}\u{15}\rMiddle\r\u{13} EMBED chart \u{14}\u{15}\r";
    let reader = assembler(text, &bytes, known());
    let output = md(reader.build_blocks(0, text.len(), None, None).unwrap());
    assert_eq!(output.matches("Revenue").count(), 2);
    let middle = output.find("Middle").unwrap();
    assert!(output.find("Revenue").unwrap() < middle && output.rfind("Revenue").unwrap() > middle);
}

#[test]
fn zombie_private_wrong_type_and_malformed_metadata_do_not_expose_stored_data() {
    let a = chart();
    let bytes = pool(&[(7, &a)]);
    let text = "Visible\u{13} EMBED chart \u{14}\u{15}\r";
    for metadata in [
        objects::Field { end_flags: Some(0x82), ..known() },
        objects::Field { end_flags: Some(0xA0), ..known() },
        objects::Field { kind: Some(0x58), ..known() },
        objects::Field { malformed_metadata: true, ..known() },
    ] {
        let reader = assembler(text, &bytes, metadata);
        assert_eq!(md(reader.build_blocks(0, text.len(), None, None).unwrap()).trim(), "Visible");
        assert!(!reader.objects.borrow().as_ref().unwrap().warnings.is_empty());
    }
}

#[test]
fn missing_index_recovers_with_warning_but_invalid_properties_do_not() {
    let a = chart();
    let bytes = pool(&[(7, &a)]);
    let text = "\u{13} EMBED chart \u{14}\u{15}\r";
    let mut reader = assembler(text, &bytes, objects::Field::default());
    assert!(md(reader.build_blocks(0, text.len(), None, None).unwrap()).contains("1250.50"));
    assert!(reader.objects.borrow().as_ref().unwrap().warnings[0].contains("field index omits"));
    reader.prcs = vec![vec![0x0A, 0x08, 0]];
    reader.piece_prcs = vec![Some(0)];
    assert!(!md(reader.build_blocks(0, text.len(), None, None).unwrap()).contains("1250.50"));
}

#[test]
fn missing_or_corrupt_package_has_a_warning_and_signed_names_resolve_exactly() {
    let text = "\u{13} EMBED chart \u{14}\u{15}\r";
    let bytes = pool(&[(99, b"not a package")]);
    let reader = assembler(text, &bytes, known());
    assert!(
        reader
            .build_blocks(0, text.len(), None, None)
            .unwrap()
            .iter()
            .all(|b| !matches!(b, Block::Table(_)))
    );
    assert!(
        reader.objects.borrow().as_ref().unwrap().warnings[0]
            .contains("missing, unsupported or unreadable")
    );
    let a = chart();
    let bytes = pool(&[(-1, &a)]);
    let mut reader = assembler(text, &bytes, known());
    reader.chpx.runs[0].props.chpx = props(-1);
    assert!(md(reader.build_blocks(0, text.len(), None, None).unwrap()).contains("1250.50"));
}

#[test]
fn repeated_object_text_budget_propagates_instead_of_cloning_past_the_limit() {
    let xml = CHART.replace("Revenue", &"x".repeat(65_536));
    let mut zip = zip::ZipWriter::new(Cursor::new(Vec::new()));
    zip.start_file("content.xml", zip::write::SimpleFileOptions::default()).unwrap();
    zip.write_all(xml.as_bytes()).unwrap();
    let package = zip.finish().unwrap().into_inner();
    let bytes = pool(&[(7, &package)]);
    let reader = assembler("", &bytes, known());
    let mut objects = reader.objects.borrow_mut();
    let objects = objects.as_mut().unwrap();
    let mut limit = None;
    for cp in 0..2_000 {
        match objects.blocks(7, "EMBED chart", known(), cp) {
            Ok(_) => {}
            Err(error) => {
                limit = Some(error);
                break;
            }
        }
    }
    assert!(matches!(
        limit,
        Some(ConvertError::ResourceLimit { limit: "max_expansion_text_bytes", .. })
    ));
}

#[test]
fn field_pairs_use_utf16_cps_and_reject_unmatched_or_repeated_separators() {
    let chars: Vec<_> =
        "😀\u{13} EMBED x \u{14}\u{15}\r\u{14}\r\u{13} EMBED x \u{14}\u{14}\u{15}\r"
            .chars()
            .collect();
    let mut cp = 0;
    let cps = chars
        .iter()
        .map(|c| {
            let at = cp;
            cp += c.len_utf16() as u32;
            at
        })
        .collect();
    let text = TextStream { chars, fcs: Vec::new(), cps, piece_of: Vec::new() };
    let fields = objects::fields(&[0; 0x222], &[], &text, &[(0x11A, 0, cp as usize)]).unwrap();
    assert_eq!(fields.at_separator.len(), 1);
    assert!(fields.at_separator.values().all(|field| field.end_flags.is_none()));
}

#[test]
fn field_index_end_flags_and_conflicting_character_positions_are_preserved() {
    let chars: Vec<_> = "\u{13} EMBED x \u{14}\u{15}\r".chars().collect();
    let count = chars.len();
    let separator = chars.iter().position(|&c| c == '\u{14}').unwrap();
    let end = separator + 1;
    let text = TextStream {
        chars,
        fcs: Vec::new(),
        cps: (0..count as u32).collect(),
        piece_of: Vec::new(),
    };
    let mut table = [0u32, separator as u32, end as u32, count as u32]
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect::<Vec<_>>();
    table.extend([0x13, 0x3A, 0x14, 0xFF, 0x15, 0x82]);
    let mut word = vec![0u8; 0x222];
    word[0x11E..0x122].copy_from_slice(&(table.len() as u32).to_le_bytes());
    let parsed = objects::fields(&word, &table, &text, &[(0x11A, 0, count)]).unwrap();
    assert_eq!(parsed.at_separator[&separator].kind, Some(0x3A));
    assert_eq!(parsed.at_separator[&separator].end_flags, Some(0x82));
    assert!(!parsed.at_separator[&separator].malformed_metadata);
    table[4..8].copy_from_slice(&0u32.to_le_bytes());
    let malformed = objects::fields(&word, &table, &text, &[(0x11A, 0, count)]).unwrap();
    assert!(malformed.at_separator[&separator].malformed_metadata);
}

#[test]
fn piece_property_toggles_can_clear_the_field_marker_or_object_identity() {
    let a = chart();
    let bytes = pool(&[(7, &a)]);
    let text = "\u{13} EMBED chart \u{14}\u{15}\r";
    for isprm in [0x4B, 0x75, 0x76] {
        let mut reader = assembler(text, &bytes, known());
        reader.prcs = vec![prm0_grpprl(isprm << 1).unwrap()];
        reader.piece_prcs = vec![Some(0)];
        assert!(!md(reader.build_blocks(0, text.len(), None, None).unwrap()).contains("1250.50"));
    }
}
// Metadata is serialized into a real PLC and interpreted by the production
// prepass. Each field has its own kind/nested/separator/end flags.
fn field_plc(text: &str, overrides: &[(usize, u8)]) -> Vec<u8> {
    let mut records = Vec::new();
    let mut stack: Vec<(bool, bool)> = Vec::new();
    let mut cp = 0u32;
    for (index, c) in text.chars().enumerate() {
        let value = match c {
            '\u{13}' => {
                let nested = !stack.is_empty();
                stack.push((nested, false));
                let tail: String = text.chars().skip(index + 1).take(8).collect();
                Some(if tail.trim_start().starts_with("EMBED") { 0x3A } else { 0x07 })
            }
            '\u{14}' => {
                if let Some(last) = stack.last_mut() {
                    last.1 = true;
                }
                Some(0xFF)
            }
            '\u{15}' => {
                let (nested, separator) = stack.pop().unwrap();
                Some((if nested { 0x40 } else { 0 }) | if separator { 0x80 } else { 0 })
            }
            _ => None,
        };
        if let Some(flags) = value {
            records.push((
                cp,
                c as u8,
                overrides
                    .iter()
                    .find(|&&(at, _)| at == index)
                    .map(|&(_, flags)| flags)
                    .unwrap_or(flags),
            ));
        }
        cp += c.len_utf16() as u32;
    }
    let mut plc: Vec<u8> =
        records.iter().map(|&(cp, _, _)| cp).chain([cp + 17]).flat_map(u32::to_le_bytes).collect();
    for (_, character, flags) in records {
        plc.extend([character, flags]);
    }
    plc
}

fn put_part(word: &mut [u8], table: &mut Vec<u8>, fib: usize, data: &[u8]) {
    word[fib..fib + 4].copy_from_slice(&(table.len() as u32).to_le_bytes());
    word[fib + 4..fib + 8].copy_from_slice(&(data.len() as u32).to_le_bytes());
    table.extend(data);
}

fn indexed<'a>(text: &str, bytes: &'a [u8], overrides: &[(usize, u8)]) -> Assembler<'a> {
    let mut reader = assembler(text, bytes, known());
    let mut word = vec![0; 0x310];
    let mut table = Vec::new();
    put_part(&mut word, &mut table, 0x11A, &field_plc(text, overrides));
    reader.fields =
        objects::fields(&word, &table, &reader.text, &[(0x11A, 0, text.chars().count())]).unwrap();
    reader
}

#[test]
fn private_and_zombie_ancestors_suppress_stored_reads_in_and_across_paragraphs() {
    let package = chart();
    let bytes = pool(&[(7, &package)]);
    for boundary in ["", "\r", "\u{7}", "\u{c}"] {
        let text = format!(
            "Before\u{13} IF true \u{14}{boundary}\u{13} EMBED chart \u{14}\u{15}\u{15}\rAfter\r"
        );
        let outer_end =
            text.chars().collect::<Vec<_>>().iter().rposition(|&c| c == '\u{15}').unwrap();
        for flags in [0xA0, 0x82] {
            let reader = indexed(&text, &bytes, &[(outer_end, flags)]);
            let output = md(reader.build_blocks(0, text.chars().count(), None, None).unwrap());
            assert!(!output.contains("Revenue"), "{flags:x}: {output}");
            assert!(reader.objects.borrow().as_ref().unwrap().decoded_count() == 0);
            assert!(
                reader
                    .objects
                    .borrow()
                    .as_ref()
                    .unwrap()
                    .warnings
                    .iter()
                    .any(|warning| warning.contains("ancestor"))
            );
        }
    }
}

#[test]
fn whole_story_instruction_spans_block_cross_paragraph_objects_but_visible_results_repeat() {
    let package = chart();
    let bytes = pool(&[(7, &package)]);
    for boundary in ["", "\r"] {
        let text = format!(
            "\u{13} IF {boundary}\u{13} EMBED chart \u{14}\u{15}{boundary}\u{14}Shown\u{15}\r"
        );
        let reader = indexed(&text, &bytes, &[]);
        assert!(
            !md(reader.build_blocks(0, text.chars().count(), None, None).unwrap())
                .contains("Revenue")
        );
        assert!(reader.objects.borrow().as_ref().unwrap().decoded_count() == 0);
        let text = format!(
            "\u{13} IF \u{14}{boundary}\u{13} EMBED chart \u{14}\u{15}\u{15}\rMiddle\r\u{13} EMBED chart \u{14}\u{15}\r"
        );
        let reader = indexed(&text, &bytes, &[]);
        let output = md(reader.build_blocks(0, text.chars().count(), None, None).unwrap());
        assert_eq!(output.matches("Revenue").count(), 2, "{output}");
        assert!(output.find("Revenue").unwrap() < output.find("Middle").unwrap());
        assert!(output.rfind("Revenue").unwrap() > output.find("Middle").unwrap());
    }
}

#[test]
fn known_has_separator_and_nested_conflicts_cannot_become_missing_index_recovery() {
    let package = chart();
    let bytes = pool(&[(7, &package)]);
    let text = "\u{13} EMBED chart \u{14}\u{15}\r";
    let end = text.chars().position(|c| c == '\u{15}').unwrap();
    for flags in [0x00, 0x40, 0xC0] {
        let reader = indexed(text, &bytes, &[(end, flags)]);
        assert!(
            !md(reader.build_blocks(0, text.chars().count(), None, None).unwrap())
                .contains("Revenue")
        );
        assert!(reader.objects.borrow().as_ref().unwrap().decoded_count() == 0);
        assert!(
            reader
                .objects
                .borrow()
                .as_ref()
                .unwrap()
                .warnings
                .iter()
                .all(|warning| !warning.contains("index omits"))
        );
    }
    let text = "\u{13} IF \u{14}\u{13} EMBED chart \u{14}\u{15}\u{15}\r";
    let inner_end = text.chars().position(|c| c == '\u{15}').unwrap();
    let reader = indexed(text, &bytes, &[(inner_end, 0x80)]);
    assert!(
        !md(reader.build_blocks(0, text.chars().count(), None, None).unwrap()).contains("Revenue")
    );
}

#[test]
fn plc_sorting_sentinel_conflicts_block_data_but_ignored_character_bits_do_not() {
    let package = chart();
    let bytes = pool(&[(7, &package)]);
    let text = "\u{13} EMBED chart \u{14}\u{15}\r";
    let mut reader = assembler(text, &bytes, known());
    let mut word = vec![0; 0x310];
    let mut table = field_plc(text, &[]);
    // Last CP is only ordered, not constrained to the story count.
    let cp_bytes = 16;
    for n in 0..3 {
        table[cp_bytes + 2 * n] |= 0xE0;
    }
    word[0x11E..0x122].copy_from_slice(&(table.len() as u32).to_le_bytes());
    reader.fields =
        objects::fields(&word, &table, &reader.text, &[(0x11A, 0, text.len())]).unwrap();
    assert!(md(reader.build_blocks(0, text.len(), None, None).unwrap()).contains("Revenue"));
    table[12..16].copy_from_slice(&0u32.to_le_bytes());
    reader.fields =
        objects::fields(&word, &table, &reader.text, &[(0x11A, 0, text.len())]).unwrap();
    assert!(!md(reader.build_blocks(0, text.len(), None, None).unwrap()).contains("Revenue"));
    let word = vec![0; 0x310];
    reader.fields = objects::fields(&word, &[], &reader.text, &[(0x11A, 0, text.len())]).unwrap();
    assert!(md(reader.build_blocks(0, text.len(), None, None).unwrap()).contains("Revenue"));
    assert!(
        reader
            .objects
            .borrow()
            .as_ref()
            .unwrap()
            .warnings
            .iter()
            .any(|w| w.contains("index omits"))
    );
}

fn drawing_record(kind: u16, version: u16, body: &[u8]) -> Vec<u8> {
    let mut data = version.to_le_bytes().into_iter().chain(kind.to_le_bytes()).collect::<Vec<_>>();
    data.extend((body.len() as u32).to_le_bytes());
    data.extend(body);
    data
}

fn textbox_metadata(
    word: &mut [u8],
    table: &mut Vec<u8>,
    boundaries: &[u32],
    boxes: &[(u32, u32, u16)],
    anchors: &[(u32, u32)],
    main_count: u32,
    shapes: &[(u32, u32)],
) {
    assert_eq!(boundaries.len(), boxes.len() + 1);
    let mut plc: Vec<u8> = boundaries.iter().copied().flat_map(u32::to_le_bytes).collect();
    for &(id, chain, reusable) in boxes {
        plc.extend(chain.to_le_bytes());
        plc.extend([0; 4]);
        plc.extend(reusable.to_le_bytes());
        plc.extend(u32::MAX.to_le_bytes());
        plc.extend(id.to_le_bytes());
        plc.extend([0; 4]);
    }
    put_part(word, table, 0x25A, &plc);
    let mut plc: Vec<u8> = anchors
        .iter()
        .map(|&(cp, _)| cp)
        .chain([main_count + 17])
        .flat_map(u32::to_le_bytes)
        .collect();
    for &(_, id) in anchors {
        plc.extend(id.to_le_bytes());
        plc.extend([0; 22]);
    }
    put_part(word, table, 0x1DA, &plc);
    let mut drawing = Vec::new();
    for &(id, flags) in shapes {
        let fsp: Vec<u8> = id.to_le_bytes().into_iter().chain(flags.to_le_bytes()).collect();
        drawing.extend(drawing_record(0xF004, 0xF, &drawing_record(0xF00A, 2, &fsp)));
    }
    let mut drawing_data = drawing_record(0xF000, 0xF, &[]);
    drawing_data.push(0);
    drawing_data.extend(drawing_record(0xF002, 0xF, &drawing));
    put_part(word, table, 0x22A, &drawing_data);
}

fn with_textbox<'a>(
    main: &str,
    boxed: &str,
    bytes: &'a [u8],
    main_flags: &[(usize, u8)],
    box_flags: &[(usize, u8)],
) -> (Assembler<'a>, Vec<u8>, Vec<u8>) {
    let text = format!("{main}{boxed}\r");
    let mut reader = assembler(&text, bytes, known());
    let mut word = vec![0; 0x310];
    let mut table = Vec::new();
    put_part(&mut word, &mut table, 0x11A, &field_plc(main, main_flags));
    put_part(&mut word, &mut table, 0x262, &field_plc(&format!("{boxed}\r"), box_flags));
    let anchor = main.chars().position(|c| c == '\u{8}').unwrap() as u32;
    textbox_metadata(
        &mut word,
        &mut table,
        &[0, boxed.len() as u32, boxed.len() as u32 + 1],
        &[(7, 1, 0), (0, 0, 0)],
        &[(anchor, 7)],
        main.len() as u32,
        &[(7, 0x200)],
    );
    reader.fields = objects::fields(
        &word,
        &table,
        &reader.text,
        &[(0x11A, 0, main.len()), (0x262, main.len(), boxed.len() + 1)],
    )
    .unwrap();
    reader.textboxes = objects::Textboxes::read(
        &word,
        &table,
        &reader.text,
        main.len(),
        main.len(),
        boxed.len() + 1,
    )
    .unwrap();
    (reader, word, table)
}

#[test]
fn a_live_non_ole_textbox_uses_its_exact_main_anchor_and_complete_story_field_index() {
    let package = chart();
    let bytes = pool(&[(7, &package)]);
    let main = "Before\r\u{8}\rAfter\r";
    let boxed = "\u{13} EMBED chart \u{14}\u{15}\r";
    let (reader, _, _) = with_textbox(main, boxed, &bytes, &[], &[]);
    let output = md(reader.build_blocks(0, main.len(), None, None).unwrap());
    assert_eq!(output.matches("Revenue").count(), 1, "{output}");
    assert!(output.find("Before").unwrap() < output.find("Revenue").unwrap());
    assert!(output.find("Revenue").unwrap() < output.find("After").unwrap());
    assert!(reader.objects.borrow().as_ref().unwrap().warnings.is_empty());
}

#[test]
fn textbox_spares_orphans_wrong_anchors_chains_and_deleted_or_duplicate_shapes_are_not_read() {
    let package = chart();
    let bytes = pool(&[(7, &package)]);
    let main = "Before\r\u{8}\rAfter\r";
    let boxed = "\u{13} EMBED chart \u{14}\u{15}\r";
    let anchor = main.chars().position(|c| c == '\u{8}').unwrap() as u32;
    for (id, chain, reuse, shape_flags, duplicate) in [
        (9, 1, 0, 0x200, false),
        (7, 2, 0, 0x200, false),
        (7, 1, 1, 0x200, false),
        (7, 1, 0, 0x208, false),
        (7, 1, 0, 0x200, true),
    ] {
        let (mut reader, mut word, mut table) = with_textbox(main, boxed, &bytes, &[], &[]);
        let shapes = if duplicate {
            vec![(7, shape_flags), (7, shape_flags)]
        } else {
            vec![(7, shape_flags)]
        };
        textbox_metadata(
            &mut word,
            &mut table,
            &[0, boxed.len() as u32, boxed.len() as u32 + 1],
            &[(id, chain, reuse), (7, 1, 0)],
            &[(anchor, 7)],
            main.len() as u32,
            &shapes,
        );
        reader.textboxes = objects::Textboxes::read(
            &word,
            &table,
            &reader.text,
            main.len(),
            main.len(),
            boxed.len() + 1,
        )
        .unwrap();
        assert!(reader.textboxes.ranges.is_empty());
        assert!(!md(reader.build_blocks(0, main.len(), None, None).unwrap()).contains("Revenue"));
        assert!(reader.objects.borrow().as_ref().unwrap().decoded_count() == 0);
    }
    for cp in [0, main.len() as u32 + 1] {
        let (mut reader, mut word, mut table) = with_textbox(main, boxed, &bytes, &[], &[]);
        textbox_metadata(
            &mut word,
            &mut table,
            &[0, boxed.len() as u32, boxed.len() as u32 + 1],
            &[(7, 1, 0), (0, 0, 0)],
            &[(cp, 7)],
            main.len() as u32,
            &[(7, 0x200)],
        );
        reader.textboxes = objects::Textboxes::read(
            &word,
            &table,
            &reader.text,
            main.len(),
            main.len(),
            boxed.len() + 1,
        )
        .unwrap();
        assert!(reader.textboxes.ranges.is_empty());
    }
    let (mut reader, _, _) = with_textbox(main, boxed, &bytes, &[], &[]);
    reader.chpx.runs[0].props.chpx = vec![0x0A, 0x08, 1];
    assert!(!md(reader.build_blocks(0, main.len(), None, None).unwrap()).contains("Revenue"));
    assert!(
        reader.objects.borrow().as_ref().unwrap().warnings.iter().any(|w| w.contains("CFSpec"))
    );
}

#[test]
fn main_and_textbox_private_or_instruction_ancestors_both_block_package_reads() {
    let package = chart();
    let bytes = pool(&[(7, &package)]);
    let boxed = "\u{13} EMBED chart \u{14}\u{15}\r";
    for boundary in ["", "\r"] {
        let main = format!("Before\r\u{13} IF \u{14}{boundary}\u{8}\u{15}\rAfter\r");
        let end = main.chars().position(|c| c == '\u{15}').unwrap();
        for flags in [0xA0, 0x82] {
            let (reader, _, _) = with_textbox(&main, boxed, &bytes, &[(end, flags)], &[]);
            assert!(
                !md(reader.build_blocks(0, main.len(), None, None).unwrap()).contains("Revenue")
            );
            assert!(reader.objects.borrow().as_ref().unwrap().decoded_count() == 0);
        }
        let main = format!("\u{13} IF {boundary}\u{8}{boundary}\u{14}Shown\u{15}\r");
        let (reader, _, _) = with_textbox(&main, boxed, &bytes, &[], &[]);
        assert!(!md(reader.build_blocks(0, main.len(), None, None).unwrap()).contains("Revenue"));
        let main = "Before\r\u{8}\rAfter\r";
        let boxed = format!("\u{13} IF \u{14}{boundary}\u{13} EMBED chart \u{14}\u{15}\u{15}\r");
        let end = boxed.chars().collect::<Vec<_>>().iter().rposition(|&c| c == '\u{15}').unwrap();
        let (reader, _, _) = with_textbox(main, &boxed, &bytes, &[], &[(end, 0xA0)]);
        assert!(!md(reader.build_blocks(0, main.len(), None, None).unwrap()).contains("Revenue"));
        assert!(reader.objects.borrow().as_ref().unwrap().decoded_count() == 0);
    }
}

#[test]
fn two_textboxes_follow_anchor_order_instead_of_storage_or_story_order() {
    let package = chart();
    let bytes = pool(&[(7, &package)]);
    let main = "Start\r\u{8}\rMiddle\r\u{8}\rEnd\r";
    let first = "First box\u{13} EMBED chart \u{14}\u{15}\r";
    let second = "Second box\u{13} EMBED chart \u{14}\u{15}\r";
    let boxed = format!("{first}{second}");
    let (mut reader, mut word, mut table) = with_textbox(main, &boxed, &bytes, &[], &[]);
    let anchors: Vec<_> =
        main.chars().enumerate().filter(|&(_, c)| c == '\u{8}').map(|(n, _)| n as u32).collect();
    textbox_metadata(
        &mut word,
        &mut table,
        &[0, first.len() as u32, boxed.len() as u32, boxed.len() as u32 + 1],
        &[(7, 1, 0), (9, 1, 0), (0, 0, 0)],
        &[(anchors[0], 9), (anchors[1], 7)],
        main.len() as u32,
        &[(7, 0x200), (9, 0x200)],
    );
    reader.textboxes = objects::Textboxes::read(
        &word,
        &table,
        &reader.text,
        main.len(),
        main.len(),
        boxed.len() + 1,
    )
    .unwrap();
    let output = md(reader.build_blocks(0, main.len(), None, None).unwrap());
    assert_eq!(output.matches("Revenue").count(), 2, "{output}");
    let order: Vec<_> = ["Start", "Second box", "Middle", "First box", "End"]
        .iter()
        .map(|token| output.find(token).unwrap())
        .collect();
    assert!(order.windows(2).all(|pair| pair[0] < pair[1]), "{output}");
    assert_eq!(reader.objects.borrow().as_ref().unwrap().decoded_count(), 1);
}

#[test]
fn newly_read_textbox_plain_text_does_not_promote_hidden_results_or_cross_paragraph_instructions() {
    let bytes = pool(&[]);
    let main = "Before\r\u{8}\rAfter\r";
    for boundary in ["", "\r"] {
        let boxed = format!("Visible box\r\u{13} IF \u{14}{boundary}PRIVATE SENTINEL\u{15}\r");
        let end = boxed.chars().position(|c| c == '\u{15}').unwrap();
        let (reader, _, _) = with_textbox(main, &boxed, &bytes, &[], &[(end, 0xA0)]);
        let output = md(reader.build_blocks(0, main.len(), None, None).unwrap());
        assert!(output.contains("Visible box"), "{output}");
        assert!(!output.contains("PRIVATE SENTINEL"), "{output}");
        let boxed = format!("\u{13} IF {boundary}INSTRUCTION SENTINEL\u{14}Visible result\u{15}\r");
        let (reader, _, _) = with_textbox(main, &boxed, &bytes, &[], &[]);
        let output = md(reader.build_blocks(0, main.len(), None, None).unwrap());
        assert!(!output.contains("INSTRUCTION SENTINEL"), "{output}");
        assert!(output.contains("Visible result"), "{output}");
    }
}

#[test]
fn textbox_tables_keep_their_list_item_and_table_cell_instead_of_escaping_to_the_body() {
    let package = chart();
    let bytes = pool(&[(7, &package)]);
    let boxed = "\u{13} EMBED chart \u{14}\u{15}\r";
    let main = "Item\u{8}\rSecond\r";
    let (mut reader, _, _) = with_textbox(main, boxed, &bytes, &[], &[]);
    reader.papx = Runs::new(vec![Run {
        fc_start: 0,
        fc_end: main.len() as u32,
        props: RunProps {
            pap: PapDelta { ilfo: Some(1), ..Default::default() },
            ..Default::default()
        },
    }]);
    let blocks = reader.build_blocks(0, main.len(), None, None).unwrap();
    let [Block::List(list)] = blocks.as_slice() else { panic!("{blocks:?}") };
    assert_eq!(list.items.len(), 2);
    assert!(list.items[0].blocks.iter().any(|b| matches!(b, Block::Table(_))));
    assert!(!list.items[1].blocks.iter().any(|b| matches!(b, Block::Table(_))));
    let main = "Cell\u{8}\u{7}\u{7}";
    let (mut reader, _, _) = with_textbox(main, boxed, &bytes, &[], &[]);
    let end = main.len() as u32;
    let cell = PapDelta { in_table: Some(true), ..Default::default() };
    reader.papx = Runs::new(vec![
        Run {
            fc_start: 0,
            fc_end: end - 1,
            props: RunProps { pap: cell.clone(), ..Default::default() },
        },
        Run {
            fc_start: end - 1,
            fc_end: end,
            props: RunProps {
                pap: PapDelta {
                    ttp: Some(true),
                    tap: Some(Tap {
                        boundaries: vec![0, 4000],
                        cells: vec![sprm::TapCell::default()],
                        ..Default::default()
                    }),
                    ..cell
                },
                ..Default::default()
            },
        },
    ]);
    let blocks = reader.build_blocks(0, main.len(), None, None).unwrap();
    let [Block::Table(table)] = blocks.as_slice() else { panic!("{blocks:?}") };
    let CellSlot::Origin(cell) = &table.grid[0][0] else { panic!("source cell") };
    assert!(cell.blocks.iter().any(|b| matches!(b, Block::Table(_))));
}

#[test]
fn invalid_textbox_boundaries_and_duplicate_anchor_or_range_identities_never_choose_a_first_match()
{
    let package = chart();
    let bytes = pool(&[(7, &package)]);
    let main = "Before\r\u{8}\rAfter\r";
    let one = "\u{13} EMBED chart \u{14}\u{15}\r";
    let boxed = format!("{one}{one}");
    let (reader, mut word, mut table) = with_textbox(main, &boxed, &bytes, &[], &[]);
    let anchor = main.chars().position(|c| c == '\u{8}').unwrap() as u32;
    for (boundaries, boxes, anchors) in [
        (
            vec![0, one.len() as u32, boxed.len() as u32, boxed.len() as u32 + 1],
            vec![(7, 1, 0), (7, 1, 0), (0, 0, 0)],
            vec![(anchor, 7)],
        ),
        (
            vec![0, boxed.len() as u32, boxed.len() as u32 + 1],
            vec![(7, 1, 0), (0, 0, 0)],
            vec![(anchor, 7), (anchor + 1, 7)],
        ),
        (
            vec![0, boxed.len() as u32 + 10, boxed.len() as u32 + 11],
            vec![(7, 1, 0), (0, 0, 0)],
            vec![(anchor, 7)],
        ),
        (vec![0, 0, boxed.len() as u32 + 1], vec![(7, 1, 0), (0, 0, 0)], vec![(anchor, 7)]),
    ] {
        textbox_metadata(
            &mut word,
            &mut table,
            &boundaries,
            &boxes,
            &anchors,
            main.len() as u32,
            &[(7, 0x200)],
        );
        let boxes = objects::Textboxes::read(
            &word,
            &table,
            &reader.text,
            main.len(),
            main.len(),
            boxed.len() + 1,
        )
        .unwrap();
        assert!(boxes.ranges.is_empty());
        assert!(!boxes.warnings.is_empty());
    }
}
