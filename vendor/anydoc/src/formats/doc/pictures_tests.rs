//! Synthetic OfficeArt/PLC graphs test identity resolution independently of
//! production metadata builders. No real document or Office process is needed.
use super::*;
use std::io::Write;

// The extractor preserves compressed image bytes; decoding pixels is a
// downstream concern. Distinct markers make wrong-slot recovery observable.
const PAYLOAD: &[u8] = b"\xFF\xD8SYNTHETIC-PICTURE\xFF\xD9";

fn record(kind: u16, version: u16, body: &[u8]) -> Vec<u8> {
    let mut bytes = version.to_le_bytes().to_vec();
    bytes.extend(kind.to_le_bytes());
    bytes.extend((body.len() as u32).to_le_bytes());
    bytes.extend(body);
    bytes
}

fn put(word: &mut [u8], table: &mut Vec<u8>, fib: usize, bytes: &[u8]) {
    word[fib..fib + 4].copy_from_slice(&(table.len() as u32).to_le_bytes());
    word[fib + 4..fib + 8].copy_from_slice(&(bytes.len() as u32).to_le_bytes());
    table.extend(bytes);
}

fn blip(payload: &[u8]) -> Vec<u8> {
    let mut body = vec![0x42; 16];
    body.push(0xFF);
    body.extend(payload);
    record(0xF01D, 0x46A0, &body)
}

fn entry(word: &mut Vec<u8>, payload: &[u8], embedded: bool) -> Vec<u8> {
    let image = blip(payload);
    let mut fbse = vec![0u8; 36];
    fbse[0] = 5;
    fbse[1] = 5;
    fbse[2..18].fill(0x42);
    fbse[20..24].copy_from_slice(&(image.len() as u32).to_le_bytes());
    fbse[24..28].copy_from_slice(&1u32.to_le_bytes());
    fbse[28..32].copy_from_slice(&(word.len() as u32).to_le_bytes());
    if embedded {
        fbse.extend(image);
    } else {
        word.extend(image);
    }
    record(0xF007, 0x52, &fbse)
}

fn shape(id: u32, flags: u32, pib: u32) -> Vec<u8> {
    let mut identity = id.to_le_bytes().to_vec();
    identity.extend(flags.to_le_bytes());
    let mut body = record(0xF00A, 0x4B2, &identity);
    if pib != 0 {
        let mut property = 0x4104u16.to_le_bytes().to_vec();
        property.extend(pib.to_le_bytes());
        body.extend(record(0xF00B, 0x13, &property));
    }
    record(0xF004, 0xF, &body)
}

fn fixture(
    text: &str,
    anchors: &[(u32, u32)],
    shapes: &[(u32, u32, u32)],
    embedded: bool,
) -> (Vec<u8>, Vec<u8>) {
    let mut word = vec![0; 1024];
    // A valid but unreferenced entry must not become the chosen picture.
    let mut entries = entry(&mut word, b"UNREFERENCED", embedded);
    entries.extend(entry(&mut word, PAYLOAD, embedded));
    let store = record(0xF001, 0x2F, &entries);
    let mut art = record(0xF000, 0xF, &store);
    let mut group = shape(1024, 5, 0);
    for &(id, flags, pib) in shapes {
        group.extend(shape(id, flags, pib));
    }
    art.push(0); // Main drawing, not the header drawing.
    art.extend(record(0xF002, 0xF, &record(0xF003, 0xF, &group)));
    let mut table = Vec::new();
    put(&mut word, &mut table, 0x22A, &art);
    let mut plc: Vec<u8> = anchors.iter().flat_map(|&(cp, _)| cp.to_le_bytes()).collect();
    plc.extend((text.encode_utf16().count() as u32 + 1).to_le_bytes());
    for &(_, id) in anchors {
        plc.extend(id.to_le_bytes());
        plc.extend([0; 22]);
    }
    put(&mut word, &mut table, 0x1DA, &plc);
    (word, table)
}

fn reader<'a>(text: &str, word: &'a [u8], table: &'a [u8]) -> Assembler<'a> {
    let mut cp = 0u32;
    let cps: Vec<_> = text
        .chars()
        .map(|c| {
            let at = cp;
            cp += c.len_utf16() as u32;
            at
        })
        .collect();
    let count = cps.len();
    let text = TextStream {
        chars: text.chars().collect(),
        fcs: cps.clone(),
        cps,
        piece_of: vec![0; count],
    };
    let pictures = pictures::Pictures::read(word, table, &text, cp as usize).unwrap();
    Assembler {
        fields: objects::fields(word, table, &text, &[(0x11A, 0, cp as usize)]).unwrap(),
        text,
        pictures,
        chpx: Runs::new(vec![Run {
            fc_start: 0,
            fc_end: cp,
            props: RunProps { chpx: vec![0x55, 0x08, 1], ..RunProps::default() },
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
        textboxes: objects::Textboxes::default(),
        objects: std::cell::RefCell::new(None),
    }
}

fn assembled_debug(reader: &Assembler<'_>) -> String {
    let blocks = reader.build_blocks(0, reader.text.chars.len(), None, None).unwrap();
    format!("{blocks:?}")
}

#[test]
fn floating_picture_uses_its_slot_and_utf16_anchor_in_body_order() {
    let text = "Before😀\r\u{8}\rAfter\r";
    for embedded in [false, true] {
        let (word, table) = fixture(text, &[(9, 1027)], &[(1027, 0xA00, 2)], embedded);
        let reader = reader(text, &word, &table);
        let output = assembled_debug(&reader);
        assert!(output.find("Before").unwrap() < output.find("Image {").unwrap(), "{output}");
        assert!(output.find("Image {").unwrap() < output.find("After").unwrap(), "{output}");
        let assets = reader.assets.borrow();
        assert_eq!(assets.assets.len(), 1);
        assert_eq!(assets.assets[0].bytes, PAYLOAD);
        assert_eq!(assets.assets[0].media_type, "image/jpeg");
    }
}

#[test]
fn distinct_shapes_can_share_one_asset_without_losing_either_anchor() {
    let text = "A\u{8}\rB\u{8}\r";
    let (word, table) = fixture(text, &[(1, 7), (4, 8)], &[(7, 0x200, 2), (8, 0x200, 2)], false);
    let reader = reader(text, &word, &table);
    assert_eq!(assembled_debug(&reader).matches("Image {").count(), 2);
    assert_eq!(reader.assets.borrow().assets.len(), 1);
}

#[test]
fn wrong_or_duplicate_anchors_deleted_shapes_and_out_of_range_slots_are_not_guessed() {
    let text = "A\u{8}\rB\u{8}\r";
    for (anchors, shapes) in [
        (vec![(0, 7)], vec![(7, 0x200, 2)]),
        (vec![(1, 99)], vec![(7, 0x200, 2)]),
        (vec![(1, 7), (4, 7)], vec![(7, 0x200, 2)]),
        (vec![(1, 7)], vec![(7, 0x208, 2)]),
        (vec![(1, 7)], vec![(7, 0x200, 2), (7, 0x208, 2)]),
        (vec![(1, 7)], vec![(7, 0x200, 3)]),
    ] {
        let (word, table) = fixture(text, &anchors, &shapes, false);
        let reader = reader(text, &word, &table);
        assert!(!assembled_debug(&reader).contains("Image {"));
        assert!(reader.assets.borrow().assets.is_empty());
    }
}

#[test]
fn header_drawings_hidden_fields_and_non_special_characters_do_not_read_assets() {
    let text = "\u{13} HIDDEN\u{8}\rSTILL HIDDEN\u{15}\r";
    let cp = text.chars().position(|c| c == '\u{8}').unwrap() as u32;
    let (word, table) = fixture(text, &[(cp, 7)], &[(7, 0x200, 2)], false);
    let hidden = reader(text, &word, &table);
    assert!(!assembled_debug(&hidden).contains("Image {"));
    assert!(hidden.assets.borrow().assets.is_empty());
    let text = "A\u{8}\r";
    let (word, mut table) = fixture(text, &[(1, 7)], &[(7, 0x200, 2)], false);
    let mut missing_special = reader(text, &word, &table);
    missing_special.chpx = Runs::new(Vec::new());
    assert!(!assembled_debug(&missing_special).contains("Image {"));
    assert!(missing_special.assets.borrow().assets.is_empty());
    drop(missing_special);
    let group_size = get_u32(&table, 4).unwrap() as usize;
    table[8 + group_size] = 1;
    let header = reader(text, &word, &table);
    assert!(!assembled_debug(&header).contains("Image {"));
    assert!(header.assets.borrow().assets.is_empty());
}

#[test]
fn invalid_delay_length_type_and_uid_do_not_fall_back_to_an_unrelated_blip() {
    let text = "A\u{8}\r";
    for corruption in 0..4 {
        let (mut word, mut table) = fixture(text, &[(1, 7)], &[(7, 0x200, 2)], false);
        // DG header + BStore header + first FBSE (44) + second header.
        let fbse = 8 + 8 + 44 + 8;
        let offset = get_u32(&table, fbse + 28).unwrap() as usize;
        match corruption {
            0 => table[fbse + 28..fbse + 32].copy_from_slice(&u32::MAX.to_le_bytes()),
            1 => table[fbse + 20..fbse + 24].copy_from_slice(&u32::MAX.to_le_bytes()),
            2 => word[offset + 2..offset + 4].copy_from_slice(&0xF01Eu16.to_le_bytes()),
            _ => word[offset + 8] ^= 1,
        }
        let reader = reader(text, &word, &table);
        assert!(!assembled_debug(&reader).contains("Image {"));
        assert!(reader.assets.borrow().assets.is_empty());
    }
}

#[test]
fn parse_cfb_resolves_floating_picture_without_a_data_stream() {
    let text = "Before\r\u{8}\rAfter\r";
    let (mut word, mut table) = fixture(text, &[(7, 7)], &[(7, 0x200, 2)], false);
    word[..2].copy_from_slice(&0xA5ECu16.to_le_bytes());
    word[0x4C..0x50].copy_from_slice(&(text.len() as u32).to_le_bytes());
    let fc = word.len() as u32;
    word[0x18..0x1C].copy_from_slice(&fc.to_le_bytes());
    word[0x1C..0x20].copy_from_slice(&(fc + text.len() as u32).to_le_bytes());
    word.extend(text.bytes());
    // One CHPX FKP over the text range, setting CFSpec for the anchor.
    let page = word.len().div_ceil(512);
    word.resize((page + 1) * 512, 0);
    let fkp = &mut word[page * 512..(page + 1) * 512];
    fkp[..4].copy_from_slice(&fc.to_le_bytes());
    fkp[4..8].copy_from_slice(&(fc + text.len() as u32).to_le_bytes());
    fkp[8] = 10;
    fkp[20..24].copy_from_slice(&[3, 0x55, 0x08, 1]);
    fkp[511] = 1;
    let mut bte = fc.to_le_bytes().to_vec();
    bte.extend((fc + text.len() as u32).to_le_bytes());
    bte.extend((page as u32).to_le_bytes());
    put(&mut word, &mut table, 0xFA, &bte);
    let mut ole = cfb::CompoundFile::create(Cursor::new(Vec::new())).unwrap();
    ole.create_stream("/WordDocument").unwrap().write_all(&word).unwrap();
    ole.create_stream("/0Table").unwrap().write_all(&table).unwrap();
    let document = parse(&ole.into_inner().into_inner()).unwrap();
    assert_eq!(document.assets.len(), 1);
    assert_eq!(document.assets[0].bytes, PAYLOAD);
    let output = format!("{:?}", document.blocks);
    assert_eq!(output.matches("Image {").count(), 1);
}

#[test]
fn complete_store_slots_survive_a_stale_count_and_an_empty_prior_slot() {
    let text = "A\u{8}\r";
    let (word, mut table) = fixture(text, &[(1, 7)], &[(7, 0x200, 2)], false);
    table[8..10].copy_from_slice(&0x1Fu16.to_le_bytes());
    table[8 + 8 + 8 + 24..8 + 8 + 8 + 28].copy_from_slice(&0u32.to_le_bytes());
    let reader = reader(text, &word, &table);
    assert_eq!(assembled_debug(&reader).matches("Image {").count(), 1);
    assert_eq!(reader.assets.borrow().assets[0].bytes, PAYLOAD);
    assert!(reader.pictures.warnings.iter().any(|warning| warning.contains("inconsistent count")));
}

#[test]
fn ordinary_textbox_shape_without_picture_reference_needs_no_store() {
    let text = "A\u{8}\r";
    let (word, mut table) = fixture(text, &[(1, 7)], &[(7, 0x200, 0)], false);
    // Replace the Dgg's BStore with a same-sized unrelated record. The
    // picture reader should never require it for this ordinary shape.
    table[10..12].copy_from_slice(&0xF11Eu16.to_le_bytes());
    let reader = reader(text, &word, &table);
    assert!(!assembled_debug(&reader).contains("Image {"));
    assert!(reader.pictures.warnings.is_empty());
}
