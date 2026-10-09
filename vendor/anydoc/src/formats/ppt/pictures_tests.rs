use super::*;

fn rec(kind: u16, ver: u16, instance: u16, body: &[u8]) -> Vec<u8> {
    let mut r = (ver | instance << 4).to_le_bytes().to_vec();
    r.extend(kind.to_le_bytes());
    r.extend((body.len() as u32).to_le_bytes());
    r.extend(body);
    r
}
fn png(payload: &[u8]) -> Vec<u8> {
    let mut body = vec![0; 17];
    body.extend(payload);
    rec(0xF01E, 0, 0x6E0, &body)
}
fn fbse(blip: &[u8], delay: u32, refs: u32) -> Vec<u8> {
    let mut body = vec![0; 36];
    body[0] = 6;
    body[1] = 6;
    body[20..24].copy_from_slice(&(blip.len() as u32).to_le_bytes());
    body[24..28].copy_from_slice(&refs.to_le_bytes());
    body[28..32].copy_from_slice(&delay.to_le_bytes());
    body.extend(blip);
    rec(0xF007, 2, 6, &body)
}
fn delayed(delay: u32, size: usize) -> Vec<u8> {
    let mut out = fbse(&[], delay, 1);
    out[28..32].copy_from_slice(&(size as u32).to_le_bytes());
    out
}
fn bank(slots: &[Vec<u8>]) -> Vec<u8> {
    rec(0xF000, 15, 0, &rec(0xF001, 15, slots.len() as u16, &slots.concat()))
}
fn sp(opid: u16, pib: u32, flags: u32, hidden: bool) -> Vec<u8> {
    let mut fsp = 17u32.to_le_bytes().to_vec();
    fsp.extend(flags.to_le_bytes());
    let mut props = opid.to_le_bytes().to_vec();
    props.extend(pib.to_le_bytes());
    if hidden {
        props.extend(0x3BFu16.to_le_bytes());
        props.extend(0x20002u32.to_le_bytes());
    }
    let mut body = rec(0xF00A, 2, 75, &fsp);
    body.extend(rec(0xF00B, 3, if hidden { 2 } else { 1 }, &props));
    body
}

#[test]
fn unsupported_and_empty_slots_do_not_renumber_referenced_pictures() {
    let group =
        bank(&[fbse(&rec(0xF01F, 0, 0, b"unsupported"), 0, 1), fbse(&png(b"diagram"), 0, 1)]);
    let mut b = Bank::read(&group, &[]).unwrap();
    assert!(b.image(1).unwrap().is_none());
    assert!(b.image(2).unwrap().is_some());
    assert_eq!(b.assets.assets[0].bytes, b"diagram");
    assert!(b.image(0).unwrap().is_none());
    assert!(b.image(3).unwrap().is_none());
    let group = bank(&[fbse(&png(b"empty"), 0, 0), fbse(&png(b"live"), 0, 1)]);
    let mut b = Bank::read(&group, &[]).unwrap();
    assert!(b.image(1).unwrap().is_none());
    assert!(b.image(2).unwrap().is_some());
    assert_eq!(b.assets.assets[0].bytes, b"live");
}
#[test]
fn only_requested_slots_are_kept_and_duplicate_references_share_bytes() {
    let group = bank(&[fbse(&png(b"orphan"), 0, 1), fbse(&png(b"live"), 0, 1)]);
    let mut b = Bank::read(&group, &[]).unwrap();
    assert!(b.assets.assets.is_empty());
    let a = b.image(2).unwrap().unwrap();
    let c = b.image(2).unwrap().unwrap();
    assert!(
        matches!((a,c),(Inline::Image{source:ImageSource::Asset(a),..},Inline::Image{source:ImageSource::Asset(b),..})if a==b)
    );
    assert_eq!(b.assets.assets.len(), 1);
    assert_eq!(b.assets.assets[0].bytes, b"live");
}
#[test]
fn delayed_slots_use_record_boundaries_and_embedded_data_has_priority() {
    let first = png(b"first");
    let second = png(b"second");
    let delay = [first.clone(), second].concat();
    let group = bank(&[
        delayed(first.len() as u32, delay.len() - first.len()),
        delayed(1, delay.len() - first.len()),
        fbse(&png(b"embedded"), 0, 1),
    ]);
    let mut b = Bank::read(&group, &delay).unwrap();
    assert!(b.image(1).unwrap().is_some());
    assert_eq!(b.assets.assets[0].bytes, b"second");
    assert!(b.image(2).unwrap().is_none());
    assert!(b.image(3).unwrap().is_some());
    assert_eq!(b.assets.assets[1].bytes, b"embedded");
}
#[test]
fn truncated_and_conflicting_banks_are_rejected_and_count_mismatch_cannot_invent_slots() {
    let mut bad = bank(&[fbse(&png(b"image"), 0, 1)]);
    bad.pop();
    assert!(Bank::read(&bad, &[]).unwrap().image(1).unwrap().is_none());
    let one = bank(&[fbse(&png(b"one"), 0, 1)]);
    let two = bank(&[fbse(&png(b"two"), 0, 1)]);
    assert!(Bank::read(&[one, two].concat(), &[]).unwrap().image(1).unwrap().is_none());
    let group = rec(0xF000, 15, 0, &rec(0xF001, 15, 2, &fbse(&png(b"one"), 0, 1)));
    let mut recovered = Bank::read(&group, &[]).unwrap();
    assert!(recovered.image(1).unwrap().is_some());
    assert_eq!(recovered.assets.assets[0].bytes, b"one");
    assert!(recovered.warnings.iter().any(|w| w.contains("declared slot count")));
    assert!(recovered.image(2).unwrap().is_none(), "declared count cannot invent missing slots");
}
#[test]
fn picture_shape_requires_a_live_noncomplex_blip_id() {
    assert_eq!(shape(&sp(0x4104, 2, 2560, false)).pib, Some(2));
    for (opid, pib, flags, hidden) in [
        (0x104, 2, 2560, false),
        (0xC104, 2, 2560, false),
        (0x4104, 0, 2560, false),
        (0x4104, 2, 2568, false),
        (0x4104, 2, 2560, true),
    ] {
        assert!(shape(&sp(opid, pib, flags, hidden)).pib.is_none());
    }
    let mut dup = sp(0x4104, 1, 2560, false);
    dup.extend(rec(0xF00B, 3, 1, &[0x04, 0x41, 2, 0, 0, 0]));
    assert!(shape(&dup).pib.is_none());
}
#[test]
fn group_state_is_inherited_and_record_budget_is_a_hard_error() {
    let visible = rec(0xF004, 15, 0, &sp(0x4104, 1, 5, false));
    let hidden = rec(0xF004, 15, 0, &sp(0x4104, 1, 5, true));
    assert!(!group_hidden(&visible));
    assert!(group_hidden(&hidden));
    assert!(group_hidden(&[]));
    let mut n = limits::MAX_RECORDS;
    assert!(matches!(
        charge(&mut n),
        Err(ConvertError::ResourceLimit { limit: "max_records", .. })
    ));
}

#[test]
fn an_unknown_bank_child_cannot_shift_a_real_picture_ordinal() {
    let group = bank(&[rec(0xF00A, 2, 0, &[0; 8]), fbse(&png(b"live"), 0, 1)]);
    let mut b = Bank::read(&group, &[]).unwrap();
    assert!(b.image(1).unwrap().is_none());
    assert!(b.image(2).unwrap().is_none());
    assert!(b.assets.assets.is_empty());
    assert!(b.warnings.iter().any(|w| w.contains("invalid record")));
}

#[test]
fn picture_id_comes_from_valid_primary_properties_and_exact_fsp_metadata() {
    let correct = sp(0x4104, 1, 2560, false);
    assert_eq!(shape(&correct).pib, Some(1));
    let mut tertiary = correct.clone();
    tertiary[18..20].copy_from_slice(&0xF122u16.to_le_bytes());
    assert!(shape(&tertiary).pib.is_none(), "tertiary pib is not a primary reference");
    let mut bad_fopt = correct.clone();
    bad_fopt[16..18].copy_from_slice(&0x12u16.to_le_bytes());
    assert!(shape(&bad_fopt).pib.is_none());
    let mut bad_fsp = correct.clone();
    bad_fsp[0] = (bad_fsp[0] & 0xF0) | 1;
    assert!(shape(&bad_fsp).pib.is_none());
    for len in [4, 9] {
        let fsp = vec![0; len];
        let b = [rec(0xF00A, 2, 75, &fsp), correct[16..].to_vec()].concat();
        assert!(shape(&b).pib.is_none());
    }
    assert!(shape(&correct[16..]).pib.is_none(), "properties cannot invent a live FSP");
}

#[test]
fn script_anchor_hidden_requires_all_specification_use_and_value_bits() {
    // MS-ODRAW 2.3.4.44: value bits 7/8, use bits 23/24.
    let all = 0x0180_0180u32;
    for kind in [0xF00B, 0xF122] {
        let prop = |opid: u16, value: u32| {
            let mut b = sp(0x4104, 1, 2560, false);
            let props = [opid.to_le_bytes().as_slice(), value.to_le_bytes().as_slice()].concat();
            b.extend(rec(kind, 3, 1, &props));
            b
        };
        assert!(shape(&prop(0x3BF, all)).hidden);
        for bit in [0x80, 0x100, 0x0080_0000, 0x0100_0000] {
            assert_eq!(shape(&prop(0x3BF, all & !bit)).pib, Some(1));
        }
        assert_eq!(shape(&prop(0x3BF, 0x0100_0100)).pib, Some(1));
        // fBid/fComplex are both zero for this Boolean property.
        assert_eq!(shape(&prop(0x43BF, all)).pib, Some(1));
        assert_eq!(shape(&prop(0x83BF, all)).pib, Some(1));
    }
}

#[test]
fn delayed_fbse_size_measures_the_complete_inner_blip_not_the_wrapper() {
    let blip = png(b"wrapped PNG");
    let wrapped = fbse(&blip, 0, 1);
    let group = bank(&[delayed(0, blip.len())]);
    let mut b = Bank::read(&group, &wrapped).unwrap();
    assert!(b.image(1).unwrap().is_some());
    assert_eq!(b.assets.assets[0].bytes, b"wrapped PNG");
    let wrong_size = bank(&[delayed(0, wrapped.len())]);
    assert!(Bank::read(&wrong_size, &wrapped).unwrap().image(1).unwrap().is_none());
    for corrupt in 0..4 {
        let mut bad = wrapped.clone();
        match corrupt {
            0 => bad[0] = (bad[0] & 0xF0) | 1,
            1 => bad[41] = 1, // cbName must be even.
            2 => bad[28..32].copy_from_slice(&1u32.to_le_bytes()), // inner size
            _ => {
                bad.pop();
            }
        }
        assert!(Bank::read(&group, &bad).unwrap().image(1).unwrap().is_none());
    }
}

#[test]
fn malformed_embedded_picture_never_falls_back_to_a_valid_delay_record() {
    let correct = png(b"delay");
    let mut bad = fbse(&png(b"embedded"), 0, 1);
    bad[28..32].copy_from_slice(&(correct.len() as u32).to_le_bytes());
    let group = bank(&[bad]);
    assert!(Bank::read(&group, &correct).unwrap().image(1).unwrap().is_none());
    let mut named = fbse(&png(b"embedded"), 0, 1);
    named[41] = 2;
    // Missing terminating null at the newly claimed name boundary.
    let group = bank(&[named]);
    assert!(Bank::read(&group, &correct).unwrap().image(1).unwrap().is_none());
}

#[test]
fn direct_blip_file_blocks_keep_ordinals_without_loading_orphan_slots() {
    let group = bank(&[png(b"orphan"), rec(0xF01F, 0, 0, b"unsupported"), png(b"chosen")]);
    let mut b = Bank::read(&group, &[]).unwrap();
    assert!(b.assets.assets.is_empty());
    assert!(b.image(3).unwrap().is_some());
    assert_eq!(b.assets.assets[0].bytes, b"chosen");
    assert!(b.image(2).unwrap().is_none());
    assert_eq!(b.assets.assets.len(), 1);
    let mut bad_version = png(b"bad version");
    bad_version[0] |= 1;
    let group = bank(&[bad_version]);
    assert!(Bank::read(&group, &[]).unwrap().image(1).unwrap().is_none());
}

#[test]
fn current_document_drawing_group_must_be_unique_without_dropping_body_text() {
    use super::super::Extractor;
    let mut text = rec(0x0F9F, 0, 0, &1u32.to_le_bytes());
    let units: Vec<u8> = "retained body".encode_utf16().flat_map(u16::to_le_bytes).collect();
    text.extend(rec(0x0FA0, 0, 0, &units));
    let group_a = bank(&[png(b"first")]);
    let group_b = bank(&[png(b"conflicting")]);
    for duplicate in [false, true] {
        let mut doc = rec(0x0FF0, 15, 0, &text);
        doc.extend(rec(0x040B, 15, 0, &group_a));
        if duplicate {
            doc.extend(rec(0x040B, 15, 0, &group_b));
        }
        let mut stream = rec(0x03E8, 15, 0, &doc);
        let dir_at = stream.len() as u32;
        let dir = [0x0010_0001u32.to_le_bytes(), 0u32.to_le_bytes()].concat();
        stream.extend(rec(0x1772, 0, 0, &dir));
        let edit_at = stream.len() as u32;
        let mut edit = vec![0; 28];
        edit[12..16].copy_from_slice(&dir_at.to_le_bytes());
        edit[16..20].copy_from_slice(&1u32.to_le_bytes());
        stream.extend(rec(0x0FF5, 0, 0, &edit));
        let mut user = vec![0; 20];
        user[16..20].copy_from_slice(&edit_at.to_le_bytes());
        let mut e = Extractor::default();
        assert!(e.parse_slides(&stream, &user, &[]).unwrap());
        if duplicate {
            assert!(e.pictures.image(1).unwrap().is_none());
            assert!(e.pictures.warnings.iter().any(|w| w.contains("Conflicting drawing groups")));
        } else {
            assert!(e.pictures.image(1).unwrap().is_some());
            assert_eq!(e.pictures.assets.assets[0].bytes, b"first");
        }
        let (blocks, _) = e.into_blocks();
        let kept = blocks.iter().any(|b| matches!(b,
            crate::model::Block::Paragraph(p) if crate::model::inlines_to_plain_text(p) == "retained body"));
        assert!(kept, "ambiguity only suppresses newly resolved figures");
    }
}

/// markitai: an EMF blip as PowerPoint stores it: uid, the 34-byte
/// metafile header, then the zlib-compressed metafile.
fn emf(metafile: &[u8]) -> Vec<u8> {
    use std::io::Write;
    let mut encoder = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
    encoder.write_all(metafile).unwrap();
    let compressed = encoder.finish().unwrap();
    let mut body = vec![0; 16 + 34];
    body[16..20].copy_from_slice(&(metafile.len() as u32).to_le_bytes());
    body[44..48].copy_from_slice(&(compressed.len() as u32).to_le_bytes());
    body[49] = 0xFE;
    body.extend(compressed);
    rec(0xF01A, 0, 0x3D4, &body)
}

#[test]
fn complete_emf_pictures_are_kept_and_partial_ones_omitted() {
    let mut embedded = fbse(&emf(b"EMBEDDED EMF"), 0, 1);
    // An FBSE names its blip type (msoblipEMF = 2) in both type bytes and
    // its instance.
    embedded[0..2].copy_from_slice(&(2u16 | 2 << 4).to_le_bytes());
    embedded[8] = 2;
    embedded[9] = 2;
    // A metafile that inflates to less than its declared size.
    let mut partial = emf(b"PARTIAL EMF");
    partial[24..28].copy_from_slice(&64u32.to_le_bytes());
    let group = bank(&[embedded, emf(b"DIRECT EMF"), partial]);
    let mut b = Bank::read(&group, &[]).unwrap();
    assert!(b.image(1).unwrap().is_some());
    assert!(b.image(2).unwrap().is_some());
    assert!(b.image(3).unwrap().is_none());
    let kept: Vec<_> = b.assets.assets.iter().map(|a| a.bytes.as_slice()).collect();
    assert_eq!(kept, [b"EMBEDDED EMF".as_slice(), b"DIRECT EMF"]);
    assert!(b.assets.assets.iter().all(|a| a.media_type == "image/emf"));
    assert_eq!(b.warnings, ["A referenced OfficeArt picture format is unsupported and was omitted."]);
}
