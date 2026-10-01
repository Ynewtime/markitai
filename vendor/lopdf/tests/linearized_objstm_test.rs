use lopdf::Document;

/// Build a minimal PDF where two ObjStm streams contain conflicting versions
/// of the same object (the Pages tree root). The xref Compressed entry points
/// to the higher-numbered ObjStm as authoritative, but the lower-numbered one
/// is processed first during loading.
///
/// Without the fix in load_objects_raw, the stale copy from the lower-numbered
/// ObjStm would win via or_insert, causing the document to report fewer pages.
fn build_conflicting_objstm_pdf() -> Vec<u8> {
    let mut buf = Vec::new();

    // Header
    buf.extend_from_slice(b"%PDF-1.5\n");

    // Object 1: Catalog
    let off_1 = buf.len();
    buf.extend_from_slice(b"1 0 obj\n<</Type/Catalog/Pages 3 0 R>>\nendobj\n");

    // Object 2: ObjStm with STALE copy of object 3 (Count=1, only 1 kid)
    let off_2 = buf.len();
    let stale_data = b"3 0 <</Type/Pages/Count 1/Kids[5 0 R]>>";
    buf.extend_from_slice(
        format!(
            "2 0 obj\n<</Type/ObjStm/N 1/First 4/Length {}>>\nstream\n",
            stale_data.len()
        )
        .as_bytes(),
    );
    buf.extend_from_slice(stale_data);
    buf.extend_from_slice(b"\nendstream\nendobj\n");

    // Object 4: ObjStm with CORRECT copy of object 3 (Count=2, 2 kids)
    let off_4 = buf.len();
    let correct_data = b"3 0 <</Type/Pages/Count 2/Kids[5 0 R 6 0 R]>>";
    buf.extend_from_slice(
        format!(
            "4 0 obj\n<</Type/ObjStm/N 1/First 4/Length {}>>\nstream\n",
            correct_data.len()
        )
        .as_bytes(),
    );
    buf.extend_from_slice(correct_data);
    buf.extend_from_slice(b"\nendstream\nendobj\n");

    // Object 5: Page
    let off_5 = buf.len();
    buf.extend_from_slice(b"5 0 obj\n<</Type/Page/Parent 3 0 R/MediaBox[0 0 612 792]>>\nendobj\n");

    // Object 6: Page
    let off_6 = buf.len();
    buf.extend_from_slice(b"6 0 obj\n<</Type/Page/Parent 3 0 R/MediaBox[0 0 612 792]>>\nendobj\n");

    // Object 7: Cross-reference stream
    // W=[1 3 1]: type (1 byte), field2 (3 bytes), field3 (1 byte)
    // Entries for objects 0-7:
    //   0: Free
    //   1: Normal (Catalog)
    //   2: Normal (stale ObjStm)
    //   3: Compressed in container=4, index=0  <-- authoritative
    //   4: Normal (correct ObjStm)
    //   5: Normal (Page)
    //   6: Normal (Page)
    //   7: Normal (xref stream itself)
    let off_7 = buf.len();

    let mut xref_data = Vec::new();
    let offsets = [
        0u32,
        off_1 as u32,
        off_2 as u32,
        0,
        off_4 as u32,
        off_5 as u32,
        off_6 as u32,
        off_7 as u32,
    ];

    // Object 0: Free
    xref_data.extend_from_slice(&[0, 0, 0, 0, 0]);

    // Objects 1, 2: Normal
    for &off in &offsets[1..=2] {
        xref_data.push(1);
        xref_data.push((off >> 16) as u8);
        xref_data.push((off >> 8) as u8);
        xref_data.push(off as u8);
        xref_data.push(0);
    }

    // Object 3: Compressed { container: 4, index: 0 }
    xref_data.push(2);
    xref_data.extend_from_slice(&[0, 0, 4]); // container = 4
    xref_data.push(0); // index = 0

    // Objects 4, 5, 6, 7: Normal
    for &off in &offsets[4..=7] {
        xref_data.push(1);
        xref_data.push((off >> 16) as u8);
        xref_data.push((off >> 8) as u8);
        xref_data.push(off as u8);
        xref_data.push(0);
    }

    buf.extend_from_slice(
        format!(
            "7 0 obj\n<</Type/XRef/Size 8/W[1 3 1]/Root 1 0 R/Length {}>>\nstream\n",
            xref_data.len()
        )
        .as_bytes(),
    );
    buf.extend_from_slice(&xref_data);
    buf.extend_from_slice(b"\nendstream\nendobj\n");

    buf.extend_from_slice(format!("startxref\n{}\n%%EOF", off_7).as_bytes());
    buf
}

#[test]
fn test_conflicting_objstm_uses_xref_container() {
    let pdf = build_conflicting_objstm_pdf();
    let doc = Document::load_mem(&pdf).unwrap();

    // The xref says object 3 (Pages root) belongs to ObjStm 4 (Count=2).
    // ObjStm 2 has a stale copy with Count=1. The correct version must win.
    assert_eq!(
        doc.get_pages().len(),
        2,
        "Should load 2 pages from ObjStm 4, not 1 from stale ObjStm 2"
    );
}

/// markitai: an object stream holding `objects` (number and text) as object
/// `number`.
fn object_stream(number: u32, objects: &[(u32, &str)]) -> Vec<u8> {
    let (mut header, mut body) = (String::new(), String::new());
    for (id, object) in objects {
        header.push_str(&format!("{id} {} ", body.len()));
        body.push_str(object);
        body.push(' ');
    }
    let data = format!("{header}{body}");
    format!(
        "{number} 0 obj\n<</Type/ObjStm/N {}/First {}/Length {}>>\nstream\n{data}\nendstream\nendobj\n",
        objects.len(),
        header.len(),
        data.len()
    )
    .into_bytes()
}

/// markitai: as `build_conflicting_objstm_pdf`, and each stream also holds an
/// object that only it holds: object 8 in ObjStm 2, where the xref puts it,
/// and object 9 in ObjStm 4, though the xref puts it in ObjStm 2. Whichever
/// stream is read first, object 8 must load and object 9 must not.
fn build_two_container_pdf() -> Vec<u8> {
    let mut buf = b"%PDF-1.5\n".to_vec();
    let mut offsets = [0u32; 10];
    offsets[1] = buf.len() as u32;
    buf.extend_from_slice(b"1 0 obj\n<</Type/Catalog/Pages 3 0 R>>\nendobj\n");
    offsets[2] = buf.len() as u32;
    buf.extend(object_stream(
        2,
        &[(3, "<</Type/Pages/Count 1/Kids[5 0 R]>>"), (8, "<</Marker/Two>>")],
    ));
    offsets[4] = buf.len() as u32;
    buf.extend(object_stream(
        4,
        &[
            (3, "<</Type/Pages/Count 2/Kids[5 0 R 6 0 R]>>"),
            (9, "<</Marker/Stale>>"),
        ],
    ));
    for id in [5, 6] {
        offsets[id as usize] = buf.len() as u32;
        buf.extend_from_slice(
            format!("{id} 0 obj\n<</Type/Page/Parent 3 0 R/MediaBox[0 0 612 792]>>\nendobj\n").as_bytes(),
        );
    }
    offsets[7] = buf.len() as u32;
    let mut xref = vec![0, 0, 0, 0, 0];
    for (id, &off) in offsets.iter().enumerate().skip(1) {
        match id {
            3 => xref.extend_from_slice(&[2, 0, 0, 4, 0]),
            8 => xref.extend_from_slice(&[2, 0, 0, 2, 1]),
            9 => xref.extend_from_slice(&[2, 0, 0, 2, 2]),
            _ => xref.extend_from_slice(&[1, (off >> 16) as u8, (off >> 8) as u8, off as u8, 0]),
        }
    }
    buf.extend_from_slice(
        format!(
            "7 0 obj\n<</Type/XRef/Size 10/W[1 3 1]/Root 1 0 R/Length {}>>\nstream\n",
            xref.len()
        )
        .as_bytes(),
    );
    buf.extend_from_slice(&xref);
    buf.extend_from_slice(b"\nendstream\nendobj\n");
    buf.extend_from_slice(format!("startxref\n{}\n%%EOF", offsets[7]).as_bytes());
    buf
}

/// markitai: each compressed object loads only from the container the xref
/// names, with and without a filter function (the two branches of
/// `load_objects_raw`).
#[test]
fn test_compressed_objects_load_only_from_their_xref_container() {
    fn keep(id: (u32, u16), object: &mut lopdf::Object) -> Option<((u32, u16), lopdf::Object)> {
        Some((id, object.clone()))
    }
    let pdf = build_two_container_pdf();
    for doc in [
        Document::load_mem(&pdf).unwrap(),
        Document::load_mem_with_options(&pdf, lopdf::LoadOptions::with_filter(keep)).unwrap(),
    ] {
        assert_eq!(
            doc.get_pages().into_iter().collect::<Vec<_>>(),
            [(1, (5, 0)), (2, (6, 0))]
        );
        assert!(doc.objects.contains_key(&(8, 0)), "object 8 is in its own container");
        assert!(
            !doc.objects.contains_key(&(9, 0)),
            "object 9 is not in the container the xref names"
        );
    }
}
