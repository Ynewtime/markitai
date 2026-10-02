//! Slide markers on the presentations anydoc reads, ODP and legacy PPT: they
//! are numbered as the PPTX reader numbers its slides.

use super::*;
use std::io::{Cursor, Write};

fn paragraph(text: &str) -> Block {
    Block::Paragraph(vec![Inline::plain(text)])
}

fn renderer() -> Renderer<'static> {
    Renderer {
        asset_names: &[],
        merged_cells: false,
        in_cell: false,
        anchors: BTreeSet::new(),
        extension: "odp",
    }
}

#[test]
fn blocks_are_written_behind_numbered_markers_and_blank_slides_keep_theirs() {
    let blocks = [
        paragraph("Before any slide"),
        Block::heading(2, vec![Inline::plain("Title")]),
        paragraph("Body"),
        paragraph("Third"),
    ];
    // Slide 2 has no blocks; slide 4 has none and comes last.
    let starts = [1, 3, 3, 4];
    assert_eq!(
        renderer().slides(&blocks, &starts),
        "Before any slide\n\n\
         <!-- Slide number: 1 -->\n## Title\n\nBody\n\n\
         <!-- Slide number: 2 -->\n\n\
         <!-- Slide number: 3 -->\nThird\n\n\
         <!-- Slide number: 4 -->"
    );
    // A start past the end of the blocks is a blank slide, not a panic.
    assert_eq!(
        renderer().slides(&blocks[..1], &[0, 9]),
        "<!-- Slide number: 1 -->\nBefore any slide\n\n<!-- Slide number: 2 -->"
    );
}

const PRESENTATION: &str = r#"<office:document-content
    xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
    xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"
    xmlns:draw="urn:oasis:names:tc:opendocument:xmlns:drawing:1.0"
    xmlns:presentation="urn:oasis:names:tc:opendocument:xmlns:presentation:1.0">
  <office:body><office:presentation>{pages}</office:presentation></office:body>
</office:document-content>"#;

fn odp(pages: &str) -> Vec<u8> {
    let mut zip = zip::ZipWriter::new(Cursor::new(Vec::new()));
    let options = zip::write::SimpleFileOptions::default();
    zip.start_file("mimetype", options).unwrap();
    zip.write_all(b"application/vnd.oasis.opendocument.presentation")
        .unwrap();
    zip.start_file("content.xml", options).unwrap();
    zip.write_all(PRESENTATION.replace("{pages}", pages).as_bytes())
        .unwrap();
    zip.finish().unwrap().into_inner()
}

#[test]
fn odp_slides_are_numbered_like_pptx_slides_and_blank_ones_keep_their_marker() {
    let bytes = odp(r#"<draw:page>
             <draw:frame presentation:class="title"><draw:text-box>
               <text:p>First</text:p></draw:text-box></draw:frame>
             <draw:frame><draw:text-box><text:p>First body</text:p></draw:text-box></draw:frame>
           </draw:page>
           <draw:page/>
           <draw:page>
             <presentation:notes><draw:frame><draw:text-box>
               <text:p>Note three</text:p></draw:text-box></draw:frame></presentation:notes>
           </draw:page>
           <draw:page>
             <draw:frame><draw:text-box><text:p>Fourth body</text:p></draw:text-box></draw:frame>
           </draw:page>
           <draw:page/>"#);
    let document = extract(&bytes, "odp").unwrap();
    assert_eq!(
        document.markdown,
        "<!-- Slide number: 1 -->\n## First\n\nFirst body\n\n\
         <!-- Slide number: 2 -->\n\n\
         <!-- Slide number: 3 -->\n> Note three\n\n\
         <!-- Slide number: 4 -->\nFourth body\n\n\
         <!-- Slide number: 5 -->"
    );
    // A deck of blank slides is still a numbered deck.
    assert_eq!(
        extract(&odp("<draw:page/><draw:page/>"), "odp")
            .unwrap()
            .markdown,
        "<!-- Slide number: 1 -->\n\n<!-- Slide number: 2 -->"
    );
}

fn record(version: u16, instance: u16, kind: u16, body: &[u8]) -> Vec<u8> {
    let mut out = (version | instance << 4).to_le_bytes().to_vec();
    out.extend(kind.to_le_bytes());
    out.extend((body.len() as u32).to_le_bytes());
    out.extend(body);
    out
}

/// A text shape: a TextHeaderAtom of the given text type, then its text.
fn text(kind: u32, chars: &str) -> Vec<u8> {
    let units: Vec<u8> = chars.encode_utf16().flat_map(u16::to_le_bytes).collect();
    let mut out = record(0, 0, 0x0F9F, &kind.to_le_bytes());
    out.extend(record(0, 0, 0x0FA0, &units));
    out
}

/// A SlidePersistAtom: persistIdRef at 0 and slideId at 12.
fn slide_atom(persist_ref: u32, slide_id: u32) -> Vec<u8> {
    let mut body = [0u8; 20];
    body[..4].copy_from_slice(&persist_ref.to_le_bytes());
    body[12..16].copy_from_slice(&slide_id.to_le_bytes());
    record(0, 0, 0x03F3, &body)
}

/// A five-slide deck in the legacy PowerPoint layout: a titled slide, a blank
/// one, a blank one with speaker notes, one with a body, and a blank last one.
/// Slide text is outline text of the slide list; the notes page is a
/// container the persist directory points to (its NotesAtom names slide id
/// 258, the third slide).
fn ppt() -> Vec<u8> {
    let mut notes = record(0, 0, 0x03F1, &[2, 1, 0, 0, 0, 0, 0, 0]);
    notes.extend(text(2, "Note three"));
    let notes = record(0xF, 0, 0x03F0, &notes);

    let mut slides = Vec::new();
    slides.extend(slide_atom(10, 256));
    slides.extend(text(0, "First slide"));
    slides.extend(text(1, "First body"));
    slides.extend(slide_atom(11, 257));
    slides.extend(slide_atom(12, 258));
    slides.extend(slide_atom(13, 259));
    slides.extend(text(1, "Fourth body"));
    slides.extend(slide_atom(14, 260));
    let mut document = record(0xF, 0, 0x0FF0, &slides);
    document.extend(record(0xF, 2, 0x0FF0, &slide_atom(2, 0)));
    let document = record(0xF, 0, 0x03E8, &document);

    let mut stream = notes;
    let document_at = stream.len() as u32;
    stream.extend(document);
    // Persist ids 1 (the document) and 2 (the notes container).
    let mut directory = (1u32 | 2 << 20).to_le_bytes().to_vec();
    directory.extend(document_at.to_le_bytes());
    directory.extend(0u32.to_le_bytes());
    let directory_at = stream.len() as u32;
    stream.extend(record(0, 0, 0x1772, &directory));
    let mut edit = [0u8; 28];
    edit[12..16].copy_from_slice(&directory_at.to_le_bytes());
    edit[16..20].copy_from_slice(&1u32.to_le_bytes());
    let edit_at = stream.len() as u32;
    stream.extend(record(0, 0, 0x0FF5, &edit));

    let mut user = 20u32.to_le_bytes().to_vec();
    user.extend(0xE391_C05Fu32.to_le_bytes());
    user.extend(edit_at.to_le_bytes());

    let mut file = cfb::CompoundFile::create(Cursor::new(Vec::new())).unwrap();
    file.create_stream("PowerPoint Document")
        .unwrap()
        .write_all(&stream)
        .unwrap();
    file.create_stream("Current User")
        .unwrap()
        .write_all(&record(0, 0, 0x0FF6, &user))
        .unwrap();
    file.into_inner().into_inner()
}

#[test]
fn ppt_slides_are_numbered_like_pptx_slides_and_blank_ones_keep_their_marker() {
    let document = extract(&ppt(), "ppt").unwrap();
    // A legacy PowerPoint file's output ends with a newline (see `extract`).
    assert_eq!(
        document.markdown,
        "<!-- Slide number: 1 -->\n## First slide\n\nFirst body\n\n\
         <!-- Slide number: 2 -->\n\n\
         <!-- Slide number: 3 -->\n> Note three\n\n\
         <!-- Slide number: 4 -->\nFourth body\n\n\
         <!-- Slide number: 5 -->\n"
    );
}
