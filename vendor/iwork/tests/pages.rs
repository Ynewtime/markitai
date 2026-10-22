//! The Pages document spine: modes, sections, headers, footers and flows.
//!
//! Three claims, in the order they have to hold.
//!
//! **A section covers the text the app says it covers.** The range comes out of
//! the body storage's table 17 by arithmetic — the entry sits on the character
//! after a `U+0004`, and the break belongs to neither side — and behind
//! `IWORK_APP_CHECK=1` every range is compared against `body text of section i`
//! character for character, which is the only structural property Pages'
//! dictionary will report.
//!
//! **The shape is the shape everywhere.** Three headers and three footers per
//! section template, in every document in the corpus; page templates exactly
//! where the document is a page-layout one; column fractions that add up.
//!
//! **A header is a text storage, and writing one survives the app.** Pages has
//! no header property at all, so `app-check.sh` can only say the document
//! opened. `resave.sh` says more: the app is made to open the edited document
//! and write it out again, and what is on disk afterwards was written by Pages.
//! A header this crate invented badly does not come back from that.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use iwork::pages::{Mode, Zone};
use iwork::{iwa, pb, Document, Error, Package};

fn generated(name: &str) -> Option<PathBuf> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/generated")
        .join(name);
    path.exists().then_some(path)
}

macro_rules! fixture {
    ($name:expr) => {
        match generated($name) {
            Some(path) => path,
            None => {
                eprintln!("no {} — skipping (run scripts/make-fixtures.sh)", $name);
                return;
            }
        }
    };
}

/// Every Pages fixture, which is the only kind this file is about.
/// A password-protected package, which is not a fixture any test here can use:
/// its object streams are ciphertext and `Document::open` refuses it by design.
/// `tests/fixtures.rs` is where that refusal is asserted.
fn encrypted(path: &Path) -> bool {
    iwork::Package::read(path).is_ok_and(|package| package.contains(".iwpv2"))
}

/// The corpus is generated, never committed, so a sweep over it skips where
/// it is absent — the same rule `tests/fixtures.rs` states as "so a fresh
/// clone is green". The counting assertions below keep their teeth wherever
/// the corpus *is* present, which is the only place they could say anything.
macro_rules! corpus {
    () => {
        if pages_fixtures().is_empty() {
            eprintln!("no .pages corpus — skipping (run scripts/make-fixtures.sh)");
            return;
        }
    };
}

fn pages_fixtures() -> Vec<PathBuf> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/generated");
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return Vec::new();
    };
    let mut found: Vec<PathBuf> = entries
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("pages"))
        .filter(|path| !encrypted(path))
        .collect();
    found.sort();
    found
}

// -- reading -----------------------------------------------------------------

/// Only Pages has this. A Numbers or Keynote document has no
/// `TP.DocumentArchive` and must come back with nothing rather than with
/// something invented.
#[test]
fn only_a_pages_document_has_a_structure() {
    for name in ["numbers-values.numbers", "keynote-deck.key"] {
        let Some(path) = generated(name) else {
            continue;
        };
        let doc = Document::open(&path).unwrap();
        assert!(doc.structure().is_none(), "{name} reported a TP structure");
        assert!(doc.sections().is_empty());
        assert!(doc.header_footers().is_empty());
    }
    let path = fixture!("pages-report.pages");
    assert!(Document::open(&path).unwrap().structure().is_some());
}

/// The arithmetic, stated once and checked against every fixture: a section
/// begins after a `U+0004` and ends one short of the next section's start, so
/// the ranges tile the body with exactly one character missing between each
/// pair — the break.
#[test]
fn sections_tile_the_body_with_the_break_between_them() {
    corpus!();
    let mut checked = 0usize;
    for path in pages_fixtures() {
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        let doc = Document::open(&path).unwrap();
        let structure = doc.structure().unwrap();
        let Some(body) = structure.body_storage else {
            continue;
        };
        let text: Vec<u16> = doc.storage_text(body).unwrap().encode_utf16().collect();
        assert!(
            !structure.sections.is_empty(),
            "{name}: a Pages document has at least one section"
        );
        assert_eq!(structure.sections[0].start, 0, "{name}");
        for (index, section) in structure.sections.iter().enumerate() {
            assert!(section.end >= section.start, "{name}: section {index}");
            assert!(
                section.end <= text.len() as u64,
                "{name}: section {index} runs past the body"
            );
            if index > 0 {
                assert_eq!(
                    text.get(section.start as usize - 1).copied(),
                    Some(0x0004),
                    "{name}: section {index} does not begin after a break"
                );
            }
            if let Some(next) = structure.sections.get(index + 1) {
                assert_eq!(
                    section.end + 1,
                    next.start,
                    "{name}: section {index} does not reach the next one's break"
                );
            } else {
                assert_eq!(section.end, text.len() as u64, "{name}: the last section");
            }
            checked += 1;
        }
    }
    assert!(checked >= 10, "only {checked} sections were checked");
}

/// Three headers and three footers, per template page, per section, in every
/// document — never any other count. It is the rule `iwork check` enforces and
/// the reason a zone can be named left, centre or right at all.
#[test]
fn every_section_template_carries_three_headers_and_three_footers() {
    corpus!();
    let mut templates = 0usize;
    for path in pages_fixtures() {
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        let doc = Document::open(&path).unwrap();
        let structure = doc.structure().unwrap();
        let mut counted: BTreeMap<(u64, bool), Vec<&str>> = BTreeMap::new();
        for entry in &structure.header_footers {
            counted
                .entry((entry.section_template, entry.footer))
                .or_default()
                .push(entry.zone.as_str());
        }
        for ((template, footer), zones) in &counted {
            assert_eq!(
                zones,
                &vec!["left", "centre", "right"],
                "{name}: section template {template} {}",
                if *footer { "footers" } else { "headers" }
            );
        }
        // One `first`, one `even` and one `odd` page per section, and a header
        // set and a footer set on each: six keys per section.
        assert_eq!(
            counted.len(),
            structure.sections.len() * 6,
            "{name}: section templates per section"
        );
        templates += counted.len();
        assert!(doc.problems().is_empty(), "{name}: {:?}", doc.problems());
    }
    assert!(templates >= 60, "only {templates} zone sets were seen");
}

/// The two document modes, and the second signal that says the same thing.
///
/// `TP.SettingsArchive.body` is the flag; a `TP.PageTemplateArchive` is the
/// consequence. Over all 640 bundled Pages templates the two sets are equal,
/// and the corpus has one document on each side of it.
#[test]
fn a_page_layout_document_is_the_one_with_page_templates() {
    corpus!();
    let mut modes = BTreeMap::new();
    for path in pages_fixtures() {
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        let structure = Document::open(&path).unwrap().structure().unwrap();
        *modes.entry(structure.mode).or_insert(0usize) += 1;
        assert_eq!(
            structure.mode == Mode::PageLayout,
            !structure.page_templates.is_empty(),
            "{name}: mode {} with {} page template(s)",
            structure.mode.as_str(),
            structure.page_templates.len()
        );
    }
    assert!(
        modes.get(&Mode::PageLayout).copied().unwrap_or(0) >= 1
            && modes.get(&Mode::WordProcessing).copied().unwrap_or(0) >= 1,
        "the corpus needs one of each: {modes:?}"
    );
}

/// The one fixture with page numbering that is not the default, read field by
/// field. `65_Sales_Bold_Report_PM` restarts at 2 in its second section and
/// hides the header and footer on the first page of its first.
#[test]
fn page_numbering_and_the_section_switches_are_read() {
    let path = fixture!("pages-numbering.pages");
    let structure = Document::open(&path).unwrap().structure().unwrap();
    assert_eq!(structure.sections.len(), 2);

    let first = &structure.sections[0];
    assert_eq!(first.page_number_kind, 0);
    assert_eq!(first.numbering(), "continue from the previous section");
    assert!(first.hides_header_footer_on_first_page);
    assert!(first.has_background, "the cover paints its own background");
    assert!(first.hyperlink_uuid, "a section a link can point at");

    let second = &structure.sections[1];
    assert_eq!((second.page_number_kind, second.page_number_start), (1, 2));
    assert_eq!(second.numbering(), "start at 2");
    assert!(!second.hides_header_footer_on_first_page);
}

/// Facing pages, which only the eighteen novel-shaped templates have.
#[test]
fn facing_pages_and_six_sections() {
    let path = fixture!("pages-book.pages");
    let structure = Document::open(&path).unwrap().structure().unwrap();
    assert!(structure.setup.facing_pages);
    assert_eq!(structure.sections.len(), 6);
    assert!(structure.setup.portrait());
    // A novel has no page templates, being a word-processing document.
    assert!(structure.page_templates.is_empty());
}

/// The linked-text-box thread: one storage, several boxes, in flow order. A
/// thread is numbered rather than named — `user_interface_identifier` is its
/// only identity.
#[test]
fn a_thread_joins_two_boxes_to_one_storage() {
    let path = fixture!("pages-layout.pages");
    let doc = Document::open(&path).unwrap();
    let structure = doc.structure().unwrap();
    assert_eq!(structure.threads.len(), 1);
    let thread = &structure.threads[0];
    assert_eq!(thread.boxes.len(), 2, "the newsletter flows through two");
    let storage = thread.storage.expect("a thread lays out one storage");
    assert!(
        doc.storage_text(storage).unwrap().len() > 100,
        "and that storage holds the article"
    );
    // Every box exists and every one is a different object.
    assert_ne!(thread.boxes[0], thread.boxes[1]);
    for object in &thread.boxes {
        assert!(doc.object(*object).is_some());
    }
}

/// A table of contents in its two halves: the document's own style-inclusion
/// map, and the placed list's copy — which do not agree, and reading only one
/// of them is reading the wrong one.
#[test]
fn a_table_of_contents_has_two_settings_archives_that_disagree() {
    let path = fixture!("pages-toc.pages");
    let structure = Document::open(&path).unwrap().structure().unwrap();
    assert_eq!(structure.contents.len(), 2);

    let document_wide = &structure.contents[0];
    assert!(document_wide.placed_in.is_none());
    assert_eq!(document_wide.scope, 0);
    assert_eq!(document_wide.rules.len(), 2);
    assert!(document_wide.rules.iter().all(|r| r.show));

    let placed = &structure.contents[1];
    assert!(placed.placed_in.is_some(), "owned by a TSWP.TOCInfoArchive");
    assert_eq!(placed.scope, 1);
    assert_eq!(placed.rules.len(), 6);
    assert_eq!(placed.rules.iter().filter(|r| r.show).count(), 2);

    // And the entries the last layout produced, which carry the heading text
    // and the page it landed on.
    let headings: Vec<&str> = placed.entries.iter().map(|(h, _)| h.as_str()).collect();
    assert_eq!(headings, vec!["Chapter Title", "Heading"]);
    assert_eq!(placed.entries[0].1, 3);

    // Every rule names a paragraph style that exists.
    let doc = Document::open(&path).unwrap();
    for rule in placed.rules.iter().chain(document_wide.rules.iter()) {
        let style = rule.paragraph_style.expect("a rule names a style");
        assert!(doc.text_style(style).is_some(), "style {style}");
    }
}

/// Columns, and the unit finding: widths and gaps are fractions of the text
/// width, not points. The one non-equal layout in the whole install adds up to
/// exactly one.
#[test]
fn column_widths_are_fractions_that_add_up() {
    let path = fixture!("pages-columns.pages");
    let doc = Document::open(&path).unwrap();
    let structure = doc.structure().unwrap();
    let body = structure.body_storage.unwrap();
    let layouts = doc.column_layouts(body);
    assert_eq!(layouts.len(), 2, "one non-equal layout and one equal one");

    let unequal = layouts
        .iter()
        .find(|l| l.unequal.is_some())
        .expect("the only non-equal columns in the install");
    assert_eq!(unequal.count(), 2);
    let sum: f32 = unequal.fractions().iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-4,
        "the fractions cover the text width: {sum} from {:?}",
        unequal.fractions()
    );

    let equal = layouts
        .iter()
        .find(|l| l.equal.is_some())
        .expect("and an equal two-column one");
    assert_eq!(equal.count(), 2);
    let (_, gap) = equal.equal.unwrap();
    assert!(
        gap > 0.0 && gap < 0.2,
        "a gap of {gap} is a fraction, not a point measurement"
    );
    let sum: f32 = equal.fractions().iter().sum();
    assert!((sum - 1.0).abs() < 1e-4, "{sum}");

    // The two layouts tile the body, one entry per paragraph range.
    assert_eq!(layouts[0].end, layouts[1].start);
}

/// The tripwire fired: `pages-footnotes.pages` and `pages-bookmarks.pages`
/// exist now, made through the Insert menu on an unlocked screen. Footnotes
/// and bookmarks live exactly where the schema said they would, and every
/// document that was not given one still has none.
#[test]
fn footnotes_and_bookmarks_exist_exactly_where_they_were_made() {
    corpus!();
    let mut checked = 0usize;
    for path in pages_fixtures() {
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        let structure = Document::open(&path).unwrap().structure().unwrap();
        match name.as_str() {
            "pages-footnotes.pages" => {
                assert_eq!(structure.footnotes.len(), 2, "{name}");
                for note in &structure.footnotes {
                    assert!(
                        note.attachment.is_some(),
                        "{name}: a mark with no attachment"
                    );
                    assert!(note.body.is_some(), "{name}: a mark with no note body");
                    assert!(
                        note.text.contains("Fussnote"),
                        "{name}: the note text did not come back: {:?}",
                        note.text
                    );
                }
                let mut indexes: Vec<u64> = structure.footnotes.iter().map(|n| n.index).collect();
                indexes.sort_unstable();
                assert_eq!(indexes, vec![26, 77], "{name}: marks moved");
            }
            "pages-bookmarks.pages" => {
                // Two were made through Insert > Bookmark; the table also
                // carries a terminator entry, which is not a bookmark.
                assert_eq!(structure.bookmarks.len(), 2, "{name}");
                assert!(
                    structure.bookmarks.iter().all(|(_, _, b)| b.is_some()),
                    "{name}: a bookmark with no archive"
                );
                assert!(structure.footnotes.is_empty(), "{name}");
            }
            _ => {
                assert!(structure.footnotes.is_empty(), "{name} grew a footnote");
                assert!(structure.bookmarks.is_empty(), "{name} grew a bookmark");
            }
        }
        assert_eq!(structure.footnote_settings.kind, 0, "{name}");
        assert_eq!(structure.footnote_settings.format, 0, "{name}");
        assert_eq!(structure.footnote_settings.numbering, 0, "{name}");
        assert_eq!(structure.footnote_settings.gap, 10, "{name}");
        checked += 1;
    }
    assert!(checked >= 5);
}

/// A storage of kind 2 is a footnote body. `pages-footnotes.pages` has
/// exactly its two, and no other document has any — the boundary the phase
/// once reported, now measured from the observed side.
#[test]
fn footnote_bodies_are_kind_2_storages_and_only_where_footnotes_are() {
    corpus!();
    let mut storages = 0usize;
    for path in pages_fixtures() {
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        let doc = Document::open(&path).unwrap();
        let bodies = doc.storages().iter().filter(|s| s.kind == 2).count();
        if name == "pages-footnotes.pages" {
            assert_eq!(bodies, 2, "{name}: two footnotes, two note bodies");
        } else {
            assert_eq!(bodies, 0, "{name}: a footnote body with no footnote");
        }
        storages += doc.storages().len();
    }
    assert!(storages >= 100, "only {storages} storages were seen");
}

// -- writing -----------------------------------------------------------------

/// Deleting the `U+0004` a section begins after is refused by name, and the
/// document is left untouched.
#[test]
fn deleting_a_section_break_is_refused() {
    let path = fixture!("pages-report.pages");
    let doc = Document::open(&path).unwrap();
    let structure = doc.structure().unwrap();
    let body = structure.body_storage.unwrap();
    let second = &structure.sections[1];
    let break_at = second.start - 1;

    // Exactly the break, and a range that swallows it along with text either
    // side. (A full-range replace is refused too, but by
    // `Error::AnchoredObject` — `pages-report` has a photo anchored at
    // character 12 and that objection comes first.)
    for range in [break_at..break_at + 1, break_at - 5..break_at + 5] {
        let mut doc = Document::open(&path).unwrap();
        let before = doc.storage_text(body).unwrap();
        match doc.delete_text(body, range.clone()) {
            Err(Error::SectionBreak {
                storage,
                index,
                section,
            }) => {
                assert_eq!(storage, body);
                assert_eq!(index, break_at);
                assert_eq!(section, Some(second.identifier));
            }
            other => panic!("{range:?} was not refused: {other:?}"),
        }
        assert_eq!(doc.storage_text(body).unwrap(), before, "{range:?}");
        assert!(doc.changed_streams().is_empty(), "{range:?}");
    }

    // And an edit that stays inside a section is not refused.
    let mut doc = Document::open(&path).unwrap();
    doc.delete_text(body, 100..140).unwrap();
    assert!(!doc.changed_streams().is_empty());
}

/// Writing a header is writing a text storage: the remapping is the same, the
/// document still checks out, and only the stream the storage lives in is
/// rewritten.
#[test]
fn writing_a_header_rewrites_one_stream_and_nothing_else() {
    let path = fixture!("pages-layout.pages");
    let doc = Document::open(&path).unwrap();
    // A header or footer with text and no smart field: writing into one that
    // carries a date field is not a no-op (it freezes the field — see the next
    // test), so the byte-identity assertion at the end needs a plain one, and
    // asserting about the very storage just edited means it has to be plain
    // from the top. (pages-layout's plain text storage is a footer.)
    let strip = doc
        .header_footers()
        .into_iter()
        .find(|hf| !hf.text.is_empty() && smart_fields_of(&doc, hf.storage) == 0)
        .expect("pages-layout has a plain header or footer with text");

    let mut edited = Document::open(&path).unwrap();
    edited
        .set_text(strip.storage, "Neuer Text von iwork-rs")
        .unwrap();
    assert_eq!(edited.changed_streams().len(), 1);
    assert!(edited.problems().is_empty(), "{:?}", edited.problems());

    let found = edited
        .header_footers()
        .into_iter()
        .find(|hf| hf.storage == strip.storage)
        .unwrap();
    assert_eq!(found.text, "Neuer Text von iwork-rs");
    assert_eq!(found.zone, strip.zone);
    assert_eq!(found.section, strip.section);
    assert_eq!(found.footer, strip.footer);

    // Writing a storage the text it already holds changes nothing at all — and
    // this is the very storage just edited, re-opened clean, not some other
    // strip that happens to be plain.
    let mut same = Document::open(&path).unwrap();
    same.set_text(strip.storage, &strip.text).unwrap();
    assert!(
        same.changed_streams().is_empty(),
        "rewriting a storage its own text is a no-op"
    );
}

/// How many smart-field runs a storage carries — the reason rewriting a header
/// is not always a no-op.
fn smart_fields_of(doc: &Document, storage: u64) -> usize {
    doc.storages()
        .into_iter()
        .find(|s| s.identifier == storage)
        .map(|s| {
            s.tables
                .iter()
                .filter(|t| t.field == 11)
                .map(|t| t.entries)
                .sum()
        })
        .unwrap_or(0)
}

/// **A header's text is often not text.** The date in `pages-layout`'s header
/// is a `TSWP.DateTimeSmartFieldArchive`, and the storage holds the string it
/// last rendered to; replacing the text removes the field and freezes the date,
/// so the header stops updating. That is a consequence worth failing over if it
/// ever changes silently, and it is why the crate reports the tables it
/// rewrote.
#[test]
fn rewriting_a_header_that_holds_a_date_field_replaces_the_field_with_its_text() {
    let path = fixture!("pages-layout.pages");
    let doc = Document::open(&path).unwrap();
    let dated = doc
        .header_footers()
        .into_iter()
        .find(|hf| smart_fields_of(&doc, hf.storage) > 0)
        .expect("the newsletter's header has a date field");
    // The field rendered a full date, so the frozen text carries a four-digit
    // 20xx year — not merely "20", which any time of day contains too.
    let has_year =
        dated.text.as_bytes().windows(4).any(|w| {
            w == b"2026" || (w.starts_with(b"20") && w.iter().all(|b| b.is_ascii_digit()))
        });
    assert!(
        has_year,
        "the header's date field rendered a year: {:?}",
        dated.text
    );

    let mut edited = Document::open(&path).unwrap();
    let report = edited.set_text(dated.storage, "Kopfzeile").unwrap();
    assert!(
        report.report.tables.contains(&11),
        "the smart-field table was rewritten: {:?}",
        report.report.tables
    );
    assert_eq!(
        smart_fields_of(&edited, dated.storage),
        0,
        "the date field went with the text it rendered"
    );
    assert!(edited.problems().is_empty());
}

/// Replace the payload of the object `identifier` names, wherever in the
/// package it lives, with `payload`.
fn forge_object(path: &Path, identifier: u64, payload: Vec<u8>) -> Package {
    let mut package = Package::read(path).unwrap();
    let mut done = false;
    for (name, data) in package.entries.iter_mut() {
        if !name.ends_with(".iwa") {
            continue;
        }
        let Ok(mut objects) = iwa::parse(data) else {
            continue;
        };
        let mut touched = false;
        for object in &mut objects {
            if object.identifier == identifier {
                object.messages[0].payload = payload.clone();
                touched = true;
            }
        }
        if touched {
            *data = iwa::serialize(&objects);
            done = true;
        }
    }
    assert!(done, "no object {identifier} to forge");
    package
}

/// A header strip's zones are assigned by an entry's index in the full repeated
/// field, not by its position among the entries that decoded. A nil or
/// unrecognised middle entry used to slide the right-hand storage into the
/// Centre zone; now it keeps its place and the right storage is still Right.
#[test]
fn a_nil_middle_header_does_not_mislabel_the_right_zone() {
    let path = fixture!("pages-layout.pages");
    let doc = Document::open(&path).unwrap();

    // A section template with all three header zones filled by distinct storages.
    let template = doc
        .header_footers()
        .into_iter()
        .filter(|hf| !hf.footer)
        .fold(BTreeMap::<u64, Vec<(Zone, u64)>>::new(), |mut acc, hf| {
            acc.entry(hf.section_template)
                .or_default()
                .push((hf.zone, hf.storage));
            acc
        })
        .into_iter()
        .find(|(_, zones)| {
            zones.iter().any(|(z, _)| *z == Zone::Left)
                && zones.iter().any(|(z, _)| *z == Zone::Centre)
                && zones.iter().any(|(z, _)| *z == Zone::Right)
        })
        .map(|(id, _)| id)
        .expect("pages-layout has a header strip with three zones");
    let right_storage = doc
        .header_footers()
        .into_iter()
        .find(|hf| !hf.footer && hf.section_template == template && hf.zone == Zone::Right)
        .unwrap()
        .storage;

    // Rewrite the template's header field (1) so its middle entry is a varint,
    // not a bare reference — a nil in the strip. The left and right stay bytes.
    let (_, template_archive) = doc.object(template).unwrap();
    let mut archive = pb::Message::decode(template_archive.payload()).unwrap();
    let mut header_index = 0usize;
    for field in archive.fields.iter_mut() {
        if field.number != 1 {
            continue;
        }
        if header_index == 1 {
            field.value = pb::Value::Varint(0);
        }
        header_index += 1;
    }
    assert!(header_index >= 3, "the strip had three header entries");
    let package = forge_object(&path, template, archive.encode());

    let forged = Document::from_package(package).unwrap();
    let zones: Vec<(Zone, u64)> = forged
        .header_footers()
        .into_iter()
        .filter(|hf| !hf.footer && hf.section_template == template)
        .map(|hf| (hf.zone, hf.storage))
        .collect();
    // The right storage is still Right — not slid into Centre — and the nil
    // middle entry simply has no zone.
    assert!(
        zones.contains(&(Zone::Right, right_storage)),
        "the right storage kept its zone: {zones:?}"
    );
    assert!(
        !zones.iter().any(|(z, _)| *z == Zone::Centre),
        "the nil middle entry did not become a zone: {zones:?}"
    );
}

// -- the app -----------------------------------------------------------------

fn script(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("scripts")
        .join(name)
}

/// Ask Pages what the sections are: `(body flag, [(length, text)])`.
fn ask_the_app(path: &Path) -> (bool, Vec<(usize, String)>) {
    let out = std::process::Command::new(script("section-oracle.sh"))
        .arg(path)
        .output()
        .unwrap_or_else(|e| panic!("section-oracle.sh: {e}"));
    assert!(
        out.status.success(),
        "section-oracle.sh failed for {}: {}",
        path.display(),
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8_lossy(&out.stdout);
    let mut body = true;
    let mut sections = Vec::new();
    for line in text.lines() {
        let parts: Vec<&str> = line.split('\t').collect();
        match parts.as_slice() {
            ["body", flag] => body = *flag == "true",
            ["section", _, length, rest @ ..] => sections.push((
                length.parse().unwrap_or(0),
                rest.join("\t")
                    .replace("\\n", "\n")
                    .replace("\\r", "\r")
                    .replace("\\t", "\t")
                    .replace("\\\\", "\\"),
            )),
            _ => {}
        }
    }
    (body, sections)
}

/// The section arithmetic, put to the app.
///
/// Off unless `IWORK_APP_CHECK=1`. `body text of section i` is the only thing
/// Pages' dictionary will say about document structure, and it is exactly the
/// range this crate computes from table 17: the entry sits one past the
/// `U+0004`, and the break belongs to neither side. Compared character for
/// character, not just by length.
#[test]
fn the_app_agrees_about_every_section() {
    if std::env::var("IWORK_APP_CHECK").as_deref() != Ok("1") {
        eprintln!("IWORK_APP_CHECK is not 1 — skipping the app round trip");
        return;
    }
    let mut compared = 0usize;
    for name in [
        "pages-report.pages",
        "pages-numbering.pages",
        "pages-book.pages",
        "pages-layout.pages",
    ] {
        let Some(path) = generated(name) else {
            eprintln!("no {name} — skipping");
            continue;
        };
        let (body, said) = ask_the_app(&path);
        let doc = Document::open(&path).unwrap();
        let structure = doc.structure().unwrap();

        assert_eq!(
            body,
            structure.mode == Mode::WordProcessing,
            "{name}: the app's `document body` and TP.SettingsArchive.body disagree"
        );
        // **A page-layout document has no sections to the app.** `pages-layout`
        // carries two `TP.SectionArchive`s, three section templates each
        // and its thirty-six header and footer storages, and Pages answers
        // `count of sections` with 0 — the element is word-processing only, the
        // way the Document inspector is. The archives are still there and still
        // what the headers hang off, so the decoder reports them and this
        // comparison stops here.
        if !body {
            assert_eq!(
                said.len(),
                0,
                "{name}: a page-layout document has sections?"
            );
            assert!(
                !structure.sections.is_empty(),
                "{name}: …but the archive still has them"
            );
            continue;
        }
        assert_eq!(
            said.len(),
            structure.sections.len(),
            "{name}: section count"
        );

        let text = doc
            .storage_text(structure.body_storage.unwrap())
            .unwrap_or_default();
        let units: Vec<u16> = text.encode_utf16().collect();
        for (index, section) in structure.sections.iter().enumerate() {
            let (length, words) = &said[index];
            assert_eq!(
                section.length() as usize,
                *length,
                "{name}: section {} is {} unit(s) here and {length} to the app",
                index + 1,
                section.length()
            );
            let mine = String::from_utf16_lossy(
                &units[section.start as usize..(section.end as usize).min(units.len())],
            );
            assert_eq!(
                &mine,
                words,
                "{name}: section {} reads differently",
                index + 1
            );
            compared += 1;
        }
    }
    assert!(compared >= 10, "only {compared} sections were compared");
}

/// A header this crate wrote, put through Pages and read back out.
///
/// Off unless `IWORK_APP_CHECK=1`. Pages exposes no header text to a script —
/// its dictionary has `body text` of a document, a section and a page, and
/// nothing else — so "the app read it back" has to be arranged differently:
/// the app is made to open the edited document and **save it**, and the file on
/// disk afterwards is one Pages wrote. The header either came through its model
/// or it did not.
///
/// The storage identifier survives the round trip, which is worth noting: Pages
/// rewrote the document stream and kept the object's number.
#[test]
fn pages_saves_the_document_with_the_header_this_crate_wrote() {
    if std::env::var("IWORK_APP_CHECK").as_deref() != Ok("1") {
        eprintln!("IWORK_APP_CHECK is not 1 — skipping the app round trip");
        return;
    }
    let path = fixture!("pages-layout.pages");
    let doc = Document::open(&path).unwrap();
    let header = doc
        .header_footers()
        .into_iter()
        .find(|hf| !hf.footer && !hf.text.is_empty())
        .expect("pages-layout has header text");
    let footer = doc
        .header_footers()
        .into_iter()
        .find(|hf| hf.footer && hf.section == header.section)
        .expect("and a footer beside it");

    let out = std::env::temp_dir().join("iwork-header.pages");
    let _ = std::fs::remove_file(&out);
    let mut edited = Document::open(&path).unwrap();
    edited
        .set_text(header.storage, "Kopfzeile von iwork-rs")
        .unwrap();
    edited.set_text(footer.storage, "Fusszeile 42").unwrap();
    edited.save(&out).unwrap();

    let status = std::process::Command::new(script("resave.sh"))
        .arg(&out)
        .status()
        .unwrap_or_else(|e| panic!("resave.sh: {e}"));
    assert!(status.success(), "Pages would not open and save {out:?}");

    let after = Document::open(&out).unwrap();
    let zones = after.header_footers();
    assert!(
        zones.iter().any(|hf| hf.text == "Kopfzeile von iwork-rs"),
        "Pages saved the document without the header: {:?}",
        zones
            .iter()
            .filter(|hf| !hf.text.is_empty())
            .map(|hf| (hf.storage, hf.text.clone()))
            .collect::<Vec<_>>()
    );
    assert!(
        zones.iter().any(|hf| hf.text == "Fusszeile 42"),
        "…or without the footer"
    );
    // And the document Pages wrote still keeps every rule this crate checks.
    assert!(after.problems().is_empty(), "{:?}", after.problems());
}

/// Where a page number's **format** lives, which is not on the section.
///
/// A section says whether numbering continues or restarts and at what; what
/// the number is *drawn as* is on the `TSWP.NumberAttachmentArchive` behind
/// the `U+FFFC` in the footer. So a document can number one section in roman
/// and another in arabic without either section archive saying anything.
#[test]
fn a_page_number_carries_its_own_format() {
    let path = fixture!("pages-numbering.pages");
    let doc = Document::open(&path).unwrap();
    let numbered: Vec<_> = doc
        .header_footers()
        .into_iter()
        .filter(|hf| !hf.numbers.is_empty())
        .collect();
    assert!(!numbered.is_empty(), "the footer holds a page number");
    for entry in &numbered {
        // The storage is a lone object-replacement character: the number is
        // not text, it is a thing standing in for one.
        assert_eq!(entry.text, "\u{FFFC}");
        for number in &entry.numbers {
            assert_eq!(number.kind, 0, "a page number, not a page count");
            assert_eq!(number.kind_name(), "page number");
            assert_eq!(number.format_name, "decimal");
            assert_eq!(number.index, 0);
            assert_eq!(number.storage, entry.storage);
        }
    }

    // And a header with ordinary text has none.
    let plain = doc
        .header_footers()
        .into_iter()
        .find(|hf| hf.text == "1")
        .or_else(|| {
            doc.header_footers()
                .into_iter()
                .find(|hf| hf.text.is_empty())
        })
        .unwrap();
    assert!(plain.numbers.is_empty());
}

// -- change tracking (phase 7) ------------------------------------------------

/// **A tripwire, and a survey.** Change tracking has a document half in `TP`
/// and a text half in the storage's tables. `pages-tracked.pages` — Track
/// Changes turned on from the menu on an unlocked screen — is the one document
/// that has either: `enabled` true, one session, and a `table_deletion` in its
/// body (the text half, refused by `src/annotations.rs`'s decoders and pinned
/// in `tests/text.rs`). Every other Pages document in the corpus, and all 640
/// bundled templates, keeps the ten fields at their schema defaults; Pages'
/// dictionary has no change-tracking property, so nothing here moves a display
/// switch on its own.
#[test]
fn change_tracking_is_on_exactly_where_it_was_turned_on() {
    for path in pages_fixtures() {
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        let doc = Document::open(&path).unwrap();
        let tracking = doc.structure().unwrap().change_tracking;
        if name == "pages-tracked.pages" {
            assert!(
                tracking.enabled,
                "{name}: Track Changes was on when it was saved"
            );
            assert_eq!(tracking.sessions, 1, "{name}");
            assert!(tracking.has_recent_session, "{name}");
        } else {
            assert!(
                tracking.is_default(),
                "{name}: change tracking is no longer at its defaults — {tracking:?}"
            );
        }
    }
}
