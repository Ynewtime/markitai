//! Integration tests against real documents.
//!
//! No iWork files are committed to this repository — they are other people's
//! documents, and the ones used during development are copyrighted. Drop any
//! `.pages`, `.numbers` or `.key` files into `tests/fixtures/` (or point
//! `IWORK_FIXTURES` at a directory) and these tests will exercise every one of
//! them. With no fixtures present they pass without asserting anything, and say
//! so.

use std::path::{Path, PathBuf};

use iwork::pb::Message;
use iwork::{iwa, style, Document, Kind, Package, StyleKind};

fn fixtures() -> Vec<PathBuf> {
    let mut found = all_fixtures();
    found.retain(|path| !is_encrypted(path));
    found
}

fn all_fixtures() -> Vec<PathBuf> {
    let dir = std::env::var("IWORK_FIXTURES")
        .map(PathBuf::from)
        .unwrap_or_else(|_| Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures"));
    let mut found = Vec::new();
    collect(&dir, &mut found);
    found.sort();
    found
}

/// A password-protected package, which every other test here must not be given:
/// its object streams are ciphertext, so `Document::open` refuses it by design
/// and the app will not open it without a password either.
///
/// The test is the package's own shape rather than the file's name, because a
/// reader pointing `IWORK_FIXTURES` at their own documents may well have one
/// and will not have named it `pages-locked`.
fn is_encrypted(path: &Path) -> bool {
    Package::read(path).is_ok_and(|package| package.contains(".iwpv2"))
}

/// Fixtures may be nested: `scripts/make-fixtures.sh` writes the documents it
/// builds with Pages, Numbers and Keynote into `tests/fixtures/generated/`, and
/// a directory the reader points `IWORK_FIXTURES` at is likely to have folders
/// in it too. So the search recurses, and a document dropped straight into
/// `tests/fixtures/` still works exactly as before.
fn collect(dir: &Path, found: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.filter_map(Result::ok) {
        let path = entry.path();
        let named = matches!(
            path.extension().and_then(|e| e.to_str()),
            Some("pages") | Some("numbers") | Some("key")
        );
        // Extension first: a pre-2013 `.pages` is a bundle, and descending into
        // one would find nothing and take a while about it.
        if named {
            if path.is_file() {
                found.push(path);
            }
        } else if path.is_dir() {
            collect(&path, found);
        }
    }
}

/// One fixture by file name, or `None` (and a note) when the corpus lacks it.
fn fixture_named(name: &str) -> Option<PathBuf> {
    let found = fixtures().into_iter().find(|p| {
        p.file_name()
            .map(|n| n.to_string_lossy() == name)
            .unwrap_or(false)
    });
    if found.is_none() {
        eprintln!("no {name} in tests/fixtures — skipping (scripts/make-fixtures.sh --ui)");
    }
    found
}

/// Print once per test run rather than failing, so a fresh clone is green.
fn require_fixtures() -> Vec<PathBuf> {
    let found = fixtures();
    if found.is_empty() {
        eprintln!("no fixtures in tests/fixtures — skipping (see tests/fixtures/README.md)");
    }
    found
}

#[test]
fn opens_and_identifies_every_fixture() {
    for path in require_fixtures() {
        let doc = Document::open(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        assert_ne!(
            doc.kind(),
            Kind::Unknown,
            "{}: could not identify the document",
            path.display()
        );
        assert!(
            doc.objects().count() > 0,
            "{}: no objects decoded",
            path.display()
        );
        assert!(
            doc.package().contains("Metadata/DocumentIdentifier"),
            "{}: not an iWork package",
            path.display()
        );
    }
}

/// Every entry in an iWork package is stored, never deflated, and this crate
/// must keep it that way — the media is meant to be mappable in place.
#[test]
fn writes_only_stored_entries() {
    for path in require_fixtures() {
        let doc = Document::open(&path).unwrap();
        let out = std::env::temp_dir().join(format!(
            "iwork-stored-{}",
            path.file_name().unwrap().to_string_lossy()
        ));
        doc.save(&out).unwrap();

        let bytes = std::fs::read(&out).unwrap();
        let mut zip = zip::ZipArchive::new(std::io::Cursor::new(bytes)).unwrap();
        for i in 0..zip.len() {
            let entry = zip.by_index(i).unwrap();
            assert_eq!(
                entry.compression(),
                zip::CompressionMethod::Stored,
                "{}: {} was compressed",
                path.display(),
                entry.name().unwrap() // markitai: zip 9
            );
        }
        let _ = std::fs::remove_file(&out);
    }
}

/// Saving an unedited document must reproduce every entry **byte for byte** —
/// not merely an equivalent stream. A save that rewrites what it did not change
/// makes an intended edit indistinguishable from an incidental re-compression.
#[test]
fn object_streams_survive_a_roundtrip() {
    for path in require_fixtures() {
        let original = Package::read(&path).unwrap();
        let doc = Document::open(&path).unwrap();
        let out = std::env::temp_dir().join(format!(
            "iwork-roundtrip-{}",
            path.file_name().unwrap().to_string_lossy()
        ));
        assert!(
            doc.changed_streams().is_empty(),
            "{}: an unedited document reports changed streams: {:?}",
            path.display(),
            doc.changed_streams()
        );
        doc.save(&out).unwrap();
        let rewritten = Package::read(&out).unwrap();

        assert_eq!(
            original.names().collect::<Vec<_>>(),
            rewritten.names().collect::<Vec<_>>(),
            "{}: entry names or order changed",
            path.display()
        );

        for name in original.names() {
            let before = original.get(name).unwrap();
            let after = rewritten.get(name).unwrap();
            assert_eq!(
                before,
                after,
                "{}: {name} was rewritten by a save that changed nothing",
                path.display()
            );
        }
        let _ = std::fs::remove_file(&out);
    }
}

/// Editing one storage must leave every other object in the document alone.
#[test]
fn editing_text_touches_only_its_own_stream() {
    for path in require_fixtures() {
        let doc = Document::open(&path).unwrap();
        // Not simply the first storage: one with an anchored image is refused
        // by name, which `text_that_anchors_an_object_is_not_replaced` asserts.
        let Some(target) = doc.text_storages().into_iter().find(|storage| {
            Document::open(&path)
                .unwrap()
                .set_text(storage.identifier, "x")
                .is_ok()
        }) else {
            continue; // a document with no text is fine, just not useful here
        };

        let mut edited = Document::open(&path).unwrap();
        let replacement = "Ersetzt durch iwork-rs — 🎬";
        edited.set_text(target.identifier, replacement).unwrap();
        assert_eq!(
            edited.changed_streams(),
            vec![target.stream.as_str()],
            "{}: editing one storage should change one stream",
            path.display()
        );
        let out = std::env::temp_dir().join(format!(
            "iwork-edit-{}",
            path.file_name().unwrap().to_string_lossy()
        ));
        edited.save(&out).unwrap();

        let reopened = Document::open(&out).unwrap();
        let found = reopened
            .text_storages()
            .into_iter()
            .find(|s| s.identifier == target.identifier)
            .expect("edited storage is still there");
        assert_eq!(found.text, replacement, "{}", path.display());

        assert_eq!(
            reopened.objects().count(),
            doc.objects().count(),
            "{}: object count changed",
            path.display()
        );

        // Only the stream holding the edited storage may differ.
        let before = Package::read(&path).unwrap();
        let after = Package::read(&out).unwrap();
        for name in before.names().filter(|n| n.ends_with(".iwa")) {
            if name == target.stream {
                continue;
            }
            assert_eq!(
                iwa::decompress(before.get(name).unwrap()).unwrap(),
                iwa::decompress(after.get(name).unwrap()).unwrap(),
                "{}: {name} changed while editing {}",
                path.display(),
                target.stream
            );
        }
        let _ = std::fs::remove_file(&out);
    }
}

/// Components must resolve to entries that actually exist. Numbers is the real
/// test here: it spreads ~100 components over `Index/Tables/`.
#[test]
fn components_resolve_to_real_streams() {
    for path in require_fixtures() {
        let doc = Document::open(&path).unwrap();
        for component in doc.components() {
            let name = component.stream_name();
            assert!(
                doc.package().contains(&name),
                "{}: component {} ({}) points at missing {name}",
                path.display(),
                component.identifier,
                component.preferred_name
            );
        }
    }
}

/// Every style a run of text points at must be an object that exists, and the
/// three tables must point at the kind of style this crate says they do. This
/// is the claim `style::StyleKind::attribute_table` rests on, so a document
/// that disagrees should fail loudly rather than be edited on a wrong premise.
#[test]
fn attribute_tables_point_at_styles_of_the_matching_kind() {
    for path in require_fixtures() {
        let doc = Document::open(&path).unwrap();
        let styles: Vec<_> = doc
            .text_styles()
            .into_iter()
            .map(|s| (s.identifier, s.kind))
            .collect();

        for (stream, object) in doc.objects() {
            if object.message_type() != iwork::TYPE_STORAGE {
                continue;
            }
            let Ok(storage) = Message::decode(object.payload()) else {
                continue;
            };
            for kind in [StyleKind::Character, StyleKind::Paragraph, StyleKind::List] {
                let Some(table) = storage
                    .bytes(kind.attribute_table())
                    .and_then(iwork::pb::decode_nested)
                else {
                    continue;
                };
                for run in style::runs(&table) {
                    let Some(target) = run.style else { continue };
                    assert!(
                        doc.object(target).is_some(),
                        "{}: {stream} storage {} points at missing object {target}",
                        path.display(),
                        object.identifier
                    );
                    if let Some((_, found)) = styles.iter().find(|(id, _)| *id == target) {
                        assert_eq!(
                            *found,
                            kind,
                            "{}: storage {} field {} points at a {} style",
                            path.display(),
                            object.identifier,
                            kind.attribute_table(),
                            found.as_str()
                        );
                    }
                }
            }
        }
    }
}

/// Styles say what kind they are, in their own internal identifier, and the
/// message types must agree.
///
/// This is the check that settles 2021 against 2022. Public prior art has them
/// the other way round and this crate copied that; across six documents every
/// type 2021 calls itself `character-style-…` and every type 2022
/// `…-paragraphstyle-…`, with no exceptions in either direction.
#[test]
fn style_types_match_the_identifiers_the_styles_give_themselves() {
    let mut counted = 0;
    for path in require_fixtures() {
        let doc = Document::open(&path).unwrap();
        for style in doc.text_styles() {
            let Some(identifier) = &style.style_identifier else {
                continue;
            };
            let lower = identifier.to_ascii_lowercase();
            let claimed = if lower.contains("characterstyle") || lower.contains("character-style") {
                StyleKind::Character
            } else if lower.contains("paragraphstyle") || lower.contains("paragraph-style") {
                StyleKind::Paragraph
            } else if lower.contains("liststyle") || lower.contains("list-style") {
                StyleKind::List
            } else {
                continue;
            };
            counted += 1;
            assert_eq!(
                style.kind,
                claimed,
                "{}: style {} calls itself {identifier:?} but is message type {}",
                path.display(),
                style.identifier,
                style.kind.message_type()
            );
        }
    }
    if counted > 0 {
        eprintln!("{counted} styles agreed with their own identifiers");
    }
}

/// The two tables are the other way round from what the field order suggests:
/// field 5 is the paragraph table, field 8 the character one.
///
/// The check that shows it: every entry in field 5 sits at a paragraph boundary.
/// A character-attribute table has no reason to, and field 8's does not.
#[test]
fn the_paragraph_table_holds_entries_at_paragraph_starts() {
    for path in require_fixtures() {
        let doc = Document::open(&path).unwrap();
        for (_, object) in doc.objects() {
            if object.message_type() != iwork::TYPE_STORAGE {
                continue;
            }
            let Ok(storage) = Message::decode(object.payload()) else {
                continue;
            };
            let text = doc.storage_text(object.identifier).unwrap();
            let starts: Vec<u64> = iwork::text::paragraph_ranges(&text)
                .into_iter()
                .map(|r| r.start)
                .collect();
            if starts.is_empty() {
                continue;
            }
            let Some(table) = storage
                .bytes(StyleKind::Paragraph.attribute_table())
                .and_then(iwork::pb::decode_nested)
            else {
                continue;
            };
            let length = iwork::text::length(&text);
            for run in style::runs(&table) {
                // A trailing entry at the very end of the text is normal — it is
                // where the style of a paragraph yet to be typed comes from.
                assert!(
                    starts.contains(&run.start) || run.start == length,
                    "{}: storage {} paragraph run at {} is neither a paragraph start \
                     nor the end of the text ({length}) ({starts:?})",
                    path.display(),
                    object.identifier,
                    run.start
                );
            }
        }
    }
}

/// Copying a style must add exactly one object, leave the text alone, take an
/// identifier the document has not used, and keep the copy the same *kind* of
/// style as the template — named or variation, not a mix of the two.
#[test]
fn creating_a_style_adds_one_object_and_nothing_else() {
    for path in require_fixtures() {
        let doc = Document::open(&path).unwrap();
        let Some(template) = doc.text_styles().into_iter().next() else {
            continue; // a document with no styles is fine, just not useful here
        };
        let used: Vec<u64> = doc.objects().map(|(_, o)| o.identifier).collect();

        let mut edited = Document::open(&path).unwrap();
        let created = edited
            .create_text_style(template.identifier, "iwork-rs")
            .unwrap();
        assert!(
            !used.contains(&created.identifier),
            "{}: identifier {} was already in use",
            path.display(),
            created.identifier
        );

        let out = std::env::temp_dir().join(format!(
            "iwork-new-style-{}",
            path.file_name().unwrap().to_string_lossy()
        ));
        edited.save(&out).unwrap();

        let reopened = Document::open(&out).unwrap();
        assert_eq!(
            reopened.objects().count(),
            doc.objects().count() + 1,
            "{}: object count",
            path.display()
        );
        let new = reopened.text_style(created.identifier).unwrap();
        assert_eq!(new.kind, template.kind, "{}", path.display());
        // A named template hands its copy the new name; a variation stays
        // anonymous, because a named variation is what Pages refuses to open.
        assert_eq!(
            new.name.as_deref(),
            created.name.as_deref(),
            "{}: the report disagrees with the document",
            path.display()
        );
        assert_eq!(
            created.name.is_some(),
            template.name.is_some(),
            "{}: naming must follow the template",
            path.display()
        );
        assert_eq!(
            new.parent,
            template.parent,
            "{}: the copy inherits what the template did",
            path.display()
        );
        assert_eq!(
            reopened.last_object_identifier(),
            Some(created.identifier),
            "{}: high-water mark was not bumped",
            path.display()
        );

        let before: Vec<_> = doc.text_storages().into_iter().map(|s| s.text).collect();
        let after: Vec<_> = reopened
            .text_storages()
            .into_iter()
            .map(|s| s.text)
            .collect();
        assert_eq!(before, after, "{}: text changed", path.display());
        let _ = std::fs::remove_file(&out);
    }
}

/// Applying a style must change which style a range uses and nothing else —
/// not the text, not the object count, not the other streams.
#[test]
fn applying_a_style_touches_only_its_own_stream() {
    for path in require_fixtures() {
        let doc = Document::open(&path).unwrap();
        let Some(target) = doc.text_storages().into_iter().next() else {
            continue;
        };
        let Some(style) = doc
            .text_styles()
            .into_iter()
            .find(|s| s.kind == StyleKind::Character)
        else {
            continue;
        };

        let mut edited = Document::open(&path).unwrap();
        let range = 0..1.min(target.text.encode_utf16().count() as u64);
        if range.is_empty() {
            continue;
        }
        edited
            .apply_text_style(target.identifier, range.clone(), style.identifier)
            .unwrap();
        let out = std::env::temp_dir().join(format!(
            "iwork-apply-style-{}",
            path.file_name().unwrap().to_string_lossy()
        ));
        edited.save(&out).unwrap();

        let reopened = Document::open(&out).unwrap();
        assert_eq!(
            reopened.storage_text(target.identifier).unwrap(),
            doc.storage_text(target.identifier).unwrap(),
            "{}: text changed",
            path.display()
        );
        assert_eq!(
            reopened.objects().count(),
            doc.objects().count(),
            "{}: object count changed",
            path.display()
        );
        assert!(
            reopened
                .text_style_usage(style.identifier)
                .iter()
                .any(|u| u.storage == target.identifier && u.range.start == range.start),
            "{}: the run was not applied",
            path.display()
        );

        let before = Package::read(&path).unwrap();
        let after = Package::read(&out).unwrap();
        // The metadata stream is allowed to move: pointing text at a style in
        // another component adds a declaration there.
        let metadata = metadata_stream(&doc);
        for name in before.names().filter(|n| n.ends_with(".iwa")) {
            if name == target.stream || Some(name) == metadata.as_deref() {
                continue;
            }
            assert_eq!(
                iwa::decompress(before.get(name).unwrap()).unwrap(),
                iwa::decompress(after.get(name).unwrap()).unwrap(),
                "{}: {name} changed while styling {}",
                path.display(),
                target.stream
            );
        }
        let _ = std::fs::remove_file(&out);
    }
}

/// Package entry holding `TSP.PackageMetadata`.
fn metadata_stream(doc: &Document) -> Option<String> {
    doc.objects()
        .find(|(_, o)| o.message_type() == iwork::TYPE_PACKAGE_METADATA)
        .map(|(stream, _)| stream.to_string())
}

/// A reference that leaves its component must be declared there, or iWork never
/// loads what it points at.
///
/// The documents Apple wrote satisfy this exactly — which is what makes it worth
/// asserting. A file this crate produced that violates it opened in Pages with
/// the edit silently missing, and another crashed on open.
#[test]
fn every_cross_component_reference_is_declared() {
    for path in require_fixtures() {
        let doc = Document::open(&path).unwrap();
        assert_eq!(
            doc.undeclared_references(),
            Vec::new(),
            "{}: undeclared cross-component reference(s)",
            path.display()
        );
    }
}

/// Applying a style from another component declares it, and declaring is
/// idempotent — running it again adds nothing.
#[test]
fn applying_a_style_declares_it_where_it_is_needed() {
    for path in require_fixtures() {
        let doc = Document::open(&path).unwrap();
        let Some(target) = doc.text_storages().into_iter().find(|s| !s.text.is_empty()) else {
            continue;
        };
        let Some(style) = doc
            .text_styles()
            .into_iter()
            .find(|s| s.kind == StyleKind::Paragraph && s.stream != target.stream)
        else {
            continue;
        };

        let mut edited = Document::open(&path).unwrap();
        let range = 0..1;
        edited
            .apply_text_style(target.identifier, range, style.identifier)
            .unwrap();
        assert_eq!(
            edited.undeclared_references(),
            Vec::new(),
            "{}: applying {} left it undeclared",
            path.display(),
            style.identifier
        );
        assert_eq!(
            edited.declare_external_references(),
            0,
            "{}: declaring is not idempotent",
            path.display()
        );
    }
}

/// A style that is listed in a stylesheet and names a parent is also grouped
/// under that parent. A copy that is listed but not grouped is a shape no real
/// document takes, and `Document::create_text_style` must not produce one.
#[test]
fn a_copied_style_is_grouped_under_its_parent() {
    for path in require_fixtures() {
        let doc = Document::open(&path).unwrap();
        let Some(template) = doc
            .text_styles()
            .into_iter()
            .find(|s| s.parent.is_some() && s.stylesheet.is_some())
        else {
            continue;
        };

        let mut edited = Document::open(&path).unwrap();
        let created = edited
            .create_text_style(template.identifier, "Copy")
            .unwrap();
        let sheet = Message::decode(
            edited
                .object(template.stylesheet.unwrap())
                .unwrap()
                .1
                .payload(),
        )
        .unwrap();
        assert_eq!(
            style::count_references(&sheet, created.identifier),
            style::count_references(&sheet, template.identifier),
            "{}: the copy of {} is listed in fewer places than its template",
            path.display(),
            template.identifier
        );
        assert!(
            edited.problems().is_empty(),
            "{}: {:?}",
            path.display(),
            edited.problems()
        );
    }
}

/// The one thing the rest of this file cannot prove: that the app opens it.
///
/// Off unless `IWORK_APP_CHECK=1`, because it drives Pages, Numbers or Keynote
/// through AppleScript and takes the best part of a minute per document. What
/// it runs is `scripts/app-check.sh`, the same harness every later phase uses to
/// accept a document it has written.
#[test]
fn every_fixture_opens_in_the_app_that_owns_it() {
    if std::env::var("IWORK_APP_CHECK").as_deref() != Ok("1") {
        eprintln!("IWORK_APP_CHECK is not 1 — skipping the app round-trip");
        return;
    }
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts/app-check.sh");
    for path in require_fixtures() {
        let status = std::process::Command::new(&script)
            .arg(&path)
            .status()
            .unwrap_or_else(|e| panic!("{}: {e}", script.display()));
        assert!(
            status.success(),
            "{}: the app that owns it would not open it",
            path.display()
        );
    }
}

/// Media registered in `TSP.PackageMetadata` must be present, except for theme
/// assets, which are referenced by path and deliberately not stored.
#[test]
fn registered_media_is_present() {
    for path in require_fixtures() {
        let doc = Document::open(&path).unwrap();
        for data in doc.data_files() {
            let Some(entry) = data.entry_name() else {
                assert!(
                    !data.asset_path.is_empty(),
                    "{}: media {} is neither stored nor a theme asset",
                    path.display(),
                    data.identifier
                );
                continue;
            };
            assert!(
                doc.package().contains(&entry),
                "{}: media {} points at missing {entry}",
                path.display(),
                data.identifier
            );
        }
    }
}

// -- metadata, identity and the review layer (phase 7) -----------------------

/// Every document has the three metadata entries, and they agree with one
/// another: `Metadata/DocumentIdentifier` is `documentUUID` again, and so is
/// `shareUUID`. Checked across the corpus because that agreement is what
/// `save_as_new` has to preserve.
#[test]
fn every_document_identifies_itself_the_same_way_three_times() {
    for path in require_fixtures() {
        let doc = Document::open(&path).unwrap();
        let metadata = doc
            .metadata()
            .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let properties = metadata
            .properties
            .unwrap_or_else(|| panic!("{}: no Properties.plist", path.display()));

        let document_uuid = properties
            .document_uuid
            .unwrap_or_else(|| panic!("{}: no documentUUID", path.display()));
        assert!(
            iwork::metadata::is_uuid(&document_uuid),
            "{}: documentUUID is {document_uuid}",
            path.display()
        );
        assert_eq!(
            metadata.document_identifier.as_deref(),
            Some(document_uuid.as_str()),
            "{}: Metadata/DocumentIdentifier disagrees with documentUUID",
            path.display()
        );
        assert_eq!(
            properties.share_uuid.as_deref(),
            Some(document_uuid.as_str()),
            "{}: shareUUID disagrees with documentUUID",
            path.display()
        );
        assert_eq!(
            properties.revision.as_deref(),
            Some(format!("0::{}", properties.version_uuid.clone().unwrap()).as_str()),
            "{}: revision is not 0:: plus the version UUID",
            path.display()
        );
        assert!(
            properties.other.is_empty(),
            "{}: Properties.plist has keys this crate does not name: {:?}",
            path.display(),
            properties.other
        );
    }
}

/// The document-level fields of the cross-app super, which sits at a different
/// field number in each of the three apps.
#[test]
fn every_document_names_its_locale_and_its_custom_format_list() {
    for path in require_fixtures() {
        let doc = Document::open(&path).unwrap();
        let metadata = doc.metadata().unwrap();
        assert!(
            metadata.locale.is_some(),
            "{}: no TSK.DocumentArchive.locale_identifier",
            path.display()
        );
        // The list is always there and is usually empty; what matters is that
        // it is the object `Document::custom_formats` reads.
        let list = metadata
            .custom_format_list
            .unwrap_or_else(|| panic!("{}: no custom format list", path.display()));
        let (_, object) = doc
            .object(list)
            .unwrap_or_else(|| panic!("{}: custom format list {list} is missing", path.display()));
        assert_eq!(
            object.message_type(),
            222,
            "{}: the document's custom format list is not a TSK.CustomFormatListArchive",
            path.display()
        );
    }
}

/// The tripwire fired: `pages-comments.pages` and `pages-tracked.pages`
/// exist now, made by driving the Pages UI on an unlocked screen. What used
/// to assert universal emptiness now asserts the two shapes actually
/// observed — and that every comment and change is anchored, since an
/// annotation nothing points at is a lost annotation.
#[test]
fn every_comment_and_change_is_anchored() {
    for path in require_fixtures() {
        let doc = Document::open(&path).unwrap();
        let annotations = doc.annotations();
        assert!(
            annotations.author_storage.is_some(),
            "{}: no TSK.AnnotationAuthorStorageArchive",
            path.display()
        );
        for comment in &annotations.comments {
            assert!(
                !matches!(comment.anchor, iwork::annotations::Anchor::Unattached),
                "{}: comment {} is unattached — either a real leak or an anchor \
                 route this crate does not walk",
                path.display(),
                comment.identifier
            );
        }
        for change in &annotations.changes {
            assert!(
                change.anchor.is_some(),
                "{}: tracked change {} is anchored nowhere this crate walks",
                path.display(),
                change.identifier
            );
        }
        assert!(
            annotations.unreached.is_empty(),
            "{}: annotation objects nothing points at: {:?}",
            path.display(),
            annotations.unreached
        );
    }
}

/// The comment fixture, pinned: two comments through the highlight table at
/// the character indices the UI recipe selected, and one real author.
#[test]
fn the_comment_fixture_reads_as_it_was_made() {
    let Some(path) = fixture_named("pages-comments.pages") else {
        return;
    };
    let doc = Document::open(&path).unwrap();
    let annotations = doc.annotations();
    assert_eq!(annotations.authors.len(), 1, "one comment author");
    assert_eq!(annotations.comments.len(), 2);
    let starts: Vec<u64> = annotations
        .comments
        .iter()
        .filter_map(|c| match c.anchor {
            iwork::annotations::Anchor::Text { start, .. } => Some(start),
            _ => None,
        })
        .collect();
    assert_eq!(
        starts,
        vec![0, 63],
        "the first word and the last word carry the comments"
    );
}

/// The tracked-changes fixture, pinned: two insertions and a deletion in one
/// storage, one change session, and the remapper refuses that storage.
#[test]
fn the_tracked_fixture_reads_as_it_was_made() {
    let Some(path) = fixture_named("pages-tracked.pages") else {
        return;
    };
    let doc = Document::open(&path).unwrap();
    let annotations = doc.annotations();
    let kinds: Vec<&str> = annotations
        .changes
        .iter()
        .map(|c| c.kind.as_str())
        .collect();
    assert_eq!(kinds.iter().filter(|k| k.contains("insertion")).count(), 2);
    assert_eq!(kinds.iter().filter(|k| k.contains("deletion")).count(), 1);
    assert_eq!(annotations.sessions.len(), 1, "one change session");
    assert_eq!(
        annotations.tracked_storages.len(),
        1,
        "one storage carries the changes"
    );
}

/// A copy saved with `save_as_new` is a *different document*: four UUIDs
/// change, the lineage does not, and every object stream is byte for byte the
/// one the original had.
#[test]
fn save_as_new_replaces_the_identity_and_nothing_else() {
    for path in require_fixtures() {
        let doc = Document::open(&path).unwrap();
        let before = doc.metadata().unwrap();
        let out = std::env::temp_dir().join(format!(
            "iwork-new-{}",
            path.file_name().unwrap().to_string_lossy()
        ));
        let identity = doc.save_as_new(&out).unwrap();

        let copy = Document::open(&out).unwrap();
        let after = copy.metadata().unwrap();
        let (was, now) = (before.properties.unwrap(), after.properties.unwrap());

        assert_ne!(was.document_uuid, now.document_uuid, "{}", path.display());
        assert_ne!(was.private_uuid, now.private_uuid, "{}", path.display());
        assert_ne!(was.version_uuid, now.version_uuid, "{}", path.display());
        assert_eq!(
            was.stable_document_uuid,
            now.stable_document_uuid,
            "{}: the lineage must survive a copy",
            path.display()
        );
        assert_eq!(
            now.document_uuid.as_deref(),
            Some(identity.document_uuid.as_str())
        );
        assert_eq!(now.share_uuid, now.document_uuid);
        assert_eq!(after.document_identifier, now.document_uuid);
        assert_eq!(
            after.build_versions,
            before.build_versions,
            "{}: the build history belongs to the file, not to the identity",
            path.display()
        );
        assert_eq!(was.file_format_version, now.file_format_version);

        // Everything that is not the identity is the original's bytes.
        let original = Package::read(&path).unwrap();
        let written = Package::read(&out).unwrap();
        for (name, data) in &original.entries {
            if name.starts_with("Metadata/") && name != "Metadata/BuildVersionHistory.plist" {
                continue;
            }
            assert_eq!(
                written.get(name),
                Some(data.as_slice()),
                "{}: {name} changed",
                path.display()
            );
        }
        let _ = std::fs::remove_file(&out);
    }
}

/// Two copies made in a row must not collide, or the whole exercise is
/// pointless.
#[test]
fn two_copies_of_one_document_get_two_identities() {
    let Some(path) = fixtures().into_iter().next() else {
        eprintln!("no fixtures — skipping");
        return;
    };
    let doc = Document::open(&path).unwrap();
    let a = std::env::temp_dir().join("iwork-copy-a.pages");
    let b = std::env::temp_dir().join("iwork-copy-b.pages");
    let first = doc.save_as_new(&a).unwrap();
    let second = doc.save_as_new(&b).unwrap();
    assert_ne!(first.document_uuid, second.document_uuid);
    assert_ne!(first.private_uuid, second.private_uuid);
    assert_ne!(first.version_uuid, second.version_uuid);
    assert_eq!(first.stable_document_uuid, second.stable_document_uuid);
    let _ = std::fs::remove_file(&a);
    let _ = std::fs::remove_file(&b);
}

/// A plain save leaves the identity alone — the other half of the same rule.
#[test]
fn a_plain_save_keeps_the_identity() {
    for path in require_fixtures() {
        let doc = Document::open(&path).unwrap();
        let out = std::env::temp_dir().join(format!(
            "iwork-same-{}",
            path.file_name().unwrap().to_string_lossy()
        ));
        doc.save(&out).unwrap();
        let before = Package::read(&path).unwrap();
        let after = Package::read(&out).unwrap();
        for entry in [
            "Metadata/Properties.plist",
            "Metadata/DocumentIdentifier",
            "Metadata/BuildVersionHistory.plist",
        ] {
            assert_eq!(
                before.get(entry),
                after.get(entry),
                "{}: {entry} changed on a plain save",
                path.display()
            );
        }
        let _ = std::fs::remove_file(&out);
    }
}

/// A password-protected package is refused by name, and says what it knows.
#[test]
fn a_password_protected_document_is_refused_by_name() {
    let locked: Vec<PathBuf> = all_fixtures()
        .into_iter()
        .filter(|p| is_encrypted(p))
        .collect();
    if locked.is_empty() {
        eprintln!("no password-protected fixture — run scripts/make-fixtures.sh pages-locked");
        return;
    }
    for path in locked {
        match Document::open(&path) {
            Err(iwork::Error::Encrypted { hint }) => {
                assert_eq!(
                    hint.as_deref(),
                    Some("the probe"),
                    "{}: the hint is stored in the clear and should come back",
                    path.display()
                );
            }
            Err(other) => panic!(
                "{}: refused as {other} rather than as encrypted",
                path.display()
            ),
            Ok(_) => panic!(
                "{}: opened a package this crate cannot read",
                path.display()
            ),
        }
        // The shape the probe measured, asserted rather than described.
        let package = Package::read(&path).unwrap();
        let encryption = iwork::metadata::encryption(&package).unwrap();
        assert_eq!(encryption.header_length, 104);
        assert_eq!(encryption.header_words, vec![2, 1, 100_000]);
        assert!(package.contains("Metadata/Properties.plist"));
        assert!(
            !package.contains("preview.jpg"),
            "{}: an encrypted package has no previews",
            path.display()
        );
        assert!(
            iwork::plist::parse(package.get("Metadata/Properties.plist").unwrap()).is_ok(),
            "{}: Properties.plist stays in the clear",
            path.display()
        );
        assert!(
            iwork::plist::parse(package.get("Metadata/BuildVersionHistory.plist").unwrap())
                .is_err(),
            "{}: BuildVersionHistory.plist does not",
            path.display()
        );
    }
}

/// Both plist forms in the package parse, and a binary one survives a rewrite.
#[test]
fn every_metadata_plist_round_trips() {
    for path in require_fixtures() {
        let package = Package::read(&path).unwrap();
        for name in [
            "Metadata/Properties.plist",
            "Metadata/BuildVersionHistory.plist",
        ] {
            let Some(raw) = package.get(name) else {
                continue;
            };
            let value = iwork::plist::parse(raw)
                .unwrap_or_else(|e| panic!("{}: {name}: {e}", path.display()));
            let again = iwork::plist::parse(&iwork::plist::write(&value))
                .unwrap_or_else(|e| panic!("{}: {name} rewritten: {e}", path.display()));
            assert_eq!(value, again, "{}: {name}", path.display());
        }
    }
}
