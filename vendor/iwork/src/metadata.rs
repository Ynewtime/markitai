//! Document metadata and document *identity* — everything about a package that
//! is not part of the object graph, plus the handful of object fields that
//! describe the document rather than its contents.
//!
//! ```text
//!  Metadata/Properties.plist        bplist00: ten keys, five of them UUIDs
//!  Metadata/DocumentIdentifier      36 bytes of plain text, no newline
//!  Metadata/BuildVersionHistory.plist   XML: which builds have written this file
//!
//!  <app>.DocumentArchive
//!    └── super → TSA.DocumentArchive          Pages 15, Numbers 8, Keynote 3
//!          ├── super (1) → TSK.DocumentArchive
//!          │      ├── locale_identifier (4), creation_locale_identifier (9)
//!          │      └── annotation_author_storage (7)
//!          ├── document_language (3)
//!          ├── template_identifier (9)
//!          └── custom_format_list (12) → TSK.CustomFormatListArchive (222)
//! ```
//!
//! ## The five UUIDs, and which of them may be shared
//!
//! Measured, not guessed: `pages-plain.pages` was opened by Pages and saved to
//! a second path with `save doc in file …`, and the two files compared.
//!
//! | Key | Save As | What it is |
//! |---|---|---|
//! | `documentUUID` | **new** | this file |
//! | `shareUUID` | **new**, equal to `documentUUID` | the iCloud sharing identity |
//! | `privateUUID` | **new** | |
//! | `versionUUID` | **new** | this *save* |
//! | `revision` | **new**, `"0::"` + `versionUUID` | generation and version |
//! | `stableDocumentUUID` | **unchanged** | the lineage: what the copy came from |
//! | `Metadata/DocumentIdentifier` | **new**, equal to `documentUUID` | |
//! | `BuildVersionHistory.plist` | unchanged | |
//! | `fileFormatVersion`, the three booleans | unchanged | |
//!
//! The original was left byte for byte untouched. That table is what
//! [`crate::Document::save_as_new`] implements, and the reason it exists:
//! `documentUUID` is what iCloud identifies a document by, so two edited copies
//! of one file that keep the same one are two versions of the same document as
//! far as the sync layer is concerned.
//!
//! `stableDocumentUUID` staying put is not an oversight — it is what the copy
//! *is*, and the corpus shows it doing its job: `numbers-large` and
//! `numbers-links` were built by different routes and share the stable UUID
//! `B95BC832-…`, because both descend from the same original.

use crate::document::Kind;
use crate::pb::Message;
use crate::plist::Plist;
use crate::Error;

/// `Metadata/Properties.plist`.
pub const PROPERTIES: &str = "Metadata/Properties.plist";
/// `Metadata/DocumentIdentifier` — the `documentUUID` again, as plain text.
pub const DOCUMENT_IDENTIFIER: &str = "Metadata/DocumentIdentifier";
/// `Metadata/BuildVersionHistory.plist`.
pub const BUILD_VERSION_HISTORY: &str = "Metadata/BuildVersionHistory.plist";

/// The keys of `Properties.plist` this crate names. Anything else a document
/// carries is reported by name and left alone.
pub mod key {
    pub const DOCUMENT_UUID: &str = "documentUUID";
    pub const SHARE_UUID: &str = "shareUUID";
    pub const STABLE_DOCUMENT_UUID: &str = "stableDocumentUUID";
    pub const PRIVATE_UUID: &str = "privateUUID";
    pub const VERSION_UUID: &str = "versionUUID";
    pub const REVISION: &str = "revision";
    pub const FILE_FORMAT_VERSION: &str = "fileFormatVersion";
    pub const IS_MULTI_PAGE: &str = "isMultiPage";
    pub const HAS_EXTERNAL_REFERENCE: &str = "hasExternalReferenceOrMissingData";
    pub const HAS_UNMATERIALIZED_REMOTE_DATA: &str = "hasUnmaterializedRemoteData";
}

/// Field numbers of `TSK.DocumentArchive`, the innermost document super.
pub mod tsk_field {
    pub const LOCALE_IDENTIFIER: u32 = 4;
    pub const ANNOTATION_AUTHOR_STORAGE: u32 = 7;
    pub const CREATION_LOCALE_IDENTIFIER: u32 = 9;
    pub const HAS_FLOATING_LOCALE: u32 = 11;
    pub const HAS_USER_DEFINED_LOCALE: u32 = 12;
    pub const FORMATTING_SYMBOLS: u32 = 17;
}

/// Field numbers of `TSA.DocumentArchive`, the cross-app document super.
pub mod tsa_field {
    pub const SUPER: u32 = 1;
    pub const DOCUMENT_LANGUAGE: u32 = 3;
    pub const CALCULATION_ENGINE: u32 = 4;
    pub const VIEW_STATE: u32 = 5;
    pub const TABLES_CUSTOM_FORMAT_LIST: u32 = 7;
    pub const TEMPLATE_IDENTIFIER: u32 = 9;
    pub const CUSTOM_FORMAT_LIST: u32 = 12;
}

/// Where `TSA.DocumentArchive` sits inside each app's root archive.
///
/// The apps do not agree, and there is no way to guess: Pages puts its super at
/// 15, Numbers at 8, Keynote at 3.
pub fn super_field(kind: Kind) -> Option<u32> {
    match kind {
        Kind::Pages => Some(15),
        Kind::Numbers => Some(8),
        Kind::Keynote => Some(3),
        Kind::Unknown => None,
    }
}

/// `Metadata/Properties.plist`, read.
#[derive(Debug, Clone)]
pub struct Properties {
    pub document_uuid: Option<String>,
    pub share_uuid: Option<String>,
    pub stable_document_uuid: Option<String>,
    pub private_uuid: Option<String>,
    pub version_uuid: Option<String>,
    /// `"<generation>::<versionUUID>"`. The generation is 0 in every document
    /// in this corpus, including ones saved repeatedly.
    pub revision: Option<String>,
    pub file_format_version: Option<String>,
    pub is_multi_page: Option<bool>,
    pub has_external_reference_or_missing_data: Option<bool>,
    pub has_unmaterialized_remote_data: Option<bool>,
    /// Keys this crate does not name, in file order. They are carried through
    /// a rewrite untouched; this is so a reader can see them.
    pub other: Vec<String>,
    /// The parsed plist, kept so a rewrite changes only what it means to.
    pub raw: Plist,
}

impl Properties {
    pub fn read(package: &crate::Package) -> Result<Option<Properties>, Error> {
        let Some(bytes) = package.get(PROPERTIES) else {
            return Ok(None);
        };
        let raw = crate::plist::parse(bytes)?;
        let text = |key: &str| raw.get(key).and_then(|v| v.as_str()).map(str::to_string);
        let flag = |key: &str| raw.get(key).and_then(|v| v.as_bool());
        let named = [
            key::DOCUMENT_UUID,
            key::SHARE_UUID,
            key::STABLE_DOCUMENT_UUID,
            key::PRIVATE_UUID,
            key::VERSION_UUID,
            key::REVISION,
            key::FILE_FORMAT_VERSION,
            key::IS_MULTI_PAGE,
            key::HAS_EXTERNAL_REFERENCE,
            key::HAS_UNMATERIALIZED_REMOTE_DATA,
        ];
        Ok(Some(Properties {
            document_uuid: text(key::DOCUMENT_UUID),
            share_uuid: text(key::SHARE_UUID),
            stable_document_uuid: text(key::STABLE_DOCUMENT_UUID),
            private_uuid: text(key::PRIVATE_UUID),
            version_uuid: text(key::VERSION_UUID),
            revision: text(key::REVISION),
            file_format_version: text(key::FILE_FORMAT_VERSION),
            is_multi_page: flag(key::IS_MULTI_PAGE),
            has_external_reference_or_missing_data: flag(key::HAS_EXTERNAL_REFERENCE),
            has_unmaterialized_remote_data: flag(key::HAS_UNMATERIALIZED_REMOTE_DATA),
            other: raw
                .keys()
                .into_iter()
                .filter(|k| !named.contains(k))
                .map(str::to_string)
                .collect(),
            raw,
        }))
    }
}

/// Everything [`crate::Document::metadata`] found.
#[derive(Debug, Clone, Default)]
pub struct Metadata {
    pub properties: Option<Properties>,
    /// `Metadata/DocumentIdentifier`. Equal to `documentUUID` in every document
    /// in this corpus and in all 901 bundled templates.
    pub document_identifier: Option<String>,
    /// `Metadata/BuildVersionHistory.plist`, newest last. The first line names
    /// the template or the import the document began as; the rest are build
    /// numbers, one per app version that has written the file.
    pub build_versions: Vec<String>,
    /// `TSK.DocumentArchive.locale_identifier` — e.g. `en_US`.
    pub locale: Option<String>,
    /// `TSK.DocumentArchive.creation_locale_identifier`.
    pub creation_locale: Option<String>,
    /// `TSA.DocumentArchive.document_language`.
    pub document_language: Option<String>,
    /// `TSA.DocumentArchive.template_identifier`, e.g.
    /// `Application/21_BasicWhite/Wide`.
    pub template_identifier: Option<String>,
    /// `TSA.DocumentArchive.custom_format_list` — the document-scoped list of
    /// custom cell formats, which is the `TSK.CustomFormatListArchive` (222)
    /// [`crate::Document::custom_formats`] reads.
    pub custom_format_list: Option<u64>,
    /// `TSA.DocumentArchive.tables_custom_format_list`, the second list. Absent
    /// everywhere in this corpus.
    pub tables_custom_format_list: Option<u64>,
    /// `TSK.DocumentArchive.annotation_author_storage` — the object
    /// [`crate::annotations`] starts from.
    pub annotation_author_storage: Option<u64>,
    /// Whether the package is password-protected. See [`Encryption`].
    pub encryption: Option<Encryption>,
}

/// What a password-protected package looks like from the outside.
///
/// **Measured.** `set password "p4ssw0rd" hint "the probe"` is in all three
/// scripting dictionaries — it is the only thing in this phase any of them
/// would do — so a locked document could be made and looked at, three times
/// over, one per app:
///
/// * a **`.iwpv2`** entry appears at the package root, 104 bytes in all four
///   samples, beginning `02 00 01 00 A0 86 01 00` — a version, a format and
///   then 100000, an iteration count — followed by 96 bytes that differ in
///   every document and in every re-lock of the same one;
/// * a **`.iwph`** entry holds the password hint as raw UTF-8, nothing around
///   it, and is absent when no hint was given;
/// * every `Index/*.iwa` and every `Data/*` entry is ciphertext, and so is
///   `Metadata/BuildVersionHistory.plist`;
/// * `Metadata/Properties.plist` and `Metadata/DocumentIdentifier` stay in
///   plain text and keep their usual shape;
/// * `preview.jpg`, `preview-web.jpg` and `preview-micro.jpg` are **gone**.
///
/// So detection is `.iwpv2`, and it is exact: no unencrypted document in the
/// corpus or in any of the 901 bundled templates has one, and every locked one
/// does. This crate reads the hint and refuses the document by name; it does
/// not decrypt, and it will not write an encrypted package.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Encryption {
    /// `.iwph`, the password hint, when the document has one.
    pub hint: Option<String>,
    /// Length of `.iwpv2`. 104 in every sample.
    pub header_length: usize,
    /// The three little-endian numbers at the front of `.iwpv2` — two 16-bit
    /// then one 32-bit — which are `[2, 1, 100000]` in every sample. After
    /// them, 96 bytes that differ every time. They are not named here, because
    /// nothing available established what they are beyond their being constant
    /// across three apps and four documents; 100000 is the shape of a key
    /// derivation's iteration count and that is as far as the evidence goes.
    pub header_words: Vec<u32>,
}

/// Is this package password-protected, and what does it say about itself?
pub fn encryption(package: &crate::Package) -> Option<Encryption> {
    let header = package.get(".iwpv2")?;
    Some(Encryption {
        hint: package
            .get(".iwph")
            .map(|raw| String::from_utf8_lossy(raw).into_owned()),
        header_length: header.len(),
        header_words: match header {
            [a, b, c, d, e, f, g, h, ..] => vec![
                u32::from(u16::from_le_bytes([*a, *b])),
                u32::from(u16::from_le_bytes([*c, *d])),
                u32::from_le_bytes([*e, *f, *g, *h]),
            ],
            _ => Vec::new(),
        },
    })
}

/// Read a document's metadata: the package's plists and the document-level
/// fields of its root archive.
pub fn metadata(document: &crate::Document) -> Result<Metadata, Error> {
    let package = document.package();
    let mut out = Metadata {
        properties: Properties::read(package)?,
        document_identifier: package
            .get(DOCUMENT_IDENTIFIER)
            .map(|raw| String::from_utf8_lossy(raw).trim().to_string()),
        build_versions: package
            .get(BUILD_VERSION_HISTORY)
            .and_then(|raw| crate::plist::parse(raw).ok())
            .as_ref()
            .and_then(Plist::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(|item| item.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default(),
        encryption: encryption(package),
        ..Metadata::default()
    };

    let Some(field) = super_field(document.kind()) else {
        return Ok(out);
    };
    let Some((_, root)) = root_archive(document) else {
        return Ok(out);
    };

    let Some(tsa) = nested(&root, field) else {
        return Ok(out);
    };
    out.document_language = string(&tsa, tsa_field::DOCUMENT_LANGUAGE);
    out.template_identifier = string(&tsa, tsa_field::TEMPLATE_IDENTIFIER);
    out.custom_format_list = reference(&tsa, tsa_field::CUSTOM_FORMAT_LIST);
    out.tables_custom_format_list = reference(&tsa, tsa_field::TABLES_CUSTOM_FORMAT_LIST);

    if let Some(tsk) = nested(&tsa, tsa_field::SUPER) {
        out.locale = string(&tsk, tsk_field::LOCALE_IDENTIFIER);
        out.creation_locale = string(&tsk, tsk_field::CREATION_LOCALE_IDENTIFIER);
        out.annotation_author_storage = reference(&tsk, tsk_field::ANNOTATION_AUTHOR_STORAGE);
    }
    Ok(out)
}

/// The app's root document archive: its identifier and its decoded message.
///
/// Type 10000 in Pages, 1 in Numbers and Keynote. Everything a document says
/// about *itself* rather than about its contents hangs off this one object.
pub fn root_archive(document: &crate::Document) -> Option<(u64, Message)> {
    document
        .objects()
        .find(|(_, object)| {
            let message_type = object.message_type();
            match document.kind() {
                Kind::Pages => message_type == crate::pages::TYPE_DOCUMENT,
                _ => message_type == 1,
            }
        })
        .and_then(|(_, object)| {
            Message::decode(object.payload())
                .ok()
                .map(|message| (object.identifier, message))
        })
}

/// Write `TSA.DocumentArchive.template_identifier` — which template a document
/// was made from.
///
/// The apps set it and this crate otherwise never touches it: `pages-lists`
/// says `Application/04_Real_Estate_Flyer/ISO`, `numbers-categories` says
/// `Application/21_BasicCategories/Traditional`, and both are the path of the
/// bundle inside the app, without its extension. A template bundle itself has
/// no `template_identifier` at all, so a document stamped out of one without
/// this would be the only document in the corpus that does not know where it
/// came from.
pub fn set_template_identifier(
    document: &mut crate::Document,
    identifier: &str,
) -> Result<(), Error> {
    let Some(field) = super_field(document.kind()) else {
        return Err(Error::Format(
            "the document's app is unknown, so there is no TSA.DocumentArchive to write to".into(),
        ));
    };
    let Some((root_id, mut root)) = root_archive(document) else {
        return Err(Error::Format("no root document archive".into()));
    };
    let Some(mut tsa) = nested(&root, field) else {
        return Err(Error::Format(format!(
            "the root archive has no TSA.DocumentArchive at field {field}"
        )));
    };
    tsa.set_in_order(
        tsa_field::TEMPLATE_IDENTIFIER,
        crate::pb::Value::Bytes(identifier.as_bytes().to_vec()),
    );
    root.set_in_order(field, crate::pb::Value::Bytes(tsa.encode()));
    document.set_archive_of(root_id, &root)
}

fn nested(message: &Message, field: u32) -> Option<Message> {
    crate::pb::decode_nested(message.bytes(field)?)
}

/// A string field.
///
/// **Not `decode_nested` first.** A five-byte string like `en_US` re-encodes as
/// a valid one-field message — `e` is the tag byte of a fixed32 at field 12 —
/// so a reader that tries the submessage interpretation before the string one
/// reports the document's locale as a float. `iwork dump` still does exactly
/// that, which is how this was noticed.
fn string(message: &Message, field: u32) -> Option<String> {
    crate::style::string_at(message, &[field])
}

fn reference(message: &Message, field: u32) -> Option<u64> {
    nested(message, field).and_then(|m| crate::style::reference_target(&m))
}

// -- identity ----------------------------------------------------------------

/// The identity a [`crate::Document::save_as_new`] gave a copy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NewIdentity {
    pub document_uuid: String,
    pub share_uuid: String,
    pub private_uuid: String,
    pub version_uuid: String,
    pub revision: String,
    /// Unchanged from the original for a copy, and the point: this is what says
    /// the two files are copies of one another. For a document made from a
    /// template it is the *new* `documentUUID`, because that is what the app
    /// writes — see [`Lineage`].
    pub stable_document_uuid: Option<String>,
}

/// What a new identity does with `stableDocumentUUID`, which is the whole
/// difference between the two things this crate can make out of an existing
/// package.
///
/// Measured, on three Pages fixtures the app built from templates it ships.
/// `00C_Textbook_Portrait/ISO` has `stableDocumentUUID` `A0C50246-…` and
/// `documentUUID` `65D82B29-…` — a template with a lineage of its own — and
/// `pages-toc.pages`, which Pages created from exactly that template, has
/// `stableDocumentUUID` **equal to its own new `documentUUID`**. So a document
/// made from a template is not a copy of the template: the app gives it a
/// lineage that starts with itself, and the template's is not in the file at
/// all. `pages-lists` and `numbers-categories` agree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lineage {
    /// Keep `stableDocumentUUID`: this file is a copy of the one it came from.
    /// What Save As does.
    Kept,
    /// Start a new one, equal to the new `documentUUID`: this file is a new
    /// document that happens to have been stamped out of another. What making a
    /// document from a template does.
    Fresh,
}

/// Rewrite the identity of a package in place: `Metadata/Properties.plist` and
/// `Metadata/DocumentIdentifier`, and nothing else.
///
/// Both are *stored* ZIP entries beside `Index/`, so replacing them leaves
/// every object stream byte for byte as it was — which matters, because the
/// object graph of a copy is the object graph of the original.
pub fn assign_new_identity(package: &mut crate::Package) -> Result<NewIdentity, Error> {
    assign_identity(package, Lineage::Kept)
}

/// The same rewrite, with `stableDocumentUUID` chosen rather than kept.
///
/// [`Lineage::Fresh`] is what a document made from a template gets, and the
/// only caller of it is [`crate::Document::from_template`].
pub fn assign_identity(
    package: &mut crate::Package,
    lineage: Lineage,
) -> Result<NewIdentity, Error> {
    let Some(mut properties) = Properties::read(package)? else {
        return Err(Error::Format(format!(
            "no {PROPERTIES}: this package has no identity to replace"
        )));
    };
    let document_uuid = uuid();
    let version_uuid = uuid();
    let identity = NewIdentity {
        share_uuid: document_uuid.clone(),
        private_uuid: uuid(),
        // The generation the original carried, kept: every document in this
        // corpus is at 0, and nothing here has watched Pages raise it.
        revision: format!("{}::{version_uuid}", generation(&properties)),
        stable_document_uuid: match lineage {
            Lineage::Kept => properties.stable_document_uuid.clone(),
            Lineage::Fresh => Some(document_uuid.clone()),
        },
        document_uuid: document_uuid.clone(),
        version_uuid: version_uuid.clone(),
    };

    let raw = &mut properties.raw;
    if !matches!(raw, Plist::Dictionary(_)) {
        return Err(Error::Format(format!(
            "{PROPERTIES} is not a dictionary and this crate will not guess at it"
        )));
    }
    raw.set(key::DOCUMENT_UUID, Plist::String(document_uuid.clone()));
    raw.set(key::SHARE_UUID, Plist::String(identity.share_uuid.clone()));
    raw.set(
        key::PRIVATE_UUID,
        Plist::String(identity.private_uuid.clone()),
    );
    raw.set(key::VERSION_UUID, Plist::String(version_uuid));
    raw.set(key::REVISION, Plist::String(identity.revision.clone()));
    match lineage {
        // `stableDocumentUUID` is deliberately not touched. A document that has
        // none — no such document exists here — gets none.
        Lineage::Kept => {}
        Lineage::Fresh => raw.set(
            key::STABLE_DOCUMENT_UUID,
            Plist::String(document_uuid.clone()),
        ),
    }

    package.set(PROPERTIES, crate::plist::write(raw));
    package.set(DOCUMENT_IDENTIFIER, document_uuid.into_bytes());
    Ok(identity)
}

/// Give a package that has no identity its first one.
///
/// [`assign_identity`] rewrites an identity a document already has; this writes
/// the one a document made from nothing was never given. Both end in the same
/// place — `documentUUID` == `shareUUID` == `stableDocumentUUID`, a `revision`
/// ending in the `versionUUID` — which is the state [`crate::Document::problems`]
/// checks for and what every document in the corpus that has never been copied
/// looks like.
///
/// `isMultiPage` is the one value that depends on the app, and it is the Finder's
/// question rather than the format's: a Pages document that fits on one page has
/// it false and one that does not has it true, over all fourteen Pages fixtures;
/// every Numbers and Keynote document in the corpus has it true, one sheet or
/// twelve. A new document is one page of Pages and is born false.
pub fn write_identity(package: &mut crate::Package, kind: Kind) -> NewIdentity {
    let document_uuid = uuid();
    let version_uuid = uuid();
    let identity = NewIdentity {
        document_uuid: document_uuid.clone(),
        share_uuid: document_uuid.clone(),
        stable_document_uuid: Some(document_uuid.clone()),
        private_uuid: uuid(),
        revision: format!("0::{version_uuid}"),
        version_uuid: version_uuid.clone(),
    };

    let mut raw = Plist::Dictionary(Vec::new());
    raw.set(key::DOCUMENT_UUID, Plist::String(document_uuid.clone()));
    raw.set(key::SHARE_UUID, Plist::String(document_uuid.clone()));
    raw.set(
        key::STABLE_DOCUMENT_UUID,
        Plist::String(document_uuid.clone()),
    );
    raw.set(
        key::PRIVATE_UUID,
        Plist::String(identity.private_uuid.clone()),
    );
    raw.set(key::VERSION_UUID, Plist::String(version_uuid));
    raw.set(key::REVISION, Plist::String(identity.revision.clone()));
    raw.set(
        key::FILE_FORMAT_VERSION,
        Plist::String(FILE_FORMAT_VERSION.to_string()),
    );
    raw.set(key::IS_MULTI_PAGE, Plist::Bool(kind != Kind::Pages));
    raw.set(key::HAS_EXTERNAL_REFERENCE, Plist::Bool(false));
    raw.set(key::HAS_UNMATERIALIZED_REMOTE_DATA, Plist::Bool(false));

    package.set(PROPERTIES, crate::plist::write(&raw));
    package.set(DOCUMENT_IDENTIFIER, document_uuid.into_bytes());
    identity
}

/// `fileFormatVersion`, as 15.3.1 writes it.
///
/// Not the app's version and not the format's age: 26.3.1 is what every one of
/// the 26 fixtures carries, written by Pages, Numbers and Keynote 15.3.1 alike.
/// It is recorded rather than derived, and a document without the key opens
/// anyway — the reduction deleted `Properties.plist` outright and all three apps
/// still opened the result.
pub const FILE_FORMAT_VERSION: &str = "26.3.1";

/// The generation in front of a `revision`, or 0.
fn generation(properties: &Properties) -> u64 {
    properties
        .revision
        .as_deref()
        .and_then(|r| r.split("::").next())
        .and_then(|g| g.parse().ok())
        .unwrap_or(0)
}

/// A version-4 UUID in the shape Apple writes them: uppercase, hyphenated.
///
/// The bytes come from `/dev/urandom`. If that cannot be read — it always can
/// on the platforms this crate builds for — the fallback mixes the clock, the
/// process id and a counter, which is weaker but never returns the same value
/// twice in one process.
pub fn uuid() -> String {
    let mut bytes = [0u8; 16];
    let random = std::fs::File::open("/dev/urandom")
        .and_then(|mut file| std::io::Read::read_exact(&mut file, &mut bytes));
    if random.is_err() {
        fill_from_clock(&mut bytes);
    }
    bytes[6] = (bytes[6] & 0x0f) | 0x40; // version 4
    bytes[8] = (bytes[8] & 0x3f) | 0x80; // variant 1
    let hex: String = bytes.iter().map(|b| format!("{b:02X}")).collect();
    format!(
        "{}-{}-{}-{}-{}",
        &hex[0..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..32]
    )
}

fn fill_from_clock(bytes: &mut [u8; 16]) {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let mut state = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
        ^ (std::process::id() as u64) << 32
        ^ COUNTER
            .fetch_add(1, Ordering::Relaxed)
            .wrapping_mul(0x9e37_79b9_7f4a_7c15);
    for chunk in bytes.chunks_mut(8) {
        // splitmix64.
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        chunk.copy_from_slice(&z.to_le_bytes()[..chunk.len()]);
    }
}

/// A value read out of `TSK.DocumentArchive`, decoded the way iWork writes it.
pub fn is_uuid(text: &str) -> bool {
    text.len() == 36
        && text.bytes().enumerate().all(|(i, b)| match i {
            8 | 13 | 18 | 23 => b == b'-',
            _ => b.is_ascii_hexdigit(),
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_generated_uuid_has_apple_s_shape() {
        let value = uuid();
        assert!(is_uuid(&value), "{value}");
        assert_eq!(&value[14..15], "4", "version 4");
        assert!(
            matches!(&value[19..20], "8" | "9" | "A" | "B"),
            "variant, got {}",
            &value[19..20]
        );
        assert_eq!(value, value.to_uppercase(), "Apple writes them uppercase");
        assert_ne!(uuid(), uuid());
    }

    #[test]
    fn the_clock_fallback_still_produces_distinct_values() {
        let mut a = [0u8; 16];
        let mut b = [0u8; 16];
        fill_from_clock(&mut a);
        fill_from_clock(&mut b);
        assert_ne!(a, b);
    }

    /// The rule the app was measured performing: four UUIDs change, the stable
    /// one does not, and the revision follows the version.
    #[test]
    fn a_new_identity_keeps_the_lineage_and_replaces_the_rest() {
        let original = Plist::Dictionary(vec![
            (
                key::REVISION.into(),
                Plist::String("0::5BC1D942-7001-4A50-92DC-102C6C156FEE".into()),
            ),
            (
                key::DOCUMENT_UUID.into(),
                Plist::String("25578022-75EE-41AB-831C-48D4A86E704D".into()),
            ),
            (
                key::SHARE_UUID.into(),
                Plist::String("25578022-75EE-41AB-831C-48D4A86E704D".into()),
            ),
            (
                key::STABLE_DOCUMENT_UUID.into(),
                Plist::String("25578022-75EE-41AB-831C-48D4A86E704D".into()),
            ),
            (
                key::PRIVATE_UUID.into(),
                Plist::String("A9D83E5A-E164-4A28-B540-CBFA4B0AE15B".into()),
            ),
            (
                key::VERSION_UUID.into(),
                Plist::String("5BC1D942-7001-4A50-92DC-102C6C156FEE".into()),
            ),
            (key::IS_MULTI_PAGE.into(), Plist::Bool(false)),
            (
                key::FILE_FORMAT_VERSION.into(),
                Plist::String("26.3.1".into()),
            ),
            ("somethingNew".into(), Plist::Integer(7)),
        ]);
        let mut package = crate::Package::default();
        package.set(PROPERTIES, crate::plist::write(&original));
        package.set(
            DOCUMENT_IDENTIFIER,
            b"25578022-75EE-41AB-831C-48D4A86E704D".to_vec(),
        );
        package.set("Index/Document.iwa", vec![1, 2, 3]);

        let identity = assign_new_identity(&mut package).unwrap();
        let after = Properties::read(&package).unwrap().unwrap();

        assert_eq!(
            after.document_uuid.as_deref(),
            Some(identity.document_uuid.as_str())
        );
        assert_eq!(
            after.share_uuid, after.document_uuid,
            "share follows document"
        );
        assert_eq!(
            after.stable_document_uuid.as_deref(),
            Some("25578022-75EE-41AB-831C-48D4A86E704D"),
            "the lineage is what a copy keeps"
        );
        assert_ne!(
            after.private_uuid.as_deref(),
            Some("A9D83E5A-E164-4A28-B540-CBFA4B0AE15B")
        );
        assert_eq!(
            after.revision,
            Some(format!("0::{}", after.version_uuid.clone().unwrap()))
        );
        assert_eq!(
            package
                .get(DOCUMENT_IDENTIFIER)
                .map(|b| String::from_utf8_lossy(b).into_owned()),
            after.document_uuid,
            "DocumentIdentifier is documentUUID again"
        );
        // Everything else survives, including a key this crate does not name.
        assert_eq!(after.is_multi_page, Some(false));
        assert_eq!(after.file_format_version.as_deref(), Some("26.3.1"));
        assert_eq!(after.other, vec!["somethingNew".to_string()]);
        assert_eq!(
            package.get("Index/Document.iwa"),
            Some([1u8, 2, 3].as_slice())
        );
    }

    #[test]
    fn a_package_without_properties_is_refused_by_name() {
        let mut package = crate::Package::default();
        assert!(matches!(
            assign_new_identity(&mut package),
            Err(Error::Format(_))
        ));
    }

    #[test]
    fn the_super_field_differs_per_app() {
        assert_eq!(super_field(Kind::Pages), Some(15));
        assert_eq!(super_field(Kind::Numbers), Some(8));
        assert_eq!(super_field(Kind::Keynote), Some(3));
        assert_eq!(super_field(Kind::Unknown), None);
    }

    #[test]
    fn an_encrypted_package_is_recognised_by_one_entry() {
        let mut package = crate::Package::default();
        assert_eq!(encryption(&package), None);
        let mut header = vec![2u8, 0, 1, 0, 0xa0, 0x86, 0x01, 0x00];
        header.extend(std::iter::repeat(0xab).take(96));
        package.set(".iwpv2", header);
        package.set(".iwph", b"the probe".to_vec());
        let found = encryption(&package).unwrap();
        assert_eq!(found.hint.as_deref(), Some("the probe"));
        assert_eq!(found.header_length, 104);
        assert_eq!(found.header_words, vec![2, 1, 100_000]);
    }
}
