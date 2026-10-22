//! A hostile file must fail, and only fail.
//!
//! Every other test in this repository asks whether a *good* document is read
//! correctly. This one asks what happens to a bad one, and the bar is much
//! lower and much more absolute: whatever the bytes are, the crate returns an
//! `Err`, or a value, and never panics, never reads outside a buffer and never
//! tries to allocate a number it read out of the file. A parser for other
//! people's documents is the place where a corrupt integer becomes a crash.
//!
//! ## The harness
//!
//! It is a dumb mutation fuzzer, deliberately: seeds come from the corpus, a
//! deterministic generator mutates one, and the result goes through the whole
//! stack under [`std::panic::catch_unwind`]. A panic is a test failure, with
//! the seed number that produced it and the offending bytes written out to
//! `/tmp` so the case can be replayed — `IWORK_FUZZ_SEED=<n> cargo test
//! --test fuzz` reruns exactly it.
//!
//! Six levels, because the interesting decoders are not reachable from the
//! outside without getting past the ones in front of them:
//!
//! | Level | What it feeds | What it reaches |
//! |---|---|---|
//! | `container` | mutated whole files | the ZIP reader (single-file form), the encryption probe |
//! | `entry` | a package with one entry mutated | IWA framing, Snappy, the object stream |
//! | `object` | one object's payload mutated, the framing rebuilt | **every reader in the crate**, on a document that opens |
//! | `iwa` | mutated `Index/*.iwa` bytes | framing and Snappy on their own |
//! | `plist` | mutated `Metadata/*.plist` bytes | `plist.rs`, binary and XML |
//! | `protobuf` | mutated object payloads | `pb.rs` and the nested-message walk |
//!
//! The `container` level reaches only the *single-file* reader:
//! `Package::from_bytes` always yields a `SingleFile` package, and there is no
//! ZIP-shaped mutation that turns into a directory. The directory reader and
//! writer — `read_directory`, `write_directory` and its stale-stream sweep,
//! where the filesystem-safety findings live — are fuzzed on their own by
//! `the_directory_form_survives_hostile_bytes` below.
//!
//! The `object` level is where the value is, and it exists because the `entry`
//! level turned out not to be: a mutation of the compressed bytes almost always
//! dies in Snappy, which proves nothing about anything above it. So the object
//! level puts the mutated payload *back* into its object and re-frames the
//! stream, which makes a document that opens and whose object 1732608 is
//! nonsense — a plausible document with wrong numbers in it, which is what runs
//! through the table decoder, the formula ASTs, the attribute tables and the
//! chart grids. Every reader the crate exposes is called on whatever comes out,
//! and so is every **pretty-printer**: the formula and predicate text, which
//! recurse, and which a review found two ways to hang because this sweep used
//! to stop at the decoders and never print what they produced.
//!
//! ## The budget
//!
//! Bounded by time, not by iterations, so it is the same test on a slow
//! machine: `IWORK_FUZZ_SECONDS` (default 20) and `IWORK_FUZZ_ITERATIONS`
//! (default 4000), whichever runs out first. At the default this is a
//! tripwire — a few dozen cases at the object level, thousands at the cheap
//! ones — and it is meant to be raised:
//! `IWORK_FUZZ_SECONDS=1200 cargo test --release --test fuzz` is a real run.
//! It is deterministic given the seed, so a failure found in a long run
//! reproduces in a short one.
//!
//! With no fixtures in the tree the seeds are the synthetic ones below, so a
//! fresh clone still fuzzes; with the corpus present every fixture is a seed.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use iwork::package::Form;
use iwork::{iwa, pb, plist, Document, Package};

// -- the corpus --------------------------------------------------------------

fn corpus() -> Vec<PathBuf> {
    let dir = std::env::var("IWORK_FIXTURES")
        .map(PathBuf::from)
        .unwrap_or_else(|_| Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures"));
    let mut found = Vec::new();
    collect(&dir, &mut found);
    // Password-protected packages are refused before anything decodes them, by
    // shape rather than by name — they are no use as a seed for the deep
    // levels and every corpus walker in this repository skips them.
    found.retain(|path| !Package::read(path).is_ok_and(|p| p.contains(".iwpv2")));
    found.sort();
    found
}

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
        if named {
            if path.is_file() {
                found.push(path);
            }
        } else if path.is_dir() {
            collect(&path, found);
        }
    }
}

/// What a mutated buffer is fed to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Level {
    Container,
    Entry,
    Object,
    Iwa,
    Plist,
    Protobuf,
}

impl Level {
    fn as_str(self) -> &'static str {
        match self {
            Level::Container => "container",
            Level::Entry => "entry",
            Level::Object => "object",
            Level::Iwa => "iwa",
            Level::Plist => "plist",
            Level::Protobuf => "protobuf",
        }
    }
}

/// One thing to mutate: the bytes, what they are, and — for an entry — the
/// package they have to go back into before anything can read them.
struct Seed {
    name: String,
    level: Level,
    bytes: Vec<u8>,
    /// The package this buffer came out of, with the entry left in place. The
    /// mutated bytes replace `entry` before the document is opened.
    package: Option<(Package, String)>,
    /// For the `object` level: which object of that entry the buffer is the
    /// payload of. The payload goes back into the object and the stream is
    /// re-framed, so the container and the framing are always valid and the
    /// document that comes out is a *plausible* one with wrong numbers in it —
    /// which is the input the readers above the framing never otherwise see.
    object: Option<u64>,
}

/// A package built here rather than by an app, so this test does something on
/// a machine with no fixtures at all.
///
/// It is a real one as far as the framing goes — a stored ZIP, a Snappy-framed
/// object stream, a binary plist, an XML plist — and a nonsense one as far as
/// the object graph goes, which does not matter: the levels that care are fed
/// real documents when there are any.
fn synthetic() -> Package {
    let mut root = pb::Message::default();
    root.set(1, pb::Value::Varint(1));
    root.set(2, pb::Value::Bytes(b"synthetic".to_vec()));
    let mut storage = pb::Message::default();
    storage.set(1, pb::Value::Varint(0));
    storage.set(
        3,
        pb::Value::Bytes("Hello, hostile world".as_bytes().to_vec()),
    );

    let objects = vec![
        object(1, 10000, root.encode()),
        object(2, iwork::TYPE_STORAGE, storage.encode()),
    ];

    let mut properties = plist::Plist::Dictionary(Vec::new());
    properties.set(
        "documentUUID",
        plist::Plist::String("6E1E5A2C-0000-0000-0000-000000000000".into()),
    );
    properties.set("isMultiPage", plist::Plist::Bool(false));

    Package {
        entries: vec![
            ("Index/Document.iwa".to_string(), iwa::serialize(&objects)),
            (
                "Metadata/Properties.plist".to_string(),
                plist::write(&properties),
            ),
            (
                "Metadata/DocumentIdentifier".to_string(),
                b"6E1E5A2C-0000-0000-0000-000000000000".to_vec(),
            ),
            (
                "Metadata/BuildVersionHistory.plist".to_string(),
                b"<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<plist version=\"1.0\">\
                  <array><string>Template: Blank</string></array></plist>"
                    .to_vec(),
            ),
            ("Data/photo.jpeg".to_string(), vec![0xFF, 0xD8, 0xFF, 0xE0]),
        ],
        ..Default::default()
    }
}

fn object(identifier: u64, message_type: u32, payload: Vec<u8>) -> iwa::ArchiveObject {
    iwa::ArchiveObject {
        identifier,
        messages: vec![iwa::ArchiveMessage {
            message_type,
            version: vec![1, 0, 5],
            extra: Vec::new(),
            payload,
        }],
        extra: Vec::new(),
    }
}

/// Everything worth mutating, from every document there is.
fn seeds() -> Vec<Seed> {
    let mut seeds = Vec::new();
    let mut add_package = |name: &str, package: Package, raw: Option<Vec<u8>>| {
        if let Some(raw) = raw {
            seeds.push(Seed {
                name: name.to_string(),
                level: Level::Container,
                bytes: raw,
                package: None,
                object: None,
            });
        }
        for (entry, data) in &package.entries {
            // Everything is an entry-level seed; the streams and the plists are
            // also fed to their own decoder on their own, where a failure is
            // easier to read.
            seeds.push(Seed {
                name: format!("{name}!{entry}"),
                level: Level::Entry,
                bytes: data.clone(),
                package: Some((package.clone(), entry.clone())),
                object: None,
            });
            if entry.ends_with(".iwa") {
                seeds.push(Seed {
                    name: format!("{name}!{entry}"),
                    level: Level::Iwa,
                    bytes: data.clone(),
                    package: None,
                    object: None,
                });
                // And the object payloads inside it, which is what `pb.rs` and
                // every field walk in the crate actually see.
                if let Ok(objects) = iwa::parse(data) {
                    for archive in objects.iter().take(24) {
                        seeds.push(Seed {
                            name: format!("{name}!{entry}#{}", archive.identifier),
                            level: Level::Protobuf,
                            bytes: archive.payload().to_vec(),
                            package: None,
                            object: None,
                        });
                        seeds.push(Seed {
                            name: format!("{name}!{entry}#{}", archive.identifier),
                            level: Level::Object,
                            bytes: archive.payload().to_vec(),
                            package: Some((package.clone(), entry.clone())),
                            object: Some(archive.identifier),
                        });
                    }
                }
            }
            if entry.ends_with(".plist") {
                seeds.push(Seed {
                    name: format!("{name}!{entry}"),
                    level: Level::Plist,
                    bytes: data.clone(),
                    package: None,
                    object: None,
                });
            }
        }
    };

    add_package("synthetic", synthetic(), None);
    for path in corpus() {
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        let Ok(raw) = std::fs::read(&path) else {
            continue;
        };
        let Ok(package) = Package::from_bytes(&raw) else {
            continue;
        };
        add_package(&name, package, Some(raw));
    }
    seeds
}

// -- the mutator -------------------------------------------------------------

/// splitmix64, so a seed number is a whole run and a run is reproducible.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next() % n as u64) as usize
        }
    }
}

/// Numbers a hand-written decoder gets wrong: the ends of every integer width,
/// the varint continuation bit, the lengths that overflow when doubled.
const EDGES: [u8; 10] = [0x00, 0x01, 0x7f, 0x80, 0xfe, 0xff, 0x0a, 0x08, 0x40, 0xf0];

/// One mutation of `input`. Weighted towards the front of the buffer, because
/// that is where every header, every length and every offset table lives, and
/// a bit flip in the middle of a JPEG proves nothing.
fn mutate(input: &[u8], rng: &mut Rng) -> Vec<u8> {
    let mut out = input.to_vec();
    let rounds = 1 + rng.below(4);
    for _ in 0..rounds {
        if out.is_empty() {
            out.push(rng.next() as u8);
            continue;
        }
        // Two thirds of the edits land in the first 64 bytes or in the last
        // 64 — headers and trailers, which is where a plist keeps its offset
        // table and a ZIP its central directory.
        let at = match rng.below(3) {
            0 => rng.below(out.len()),
            1 => rng.below(64.min(out.len())),
            _ => out.len().saturating_sub(rng.below(64.min(out.len())) + 1),
        };
        match rng.below(8) {
            0 => out[at] ^= 1 << rng.below(8),
            1 => out[at] = EDGES[rng.below(EDGES.len())],
            2 => out[at] = rng.next() as u8,
            3 => {
                // A length or an offset, made enormous: eight 0xff bytes is
                // every hand-rolled length field's worst input.
                let end = (at + 8).min(out.len());
                for byte in &mut out[at..end] {
                    *byte = 0xff;
                }
            }
            4 => out.truncate(at),
            5 => {
                let n = rng.below(64) + 1;
                let filler = rng.next() as u8;
                out.splice(at..at, std::iter::repeat(filler).take(n));
            }
            6 => {
                // Move a chunk somewhere else: the way to make a structurally
                // plausible stream with the wrong things in it.
                let len = rng.below(64).min(out.len() - at);
                let to = rng.below(out.len() - len.max(1) + 1);
                let chunk: Vec<u8> = out[at..at + len].to_vec();
                out.splice(to..to + len, chunk);
            }
            _ => {
                let end = (at + rng.below(32) + 1).min(out.len());
                for byte in &mut out[at..end] {
                    *byte = 0;
                }
            }
        }
    }
    out
}

// -- the targets -------------------------------------------------------------

/// Read everything the crate can read out of a document, ignoring every error
/// and asserting nothing. What is under test is that it *returns*.
fn read_everything(doc: &mut Document) {
    let _ = doc.kind();
    let _ = doc.components();
    let _ = doc.data_files();
    let _ = doc.patched_objects();
    let _ = doc.last_object_identifier();
    let _ = doc.metadata();
    let _ = doc.annotations().summary();
    let _ = doc.storages();
    let _ = doc.smart_fields();
    let _ = doc.text_styles();
    let _ = doc.problems();
    let _ = doc.undeclared_references();
    let _ = doc.drawables();
    let _ = doc.charts();
    let _ = doc.structure();
    let _ = doc.sections();
    let _ = doc.header_footers();
    let _ = doc.show();
    let _ = doc.slide_layouts();
    let _ = doc.custom_formats();

    for storage in doc.text_storages() {
        let _ = doc.list_paragraphs(storage.identifier);
        let _ = doc.paragraph_ranges(storage.identifier);
        let _ = doc.storage_text(storage.identifier);
        let _ = doc.column_layouts(storage.identifier);
        let _ = doc.style_of_run(storage.identifier, 0, iwork::StyleKind::Paragraph);
        let _ = doc.style_of_run(storage.identifier, 0, iwork::StyleKind::Character);
    }
    for style in doc.text_styles() {
        let _ = doc.text_style(style.identifier);
        let _ = doc.text_style_usage(style.identifier);
    }
    let tables = doc.tables();
    for table in &tables {
        let _ = table.rows;
        for row in 0..table.rows.min(64) {
            for column in 0..table.columns.min(64) {
                let _ = table.cell(row, column);
                let _ = table.formula(row, column);
            }
        }
        let _ = table.merge_at(0, 0);
        let _ = table.audit();
        let _ = table.to_rows();
        let _ = table.formula_cells();
        let _ = table.names();
    }
    // The pretty-printer — the largest recursive surface in the crate, and the
    // one a mutation fuzzer never reached before the review found two ways to
    // hang it there. Every formula and every rule condition is printed, which
    // is what drives `formula::Ast::text` over hostile bytes.
    let index = iwork::table::names(&tables);
    for cell in iwork::table::formulas(&tables) {
        let _ = cell.text;
    }
    for (position, table) in tables.iter().enumerate() {
        if let Some(filter) = &table.filter {
            for rule in &filter.rules {
                let _ = iwork::table::predicate_text(
                    &rule.predicate,
                    &index,
                    position,
                    rule.column.unwrap_or(0),
                );
            }
        }
        for set in &table.conditional_styles {
            for rule in &set.rules {
                let _ = iwork::table::predicate_text(&rule.predicate, &index, position, 0);
            }
        }
        let _ = &table.sort_rules;
        let _ = &table.categories;
        let _ = &table.pivot;
        let _ = table.row_states.user_hidden.len();
        let _ = table.column_states.user_hidden.len();
    }
    for chart in doc.charts() {
        let _ = chart.series();
        let _ = chart.categories();
        if let Some(references) = &chart.references {
            for reference in &references.data {
                let _ = reference.to_text();
            }
        }
    }
    for slide in doc.slides() {
        let _ = slide.title;
        let _ = slide.transition;
    }
    for drawable in doc.drawables() {
        let _ = doc.object_style(drawable.identifier);
    }
    // And the write path, on a document whose bytes are nonsense: re-encoding
    // every object is what `iwork roundtrip` does, and a document that only
    // survives being read is half a guarantee.
    let _ = doc.changed_streams();
}

fn run(level: Level, bytes: &[u8], package: Option<&(Package, String)>, object: Option<u64>) {
    match level {
        Level::Container => {
            if let Ok(package) = Package::from_bytes(bytes) {
                let _ = package.iwa_names();
                let _ = package.data_names();
                if let Ok(mut doc) = Document::from_package(package) {
                    read_everything(&mut doc);
                }
            }
        }
        Level::Entry => {
            let Some((package, entry)) = package else {
                return;
            };
            let mut package = package.clone();
            package.set(entry, bytes.to_vec());
            if let Ok(mut doc) = Document::from_package(package) {
                read_everything(&mut doc);
            }
        }
        // The deep one: the framing stays correct and one object's payload is
        // nonsense, so the document opens and every reader above the framing
        // runs on it. A mutation of the compressed bytes almost always dies in
        // Snappy instead, which proves nothing about the decoders.
        Level::Object => {
            let (Some((package, entry)), Some(identifier)) = (package, object) else {
                return;
            };
            let mut package = package.clone();
            let Some(raw) = package.get(entry) else {
                return;
            };
            let Ok(mut objects) = iwa::parse(raw) else {
                return;
            };
            for archive in &mut objects {
                if archive.identifier == identifier {
                    if let Some(message) = archive.messages.first_mut() {
                        message.payload = bytes.to_vec();
                    }
                }
            }
            package.set(entry, iwa::serialize(&objects));
            if let Ok(mut doc) = Document::from_package(package) {
                read_everything(&mut doc);
            }
        }
        Level::Iwa => {
            let _ = iwa::decompress(bytes);
            if let Ok(objects) = iwa::parse(bytes) {
                let _ = iwa::serialize(&objects);
            }
        }
        Level::Plist => {
            if let Ok(value) = plist::parse(bytes) {
                let _ = plist::write(&value);
            }
        }
        Level::Protobuf => {
            let _ = pb::decode_nested(bytes);
            if let Ok(message) = pb::Message::decode(bytes) {
                let _ = message.encode();
                for field in &message.fields {
                    if let pb::Value::Bytes(raw) = &field.value {
                        let _ = pb::decode_nested(raw);
                    }
                }
            }
        }
    }
}

// -- the test ----------------------------------------------------------------

fn budget(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Thousands of hostile inputs, and not one of them may panic.
#[test]
fn hostile_bytes_fail_rather_than_panic() {
    let seeds = seeds();
    assert!(!seeds.is_empty(), "the synthetic seed is always there");
    let iterations = budget("IWORK_FUZZ_ITERATIONS", 4000);
    let seconds = budget("IWORK_FUZZ_SECONDS", 20);
    let base = budget("IWORK_FUZZ_SEED", 0x1F02_6D0C);
    let deadline = Instant::now() + Duration::from_secs(seconds);

    // The panic hook would print a page of backtrace for every case; the
    // failure is reported below with the seed, which is the part worth having.
    // Replaying one case is the exception — `IWORK_FUZZ_SEED=<n>
    // IWORK_FUZZ_ITERATIONS=1 IWORK_FUZZ_BACKTRACE=1 RUST_BACKTRACE=1` leaves
    // the hook alone, and the backtrace is the point of the replay.
    let previous = std::panic::take_hook();
    let quiet = std::env::var("IWORK_FUZZ_BACKTRACE").is_err();
    if quiet {
        std::panic::set_hook(Box::new(|_| {}));
    }

    let mut done = 0u64;
    let mut failure = None;
    for iteration in 0..iterations {
        if Instant::now() >= deadline {
            break;
        }
        let seed = base.wrapping_add(iteration);
        let mut rng = Rng(seed);
        let picked = &seeds[rng.below(seeds.len())];
        let mutated = mutate(&picked.bytes, &mut rng);
        let level = picked.level;
        let package = picked.package.clone();
        let object = picked.object;
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            run(level, &mutated, package.as_ref(), object)
        }));
        done += 1;
        if let Err(panic) = outcome {
            let message = panic
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| panic.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "a panic with no message".to_string());
            let dump = std::env::temp_dir().join(format!("iwork-fuzz-{seed:016x}.bin"));
            let _ = std::fs::write(&dump, &mutated);
            failure = Some(format!(
                "IWORK_FUZZ_SEED={seed} — {} level, seed document {}, panicked: {message}\n\
                 the input is in {}",
                level.as_str(),
                picked.name,
                dump.display()
            ));
            break;
        }
    }

    if quiet {
        std::panic::set_hook(previous);
    }
    if let Some(failure) = failure {
        panic!("{failure}");
    }
    eprintln!(
        "fuzz: {done} mutations over {} seeds, no panics",
        seeds.len()
    );
}

// -- the cases the fuzzer found ---------------------------------------------
//
// Each of these is a shape rather than a specific mutated file: the fuzzer
// found it, the fix is in the decoder, and the test states what the input
// looked like so the next person does not have to rediscover it.

/// A Snappy block declares its uncompressed size in its first bytes, and
/// nothing in the format stops it declaring four gigabytes in five bytes. The
/// decoder believed it and allocated it.
#[test]
fn a_snappy_block_may_not_claim_more_than_a_block() {
    // A raw Snappy block: uncompressed length 0xFFFFFFF as a varint, then a
    // literal that supplies nothing like that many bytes.
    let block = [0xff, 0xff, 0xff, 0x7f, 0x00, 0x41];
    let mut stream = vec![0x00];
    stream.extend_from_slice(&(block.len() as u32).to_le_bytes()[..3]);
    stream.extend_from_slice(&block);

    let error = iwa::decompress(&stream).unwrap_err();
    assert!(
        error.contains("block size"),
        "a block claiming to decompress to 256 MB should be refused by size: {error}"
    );
}

/// `MessageInfo.length` is a varint, and the payload it describes is supposed
/// to be the next `length` bytes of the stream. A stream that says 2^60 was an
/// allocation of 2^60 bytes before anything looked at what was left.
#[test]
fn a_message_length_is_not_an_allocation_request() {
    let mut info = pb::Message::default();
    info.set(1, pb::Value::Varint(1));
    let mut message_info = pb::Message::default();
    message_info.set(1, pb::Value::Varint(2001));
    message_info.set(3, pb::Value::Varint(1 << 60));
    info.set(2, pb::Value::Bytes(message_info.encode()));

    let encoded = info.encode();
    let mut stream = Vec::new();
    pb::write_varint(&mut stream, encoded.len() as u64);
    stream.extend_from_slice(&encoded);

    let error = iwa::parse(&iwa::compress(&stream)).unwrap_err();
    assert!(
        error.contains("remain in the stream"),
        "a message claiming 2^60 bytes should be refused against what is there: {error}"
    );
}

/// Lengths in a binary property list are eight-byte integers out of the file,
/// and a dictionary's is used twice — once for the keys, once for the values.
/// `length * 2` wraps, and `at + length` wraps, and both were unchecked.
#[test]
fn a_plist_length_that_overflows_is_refused() {
    // bplist00, one object, whose marker is 0xDF — a dictionary with an
    // extended length — followed by 0x13 (an eight-byte integer) and
    // 0xFFFFFFFFFFFFFFFF as the count.
    let mut bytes = b"bplist00".to_vec();
    let object_at = bytes.len();
    bytes.extend_from_slice(&[0xdf, 0x13]);
    bytes.extend_from_slice(&[0xff; 8]);
    let table_at = bytes.len();
    bytes.push(object_at as u8);
    // The trailer: sort version, offset size 1, ref size 1, one object, root 0.
    bytes.extend_from_slice(&[0, 0, 0, 0, 0, 0, 1, 1]);
    bytes.extend_from_slice(&1u64.to_be_bytes());
    bytes.extend_from_slice(&0u64.to_be_bytes());
    bytes.extend_from_slice(&(table_at as u64).to_be_bytes());

    let error = plist::parse(&bytes).unwrap_err();
    assert!(
        matches!(error, iwork::Error::Format(_)),
        "an overflowing dictionary length should be a format error: {error}"
    );
}

/// The directory form of a package never reaches the main loop: every seed is
/// read with `Package::from_bytes`, which always yields the single-file form, so
/// `read_directory`, `write_directory` and the stale-stream sweep — the code
/// carrying the filesystem-safety findings — would otherwise see no hostile
/// input at all. This drives them directly on mutated bytes: write a package to
/// a directory, read it back, and write it again over itself (which is where the
/// sweep runs, the target now being a recognisable package). Not one round trip
/// may panic.
#[test]
fn the_directory_form_survives_hostile_bytes() {
    let seeds = seeds();
    let base = budget("IWORK_FUZZ_SEED", 0x1F02_6D0C);
    let iterations = budget("IWORK_FUZZ_DIR_ITERATIONS", 200);
    let root = std::env::temp_dir().join(format!("iwork-fuzz-dir-{}", std::process::id()));

    for iteration in 0..iterations {
        let seed = base.wrapping_add(iteration);
        let mut rng = Rng(seed);
        let picked = &seeds[rng.below(seeds.len())];
        // Only the whole-package seeds carry the entries a directory needs.
        let Some((package, _)) = &picked.package else {
            continue;
        };
        let mut package = package.clone();
        if !package.entries.is_empty() {
            let which = rng.below(package.entries.len());
            package.entries[which].1 = mutate(&package.entries[which].1, &mut rng);
        }

        let document = root.join("Probe.pages");
        let _ = std::fs::remove_dir_all(&document);
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            if package.write_as(&document, Form::Directory).is_ok() {
                if let Ok(reread) = Package::read_directory(&document) {
                    let _ = reread.write_as(&document, Form::Directory);
                }
            }
        }));
        assert!(
            outcome.is_ok(),
            "IWORK_FUZZ_SEED={seed} — the directory round trip panicked"
        );
    }
    let _ = std::fs::remove_dir_all(&root);
}

/// A ZIP entry's size in the central directory is a claim, not a measurement.
/// Reserving room for the claim let a two-hundred-byte file ask for two
/// gigabytes.
#[test]
fn a_zip_entry_size_is_not_an_allocation_request() {
    let mut package = Package::default();
    package.set("Index/Document.iwa", b"x".to_vec());
    let path = std::env::temp_dir().join(format!("iwork-fuzz-size-{}.zip", std::process::id()));
    package.write(&path).unwrap();
    let mut bytes = std::fs::read(&path).unwrap();
    let _ = std::fs::remove_file(&path);

    // Both copies of the uncompressed size — local header and central
    // directory — set to 0x7FFFFFFF, which is two gigabytes of nothing.
    let mut patched = 0;
    for at in 0..bytes.len().saturating_sub(4) {
        if bytes[at..at + 4] == [1, 0, 0, 0] {
            bytes[at..at + 4].copy_from_slice(&[0xff, 0xff, 0xff, 0x7f]);
            patched += 1;
        }
    }
    assert!(patched > 0, "the sizes should be in there to patch");

    // The mitigation is in the *reservation*: capacity is capped at what is
    // actually left in the archive, not the claimed size, so a two-gigabyte
    // claim over a one-byte entry never reserves two gigabytes. Whatever the
    // reader makes of the size mismatch, it returns rather than hanging, and
    // every entry it hands back is bounded by the file it came out of — not a
    // gigabyte of reserved nothing.
    if let Ok(package) = Package::from_bytes(&bytes) {
        for (name, data) in &package.entries {
            assert!(
                data.len() <= bytes.len(),
                "entry {name} grew to {} bytes, past the {}-byte archive",
                data.len(),
                bytes.len()
            );
        }
    }
}
