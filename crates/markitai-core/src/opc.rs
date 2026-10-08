//! Bounded ZIP part reads and OPC relationship targets for the package readers.
//! A `.numbers` file is an Apple ZIP, not OPC, and shares only the ZIP reads.

use std::io::{Cursor, Read};

/// The entry ceiling of the Office package readers.
pub(crate) const MAX_ENTRIES: usize = 16_384;

pub(crate) type Result<T> = std::result::Result<T, String>;

/// A ZIP package with one entry per name, read one bounded part at a time.
pub(crate) struct Zip<'a> {
    pub(crate) archive: zip::ZipArchive<Cursor<&'a [u8]>>,
}

impl<'a> Zip<'a> {
    /// Open `bytes` as a ZIP of at most `max_entries` entries. A repeated name
    /// is rejected: the ZIP library silently keeps one of its entries, and
    /// another reader may read the other.
    pub(crate) fn open(bytes: &'a [u8], max_entries: usize) -> Result<Self> {
        let archive = zip::ZipArchive::new(Cursor::new(bytes))
            .map_err(|e| format!("invalid ZIP package: {e}"))?;
        let records = central_records(bytes, archive.central_directory_start());
        if records > max_entries {
            return Err(format!("ZIP package has more than {max_entries} entries"));
        }
        if records != archive.len() {
            return Err("ZIP package repeats an entry name".into());
        }
        Ok(Self { archive })
    }

    /// Every entry name in archive order, after rejecting unsafe names, links
    /// and other special files, and declared contents above `max_total` bytes.
    pub(crate) fn names(&mut self, max_total: u64) -> Result<Vec<String>> {
        let mut names = Vec::with_capacity(self.archive.len());
        let mut total = 0u64;
        for index in 0..self.archive.len() {
            let file = self
                .archive
                .by_index_raw(index)
                .map_err(|e| format!("unreadable ZIP entry: {e}"))?;
            let name = std::str::from_utf8(file.name_raw())
                .map_err(|_| "ZIP entry name is not UTF-8".to_owned())?;
            if name.is_empty()
                || name.len() > 4096
                || name.starts_with('/')
                || name.contains(['\\', '\0'])
                || name.split('/').any(|part| part == "." || part == "..")
            {
                return Err("unsafe ZIP entry name".into());
            }
            if file
                .unix_mode()
                .is_some_and(|mode| !matches!(mode & 0o170000, 0 | 0o100000 | 0o040000))
            {
                return Err("ZIP package contains a link or special file".into());
            }
            total = total
                .checked_add(file.size())
                .filter(|total| *total <= max_total)
                .ok_or_else(|| "ZIP package exceeds its expanded size limit".to_owned())?;
            names.push(name.to_owned());
        }
        Ok(names)
    }

    /// Part `name`: at most `limit` bytes and exactly its declared size;
    /// `None` when the package has no such entry.
    pub(crate) fn read(&mut self, name: &str, limit: u64) -> Result<Option<Vec<u8>>> {
        let file = match self.archive.by_name(name) {
            Ok(file) => file,
            Err(zip::result::ZipError::FileNotFound) => return Ok(None),
            Err(e) => return Err(format!("part {name} cannot be read: {e}")),
        };
        let size = file.size();
        if file.is_dir() || size > limit {
            return Err(format!("part {name} exceeds its size limit"));
        }
        let mut bytes = Vec::new();
        file.take(size.saturating_add(1))
            .read_to_end(&mut bytes)
            .map_err(|e| format!("part {name} cannot be read: {e}"))?;
        if bytes.len() as u64 != size {
            return Err(format!("part {name} does not match its declared size"));
        }
        Ok(Some(bytes))
    }
}

/// The central-directory file headers from `start`. The ZIP library keeps one
/// entry per name, so more headers than entries means a repeated name.
fn central_records(bytes: &[u8], start: u64) -> usize {
    let mut at = usize::try_from(start).unwrap_or(usize::MAX);
    let mut count = 0;
    while let Some(header) = bytes
        .get(at..at.saturating_add(46))
        .filter(|header| header.starts_with(b"PK\x01\x02"))
    {
        let length =
            |offset: usize| usize::from(u16::from_le_bytes([header[offset], header[offset + 1]]));
        at += 46 + length(28) + length(30) + length(32);
        count += 1;
    }
    count
}

/// A package relationship: its type, its target as written, and whether the
/// target lies outside the package.
pub(crate) struct Relationship {
    pub kind: String,
    pub target: String,
    pub external: bool,
}

impl Relationship {
    /// `mode` is the `TargetMode` attribute. Only an absent or `Internal`
    /// mode names a part; any other value is never resolved in the package.
    pub(crate) fn new(kind: &str, target: &str, mode: Option<&str>) -> Self {
        Self {
            kind: kind.to_owned(),
            target: target.to_owned(),
            external: mode.is_some_and(|mode| mode != "Internal"),
        }
    }
}

/// The relationships part of `source`, or of the package for `""`.
pub(crate) fn relationships_part(source: &str) -> String {
    match source.rsplit_once('/') {
        _ if source.is_empty() => "_rels/.rels".into(),
        Some((parent, name)) => format!("{parent}/_rels/{name}.rels"),
        None => format!("_rels/{source}.rels"),
    }
}

/// The part an internal relationship of part `source` (`""` for the package)
/// targets, resolved as ECMA-376 Part 2 resolves a relative reference: escapes
/// of exactly two hex digits are decoded and `.`/`..` segments removed. A
/// scheme, authority, query or fragment, an escaped `/`, an empty segment, a
/// folder and a path above the package root are rejected.
pub(crate) fn resolve(source: &str, target: &str) -> Result<String> {
    if target.is_empty() || target.starts_with("//") || target.contains(['\\', '\0', ':', '?', '#'])
    {
        return Err("invalid internal relationship target".into());
    }
    let mut decoded = Vec::with_capacity(target.len());
    let mut bytes = target.bytes();
    while let Some(byte) = bytes.next() {
        if byte != b'%' {
            decoded.push(byte);
            continue;
        }
        let mut digit = || bytes.next().and_then(|byte| char::from(byte).to_digit(16));
        match (digit(), digit()) {
            // An escaped `/` would split a segment the target names as one.
            (Some(2), Some(15)) => return Err("unsafe escaped part name".into()),
            (Some(high), Some(low)) => decoded.push((high * 16 + low) as u8),
            _ => return Err("invalid escaped part name".into()),
        }
    }
    let decoded = String::from_utf8(decoded).map_err(|_| "part name is not UTF-8".to_owned())?;
    if decoded.contains(['\\', '\0', ':', '?', '#']) {
        return Err("unsafe escaped part name".into());
    }
    let (mut segments, path) = match decoded.strip_prefix('/') {
        Some(path) => (Vec::new(), path),
        None => (
            source
                .rsplit_once('/')
                .map_or(Vec::new(), |(parent, _)| parent.split('/').collect()),
            decoded.as_str(),
        ),
    };
    for segment in path.split('/') {
        match segment {
            "" => return Err("relationship target has an empty segment".into()),
            "." => {}
            ".." => {
                segments
                    .pop()
                    .ok_or_else(|| "relationship escapes package root".to_owned())?;
            }
            value => segments.push(value),
        }
    }
    // A final `.` or `..` segment names a folder.
    if segments.is_empty() || matches!(path.rsplit('/').next(), Some("." | "..")) {
        return Err("relationship has no part name".into());
    }
    Ok(segments.join("/"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn archive(parts: &[(&str, &[u8])]) -> Vec<u8> {
        let mut writer = zip::ZipWriter::new(Cursor::new(Vec::new()));
        for (name, bytes) in parts {
            writer
                .start_file(*name, zip::write::SimpleFileOptions::default())
                .unwrap();
            writer.write_all(bytes).unwrap();
        }
        writer.finish().unwrap().into_inner()
    }

    /// `bytes` with every occurrence of `from` replaced by `to` (same length).
    fn patch(mut bytes: Vec<u8>, from: &[u8], to: &[u8]) -> Vec<u8> {
        for at in 0..=bytes.len() - from.len() {
            if bytes[at..].starts_with(from) {
                bytes[at..at + to.len()].copy_from_slice(to);
            }
        }
        bytes
    }

    #[test]
    fn repeated_entry_names_are_rejected_although_the_zip_library_keeps_one() {
        let bytes = patch(
            archive(&[("a.xml", b"one"), ("b.xml", b"two")]),
            b"b.xml",
            b"a.xml",
        );
        assert_eq!(
            zip::ZipArchive::new(Cursor::new(&bytes[..])).unwrap().len(),
            1
        );
        assert!(
            Zip::open(&bytes, MAX_ENTRIES)
                .err()
                .unwrap()
                .contains("repeats")
        );
        let bytes = archive(&[("a.xml", b"one"), ("b.xml", b"two")]);
        assert!(Zip::open(&bytes, 2).is_ok());
        assert!(Zip::open(&bytes, 1).err().unwrap().contains("more than 1"));
    }

    #[test]
    fn a_part_must_hold_exactly_its_declared_size_within_its_limit() {
        let bytes = archive(&[("a.xml", b"12345")]);
        let mut zip = Zip::open(&bytes, MAX_ENTRIES).unwrap();
        assert_eq!(zip.read("a.xml", 5).unwrap().unwrap(), b"12345");
        assert!(zip.read("a.xml", 4).is_err());
        assert_eq!(zip.read("missing.xml", 5).unwrap(), None);
        // A central directory that declares 3 bytes for 5 stored ones.
        let mut bytes = bytes;
        let central = bytes.len() - 22 - (46 + 5);
        bytes[central + 24..central + 28].copy_from_slice(&3u32.to_le_bytes());
        let mut zip = Zip::open(&bytes, MAX_ENTRIES).unwrap();
        assert!(zip.read("a.xml", 5).unwrap_err().contains("declared size"));
    }

    #[test]
    fn entry_names_kinds_and_declared_totals_are_checked() {
        let bytes = archive(&[("a/b:c.xml", b"12"), ("d/", b""), ("e.xml", b"345")]);
        let mut zip = Zip::open(&bytes, MAX_ENTRIES).unwrap();
        assert_eq!(zip.names(5).unwrap(), ["a/b:c.xml", "d/", "e.xml"]);
        assert!(zip.names(4).is_err());
        for name in ["/a.xml", "a/../b.xml", "./a.xml", "a\\b.xml"] {
            let bytes = patch(
                archive(&[(&"x".repeat(name.len()), b"")]),
                &b"x".repeat(name.len()),
                name.as_bytes(),
            );
            assert!(
                Zip::open(&bytes, MAX_ENTRIES).unwrap().names(5).is_err(),
                "{name}"
            );
        }
        let mut writer = zip::ZipWriter::new(Cursor::new(Vec::new()));
        writer
            .add_symlink("link", "target", zip::write::SimpleFileOptions::default())
            .unwrap();
        let bytes = writer.finish().unwrap().into_inner();
        assert!(
            Zip::open(&bytes, MAX_ENTRIES)
                .unwrap()
                .names(5)
                .unwrap_err()
                .contains("special")
        );
    }

    #[test]
    fn relationship_targets_resolve_as_relative_references() {
        let source = "xl/worksheets/sheet1.xml";
        for (target, part) in [
            ("../drawings/drawing1.xml", "xl/drawings/drawing1.xml"),
            ("./sheet2.xml", "xl/worksheets/sheet2.xml"),
            ("/xl/media/a%20b.png", "xl/media/a b.png"),
            ("media/%E2%82%AC.png", "xl/worksheets/media/€.png"),
        ] {
            assert_eq!(resolve(source, target).unwrap(), part, "{target}");
        }
        assert_eq!(
            resolve("", "ppt/presentation.xml").unwrap(),
            "ppt/presentation.xml"
        );
        for target in [
            "",
            "//server/share.xml",
            "file:///secret",
            "a.xml#part",
            "..\\secret",
            "a%00b",
            "%+f.xml",
            "%2.xml",
            "a%2Fb.xml",
            "a%5Cb.xml",
            "../../../secret",
            "%2e%2e/%2e%2e/%2e%2e/secret",
            "a//b.xml",
            "media/",
            "media/.",
            "/",
        ] {
            assert!(resolve(source, target).is_err(), "{target}");
        }
    }

    #[test]
    fn only_an_absent_or_internal_target_mode_names_a_part() {
        for (mode, external) in [
            (None, false),
            (Some("Internal"), false),
            (Some("External"), true),
            (Some("external"), true),
        ] {
            assert_eq!(
                Relationship::new("t", "a.xml", mode).external,
                external,
                "{mode:?}"
            );
        }
        assert_eq!(relationships_part(""), "_rels/.rels");
        assert_eq!(relationships_part("a.xml"), "_rels/a.xml.rels");
        assert_eq!(
            relationships_part("xl/workbook.xml"),
            "xl/_rels/workbook.xml.rels"
        );
    }
}
