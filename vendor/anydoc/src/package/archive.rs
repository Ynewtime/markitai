//! ZIP archive access with decompression limits.

use crate::error::ConvertError;
use crate::package::limits;
use crate::package::xml::{Element, parse_xml};
use std::collections::HashMap;
use std::io::{Cursor, Read};
use std::rc::Rc;

/// A ZIP-based document package (OOXML, ODF, EPUB).
pub struct Package<'a> {
    zip: zip::ZipArchive<Cursor<&'a [u8]>>,
    total_read: u64,
    /// Decompressed parts by normalized name: repeated references are served
    /// from the cache instead of re-decompressing and re-charging the
    /// total-bytes budget (which would falsely trip on valid documents that
    /// reference one part many times). Bounded by `MAX_TOTAL_BYTES`. Buffers
    /// are shared (`Rc`), so a cache hit never copies the bytes.
    cache: HashMap<String, Rc<[u8]>>,
}

impl<'a> Package<'a> {
    pub fn open(bytes: &'a [u8]) -> Result<Self, ConvertError> {
        let zip = zip::ZipArchive::new(Cursor::new(bytes))
            .map_err(|e| ConvertError::malformed(format!("not a readable zip archive: {e}")))?;
        if zip.len() > limits::MAX_ENTRY_COUNT {
            return Err(ConvertError::ResourceLimit {
                limit: "max_entry_count",
                detail: format!("archive contains {} entries", zip.len()),
            });
        }
        Ok(Package { zip, total_read: 0, cache: HashMap::new() })
    }

    /// markitai: an archive stored inside another one (a Word document's
    /// `altChunk`), whose reads count on from the `spent` bytes the outer
    /// archive has read, so nesting cannot multiply the total budget.
    pub fn open_spent(bytes: &'a [u8], spent: u64) -> Result<Self, ConvertError> {
        let mut package = Package::open(bytes)?;
        package.total_read = spent;
        Ok(package)
    }

    /// markitai: the decompressed bytes read so far, those of archives
    /// stored inside it included (see [`Package::charge`]).
    pub fn total_read(&self) -> u64 {
        self.total_read
    }

    /// markitai: take over the total an archive stored inside this one
    /// reached (see [`Package::open_spent`]); past the total budget it is
    /// the same resource-limit error a read gives.
    pub fn charge(&mut self, total: u64) -> Result<(), ConvertError> {
        self.total_read = self.total_read.max(total);
        if self.total_read > limits::MAX_TOTAL_BYTES {
            return Err(ConvertError::ResourceLimit {
                limit: "max_total_bytes",
                detail: "embedded parts exceed the archive's decompression budget".into(),
            });
        }
        Ok(())
    }

    /// Read a part's bytes. `Ok(None)` means the part is absent (a valid
    /// state for optional parts); `Err` means it exists but cannot be read.
    /// Callers apply the unified policy: skip + log when useful output
    /// remains, propagate when the part is the primary content.
    pub fn part(&mut self, name: &str) -> Result<Option<Rc<[u8]>>, ConvertError> {
        // OPC part URIs may carry a leading slash; entries never do.
        let name = name.trim_start_matches('/');
        if let Some(bytes) = self.cache.get(name) {
            return Ok(Some(Rc::clone(bytes)));
        }
        let mut file = match self.zip.by_name(name) {
            Ok(f) => f,
            Err(zip::result::ZipError::FileNotFound) => return Ok(None),
            Err(e) => {
                return Err(ConvertError::Malformed {
                    part: Some(name.to_string()),
                    detail: format!("unreadable archive entry: {e}"),
                });
            }
        };
        if file.size() > limits::MAX_ENTRY_BYTES {
            return Err(ConvertError::ResourceLimit {
                limit: "max_entry_bytes",
                detail: format!("{name} declares {} decompressed bytes", file.size()),
            });
        }
        // The declared size can lie; read through a hard-capped reader. The
        // cap is whichever budget has less room: the per-entry limit or what
        // remains of the whole-archive total.
        let remaining_total = limits::MAX_TOTAL_BYTES.saturating_sub(self.total_read);
        let cap = limits::MAX_ENTRY_BYTES.min(remaining_total);
        let mut bytes = Vec::new();
        let read = (&mut file).take(cap + 1).read_to_end(&mut bytes).map_err(|e| {
            ConvertError::Malformed {
                part: Some(name.to_string()),
                detail: format!("corrupt archive entry: {e}"),
            }
        })? as u64;
        if read > cap {
            return Err(if remaining_total < limits::MAX_ENTRY_BYTES {
                ConvertError::ResourceLimit {
                    limit: "max_total_bytes",
                    detail: format!("{name} exceeds the archive's remaining decompression budget"),
                }
            } else {
                ConvertError::ResourceLimit {
                    limit: "max_entry_bytes",
                    detail: format!("{name} exceeds the decompression cap"),
                }
            });
        }
        self.total_read += read;
        let bytes: Rc<[u8]> = Rc::from(bytes);
        self.cache.insert(name.to_string(), Rc::clone(&bytes));
        Ok(Some(bytes))
    }

    /// Read a part that must exist for any meaningful output.
    /// True when a part exists, without reading (or budget-charging) it.
    pub fn has_part(&self, name: &str) -> bool {
        self.zip.index_for_name(name.trim_start_matches('/')).is_some()
    }

    pub fn required_part(&mut self, name: &str) -> Result<Rc<[u8]>, ConvertError> {
        self.part(name)?.ok_or_else(|| ConvertError::MissingPart { part: name.to_string() })
    }

    /// Read an optional part under the unified recovery policy: absent is a
    /// valid state (`Ok(None)`, silent); an unreadable part is skipped with a
    /// log (`Ok(None)`); fatal resource-limit errors always propagate.
    pub fn optional_part(&mut self, name: &str) -> Result<Option<Rc<[u8]>>, ConvertError> {
        match self.part(name) {
            Ok(bytes) => Ok(bytes),
            Err(e) if e.is_fatal() => Err(e),
            Err(e) => {
                log::warn!("skipping unreadable part {name}: {e}");
                Ok(None)
            }
        }
    }

    /// Read and parse an optional XML part under the unified recovery policy:
    /// absent -> `Ok(None)`; unreadable or corrupt -> skipped with a log;
    /// fatal resource-limit errors always propagate.
    pub fn optional_xml_part(&mut self, name: &str) -> Result<Option<Element>, ConvertError> {
        let Some(bytes) = self.optional_part(name)? else {
            return Ok(None);
        };
        match parse_xml(&bytes) {
            Ok(tree) => Ok(Some(tree)),
            Err(e) if e.is_fatal() => Err(e),
            Err(e) => {
                log::warn!("skipping corrupt part {name}: {e}");
                Ok(None)
            }
        }
    }

    /// Read and parse an XML part that must exist and parse for any
    /// meaningful output.
    pub fn required_xml_part(&mut self, name: &str) -> Result<Element, ConvertError> {
        let bytes = self.required_part(name)?;
        parse_xml(&bytes)
    }
}

/// A zip-open failure on OOXML input may actually be an OLE compound file:
/// an encrypted package, or a legacy binary document with the wrong
/// extension.
pub fn probe_ole(bytes: &[u8]) -> Option<ConvertError> {
    const OLE_MAGIC: [u8; 8] = [0xD0, 0xCF, 0x11, 0xE0, 0xA1, 0xB1, 0x1A, 0xE1];
    if !bytes.starts_with(&OLE_MAGIC) {
        return None;
    }
    let cursor = Cursor::new(bytes);
    if let Ok(file) = cfb::CompoundFile::open(cursor)
        && (file.exists("EncryptionInfo") || file.exists("EncryptedPackage"))
    {
        return Some(ConvertError::Encrypted);
    }
    Some(ConvertError::malformed(
        "OLE compound document where an OOXML package was expected (legacy binary format?)",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn one_part_zip(name: &str, bytes: &[u8]) -> Vec<u8> {
        let mut w = zip::ZipWriter::new(Cursor::new(Vec::new()));
        w.start_file(name, zip::write::SimpleFileOptions::default()).unwrap();
        w.write_all(bytes).unwrap();
        w.finish().unwrap().into_inner()
    }

    #[test]
    fn an_archive_inside_another_spends_from_the_outer_budget() {
        // markitai: what the outer archive has read counts, and the inner
        // total comes back to it.
        let data = one_part_zip("a.bin", &[7u8; 4096]);
        let spent = limits::MAX_TOTAL_BYTES - 100;
        let mut inner = Package::open_spent(&data, spent).unwrap();
        let error = inner.part("a.bin").unwrap_err();
        assert!(matches!(error, ConvertError::ResourceLimit { limit: "max_total_bytes", .. }));
        let mut inner = Package::open_spent(&data, 1000).unwrap();
        assert!(inner.part("a.bin").unwrap().is_some());
        assert_eq!(inner.total_read(), 5096);
        let mut outer = Package::open(&data).unwrap();
        outer.charge(inner.total_read()).unwrap();
        assert_eq!(outer.total_read(), 5096);
        let error = outer.charge(limits::MAX_TOTAL_BYTES + 1).unwrap_err();
        assert!(matches!(error, ConvertError::ResourceLimit { limit: "max_total_bytes", .. }));
    }

    #[test]
    fn repeated_reads_are_cached_and_charged_once() {
        let data = one_part_zip("media/a.bin", &[7u8; 4096]);
        let mut pkg = Package::open(&data).unwrap();
        for _ in 0..5 {
            assert_eq!(pkg.part("media/a.bin").unwrap().unwrap().len(), 4096);
        }
        assert_eq!(pkg.total_read, 4096, "repeated reads must not re-charge the budget");
    }

    #[test]
    fn total_budget_exhaustion_reports_max_total_bytes() {
        let mut w = zip::ZipWriter::new(Cursor::new(Vec::new()));
        for name in ["a.bin", "b.bin"] {
            w.start_file(name, zip::write::SimpleFileOptions::default()).unwrap();
            w.write_all(&[7u8; 4096]).unwrap();
        }
        let data = w.finish().unwrap().into_inner();
        let mut pkg = Package::open(&data).unwrap();
        assert!(pkg.part("a.bin").unwrap().is_some());
        // Simulate a large archive having consumed almost the whole total
        // budget across earlier entries; the next entry no longer fits.
        pkg.total_read = limits::MAX_TOTAL_BYTES - 100;
        let err = pkg.part("b.bin").unwrap_err();
        assert!(
            matches!(err, ConvertError::ResourceLimit { limit: "max_total_bytes", .. }),
            "expected max_total_bytes, got: {err}"
        );
    }

    #[test]
    fn leading_slash_part_names_normalize() {
        let data = one_part_zip("word/document.xml", b"<x/>");
        let mut pkg = Package::open(&data).unwrap();
        assert!(pkg.part("/word/document.xml").unwrap().is_some());
    }
}
