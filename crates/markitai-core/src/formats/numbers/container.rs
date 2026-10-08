//! Allocation checks that run before the third-party table decoder.

#[path = "directory.rs"]
mod directory;

use super::{Result, error};
use iwork::pb::{Message, Reader, Value};
use std::collections::{HashMap, HashSet};

const MAX_ENTRIES: usize = 4096;
const MAX_PART: usize = 32 * 1024 * 1024;
const MAX_PACKAGE: usize = 128 * 1024 * 1024;
const MAX_IWA: usize = 64 * 1024 * 1024;
const MAX_OBJECTS: usize = 100_000;
pub(super) const MAX_CELLS: usize = 2_000_000;
const MAX_TABLES: usize = 1024;

#[derive(Default)]
struct Budget {
    expanded: usize,
    objects: HashSet<u64>,
    cells: usize,
    tables: usize,
    infos: usize,
    model_refs: HashSet<u64>,
    model_ids: HashSet<u64>,
    max_text: usize,
    payload_sizes: HashMap<u64, usize>,
    side_refs: Vec<u64>,
}

pub(super) fn open(bytes: &[u8]) -> Result<(iwork::Document, usize)> {
    if bytes.len() > MAX_PACKAGE {
        return Err(error("package exceeds the 128 MiB limit"));
    }
    let mut zip = crate::opc::Zip::open(bytes, MAX_ENTRIES).map_err(error)?;
    let names = zip.names(MAX_PACKAGE as u64).map_err(error)?;
    if names.iter().any(|name| name == ".iwpv2") {
        return Err(crate::Error::Unsupported(
            "Encrypted Numbers documents are not supported".into(),
        ));
    }
    let mut entries = Vec::new();
    let mut budget = Budget::default();
    for name in names.into_iter().filter(|name| !name.ends_with('/')) {
        let data = zip
            .read(&name, MAX_PART as u64)
            .map_err(error)?
            .ok_or_else(|| error("unreadable ZIP entry"))?;
        if name.ends_with(".iwa") {
            validate_iwa(&data, &mut budget)?;
        }
        entries.push((name, data));
    }
    finish(entries, iwork::package::Form::SingleFile, budget)
}

pub(super) fn open_directory(path: &std::path::Path) -> Result<(iwork::Document, usize)> {
    directory::open(path)
}

fn finish(
    entries: Vec<(String, Vec<u8>)>,
    form: iwork::package::Form,
    budget: Budget,
) -> Result<(iwork::Document, usize)> {
    if !entries.iter().any(|(name, _)| name.ends_with(".iwa"))
        && entries
            .iter()
            .any(|(name, _)| matches!(name.as_str(), "index.xml" | "index.xml.gz" | "Index.zip"))
    {
        return Err(crate::Error::Unsupported(
            "Legacy XML and Index.zip-only Numbers packages are not supported".into(),
        ));
    }
    if !budget.model_refs.is_subset(&budget.model_ids) {
        return Err(error("table references a missing model"));
    }
    validate_decoded_budget(&budget)?;
    let doc = iwork::Document::from_package(iwork::Package { entries, form })
        .map_err(|_| error("invalid or unsupported IWA document"))?;
    if doc.kind() != iwork::Kind::Numbers {
        return Err(error("package is not a modern Numbers document"));
    }
    Ok((doc, budget.infos))
}

fn validate_iwa(data: &[u8], budget: &mut Budget) -> Result<()> {
    let mut pos = 0usize;
    let mut expanded = 0usize;
    while pos < data.len() {
        let header = data
            .get(pos..pos.saturating_add(4))
            .ok_or_else(|| error("truncated IWA block header"))?;
        if header[0] != 0 {
            return Err(error("unsupported IWA block type"));
        }
        let length = u32::from_le_bytes([header[1], header[2], header[3], 0]) as usize;
        pos += 4;
        let end = pos
            .checked_add(length)
            .ok_or_else(|| error("IWA block size overflow"))?;
        let block = data
            .get(pos..end)
            .ok_or_else(|| error("truncated IWA block"))?;
        let mut reader = Reader::new(block);
        let size = reader
            .varint()
            .map_err(|_| error("invalid Snappy length"))?;
        // Snappy raw lengths are uint32, encoded in at most five bytes.
        if reader.pos > 5 || size > 65_536 {
            return Err(error("Snappy block exceeds the 64 KiB limit"));
        }
        expanded = expanded
            .checked_add(size as usize)
            .ok_or_else(|| error("IWA size overflow"))?;
        if expanded > MAX_PART || expanded > MAX_IWA.saturating_sub(budget.expanded) {
            return Err(error("expanded IWA data exceeds the size limit"));
        }
        pos = end;
    }
    budget.expanded += expanded;
    let stream = iwork::iwa::decompress(data).map_err(|_| error("invalid Snappy data"))?;
    if stream.len() != expanded {
        return Err(error("inconsistent Snappy length"));
    }
    let mut reader = Reader::new(&stream);
    while !reader.done() {
        let length = reader
            .varint()
            .map_err(|_| error("invalid IWA object header"))?;
        if length > 65_536 || budget.objects.len() >= MAX_OBJECTS {
            return Err(error("IWA object limit exceeded"));
        }
        let info = Message::decode(
            reader
                .take(length as usize)
                .map_err(|_| error("truncated IWA object"))?,
        )
        .map_err(|_| error("invalid IWA object header"))?;
        let id = info
            .varint(1)
            .ok_or_else(|| error("IWA object has no identity"))?;
        if !budget.objects.insert(id) {
            return Err(error("duplicate IWA object identity"));
        }
        let mut messages = Vec::new();
        for value in info.all(2) {
            let Value::Bytes(raw) = value else {
                return Err(error("invalid IWA message header"));
            };
            if messages.len() >= 64 {
                return Err(error("too many messages in one IWA object"));
            }
            let message = Message::decode(raw).map_err(|_| error("invalid IWA message header"))?;
            let length = message.varint(3).unwrap_or(0);
            if length > MAX_PART as u64 {
                return Err(error("IWA object payload exceeds the size limit"));
            }
            messages.push((message.varint(1).unwrap_or(0), length as usize));
        }
        for (kind, length) in messages {
            let payload = reader
                .take(length)
                .map_err(|_| error("truncated IWA object payload"))?;
            budget.payload_sizes.insert(id, length);
            if kind == 6000 {
                let info =
                    Message::decode(payload).map_err(|_| error("invalid table reference"))?;
                let model = info
                    .bytes(2)
                    .and_then(iwork::table::reference)
                    .ok_or_else(|| error("table has no model reference"))?;
                // iwork decodes each TableInfo separately. A shared model
                // would multiply its allocation without charging dimensions.
                if !budget.model_refs.insert(model) {
                    return Err(error("multiple tables reference the same model"));
                }
                budget.infos += 1;
                if budget.infos > MAX_TABLES {
                    return Err(error("table count limit exceeded"));
                }
            } else if kind == 6001 {
                budget.model_ids.insert(id);
                let model = Message::decode(payload).map_err(|_| error("invalid table model"))?;
                validate_dimensions(
                    model.varint(6).unwrap_or(0),
                    model.varint(7).unwrap_or(0),
                    budget,
                )?;
                if let Some(store) = model.bytes(4).and_then(iwork::pb::decode_nested) {
                    for field in [4, 6, 17, 21, 22] {
                        if let Some(id) = store.bytes(field).and_then(iwork::table::reference) {
                            budget.side_refs.push(id);
                        }
                    }
                }
            } else if matches!(kind, 6005 | 6201) {
                let list =
                    Message::decode(payload).map_err(|_| error("invalid table data list"))?;
                for value in list.all(3) {
                    let Value::Bytes(raw) = value else { continue };
                    let entry =
                        Message::decode(raw).map_err(|_| error("invalid table data entry"))?;
                    if let Some(text) = entry.bytes(3) {
                        text_budget(text, budget)?;
                    }
                }
            } else if kind == 2001 {
                let storage =
                    Message::decode(payload).map_err(|_| error("invalid rich text storage"))?;
                let mut total = 0usize;
                for value in storage.all(3) {
                    if let Value::Bytes(text) = value {
                        text_budget(text, budget)?;
                        total += text.len();
                    }
                }
                budget.max_text = budget.max_text.max(total);
            }
        }
    }
    Ok(())
}

fn text_budget(text: &[u8], budget: &mut Budget) -> Result<()> {
    if text.len() > super::MAX_TEXT || std::str::from_utf8(text).is_err() {
        return Err(error("invalid or oversized table text"));
    }
    budget.max_text = budget.max_text.max(text.len());
    Ok(())
}

fn validate_decoded_budget(budget: &Budget) -> Result<()> {
    // The reader materializes interned text separately in every cell. Charge
    // a conservative upper bound before calling tables(), including blanks.
    let per_cell = budget
        .max_text
        .checked_add(std::mem::size_of::<iwork::table::Cell>())
        .ok_or_else(|| error("decoded table size overflow"))?;
    if budget.max_text > super::MAX_TEXT
        || budget
            .cells
            .checked_mul(per_cell)
            .is_none_or(|n| n > 256 * 1024 * 1024)
    {
        return Err(error("decoded table text exceeds the 256 MiB budget"));
    }
    let side_bytes = budget.side_refs.iter().try_fold(0usize, |total, id| {
        total.checked_add(*budget.payload_sizes.get(id)?)
    });
    if side_bytes.is_none_or(|n| n > MAX_IWA) {
        return Err(error("missing or excessive referenced table data"));
    }
    Ok(())
}

fn validate_dimensions(rows: u64, columns: u64, budget: &mut Budget) -> Result<()> {
    if rows == 0 || columns == 0 || rows > 100_000 || columns > 1024 {
        return Err(error("table dimensions exceed the supported limit"));
    }
    let cells = rows
        .checked_mul(columns)
        .ok_or_else(|| error("table size overflow"))?;
    if cells > MAX_CELLS as u64 || cells as usize > MAX_CELLS.saturating_sub(budget.cells) {
        return Err(error("document exceeds the two-million-cell limit"));
    }
    budget.cells += cells as usize;
    budget.tables += 1;
    if budget.tables > MAX_TABLES {
        return Err(error("table count limit exceeded"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Cursor, Write};

    fn zip(entries: &[(&str, &[u8])]) -> Vec<u8> {
        let mut writer = zip::ZipWriter::new(Cursor::new(Vec::new()));
        for (name, bytes) in entries {
            writer
                .start_file(*name, zip::write::SimpleFileOptions::default())
                .unwrap();
            writer.write_all(bytes).unwrap();
        }
        writer.finish().unwrap().into_inner()
    }

    #[test]
    fn rejects_traversal_encryption_and_truncated_containers() {
        assert!(open(&zip(&[("../Index/Document.iwa", b"")])).is_err());
        assert!(matches!(
            open(&zip(&[(".iwpv2", b"private hint")])),
            Err(crate::Error::Unsupported(_))
        ));
        assert!(open(b"PK\x03\x04").is_err());
        assert!(validate_iwa(&[0, 10, 0, 0, 1], &mut Budget::default()).is_err());
    }

    #[test]
    fn snappy_claim_and_aggregate_are_checked_before_decompression() {
        // A tiny invalid block claims 2^32-1 expanded bytes.
        assert!(
            validate_iwa(
                &[0, 5, 0, 0, 255, 255, 255, 255, 15],
                &mut Budget::default()
            )
            .unwrap_err()
            .to_string()
            .contains("64 KiB")
        );
        let mut budget = Budget {
            expanded: MAX_IWA,
            ..Budget::default()
        };
        assert!(
            validate_iwa(&[0, 1, 0, 0, 1], &mut budget)
                .unwrap_err()
                .to_string()
                .contains("size limit")
        );
    }

    #[test]
    fn hostile_dimensions_are_rejected_before_table_allocation() {
        assert!(validate_dimensions(u64::MAX, 1, &mut Budget::default()).is_err());
        assert!(validate_dimensions(100_000, 1024, &mut Budget::default()).is_err());
        let mut budget = Budget::default();
        validate_dimensions(1000, 1000, &mut budget).unwrap();
        assert!(validate_dimensions(1001, 1000, &mut budget).is_err());
    }

    #[test]
    fn repeated_interned_text_is_bounded_before_cells_are_materialized() {
        let budget = Budget {
            cells: 100_000,
            max_text: super::super::MAX_TEXT,
            ..Budget::default()
        };
        assert!(
            validate_decoded_budget(&budget)
                .unwrap_err()
                .to_string()
                .contains("256 MiB")
        );
    }

    #[test]
    fn zip_expansion_limit_is_enforced() {
        let mut writer = zip::ZipWriter::new(Cursor::new(Vec::new()));
        writer
            .start_file(
                "preview.bin",
                zip::write::SimpleFileOptions::default()
                    .compression_method(zip::CompressionMethod::Deflated),
            )
            .unwrap();
        let block = [0u8; 65536];
        for _ in 0..=MAX_PART / block.len() {
            writer.write_all(&block).unwrap();
        }
        let bytes = writer.finish().unwrap().into_inner();
        assert!(
            open(&bytes)
                .err()
                .unwrap()
                .to_string()
                .contains("exceeds its size limit")
        );
    }
}
