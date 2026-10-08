//! Bounded repair of OLE compound files whose unused mini stream is malformed.
//!
//! macOS's Word 97 exporter (TextEdit, `textutil -convert doc`) declares a
//! mini stream and its MiniFAT although every stream is stored in regular
//! sectors, and writes them inconsistently: the MiniFAT chains unused mini
//! sectors to sector 0, or both lie beyond the sectors the FAT describes.
//! Strict readers reject the file; Word and other readers ignore the unused
//! structure. When no stream lives in the mini stream, a copy that declares
//! no mini stream (the header's MiniFAT and the root entry's stream emptied,
//! as a file without small streams is written) reads the same content.

const SIGNATURE: [u8; 8] = [0xD0, 0xCF, 0x11, 0xE0, 0xA1, 0xB1, 0x1A, 0xE1];
const FREE_SECTOR: u32 = 0xFFFF_FFFF;
const END_OF_CHAIN: u32 = 0xFFFF_FFFE;
const FAT_SECTOR: u32 = 0xFFFF_FFFD;
/// A FAT short by more than this many sectors is not the exporter's slip.
const MAX_ADDED_FAT_SECTORS: usize = 1;
const MAX_REGULAR_SECTOR: u32 = 0xFFFF_FFFA;
const HEADER_DIFAT_ENTRIES: usize = 109;
const DIRECTORY_ENTRY: usize = 128;
const STREAM: u8 = 2;
const ROOT: u8 = 5;

/// A repaired copy of `bytes`, when the file is a compound file this reader
/// can walk, no stream is stored in its mini stream, and it needs repair: a
/// FAT one sector short of the file gets the missing entries in a sector
/// appended to the copy, and a declared mini stream is detached. `None`
/// leaves the file as it is.
pub(super) fn repaired(bytes: &[u8]) -> Option<Vec<u8>> {
    if bytes.len() < 512 || bytes[..8] != SIGNATURE {
        return None;
    }
    let u16_at = |at: usize| {
        bytes
            .get(at..at + 2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
    };
    let u32_at = |at: usize| {
        bytes
            .get(at..at + 4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    };
    let major = u16_at(0x1A)?;
    let sector = match (major, u16_at(0x1E)?) {
        (3, 9) => 512,
        (4, 12) => 4096,
        _ => return None,
    };
    let sectors = bytes.len() / sector;
    // The header and at least one sector for the FAT.
    if sectors < 2 {
        return None;
    }
    // Sector n starts after the header, which takes one sector.
    let offset = |n: u32| -> Option<usize> {
        let at = (n <= MAX_REGULAR_SECTOR)
            .then(|| (n as usize + 1).checked_mul(sector))
            .flatten()?;
        (at + sector <= bytes.len()).then_some(at)
    };

    // The FAT's sectors, from the header's DIFAT entries and their chain.
    let fat_count = u32_at(0x2C)? as usize;
    if fat_count > sectors {
        return None;
    }
    let mut fat_sectors = Vec::with_capacity(fat_count);
    let take = |n: u32, fat_sectors: &mut Vec<u32>| {
        if fat_sectors.len() < fat_count && n <= MAX_REGULAR_SECTOR {
            fat_sectors.push(n);
        }
    };
    for index in 0..HEADER_DIFAT_ENTRIES {
        take(u32_at(0x4C + index * 4)?, &mut fat_sectors);
    }
    let mut difat = u32_at(0x44)?;
    // The chain cannot have more sectors than the file, however many the
    // header claims (a sector can point back at itself).
    let mut difat_left = (u32_at(0x48)? as usize).min(sectors);
    while fat_sectors.len() < fat_count && difat_left > 0 {
        let at = offset(difat)?;
        let per_sector = sector / 4 - 1;
        for index in 0..per_sector {
            take(u32_at(at + index * 4)?, &mut fat_sectors);
        }
        difat = u32_at(at + per_sector * 4)?;
        difat_left -= 1;
    }
    if fat_sectors.len() != fat_count {
        return None;
    }
    let per_sector = sector / 4;
    let mut fat = Vec::with_capacity(fat_count * per_sector);
    for &n in &fat_sectors {
        let at = offset(n)?;
        for index in 0..per_sector {
            fat.push(u32_at(at + index * 4)?);
        }
    }
    // The macOS exporter can write one FAT sector too few, so its trailing
    // directory, MiniFAT and mini-stream sectors lie beyond the entries the
    // FAT has. Each of them ends its chain (they are single sectors or the
    // directory's last), a FAT sector among them is one, and the entries go
    // into FAT sectors appended after the file's last sector.
    let file_sectors = sectors - 1;
    let mut added = 0;
    while fat.len() + added * per_sector < file_sectors + added {
        added += 1;
    }
    if added > MAX_ADDED_FAT_SECTORS || fat_count + added > HEADER_DIFAT_ENTRIES {
        return None;
    }
    let appended_from = fat.len();
    for n in appended_from..fat_count * per_sector + added * per_sector {
        fat.push(if n < file_sectors {
            if fat_sectors.contains(&(n as u32)) {
                FAT_SECTOR
            } else {
                END_OF_CHAIN
            }
        } else if n < file_sectors + added {
            FAT_SECTOR
        } else {
            FREE_SECTOR
        });
    }
    // A directory whose tail lies beyond the old FAT continues into the next
    // such sector while the entries read so far name siblings not yet read.
    let first_directory = u32_at(0x30)?;
    let entries_per_sector = sector / DIRECTORY_ENTRY;
    let mut directory = None;
    // Each round links one more sector, so the file's sector count bounds it.
    for _ in 0..=file_sectors {
        let current = follow(&fat, first_directory, sectors)?;
        let mut highest = 0;
        for &n in &current {
            let at = offset(n)?;
            for entry in bytes[at..at + sector].as_chunks::<DIRECTORY_ENTRY>().0 {
                if entry[66] == 0 {
                    continue;
                }
                for field in [68, 72, 76] {
                    let id = u32::from_le_bytes([
                        entry[field],
                        entry[field + 1],
                        entry[field + 2],
                        entry[field + 3],
                    ]);
                    if id <= MAX_REGULAR_SECTOR {
                        highest = highest.max(id as usize);
                    }
                }
            }
        }
        let last = *current.last()? as usize;
        let next = last + 1;
        if current.len() > highest / entries_per_sector
            || last < appended_from
            || next >= file_sectors
            || fat[next] != END_OF_CHAIN
        {
            directory = Some(current);
            break;
        }
        fat[last] = next as u32;
    }
    let directory = directory?;

    // Only a file with nothing in the mini stream is repaired.
    let cutoff = u64::from(u32_at(0x38)?);
    let root = offset(*directory.first()?)?;
    if bytes[root + 66] != ROOT {
        return None;
    }
    for &n in &directory {
        let at = offset(n)?;
        for entry in bytes[at..at + sector].as_chunks::<DIRECTORY_ENTRY>().0 {
            let low = u32::from_le_bytes([entry[120], entry[121], entry[122], entry[123]]);
            let high = u32::from_le_bytes([entry[124], entry[125], entry[126], entry[127]]);
            // Version 3 files leave the high half undefined.
            let size = if major == 3 {
                u64::from(low)
            } else {
                u64::from(low) | (u64::from(high) << 32)
            };
            if entry[66] == STREAM && size > 0 && size < cutoff {
                return None;
            }
        }
    }
    let declared = u32_at(0x3C)? != END_OF_CHAIN
        || u32_at(0x40)? != 0
        || u32_at(root + 116)? != END_OF_CHAIN
        || bytes[root + 120..root + 128].iter().any(|b| *b != 0);
    if !declared && added == 0 {
        return None;
    }
    // Whole sectors only: a trailing partial sector is not part of the file.
    let mut repaired = bytes[..sectors * sector].to_vec();
    for index in 0..added {
        let entries =
            &fat[appended_from + index * per_sector..appended_from + (index + 1) * per_sector];
        repaired.extend(entries.iter().flat_map(|entry| entry.to_le_bytes()));
        let slot = 0x4C + (fat_count + index) * 4;
        repaired[slot..slot + 4].copy_from_slice(&((file_sectors + index) as u32).to_le_bytes());
    }
    repaired[0x2C..0x30].copy_from_slice(&((fat_count + added) as u32).to_le_bytes());
    repaired[0x3C..0x40].copy_from_slice(&END_OF_CHAIN.to_le_bytes());
    repaired[0x40..0x44].fill(0);
    repaired[root + 116..root + 120].copy_from_slice(&END_OF_CHAIN.to_le_bytes());
    repaired[root + 120..root + 128].fill(0);
    Some(repaired)
}

/// The chain of sectors from `start` in `fat`; `None` for a chain that
/// leaves the table or runs longer than the file has `sectors` (a cycle).
fn follow(fat: &[u32], start: u32, sectors: usize) -> Option<Vec<u32>> {
    let mut out = Vec::new();
    let mut n = start;
    while n != END_OF_CHAIN {
        if n > MAX_REGULAR_SECTOR || out.len() > sectors {
            return None;
        }
        out.push(n);
        n = *fat.get(n as usize)?;
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Cursor, Write};

    const TEXTEDIT_DOC: &[u8] = include_bytes!("fixtures/textedit-word97.doc");
    const TEXTEDIT_LONG_DOC: &[u8] = include_bytes!("fixtures/textedit-word97-long.doc");

    #[test]
    fn a_textedit_word_97_file_with_a_short_fat_reads_every_paragraph() {
        // The FAT sector, the two directory sectors and the mini stream lie
        // beyond the 128 entries of the only FAT sector.
        let error = cfb::CompoundFile::open(Cursor::new(TEXTEDIT_LONG_DOC))
            .expect_err("the exported file has a short FAT");
        assert!(error.to_string().contains("is invalid"), "{error}");
        let repaired = repaired(TEXTEDIT_LONG_DOC).unwrap();
        assert_eq!(repaired.len(), TEXTEDIT_LONG_DOC.len() + 512);
        let file = cfb::CompoundFile::open(Cursor::new(&repaired)).unwrap();
        let names: Vec<_> = file
            .read_root_storage()
            .map(|entry| entry.name().to_owned())
            .collect();
        assert_eq!(names.len(), 4, "{names:?}");
        let document = super::super::extract(TEXTEDIT_LONG_DOC, "doc").unwrap();
        for paragraph in [
            "Paragraph 1 of",
            "Paragraph 205 of",
            "closing paragraph ends the long",
        ] {
            assert!(document.markdown.contains(paragraph), "{paragraph}");
        }
        assert!(
            document
                .warnings
                .iter()
                .any(|w| w.contains("repaired copy")),
            "{:?}",
            document.warnings
        );
    }

    #[test]
    fn a_textedit_word_97_file_reads_after_its_unused_mini_stream_is_detached() {
        let error = cfb::CompoundFile::open(Cursor::new(TEXTEDIT_DOC))
            .expect_err("the exported file has the malformed MiniFAT");
        assert!(error.to_string().contains("pointed to twice"), "{error}");
        let repaired = repaired(TEXTEDIT_DOC).unwrap();
        assert_eq!(repaired.len(), TEXTEDIT_DOC.len());
        let mut original = Vec::new();
        {
            let mut file = cfb::CompoundFile::open(Cursor::new(&repaired)).unwrap();
            std::io::Read::read_to_end(
                &mut file.open_stream("/WordDocument").unwrap(),
                &mut original,
            )
            .unwrap();
        }
        assert!(original.len() >= 4096);
        let document = super::super::extract(TEXTEDIT_DOC, "doc").unwrap();
        assert!(
            document.markdown.contains("Compound repair"),
            "{}",
            document.markdown
        );
        assert!(
            document.markdown.contains("Second listed point"),
            "{}",
            document.markdown
        );
        assert!(
            document
                .warnings
                .iter()
                .any(|w| w.contains("repaired copy")),
            "{:?}",
            document.warnings
        );
    }

    #[test]
    fn a_file_with_a_live_mini_stream_or_none_is_left_alone() {
        // A small stream lives in the mini stream: its table is live data.
        let mut file = cfb::CompoundFile::create(Cursor::new(Vec::new())).unwrap();
        file.create_stream("/small")
            .unwrap()
            .write_all(b"a stream under the mini-stream cutoff")
            .unwrap();
        let small = file.into_inner().into_inner();
        assert!(repaired(&small).is_none());
        // A file with only regular streams has no MiniFAT to free.
        let mut file = cfb::CompoundFile::create(Cursor::new(Vec::new())).unwrap();
        file.create_stream("/large")
            .unwrap()
            .write_all(&vec![7u8; 8192])
            .unwrap();
        let large = file.into_inner().into_inner();
        assert!(repaired(&large).is_none());
        assert!(repaired(b"not a compound file").is_none());
        let mut truncated = TEXTEDIT_DOC.to_vec();
        truncated.truncate(1024);
        assert!(repaired(&truncated).is_none());
    }

    /// A header with the given version and every DIFAT entry free.
    fn header(major: u16, shift: u16, len: usize) -> Vec<u8> {
        let mut bytes = vec![0xFF; len];
        bytes[..8].copy_from_slice(&SIGNATURE);
        bytes[0x1A..0x1C].copy_from_slice(&major.to_le_bytes());
        bytes[0x1E..0x20].copy_from_slice(&shift.to_le_bytes());
        bytes
    }

    #[test]
    fn a_difat_chain_that_loops_on_itself_ends_within_the_file() {
        // The header claims 2^32 - 1 DIFAT sectors; the only one is free
        // entries pointing back at itself, so no round finds a FAT sector.
        let mut bytes = header(3, 9, 1024);
        bytes[0x2C..0x30].copy_from_slice(&1u32.to_le_bytes());
        bytes[0x44..0x48].copy_from_slice(&0u32.to_le_bytes());
        bytes[0x48..0x4C].copy_from_slice(&u32::MAX.to_le_bytes());
        bytes[1020..1024].copy_from_slice(&0u32.to_le_bytes());
        let (sender, receiver) = std::sync::mpsc::channel();
        std::thread::spawn(move || sender.send(repaired(&bytes)).unwrap());
        let result = receiver
            .recv_timeout(std::time::Duration::from_secs(30))
            .expect("the DIFAT walk is bounded by the file's sectors");
        assert!(result.is_none());
    }

    #[test]
    fn a_version_4_file_shorter_than_one_sector_is_left_alone() {
        let mut bytes = header(4, 12, 1024);
        bytes[0x2C..0x30].copy_from_slice(&0u32.to_le_bytes());
        assert!(repaired(&bytes).is_none());
    }
}
