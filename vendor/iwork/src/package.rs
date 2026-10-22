//! The package container shared by Pages, Numbers and Keynote, in its two
//! shapes: a ZIP archive whose entries are all *stored* (compression method 0),
//! or a **directory** holding the same entries as real files.
//!
//! iWork relies on entries being uncompressed so media can be mapped straight
//! out of the file, and the `Index/*.iwa` payloads carry their own Snappy
//! compression, so deflating them again would only cost CPU. All three apps
//! behave identically here.
//!
//! The two shapes are two content types rather than a detail of storage.
//! `mdls` on the same document written both ways answers
//! `com.apple.iwork.pages.sffpages` for the file — *single-file format* — and
//! `com.apple.iwork.pages.pages`, conforming to `com.apple.package` and
//! `public.directory`, for the directory. What is inside is identical: the same
//! entry names, the same bytes.

use std::io::{Cursor, Read, Write};
use std::path::{Component, Path};
use zip::write::SimpleFileOptions;

use crate::Error;

/// Which of the two shapes a package has on disk.
///
/// A document keeps the one it was read in — `File > Advanced > Change File
/// Type` is the user's choice, not this crate's, and the app itself is
/// perfectly happy with either.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Form {
    /// One file: the ZIP. What all three apps write unless told otherwise.
    #[default]
    SingleFile,
    /// A directory, which the Finder shows as one document because the
    /// extension is registered as a package type. Apple recommends it for
    /// documents past a few hundred megabytes, where rewriting one huge file
    /// for every save is the expensive part.
    Directory,
}

impl Form {
    /// What is at this path, if anything: a directory is a package, everything
    /// else is read as a ZIP.
    pub fn of(path: impl AsRef<Path>) -> Form {
        if path.as_ref().is_dir() {
            Form::Directory
        } else {
            Form::SingleFile
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Form::SingleFile => "single file",
            Form::Directory => "package (a directory)",
        }
    }
}

/// A raw iWork package: an ordered list of entries, and the shape it had.
///
/// Entry order is preserved on write. iWork does not appear to depend on it,
/// but keeping it makes diffs against the original file readable. A package
/// read from a directory is ordered by name, because a directory has no order
/// of its own to keep.
#[derive(Clone, Default)]
pub struct Package {
    pub entries: Vec<(String, Vec<u8>)>,
    /// The shape this package was read in, and the one [`Package::write`] will
    /// write it back in.
    pub form: Form,
}

/// How deep a package directory may nest before this reader stops descending.
///
/// Real ones are two levels — `Index/Tables/DataList-1234.iwa` is the deepest
/// entry any of the 901 bundled templates or 26 fixtures has. The limit is
/// there so a directory a hostile program built cannot walk this reader into a
/// stack overflow.
const MAX_DEPTH: usize = 8;

impl Package {
    pub fn read(path: impl AsRef<std::path::Path>) -> Result<Package, Error> {
        let path = path.as_ref();
        if path.is_dir() {
            return Package::read_directory(path);
        }
        let bytes = std::fs::read(path)?;
        Package::from_bytes(&bytes)
    }

    /// Read the package form: a directory whose files are the entries.
    ///
    /// Names are the paths below `root`, joined with `/`, which is exactly what
    /// the same document's ZIP calls them — so a package and a single file read
    /// into the same `Package` and everything above this layer is unaware of
    /// the difference.
    ///
    /// Three things are deliberately not entries. `.DS_Store` is the Finder's,
    /// not the document's, and iWork never writes one — carrying it into a ZIP
    /// on the next save would add an entry the app did not put there. Symbolic
    /// links are skipped rather than followed, because a link is a name
    /// pointing outside the package and following one turns "read this
    /// document" into "read that file". Anything that is not a regular file —
    /// a fifo, a device — is skipped for the same reason: reading it is not a
    /// bounded operation.
    pub fn read_directory(root: impl AsRef<Path>) -> Result<Package, Error> {
        let root = root.as_ref();
        let mut entries = Vec::new();
        collect(root, "", 0, &mut entries)?;
        // A directory has no order, so give it one that is the same twice.
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(Package {
            entries,
            form: Form::Directory,
        })
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Package, Error> {
        let mut zip = zip::ZipArchive::new(Cursor::new(bytes))?;
        let mut entries = Vec::with_capacity(zip.len());
        for i in 0..zip.len() {
            let mut file = zip.by_index(i)?;
            // Entry names are read as raw bytes and decoded as UTF-8, not taken
            // from `name()`.
            //
            // iWork writes them as UTF-8 but does not set the ZIP general
            // purpose flag that says so (bit 11), so a reader following the
            // spec falls back to CP437 — and a document with an umlaut in a
            // media filename comes back as mojibake. That would be a cosmetic
            // problem if the name were only a label, but `TSP.PackageMetadata`
            // refers to media by this exact name, and writing the mojibake back
            // out renames the file out from under the reference: `Schu` + U+0308
            // is stored as CC 88 and would be rewritten as E2 95 A0 C3 AA.
            // Pages then opens a document whose images have gone missing.
            let name = String::from_utf8_lossy(file.name_raw()).into_owned();
            // `size()` is what the central directory *claims*, and a ZIP a
            // stranger built can claim four gigabytes for an entry containing
            // nothing. Reserving room for the claim would let a 200-byte file
            // ask for all the memory on the machine, so the capacity is the
            // smaller of the claim and what is left of the archive; the vector
            // grows if the entry really is that big.
            let ceiling = bytes.len().min(1 << 20);
            let mut data = Vec::with_capacity((file.size() as usize).min(ceiling));
            file.read_to_end(&mut data)?;
            entries.push((name, data));
        }
        Ok(Package {
            entries,
            form: Form::SingleFile,
        })
    }

    /// Write the package back in the shape it came in.
    pub fn write(&self, path: impl AsRef<std::path::Path>) -> Result<(), Error> {
        self.write_as(path, self.form)
    }

    /// Write the package in a chosen shape, whatever shape it was read in.
    ///
    /// The entries are the same bytes either way; this is the conversion
    /// `File > Advanced > Change File Type` performs.
    pub fn write_as(&self, path: impl AsRef<std::path::Path>, form: Form) -> Result<(), Error> {
        match form {
            Form::SingleFile => self.write_zip(path),
            Form::Directory => self.write_directory(path),
        }
    }

    fn write_zip(&self, path: impl AsRef<std::path::Path>) -> Result<(), Error> {
        let file = std::fs::File::create(path)?;
        let mut zip = zip::ZipWriter::new(file);
        // Stored, never deflated — see the module comment.
        //
        // And stamped with a **fixed** time rather than the clock. The default
        // is "now", which makes the same document saved twice two different
        // files: the entry contents match byte for byte and the local headers
        // do not, because a second ticked over between the saves. That is a
        // promise this crate makes and was quietly breaking — and it broke two
        // tests that compare two identical saves, but only when the machine was
        // loaded enough to straddle a second. Nothing in iWork reads an entry's
        // time: the archives refer to entries by name, and the app re-stamps
        // the whole package when it saves. So the deterministic answer is the
        // right one, and the ZIP epoch is the obviously synthetic value.
        let options = SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Stored)
            .last_modified_time(zip::DateTime::default())
            .large_file(true);
        for (name, data) in &self.entries {
            zip.start_file(name.as_str(), options)?;
            zip.write_all(data)?;
        }
        zip.finish()?;
        Ok(())
    }

    /// Write the package form: one file per entry, under a directory.
    ///
    /// **This crate never deletes a file it did not write.** Saving over a
    /// directory that already holds an iWork package *does* sweep away streams
    /// the package no longer lists — a save that rewrote `Index/Document.iwa`
    /// and left the previous `Index/Tables/DataList-9.iwa` beside it would leave
    /// two documents' worth of streams in one directory, and the component index
    /// names only one of them. But the sweep is fenced twice, because the target
    /// path is a caller's choice and `iwork duplicate Report.pages ~/Desktop`
    /// must not empty the Desktop:
    ///
    /// 1. It runs **only** when the directory was recognisably a package
    ///    *before* this save — it had an `Index/` directory and a
    ///    `Metadata/Properties.plist` (`is_package_directory`). A directory that
    ///    was something else keeps every file it held; the document is written
    ///    alongside them.
    /// 2. Even then it removes **only** files shaped like a former package entry
    ///    (`looks_like_package_entry`) — `Index/*`, `Data/*`, `Metadata/*`, any
    ///    `*.iwa`, a `preview*.jpg`. A stray file inside a package directory is
    ///    left where it is.
    ///
    /// See FORMAT.md §1.
    fn write_directory(&self, path: impl AsRef<std::path::Path>) -> Result<(), Error> {
        let root = path.as_ref();

        // Answer this *before* writing anything: the writes below create
        // `Index/` and `Metadata/Properties.plist`, which would make every
        // target look like a package and unfence the sweep.
        let overwriting_a_package = is_package_directory(root);

        // Validate every entry name up front, so a hostile one aborts the save
        // rather than leaving half a document — and half a sweep — on disk.
        let mut targets = Vec::with_capacity(self.entries.len());
        for (name, data) in &self.entries {
            targets.push((entry_path(name)?, name.as_str(), data));
        }

        std::fs::create_dir_all(root)?;
        for (relative, _, data) in &targets {
            let target = root.join(relative);
            if let Some(parent) = target.parent() {
                std::fs::create_dir_all(parent)?;
            }
            write_regular_file(&target, data)?;
        }

        if overwriting_a_package {
            let mut present = Vec::new();
            collect_names(root, "", 0, &mut present);
            for name in present {
                let wanted = targets.iter().any(|(_, n, _)| *n == name);
                if !wanted && looks_like_package_entry(&name) {
                    if let Ok(relative) = entry_path(&name) {
                        let _ = std::fs::remove_file(root.join(relative));
                    }
                }
            }
            prune_empty(root, 0);
        }
        Ok(())
    }

    pub fn get(&self, name: &str) -> Option<&[u8]> {
        self.entries
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, d)| d.as_slice())
    }

    pub fn contains(&self, name: &str) -> bool {
        self.entries.iter().any(|(n, _)| n == name)
    }

    /// Replace an entry in place, or append it if it does not exist yet.
    pub fn set(&mut self, name: &str, data: Vec<u8>) {
        match self.entries.iter_mut().find(|(n, _)| n == name) {
            Some(entry) => entry.1 = data,
            None => self.entries.push((name.to_string(), data)),
        }
    }

    pub fn remove(&mut self, name: &str) -> Option<Vec<u8>> {
        let index = self.entries.iter().position(|(n, _)| n == name)?;
        Some(self.entries.remove(index).1)
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.entries.iter().map(|(n, _)| n.as_str())
    }

    /// Names of every object-archive stream, in package order.
    ///
    /// Numbers keeps most of its components under `Index/Tables/`, so this is
    /// not a flat listing of `Index/`.
    pub fn iwa_names(&self) -> Vec<String> {
        self.names()
            .filter(|n| n.ends_with(".iwa"))
            .map(str::to_string)
            .collect()
    }

    /// Names of media entries under `Data/`.
    pub fn data_names(&self) -> Vec<String> {
        self.names()
            .filter(|n| n.starts_with("Data/"))
            .map(str::to_string)
            .collect()
    }
}

/// The path an entry name stands for below the package root, refusing any name
/// that would leave it.
///
/// Public because it is not only the package writer that turns an entry name
/// into a path on this machine: `iwork extract` does the same with the names
/// under `Data/`.
///
/// An entry name arrives from a ZIP a stranger wrote and is used to build a
/// path on this machine, which is the whole of the zip-slip problem: a package
/// holding `../../../../Library/LaunchAgents/x.plist` would otherwise write
/// there. Names are relative, `/`-separated and contain no `..`, in every
/// document and every template here; anything else is refused by name.
pub fn entry_path(name: &str) -> Result<std::path::PathBuf, Error> {
    let refuse = |why: &str| {
        Err(Error::Format(format!(
            "package entry {name:?} {why}, and this crate will not write it to disk"
        )))
    };
    if name.is_empty() {
        return refuse("has no name");
    }
    if name.starts_with('/') {
        return refuse("is an absolute path");
    }
    let path = Path::new(name);
    for component in path.components() {
        match component {
            Component::Normal(_) => {}
            _ => return refuse("is not a plain relative path"),
        }
    }
    Ok(path.to_path_buf())
}

/// Walk a package directory, reading every regular file into an entry.
fn collect(
    dir: &Path,
    prefix: &str,
    depth: usize,
    entries: &mut Vec<(String, Vec<u8>)>,
) -> Result<(), Error> {
    if depth > MAX_DEPTH {
        return Err(Error::Format(format!(
            "package directory nests more than {MAX_DEPTH} levels deep at {prefix:?}"
        )));
    }
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if name == ".DS_Store" {
            continue;
        }
        // `file_type` does not follow links, which is the point: a link is
        // skipped rather than read through.
        let kind = entry.file_type()?;
        let path = entry.path();
        let full = format!("{prefix}{name}");
        if kind.is_dir() {
            collect(&path, &format!("{full}/"), depth + 1, entries)?;
        } else if kind.is_file() {
            entries.push((full, std::fs::read(&path)?));
        }
    }
    Ok(())
}

/// Whether a directory already holds an iWork package: the `Index/` component
/// directory and the `Metadata/Properties.plist` every document carries. Only
/// then is sweeping away files the package no longer lists safe — a target
/// directory that is *not* a package may be full of a caller's own files
/// (`iwork duplicate Report.pages ~/Desktop`), and this crate never deletes
/// those.
fn is_package_directory(root: &Path) -> bool {
    root.join("Index").is_dir() && root.join("Metadata/Properties.plist").is_file()
}

/// Whether an entry name is shaped like something a package writes, and so is a
/// stale stream safe to remove when overwriting a package this crate owns.
/// Anything else — a file a caller left in the target directory — is left where
/// it is even inside a package.
fn looks_like_package_entry(name: &str) -> bool {
    name.starts_with("Index/")
        || name.starts_with("Data/")
        || name.starts_with("Metadata/")
        || name.ends_with(".iwa")
        || (name.starts_with("preview") && name.ends_with(".jpg"))
}

/// Write a regular file, first unlinking any symlink — or other non-regular
/// file — already at the path.
///
/// `std::fs::write` opens the path following symlinks, so a link planted at an
/// entry path (`Index/Document.iwa` pointing at `~/.ssh/authorized_keys`) would
/// be written *through*, turning a save into an overwrite of a file outside the
/// package. The read side already refuses to follow links (see `collect`); this
/// is the write side of the same rule, and `iwork extract` shares it.
pub fn write_regular_file(target: &Path, data: &[u8]) -> Result<(), Error> {
    if let Ok(meta) = std::fs::symlink_metadata(target) {
        if !meta.is_file() {
            if meta.is_dir() {
                std::fs::remove_dir_all(target)?;
            } else {
                std::fs::remove_file(target)?;
            }
        }
    }
    std::fs::write(target, data)?;
    Ok(())
}

/// Every file already under a package directory, by entry name. Best effort:
/// anything that cannot be read cannot be a stale entry worth deleting either.
fn collect_names(dir: &Path, prefix: &str, depth: usize, names: &mut Vec<String>) {
    if depth > MAX_DEPTH {
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.filter_map(Result::ok) {
        let name = entry.file_name().to_string_lossy().into_owned();
        let Ok(kind) = entry.file_type() else {
            continue;
        };
        let full = format!("{prefix}{name}");
        if kind.is_dir() {
            collect_names(&entry.path(), &format!("{full}/"), depth + 1, names);
        } else if kind.is_file() && name != ".DS_Store" {
            names.push(full);
        }
    }
}

/// Remove directories a save emptied, deepest first. The root itself stays.
fn prune_empty(dir: &Path, depth: usize) {
    if depth > MAX_DEPTH {
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.filter_map(Result::ok) {
        if entry.file_type().is_ok_and(|k| k.is_dir()) {
            let path = entry.path();
            prune_empty(&path, depth + 1);
            let empty = std::fs::read_dir(&path).is_ok_and(|mut d| d.next().is_none());
            if empty {
                let _ = std::fs::remove_dir(&path);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn crc32(data: &[u8]) -> u32 {
        let mut crc = 0xFFFF_FFFFu32;
        for byte in data {
            crc ^= u32::from(*byte);
            for _ in 0..8 {
                let mask = (crc & 1).wrapping_neg();
                crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
            }
        }
        !crc
    }

    /// A one-entry stored ZIP with the given raw filename bytes and the UTF-8
    /// flag left clear — which is how iWork writes packages, and which the
    /// `zip` writer will not produce, so it has to be built by hand.
    fn zip_with_raw_name(name: &[u8], data: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        let crc = crc32(data).to_le_bytes();
        let size = (data.len() as u32).to_le_bytes();
        let name_len = (name.len() as u16).to_le_bytes();

        out.extend_from_slice(&0x0403_4b50u32.to_le_bytes());
        out.extend_from_slice(&[20, 0, 0, 0, 0, 0, 0, 0, 0, 0]); // version, flags, method, time
        out.extend_from_slice(&crc);
        out.extend_from_slice(&size);
        out.extend_from_slice(&size);
        out.extend_from_slice(&name_len);
        out.extend_from_slice(&[0, 0]); // extra length
        out.extend_from_slice(name);
        out.extend_from_slice(data);

        let central = out.len() as u32;
        out.extend_from_slice(&0x0201_4b50u32.to_le_bytes());
        out.extend_from_slice(&[20, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        out.extend_from_slice(&crc);
        out.extend_from_slice(&size);
        out.extend_from_slice(&size);
        out.extend_from_slice(&name_len);
        out.extend_from_slice(&[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]); // extra, comment, disk, attrs
        out.extend_from_slice(&0u32.to_le_bytes()); // local header offset
        out.extend_from_slice(name);

        let central_size = out.len() as u32 - central;
        out.extend_from_slice(&0x0605_4b50u32.to_le_bytes());
        out.extend_from_slice(&[0, 0, 0, 0, 1, 0, 1, 0]);
        out.extend_from_slice(&central_size.to_le_bytes());
        out.extend_from_slice(&central.to_le_bytes());
        out.extend_from_slice(&[0, 0]); // comment length
        out
    }

    fn raw_names(zip: &[u8]) -> Vec<Vec<u8>> {
        let mut archive = zip::ZipArchive::new(Cursor::new(zip)).unwrap();
        (0..archive.len())
            .map(|i| archive.by_index(i).unwrap().name_raw().to_vec())
            .collect()
    }

    /// `TSP.PackageMetadata` refers to media by exact filename, so a name must
    /// survive a read and a write byte for byte. iWork writes UTF-8 names
    /// without setting the flag that says so, and decoding those as CP437 —
    /// which is what the ZIP spec asks for — turns `ü` into two characters that
    /// re-encode to entirely different bytes, renaming the file out from under
    /// the document's own reference.
    #[test]
    fn a_utf8_name_without_the_utf8_flag_survives_a_roundtrip() {
        // "Schu" + U+0308 COMBINING DIAERESIS, exactly as macOS stores it.
        let name = b"Data/Schu\xcc\x88lerblatt Logo.png";
        let package = Package::from_bytes(&zip_with_raw_name(name, b"PNG")).unwrap();

        assert_eq!(package.names().count(), 1);
        assert_eq!(
            package.data_names(),
            vec!["Data/Schu\u{308}lerblatt Logo.png".to_string()],
            "the name decodes as the UTF-8 it is"
        );
        assert!(package.contains("Data/Schu\u{308}lerblatt Logo.png"));

        let path = std::env::temp_dir().join("iwork-package-name-test.zip");
        package.write(&path).unwrap();
        let written = std::fs::read(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        assert_eq!(
            raw_names(&written),
            vec![name.to_vec()],
            "filename bytes changed on write"
        );
        assert_eq!(
            Package::from_bytes(&written).unwrap().data_names(),
            package.data_names()
        );
    }

    #[test]
    fn ascii_names_are_unaffected() {
        let package = Package::from_bytes(&zip_with_raw_name(b"Index/Document.iwa", b"x")).unwrap();
        assert_eq!(package.iwa_names(), vec!["Index/Document.iwa".to_string()]);
    }

    fn scratch(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("iwork-package-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// The two shapes are one document. A package read from a directory has the
    /// same entry names and the same bytes as the ZIP of it, and saving it puts
    /// a directory back rather than quietly converting the user's document to
    /// the other file type.
    #[test]
    fn a_directory_is_a_package_and_stays_one() {
        let dir = scratch("form");
        let document = dir.join("Probe.pages");
        std::fs::create_dir_all(document.join("Index")).unwrap();
        std::fs::write(document.join("Index/Document.iwa"), b"stream").unwrap();
        std::fs::write(document.join("Metadata"), b"not a directory here").unwrap();

        let package = Package::read(&document).unwrap();
        assert_eq!(package.form, Form::Directory);
        assert_eq!(
            package.names().collect::<Vec<_>>(),
            vec!["Index/Document.iwa", "Metadata"]
        );
        assert_eq!(package.get("Index/Document.iwa"), Some(&b"stream"[..]));

        let out = dir.join("Copy.pages");
        package.write(&out).unwrap();
        assert!(out.is_dir(), "a package saved as a single file");
        assert_eq!(
            std::fs::read(out.join("Index/Document.iwa")).unwrap(),
            b"stream"
        );

        // And the other shape on request, which is the Change File Type
        // conversion, with the entries unchanged.
        let zipped = dir.join("Zipped.pages");
        package.write_as(&zipped, Form::SingleFile).unwrap();
        let reread = Package::read(&zipped).unwrap();
        assert_eq!(reread.form, Form::SingleFile);
        assert_eq!(reread.entries, package.entries);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A save over an existing package must not leave the streams of the
    /// document that was there before: the component index names the new ones,
    /// and a reader that met both would have two documents in one directory.
    #[test]
    fn saving_over_a_package_removes_what_is_no_longer_in_it() {
        let dir = scratch("stale");
        let document = dir.join("Probe.pages");
        std::fs::create_dir_all(document.join("Index/Tables")).unwrap();
        std::fs::create_dir_all(document.join("Metadata")).unwrap();
        std::fs::write(document.join("Index/Document.iwa"), b"one").unwrap();
        std::fs::write(document.join("Index/Tables/DataList-9.iwa"), b"two").unwrap();
        std::fs::write(document.join("Metadata/Properties.plist"), b"props").unwrap();

        let mut package = Package::read(&document).unwrap();
        package.remove("Index/Tables/DataList-9.iwa").unwrap();
        package.set("Index/Document.iwa", b"edited".to_vec());
        package.write(&document).unwrap();

        assert_eq!(
            std::fs::read(document.join("Index/Document.iwa")).unwrap(),
            b"edited"
        );
        assert!(!document.join("Index/Tables/DataList-9.iwa").exists());
        assert!(
            !document.join("Index/Tables").exists(),
            "the directory the stale stream was in should have been pruned"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The reviewer's scenario: `iwork duplicate Report.pages ~/Desktop`, where
    /// the destination is a directory full of the user's own files rather than a
    /// package. Writing the document there must place it *alongside* them and
    /// delete nothing — the earlier writer swept every file it did not itself
    /// write, which turned a duplicate into `rm -rf ~/Desktop`.
    #[test]
    fn saving_into_a_directory_of_unrelated_files_deletes_none_of_them() {
        let dir = scratch("bystander");
        std::fs::write(dir.join("resume.pdf"), b"the user's resume").unwrap();
        std::fs::create_dir_all(dir.join("Photos")).unwrap();
        std::fs::write(dir.join("Photos/holiday.jpg"), b"a photo").unwrap();

        let package = Package {
            entries: vec![
                ("Index/Document.iwa".to_string(), b"stream".to_vec()),
                ("Metadata/Properties.plist".to_string(), b"props".to_vec()),
            ],
            form: Form::Directory,
        };
        package.write(&dir).unwrap();

        assert_eq!(
            std::fs::read(dir.join("resume.pdf")).unwrap(),
            b"the user's resume",
            "an unrelated file was deleted by a package save"
        );
        assert!(dir.join("Photos/holiday.jpg").exists());
        // And the document really was written among them.
        assert_eq!(
            std::fs::read(dir.join("Index/Document.iwa")).unwrap(),
            b"stream"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A symlink planted where an entry will be written must not be followed:
    /// `std::fs::write` would open the link and overwrite the file it points at,
    /// somewhere outside the package entirely. The read side already refuses to
    /// follow links; this is the write side of the same rule.
    #[test]
    fn a_symlink_at_an_entry_path_is_not_written_through() {
        let dir = scratch("write-symlink");
        let document = dir.join("Probe.pages");
        std::fs::create_dir_all(document.join("Index")).unwrap();
        let outside = dir.join("outside-secret");
        std::fs::write(&outside, b"do not touch").unwrap();
        std::os::unix::fs::symlink(&outside, document.join("Index/Document.iwa")).unwrap();

        let package = Package {
            entries: vec![
                ("Index/Document.iwa".to_string(), b"stream".to_vec()),
                ("Metadata/Properties.plist".to_string(), b"props".to_vec()),
            ],
            form: Form::Directory,
        };
        package.write(&document).unwrap();

        assert_eq!(
            std::fs::read(&outside).unwrap(),
            b"do not touch",
            "the symlink was followed and the file it points at was overwritten"
        );
        let written = document.join("Index/Document.iwa");
        assert!(
            std::fs::symlink_metadata(&written).unwrap().is_file(),
            "the entry should be a real file where the link was"
        );
        assert_eq!(std::fs::read(&written).unwrap(), b"stream");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Entry names come out of a ZIP a stranger wrote, and writing the package
    /// form turns each one into a path on this machine. `../` in a name is the
    /// oldest archive attack there is, and it is refused rather than written.
    #[test]
    fn an_entry_name_may_not_escape_the_package() {
        let dir = scratch("escape");
        for hostile in ["../escaped.txt", "/etc/hostile", "Index/../../out", ""] {
            let package = Package {
                entries: vec![(hostile.to_string(), b"x".to_vec())],
                form: Form::Directory,
            };
            let error = package.write(dir.join("Probe.pages")).unwrap_err();
            assert!(
                matches!(error, Error::Format(_)),
                "{hostile:?} was not refused: {error}"
            );
        }
        assert!(!dir.join("escaped.txt").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A symlink in a package directory is a name pointing somewhere else, and
    /// following one would let a document read a file it does not contain.
    #[test]
    fn a_symlink_is_not_an_entry() {
        let dir = scratch("symlink");
        let document = dir.join("Probe.pages");
        std::fs::create_dir_all(document.join("Index")).unwrap();
        std::fs::write(dir.join("secret"), b"not part of the document").unwrap();
        std::fs::write(document.join("Index/Document.iwa"), b"stream").unwrap();
        std::os::unix::fs::symlink(dir.join("secret"), document.join("Index/Secret.iwa")).unwrap();

        let package = Package::read(&document).unwrap();
        assert_eq!(package.iwa_names(), vec!["Index/Document.iwa".to_string()]);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
