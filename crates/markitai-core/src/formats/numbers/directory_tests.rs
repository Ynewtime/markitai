use super::super::super::{extract, extract_directory};
use super::*;
use std::io::{Cursor, Read};

const FIXTURE: &[u8] = include_bytes!("fixtures/test-1.numbers");

fn expand(root: &Path, bytes: &[u8]) {
    fs::create_dir_all(root).unwrap();
    let mut zip = zip::ZipArchive::new(Cursor::new(bytes)).unwrap();
    // Creation order deliberately differs from the package's ZIP order.
    for index in (0..zip.len()).rev() {
        let mut part = zip.by_index(index).unwrap();
        let path = root.join(part.enclosed_name().unwrap());
        if part.is_dir() {
            fs::create_dir_all(path).unwrap();
        } else {
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            let mut bytes = Vec::new();
            part.read_to_end(&mut bytes).unwrap();
            fs::write(path, bytes).unwrap();
        }
    }
}

fn fixture() -> (tempfile::TempDir, PathBuf) {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("表格 ❄.NuMbErS");
    expand(&path, FIXTURE);
    (temp, path)
}

#[test]
fn both_independent_mit_packages_have_identical_zip_and_directory_results() {
    for bytes in [
        FIXTURE,
        include_bytes!("fixtures/test-formats.numbers").as_slice(),
    ] {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("表格 ❄.NuMbErS");
        expand(&path, bytes);
        fs::write(path.join(".DS_Store"), b"ignored Finder metadata").unwrap();
        fs::write(path.join("Index/.DS_Store"), b"nested Finder metadata").unwrap();
        let zip = extract(bytes).unwrap();
        let directory = extract_directory(&path).unwrap();
        assert_eq!(directory.markdown, zip.markdown);
        assert_eq!(directory.metadata, zip.metadata);
        assert_eq!(directory.warnings, zip.warnings);
        assert!(directory.assets.is_empty());
        assert!(zip.assets.is_empty());
        assert!(directory.metadata["table_count"].as_u64().unwrap() > 0);
        let (archive, _) = open(&path).unwrap();
        assert!(matches!(
            archive.package().form,
            iwork::package::Form::Directory
        ));
    }
}

#[test]
fn directory_depth_boundary_retains_tables_and_rejects_deeper_entries() {
    let (_temp, root) = fixture();
    let mut folder = root.clone();
    for _ in 0..MAX_DEPTH - 1 {
        folder.push("nested");
    }
    fs::create_dir_all(&folder).unwrap();
    fs::write(folder.join("metadata"), b"unused").unwrap();
    assert!(
        extract_directory(&root)
            .unwrap()
            .markdown
            .contains("YYY\\_4\\_2")
    );
    fs::create_dir(folder.join("deeper")).unwrap();
    fs::write(folder.join("deeper/metadata"), b"unused").unwrap();
    assert!(
        extract_directory(&root)
            .unwrap_err()
            .to_string()
            .contains("8-level")
    );
}

#[test]
fn all_files_and_directories_count_toward_the_node_boundary() {
    let (_temp, root) = fixture();
    let initial = inventory(&root).unwrap().entries.len();
    fs::create_dir(root.join("unused")).unwrap();
    for index in initial + 1..MAX_ENTRIES {
        fs::write(root.join(format!("unused/{index}")), []).unwrap();
    }
    assert_eq!(inventory(&root).unwrap().entries.len(), MAX_ENTRIES);
    assert!(
        extract_directory(&root)
            .unwrap()
            .markdown
            .contains("YYY\\_4\\_2")
    );
    fs::write(root.join(".DS_Store"), []).unwrap();
    assert!(
        extract_directory(&root)
            .unwrap_err()
            .to_string()
            .contains("4096-node")
    );
}

#[test]
fn declared_file_and_aggregate_limits_reject_before_large_reads() {
    let temp = tempfile::tempdir().unwrap();
    let oversized = temp.path().join("oversized.numbers");
    fs::create_dir(&oversized).unwrap();
    File::create(oversized.join("part"))
        .unwrap()
        .set_len(MAX_PART as u64 + 1)
        .unwrap();
    assert!(
        extract_directory(&oversized)
            .unwrap_err()
            .to_string()
            .contains("32 MiB")
    );
    let aggregate = temp.path().join("aggregate.numbers");
    fs::create_dir(&aggregate).unwrap();
    for index in 0..5 {
        File::create(aggregate.join(format!("part{index}")))
            .unwrap()
            .set_len(MAX_PART as u64)
            .unwrap();
    }
    assert!(
        extract_directory(&aggregate)
            .unwrap_err()
            .to_string()
            .contains("128 MiB")
    );
    // Finder data is ignored semantically, not a way around the size budget.
    fs::rename(oversized.join("part"), oversized.join(".DS_Store")).unwrap();
    assert!(
        extract_directory(&oversized)
            .unwrap_err()
            .to_string()
            .contains("32 MiB")
    );
}

#[test]
fn unsupported_containers_are_explicit_and_broken_iwa_is_not_empty_success() {
    for marker in ["index.xml", "index.xml.gz", "Index.zip", ".iwpv2"] {
        let temp = tempfile::tempdir().unwrap();
        fs::write(temp.path().join(marker), b"unsupported container").unwrap();
        assert!(matches!(
            extract_directory(temp.path()),
            Err(crate::Error::Unsupported(_))
        ));
    }
    let (_temp, root) = fixture();
    let name = inventory(&root)
        .unwrap()
        .entries
        .into_iter()
        .find(|entry| entry.name.ends_with(".iwa"))
        .unwrap()
        .name;
    fs::write(root.join(name), [0, 255, 255, 255]).unwrap();
    assert!(
        extract_directory(&root)
            .unwrap_err()
            .to_string()
            .contains("truncated IWA")
    );
    let temp = tempfile::tempdir().unwrap();
    assert!(extract_directory(temp.path()).is_err());
}

#[test]
fn changed_file_length_is_rejected_against_the_inventory() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("part");
    fs::write(&path, b"original").unwrap();
    let saved = inventory(temp.path()).unwrap();
    fs::write(&path, b"short").unwrap();
    assert!(
        read_entry(&saved.root, &saved.entries[0])
            .unwrap_err()
            .to_string()
            .contains("changed")
    );
}

#[cfg(unix)]
#[test]
fn internal_file_directory_and_finder_symlinks_are_rejected() {
    use std::os::unix::fs::symlink;
    for name in ["file", "directory", ".DS_Store"] {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("package.numbers");
        expand(&root, FIXTURE);
        let target = temp.path().join("outside");
        if name == "directory" {
            fs::create_dir(&target).unwrap();
        } else {
            fs::write(&target, b"outside bytes").unwrap();
        }
        symlink(&target, root.join(name)).unwrap();
        assert!(
            extract_directory(&root)
                .unwrap_err()
                .to_string()
                .contains("symbolic link")
        );
    }
}

#[cfg(unix)]
#[test]
fn non_utf8_names_are_rejected_by_the_filesystem_or_the_reader() {
    use std::os::unix::ffi::OsStringExt;
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join(std::ffi::OsString::from_vec(vec![0xff]));
    match fs::write(path, []) {
        Ok(()) => assert!(
            extract_directory(temp.path())
                .unwrap_err()
                .to_string()
                .contains("not UTF-8")
        ),
        Err(error) => {
            assert_eq!(error.raw_os_error(), Some(libc::EILSEQ));
            eprintln!(
                "Filesystem rejected the non-UTF-8 fixture with EILSEQ; reader rejection is not exercised on this filesystem."
            );
        }
    }
}

#[cfg(unix)]
#[test]
fn backslash_and_socket_entries_fail_without_opening_them() {
    use std::os::unix::net::UnixListener;
    let temp = tempfile::tempdir().unwrap();
    fs::write(temp.path().join("unsafe\\name"), []).unwrap();
    assert!(
        extract_directory(temp.path())
            .unwrap_err()
            .to_string()
            .contains("invalid package entry name")
    );
    let temp = tempfile::tempdir().unwrap();
    let _socket = UnixListener::bind(temp.path().join("socket")).unwrap();
    assert!(
        extract_directory(temp.path())
            .unwrap_err()
            .to_string()
            .contains("non-regular")
    );
}

#[cfg(unix)]
#[test]
fn fifo_is_rejected_before_open_and_same_size_replacement_is_detected() {
    use std::os::unix::ffi::OsStrExt;
    let temp = tempfile::tempdir().unwrap();
    let fifo = temp.path().join("fifo");
    let c_path = std::ffi::CString::new(fifo.as_os_str().as_bytes()).unwrap();
    // The NUL-terminated path is private to this test and remains alive during
    // the syscall. No reader or writer is needed to reject the resulting FIFO.
    assert_eq!(unsafe { libc::mkfifo(c_path.as_ptr(), 0o600) }, 0);
    assert!(
        extract_directory(temp.path())
            .unwrap_err()
            .to_string()
            .contains("non-regular")
    );
    fs::remove_file(fifo).unwrap();
    fs::write(temp.path().join("part"), b"original").unwrap();
    let saved = inventory(temp.path()).unwrap();
    fs::rename(temp.path().join("part"), temp.path().join("old")).unwrap();
    fs::write(temp.path().join("part"), b"replaced").unwrap();
    assert!(
        read_entry(&saved.root, &saved.entries[0])
            .unwrap_err()
            .to_string()
            .contains("changed")
    );
}

#[test]
fn the_inventory_lists_entries_by_name_whatever_the_directory_order() {
    let (_temp, root) = fixture();
    let names: Vec<String> = inventory(&root)
        .unwrap()
        .entries
        .into_iter()
        .map(|entry| entry.name)
        .collect();
    assert!(names.len() > 3 && names.iter().any(|name| name.contains('/')));
    assert!(names.windows(2).all(|pair| pair[0] < pair[1]), "{names:?}");
}
