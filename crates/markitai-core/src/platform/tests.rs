use super::*;
use std::io::Write;

#[test]
fn verbatim_drive_and_unc_paths_lose_the_prefix_only_when_spelled_alike() {
    for (verbatim, plain) in [
        (r"\\?\C:\", Some(r"C:\")),
        (
            r"\\?\C:\Users\runneradmin\out",
            Some(r"C:\Users\runneradmin\out"),
        ),
        (r"\\?\d:\a b\c.d.md", Some(r"d:\a b\c.d.md")),
        (r"\\?\UNC\server\share\x", Some(r"\\server\share\x")),
        (r"\\?\UNC\server", None),
        (r"\\?\C:\out\CON", None),
        (r"\\?\C:\out\con.md", None),
        (r"\\?\C:\out\Lpt1.txt", None),
        (r"\\?\C:\out\COM².log", None),
        (r"\\?\C:\out\CONOUT$", None),
        (r"\\?\C:\out\trailing.", None),
        (r"\\?\C:\out\trailing ", None),
        (r"\\?\C:\out\a:b", None),
        (r"\\?\C:\out\a|b", None),
        (r"\\?\C:\out\..\x", None),
        (r"\\?\Volume{0f0f0f0f-0000-0000-0000-000000000000}\x", None),
        (r"\\?\GLOBALROOT\Device\x", None),
        (r"C:\already\plain", None),
    ] {
        assert_eq!(plain_spelling(verbatim).as_deref(), plain, "{verbatim}");
    }
    // Names that only look like devices stay ordinary.
    for name in [
        "console",
        "COM10",
        "LPT",
        "nul0",
        "AUX_",
        ".markitai",
        "a.b.c",
    ] {
        assert!(ordinary_name(name), "{name}");
    }
}

#[test]
fn only_transient_windows_rename_errors_are_retried_within_the_bound() {
    for code in [5, 32, 33] {
        let delays: Vec<_> = (0..RENAME_ATTEMPTS)
            .map(|attempt| retry_delay(true, Some(code), attempt))
            .collect();
        assert_eq!(
            delays,
            [
                Some(RENAME_BACKOFF),
                Some(RENAME_BACKOFF * 2),
                Some(RENAME_BACKOFF * 3),
                Some(RENAME_BACKOFF * 4),
                None,
            ]
        );
        // Unix never retries: a rename there is not blocked by other opens.
        assert_eq!(retry_delay(false, Some(code), 0), None);
    }
    // ERROR_FILE_EXISTS, ERROR_ALREADY_EXISTS, ERROR_PATH_NOT_FOUND, none.
    for code in [Some(80), Some(183), Some(3), None] {
        assert_eq!(retry_delay(true, code, 0), None);
    }
}

#[test]
fn identity_is_shared_by_path_and_handle_and_survives_rename() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("file");
    std::fs::write(&path, b"bytes").unwrap();
    let file = File::open(&path).unwrap();
    let by_path = status(&path).unwrap();
    let by_handle = file_status(&file).unwrap();
    assert_eq!(by_path.id(), by_handle.id());
    assert_eq!(by_path.links(), 1);
    assert!(by_path.metadata().is_file());
    let renamed = root.path().join("renamed");
    std::fs::rename(&path, &renamed).unwrap();
    assert_eq!(status(&renamed).unwrap().id(), by_path.id());
    std::fs::write(&path, b"other").unwrap();
    assert_ne!(status(&path).unwrap().id(), by_path.id());
    // Directories have identities too, on the same volume.
    let directory = status(root.path()).unwrap();
    assert!(directory.metadata().is_dir());
    assert_ne!(directory.id(), by_path.id());
    assert_eq!(directory.id().volume, by_path.id().volume);
    assert_eq!(
        file_status(&open_directory(root.path()).unwrap())
            .unwrap()
            .id(),
        directory.id()
    );
}

#[test]
fn hard_links_are_counted() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("file");
    std::fs::write(&path, b"bytes").unwrap();
    std::fs::hard_link(&path, root.path().join("alias")).unwrap();
    assert_eq!(status(&path).unwrap().links(), 2);
    assert_eq!(
        status(&path).unwrap().id(),
        status(&root.path().join("alias")).unwrap().id()
    );
}

#[test]
fn entries_this_process_creates_privately_are_private_and_its_own() {
    let root = tempfile::tempdir().unwrap();
    let directory = root.path().join("private");
    private_directory().create(&directory).unwrap();
    let path = directory.join("file");
    private_file(OpenOptions::new().write(true).create_new(true))
        .open(&path)
        .unwrap();
    for entry in [&directory, &path] {
        let observed = status(entry).unwrap();
        assert!(observed.private(), "{entry:?}");
        assert!(observed.owned_by_current_user(), "{entry:?}");
    }
    assert!(
        status(&directory)
            .unwrap()
            .same_owner(&status(&path).unwrap())
    );
}

#[cfg(unix)]
#[test]
fn unix_privacy_is_the_mode_and_ownership_the_user_id() {
    use std::os::unix::fs::PermissionsExt;
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("file");
    std::fs::write(&path, b"x").unwrap();
    for (mode, private) in [(0o600, true), (0o640, false), (0o604, false), (0o700, true)] {
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(mode)).unwrap();
        assert_eq!(status(&path).unwrap().private(), private, "{mode:o}");
    }
    assert!(status(&path).unwrap().owned_by_current_user());
}

#[test]
fn directories_open_only_as_directories_and_sync() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("file");
    std::fs::write(&path, b"x").unwrap();
    assert!(open_directory(&path).is_err());
    assert!(open_directory(&root.path().join("missing")).is_err());
    open_directory(root.path()).unwrap();
    sync_directory(root.path()).unwrap();
    assert_eq!(
        sync_directory(&root.path().join("missing"))
            .unwrap_err()
            .kind(),
        io::ErrorKind::NotFound
    );
    sync_file(&path).unwrap();
    let file = open_for_sync(&path).unwrap();
    file.sync_all().unwrap();
    sync_renamed(&file).unwrap();
    sync_renamed_path(&path).unwrap();
}

#[test]
fn persisting_publishes_or_refuses_without_retrying_an_existing_name() {
    let root = tempfile::tempdir().unwrap();
    let target = root.path().join("target");
    let mut staged = tempfile::NamedTempFile::new_in(root.path()).unwrap();
    staged.write_all(b"first").unwrap();
    let published = persist_noclobber(staged, &target).unwrap();
    sync_renamed(&published).unwrap();
    // Windows cannot replace a file this process still holds open.
    drop(published);
    assert_eq!(std::fs::read(&target).unwrap(), b"first");
    let mut staged = tempfile::NamedTempFile::new_in(root.path()).unwrap();
    staged.write_all(b"second").unwrap();
    let error = persist_noclobber(staged, &target).unwrap_err();
    assert_eq!(error.error.kind(), io::ErrorKind::AlreadyExists);
    assert_eq!(std::fs::read(&target).unwrap(), b"first");
    persist(error.file, &target).unwrap();
    assert_eq!(std::fs::read(&target).unwrap(), b"second");
    let moved = root.path().join("moved");
    rename(&target, &moved).unwrap();
    assert!(rename(&target, &moved).is_err());
    assert_eq!(std::fs::read(&moved).unwrap(), b"second");
}

#[test]
fn canonical_spellings_agree_for_a_path_and_its_parent() {
    let root = tempfile::tempdir().unwrap();
    std::fs::create_dir(root.path().join("child")).unwrap();
    let parent = canonicalize(root.path()).unwrap();
    let child = canonicalize(&root.path().join("child")).unwrap();
    assert_eq!(child, parent.join("child"));
    assert!(child.starts_with(&parent));
    assert!(canonicalize(&root.path().join("missing")).is_err());
}

#[test]
fn no_follow_opens_regular_files_and_creates_new_ones() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("file");
    std::fs::write(&path, b"bytes").unwrap();
    let mut read = OpenOptions::new();
    read.read(true);
    let file = open_no_follow(&read, &path).unwrap();
    assert_eq!(
        file_status(&file).unwrap().id(),
        status(&path).unwrap().id()
    );
    open_read(&path, true).unwrap();
    open_read(&path, false).unwrap();
    let created = root.path().join("created");
    let mut create = OpenOptions::new();
    create.read(true).write(true).create(true).truncate(false);
    open_no_follow(&create, &created).unwrap();
    assert!(created.is_file());
}

#[cfg(unix)]
#[test]
fn unix_no_follow_refuses_links_and_never_blocks_on_a_fifo() {
    let root = tempfile::tempdir().unwrap();
    let target = root.path().join("target");
    std::fs::write(&target, b"protected").unwrap();
    let link = root.path().join("link");
    std::os::unix::fs::symlink(&target, &link).unwrap();
    let mut write = OpenOptions::new();
    write.write(true).create(true).truncate(true);
    assert!(open_no_follow(&write, &link).is_err());
    assert!(open_directory(&link).is_err());
    assert_eq!(std::fs::read(&target).unwrap(), b"protected");
    assert!(status(&link).unwrap().metadata().file_type().is_symlink());
    open_read(&link, true).unwrap();
    assert!(open_read(&link, false).is_err());
    let fifo = root.path().join("fifo");
    let name = std::ffi::CString::new(fifo.as_os_str().as_encoded_bytes()).unwrap();
    // SAFETY: a valid C string naming a new entry in a private directory.
    assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);
    let opened = open_read(&fifo, false).unwrap();
    assert!(!opened.metadata().unwrap().is_file());
}

#[cfg(windows)]
mod windows_paths {
    use super::*;
    use std::process::Command;

    /// A directory junction, which needs no privilege to create.
    fn junction(link: &Path, target: &Path) {
        let status = Command::new("cmd")
            .arg("/d")
            .arg("/c")
            .arg("mklink")
            .arg("/J")
            .arg(link)
            .arg(target)
            .stdout(std::process::Stdio::null())
            .status()
            .unwrap();
        assert!(status.success(), "mklink /J failed");
    }

    #[test]
    fn a_junction_is_observed_as_a_link_and_never_followed() {
        let root = tempfile::tempdir().unwrap();
        let target = root.path().join("target");
        std::fs::create_dir(&target).unwrap();
        std::fs::write(target.join("inside"), b"protected").unwrap();
        let link = root.path().join("link");
        junction(&link, &target);
        let observed = status(&link).unwrap();
        assert!(observed.metadata().file_type().is_symlink());
        assert!(!observed.metadata().is_dir());
        assert_ne!(observed.id(), status(&target).unwrap().id());
        assert!(open_directory(&link).is_err());
        let mut read = OpenOptions::new();
        read.read(true);
        assert!(open_no_follow(&read, &link).is_err());
        // Following it reaches the target's spelling.
        assert_eq!(
            canonicalize(&link.join("inside")).unwrap(),
            canonicalize(&target.join("inside")).unwrap()
        );
        assert_eq!(std::fs::read(target.join("inside")).unwrap(), b"protected");
    }

    #[test]
    fn case_aliases_share_one_identity_and_one_spelling() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("Member.md");
        std::fs::write(&path, b"x").unwrap();
        let alias = root.path().join("mEMBER.MD");
        assert_eq!(status(&alias).unwrap().id(), status(&path).unwrap().id());
        assert_eq!(canonicalize(&alias).unwrap(), canonicalize(&path).unwrap());
        assert!(canonicalize(&alias).unwrap().ends_with("Member.md"));
    }

    #[test]
    fn verbatim_and_plain_spellings_name_the_same_file() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("file.md");
        std::fs::write(&path, b"x").unwrap();
        let verbatim = std::fs::canonicalize(&path).unwrap();
        assert!(verbatim.to_str().unwrap().starts_with(r"\\?\"));
        let plain = canonicalize(&path).unwrap();
        assert!(!plain.to_str().unwrap().starts_with(r"\\?\"));
        assert_eq!(
            status(&verbatim).unwrap().id(),
            status(&plain).unwrap().id()
        );
        assert_eq!(canonicalize(&verbatim).unwrap(), plain);
        assert_eq!(resolve(&verbatim).unwrap(), plain);
    }

    #[test]
    fn resolution_spells_existing_prefixes_finally_and_keeps_the_missing_rest() {
        let root = tempfile::tempdir().unwrap();
        let base = canonicalize(root.path()).unwrap();
        assert_eq!(
            resolve(&root.path().join(r"missing\..\future\.\file")).unwrap(),
            base.join("future").join("file")
        );
        assert!(!root.path().join("future").exists());
        assert_eq!(resolve(root.path()).unwrap(), base);
        // A short (8.3) or differently cased spelling resolves to the stored one.
        let upper = PathBuf::from(root.path().to_str().unwrap().to_uppercase());
        assert_eq!(resolve(&upper.join("x")).unwrap(), base.join("x"));
    }

    #[test]
    fn drive_relative_paths_resolve_against_that_drives_directory() {
        // `D:name` names an entry of the current directory when D: is the
        // current drive. The working directory is only read, never changed.
        let current = std::env::current_dir().unwrap();
        let text = current.to_str().unwrap();
        if text.as_bytes().get(1) != Some(&b':') {
            return; // A UNC working directory has no drive-relative form.
        }
        let name = "markitai-absent-drive-relative.md";
        assert_eq!(
            resolve(Path::new(&format!("{}{name}", &text[..2]))).unwrap(),
            canonicalize(&current).unwrap().join(name)
        );
    }
}
