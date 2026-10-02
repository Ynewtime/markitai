//! Shared private installation paths, stable no-follow files and bounded downloads.
use crate::{Error, Result, config, platform};
use reqwest::blocking::Client;
use sha2::{Digest, Sha256};
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

pub(crate) type Failure = fn(&str) -> Error;

fn at(failure: Failure, path: &Path, message: &str) -> Error {
    failure(&format!("{}: {message}", path.display()))
}

fn link(status: &platform::Status) -> bool {
    if status.metadata().file_type().is_symlink() {
        return true;
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        status.metadata().file_attributes()
            & windows_sys::Win32::Storage::FileSystem::FILE_ATTRIBUTE_REPARSE_POINT
            != 0
    }
    #[cfg(not(windows))]
    false
}

fn container(status: &platform::Status) -> bool {
    if !status.owned_by_current_user() {
        return false;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        status.metadata().mode() & 0o022 == 0
    }
    #[cfg(not(unix))]
    status.private()
}

struct Step {
    path: PathBuf,
    file: File,
    named: platform::Status,
    trusted_link: bool,
    private: bool,
    container: bool,
}

/// Held directory identities. Ordinary public ancestors (such as /tmp) are
/// outside the managed tree; only root-owned system links there are followed.
pub(crate) struct Directories(Vec<Step>);
impl Directories {
    pub(crate) fn validate(&self, failure: Failure) -> Result<()> {
        for step in &self.0 {
            let named = platform::status(&step.path)
                .map_err(|_| at(failure, &step.path, "installation directory disappeared"))?;
            let held = platform::file_status(&step.file)?;
            if named.id() != step.named.id()
                || (!step.trusted_link && (link(&named) || named.id() != held.id()))
                || (step.trusted_link
                    && (!link(&named)
                        || !platform::root_owned(named.metadata())
                        || platform::followed_status(&step.path)?.id() != held.id()))
                || !held.metadata().is_dir()
                || (step.private && (!named.private() || !named.owned_by_current_user()))
                || (step.container && !container(&named))
            {
                return Err(at(
                    failure,
                    &step.path,
                    "installation directory changed or is unsafe",
                ));
            }
        }
        Ok(())
    }
}

/// Observe every path component, optionally creating missing managed entries.
/// `anchor` is the owned Markitai home container (0755 is permitted there);
/// descendants must be private. Missing paths return None without creating.
pub(crate) fn directories(
    anchor: &Path,
    end: &Path,
    create: bool,
    failure: Failure,
) -> Result<Option<Directories>> {
    let absolute = |path: &Path| -> Result<PathBuf> {
        let path = if path.is_absolute() {
            path.to_owned()
        } else {
            std::env::current_dir()?.join(path)
        };
        if path
            .components()
            .any(|part| matches!(part, std::path::Component::ParentDir))
        {
            return Err(at(
                failure,
                &path,
                "installation paths must not contain parent components",
            ));
        }
        Ok(path)
    };
    let anchor = absolute(anchor)?;
    let end = absolute(end)?;
    if !end.starts_with(&anchor) {
        return Err(at(
            failure,
            &end,
            "installation path is outside its managed root",
        ));
    }
    let mut paths: Vec<_> = end.ancestors().map(Path::to_path_buf).collect();
    paths.reverse();
    let mut observed = Directories(Vec::new());
    for path in paths {
        observed.validate(failure)?;
        let in_tree = path.starts_with(&anchor);
        let named = match platform::status(&path) {
            Ok(status) => status,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                if !create {
                    return Ok(None);
                }
                if !in_tree {
                    return Err(at(
                        failure,
                        &path,
                        "parent of the managed root does not exist",
                    ));
                }
                match platform::private_directory().create(&path) {
                    Ok(()) => {}
                    Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
                    Err(_) => {
                        return Err(at(
                            failure,
                            &path,
                            "cannot create private installation directory",
                        ));
                    }
                }
                observed.validate(failure)?;
                platform::status(&path)?
            }
            Err(_) => return Err(at(failure, &path, "cannot inspect installation directory")),
        };
        let trusted_link = !in_tree && link(&named) && platform::root_owned(named.metadata());
        if (link(&named) && !trusted_link) || (!trusted_link && !named.metadata().is_dir()) {
            return Err(at(
                failure,
                &path,
                "installation directory is a link, junction or special file",
            ));
        }
        let is_container = path == anchor;
        if is_container && !container(&named) {
            return Err(at(
                failure,
                &path,
                "managed home must be owned by this user and not writable by others",
            ));
        }
        if in_tree && !is_container && (!named.private() || !named.owned_by_current_user()) {
            return Err(at(
                failure,
                &path,
                "managed installation directory must be owned by this user and private",
            ));
        }
        let actual = if trusted_link {
            platform::canonicalize(&path)?
        } else {
            path.clone()
        };
        let file = platform::open_directory(&actual).map_err(|_| {
            at(
                failure,
                &path,
                "cannot open installation directory without following a link",
            )
        })?;
        let held = platform::file_status(&file)?;
        let current = platform::status(&path)?;
        if current.id() != named.id() || (!trusted_link && held.id() != named.id()) {
            return Err(at(
                failure,
                &path,
                "installation directory changed while opening",
            ));
        }
        observed.0.push(Step {
            path,
            file,
            named,
            trusted_link,
            private: in_tree && !is_container,
            container: is_container,
        });
    }
    observed.validate(failure)?;
    Ok(Some(observed))
}

/// Browser extraction also uses this helper. Within MARKITAI_HOME it validates
/// the whole managed chain; isolated extraction roots use their requested leaf
/// as the boundary, including all missing parents created along the way. An
/// existing isolated container may be 0755, while managed descendants are private.
pub(crate) fn private_directory(path: &Path, failure: Failure) -> Result<()> {
    let home = config::home();
    let managed = path.starts_with(&home);
    let anchor = if managed {
        home
    } else {
        let mut anchor = path.to_owned();
        while platform::status(&anchor)
            .is_err_and(|error| error.kind() == std::io::ErrorKind::NotFound)
        {
            let parent = anchor
                .parent()
                .filter(|p| !p.as_os_str().is_empty())
                .ok_or_else(|| at(failure, path, "missing installation parent"))?;
            anchor = parent.to_owned();
        }
        anchor
    };
    let dirs = directories(&anchor, path, true, failure)?
        .ok_or_else(|| at(failure, path, "missing installation directory"))?;
    let status = platform::status(path)?;
    if (managed || path != anchor) && (!status.private() || !status.owned_by_current_user()) {
        return Err(at(
            failure,
            path,
            "installation directory must be private and owned by this user",
        ));
    }
    dirs.validate(failure)
}

/// A private, singly linked ordinary file, held without following its leaf.
pub(crate) struct HeldFile {
    pub(crate) file: File,
    pub(crate) named: platform::Status,
    path: PathBuf,
}
impl HeldFile {
    pub(crate) fn open(path: &Path, failure: Failure) -> Result<Self> {
        let named = platform::status(path)
            .map_err(|_| at(failure, path, "cannot inspect installation file"))?;
        safe_file(&named, path, failure)?;
        let file = platform::open_read(path, false).map_err(|_| {
            at(
                failure,
                path,
                "cannot read installation file without following a link",
            )
        })?;
        let held = Self {
            file,
            named,
            path: path.to_owned(),
        };
        held.validate(failure)?;
        Ok(held)
    }
    pub(crate) fn validate(&self, failure: Failure) -> Result<()> {
        let held = platform::file_status(&self.file)?;
        let named = platform::status(&self.path)
            .map_err(|_| at(failure, &self.path, "installation file disappeared"))?;
        safe_file(&held, &self.path, failure)?;
        safe_file(&named, &self.path, failure)?;
        if named.id() != held.id()
            || named.id() != self.named.id()
            || named.changed() != self.named.changed()
            || held.changed() != self.named.changed()
            || named.metadata().len() != self.named.metadata().len()
        {
            return Err(at(
                failure,
                &self.path,
                "installation file changed while it was held",
            ));
        }
        Ok(())
    }
    #[cfg_attr(
        all(target_os = "macos", not(feature = "portable-media")),
        allow(dead_code)
    )]
    pub(crate) fn digest(&mut self, limit: u64, failure: Failure) -> Result<String> {
        self.validate(failure)?;
        self.file.seek(SeekFrom::Start(0))?;
        let mut hash = Sha256::new();
        let mut count = 0u64;
        let mut buffer = [0u8; 64 * 1024];
        loop {
            let wanted = (limit.saturating_sub(count) + 1).min(buffer.len() as u64) as usize;
            let read = self.file.read(&mut buffer[..wanted])?;
            if read == 0 {
                break;
            }
            count += read as u64;
            if count > limit {
                return Err(at(
                    failure,
                    &self.path,
                    "installation file exceeds its byte limit",
                ));
            }
            hash.update(&buffer[..read]);
        }
        self.validate(failure)?;
        Ok(crate::hex(hash.finalize()))
    }
    #[cfg_attr(
        all(target_os = "macos", not(feature = "portable-media")),
        allow(dead_code)
    )]
    pub(crate) fn read(&mut self, limit: u64, failure: Failure) -> Result<Vec<u8>> {
        self.validate(failure)?;
        self.file.seek(SeekFrom::Start(0))?;
        let mut bytes = Vec::with_capacity(self.named.metadata().len().min(limit) as usize);
        (&mut self.file).take(limit + 1).read_to_end(&mut bytes)?;
        self.validate(failure)?;
        if bytes.len() as u64 > limit {
            return Err(at(
                failure,
                &self.path,
                "installation file exceeds its byte limit",
            ));
        }
        Ok(bytes)
    }
}
fn safe_file(status: &platform::Status, path: &Path, failure: Failure) -> Result<()> {
    if link(status) || !status.metadata().is_file() {
        return Err(at(
            failure,
            path,
            "installation file is a link, junction or special file",
        ));
    }
    if !status.owned_by_current_user() || !status.private() || status.links() != 1 {
        return Err(at(
            failure,
            path,
            "installation file must be private, owned by this user and have one hard link",
        ));
    }
    Ok(())
}

pub(crate) struct InstallLock {
    file: File,
    path: PathBuf,
    dirs: Directories,
}
impl Drop for InstallLock {
    fn drop(&mut self) {
        let _ = self.file.unlock();
    }
}
#[derive(Clone, Copy)]
pub(crate) enum Contention {
    Refuse(&'static str),
    #[cfg_attr(
        all(target_os = "macos", not(feature = "portable-media")),
        allow(dead_code)
    )]
    Wait,
}
impl InstallLock {
    pub(crate) fn validate(&self, failure: Failure) -> Result<()> {
        self.dirs.validate(failure)?;
        let held = platform::file_status(&self.file)?;
        let named = platform::status(&self.path)
            .map_err(|_| at(failure, &self.path, "installation lock disappeared"))?;
        safe_file(&held, &self.path, failure)?;
        safe_file(&named, &self.path, failure)?;
        if held.id() != named.id() {
            return Err(at(failure, &self.path, "installation lock was replaced"));
        }
        Ok(())
    }
}
pub(crate) fn lock(root: &Path, contention: Contention, failure: Failure) -> Result<InstallLock> {
    private_directory(root, failure)?;
    let home = config::home();
    let anchor = if root.starts_with(&home) {
        home.as_path()
    } else {
        root
    };
    let dirs = directories(anchor, root, false, failure)?
        .ok_or_else(|| at(failure, root, "missing lock directory"))?;
    let path = root.join("install.lock");
    let expected = match platform::status(&path) {
        Ok(status) => {
            safe_file(&status, &path, failure)?;
            Some(status.id())
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(_) => return Err(at(failure, &path, "cannot inspect installation lock")),
    };
    let mut options = OpenOptions::new();
    options.read(true).write(true).create(true).truncate(false);
    platform::private_file(&mut options);
    let file = platform::open_no_follow(&options, &path).map_err(|_| {
        at(
            failure,
            &path,
            "cannot open installation lock without following a link",
        )
    })?;
    if expected.is_some_and(|id| platform::file_status(&file).is_ok_and(|held| held.id() != id)) {
        return Err(at(
            failure,
            &path,
            "installation lock changed while opening",
        ));
    }
    let lock = InstallLock { file, path, dirs };
    lock.validate(failure)?;
    match contention {
        Contention::Refuse(busy) => lock.file.try_lock().map_err(|_| failure(busy))?,
        Contention::Wait => lock
            .file
            .lock()
            .map_err(|_| failure("locking the installation is unavailable"))?,
    }
    lock.validate(failure)?;
    Ok(lock)
}

/// Streams `url` into `output`, at most `limit` bytes, and returns the
/// SHA-256 of the bytes written, in lowercase hexadecimal.
pub(crate) fn download(
    client: &Client,
    url: &str,
    limit: u64,
    output: &mut impl Write,
    failure: Failure,
) -> Result<String> {
    let mut response = client
        .get(url)
        .send()
        .map_err(|_| failure("official download could not be reached"))?;
    if !response.status().is_success() {
        return Err(failure("official download returned an unsuccessful status"));
    }
    if response.content_length().is_some_and(|n| n > limit) {
        return Err(failure("download exceeds its byte limit"));
    }
    let mut count = 0u64;
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let len = response
            .read(&mut buffer)
            .map_err(|_| failure("official download was interrupted"))?;
        if len == 0 {
            break;
        }
        count += len as u64;
        if count > limit {
            return Err(failure("download exceeds its byte limit"));
        }
        hash.update(&buffer[..len]);
        output
            .write_all(&buffer[..len])
            .map_err(|_| failure("cannot save the download"))?;
    }
    Ok(crate::hex(hash.finalize()))
}

/// Browser executable hashing: no-follow, regular file, bounded by its
/// observed size and the browser installer's 1.5 GiB expanded-archive limit.
pub(crate) fn hash_file(path: &Path) -> Result<String> {
    fn failure(text: &str) -> Error {
        Error::Conversion(text.to_owned())
    }
    let parent = path
        .parent()
        .ok_or_else(|| at(failure, path, "missing installation parent"))?;
    let home = config::home();
    let anchor = if path.starts_with(&home) {
        home.as_path()
    } else {
        parent
    };
    let dirs = directories(anchor, parent, false, failure)?
        .ok_or_else(|| at(failure, parent, "missing installation directory"))?;
    let mut held = HeldFile::open(path, failure)?;
    const LIMIT: u64 = 1536 * 1024 * 1024;
    let bytes = held.named.metadata().len();
    if bytes > LIMIT {
        return Err(at(
            failure,
            path,
            "file exceeds the installation hash limit",
        ));
    }
    let mut hash = Sha256::new();
    let mut remaining = bytes;
    let mut buffer = [0u8; 64 * 1024];
    while remaining > 0 {
        let wanted = remaining.min(buffer.len() as u64) as usize;
        let read = held.file.read(&mut buffer[..wanted])?;
        if read == 0 {
            return Err(at(failure, path, "file changed while hashing"));
        }
        hash.update(&buffer[..read]);
        remaining -= read as u64;
    }
    if held.file.read(&mut [0u8; 1])? != 0 {
        return Err(at(failure, path, "file grew while hashing"));
    }
    held.validate(failure)?;
    dirs.validate(failure)?;
    Ok(crate::hex(hash.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    fn fail(text: &str) -> Error {
        Error::Conversion(text.into())
    }
    fn private(path: &Path, bytes: &[u8]) {
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        let mut file =
            platform::open_no_follow(platform::private_file(&mut options), path).unwrap();
        file.write_all(bytes).unwrap();
    }
    #[test]
    fn a_replaced_lock_is_refused_while_the_original_handle_stays_locked() {
        let root = tempfile::tempdir().unwrap();
        let held = lock(root.path(), Contention::Refuse("busy"), fail).unwrap();
        let name = root.path().join("install.lock");
        fs::rename(&name, root.path().join("old-lock")).unwrap();
        private(&name, b"");
        let error = held.validate(fail).unwrap_err().to_string();
        assert!(error.contains("lock was replaced"), "{error}");
        assert!(fs::read(&name).unwrap().is_empty());
    }
    #[test]
    fn a_held_file_refuses_replacement_and_hardlinks_and_hashing_is_no_follow() {
        let root = tempfile::tempdir().unwrap();
        let name = root.path().join("ordinary");
        private(&name, b"bytes");
        let held = HeldFile::open(&name, fail).unwrap();
        fs::rename(&name, root.path().join("old")).unwrap();
        private(&name, b"bytes");
        assert!(held.validate(fail).is_err());
        fs::hard_link(&name, root.path().join("alias")).unwrap();
        assert!(HeldFile::open(&name, fail).is_err());
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(root.path().join("old"), root.path().join("link")).unwrap();
            assert!(hash_file(&root.path().join("link")).is_err());
        }
    }
    #[cfg(unix)]
    #[test]
    fn unsafe_parent_links_and_public_managed_directories_are_refused() {
        use std::os::unix::fs::{PermissionsExt, symlink};
        let base = tempfile::tempdir().unwrap();
        let home = base.path().join("home");
        platform::private_directory().create(&home).unwrap();
        fs::set_permissions(&home, fs::Permissions::from_mode(0o755)).unwrap();
        let model = home.join("models/ocr");
        assert!(directories(&home, &model, true, fail).unwrap().is_some());
        fs::set_permissions(home.join("models"), fs::Permissions::from_mode(0o755)).unwrap();
        assert!(directories(&home, &model, false, fail).is_err());
        fs::remove_dir(&model).unwrap();
        fs::remove_dir(home.join("models")).unwrap();
        let outside = base.path().join("outside");
        platform::private_directory().create(&outside).unwrap();
        symlink(&outside, home.join("models")).unwrap();
        assert!(directories(&home, &model, true, fail).is_err());
        assert_eq!(fs::read_dir(&outside).unwrap().count(), 0);
    }
}
