//! Ordering fence for a staged file that a later rename makes visible.
//!
//! The core never synchronizes the directory entry of a document, asset or
//! sidecar it publishes, so a name was never durable when a write returned;
//! what the staging synchronization guarantees is that a name which does
//! survive a crash refers to complete bytes. That needs ordering, not a drive
//! cache flush: on a verified local macOS APFS/HFS volume, `fsync` followed by
//! `F_BARRIERFSYNC` puts the staged bytes ahead of every later write to that
//! device, the rename included (fcntl(2)). A file system that rejects the
//! barrier gets `F_FULLFSYNC`. Elsewhere the staged file keeps `File::sync_all`.
use std::fs::File;
use std::io;

/// Synchronize a staged file so that no later write on its device, in
/// particular the rename publishing it, can reach stable storage first.
pub(crate) fn order_staged(file: &File) -> io::Result<()> {
    #[cfg(target_os = "macos")]
    if verified_local(file)? {
        use std::os::fd::AsRawFd;
        let descriptor = file.as_raw_fd();
        // SAFETY: the descriptor is borrowed from a live File for each call and
        // none of these operations take pointer arguments.
        retry_interrupted(|| unsafe { libc::fsync(descriptor) })?;
        let barrier =
            retry_interrupted(|| unsafe { libc::fcntl(descriptor, libc::F_BARRIERFSYNC) });
        return match barrier {
            Err(error)
                if matches!(
                    error.raw_os_error(),
                    Some(libc::ENOTSUP | libc::ENOTTY | libc::EINVAL)
                ) =>
            {
                note("full");
                retry_interrupted(|| unsafe { libc::fcntl(descriptor, libc::F_FULLFSYNC) })
            }
            result => {
                note("barrier");
                result
            }
        };
    }
    note("sync");
    file.sync_all()
}

/// A regular file or directory on a local APFS or HFS volume, the file
/// systems whose barrier operation the fcntl manual describes.
#[cfg(target_os = "macos")]
pub(crate) fn verified_local(file: &File) -> io::Result<bool> {
    use std::os::fd::AsRawFd;
    let metadata = file.metadata()?;
    if !metadata.is_file() && !metadata.is_dir() {
        return Ok(false);
    }
    let mut filesystem = std::mem::MaybeUninit::<libc::statfs>::uninit();
    // SAFETY: fstatfs initializes the provided struct on success and the
    // descriptor is borrowed from a live File for this call.
    if unsafe { libc::fstatfs(file.as_raw_fd(), filesystem.as_mut_ptr()) } != 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: the preceding call succeeded.
    let filesystem = unsafe { filesystem.assume_init() };
    let name: Vec<u8> = filesystem
        .f_fstypename
        .iter()
        .take_while(|&&byte| byte != 0)
        .map(|&byte| byte as u8)
        .collect();
    let local = (filesystem.f_flags as u64 & libc::MNT_LOCAL as u64) != 0;
    Ok(local && matches!(name.as_slice(), b"apfs" | b"hfs"))
}

#[cfg(target_os = "macos")]
fn retry_interrupted(mut operation: impl FnMut() -> libc::c_int) -> io::Result<()> {
    loop {
        if operation() != -1 {
            return Ok(());
        }
        let error = io::Error::last_os_error();
        if error.kind() != io::ErrorKind::Interrupted {
            return Err(error);
        }
    }
}

#[cfg(test)]
thread_local! {
    static EVENTS: std::cell::RefCell<Vec<&'static str>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Record a fence kind or a publication step for this test thread.
#[cfg_attr(not(test), inline(always))]
pub(crate) fn note(_event: &'static str) {
    #[cfg(test)]
    EVENTS.with(|events| events.borrow_mut().push(_event));
}

/// Fences and publication steps noted by this test thread since the last call.
#[cfg(test)]
pub(crate) fn take() -> Vec<&'static str> {
    EVENTS.with(|events| events.borrow_mut().drain(..).collect())
}

/// The fence a staged file in this directory receives: the barrier on a
/// verified local volume, otherwise the per-object durable synchronization.
#[cfg(test)]
pub(crate) fn expected(directory: &std::path::Path) -> &'static str {
    #[cfg(target_os = "macos")]
    if verified_local(&File::open(directory).unwrap()).unwrap() {
        return "barrier";
    }
    let _ = directory;
    "sync"
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn staged_bytes_are_fenced_once_and_remain_readable() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("staged");
        let mut file = File::create(&path).unwrap();
        file.write_all(b"complete staged bytes").unwrap();
        take();
        order_staged(&file).unwrap();
        assert_eq!(take(), [expected(directory.path())]);
        assert_eq!(std::fs::read(path).unwrap(), b"complete staged bytes");
    }

    /// The system temporary directory of a macOS host is on its local APFS
    /// data volume, checked here independently of `verified_local`.
    #[cfg(target_os = "macos")]
    #[test]
    fn local_apfs_temporary_directory_takes_the_barrier() {
        use std::os::unix::ffi::OsStrExt;
        let directory = tempfile::tempdir().unwrap();
        let spelling = std::ffi::CString::new(directory.path().as_os_str().as_bytes()).unwrap();
        let mut filesystem = std::mem::MaybeUninit::<libc::statfs>::uninit();
        // SAFETY: a valid C string and an output struct that statfs initializes.
        assert_eq!(
            unsafe { libc::statfs(spelling.as_ptr(), filesystem.as_mut_ptr()) },
            0
        );
        // SAFETY: statfs succeeded.
        let filesystem = unsafe { filesystem.assume_init() };
        // SAFETY: f_fstypename is a NUL-terminated array filled by statfs.
        let name = unsafe { std::ffi::CStr::from_ptr(filesystem.f_fstypename.as_ptr()) };
        let apfs =
            name.to_bytes() == b"apfs" && (filesystem.f_flags as u64 & libc::MNT_LOCAL as u64) != 0;
        let file = File::create(directory.path().join("staged")).unwrap();
        assert_eq!(verified_local(&file).unwrap(), apfs);
        take();
        order_staged(&file).unwrap();
        assert_eq!(take(), [if apfs { "barrier" } else { "sync" }]);
    }
}
