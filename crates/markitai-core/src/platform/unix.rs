//! Unix: every fact comes from `lstat`/`fstat`; directories are `fsync`ed.
use super::{FileId, InterfaceAddress, Owner, Status};
use std::fs::{self, DirBuilder, File, Metadata, OpenOptions};
use std::io;
use std::net::Ipv4Addr;
use std::os::unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt};
use std::path::{Path, PathBuf};

fn from_metadata(metadata: Metadata) -> Status {
    Status {
        id: FileId {
            volume: metadata.dev(),
            file: u128::from(metadata.ino()),
        },
        links: metadata.nlink(),
        changed: (metadata.ctime(), metadata.ctime_nsec()),
        private: metadata.mode() & 0o077 == 0,
        owner: Owner::User(metadata.uid()),
        metadata,
    }
}

pub(super) fn status(path: &Path) -> io::Result<Status> {
    fs::symlink_metadata(path).map(from_metadata)
}

pub(super) fn file_status(file: &File) -> io::Result<Status> {
    file.metadata().map(from_metadata)
}

pub(super) fn followed_status(path: &Path) -> io::Result<Status> {
    fs::metadata(path).map(from_metadata)
}

pub(super) fn owned_by_current_user(status: &Status) -> bool {
    // SAFETY: geteuid has no preconditions and cannot fail.
    status.owner == Owner::User(unsafe { libc::geteuid() })
}

pub(super) fn root_owned(metadata: &Metadata) -> bool {
    metadata.uid() == 0
}

pub(super) fn open_no_follow(options: &OpenOptions, path: &Path) -> io::Result<File> {
    let mut options = options.clone();
    options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    options.open(path)
}

pub(super) fn open_nonblocking(options: &OpenOptions, path: &Path) -> io::Result<File> {
    let mut options = options.clone();
    options.custom_flags(libc::O_NONBLOCK);
    options.open(path)
}

pub(super) fn open_directory(path: &Path) -> io::Result<File> {
    OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)
}

pub(super) fn private_file(options: &mut OpenOptions) -> &mut OpenOptions {
    options.mode(0o600)
}

pub(super) fn restrict_to_owner(path: &Path) -> io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(0o600))
}

pub(super) fn private_directory() -> DirBuilder {
    let mut builder = DirBuilder::new();
    builder.mode(0o700);
    builder
}

pub(super) fn sync_directory(path: &Path) -> io::Result<()> {
    File::open(path)?.sync_all()
}

pub(super) fn sync_renamed(_: &File) -> io::Result<()> {
    Ok(())
}

pub(super) fn sync_renamed_path(_: &Path) -> io::Result<()> {
    Ok(())
}

pub(super) fn sync_file(path: &Path) -> io::Result<()> {
    File::open(path)?.sync_all()
}

pub(super) fn open_for_sync(path: &Path) -> io::Result<File> {
    File::open(path)
}

pub(super) fn canonicalize(path: &Path) -> io::Result<PathBuf> {
    fs::canonicalize(path)
}

/// `getifaddrs`: a tunnel is a point-to-point interface.
pub(super) fn interface_addresses() -> Vec<InterfaceAddress> {
    let mut list = std::ptr::null_mut();
    // SAFETY: `list` receives a list that is freed below.
    if unsafe { libc::getifaddrs(&mut list) } != 0 {
        return Vec::new();
    }
    let mut found = Vec::new();
    let mut entry = list;
    while !entry.is_null() {
        // SAFETY: the list stays valid until `freeifaddrs`.
        let interface = unsafe { &*entry };
        entry = interface.ifa_next;
        let flags = interface.ifa_flags as libc::c_int;
        let address = interface.ifa_addr;
        if flags & libc::IFF_UP == 0 || flags & libc::IFF_RUNNING == 0 || address.is_null() {
            continue;
        }
        // SAFETY: a non-null `ifa_addr` is a socket address.
        if i32::from(unsafe { (*address).sa_family }) != libc::AF_INET {
            continue;
        }
        // SAFETY: a socket address of the family `AF_INET` is a `sockaddr_in`.
        let address = unsafe { address.cast::<libc::sockaddr_in>().read_unaligned() };
        found.push(InterfaceAddress {
            ip: Ipv4Addr::from(u32::from_be(address.sin_addr.s_addr)),
            on_local_network: flags & libc::IFF_POINTOPOINT == 0,
        });
    }
    // SAFETY: `list` came from `getifaddrs` and is freed once.
    unsafe { libc::freeifaddrs(list) };
    found
}
