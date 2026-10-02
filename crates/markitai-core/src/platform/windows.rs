//! Windows: identity, link count and change time come from
//! `GetFileInformationByHandleEx`, ownership from the security descriptor.
//! Entries are observed through handles opened with
//! `FILE_FLAG_BACKUP_SEMANTICS` (so directories open) and
//! `FILE_FLAG_OPEN_REPARSE_POINT` (so a link or junction is observed itself).
use super::{FileId, Owner, Status};
use std::ffi::{OsStr, c_void};
use std::fs::{self, DirBuilder, File, OpenOptions};
use std::io;
use std::mem::{MaybeUninit, size_of};
use std::os::windows::fs::{MetadataExt, OpenOptionsExt};
use std::os::windows::io::AsRawHandle;
use std::path::{Path, PathBuf};
use std::ptr::null_mut;
use std::sync::OnceLock;
use windows_sys::Win32::Foundation::{CloseHandle, ERROR_SUCCESS, HANDLE, LocalFree};
use windows_sys::Win32::Security::Authorization::{GetSecurityInfo, SE_FILE_OBJECT};
use windows_sys::Win32::Security::{
    GetLengthSid, GetTokenInformation, IsValidSid, OWNER_SECURITY_INFORMATION,
    PSECURITY_DESCRIPTOR, PSID, TOKEN_INFORMATION_CLASS, TOKEN_OWNER, TOKEN_QUERY, TOKEN_USER,
    TokenOwner, TokenUser,
};
use windows_sys::Win32::Storage::FileSystem::{
    BY_HANDLE_FILE_INFORMATION, FILE_ATTRIBUTE_REPARSE_POINT, FILE_BASIC_INFO,
    FILE_FLAG_BACKUP_SEMANTICS, FILE_FLAG_OPEN_REPARSE_POINT, FILE_ID_INFO,
    FILE_INFO_BY_HANDLE_CLASS, FILE_READ_ATTRIBUTES, FILE_STANDARD_INFO, FileBasicInfo, FileIdInfo,
    FileStandardInfo, GetFileInformationByHandle, GetFileInformationByHandleEx, READ_CONTROL,
};
use windows_sys::Win32::System::Threading::{GetCurrentProcess, OpenProcessToken};

fn handle(file: &File) -> HANDLE {
    file.as_raw_handle() as HANDLE
}

/// One fixed-size `GetFileInformationByHandleEx` record.
fn query<T>(file: &File, class: FILE_INFO_BY_HANDLE_CLASS) -> io::Result<T> {
    let mut value = MaybeUninit::<T>::zeroed();
    // SAFETY: the handle is borrowed from a live File, and the buffer is
    // exactly the size of the record this information class writes.
    let ok = unsafe {
        GetFileInformationByHandleEx(
            handle(file),
            class,
            value.as_mut_ptr().cast::<c_void>(),
            size_of::<T>() as u32,
        )
    };
    if ok == 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: the call succeeded and filled the record; every field of these
    // plain records is valid for any bit pattern the system writes.
    Ok(unsafe { value.assume_init() })
}

fn identity(file: &File) -> io::Result<FileId> {
    if let Ok(info) = query::<FILE_ID_INFO>(file, FileIdInfo) {
        return Ok(FileId {
            volume: info.VolumeSerialNumber,
            file: u128::from_le_bytes(info.FileId.Identifier),
        });
    }
    // File systems without FileIdInfo still report the 64-bit file index.
    let mut info = MaybeUninit::<BY_HANDLE_FILE_INFORMATION>::zeroed();
    // SAFETY: a live handle and a writable record of the documented type.
    if unsafe { GetFileInformationByHandle(handle(file), info.as_mut_ptr()) } == 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: the call succeeded.
    let info = unsafe { info.assume_init() };
    Ok(FileId {
        volume: u64::from(info.dwVolumeSerialNumber),
        file: (u128::from(info.nFileIndexHigh) << 32) | u128::from(info.nFileIndexLow),
    })
}

/// A copy of a SID the system returned.
///
/// # Safety
/// `sid` is null or points to memory the system owns for the whole call.
unsafe fn sid_bytes(sid: PSID) -> Option<Vec<u8>> {
    // SAFETY: IsValidSid accepts any pointer to a candidate SID; the length
    // is read only from a valid one, and exactly that many bytes are copied.
    unsafe {
        if sid.is_null() || IsValidSid(sid) == 0 {
            return None;
        }
        let length = GetLengthSid(sid) as usize;
        Some(std::slice::from_raw_parts(sid.cast::<u8>(), length).to_vec())
    }
}

fn owner(file: &File) -> Owner {
    let mut sid: PSID = null_mut();
    let mut descriptor: PSECURITY_DESCRIPTOR = null_mut();
    // SAFETY: a live handle; the out pointers are valid for the call, and the
    // descriptor the system allocates is released below after the owner SID
    // inside it was copied.
    let status = unsafe {
        GetSecurityInfo(
            handle(file),
            SE_FILE_OBJECT,
            OWNER_SECURITY_INFORMATION,
            &mut sid,
            null_mut(),
            null_mut(),
            null_mut(),
            &mut descriptor,
        )
    };
    if status != ERROR_SUCCESS {
        return Owner::Unknown;
    }
    // SAFETY: the SID points into the descriptor, which is still allocated.
    let owner = unsafe { sid_bytes(sid) }.map_or(Owner::Unknown, Owner::Sid);
    // SAFETY: GetSecurityInfo allocated this descriptor with LocalAlloc.
    unsafe { LocalFree(descriptor) };
    owner
}

/// The SIDs that own what this process creates: its user, and the default
/// owner of its token when that differs (an elevated administrator's token
/// assigns `BUILTIN\Administrators`). Empty when the token cannot be read,
/// which makes nothing private.
fn current_owners() -> &'static [Vec<u8>] {
    static OWNERS: OnceLock<Vec<Vec<u8>>> = OnceLock::new();
    OWNERS.get_or_init(|| token_owners().unwrap_or_default())
}

fn token_owners() -> Option<Vec<Vec<u8>>> {
    let mut token: HANDLE = null_mut();
    // SAFETY: the pseudo-handle of this process and a writable handle slot.
    if unsafe { OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &mut token) } == 0 {
        return None;
    }
    let owners = (|| {
        let user = token_information(token, TokenUser)?;
        // SAFETY: a successful TokenUser query starts with a TOKEN_USER whose
        // SID points into the same (8-byte aligned) buffer.
        let user = unsafe { sid_bytes((*user.as_ptr().cast::<TOKEN_USER>()).User.Sid) }?;
        let mut owners = vec![user];
        if let Some(owner) = token_information(token, TokenOwner)
            // SAFETY: as above, for TOKEN_OWNER.
            && let Some(owner) = unsafe { sid_bytes((*owner.as_ptr().cast::<TOKEN_OWNER>()).Owner) }
            && !owners.contains(&owner)
        {
            owners.push(owner);
        }
        Some(owners)
    })();
    // SAFETY: the token handle was opened above and is closed once.
    unsafe { CloseHandle(token) };
    owners
}

fn token_information(token: HANDLE, class: TOKEN_INFORMATION_CLASS) -> Option<Vec<u64>> {
    let mut needed = 0u32;
    // SAFETY: a size query with no buffer; it fails and reports the size.
    unsafe { GetTokenInformation(token, class, null_mut(), 0, &mut needed) };
    if needed == 0 {
        return None;
    }
    let mut buffer = vec![0u64; (needed as usize).div_ceil(size_of::<u64>())];
    // SAFETY: the buffer holds at least `needed` writable bytes.
    let ok = unsafe {
        GetTokenInformation(
            token,
            class,
            buffer.as_mut_ptr().cast::<c_void>(),
            (buffer.len() * size_of::<u64>()) as u32,
            &mut needed,
        )
    };
    (ok != 0).then_some(buffer)
}

/// Open an entry to read its attributes and owner, without following a final
/// reparse point. An ACL that denies reading the owner still yields the
/// attributes (and an unknown owner).
fn open_attributes(path: &Path) -> io::Result<File> {
    open_attributes_with(
        path,
        FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT,
    )
}

fn open_attributes_with(path: &Path, flags: u32) -> io::Result<File> {
    let open = |access: u32| {
        OpenOptions::new()
            .access_mode(access)
            .custom_flags(flags)
            .open(path)
    };
    match open(FILE_READ_ATTRIBUTES | READ_CONTROL) {
        Err(error) if error.kind() == io::ErrorKind::PermissionDenied => open(FILE_READ_ATTRIBUTES),
        result => result,
    }
}

pub(super) fn status(path: &Path) -> io::Result<Status> {
    file_status(&open_attributes(path)?)
}

pub(super) fn followed_status(path: &Path) -> io::Result<Status> {
    file_status(&open_attributes_with(path, FILE_FLAG_BACKUP_SEMANTICS)?)
}

pub(super) fn file_status(file: &File) -> io::Result<Status> {
    let metadata = file.metadata()?;
    let id = identity(file)?;
    let standard = query::<FILE_STANDARD_INFO>(file, FileStandardInfo)?;
    let basic = query::<FILE_BASIC_INFO>(file, FileBasicInfo)?;
    let owner = owner(file);
    let private = matches!(&owner, Owner::Sid(sid) if current_owners().contains(sid));
    Ok(Status {
        metadata,
        id,
        links: u64::from(standard.NumberOfLinks),
        changed: (basic.ChangeTime, 0),
        private,
        owner,
    })
}

pub(super) fn owned_by_current_user(status: &Status) -> bool {
    status.private
}

pub(super) fn root_owned(_: &fs::Metadata) -> bool {
    false
}

fn link_error(path: &Path) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidInput,
        format!("{} is a symbolic link or junction", path.display()),
    )
}

pub(super) fn open_no_follow(options: &OpenOptions, path: &Path) -> io::Result<File> {
    let mut entry = options.clone();
    entry.custom_flags(FILE_FLAG_OPEN_REPARSE_POINT);
    let file = entry.open(path)?;
    let metadata = file.metadata()?;
    if metadata.file_type().is_symlink() {
        return Err(link_error(path));
    }
    if metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT == 0 {
        return Ok(file);
    }
    // A reparse point that is not a link (a cloud placeholder, a deduplicated
    // file) stands for data only a normal open provides. That open must reach
    // the same file the entry named.
    let opened = options.open(path)?;
    if identity(&opened)? != identity(&file)? {
        return Err(io::Error::other(format!(
            "{} changed while it was opened",
            path.display()
        )));
    }
    Ok(opened)
}

pub(super) fn open_nonblocking(options: &OpenOptions, path: &Path) -> io::Result<File> {
    options.open(path)
}

pub(super) fn open_directory(path: &Path) -> io::Result<File> {
    let file = open_attributes(path)?;
    if !file.metadata()?.is_dir() {
        return Err(io::Error::new(
            io::ErrorKind::NotADirectory,
            format!("{} is not a directory", path.display()),
        ));
    }
    Ok(file)
}

pub(super) fn private_file(options: &mut OpenOptions) -> &mut OpenOptions {
    options
}

pub(super) fn private_directory() -> DirBuilder {
    DirBuilder::new()
}

pub(super) fn sync_directory(path: &Path) -> io::Result<()> {
    // No directory flush exists; keep the error a missing directory gives
    // on Unix.
    fs::metadata(path).map(|_| ())
}

pub(super) fn sync_renamed(file: &File) -> io::Result<()> {
    file.sync_all()
}

pub(super) fn sync_renamed_path(path: &Path) -> io::Result<()> {
    sync_file(path)
}

pub(super) fn sync_file(path: &Path) -> io::Result<()> {
    open_for_sync(path)?.sync_all()
}

pub(super) fn open_for_sync(path: &Path) -> io::Result<File> {
    // FlushFileBuffers requires write access; opening for writing neither
    // creates nor truncates.
    OpenOptions::new().write(true).open(path)
}

fn simplified(path: PathBuf) -> PathBuf {
    match path.to_str().and_then(super::plain_spelling) {
        Some(plain) => PathBuf::from(plain),
        None => path,
    }
}

pub(super) fn canonicalize(path: &Path) -> io::Result<PathBuf> {
    fs::canonicalize(path).map(simplified)
}

/// Errors after which a prefix is treated as absent, as Python's non-strict
/// `realpath` does: missing or inaccessible entries, unavailable devices and
/// shares, and names or file systems the final-path query does not support.
fn unresolvable(error: &io::Error) -> bool {
    matches!(
        error.kind(),
        io::ErrorKind::NotFound | io::ErrorKind::NotADirectory | io::ErrorKind::PermissionDenied
    ) || matches!(
        error.raw_os_error(),
        Some(1 | 21 | 32 | 50 | 53 | 65 | 67 | 87 | 123 | 161 | 1920 | 1921)
    )
}

pub(super) fn resolve(path: &Path) -> io::Result<PathBuf> {
    // GetFullPathNameW: an absolute path with `.` and `..` already applied.
    let absolute = std::path::absolute(path)?;
    let mut existing = absolute.as_path();
    let mut missing: Vec<&OsStr> = Vec::new();
    loop {
        match fs::canonicalize(existing) {
            Ok(found) => {
                let mut resolved = simplified(found);
                resolved.extend(missing.iter().rev().copied());
                return Ok(resolved);
            }
            Err(error) if unresolvable(&error) => match (existing.parent(), existing.file_name()) {
                (Some(parent), Some(name)) => {
                    missing.push(name);
                    existing = parent;
                }
                // Not even the root exists (a missing drive): nothing to spell.
                _ => return Ok(absolute),
            },
            Err(error) => return Err(error),
        }
    }
}
