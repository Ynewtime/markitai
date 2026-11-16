//! The background check's process: it outlives the run and holds nothing of
//! the run's but the null device.
//!
//! A child that `std::process::Command` starts keeps every descriptor not
//! marked close-on-exec and, on Windows, every inheritable handle. The run
//! marks its own, but not those it was given: a shell's process substitution
//! (`markitai a.pdf > >(tee log)`) or a Node parent's IPC channel on 3 and up,
//! and on Windows the pipe its standard output is (`markitai a.pdf | findstr
//! x`). A check holding one keeps that reader waiting as long as the check
//! runs, 30 seconds and more for a release host that does not answer.

use std::io;
use std::path::Path;

/// Start `program` with `args` in a process group of its own (on Windows
/// without a console), its standard streams the null device and no other file
/// of this process open in it. It is not waited for.
pub(crate) fn spawn(program: &Path, args: &[&str]) -> io::Result<()> {
    imp::spawn(program, args)
}

/// macOS: `posix_spawn` closes every descriptor its file actions do not name
/// (`POSIX_SPAWN_CLOEXEC_DEFAULT`) as the kernel starts the program.
#[cfg(target_vendor = "apple")]
mod imp {
    use std::ffi::{CString, OsStr};
    use std::io;
    use std::mem::MaybeUninit;
    use std::os::unix::ffi::OsStrExt;
    use std::path::Path;

    fn c(text: &OsStr) -> io::Result<CString> {
        CString::new(text.as_bytes()).map_err(io::Error::other)
    }

    fn done(code: libc::c_int) -> io::Result<()> {
        match code {
            0 => Ok(()),
            code => Err(io::Error::from_raw_os_error(code)),
        }
    }

    fn pointers(strings: &[CString]) -> Vec<*mut libc::c_char> {
        strings
            .iter()
            .map(|string| string.as_ptr().cast_mut())
            .chain(std::iter::once(std::ptr::null_mut()))
            .collect()
    }

    struct Actions(libc::posix_spawn_file_actions_t);

    impl Drop for Actions {
        fn drop(&mut self) {
            unsafe { libc::posix_spawn_file_actions_destroy(&mut self.0) };
        }
    }

    struct Attributes(libc::posix_spawnattr_t);

    impl Drop for Attributes {
        fn drop(&mut self) {
            unsafe { libc::posix_spawnattr_destroy(&mut self.0) };
        }
    }

    pub(super) fn spawn(program: &Path, args: &[&str]) -> io::Result<()> {
        let path = c(program.as_os_str())?;
        let argv = std::iter::once(Ok(path.clone()))
            .chain(args.iter().map(|arg| c(OsStr::new(arg))))
            .collect::<io::Result<Vec<_>>>()?;
        // The environment as the standard library reads it, under its lock.
        let environment = std::env::vars_os()
            .map(|(mut pair, value)| {
                pair.push("=");
                pair.push(value);
                c(&pair)
            })
            .collect::<io::Result<Vec<_>>>()?;
        let (argv, envp) = (pointers(&argv), pointers(&environment));
        unsafe {
            let mut actions = MaybeUninit::uninit();
            done(libc::posix_spawn_file_actions_init(actions.as_mut_ptr()))?;
            let mut actions = Actions(actions.assume_init());
            for (descriptor, access) in [
                (0, libc::O_RDONLY),
                (1, libc::O_WRONLY),
                (2, libc::O_WRONLY),
            ] {
                done(libc::posix_spawn_file_actions_addopen(
                    &mut actions.0,
                    descriptor,
                    c"/dev/null".as_ptr(),
                    access,
                    0,
                ))?;
            }
            let mut attributes = MaybeUninit::uninit();
            done(libc::posix_spawnattr_init(attributes.as_mut_ptr()))?;
            let mut attributes = Attributes(attributes.assume_init());
            // As the standard library starts a child: no signal blocked, and
            // SIGPIPE, which this process ignores, back to its default.
            let mut signals = MaybeUninit::uninit();
            libc::sigemptyset(signals.as_mut_ptr());
            let mut signals = signals.assume_init();
            done(libc::posix_spawnattr_setsigmask(
                &mut attributes.0,
                &signals,
            ))?;
            libc::sigaddset(&mut signals, libc::SIGPIPE);
            done(libc::posix_spawnattr_setsigdefault(
                &mut attributes.0,
                &signals,
            ))?;
            done(libc::posix_spawnattr_setpgroup(&mut attributes.0, 0))?;
            let flags = libc::POSIX_SPAWN_SETPGROUP
                | libc::POSIX_SPAWN_SETSIGMASK
                | libc::POSIX_SPAWN_SETSIGDEF
                | libc::POSIX_SPAWN_CLOEXEC_DEFAULT;
            done(libc::posix_spawnattr_setflags(
                &mut attributes.0,
                flags as libc::c_short,
            ))?;
            let mut pid = 0;
            done(libc::posix_spawn(
                &mut pid,
                path.as_ptr(),
                &actions.0,
                &attributes.0,
                argv.as_ptr(),
                envp.as_ptr(),
            ))
        }
    }
}

/// Linux: the standard library's spawn, which marks 3 and up close-on-exec in
/// the child before the program starts.
#[cfg(all(unix, not(target_vendor = "apple")))]
mod imp {
    use std::io;
    use std::os::unix::process::CommandExt;
    use std::path::Path;
    use std::process::{Command, Stdio};

    pub(super) fn spawn(program: &Path, args: &[&str]) -> io::Result<()> {
        let mut command = Command::new(program);
        command
            .args(args)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .process_group(0);
        let limit = limit();
        // Between fork and exec: async-signal-safe calls only.
        unsafe {
            command.pre_exec(move || {
                close_on_exec_from_3(limit);
                Ok(())
            });
        }
        command.spawn().map(drop)
    }

    /// The numbers to mark one by one where the kernel cannot mark them all
    /// in one call: below the soft limit, or Linux's own default ceiling.
    fn limit() -> libc::c_int {
        const CEILING: libc::c_long = 1 << 20;
        match unsafe { libc::sysconf(libc::_SC_OPEN_MAX) } {
            open if (0..=CEILING).contains(&open) => open as libc::c_int,
            _ => CEILING as libc::c_int,
        }
    }

    fn close_on_exec_from_3(limit: libc::c_int) {
        // One call from Linux 5.11.
        #[cfg(any(target_os = "linux", target_os = "android"))]
        {
            let marked = unsafe {
                libc::syscall(
                    libc::SYS_close_range,
                    3 as libc::c_uint,
                    libc::c_uint::MAX,
                    libc::CLOSE_RANGE_CLOEXEC,
                )
            };
            if marked == 0 {
                return;
            }
        }
        for descriptor in 3..limit {
            unsafe { libc::fcntl(descriptor, libc::F_SETFD, libc::FD_CLOEXEC) };
        }
    }
}

/// Windows: `CreateProcessW` with the list of handles the child inherits, the
/// null device alone; the standard library has it inherit every inheritable
/// handle of this process.
#[cfg(windows)]
mod imp {
    use std::ffi::OsString;
    use std::io;
    use std::os::windows::ffi::OsStrExt;
    use std::os::windows::io::AsRawHandle;
    use std::path::Path;
    use windows_sys::Win32::Foundation::{
        CloseHandle, HANDLE, HANDLE_FLAG_INHERIT, SetHandleInformation,
    };
    use windows_sys::Win32::System::Threading::{
        CREATE_NEW_PROCESS_GROUP, CREATE_NO_WINDOW, CreateProcessW, DETACHED_PROCESS,
        DeleteProcThreadAttributeList, EXTENDED_STARTUPINFO_PRESENT,
        InitializeProcThreadAttributeList, LPPROC_THREAD_ATTRIBUTE_LIST,
        PROC_THREAD_ATTRIBUTE_HANDLE_LIST, PROCESS_INFORMATION, STARTF_USESTDHANDLES,
        STARTUPINFOEXW, UpdateProcThreadAttribute,
    };

    fn done(succeeded: windows_sys::core::BOOL) -> io::Result<()> {
        if succeeded == 0 {
            Err(io::Error::last_os_error())
        } else {
            Ok(())
        }
    }

    struct List(LPPROC_THREAD_ATTRIBUTE_LIST);

    impl Drop for List {
        fn drop(&mut self) {
            unsafe { DeleteProcThreadAttributeList(self.0) };
        }
    }

    pub(super) fn spawn(program: &Path, args: &[&str]) -> io::Result<()> {
        let null = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open("NUL")?;
        let handle = null.as_raw_handle() as HANDLE;
        // A handle the list names must be inheritable.
        done(unsafe { SetHandleInformation(handle, HANDLE_FLAG_INHERIT, HANDLE_FLAG_INHERIT) })?;
        let mut size = 0;
        // Asks for the size, failing with ERROR_INSUFFICIENT_BUFFER as it does.
        unsafe { InitializeProcThreadAttributeList(std::ptr::null_mut(), 1, 0, &mut size) };
        let mut storage = vec![0_usize; size.div_ceil(size_of::<usize>())];
        let list: LPPROC_THREAD_ATTRIBUTE_LIST = storage.as_mut_ptr().cast();
        done(unsafe { InitializeProcThreadAttributeList(list, 1, 0, &mut size) })?;
        let list = List(list);
        let inherited = [handle];
        done(unsafe {
            UpdateProcThreadAttribute(
                list.0,
                0,
                PROC_THREAD_ATTRIBUTE_HANDLE_LIST as usize,
                inherited.as_ptr().cast(),
                size_of_val(&inherited),
                std::ptr::null_mut(),
                std::ptr::null(),
            )
        })?;
        let mut startup: STARTUPINFOEXW = unsafe { std::mem::zeroed() };
        startup.StartupInfo.cb = size_of::<STARTUPINFOEXW>() as u32;
        startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
        startup.StartupInfo.hStdInput = handle;
        startup.StartupInfo.hStdOutput = handle;
        startup.StartupInfo.hStdError = handle;
        startup.lpAttributeList = list.0;
        let application: Vec<u16> = program.as_os_str().encode_wide().chain([0]).collect();
        // The program in quotes (a path holds none), then words that need none.
        let mut line = OsString::from("\"");
        line.push(program);
        line.push("\"");
        for arg in args {
            line.push(" ");
            line.push(arg);
        }
        let mut line: Vec<u16> = line.encode_wide().chain([0]).collect();
        let mut process: PROCESS_INFORMATION = unsafe { std::mem::zeroed() };
        done(unsafe {
            CreateProcessW(
                application.as_ptr(),
                line.as_mut_ptr(),
                std::ptr::null(),
                std::ptr::null(),
                1,
                DETACHED_PROCESS
                    | CREATE_NEW_PROCESS_GROUP
                    | CREATE_NO_WINDOW
                    | EXTENDED_STARTUPINFO_PRESENT,
                std::ptr::null(),
                std::ptr::null(),
                &startup.StartupInfo,
                &mut process,
            )
        })?;
        unsafe {
            CloseHandle(process.hProcess);
            CloseHandle(process.hThread);
        }
        Ok(())
    }
}
