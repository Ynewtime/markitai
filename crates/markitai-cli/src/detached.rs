//! Processes that outlive the one starting them and hold nothing of its but
//! the null device: the background update check, and on Windows the editor
//! the server opens its configuration in.
//!
//! A child that `std::process::Command` starts keeps every descriptor not
//! marked close-on-exec and, on Windows, every inheritable handle. A run
//! marks its own, but not those it was given: a shell's process substitution
//! (`markitai a.pdf > >(tee log)`) or a Node parent's IPC channel on 3 and up,
//! and on Windows the pipe or file its standard output is (`markitai a.pdf |
//! findstr x`, the desktop app's log of the server). A check holding one
//! keeps that reader waiting as long as the check runs, 30 seconds and more
//! for a release host that does not answer; an editor, as long as it is open.

use std::ffi::OsStr;
use std::io;
use std::path::Path;

/// Start `program` with `args` in a process group of its own (on Windows
/// without a console), its standard streams the null device and no other file
/// of this process open in it. It is not waited for.
pub(crate) fn spawn(program: &Path, args: &[&OsStr]) -> io::Result<()> {
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

    pub(super) fn spawn(program: &Path, args: &[&OsStr]) -> io::Result<()> {
        let path = c(program.as_os_str())?;
        let argv = std::iter::once(Ok(path.clone()))
            .chain(args.iter().map(|arg| c(arg)))
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
    use std::ffi::OsStr;
    use std::io;
    use std::os::unix::process::CommandExt;
    use std::path::Path;
    use std::process::{Command, Stdio};

    pub(super) fn spawn(program: &Path, args: &[&OsStr]) -> io::Result<()> {
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
    use std::ffi::OsStr;
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

    /// `arg` as the C runtime's parsing of a command line reads it back: as
    /// it is when it has no space, tab or quote, else in quotes with the
    /// backslashes before a quote, or before the closing one, doubled.
    pub(super) fn quote(arg: &OsStr, line: &mut Vec<u16>) {
        const QUOTE: u16 = b'"' as u16;
        const BACKSLASH: u16 = b'\\' as u16;
        let wide: Vec<u16> = arg.encode_wide().collect();
        let special = |c: &u16| b" \t\"".iter().any(|&b| u16::from(b) == *c);
        if !wide.is_empty() && !wide.iter().any(special) {
            line.extend(wide);
            return;
        }
        line.push(QUOTE);
        let mut backslashes = 0;
        for c in wide {
            match c {
                BACKSLASH => backslashes += 1,
                QUOTE => {
                    line.extend(std::iter::repeat_n(BACKSLASH, backslashes * 2 + 1));
                    line.push(QUOTE);
                    backslashes = 0;
                }
                _ => {
                    line.extend(std::iter::repeat_n(BACKSLASH, backslashes));
                    line.push(c);
                    backslashes = 0;
                }
            }
        }
        line.extend(std::iter::repeat_n(BACKSLASH, backslashes * 2));
        line.push(QUOTE);
    }

    struct List(LPPROC_THREAD_ATTRIBUTE_LIST);

    impl Drop for List {
        fn drop(&mut self) {
            unsafe { DeleteProcThreadAttributeList(self.0) };
        }
    }

    pub(super) fn spawn(program: &Path, args: &[&OsStr]) -> io::Result<()> {
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
        let mut line = Vec::new();
        quote(program.as_os_str(), &mut line);
        for arg in args {
            line.push(u16::from(b' '));
            quote(arg, &mut line);
        }
        line.push(0);
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

#[cfg(all(test, windows))]
mod tests {
    use std::ffi::OsStr;
    use windows_sys::Win32::Foundation::LocalFree;
    use windows_sys::Win32::UI::Shell::CommandLineToArgvW;

    /// Each argument reads back as it was, as Windows parses a command line.
    #[test]
    fn an_argument_reads_back_as_it_was() {
        let args = [
            "--json",
            r"C:\Users\A B\.markitai\config.json",
            "",
            r"C:\dir with space\",
            r#"say "hi""#,
            r#"a\"b"#,
            r"\\server\share\x.json",
        ];
        let mut line = Vec::new();
        super::imp::quote(OsStr::new(r"C:\Windows\System32\notepad.exe"), &mut line);
        for arg in args {
            line.push(u16::from(b' '));
            super::imp::quote(OsStr::new(arg), &mut line);
        }
        line.push(0);
        let mut count = 0;
        let parsed = unsafe { CommandLineToArgvW(line.as_ptr(), &mut count) };
        assert!(!parsed.is_null());
        let read: Vec<String> = (0..count as usize)
            .map(|index| unsafe {
                let arg = *parsed.add(index);
                let length = (0..).take_while(|&at| *arg.add(at) != 0).count();
                String::from_utf16(std::slice::from_raw_parts(arg, length)).unwrap()
            })
            .collect();
        unsafe { LocalFree(parsed.cast()) };
        assert_eq!(read[1..], args);
    }
}
