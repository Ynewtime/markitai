//! Manual operating-system proxy settings, read without side effects.
//!
//! macOS reads `scutil --proxy` by key, independent of its line order; the
//! reference's line-order parser misses the common sorted output. Windows reads
//! the user's Internet Settings registry values. Linux follows the reference
//! desktop scope: KDE (which wins mixed markers) through kreadconfig, GNOME and
//! Unity through a whitelist of `gsettings get` keys that never includes stored
//! credentials. Automatic/PAC/WPAD modes, SOCKS-only settings and authenticated
//! GNOME proxies yield no system proxy. One second bounds all subprocesses.
use super::{MAX_SETTING, SystemProxy, endpoint};
use std::collections::HashMap;
#[cfg(unix)]
use std::time::Duration;
#[cfg(any(unix, test))]
use std::time::Instant;
#[cfg(any(target_os = "linux", target_os = "macos", test))]
use url::Url;

#[cfg(any(unix, test))]
pub(super) trait Reader {
    #[cfg_attr(not(any(target_os = "linux", test)), allow(dead_code))]
    fn available(&self, program: &str) -> bool;
    fn read(&mut self, program: &str, args: &[&str], deadline: Instant) -> Option<String>;
}

pub(super) fn discover(env: &HashMap<String, String>) -> Option<SystemProxy> {
    #[cfg(target_os = "linux")]
    {
        linux(
            env,
            &mut ProcessReader { env },
            Instant::now() + Duration::from_secs(1),
        )
    }
    #[cfg(target_os = "macos")]
    {
        let deadline = Instant::now() + Duration::from_secs(1);
        let raw = ProcessReader { env }.read("scutil", &["--proxy"], deadline)?;
        macos(&raw)
    }
    #[cfg(windows)]
    {
        let _ = env;
        windows(read_registry()?)
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
    {
        let _ = env;
        None
    }
}

/// A host setting (never a URL or credentials) and a decimal port.
#[cfg(any(target_os = "linux", target_os = "macos", test))]
fn host_port(host: &str, port: &str) -> Option<Url> {
    let host = host.trim();
    let port = port.trim().parse::<u16>().ok().filter(|port| *port != 0)?;
    if host.is_empty() || host.contains(['/', '@', '?', '#', '\\']) {
        return None;
    }
    let host = if host.contains(':') && !host.starts_with('[') {
        format!("[{host}]")
    } else {
        host.to_owned()
    };
    endpoint(&format!("http://{host}:{port}"), true).ok()
}

/// KConfig stores proxies either as URLs or as "http://host port".
#[cfg(any(target_os = "linux", test))]
fn manual(raw: &str) -> Option<Url> {
    let raw = raw.trim();
    if raw.is_empty() {
        return None;
    }
    let normalized = match raw.rsplit_once(char::is_whitespace) {
        Some((host, port)) if !port.is_empty() && port.bytes().all(|b| b.is_ascii_digit()) => {
            format!("{}:{port}", host.trim_end())
        }
        _ => raw.to_owned(),
    };
    endpoint(&normalized, true).ok()
}

#[cfg(any(target_os = "linux", test))]
pub(super) fn linux(
    env: &HashMap<String, String>,
    reader: &mut impl Reader,
    deadline: Instant,
) -> Option<SystemProxy> {
    let desktops: Vec<String> = env
        .get("XDG_CURRENT_DESKTOP")
        .map(|value| value.to_ascii_uppercase())
        .unwrap_or_default()
        .split(':')
        .map(str::to_owned)
        .collect();
    if desktops.iter().any(|name| name == "KDE") {
        let tool = ["kreadconfig6", "kreadconfig5"]
            .into_iter()
            .find(|tool| reader.available(tool))?;
        let mut get = |key: &str| {
            reader.read(
                tool,
                &[
                    "--file",
                    "kioslaverc",
                    "--group",
                    "Proxy Settings",
                    "--key",
                    key,
                ],
                deadline,
            )
        };
        if get("ProxyType")?.trim() != "1" {
            return None;
        }
        if !["", "false", "0"].contains(
            &get("ReversedException")?
                .trim()
                .to_ascii_lowercase()
                .as_str(),
        ) {
            return None;
        }
        let endpoint = match manual(&get("httpsProxy")?) {
            Some(endpoint) => endpoint,
            None => manual(&get("httpProxy")?)?,
        };
        let bypass = get("NoProxyFor")?.trim().replace(';', ",");
        return Some(SystemProxy { endpoint, bypass });
    }
    if !desktops
        .iter()
        .any(|name| matches!(name.as_str(), "GNOME" | "UNITY"))
        || !reader.available("gsettings")
    {
        return None;
    }
    // A whitelist of keys: list-recursively would also return stored passwords.
    let mut get =
        |schema: &str, key: &str| reader.read("gsettings", &["get", schema, key], deadline);
    if quoted(&get("org.gnome.system.proxy", "mode")?)? != "manual" {
        return None;
    }
    if get("org.gnome.system.proxy.http", "use-authentication")?.trim() == "true" {
        return None;
    }
    let protocols: &[&str] = match get("org.gnome.system.proxy", "use-same-proxy")?.trim() {
        "true" => &["http"],
        _ => &["https", "http"],
    };
    for protocol in protocols {
        let schema = format!("org.gnome.system.proxy.{protocol}");
        let Some(host) = quoted(&get(&schema, "host")?) else {
            continue;
        };
        if let Some(endpoint) = host_port(&host, &get(&schema, "port")?) {
            let bypass = strings(&get("org.gnome.system.proxy", "ignore-hosts")?)?.join(",");
            return Some(SystemProxy { endpoint, bypass });
        }
    }
    None
}

/// One GVariant string literal starting at `position`.
#[cfg(any(target_os = "linux", test))]
fn string_at(raw: &str, position: &mut usize) -> Option<String> {
    let quote = *raw.as_bytes().get(*position)?;
    if quote != b'\'' && quote != b'"' {
        return None;
    }
    *position += 1;
    let mut result = String::new();
    while *position < raw.len() {
        let ch = raw[*position..].chars().next()?;
        *position += ch.len_utf8();
        if ch as u32 == u32::from(quote) {
            return Some(result);
        }
        if ch != '\\' {
            result.push(ch);
            continue;
        }
        let escaped = raw[*position..].chars().next()?;
        *position += escaped.len_utf8();
        match escaped {
            '\\' | '\'' | '"' => result.push(escaped),
            'n' => result.push('\n'),
            'r' => result.push('\r'),
            't' => result.push('\t'),
            'b' => result.push('\u{8}'),
            'f' => result.push('\u{c}'),
            'u' | 'U' => {
                let length = if escaped == 'u' { 4 } else { 8 };
                let end = position.checked_add(length)?;
                let digits = raw.get(*position..end)?;
                result.push(
                    u32::from_str_radix(digits, 16)
                        .ok()
                        .and_then(char::from_u32)?,
                );
                *position = end;
            }
            _ => return None,
        }
    }
    None
}

#[cfg(any(target_os = "linux", test))]
fn quoted(raw: &str) -> Option<String> {
    let raw = raw.trim();
    let mut position = 0;
    let value = string_at(raw, &mut position)?;
    (position == raw.len()).then_some(value)
}

/// A GVariant `as` value such as `['localhost', '::1']` or `@as []`.
#[cfg(any(target_os = "linux", test))]
fn strings(raw: &str) -> Option<Vec<String>> {
    let raw = raw.trim();
    let raw = raw.strip_prefix("@as ").unwrap_or(raw);
    let raw = raw.strip_prefix('[')?.strip_suffix(']')?.trim();
    let bytes = raw.as_bytes();
    let mut values = Vec::new();
    let mut position = 0;
    while position < raw.len() {
        values.push(string_at(raw, &mut position)?);
        while bytes.get(position).is_some_and(u8::is_ascii_whitespace) {
            position += 1;
        }
        if position == raw.len() {
            break;
        }
        if bytes[position] != b',' {
            return None;
        }
        position += 1;
        while bytes.get(position).is_some_and(u8::is_ascii_whitespace) {
            position += 1;
        }
    }
    Some(values)
}

/// `scutil --proxy` dictionary output, by key rather than by line order.
#[cfg(any(target_os = "macos", test))]
pub(super) fn macos(raw: &str) -> Option<SystemProxy> {
    if raw.len() > MAX_SETTING {
        return None;
    }
    let mut fields = HashMap::new();
    let mut bypass = Vec::new();
    let mut depth = 0_usize;
    let mut exceptions = false;
    for line in raw.lines().map(str::trim) {
        if line.ends_with('{') {
            exceptions = depth == 1 && line.starts_with("ExceptionsList :");
            depth = depth.checked_add(1)?;
            continue;
        }
        if line == "}" {
            depth = depth.checked_sub(1)?;
            if depth <= 1 {
                exceptions = false;
            }
            continue;
        }
        if let Some((key, value)) = line.split_once(" : ") {
            if depth == 1 {
                fields.insert(key, value);
            } else if depth == 2 && exceptions && key.parse::<usize>().is_ok() {
                bypass.push(value);
            }
        }
    }
    if depth != 0 {
        return None;
    }
    let kind = if fields.get("HTTPSEnable") == Some(&"1") {
        "HTTPS"
    } else if fields.get("HTTPEnable") == Some(&"1") {
        "HTTP"
    } else {
        return None;
    };
    let endpoint = host_port(
        fields.get(format!("{kind}Proxy").as_str())?,
        fields.get(format!("{kind}Port").as_str())?,
    )?;
    Some(SystemProxy {
        endpoint,
        bypass: bypass.join(","),
    })
}

#[cfg(any(windows, test))]
pub(super) struct WindowsValues {
    pub(super) enabled: bool,
    pub(super) server: String,
    pub(super) bypass: String,
}

/// `ProxyServer` is either `host:port` or `http=host:port;https=...`; like the
/// reference, the first HTTP(S) entry wins. `ProxyOverride` uses semicolons.
#[cfg(any(windows, test))]
pub(super) fn windows(values: WindowsValues) -> Option<SystemProxy> {
    if !values.enabled || values.server.len() > MAX_SETTING || values.bypass.len() > MAX_SETTING {
        return None;
    }
    let raw = if values.server.contains('=') {
        values.server.split(';').map(str::trim).find_map(|item| {
            item.strip_prefix("https=")
                .or_else(|| item.strip_prefix("http="))
        })?
    } else {
        values.server.trim()
    };
    Some(SystemProxy {
        endpoint: endpoint(raw, true).ok()?,
        bypass: values.bypass.replace(';', ","),
    })
}

#[cfg(windows)]
fn read_registry() -> Option<WindowsValues> {
    read_registry_at(r"Software\Microsoft\Windows\CurrentVersion\Internet Settings")
}

/// Internal path injection for native registry tests. Production reads only
/// the fixed Internet Settings key; no environment override exposes others.
#[cfg(windows)]
pub(super) fn read_registry_at(path: &str) -> Option<WindowsValues> {
    read_registry_at_with(path, || {
        tracing::warn!(
            "Windows automatic proxy (PAC) settings are not supported; set HTTPS_PROXY or HTTP_PROXY for network access"
        );
    })
}

#[cfg(windows)]
fn read_registry_at_with(path: &str, automatic: impl FnOnce()) -> Option<WindowsValues> {
    use windows_sys::Win32::Foundation::ERROR_SUCCESS;
    use windows_sys::Win32::System::Registry::{
        HKEY, HKEY_CURRENT_USER, KEY_QUERY_VALUE, REG_DWORD, REG_SZ, RegCloseKey, RegOpenKeyExW,
        RegQueryValueExW,
    };
    struct Key(HKEY);
    impl Drop for Key {
        fn drop(&mut self) {
            unsafe {
                RegCloseKey(self.0);
            }
        }
    }
    fn wide(value: &str) -> Vec<u16> {
        value.encode_utf16().chain(Some(0)).collect()
    }
    // The reported length bounds every read; REG_SZ need not be terminated.
    fn value(key: &Key, name: &str, expected: u32) -> Option<Vec<u8>> {
        let name = wide(name);
        let (mut kind, mut length) = (0_u32, 0_u32);
        let status = unsafe {
            RegQueryValueExW(
                key.0,
                name.as_ptr(),
                std::ptr::null(),
                &mut kind,
                std::ptr::null_mut(),
                &mut length,
            )
        };
        if status != ERROR_SUCCESS || kind != expected || length as usize > MAX_SETTING {
            return None;
        }
        let mut bytes = vec![0_u8; length as usize];
        let status = unsafe {
            RegQueryValueExW(
                key.0,
                name.as_ptr(),
                std::ptr::null(),
                &mut kind,
                bytes.as_mut_ptr(),
                &mut length,
            )
        };
        if status != ERROR_SUCCESS || kind != expected || length as usize > bytes.len() {
            return None;
        }
        bytes.truncate(length as usize);
        Some(bytes)
    }
    fn text(bytes: Vec<u8>) -> Option<String> {
        let (pairs, rest) = bytes.as_chunks::<2>();
        if !rest.is_empty() {
            return None;
        }
        let mut words: Vec<u16> = pairs.iter().map(|pair| u16::from_le_bytes(*pair)).collect();
        while words.last() == Some(&0) {
            words.pop();
        }
        if words.contains(&0) {
            return None;
        }
        String::from_utf16(&words).ok()
    }
    let path = wide(path);
    let mut handle: HKEY = std::ptr::null_mut();
    if unsafe {
        RegOpenKeyExW(
            HKEY_CURRENT_USER,
            path.as_ptr(),
            0,
            KEY_QUERY_VALUE,
            &mut handle,
        )
    } != ERROR_SUCCESS
    {
        return None;
    }
    let key = Key(handle);
    // Diagnose a configured automatic proxy without executing PAC or exposing
    // its URL, which can carry identity information. Manual settings stay intact.
    if value(&key, "AutoConfigURL", REG_SZ)
        .and_then(text)
        .is_some_and(|url| !url.trim().is_empty())
    {
        automatic();
    }
    let enabled = value(&key, "ProxyEnable", REG_DWORD)?;
    let enabled = u32::from_le_bytes(enabled.try_into().ok()?) != 0;
    if !enabled {
        return None;
    }
    let server = text(value(&key, "ProxyServer", REG_SZ)?)?;
    // A missing override list is an empty list; a malformed one is not trusted.
    let bypass = match value(&key, "ProxyOverride", REG_SZ) {
        Some(bytes) => text(bytes)?,
        None => String::new(),
    };
    Some(WindowsValues {
        enabled,
        server,
        bypass,
    })
}

#[cfg(unix)]
pub(super) struct ProcessReader<'a> {
    pub(super) env: &'a HashMap<String, String>,
}

#[cfg(unix)]
impl ProcessReader<'_> {
    fn program(&self, name: &str) -> Option<std::path::PathBuf> {
        use std::os::unix::fs::PermissionsExt;
        let path = self.env.get("PATH")?;
        std::env::split_paths(std::ffi::OsStr::new(path))
            .map(|directory| directory.join(name))
            .find(|path| {
                std::fs::metadata(path)
                    .is_ok_and(|meta| meta.is_file() && meta.permissions().mode() & 0o111 != 0)
            })
    }
}

#[cfg(unix)]
impl Reader for ProcessReader<'_> {
    fn available(&self, program: &str) -> bool {
        self.program(program).is_some()
    }

    fn read(&mut self, program: &str, args: &[&str], deadline: Instant) -> Option<String> {
        use std::io::{ErrorKind, Read};
        use std::os::fd::AsRawFd;
        use std::os::unix::process::CommandExt;
        use std::process::{Child, Command, Stdio};

        // The group stays registered and addressable until its leader is
        // reaped, so a fatal-signal cleanup and this guard never race a reuse.
        struct Running {
            child: Child,
            group: crate::process_groups::Slot,
            reaped: bool,
        }
        impl Drop for Running {
            fn drop(&mut self) {
                if self.reaped {
                    return;
                }
                if let Ok(pid) = i32::try_from(self.child.id())
                    && pid > 0
                {
                    unsafe {
                        libc::kill(-pid, libc::SIGKILL);
                    }
                }
                self.group.retire();
                let _ = self.child.kill();
                let _ = self.child.wait();
            }
        }
        fn nonblocking(pipe: &impl AsRawFd) -> Option<()> {
            let fd = pipe.as_raw_fd();
            let flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
            (flags >= 0 && unsafe { libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK) } >= 0)
                .then_some(())
        }
        // Some(true) at EOF, Some(false) when no data is ready yet.
        fn drain(pipe: &mut impl Read, output: &mut Vec<u8>) -> Option<bool> {
            let mut buffer = [0_u8; 4096];
            loop {
                match pipe.read(&mut buffer) {
                    Ok(0) => return Some(true),
                    Ok(length) => {
                        if output.len().checked_add(length)? > MAX_SETTING {
                            return None;
                        }
                        output.extend_from_slice(&buffer[..length]);
                    }
                    Err(error) if error.kind() == ErrorKind::Interrupted => {}
                    Err(error) if error.kind() == ErrorKind::WouldBlock => return Some(false),
                    Err(_) => return None,
                }
            }
        }

        if Instant::now() >= deadline {
            return None;
        }
        let path = self.program(program)?;
        let group = crate::process_groups::Slot::reserve()?;
        let mut command = Command::new(path);
        command
            .args(args)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .env_clear()
            .process_group(0);
        // Keep desktop IPC and the real HOME; pass no service credentials.
        for key in [
            "HOME",
            "PATH",
            "USER",
            "LOGNAME",
            "LANG",
            "LC_ALL",
            "XDG_RUNTIME_DIR",
            "XDG_CONFIG_HOME",
            "XDG_CONFIG_DIRS",
            "XDG_DATA_DIRS",
            "DBUS_SESSION_BUS_ADDRESS",
            "DISPLAY",
            "WAYLAND_DISPLAY",
        ] {
            if let Some(value) = self.env.get(key) {
                command.env(key, value);
            }
        }
        let child = command.spawn().ok()?;
        group.publish(&child);
        let mut running = Running {
            child,
            group,
            reaped: false,
        };
        let mut stdout = running.child.stdout.take()?;
        let mut stderr = running.child.stderr.take()?;
        nonblocking(&stdout)?;
        nonblocking(&stderr)?;
        let (mut out, mut err) = (Vec::new(), Vec::new());
        let (mut out_done, mut err_done) = (false, false);
        loop {
            if Instant::now() >= deadline {
                return None;
            }
            out_done = out_done || drain(&mut stdout, &mut out)?;
            err_done = err_done || drain(&mut stderr, &mut err)?;
            if out_done && err_done && running.group.exited(&running.child) {
                break;
            }
            std::thread::sleep(
                Duration::from_millis(2).min(deadline.saturating_duration_since(Instant::now())),
            );
        }
        running.group.retire();
        let status = loop {
            match running.child.wait() {
                Err(error) if error.kind() == ErrorKind::Interrupted => {}
                result => break result.ok()?,
            }
        };
        running.reaped = true;
        if !status.success() {
            return None;
        }
        String::from_utf8(out).ok()
    }
}

#[cfg(all(test, windows))]
mod registry_tests {
    use super::*;
    use windows_sys::Win32::Foundation::ERROR_SUCCESS;
    use windows_sys::Win32::System::Registry::{
        HKEY, HKEY_CURRENT_USER, KEY_ALL_ACCESS, REG_CREATED_NEW_KEY, REG_DWORD,
        REG_OPTION_NON_VOLATILE, REG_SZ, RegCloseKey, RegCreateKeyExW, RegDeleteTreeW,
        RegSetValueExW,
    };

    fn wide(value: &str) -> Vec<u16> {
        value.encode_utf16().chain(Some(0)).collect()
    }
    struct Fixture {
        key: HKEY,
        path: String,
    }
    impl Fixture {
        fn new() -> Self {
            // Each test gets a newly created child; an existing key is never
            // written or removed. No real Internet Settings are inspected.
            let unique = tempfile::tempdir().unwrap();
            let suffix = unique.path().file_name().unwrap().to_string_lossy();
            let path = format!(
                r"Software\MarkitaiValidation\proxy-{}-{suffix}",
                std::process::id()
            );
            let mut key = std::ptr::null_mut();
            let mut disposition = 0;
            assert_eq!(
                unsafe {
                    RegCreateKeyExW(
                        HKEY_CURRENT_USER,
                        wide(&path).as_ptr(),
                        0,
                        std::ptr::null(),
                        REG_OPTION_NON_VOLATILE,
                        KEY_ALL_ACCESS,
                        std::ptr::null(),
                        &mut key,
                        &mut disposition,
                    )
                },
                ERROR_SUCCESS
            );
            if disposition != REG_CREATED_NEW_KEY {
                unsafe {
                    RegCloseKey(key);
                }
                panic!("refuse an existing test registry key");
            }
            Self { key, path }
        }
        fn value(&self, name: &str, kind: u32, bytes: &[u8]) {
            assert_eq!(
                unsafe {
                    RegSetValueExW(
                        self.key,
                        wide(name).as_ptr(),
                        0,
                        kind,
                        bytes.as_ptr(),
                        bytes.len() as u32,
                    )
                },
                ERROR_SUCCESS
            );
        }
        fn text(&self, name: &str, text: &str) {
            let bytes: Vec<u8> = wide(text).iter().flat_map(|c| c.to_le_bytes()).collect();
            self.value(name, REG_SZ, &bytes);
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            unsafe {
                RegCloseKey(self.key);
                RegDeleteTreeW(HKEY_CURRENT_USER, wide(&self.path).as_ptr());
            }
        }
    }

    #[test]
    fn injected_registry_reads_manual_settings_and_refuses_disabled_or_malformed_values() {
        let fixture = Fixture::new();
        fixture.value("ProxyEnable", REG_DWORD, &1u32.to_le_bytes());
        fixture.text(
            "ProxyServer",
            "http=proxy.example.test:8080;https=secure.example.test:8443",
        );
        fixture.text("ProxyOverride", "<local>;*.example.test");
        let found = windows(read_registry_at(&fixture.path).unwrap()).unwrap();
        assert_eq!(found.endpoint.as_str(), "http://proxy.example.test:8080/");
        assert_eq!(found.bypass, "<local>,*.example.test");
        fixture.value("ProxyEnable", REG_DWORD, &0u32.to_le_bytes());
        fixture.text("AutoConfigURL", "https://pac.example.test/proxy.pac");
        let mut warned = false;
        assert!(read_registry_at_with(&fixture.path, || warned = true).is_none());
        assert!(warned);
        for value in ["", " "] {
            fixture.text("AutoConfigURL", value);
            warned = false;
            assert!(read_registry_at_with(&fixture.path, || warned = true).is_none());
            assert!(!warned);
        }
        fixture.value("AutoConfigURL", REG_SZ, &[b'x', 0, b'y']);
        warned = false;
        assert!(read_registry_at_with(&fixture.path, || warned = true).is_none());
        assert!(!warned);
        fixture.value("ProxyEnable", REG_DWORD, &1u32.to_le_bytes());
        fixture.value("ProxyServer", REG_SZ, &[b'x', 0, b'y']);
        assert!(read_registry_at(&fixture.path).is_none());
        fixture.value("ProxyServer", REG_SZ, &vec![0; MAX_SETTING + 2]);
        assert!(read_registry_at(&fixture.path).is_none());
        fixture.text("ProxyServer", "not-a-proxy/with-path");
        assert!(windows(read_registry_at(&fixture.path).unwrap()).is_none());
    }
}
