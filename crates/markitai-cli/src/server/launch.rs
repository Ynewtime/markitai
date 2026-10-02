use super::SettingsSource;
use markitai_core::config;
use serde_json::Value;
use std::{
    net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::{Duration, Instant},
};

pub(crate) fn settings_source(
    explicit: Option<&Path>,
    overrides: Option<Value>,
) -> Result<SettingsSource, String> {
    let cwd = std::env::current_dir().map_err(|error| error.to_string())?;
    let environment = config::environment();
    let configured = environment
        .get("MARKITAI_CONFIG")
        .filter(|value| !value.is_empty());
    let (path, origin) = select_source(explicit, configured.map(Path::new), &cwd, &config::home());
    let path = config::expand_home(&path);
    Ok(SettingsSource {
        path: if path.is_absolute() {
            path
        } else {
            cwd.join(path)
        },
        origin: origin.into(),
        overrides,
    })
}

fn select_source(
    explicit: Option<&Path>,
    configured: Option<&Path>,
    cwd: &Path,
    home: &Path,
) -> (PathBuf, &'static str) {
    if let Some(path) = explicit {
        (path.into(), "explicit")
    } else if let Some(path) = configured {
        (path.into(), "environment")
    } else if cwd.join("markitai.json").exists() {
        (cwd.join("markitai.json"), "project")
    } else if home.join("config.json").exists() {
        (home.join("config.json"), "user")
    } else {
        (home.join("config.json"), "default")
    }
}

pub(super) fn browser_url(mut address: SocketAddr, token: Option<&str>) -> String {
    if address.ip().is_unspecified() {
        address.set_ip(match address.ip() {
            IpAddr::V4(_) => Ipv4Addr::LOCALHOST.into(),
            IpAddr::V6(_) => Ipv6Addr::LOCALHOST.into(),
        });
    }
    address_url(address, token)
}

/// The address exactly as given, with the token (when there is one) in the fragment.
pub(super) fn address_url(address: SocketAddr, token: Option<&str>) -> String {
    let mut url = format!("http://{address}/");
    if let Some(token) = token {
        url.push('#');
        url.push_str(
            &url::form_urlencoded::Serializer::new(String::new())
                .append_pair("token", token)
                .finish(),
        );
    }
    url
}

pub(crate) fn open_config(path: &Path) -> Result<(), String> {
    open_target(path.as_os_str())
}

pub(super) fn open_browser(url: &str) -> Result<(), String> {
    open_target(std::ffi::OsStr::new(url))
}

fn open_target(target: &std::ffi::OsStr) -> Result<(), String> {
    #[cfg(target_os = "macos")]
    let mut command = Command::new("/usr/bin/open");
    #[cfg(target_os = "windows")]
    let mut command = {
        let mut command = Command::new("rundll32.exe");
        command.arg("url.dll,FileProtocolHandler");
        command
    };
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    let mut command = Command::new("xdg-open");
    let mut child = command
        .arg(target)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|_| "system opener could not be started".to_owned())?;
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                return if status.success() {
                    Ok(())
                } else {
                    Err("system opener rejected the request".into())
                };
            }
            Err(_) => return Err("system opener status could not be read".into()),
            Ok(None) if Instant::now() < deadline => std::thread::sleep(Duration::from_millis(20)),
            Ok(None) => {
                std::thread::spawn(move || {
                    let _ = child.wait();
                });
                return Ok(());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn browser_address_is_local_and_fragment_token_is_encoded() {
        assert_eq!(
            browser_url("0.0.0.0:1234".parse().unwrap(), None),
            "http://127.0.0.1:1234/"
        );
        let url = browser_url("[::]:1234".parse().unwrap(), Some("a& b#?/"));
        let parsed = url::Url::parse(&url).unwrap();
        assert_eq!(parsed.host_str(), Some("[::1]"));
        assert!(parsed.query().is_none());
        let pairs: Vec<_> =
            url::form_urlencoded::parse(parsed.fragment().unwrap().as_bytes()).collect();
        assert_eq!(pairs, [("token".into(), "a& b#?/".into())]);
        assert_eq!(
            browser_url("192.0.2.1:1234".parse().unwrap(), None),
            "http://192.0.2.1:1234/"
        );
    }

    #[test]
    fn configuration_destination_retains_precedence_and_default_creation_path() {
        let temp = tempfile::tempdir().unwrap();
        let cwd = temp.path().join("project");
        let home = temp.path().join("state");
        std::fs::create_dir_all(&cwd).unwrap();
        std::fs::create_dir_all(&home).unwrap();
        assert_eq!(
            select_source(None, None, &cwd, &home),
            (home.join("config.json"), "default")
        );
        std::fs::write(home.join("config.json"), "{}").unwrap();
        assert_eq!(select_source(None, None, &cwd, &home).1, "user");
        std::fs::write(cwd.join("markitai.json"), "{}").unwrap();
        assert_eq!(select_source(None, None, &cwd, &home).1, "project");
        assert_eq!(
            select_source(None, Some(Path::new("missing.json")), &cwd, &home),
            (PathBuf::from("missing.json"), "environment")
        );
        assert_eq!(
            select_source(
                Some(Path::new("explicit.json")),
                Some(Path::new("missing.json")),
                &cwd,
                &home
            ),
            (PathBuf::from("explicit.json"), "explicit")
        );
    }
}
