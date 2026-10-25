//! The requests: which release is the latest, its SHA256SUMS, its notes and
//! its archive. Every request names itself `User-Agent: markitai` and carries
//! nothing else about the machine.
//!
//! The daily check asks GitHub for `<base>/latest/download/SHA256SUMS` and
//! does not follow the redirect: its `Location` names the release
//! (`…/download/v2.2.0/SHA256SUMS`), so nothing is downloaded. A mirror that
//! answers with the file itself works too.
use super::version::Version;
use reqwest::blocking::{Client, Response};
use std::io::Read;
use std::time::Duration;
use url::Url;

pub(crate) const DEFAULT_BASE: &str = "https://github.com/Ynewtime/markitai/releases";
pub(crate) const USER_AGENT: &str = "markitai";
/// SHA256SUMS lists a dozen assets; anything this big is not one.
const SUMS_LIMIT: u64 = 64 * 1024;

#[derive(Clone, Copy, Debug)]
pub(crate) struct Timeouts {
    pub(crate) connect: Duration,
    /// The longest wait for the next bytes (reqwest's blocking timeout is
    /// per read, so a slow but moving download never times out).
    pub(crate) idle: Duration,
}

/// The notice's check: never more than about two seconds of anyone's time.
pub(crate) const NOTICE: Timeouts = Timeouts {
    connect: Duration::from_secs(2),
    idle: Duration::from_secs(2),
};
/// `markitai update` and the server's download.
pub(crate) const COMMAND: Timeouts = Timeouts {
    connect: Duration::from_secs(10),
    idle: Duration::from_secs(30),
};

/// Where releases come from: `<base>/latest/download/…`,
/// `<base>/download/v<version>/…` and `<base>/tag/v<version>`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Base(Url);

impl Base {
    /// `MARKITAI_DOWNLOAD_URL` (the setup scripts' mirror variable), else
    /// the base the install recorded, else GitHub. Only HTTPS, or HTTP to
    /// this computer (a local mirror, the tests' fake release server).
    pub(crate) fn resolve(recorded: Option<&str>) -> Result<Self, String> {
        let env = markitai_core::config::environment();
        let chosen = env
            .get("MARKITAI_DOWNLOAD_URL")
            .map(String::as_str)
            .filter(|value| !value.trim().is_empty())
            .or(recorded.filter(|value| !value.trim().is_empty()))
            .unwrap_or(DEFAULT_BASE);
        Self::parse(chosen)
    }

    pub(crate) fn parse(text: &str) -> Result<Self, String> {
        let trimmed = text.trim().trim_end_matches('/');
        let url = Url::parse(trimmed).map_err(|_| format!("not a URL: {trimmed}"))?;
        if url.query().is_some() || url.fragment().is_some() || !url.username().is_empty() {
            return Err(format!("release base must be a plain address: {trimmed}"));
        }
        match url.scheme() {
            "https" => {}
            "http" if loopback(&url) => {}
            _ => return Err(format!("release base is not HTTPS: {trimmed}")),
        }
        if url.host_str().is_none() {
            return Err(format!("not a URL: {trimmed}"));
        }
        Ok(Self(url))
    }

    pub(crate) fn as_str(&self) -> &str {
        self.0.as_str().trim_end_matches('/')
    }

    /// The host messages name (`github.com`).
    pub(crate) fn host(&self) -> String {
        self.0.host_str().unwrap_or_default().to_owned()
    }

    fn latest_sums(&self) -> String {
        format!("{}/latest/download/SHA256SUMS", self.as_str())
    }

    pub(crate) fn asset(&self, version: &Version, name: &str) -> String {
        format!("{}/download/v{version}/{name}", self.as_str())
    }

    pub(crate) fn notes_url(&self, version: &Version) -> String {
        format!("{}/tag/v{version}", self.as_str())
    }
}

fn loopback(url: &Url) -> bool {
    match url.host() {
        Some(url::Host::Domain(name)) => name.eq_ignore_ascii_case("localhost"),
        Some(url::Host::Ipv4(ip)) => ip.is_loopback(),
        Some(url::Host::Ipv6(ip)) => ip.is_loopback(),
        None => false,
    }
}

/// Why a request failed, in the words the messages use.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Failure {
    /// No answer: `reason` is `timeout`, `refused` or `unreachable`.
    Offline { host: String, reason: &'static str },
    /// An answer that is not the file.
    Status { host: String, code: u16 },
    /// The release does not exist (or has no such asset).
    Missing { what: String },
    /// An answer that cannot be read.
    Malformed { what: String },
    /// The proxy settings or the TLS setup are unusable.
    Setup(String),
}

impl Failure {
    /// The API's error code: every failed request reads as offline.
    pub(crate) fn code(&self) -> &'static str {
        "offline"
    }
}

fn classify(error: &reqwest::Error, host: &str) -> Failure {
    let reason = if error.is_timeout() {
        "timeout"
    } else if error.is_connect() && error_chain_mentions(error, "refused") {
        "refused"
    } else {
        "unreachable"
    };
    Failure::Offline {
        host: host.to_owned(),
        reason,
    }
}

fn error_chain_mentions(error: &(dyn std::error::Error + 'static), word: &str) -> bool {
    let mut current: Option<&(dyn std::error::Error + 'static)> = Some(error);
    while let Some(error) = current {
        if error.to_string().to_ascii_lowercase().contains(word) {
            return true;
        }
        current = error.source();
    }
    false
}

/// A client with these timeouts. `follow` follows up to five redirects that
/// stay on HTTPS (GitHub sends assets to its download host), or on HTTP to
/// this computer.
pub(crate) fn client(timeouts: Timeouts, follow: bool) -> Result<Client, Failure> {
    let redirects = if follow {
        reqwest::redirect::Policy::custom(|attempt| {
            if attempt.previous().len() >= 5 {
                attempt.error("too many redirects")
            } else if attempt.url().scheme() == "https"
                || (attempt.url().scheme() == "http" && loopback(attempt.url()))
            {
                attempt.follow()
            } else {
                attempt.error("redirect away from HTTPS")
            }
        })
    } else {
        reqwest::redirect::Policy::none()
    };
    markitai_core::http_client_builder()
        .map_err(|error| Failure::Setup(error.to_string()))?
        .connect_timeout(timeouts.connect)
        .timeout(timeouts.idle)
        .redirect(redirects)
        .user_agent(USER_AGENT)
        .build()
        .map_err(|error| Failure::Setup(error.to_string()))
}

fn get(client: &Client, url: &str, host: &str) -> Result<Response, Failure> {
    client
        .get(url)
        .send()
        .map_err(|error| classify(&error, host))
}

fn read_limited(response: Response, limit: u64, host: &str) -> Result<Vec<u8>, Failure> {
    let mut bytes = Vec::new();
    response
        .take(limit + 1)
        .read_to_end(&mut bytes)
        .map_err(|_| Failure::Offline {
            host: host.to_owned(),
            reason: "timeout",
        })?;
    if bytes.len() as u64 > limit {
        return Err(Failure::Malformed {
            what: "SHA256SUMS".into(),
        });
    }
    Ok(bytes)
}

/// The latest release's version.
pub(crate) fn latest(base: &Base, timeouts: Timeouts) -> Result<Version, Failure> {
    let host = base.host();
    let client = client(timeouts, false)?;
    let response = get(&client, &base.latest_sums(), &host)?;
    let status = response.status();
    if status.is_redirection() {
        let location = response
            .headers()
            .get(reqwest::header::LOCATION)
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default();
        return version_in_location(location).ok_or(Failure::Malformed {
            what: "the latest release's address".into(),
        });
    }
    if status.as_u16() == 404 {
        return Err(Failure::Missing {
            what: "latest release".into(),
        });
    }
    if !status.is_success() {
        return Err(Failure::Status {
            host,
            code: status.as_u16(),
        });
    }
    let bytes = read_limited(response, SUMS_LIMIT, &host)?;
    newest_in_sums(&String::from_utf8_lossy(&bytes)).ok_or(Failure::Malformed {
        what: "SHA256SUMS".into(),
    })
}

/// The release a `latest` redirect points at: the `v<version>` segment after
/// `download` (or `tag`).
pub(crate) fn version_in_location(location: &str) -> Option<Version> {
    let path = location.split(['?', '#']).next()?;
    let segments: Vec<&str> = path.split('/').collect();
    segments.windows(2).find_map(|pair| {
        (matches!(pair[0], "download" | "tag"))
            .then(|| pair[1].strip_prefix('v'))
            .flatten()
            .and_then(Version::parse)
    })
}

/// The newest `markitai-v<version>-…` archive a SHA256SUMS lists.
fn newest_in_sums(sums: &str) -> Option<Version> {
    sums.lines()
        .filter_map(|line| {
            let (_, name) = line.split_once("  ")?;
            let rest = name.trim().strip_prefix("markitai-v")?;
            let version = rest
                .split_once("-x86_64")
                .or(rest.split_once("-aarch64"))?
                .0;
            Version::parse(version)
        })
        .max()
}

/// A release's SHA256SUMS.
pub(crate) fn sums(base: &Base, version: &Version, timeouts: Timeouts) -> Result<String, Failure> {
    let host = base.host();
    let client = client(timeouts, true)?;
    let response = get(&client, &base.asset(version, "SHA256SUMS"), &host)?;
    match response.status().as_u16() {
        200 => {}
        404 => {
            return Err(Failure::Missing {
                what: format!("v{version}"),
            });
        }
        code => return Err(Failure::Status { host, code }),
    }
    let bytes = read_limited(response, SUMS_LIMIT, &host)?;
    String::from_utf8(bytes).map_err(|_| Failure::Malformed {
        what: "SHA256SUMS".into(),
    })
}

/// The archive line for `target` in a SHA256SUMS: its digest and name.
pub(crate) fn archive_line(
    sums: &str,
    version: &Version,
    target: &str,
) -> Option<(String, String)> {
    let extension = if target.contains("windows") {
        "zip"
    } else {
        "tar.gz"
    };
    let wanted = format!("markitai-v{version}-{target}.{extension}");
    sums.lines().find_map(|line| {
        let (digest, name) = line.trim_end_matches('\r').split_once("  ")?;
        let digest = digest.trim();
        (name == wanted
            && digest.len() == 64
            && digest.bytes().all(|byte| byte.is_ascii_hexdigit()))
        .then(|| (digest.to_ascii_lowercase(), wanted.clone()))
    })
}

/// The release's one-line description, from its `RELEASE-NOTES.md` (or
/// `RELEASE-NOTES.zh.md`); `None` for a release without one (the link alone
/// is shown).
pub(crate) fn lead(base: &Base, version: &Version, timeouts: Timeouts, zh: bool) -> Option<String> {
    let client = client(timeouts, true).ok()?;
    let name = if zh {
        "RELEASE-NOTES.zh.md"
    } else {
        "RELEASE-NOTES.md"
    };
    let response = get(&client, &base.asset(version, name), &base.host()).ok()?;
    if !response.status().is_success() {
        return None;
    }
    let mut bytes = Vec::new();
    response
        .take(super::notes::LIMIT)
        .read_to_end(&mut bytes)
        .ok()?;
    super::notes::lead(&String::from_utf8_lossy(&bytes))
}

/// The English and the Chinese description; the Chinese one is asked for
/// only when the release has an English one (releases before 2.2.0 have
/// neither).
pub(crate) fn leads(
    base: &Base,
    version: &Version,
    timeouts: Timeouts,
) -> (Option<String>, Option<String>) {
    let lead = lead(base, version, timeouts, false);
    let zh = lead
        .as_ref()
        .and_then(|_| self::lead(base, version, timeouts, true));
    (lead, zh)
}

/// Starts the archive download; the caller reads the body.
pub(crate) fn open(base: &Base, url: &str, timeouts: Timeouts) -> Result<Response, Failure> {
    let host = base.host();
    let response = get(&client(timeouts, true)?, url, &host)?;
    match response.status().as_u16() {
        200 => Ok(response),
        404 => Err(Failure::Missing {
            what: url.rsplit('/').next().unwrap_or(url).to_owned(),
        }),
        code => Err(Failure::Status { host, code }),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::io::{BufRead, BufReader, Write};
    use std::net::TcpListener;
    use std::sync::{Arc, Mutex};

    /// One recorded request: the request line and its headers, lowercased names.
    pub(crate) type Seen = Arc<Mutex<Vec<(String, Vec<(String, String)>)>>>;

    /// A path, the status, extra headers and the body it is answered with.
    pub(crate) type Route = (&'static str, u16, Vec<(&'static str, String)>, Vec<u8>);

    /// A tiny HTTP server answering each path with a status, headers and body.
    pub(crate) fn serve(routes: Vec<Route>) -> (String, Seen) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let seen: Seen = Arc::default();
        let record = seen.clone();
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { break };
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut line = String::new();
                if reader.read_line(&mut line).is_err() {
                    continue;
                }
                let mut headers = Vec::new();
                loop {
                    let mut header = String::new();
                    if reader.read_line(&mut header).is_err() || header.trim().is_empty() {
                        break;
                    }
                    if let Some((name, value)) = header.trim().split_once(':') {
                        headers.push((name.trim().to_ascii_lowercase(), value.trim().to_owned()));
                    }
                }
                record
                    .lock()
                    .unwrap()
                    .push((line.trim().to_owned(), headers));
                let path = line.split_whitespace().nth(1).unwrap_or("").to_owned();
                let route = routes.iter().find(|(route, ..)| *route == path);
                let (status, extra, body) = match route {
                    Some((_, status, extra, body)) => (*status, extra.clone(), body.clone()),
                    None => (404, Vec::new(), b"missing".to_vec()),
                };
                let mut head = format!(
                    "HTTP/1.1 {status} X\r\ncontent-length: {}\r\nconnection: close\r\n",
                    body.len()
                );
                for (name, value) in extra {
                    head.push_str(&format!("{name}: {value}\r\n"));
                }
                head.push_str("\r\n");
                let _ = stream.write_all(head.as_bytes());
                let _ = stream.write_all(&body);
            }
        });
        (format!("http://{address}/releases"), seen)
    }

    #[test]
    fn the_check_reads_the_redirect_without_following_it_and_names_itself_only_markitai() {
        let (base, seen) = serve(vec![(
            "/releases/latest/download/SHA256SUMS",
            302,
            vec![(
                "location",
                "https://github.com/Ynewtime/markitai/releases/download/v2.2.0/SHA256SUMS".into(),
            )],
            Vec::new(),
        )]);
        let base = Base::parse(&base).unwrap();
        assert_eq!(
            latest(&base, NOTICE).unwrap(),
            Version::parse("2.2.0").unwrap()
        );
        let seen = seen.lock().unwrap();
        // One request: the redirect target (github.com) was never fetched.
        assert_eq!(seen.len(), 1, "{seen:?}");
        let (line, headers) = &seen[0];
        assert_eq!(line, "GET /releases/latest/download/SHA256SUMS HTTP/1.1");
        let agent: Vec<_> = headers
            .iter()
            .filter(|(name, _)| name == "user-agent")
            .collect();
        assert_eq!(agent, [&("user-agent".to_owned(), "markitai".to_owned())]);
        assert!(
            !headers
                .iter()
                .any(|(name, _)| name == "cookie" || name == "referer")
        );
    }

    #[test]
    fn a_mirror_that_serves_the_file_and_failures_are_told_apart() {
        let sums = format!(
            "{}  markitai-v2.1.0-x86_64-unknown-linux-gnu.tar.gz\n{}  markitai-v2.10.0-aarch64-apple-darwin.tar.gz\n{}  markitai-2.10.0-cp310-abi3-win_amd64.whl\n",
            "a".repeat(64),
            "b".repeat(64),
            "c".repeat(64)
        );
        let (base, _) = serve(vec![
            (
                "/releases/latest/download/SHA256SUMS",
                200,
                vec![],
                sums.into_bytes(),
            ),
            (
                "/broken/latest/download/SHA256SUMS",
                500,
                vec![],
                Vec::new(),
            ),
        ]);
        assert_eq!(
            latest(&Base::parse(&base).unwrap(), NOTICE)
                .unwrap()
                .to_string(),
            "2.10.0"
        );
        let broken = Base::parse(&base.replace("/releases", "/broken")).unwrap();
        assert!(matches!(
            latest(&broken, NOTICE),
            Err(Failure::Status { code: 500, .. })
        ));
        let missing = Base::parse(&base.replace("/releases", "/none")).unwrap();
        assert!(matches!(
            latest(&missing, NOTICE),
            Err(Failure::Missing { .. })
        ));
    }

    #[test]
    fn a_silent_server_times_out_within_the_notice_cap() {
        // Accepts, never answers.
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            let mut held = Vec::new();
            for stream in listener.incoming() {
                held.push(stream);
            }
        });
        let base = Base::parse(&format!("http://{address}/releases")).unwrap();
        let started = std::time::Instant::now();
        let quick = Timeouts {
            connect: Duration::from_millis(300),
            idle: Duration::from_millis(300),
        };
        assert!(matches!(
            latest(&base, quick),
            Err(Failure::Offline {
                reason: "timeout",
                ..
            })
        ));
        assert!(
            started.elapsed() < Duration::from_secs(2),
            "{:?}",
            started.elapsed()
        );
        // Nothing listens: refused, not a timeout.
        let closed = TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap();
        let base = Base::parse(&format!("http://{closed}/releases")).unwrap();
        assert!(matches!(latest(&base, quick), Err(Failure::Offline { .. })));
    }

    #[test]
    fn bases_are_https_or_this_computer_and_locations_name_the_release() {
        assert!(Base::parse("https://github.com/Ynewtime/markitai/releases/").is_ok());
        assert!(Base::parse("http://127.0.0.1:8080/r").is_ok());
        assert!(Base::parse("http://localhost/r").is_ok());
        assert!(Base::parse("http://[::1]/r").is_ok());
        for bad in [
            "http://mirror.example/r",
            "ftp://x/r",
            "https://user:pw@x/r",
            "https://x/r?token=1",
            "nonsense",
        ] {
            assert!(Base::parse(bad).is_err(), "{bad}");
        }
        let base = Base::parse(DEFAULT_BASE).unwrap();
        let version = Version::parse("2.2.0").unwrap();
        assert_eq!(
            base.notes_url(&version),
            "https://github.com/Ynewtime/markitai/releases/tag/v2.2.0"
        );
        assert_eq!(base.host(), "github.com");
        for (location, expected) in [
            (
                "/Ynewtime/markitai/releases/download/v2.2.0/SHA256SUMS",
                Some("2.2.0"),
            ),
            (
                "https://github.com/o/r/releases/tag/v2.3.0-rc.1",
                Some("2.3.0-rc.1"),
            ),
            ("https://github.com/o/r/releases", None),
            ("", None),
        ] {
            assert_eq!(
                version_in_location(location)
                    .map(|v| v.to_string())
                    .as_deref(),
                expected,
                "{location}"
            );
        }
        let sums = format!(
            "{}  markitai-v2.2.0-x86_64-pc-windows-msvc.zip\n{}  markitai-v2.2.0-x86_64-unknown-linux-gnu.tar.gz\n",
            "A".repeat(64),
            "b".repeat(64)
        );
        assert_eq!(
            archive_line(&sums, &version, "x86_64-pc-windows-msvc"),
            Some((
                "a".repeat(64),
                "markitai-v2.2.0-x86_64-pc-windows-msvc.zip".into()
            ))
        );
        assert_eq!(
            archive_line(&sums, &version, "x86_64-unknown-linux-gnu")
                .unwrap()
                .1,
            "markitai-v2.2.0-x86_64-unknown-linux-gnu.tar.gz"
        );
        assert_eq!(archive_line(&sums, &version, "aarch64-apple-darwin"), None);
    }
}
