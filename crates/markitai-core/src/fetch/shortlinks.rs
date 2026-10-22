//! X's `t.co` shortlinks, replaced by where they lead.
//!
//! X wraps every link of a post in `https://t.co/<code>`. A shortlink left in
//! an X page's Markdown is resolved with one `HEAD` request to `t.co`, which
//! answers with a redirect and no body; the redirect is not followed. Bounded:
//! at most [`MAX_LINKS`] distinct links a page, [`CONCURRENCY`] at a time,
//! each within [`TIMEOUT`] and all within [`BUDGET`]. Answers, failures too,
//! are kept for the process, so a page converted again costs nothing.

use crate::{Result, public_net};
use std::collections::{HashMap, VecDeque};
use std::sync::{LazyLock, Mutex};
use std::time::{Duration, Instant};
use url::Url;

const MAX_LINKS: usize = 20;
const CONCURRENCY: usize = 4;
const TIMEOUT: Duration = Duration::from_secs(5);
const BUDGET: Duration = Duration::from_secs(10);
const CACHE_SIZE: usize = 1024;

static SHORTLINK: LazyLock<regex::Regex> =
    LazyLock::new(|| regex::Regex::new(r"https?://t\.co/([A-Za-z0-9]{4,32})\b").unwrap());

/// Shortlink code to destination (empty when it could not be resolved), and
/// the codes in the order they were remembered.
#[derive(Default)]
struct Resolved {
    destinations: HashMap<String, String>,
    order: VecDeque<String>,
}

static RESOLVED: LazyLock<Mutex<Resolved>> = LazyLock::new(Mutex::default);

/// Whether `url` is a page on X, whose links are shortlinks.
pub(super) fn is_x_page(url: &Url) -> bool {
    url.host_str().is_some_and(|host| {
        matches!(
            host.to_ascii_lowercase().as_str(),
            "x.com"
                | "www.x.com"
                | "mobile.x.com"
                | "twitter.com"
                | "www.twitter.com"
                | "mobile.twitter.com"
        )
    })
}

/// `markdown` with each resolvable `t.co` link replaced by its destination,
/// or `None` when it has none. Requests go out as the current fetch's
/// restriction allows ([`public_net::level`]).
pub(super) fn expand(markdown: &str) -> Option<String> {
    // Unit tests never reach the real t.co (see `expand_with`).
    if cfg!(test) {
        return None;
    }
    let client = client().ok()?;
    expand_with(markdown, &|code| {
        resolve(&client, &format!("https://t.co/{code}"))
    })
}

fn client() -> Result<reqwest::blocking::Client> {
    let builder = crate::trust::client_builder()
        .timeout(TIMEOUT)
        .connect_timeout(TIMEOUT)
        .user_agent(super::USER_AGENT);
    let builder = match public_net::level() {
        public_net::Level::PublicOnly => public_net::client(builder),
        _ => crate::proxy::http(builder.redirect(reqwest::redirect::Policy::none()))?,
    };
    builder.build().map_err(super::request_error)
}

/// Where the shortlink at `address` leads, or an empty string: its redirect
/// to an http(s) address that is not `t.co` again.
fn resolve(client: &reqwest::blocking::Client, address: &str) -> String {
    let Ok(response) = client.head(address).send() else {
        return String::new();
    };
    if !response.status().is_redirection() {
        return String::new();
    }
    let destination = response
        .headers()
        .get(reqwest::header::LOCATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|location| Url::parse(location).ok())
        .filter(|url| {
            matches!(url.scheme(), "http" | "https")
                && url
                    .host_str()
                    .is_some_and(|host| !host.eq_ignore_ascii_case("t.co"))
        });
    // Parentheses would end a Markdown link early.
    destination.map_or_else(String::new, |url| {
        url.as_str().replace('(', "%28").replace(')', "%29")
    })
}

fn expand_with(markdown: &str, resolve: &(dyn Fn(&str) -> String + Sync)) -> Option<String> {
    let mut codes: Vec<&str> = Vec::new();
    for found in SHORTLINK.captures_iter(markdown) {
        let code = found.get(1).unwrap().as_str();
        if !codes.contains(&code) {
            codes.push(code);
        }
    }
    codes.truncate(MAX_LINKS);
    if codes.is_empty() {
        return None;
    }
    let pending: Mutex<Vec<&str>> = Mutex::new({
        let resolved = RESOLVED.lock().unwrap_or_else(|error| error.into_inner());
        codes
            .iter()
            .copied()
            .filter(|code| !resolved.destinations.contains_key(*code))
            .rev()
            .collect()
    });
    let deadline = Instant::now() + BUDGET;
    std::thread::scope(|scope| {
        for _ in 0..CONCURRENCY {
            scope.spawn(|| {
                loop {
                    if Instant::now() >= deadline {
                        return;
                    }
                    let Some(code) = pending
                        .lock()
                        .unwrap_or_else(|error| error.into_inner())
                        .pop()
                    else {
                        return;
                    };
                    let destination = resolve(code);
                    remember(code, destination);
                }
            });
        }
    });
    let resolved = RESOLVED.lock().unwrap_or_else(|error| error.into_inner());
    let expanded = SHORTLINK.replace_all(markdown, |found: &regex::Captures<'_>| {
        let code = &found[1];
        match resolved.destinations.get(code) {
            Some(destination) if codes.contains(&code) && !destination.is_empty() => {
                destination.clone()
            }
            _ => found[0].to_owned(),
        }
    });
    (expanded != markdown).then(|| expanded.into_owned())
}

fn remember(code: &str, destination: String) {
    let mut resolved = RESOLVED.lock().unwrap_or_else(|error| error.into_inner());
    if resolved
        .destinations
        .insert(code.to_owned(), destination)
        .is_none()
    {
        resolved.order.push_back(code.to_owned());
    }
    while resolved.order.len() > CACHE_SIZE {
        if let Some(oldest) = resolved.order.pop_front() {
            resolved.destinations.remove(&oldest);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn x_pages_are_recognized_and_others_are_not() {
        for page in [
            "https://x.com/ada/status/1",
            "https://mobile.twitter.com/ada/status/1",
            "https://WWW.X.COM/ada",
        ] {
            assert!(is_x_page(&Url::parse(page).unwrap()), "{page}");
        }
        for page in [
            "https://x.com.example/a",
            "https://t.co/abcd",
            "https://example.com/",
        ] {
            assert!(!is_x_page(&Url::parse(page).unwrap()), "{page}");
        }
    }

    #[test]
    fn shortlinks_are_replaced_once_each_and_failures_are_remembered() {
        let asked = AtomicUsize::new(0);
        let resolve = |code: &str| {
            asked.fetch_add(1, Ordering::SeqCst);
            match code {
                "Zr0Aa1" => "https://example.org/read?id=1".into(),
                _ => String::new(),
            }
        };
        let markdown = "See [https://t.co/Zr0Aa1](https://t.co/Zr0Aa1) and http://t.co/Zr0Aa1.\n\nAlso https://t.co/Fail01 and https://t.co/Zr0Aa1x9.";
        let expanded = expand_with(markdown, &resolve).unwrap();
        assert_eq!(
            expanded,
            "See [https://example.org/read?id=1](https://example.org/read?id=1) and https://example.org/read?id=1.\n\nAlso https://t.co/Fail01 and https://t.co/Zr0Aa1x9."
        );
        assert_eq!(asked.load(Ordering::SeqCst), 3);
        // Known answers, the failure too, are not asked again.
        assert!(expand_with("https://t.co/Fail01", &resolve).is_none());
        assert_eq!(
            expand_with("https://t.co/Zr0Aa1", &resolve).unwrap(),
            "https://example.org/read?id=1"
        );
        assert_eq!(asked.load(Ordering::SeqCst), 3);
        assert!(expand_with("No shortlinks here.", &resolve).is_none());
    }

    #[test]
    fn at_most_twenty_links_a_page_are_asked_for() {
        let asked = AtomicUsize::new(0);
        let resolve = |code: &str| {
            asked.fetch_add(1, Ordering::SeqCst);
            format!("https://example.org/{code}")
        };
        let markdown = (0..25)
            .map(|index| format!("https://t.co/Cap{index:03}"))
            .collect::<Vec<_>>()
            .join("\n");
        let expanded = expand_with(&markdown, &resolve).unwrap();
        assert_eq!(asked.load(Ordering::SeqCst), 20);
        assert!(expanded.contains("https://example.org/Cap019"));
        assert!(expanded.contains("https://t.co/Cap020"));
    }

    /// One HEAD request, its redirect read and not followed; a redirect back
    /// to `t.co`, a non-http destination or an answer without one resolve to
    /// nothing.
    #[test]
    fn a_shortlink_resolves_to_its_redirect_without_following_it() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let server = std::thread::spawn(move || {
            let mut methods = Vec::new();
            for location in [
                Some("https://example.org/a(1)"),
                Some("https://t.co/again"),
                Some("javascript:alert(1)"),
                None,
            ] {
                let (mut stream, _) = listener.accept().unwrap();
                let mut request = [0; 4096];
                let read = stream.read(&mut request).unwrap();
                methods.push(
                    String::from_utf8_lossy(&request[..read])
                        .split(' ')
                        .next()
                        .unwrap()
                        .to_owned(),
                );
                let reply = match location {
                    Some(location) => format!(
                        "HTTP/1.1 301 Moved Permanently\r\nLocation: {location}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                    ),
                    None => {
                        "HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".into()
                    }
                };
                stream.write_all(reply.as_bytes()).unwrap();
            }
            methods
        });
        let client = client().unwrap();
        assert_eq!(
            resolve(&client, &format!("{base}/one")),
            "https://example.org/a%281%29"
        );
        for path in ["/two", "/three", "/four"] {
            assert_eq!(resolve(&client, &format!("{base}{path}")), "", "{path}");
        }
        assert_eq!(server.join().unwrap(), ["HEAD"; 4]);
    }
}
