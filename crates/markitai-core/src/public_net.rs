//! Requests a server sends on behalf of a user token's own models (their
//! conversions, discovery and tests) reach public addresses only. A name is
//! resolved when the connection is made and every address outside the public
//! internet is dropped, so a name that points at (or rebinds to) this machine
//! or its network fails; an address written in the URL is checked the same
//! way; no proxy carries the request elsewhere and no redirect is followed.
use std::future::Future;
use std::net::{SocketAddr, ToSocketAddrs};
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};

/// The reason a public-only request was refused, without its address.
pub(crate) const REFUSED: &str = "this model's address is not on the public internet";

/// Whether `ip` is outside the public internet for these requests. A debug
/// build (never a release) also lets through the addresses listed in the
/// process variable `MARKITAI_TEST_PUBLIC_NET_ALLOW`, so integration tests
/// can stand a user's model server on this machine's network address.
fn refused_address(ip: std::net::IpAddr) -> bool {
    crate::fetch::is_private(ip)
        && !(cfg!(debug_assertions)
            && std::env::var("MARKITAI_TEST_PUBLIC_NET_ALLOW").is_ok_and(|list| {
                list.split(',')
                    .any(|allowed| allowed.trim().parse() == Ok(ip))
            }))
}

/// Whether `url` may be requested by a public-only client: an IP address
/// written in it must be public (a name is checked when it is resolved).
pub(crate) fn url_allowed(url: &url::Url) -> bool {
    match url.host() {
        Some(url::Host::Ipv4(address)) => !refused_address(address.into()),
        Some(url::Host::Ipv6(address)) => !refused_address(address.into()),
        Some(url::Host::Domain(_)) => true,
        None => false,
    }
}

/// Whether `error` is the resolver's refusal of a non-public address.
pub(crate) fn refused(error: &reqwest::Error) -> bool {
    let mut source: Option<&(dyn std::error::Error + 'static)> = Some(error);
    while let Some(current) = source {
        if current.to_string() == REFUSED {
            return true;
        }
        source = current.source();
    }
    false
}

/// `builder` restricted to public addresses: the resolver drops every other
/// address, without proxy and without redirects.
pub(crate) fn client(
    builder: reqwest::blocking::ClientBuilder,
) -> reqwest::blocking::ClientBuilder {
    builder
        .no_proxy()
        .redirect(reqwest::redirect::Policy::none())
        .dns_resolver(Arc::new(PublicResolver))
}

std::thread_local! {
    /// Whether fetches on this thread must reach public addresses only.
    static RESTRICTED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Whether the current fetch is restricted to public addresses (see
/// [`restrict`]). [`crate::fetch::client`] builds a public-only client then.
pub(crate) fn restricted() -> bool {
    RESTRICTED.with(std::cell::Cell::get)
}

/// Restricts every fetch on this thread to public addresses until the returned
/// guard drops. A fetch already restricted stays restricted (nesting keeps
/// the stricter state).
#[must_use]
pub(crate) fn restrict(restricted: bool) -> Restriction {
    Restriction(RESTRICTED.with(|cell| cell.replace(restricted || cell.get())))
}

pub(crate) struct Restriction(bool);

impl Drop for Restriction {
    fn drop(&mut self) {
        RESTRICTED.with(|cell| cell.set(self.0));
    }
}

/// Whether `url`'s host is on the public internet: a public IP literal, or a
/// name that resolves to at least one public address. A fetch of such a host
/// is held to public addresses for the rest of its hops; a private or local
/// host (a deliberate local conversion) is not. Names that do not resolve
/// count as not public, so they impose no restriction of their own.
pub(crate) fn public_host(url: &url::Url) -> bool {
    match url.host() {
        Some(url::Host::Ipv4(address)) => !refused_address(address.into()),
        Some(url::Host::Ipv6(address)) => !refused_address(address.into()),
        Some(url::Host::Domain(host)) => (host, url.port_or_known_default().unwrap_or(443))
            .to_socket_addrs()
            .map(|addresses| {
                addresses
                    .into_iter()
                    .any(|address| !refused_address(address.ip()))
            })
            .unwrap_or(false),
        None => false,
    }
}

/// Whether `url`'s host would connect only to public addresses: a public IP
/// literal, or a name whose every resolved address is public (none private).
/// Chromium resolves names itself and may pick any of them, so a host with any
/// private address is refused for a restricted browser fetch.
pub(crate) fn host_all_public(url: &url::Url) -> bool {
    match url.host() {
        Some(url::Host::Ipv4(address)) => !refused_address(address.into()),
        Some(url::Host::Ipv6(address)) => !refused_address(address.into()),
        Some(url::Host::Domain(host)) => (host, url.port_or_known_default().unwrap_or(443))
            .to_socket_addrs()
            .map(|addresses| {
                let mut addresses = addresses.peekable();
                addresses.peek().is_some()
                    && addresses.all(|address| !refused_address(address.ip()))
            })
            .unwrap_or(false),
        None => false,
    }
}

/// `builder` restricted to public addresses but following redirects (up to
/// ten), each target's literal address checked and its name resolved through
/// the public resolver, without a proxy. For page fetches, where redirects are
/// followed; model requests use [`client`], which follows none.
pub(crate) fn following_client(
    builder: reqwest::blocking::ClientBuilder,
) -> reqwest::blocking::ClientBuilder {
    builder
        .no_proxy()
        .redirect(reqwest::redirect::Policy::custom(|attempt| {
            if attempt.previous().len() >= 10 {
                attempt.error("too many redirects")
            } else if url_allowed(attempt.url()) {
                attempt.follow()
            } else {
                attempt.error(REFUSED)
            }
        }))
        .dns_resolver(Arc::new(PublicResolver))
}

/// The public addresses among `addresses`, or an error naming none of them.
fn public(addresses: impl Iterator<Item = SocketAddr>) -> std::io::Result<Vec<SocketAddr>> {
    let kept: Vec<_> = addresses
        .filter(|address| !refused_address(address.ip()))
        .collect();
    if kept.is_empty() {
        Err(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            REFUSED,
        ))
    } else {
        Ok(kept)
    }
}

struct PublicResolver;

impl reqwest::dns::Resolve for PublicResolver {
    fn resolve(&self, name: reqwest::dns::Name) -> reqwest::dns::Resolving {
        let host = name.as_str().to_owned();
        let lookup = Lookup::default();
        let shared = lookup.state.clone();
        // The system resolver blocks; it runs beside the client's runtime.
        let spawned = std::thread::Builder::new()
            .name("markitai-public-dns".into())
            .spawn(move || {
                let result = (host.as_str(), 0).to_socket_addrs().and_then(public);
                let mut state = shared.lock().unwrap_or_else(|error| error.into_inner());
                state.0 = Some(result);
                if let Some(waker) = state.1.take() {
                    waker.wake();
                }
            });
        if spawned.is_err() {
            let mut state = lookup
                .state
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            state.0 = Some(Err(std::io::Error::other("name lookup unavailable")));
        }
        Box::pin(lookup)
    }
}

type Outcome = Option<std::io::Result<Vec<SocketAddr>>>;

/// One name lookup finished by its thread.
#[derive(Default)]
struct Lookup {
    state: Arc<Mutex<(Outcome, Option<Waker>)>>,
}

impl Future for Lookup {
    type Output = Result<reqwest::dns::Addrs, Box<dyn std::error::Error + Send + Sync>>;

    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        match state.0.take() {
            Some(Ok(addresses)) => Poll::Ready(Ok(Box::new(addresses.into_iter()))),
            Some(Err(error)) => Poll::Ready(Err(Box::new(error))),
            None => {
                state.1 = Some(context.waker().clone());
                Poll::Pending
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_public_addresses_are_kept_and_none_left_is_refused() {
        let address = |text: &str| text.parse::<SocketAddr>().unwrap();
        let kept = public(
            [
                address("127.0.0.1:443"),
                address("93.184.216.34:443"),
                address("[::1]:443"),
                address("10.0.0.7:443"),
            ]
            .into_iter(),
        )
        .unwrap();
        assert_eq!(kept, [address("93.184.216.34:443")]);
        for refused in [
            "127.0.0.1:80",
            "169.254.169.254:80",
            "192.168.1.20:80",
            "[::ffff:127.0.0.1]:80",
            "[fd00::1]:80",
        ] {
            let error = public(std::iter::once(address(refused))).unwrap_err();
            assert_eq!(error.to_string(), REFUSED, "{refused}");
        }
        for (url, allowed) in [
            ("http://169.254.169.254/latest/", false),
            ("http://[::1]:11434/v1", false),
            ("http://10.1.2.3:8000/v1", false),
            ("https://api.example.com/v1", true),
            ("https://93.184.216.34/v1", true),
        ] {
            assert_eq!(
                url_allowed(&url::Url::parse(url).unwrap()),
                allowed,
                "{url}"
            );
        }
    }

    #[test]
    fn a_public_only_client_cannot_reach_this_machine() {
        // A loopback server that must never be reached.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        listener.set_nonblocking(true).unwrap();
        let client = client(reqwest::blocking::Client::builder())
            .timeout(std::time::Duration::from_secs(10))
            .build()
            .unwrap();
        // A trailing dot (the fully qualified name) changes nothing: names
        // are judged by the addresses they resolve to, not by spelling, and
        // an address written with one is still that address.
        for host in ["localhost", "localhost.", "LOCALHOST."] {
            let error = client
                .get(format!("http://{host}:{port}/v1/models"))
                .send()
                .unwrap_err();
            assert!(error.is_connect(), "{host}: {error:?}");
            assert!(listener.accept().is_err(), "{host} reached loopback");
        }
        assert!(!url_allowed(
            &url::Url::parse("http://127.0.0.1./v1").unwrap()
        ));
        assert!(!host_all_public(
            &url::Url::parse(&format!("http://localhost.:{port}/")).unwrap()
        ));
    }
}
