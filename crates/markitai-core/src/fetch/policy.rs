//! The order in which `auto` tries strategies for one URL.
//!
//! As in the reference policy: a domain profile's `strategy_priority` wins,
//! then its `prefer_strategy` (that strategy first, the default order after
//! it), then `fetch.policy.strategy_priority`. Without a priority, a learned
//! browser route, or a domain in a `fetch.fallback_patterns` list the user
//! wrote, starts with the local browser, and every other URL takes the
//! default order static, browser, defuddle, jina, cloudflare. The contract's
//! default `fallback_patterns` list (X, Instagram and other social sites) is
//! not applied: the static path and the X post reader read X posts well and
//! faster. The list is cut to `max_strategy_hops`.
//!
//! Remote services stay in the list only for a URL that may leave the
//! machine: not a local or private host, not matched by
//! `fetch.policy.local_only_patterns` (with `NO_PROXY` when
//! `inherit_no_proxy`), and without credential material. Two deliberate
//! differences from the reference keep local strategies first: a browser-first
//! domain tries static before any remote service, and a learned route keeps
//! its browser failure instead of retrying static (see `docs/fetch.md`).

use super::remote::Service;
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::net::IpAddr;
use url::Url;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Step {
    Static,
    Browser,
    Remote(Service),
}

impl Step {
    fn named(name: &str) -> Option<Self> {
        match name {
            "static" => Some(Self::Static),
            "playwright" => Some(Self::Browser),
            other => Service::named(other).map(Self::Remote),
        }
    }

    pub(crate) fn remote(self) -> Option<Service> {
        match self {
            Self::Remote(service) => Some(service),
            _ => None,
        }
    }
}

const DEFAULT: [Step; 5] = [
    Step::Static,
    Step::Browser,
    Step::Remote(Service::Defuddle),
    Step::Remote(Service::Jina),
    Step::Remote(Service::Cloudflare),
];

/// What decides the order besides the configuration.
pub(crate) struct Facts<'a> {
    pub vars: &'a HashMap<String, String>,
    /// The authority has a learned browser route that was taken for this URL.
    pub learned_route: bool,
    /// Remote services can run at all (consent is not already refused).
    pub remote_possible: bool,
    /// The host name says the URL is local or private (no DNS involved).
    pub private_name: bool,
    /// The user wrote `fetch.fallback_patterns` (file or overrides); the
    /// contract's default list does not make a domain browser-first.
    pub fallback_patterns: bool,
}

fn authority(url: &Url) -> String {
    let host = url.host_str().unwrap_or_default();
    match url.port() {
        Some(port) => format!("{host}:{port}"),
        None => host.to_owned(),
    }
}

/// The domain profile for this URL's exact authority, as the browser reads it.
fn profile<'a>(url: &Url, cfg: &'a Value) -> Option<&'a Value> {
    cfg.pointer("/fetch/domain_profiles")
        .and_then(|profiles| profiles.get(authority(url)))
}

fn steps(list: &Value) -> Option<Vec<Step>> {
    let steps: Vec<Step> = list
        .as_array()?
        .iter()
        .filter_map(|name| Step::named(name.as_str()?))
        .collect();
    (!steps.is_empty()).then_some(steps)
}

/// Whether a configured priority decides the order for this URL, or the
/// policy is off, so a learned browser route does not apply.
pub(crate) fn learned_routes_apply(url: &Url, cfg: &Value) -> bool {
    let set = |value: Option<&Value>| value.is_some_and(|value| !value.is_null());
    let profile = profile(url, cfg);
    cfg.pointer("/fetch/policy/enabled")
        .and_then(Value::as_bool)
        .unwrap_or(true)
        && !set(profile.and_then(|profile| profile.get("strategy_priority")))
        && !set(profile.and_then(|profile| profile.get("prefer_strategy")))
        && !set(cfg.pointer("/fetch/policy/strategy_priority"))
}

fn host(url: &Url) -> String {
    let host = url.host_str().unwrap_or_default();
    host.strip_prefix('[')
        .and_then(|host| host.strip_suffix(']'))
        .unwrap_or(host)
        .trim_end_matches('.')
        .to_ascii_lowercase()
}

/// A `fetch.fallback_patterns` domain: the domain itself or a subdomain.
pub(crate) fn fallback_domain(url: &Url, cfg: &Value) -> bool {
    let host = host(url);
    cfg.pointer("/fetch/fallback_patterns")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(|pattern| pattern.trim().trim_end_matches('.').to_ascii_lowercase())
        .filter(|pattern| !pattern.is_empty())
        .any(|pattern| {
            host == pattern
                || host
                    .strip_suffix(pattern.as_str())
                    .is_some_and(|prefix| prefix.ends_with('.'))
        })
}

pub(crate) fn order(url: &Url, cfg: &Value, facts: &Facts<'_>) -> Vec<Step> {
    let browser_first =
        facts.learned_route || (facts.fallback_patterns && fallback_domain(url, cfg));
    let local = || {
        if facts.learned_route {
            vec![Step::Browser]
        } else if browser_first {
            vec![Step::Browser, Step::Static]
        } else {
            vec![Step::Static, Step::Browser]
        }
    };
    let restricted = facts.private_name || local_only(url, cfg, facts.vars);
    let profile = profile(url, cfg);
    let mut order = if restricted {
        local()
    } else if let Some(list) = profile
        .and_then(|profile| profile.get("strategy_priority"))
        .and_then(steps)
    {
        list
    } else if let Some(preferred) = profile
        .and_then(|profile| profile.get("prefer_strategy"))
        .and_then(Value::as_str)
        .and_then(Step::named)
    {
        std::iter::once(preferred)
            .chain(DEFAULT.into_iter().filter(|step| *step != preferred))
            .collect()
    } else if let Some(list) = cfg
        .pointer("/fetch/policy/strategy_priority")
        .and_then(steps)
    {
        list
    } else if !cfg
        .pointer("/fetch/policy/enabled")
        .and_then(Value::as_bool)
        .unwrap_or(true)
    {
        DEFAULT.to_vec()
    } else {
        let mut order = local();
        order.extend(DEFAULT.into_iter().filter(|step| step.remote().is_some()));
        order
    };
    let hops = cfg
        .pointer("/fetch/policy/max_strategy_hops")
        .and_then(Value::as_u64)
        .unwrap_or(5)
        .clamp(1, 6);
    order.truncate(usize::try_from(hops).unwrap_or(5));
    if restricted || !facts.remote_possible || credential_material(url) {
        order.retain(|step| step.remote().is_none());
        if order.is_empty() {
            order = local();
        }
    }
    order
}

// ---- local-only patterns --------------------------------------------------

/// `fetch.policy.local_only_patterns`, with `NO_PROXY` (or `no_proxy`) when
/// `inherit_no_proxy`, in the proxy exceptions' `NO_PROXY` grammar: `*`
/// matches everything, `.name` and `*.name` match subdomains only, CIDR blocks
/// match addresses and anything else matches the host exactly.
pub(crate) fn local_only(url: &Url, cfg: &Value, vars: &HashMap<String, String>) -> bool {
    let mut patterns = crate::proxy::Bypass::default();
    for pattern in cfg
        .pointer("/fetch/policy/local_only_patterns")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
    {
        patterns.add(pattern);
    }
    if cfg
        .pointer("/fetch/policy/inherit_no_proxy")
        .and_then(Value::as_bool)
        .unwrap_or(true)
        && let Some(inherited) = ["NO_PROXY", "no_proxy"]
            .iter()
            .filter_map(|name| vars.get(*name))
            .find(|value| !value.trim().is_empty())
    {
        for pattern in inherited.split(',') {
            patterns.add(pattern);
        }
    }
    patterns.listed(url)
}

// ---- hosts and credential material ---------------------------------------

/// A host whose name or literal address says it is local or private:
/// `localhost`, the `.local`/`.internal`/`.lan`/`.home`/`.corp`/`.localhost`
/// names, single-label names, and non-public address literals.
pub(crate) fn private_name(url: &Url) -> bool {
    let host = host(url);
    if host.is_empty() || !url.username().is_empty() || url.password().is_some() {
        return true;
    }
    if host == "localhost"
        || [
            ".localhost",
            ".local",
            ".internal",
            ".lan",
            ".home",
            ".corp",
        ]
        .iter()
        .any(|suffix| host.ends_with(suffix))
    {
        return true;
    }
    match host.parse::<IpAddr>() {
        Ok(address) => super::is_private(address),
        Err(_) => !host.contains('.'),
    }
}

/// Lowercase snake form of a parameter name: `apiKey`, `api-key` and
/// `API_KEY` all become `api_key`.
fn identifier(key: &str) -> String {
    let chars: Vec<char> = key.chars().collect();
    let mut snake = String::with_capacity(key.len() + 4);
    for (index, &character) in chars.iter().enumerate() {
        if character.is_ascii_uppercase() && index > 0 {
            let before = chars[index - 1];
            let lower_next = chars
                .get(index + 1)
                .is_some_and(|next| next.is_ascii_lowercase());
            if before.is_ascii_lowercase() || before.is_ascii_digit() || lower_next {
                snake.push('_');
            }
        }
        snake.push(character.to_ascii_lowercase());
    }
    snake
        .split(|character: char| !character.is_ascii_alphanumeric())
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join("_")
}

/// A query, fragment or path parameter name that carries a secret.
fn secret_name(key: &str) -> bool {
    const WHOLE: [&str; 8] = [
        "apikey", "auth", "code", "key", "pwd", "sid", "sig", "ticket",
    ];
    const PARTS: [&str; 18] = [
        "assertion",
        "authorization",
        "auth",
        "bearer",
        "credential",
        "credentials",
        "jwt",
        "otp",
        "passcode",
        "passwd",
        "password",
        "pwd",
        "saml",
        "session",
        "secret",
        "signature",
        "ticket",
        "token",
    ];
    let normalized = identifier(key);
    let parts: HashSet<&str> = normalized.split('_').collect();
    WHOLE.contains(&normalized.as_str())
        || (parts.contains("api") && parts.contains("key"))
        || (parts.contains("key") && (parts.contains("auth") || parts.contains("access")))
        || PARTS.iter().any(|part| parts.contains(part))
}

/// A query or fragment parameter that carries a secret: [`secret_name`], or
/// one of the words this build has always screened anywhere in the name.
pub(crate) fn secret_parameter(key: &str) -> bool {
    let lower = key.to_ascii_lowercase();
    [
        "token",
        "key",
        "secret",
        "password",
        "signature",
        "credential",
    ]
    .iter()
    .any(|word| lower.contains(word))
        || secret_name(key)
}

fn entropy(value: &str) -> f64 {
    let mut counts: HashMap<char, usize> = HashMap::new();
    for character in value.chars() {
        *counts.entry(character).or_default() += 1;
    }
    let length = value.chars().count() as f64;
    counts
        .values()
        .map(|count| {
            let share = *count as f64 / length;
            -share * share.log2()
        })
        .sum()
}

fn url_safe(value: &str) -> bool {
    value
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || b"_~.-".contains(&byte))
}

fn jwt(value: &str) -> bool {
    let parts: Vec<&str> = value.split('.').collect();
    parts.len() == 3
        && parts.iter().all(|part| {
            part.len() >= 8
                && part
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || b"_-".contains(&byte))
        })
}

/// A path segment that looks like an opaque secret on its own.
fn opaque(segment: &str) -> bool {
    if jwt(segment) {
        return true;
    }
    if segment.len() < 24 || !url_safe(segment) {
        return false;
    }
    if segment
        .bytes()
        .filter(|byte| *byte != b'-')
        .all(|byte| byte.is_ascii_hexdigit())
    {
        // Hashes and UUIDs are common public identifiers.
        return false;
    }
    segment.bytes().any(|byte| byte.is_ascii_lowercase())
        && segment.bytes().any(|byte| byte.is_ascii_uppercase())
        && segment.bytes().any(|byte| byte.is_ascii_digit())
        && segment.chars().collect::<HashSet<_>>().len() >= 10
        && entropy(segment) >= 3.5
}

fn uuid(value: &str) -> bool {
    let groups: Vec<&str> = value.split('-').collect();
    groups.iter().map(|group| group.len()).eq([8, 4, 4, 4, 12])
        && groups
            .iter()
            .all(|group| group.bytes().all(|byte| byte.is_ascii_hexdigit()))
}

/// A segment after a route such as `/reset/` or `/token/` that looks like a
/// token rather than a word.
fn contextual_token(segment: &str) -> bool {
    if segment.len() < 8 {
        return false;
    }
    if uuid(segment) || opaque(segment) {
        return true;
    }
    if segment.len() >= 16 && segment.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return true;
    }
    if segment
        .split('-')
        .all(|word| !word.is_empty() && word.bytes().all(|byte| byte.is_ascii_lowercase()))
    {
        return false;
    }
    segment.len() >= 12
        && url_safe(segment)
        && segment
            .bytes()
            .any(|byte| byte.is_ascii_uppercase() || byte.is_ascii_digit())
        && entropy(segment) >= 3.2
}

/// A path segment with its `%XX` escapes decoded.
fn percent_decoded(segment: &str) -> String {
    let bytes = segment.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        let escape = (bytes[index] == b'%')
            .then(|| segment.get(index + 1..index + 3))
            .flatten()
            .and_then(|hex| u8::from_str_radix(hex, 16).ok());
        match escape {
            Some(byte) => {
                decoded.push(byte);
                index += 3;
            }
            None => {
                decoded.push(bytes[index]);
                index += 1;
            }
        }
    }
    String::from_utf8_lossy(&decoded).into_owned()
}

const SENSITIVE_ROUTES: [&str; 13] = [
    "activate",
    "activation",
    "invite",
    "invitation",
    "magic",
    "magic_link",
    "password_reset",
    "reset",
    "secret",
    "session",
    "token",
    "verify",
    "verification",
];

/// Userinfo, or a secret in the path, query or fragment: such a URL is never
/// sent to a remote service.
pub(crate) fn credential_material(url: &Url) -> bool {
    if !url.username().is_empty() || url.password().is_some() {
        return true;
    }
    let segments: Vec<String> = url.path().split('/').map(percent_decoded).collect();
    for (index, segment) in segments.iter().enumerate() {
        if segment.is_empty() {
            continue;
        }
        if opaque(segment) {
            return true;
        }
        if ['=', ':'].iter().any(|separator| {
            segment
                .split_once(*separator)
                .is_some_and(|(key, value)| !value.is_empty() && secret_name(key))
        }) {
            return true;
        }
        if index > 0 {
            let route = &segments[index - 1];
            if (SENSITIVE_ROUTES.contains(&identifier(route).as_str()) || secret_name(route))
                && contextual_token(segment)
            {
                return true;
            }
        }
    }
    if url.query_pairs().any(|(key, _)| secret_parameter(&key)) {
        return true;
    }
    url.fragment().is_some_and(|fragment| {
        fragment.contains('=')
            && url::form_urlencoded::parse(fragment.as_bytes())
                .any(|(key, _)| secret_parameter(&key))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config;
    use serde_json::json;

    fn url(text: &str) -> Url {
        Url::parse(text).unwrap()
    }

    fn facts(vars: &HashMap<String, String>) -> Facts<'_> {
        Facts {
            vars,
            learned_route: false,
            remote_possible: true,
            private_name: false,
            fallback_patterns: false,
        }
    }

    fn names(steps: &[Step]) -> Vec<&'static str> {
        steps
            .iter()
            .map(|step| match step {
                Step::Static => "static",
                Step::Browser => "playwright",
                Step::Remote(service) => service.name(),
            })
            .collect()
    }

    #[test]
    fn the_default_order_is_local_first_and_remote_needs_a_possible_consent() {
        let vars = HashMap::new();
        let cfg = config::defaults();
        let page = url("https://example.com/article");
        assert_eq!(
            names(&order(&page, &cfg, &facts(&vars))),
            ["static", "playwright", "defuddle", "jina", "cloudflare"]
        );
        let local = Facts {
            remote_possible: false,
            ..facts(&vars)
        };
        assert_eq!(names(&order(&page, &cfg, &local)), ["static", "playwright"]);
        // A learned route keeps its browser failure; remote services follow it.
        let learned = Facts {
            learned_route: true,
            ..facts(&vars)
        };
        assert_eq!(
            names(&order(&page, &cfg, &learned)),
            ["playwright", "defuddle", "jina", "cloudflare"]
        );
    }

    #[test]
    fn the_default_fallback_patterns_keep_x_static_first() {
        let vars = HashMap::new();
        let cfg = config::defaults();
        // The contract's default list names x.com, but nobody wrote it.
        assert!(fallback_domain(&url("https://x.com/user/status/1"), &cfg));
        for address in [
            "https://x.com/user/status/1",
            "https://twitter.com/user/status/1",
            "https://www.linkedin.com/in/someone",
        ] {
            assert_eq!(
                names(&order(&url(address), &cfg, &facts(&vars))),
                ["static", "playwright", "defuddle", "jina", "cloudflare"],
                "{address}"
            );
        }
    }

    #[test]
    fn written_fallback_patterns_put_the_browser_before_static_for_the_domain_and_its_subdomains() {
        let vars = HashMap::new();
        let mut cfg = config::defaults();
        cfg["fetch"]["fallback_patterns"] = json!(["x.com", "example.org"]);
        let written = Facts {
            fallback_patterns: true,
            ..facts(&vars)
        };
        for address in [
            "https://x.com/user/status/1",
            "https://www.x.com/user/status/1",
            "https://example.org/app",
        ] {
            assert_eq!(
                names(&order(&url(address), &cfg, &written)),
                ["playwright", "static", "defuddle", "jina", "cloudflare"],
                "{address}"
            );
        }
        // Only the listed domains and their subdomains: not a name that ends
        // alike, and not a default entry the written list left out.
        for address in ["https://notx.com/a", "https://www.linkedin.com/in/someone"] {
            assert_eq!(
                names(&order(&url(address), &cfg, &written))[0],
                "static",
                "{address}"
            );
        }
        // Without remote consent the browser still comes first.
        let local = Facts {
            remote_possible: false,
            ..written
        };
        assert_eq!(
            names(&order(&url("https://x.com/a"), &cfg, &local)),
            ["playwright", "static"]
        );
    }

    #[test]
    fn priorities_win_in_the_reference_precedence_and_hops_cut_the_list() {
        let vars = HashMap::new();
        let mut cfg = config::defaults();
        let page = url("https://example.com:8443/a");
        cfg["fetch"]["policy"]["strategy_priority"] = json!(["jina", "static"]);
        assert_eq!(
            names(&order(&page, &cfg, &facts(&vars))),
            ["jina", "static"]
        );
        assert!(!learned_routes_apply(&page, &cfg));
        cfg["fetch"]["domain_profiles"]["example.com:8443"] =
            json!({"prefer_strategy": "playwright"});
        assert_eq!(
            names(&order(&page, &cfg, &facts(&vars))),
            ["playwright", "static", "defuddle", "jina", "cloudflare"]
        );
        cfg["fetch"]["domain_profiles"]["example.com:8443"]["strategy_priority"] =
            json!(["cloudflare", "static"]);
        assert_eq!(
            names(&order(&page, &cfg, &facts(&vars))),
            ["cloudflare", "static"]
        );
        // Only the exact authority has the profile.
        assert_eq!(
            names(&order(&url("https://example.com/a"), &cfg, &facts(&vars))),
            ["jina", "static"]
        );
        cfg["fetch"]["policy"]["max_strategy_hops"] = json!(1);
        assert_eq!(names(&order(&page, &cfg, &facts(&vars))), ["cloudflare"]);
        // A list left with nothing that can run becomes the local default.
        let local = Facts {
            remote_possible: false,
            ..facts(&vars)
        };
        assert_eq!(names(&order(&page, &cfg, &local)), ["static", "playwright"]);

        let mut cfg = config::defaults();
        cfg["fetch"]["policy"]["enabled"] = json!(false);
        assert!(!learned_routes_apply(&url("https://x.com/a"), &cfg));
        assert_eq!(
            names(&order(&url("https://x.com/a"), &cfg, &facts(&vars))),
            ["static", "playwright", "defuddle", "jina", "cloudflare"]
        );
        assert!(learned_routes_apply(
            &url("https://x.com/a"),
            &config::defaults()
        ));
    }

    #[test]
    fn local_only_patterns_and_no_proxy_keep_a_url_on_the_machine() {
        let mut cfg = config::defaults();
        cfg["fetch"]["policy"]["local_only_patterns"] = json!([
            "intranet.example",
            ".corp.example",
            "10.0.0.0/8",
            "192.168.1.0/255.255.255.0",
            "fd00::/8"
        ]);
        let vars: HashMap<String, String> = [(
            "no_proxy".to_owned(),
            "*.internal.test, docs.example".to_owned(),
        )]
        .into();
        for (address, local) in [
            ("https://intranet.example/a", true),
            ("https://www.intranet.example/a", false),
            ("https://wiki.corp.example/a", true),
            ("https://corp.example/a", false),
            ("http://10.1.2.3/a", true),
            ("http://11.1.2.3/a", false),
            ("http://192.168.1.77/a", true),
            ("http://192.168.2.77/a", false),
            ("http://[fd12::1]/a", true),
            ("https://a.internal.test/x", true),
            ("https://docs.example/x", true),
            ("https://example.org/x", false),
        ] {
            assert_eq!(local_only(&url(address), &cfg, &vars), local, "{address}");
            let steps = order(&url(address), &cfg, &facts(&vars));
            assert_eq!(steps.iter().any(|step| step.remote().is_some()), !local);
        }
        // Names are compared as the URL parser spells them, so a pattern
        // written in Unicode matches its punycode host.
        cfg["fetch"]["policy"]["local_only_patterns"] =
            json!(["bücher.example", ".straße.example"]);
        for (address, local) in [
            ("https://bücher.example/a", true),
            ("https://xn--bcher-kva.example/a", true),
            ("https://www.straße.example/a", true),
            ("https://buecher.example/a", false),
        ] {
            assert_eq!(local_only(&url(address), &cfg, &vars), local, "{address}");
        }
        cfg["fetch"]["policy"]["inherit_no_proxy"] = json!(false);
        assert!(!local_only(&url("https://docs.example/x"), &cfg, &vars));
        let all: HashMap<String, String> = [("NO_PROXY".to_owned(), "*".to_owned())].into();
        cfg["fetch"]["policy"]["inherit_no_proxy"] = json!(true);
        assert!(local_only(&url("https://example.org/x"), &cfg, &all));
    }

    #[test]
    fn private_names_and_credentials_never_reach_a_remote_service() {
        for (address, private) in [
            ("http://localhost:8080/a", true),
            ("http://app.localhost/a", true),
            ("http://printer.local/a", true),
            ("http://build.corp/a", true),
            ("http://intranet/a", true),
            ("http://127.0.0.1/a", true),
            ("http://[::1]/a", true),
            ("http://192.168.0.4/a", true),
            ("http://100.64.0.1/a", true),
            ("https://user@example.com/a", true),
            ("https://example.com/a", false),
            ("http://93.184.216.34/a", false),
        ] {
            assert_eq!(private_name(&url(address)), private, "{address}");
        }
        for (address, secret) in [
            ("https://example.com/a?api_key=1", true),
            ("https://example.com/a?apiKey=1", true),
            ("https://example.com/a?accessKey=1", true),
            ("https://example.com/a?X-Amz-Signature=1", true),
            ("https://example.com/a?sid=1", true),
            ("https://example.com/a?code=1", true),
            ("https://example.com/a?session_id=1", true),
            ("https://example.com/a#access_token=abc", true),
            ("https://example.com/reset/Xy7kQ2pLm9Za", true),
            (
                "https://example.com/verify/3f1c2a9b-4d5e-4f60-8a7b-1c2d3e4f5a6b",
                true,
            ),
            ("https://example.com/files/token=abc", true),
            (
                "https://example.com/d/eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0In0.c2lnbmF0dXJlMTIz",
                true,
            ),
            ("https://example.com/s/aB3dE5fG7hJ9kL1mN3pQ5rS7tU9v", true),
            ("https://user:pass@example.com/", true),
            ("https://example.com/a?id=5&view=full", false),
            ("https://example.com/reset/how-to-reset-a-router", false),
            (
                "https://github.com/rust-lang/rust/commit/0123456789abcdef0123456789abcdef01234567",
                false,
            ),
            (
                "https://example.com/post/3f1c2a9b-4d5e-4f60-8a7b-1c2d3e4f5a6b",
                false,
            ),
            ("https://example.com/a#section-2", false),
        ] {
            assert_eq!(credential_material(&url(address)), secret, "{address}");
        }
        let vars = HashMap::new();
        let cfg = config::defaults();
        let steps = order(&url("https://example.com/a?token=x"), &cfg, &facts(&vars));
        assert_eq!(names(&steps), ["static", "playwright"]);
        let private = Facts {
            private_name: true,
            ..facts(&vars)
        };
        assert_eq!(
            names(&order(&url("http://localhost/a"), &cfg, &private)),
            ["static", "playwright"]
        );
    }

    #[test]
    fn parameter_names_fold_their_spellings() {
        for (key, folded) in [
            ("apiKey", "api_key"),
            ("API_KEY", "api_key"),
            ("api-key", "api_key"),
            ("APIKey", "api_key"),
            ("X-Amz-Credential", "x_amz_credential"),
        ] {
            assert_eq!(identifier(key), folded, "{key}");
        }
    }
}
