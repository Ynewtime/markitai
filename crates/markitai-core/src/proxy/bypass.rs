//! `NO_PROXY`-style exceptions with the reference matcher's semantics:
//! `*` matches everything, `.example.com` and `*.example.com` match
//! subdomains only, other names and IP addresses match exactly, and CIDR
//! blocks (a prefix length or an IPv4 netmask) match addresses. Names are
//! compared as the URL parser spells them, so Unicode names match punycode.
//! `fetch.policy.local_only_patterns` uses the same grammar. Unsupported rules such as `<local>`, port-qualified
//! hosts, other wildcards or malformed CIDR are ignored one by one, as the
//! reference ignores them; they never disable the remaining valid rules.
use super::{MAX_SETTING, invalid};
use crate::Result;
use std::collections::HashSet;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
use url::Url;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum Rule {
    All,
    Exact(String),
    Suffix(String),
    Address(IpAddr),
    Network(IpAddr, u8),
}

#[derive(Clone, Debug, Default)]
pub(crate) struct Bypass {
    rules: Vec<Rule>,
    seen: HashSet<Rule>,
}

fn unbracket(value: &str) -> &str {
    value
        .strip_prefix('[')
        .and_then(|value| value.strip_suffix(']'))
        .unwrap_or(value)
}

/// A host name as the URL parser would spell it, lowercase, without a final dot.
fn domain(value: &str) -> Option<String> {
    if value.is_empty()
        || value.chars().any(|ch| {
            ch.is_whitespace()
                || ch.is_control()
                || matches!(
                    ch,
                    '/' | '\\' | ':' | '@' | '?' | '#' | '*' | '%' | '<' | '>' | ';' | '[' | ']'
                )
        })
    {
        return None;
    }
    let url = Url::parse(&format!("http://{value}")).ok()?;
    let host = url.host_str()?.trim_end_matches('.').to_ascii_lowercase();
    (!host.is_empty()).then_some(host)
}

fn network(address: IpAddr, bits: u8) -> Option<IpAddr> {
    match address {
        IpAddr::V4(ip) if bits <= 32 => Some(
            Ipv4Addr::from(u32::from(ip) & u32::MAX.checked_shl(32 - u32::from(bits)).unwrap_or(0))
                .into(),
        ),
        IpAddr::V6(ip) if bits <= 128 => Some(
            Ipv6Addr::from(
                u128::from(ip) & u128::MAX.checked_shl(128 - u32::from(bits)).unwrap_or(0),
            )
            .into(),
        ),
        _ => None,
    }
}

fn rule(raw: &str) -> Option<Rule> {
    if raw == "*" {
        return Some(Rule::All);
    }
    if let Some((address, bits)) = raw.rsplit_once('/') {
        let address = unbracket(address);
        let address = address.split('%').next().unwrap_or(address);
        let address = address.parse::<IpAddr>().ok()?;
        let bits = bits.parse::<u8>().ok().or_else(|| {
            let mask = bits.parse::<Ipv4Addr>().ok()?;
            address
                .is_ipv4()
                .then(|| u32::from(mask).leading_ones() as u8)
        })?;
        return Some(Rule::Network(network(address, bits)?, bits));
    }
    if let Ok(address) = unbracket(raw).parse::<IpAddr>() {
        return Some(Rule::Address(address));
    }
    if let Some(suffix) = raw.strip_prefix("*.").or_else(|| raw.strip_prefix('.')) {
        return domain(suffix).map(Rule::Suffix);
    }
    domain(raw).map(Rule::Exact)
}

fn host(url: &Url) -> (String, Option<IpAddr>) {
    let name = unbracket(url.host_str().unwrap_or(""))
        .trim_end_matches('.')
        .to_ascii_lowercase();
    let address = name.parse::<IpAddr>().ok();
    (name, address)
}

fn loopback(name: &str, address: Option<IpAddr>) -> bool {
    name == "localhost"
        || name.ends_with(".localhost")
        || address.is_some_and(|ip| match ip {
            IpAddr::V4(ip) => ip.is_loopback(),
            IpAddr::V6(ip) => {
                ip.is_loopback() || ip.to_ipv4_mapped().is_some_and(|ip| ip.is_loopback())
            }
        })
}

impl Bypass {
    pub(super) fn parse(raw: &str) -> Result<Self> {
        if raw.len() > MAX_SETTING {
            return Err(invalid());
        }
        let mut this = Self::default();
        for raw in raw.split(',') {
            this.add(raw);
        }
        Ok(this)
    }

    /// One entry; an unsupported one is ignored.
    pub(crate) fn add(&mut self, raw: &str) {
        let raw = raw.trim();
        if !raw.is_empty()
            && let Some(rule) = rule(&raw.to_ascii_lowercase())
        {
            self.push(rule);
        }
    }

    fn push(&mut self, rule: Rule) {
        if self.seen.insert(rule.clone()) {
            self.rules.push(rule);
        }
    }

    pub(super) fn merge(&mut self, other: Self) -> Result<()> {
        for rule in other.rules {
            self.push(rule);
        }
        Ok(())
    }

    /// Whether a request to `url` goes direct: a loopback host or a listed one.
    pub(super) fn matches(&self, url: &Url) -> bool {
        let (name, address) = host(url);
        loopback(&name, address) || self.listed(url)
    }

    /// Whether a rule names `url`'s host, without the implicit loopback rule.
    pub(crate) fn listed(&self, url: &Url) -> bool {
        let (name, address) = host(url);
        self.rules.iter().any(|rule| match rule {
            Rule::All => true,
            Rule::Exact(value) => name == *value,
            Rule::Suffix(value) => name
                .strip_suffix(value.as_str())
                .is_some_and(|prefix| prefix.ends_with('.')),
            Rule::Address(ip) => address == Some(*ip),
            Rule::Network(ip, bits) => {
                address.and_then(|address| network(address, *bits)) == Some(*ip)
            }
        })
    }

    /// The same rules in Chromium's documented `--proxy-bypass-list` grammar:
    /// a plain host is exact, `.name` means subdomains only, IPv6 hosts are
    /// bracketed and IPv6 CIDR blocks are not.
    pub(super) fn chromium(&self) -> String {
        let mut rules = vec![
            "localhost".to_owned(),
            "*.localhost".into(),
            "127.0.0.0/8".into(),
            "[::1]".into(),
        ];
        let mut seen: HashSet<String> = rules.iter().cloned().collect();
        for rule in &self.rules {
            let value = match rule {
                Rule::All => "*".into(),
                Rule::Exact(name) => name.clone(),
                Rule::Suffix(name) => format!(".{name}"),
                Rule::Address(IpAddr::V4(ip)) => ip.to_string(),
                Rule::Address(IpAddr::V6(ip)) => format!("[{ip}]"),
                Rule::Network(ip, bits) => format!("{ip}/{bits}"),
            };
            if seen.insert(value.clone()) {
                rules.push(value);
            }
        }
        rules.join(";")
    }
}
