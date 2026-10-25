//! Release versions as the tags and archive names spell them: `X.Y.Z`, with
//! an optional `-pre.release` suffix, compared by Semantic Versioning's
//! precedence (a pre-release sorts before its release).
use std::cmp::Ordering;
use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Version {
    pub(crate) major: u64,
    pub(crate) minor: u64,
    pub(crate) patch: u64,
    /// The pre-release identifiers after `-`; empty for a release.
    pre: Vec<String>,
}

impl Version {
    /// `X.Y.Z` or `X.Y.Z-pre`, with an optional leading `v`; build metadata
    /// (`+…`) is ignored. Numbers have no leading zeros, as tags never do.
    pub(crate) fn parse(text: &str) -> Option<Self> {
        let text = text.trim();
        let text = text.strip_prefix('v').unwrap_or(text);
        let text = text.split_once('+').map_or(text, |(version, _)| version);
        let (core, pre) = match text.split_once('-') {
            Some((core, pre)) => (core, Some(pre)),
            None => (text, None),
        };
        let mut numbers = core.split('.').map(number);
        let (major, minor, patch) = (numbers.next()??, numbers.next()??, numbers.next()??);
        if numbers.next().is_some() {
            return None;
        }
        let pre = match pre {
            None => Vec::new(),
            Some(pre) => {
                let parts: Vec<String> = pre.split('.').map(str::to_owned).collect();
                let valid = |part: &String| {
                    !part.is_empty()
                        && part
                            .bytes()
                            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
                };
                if !parts.iter().all(valid) {
                    return None;
                }
                parts
            }
        };
        Some(Self {
            major,
            minor,
            patch,
            pre,
        })
    }

    /// A development or release-candidate version (`2.2.0-dev`, `2.2.0-rc.1`).
    pub(crate) fn is_prerelease(&self) -> bool {
        !self.pre.is_empty()
    }
}

fn number(text: &str) -> Option<u64> {
    if text.is_empty() || (text.len() > 1 && text.starts_with('0')) {
        return None;
    }
    if !text.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    text.parse().ok()
}

impl Ord for Version {
    fn cmp(&self, other: &Self) -> Ordering {
        (self.major, self.minor, self.patch)
            .cmp(&(other.major, other.minor, other.patch))
            .then_with(|| match (self.pre.is_empty(), other.pre.is_empty()) {
                (true, true) => Ordering::Equal,
                (true, false) => Ordering::Greater,
                (false, true) => Ordering::Less,
                (false, false) => {
                    for (left, right) in self.pre.iter().zip(&other.pre) {
                        let order = match (left.parse::<u64>(), right.parse::<u64>()) {
                            (Ok(left), Ok(right)) => left.cmp(&right),
                            (Ok(_), Err(_)) => Ordering::Less,
                            (Err(_), Ok(_)) => Ordering::Greater,
                            (Err(_), Err(_)) => left.cmp(right),
                        };
                        if order != Ordering::Equal {
                            return order;
                        }
                    }
                    self.pre.len().cmp(&other.pre.len())
                }
            })
    }
}

impl PartialOrd for Version {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl fmt::Display for Version {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}.{}.{}", self.major, self.minor, self.patch)?;
        if !self.pre.is_empty() {
            write!(formatter, "-{}", self.pre.join("."))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(text: &str) -> Version {
        Version::parse(text).unwrap_or_else(|| panic!("{text} parses"))
    }

    #[test]
    fn releases_and_prereleases_order_as_semantic_versioning_says() {
        let ordered = [
            "1.2.0",
            "2.0.0-dev",
            "2.0.0-rc.1",
            "2.0.0-rc.2",
            "2.0.0-rc.10",
            "2.0.0",
            "2.0.1",
            "2.1.0",
            "2.10.0",
            "10.0.0",
        ];
        for pair in ordered.windows(2) {
            assert!(v(pair[0]) < v(pair[1]), "{} < {}", pair[0], pair[1]);
        }
        assert_eq!(v("v2.1.0"), v("2.1.0"));
        assert_eq!(v("2.1.0+build.7"), v("2.1.0"));
        // Numeric identifiers sort before alphanumeric ones; more identifiers sort later.
        assert!(v("2.0.0-1") < v("2.0.0-alpha"));
        assert!(v("2.0.0-alpha") < v("2.0.0-alpha.1"));
    }

    #[test]
    fn only_three_plain_numbers_and_a_clean_suffix_parse() {
        for bad in [
            "",
            "2",
            "2.1",
            "2.1.0.0",
            "02.1.0",
            "2.1.x",
            "2.1.0-",
            "2.1.0-a..b",
            "2.1.0-é",
            "x2.1.0",
            " v ",
        ] {
            assert_eq!(Version::parse(bad), None, "{bad:?}");
        }
        assert!(!v("2.1.0").is_prerelease());
        assert!(v("2.2.0-dev").is_prerelease());
        assert_eq!(v("2.2.0-rc.1").to_string(), "2.2.0-rc.1");
        assert_eq!(v(" v2.2.0 ").to_string(), "2.2.0");
    }
}
