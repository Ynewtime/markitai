//! Injected settings only: no test reads the machine's real proxy configuration.
use super::system::{self, Reader};
use super::*;
use std::time::{Duration, Instant};

fn env(pairs: &[(&str, &str)]) -> HashMap<String, String> {
    pairs
        .iter()
        .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
        .collect()
}

fn url(value: &str) -> Url {
    Url::parse(value).unwrap()
}

fn system(endpoint: &str, bypass: &str) -> SystemProxy {
    SystemProxy {
        endpoint: url(endpoint),
        bypass: bypass.into(),
    }
}

fn found(endpoint: &str, bypass: &str) -> System {
    Some((url(endpoint), Bypass::parse(bypass).unwrap()))
}

fn no_discovery() -> System {
    panic!("this decision must not read operating-system settings")
}

fn decide(settings: &Settings, target: &str, system: impl FnOnce() -> System) -> Option<String> {
    settings
        .for_url(&url(target), system)
        .map(|url| url.to_string())
}

#[test]
fn environment_order_selects_one_proxy_and_skips_blank_values() {
    let order = [
        ("HTTPS_PROXY", "http://a.test:1"),
        ("HTTP_PROXY", "http://b.test:2"),
        ("ALL_PROXY", "http://c.test:3"),
        ("https_proxy", "http://d.test:4"),
        ("http_proxy", "http://e.test:5"),
        ("all_proxy", "http://f.test:6"),
    ];
    for start in 0..order.len() {
        let settings = Settings::resolve(&env(&order[start..])).unwrap();
        for target in ["http://site.test/", "https://site.test/"] {
            assert_eq!(
                decide(&settings, target, no_discovery),
                Some(url(order[start].1).to_string()),
                "{start} {target}"
            );
        }
        assert!(settings.browser(no_discovery).unwrap().0.is_some());
    }
    let blank = Settings::resolve(&env(&[
        ("HTTPS_PROXY", "  "),
        ("HTTP_PROXY", "http://b.test:2"),
    ]))
    .unwrap();
    assert_eq!(
        decide(&blank, "http://x.test/", no_discovery).as_deref(),
        Some("http://b.test:2/")
    );
    let direct = Settings::resolve(&env(&[])).unwrap();
    assert!(decide(&direct, "https://site.test/", || None).is_none());
}

#[test]
fn system_settings_are_read_only_when_needed_and_merge_only_their_own_exceptions() {
    let settings = Settings::resolve(&env(&[("NO_PROXY", "env.test")])).unwrap();
    // Loopback and NO_PROXY targets are direct without any system read.
    for direct in [
        "http://env.test/",
        "http://127.0.0.1:9/",
        "http://[::1]/",
        "http://x.localhost/",
    ] {
        assert!(
            decide(&settings, direct, no_discovery).is_none(),
            "{direct}"
        );
    }
    let os = || found("http://sys.test:3128", "os.test,<local>");
    assert!(decide(&settings, "http://os.test/", os).is_none());
    assert_eq!(
        decide(&settings, "https://other.test/", os).as_deref(),
        Some("http://sys.test:3128/")
    );
    assert!(decide(&settings, "https://other.test/", || None).is_none());
    let (server, bypass) = settings.browser(os).unwrap();
    assert_eq!(server.as_deref(), Some("http://sys.test:3128"));
    assert!(bypass.ends_with(";env.test;os.test"), "{bypass}");
    // With an environment proxy, the system exception list never applies.
    let env_proxy = Settings::resolve(&env(&[("HTTP_PROXY", "http://env.proxy:1")])).unwrap();
    assert_eq!(
        decide(&env_proxy, "http://os.test/", no_discovery).as_deref(),
        Some("http://env.proxy:1/")
    );
    assert!(
        !env_proxy
            .browser(no_discovery)
            .unwrap()
            .1
            .contains("os.test")
    );
}

#[test]
fn exceptions_prefer_nonempty_uppercase_then_lowercase() {
    let pairs = [
        ("HTTP_PROXY", "http://proxy.test:8080"),
        ("NO_PROXY", ""),
        ("no_proxy", "lower.test"),
    ];
    let settings = Settings::resolve(&env(&pairs)).unwrap();
    assert!(decide(&settings, "http://lower.test/", no_discovery).is_none());
    let pairs = [
        ("HTTP_PROXY", "http://proxy.test:8080"),
        ("NO_PROXY", "upper.test"),
        ("no_proxy", "lower.test"),
    ];
    let settings = Settings::resolve(&env(&pairs)).unwrap();
    assert!(decide(&settings, "http://upper.test/", no_discovery).is_none());
    assert!(decide(&settings, "http://lower.test/", no_discovery).is_some());
}

#[test]
fn bypass_rules_follow_reference_matching_and_ignore_unsupported_entries() {
    let rules = Bypass::parse(
        "exact.test, .suffix.test, *.star.test, 10.1.2.3, ::5, [fd00::7], 192.168.0.0/16, \
         fe80::/10, 10.9.9.9/33, <local>, port.test:8080, *mid.test, [::6]:80, bad/cidr, UPPER.Test.",
    )
    .unwrap();
    let cases = [
        ("http://exact.test/", true),
        ("http://sub.exact.test/", false),
        ("http://a.suffix.test/", true),
        ("http://suffix.test/", false),
        ("http://a.b.star.test/", true),
        ("http://star.test/", false),
        ("http://10.1.2.3/", true),
        ("http://[::5]/", true),
        ("http://[fd00::7]/", true),
        ("http://192.168.44.1/", true),
        ("http://[fe80::1]/", true),
        ("http://10.9.9.9/", false),
        ("http://port.test/", false),
        ("http://xmid.test/", false),
        ("http://upper.test./", true),
        ("http://localhost/", true),
        ("http://api.localhost/", true),
        ("http://127.8.0.1/", true),
        ("http://[::1]/", true),
        ("http://[::ffff:127.0.0.1]/", true),
        ("http://public.test/", false),
    ];
    for (target, expected) in cases {
        assert_eq!(rules.matches(&url(target)), expected, "{target}");
    }
    assert!(
        Bypass::parse("*")
            .unwrap()
            .matches(&url("https://anything.test/"))
    );
    // The local-only grammar shares this parser: an IPv4 netmask is a prefix.
    let masked = Bypass::parse("192.168.1.0/255.255.255.0, fd00::/255.0.0.0").unwrap();
    assert!(masked.matches(&url("http://192.168.1.77/")));
    assert!(!masked.matches(&url("http://192.168.2.77/")));
    assert!(!masked.matches(&url("http://[fd00::1]/")));
    assert!(Bypass::parse(&"a".repeat(MAX_SETTING + 1)).is_err());
    // Many rules are accepted and de-duplicated without quadratic work.
    let many = (0..2500)
        .map(|i| format!("h{i}.test,h{i}.test"))
        .collect::<Vec<_>>()
        .join(",");
    assert!(
        Bypass::parse(&many)
            .unwrap()
            .matches(&url("http://h2499.test/"))
    );
}

#[test]
fn endpoints_accept_owner_spelling_and_reject_unsafe_shapes() {
    assert_eq!(
        endpoint("127.0.0.1:7890", false).unwrap().as_str(),
        "http://127.0.0.1:7890/"
    );
    assert!(endpoint("http://user:pass@proxy.test:8080", false).is_ok());
    assert!(endpoint("http://user:pass@proxy.test:8080", true).is_err());
    assert!(endpoint("socks5://proxy.test:1080", false).is_ok());
    assert!(endpoint("socks5://proxy.test:1080", true).is_err());
    for bad in [
        "ftp://proxy.test",
        "http://proxy.test/path",
        "http://proxy.test/?q=1",
        "http://proxy.test:0",
        "http://pro xy.test",
        "http://",
    ] {
        assert!(endpoint(bad, false).is_err(), "{bad}");
    }
    let socks = Settings::resolve(&env(&[("ALL_PROXY", "socks5://proxy.test:1080")])).unwrap();
    assert!(matches!(socks.static_client(), Err(Error::Unsupported(_))));
    assert_eq!(
        socks.browser(no_discovery).unwrap().0.as_deref(),
        Some("socks5://proxy.test:1080")
    );
    let credentialed =
        Settings::resolve(&env(&[("HTTPS_PROXY", "http://u:secret@proxy.test:8080")])).unwrap();
    assert!(credentialed.static_client().is_ok());
    let error = credentialed.browser(no_discovery).unwrap_err().to_string();
    assert!(!error.contains("secret"));
    assert!(Settings::resolve(&env(&[("HTTPS_PROXY", "not a url")])).is_err());
}

#[test]
fn chromium_projection_keeps_loopback_and_supported_rules_only() {
    let settings = Settings::resolve(&env(&[
        ("HTTPS_PROXY", "http://proxy.test:8080/"),
        ("NO_PROXY", "a.test,.b.test,10.0.0.0/8,::9,<local>,c.test:1"),
    ]))
    .unwrap();
    let (server, bypass) = settings.browser(no_discovery).unwrap();
    assert_eq!(server.as_deref(), Some("http://proxy.test:8080"));
    assert_eq!(
        bypass,
        "localhost;*.localhost;127.0.0.0/8;[::1];a.test;.b.test;10.0.0.0/8;[::9]"
    );
}

const SCUTIL_SORTED: &str = "<dictionary> {
  ExceptionsList : <array> {
    0 : *.local
    1 : 169.254/16
    2 : internal.test
  }
  FTPPassive : 1
  HTTPEnable : 1
  HTTPPort : 8080
  HTTPProxy : http.proxy.test
  HTTPSEnable : 1
  HTTPSPort : 8443
  HTTPSProxy : secure.proxy.test
  SOCKSEnable : 1
  SOCKSPort : 1080
  SOCKSProxy : socks.proxy.test
  __SCOPED__ : <dictionary> {
    en0 : <dictionary> {
      HTTPSEnable : 1
      HTTPSPort : 1
      HTTPSProxy : scoped.test
    }
  }
}
";

#[test]
fn macos_settings_are_read_by_key_regardless_of_line_order() {
    let found = system::macos(SCUTIL_SORTED).unwrap();
    assert_eq!(
        found,
        system(
            "http://secure.proxy.test:8443",
            "*.local,169.254/16,internal.test"
        )
    );
    let http_only = SCUTIL_SORTED.replace(
        "HTTPSEnable : 1\n  HTTPSPort",
        "HTTPSEnable : 0\n  HTTPSPort",
    );
    assert_eq!(
        system::macos(&http_only).unwrap().endpoint.as_str(),
        "http://http.proxy.test:8080/"
    );
    let socks_only =
        "<dictionary> {\n  SOCKSEnable : 1\n  SOCKSPort : 1080\n  SOCKSProxy : s.test\n}\n";
    assert!(system::macos(socks_only).is_none());
    assert!(system::macos("<dictionary> {\n  HTTPEnable : 1\n").is_none());
    let ipv6 = "<dictionary> {\n  HTTPEnable : 1\n  HTTPPort : 3128\n  HTTPProxy : fd00::1\n}\n";
    assert_eq!(
        system::macos(ipv6).unwrap().endpoint.as_str(),
        "http://[fd00::1]:3128/"
    );
    let credentials =
        "<dictionary> {\n  HTTPEnable : 1\n  HTTPPort : 3128\n  HTTPProxy : user@p.test\n}\n";
    assert!(system::macos(credentials).is_none());
}

#[test]
fn windows_values_follow_the_reference_server_and_override_forms() {
    let values = |enabled, server: &str, bypass: &str| system::WindowsValues {
        enabled,
        server: server.into(),
        bypass: bypass.into(),
    };
    assert_eq!(
        system::windows(values(true, "proxy.test:8080", "*.corp;<local>")).unwrap(),
        system("http://proxy.test:8080", "*.corp,<local>")
    );
    assert_eq!(
        system::windows(values(
            true,
            "ftp=f.test:21;http=h.test:80;https=s.test:443",
            ""
        ))
        .unwrap()
        .endpoint
        .as_str(),
        "http://h.test/"
    );
    assert_eq!(
        system::windows(values(true, "https=s.test:443;http=h.test:80", ""))
            .unwrap()
            .endpoint
            .as_str(),
        "http://s.test:443/"
    );
    assert!(system::windows(values(true, "socks=s.test:1080", "")).is_none());
    assert!(system::windows(values(false, "proxy.test:8080", "")).is_none());
    assert!(system::windows(values(true, "http://u:p@proxy.test:8080", "")).is_none());
}

struct Fake {
    tools: Vec<&'static str>,
    values: HashMap<String, String>,
    calls: Vec<String>,
}

impl Reader for Fake {
    fn available(&self, program: &str) -> bool {
        self.tools.contains(&program)
    }
    fn read(&mut self, program: &str, args: &[&str], _deadline: Instant) -> Option<String> {
        let key = format!("{program} {}", args.join(" "));
        self.calls.push(key.clone());
        self.values.get(&key).cloned()
    }
}

fn kde(values: &[(&str, &str)]) -> Fake {
    Fake {
        tools: vec!["kreadconfig5", "kreadconfig6", "gsettings"],
        values: values
            .iter()
            .map(|(k, v)| {
                (
                    format!("kreadconfig6 --file kioslaverc --group Proxy Settings --key {k}"),
                    (*v).to_owned(),
                )
            })
            .collect(),
        calls: Vec::new(),
    }
}

fn gnome(values: &[(&str, &str)]) -> Fake {
    Fake {
        tools: vec!["gsettings"],
        values: values
            .iter()
            .map(|(k, v)| (format!("gsettings get {k}"), (*v).to_owned()))
            .collect(),
        calls: Vec::new(),
    }
}

#[test]
fn kde_wins_mixed_markers_and_reads_manual_settings_only() {
    let deadline = Instant::now() + Duration::from_secs(1);
    let desktop = env(&[("XDG_CURRENT_DESKTOP", "KDE:GNOME")]);
    let mut reader = kde(&[
        ("ProxyType", "1"),
        ("ReversedException", "false"),
        ("httpsProxy", ""),
        ("httpProxy", "http://kde.proxy.test 3128"),
        ("NoProxyFor", "a.test;b.test"),
    ]);
    assert_eq!(
        system::linux(&desktop, &mut reader, deadline).unwrap(),
        system("http://kde.proxy.test:3128", "a.test,b.test")
    );
    assert!(
        reader
            .calls
            .iter()
            .all(|call| !call.starts_with("gsettings"))
    );
    for (key, value) in [("ProxyType", "2"), ("ReversedException", "true")] {
        let mut reader = kde(&[
            ("ProxyType", "1"),
            ("ReversedException", "false"),
            ("httpsProxy", "https://s.test:443"),
            ("NoProxyFor", ""),
            (key, value),
        ]);
        assert!(
            system::linux(&desktop, &mut reader, deadline).is_none(),
            "{key}"
        );
    }
    let mut missing = Fake {
        tools: vec!["gsettings"],
        values: HashMap::new(),
        calls: Vec::new(),
    };
    assert!(system::linux(&desktop, &mut missing, deadline).is_none());
    assert!(missing.calls.is_empty());
}

#[test]
fn gnome_reads_a_whitelist_and_rejects_automatic_or_authenticated_modes() {
    let deadline = Instant::now() + Duration::from_secs(1);
    let desktop = env(&[("XDG_CURRENT_DESKTOP", "ubuntu:GNOME")]);
    let base = [
        ("org.gnome.system.proxy mode", "'manual'"),
        ("org.gnome.system.proxy.http use-authentication", "false"),
        ("org.gnome.system.proxy use-same-proxy", "false"),
        ("org.gnome.system.proxy.https host", "''"),
        ("org.gnome.system.proxy.https port", "0"),
        ("org.gnome.system.proxy.http host", "'gnome.proxy.test'"),
        ("org.gnome.system.proxy.http port", "8080"),
        (
            "org.gnome.system.proxy ignore-hosts",
            "['localhost', '127.0.0.0/8', '::1', 'it\\'s.test']",
        ),
    ];
    let mut reader = gnome(&base);
    assert_eq!(
        system::linux(&desktop, &mut reader, deadline).unwrap(),
        system(
            "http://gnome.proxy.test:8080",
            "localhost,127.0.0.0/8,::1,it's.test"
        )
    );
    assert!(
        reader
            .calls
            .iter()
            .all(|call| !call.contains("password") && !call.contains("list-recursively"))
    );
    let mut empty = base.to_vec();
    empty[7] = ("org.gnome.system.proxy ignore-hosts", "@as []");
    assert_eq!(
        system::linux(&desktop, &mut gnome(&empty), deadline)
            .unwrap()
            .bypass,
        ""
    );
    for (index, value) in [(0, "'auto'"), (0, "'none'"), (1, "true"), (7, "[1, 2]")] {
        let mut changed = base.to_vec();
        changed[index].1 = value;
        assert!(
            system::linux(&desktop, &mut gnome(&changed), deadline).is_none(),
            "{index} {value}"
        );
    }
    assert!(
        system::linux(
            &env(&[("XDG_CURRENT_DESKTOP", "XFCE")]),
            &mut gnome(&base),
            deadline
        )
        .is_none()
    );
}

#[cfg(unix)]
mod process {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    #[test]
    fn subprocess_reads_are_bounded_minimal_and_reap_their_group() {
        let dir = tempfile::tempdir().unwrap();
        let marker = dir.path().join("grandchild.pid");
        // One script: the first launch absorbs any OS scan of a new executable.
        let script = dir.path().join("tool");
        std::fs::write(
            &script,
            format!(
                "#!/bin/sh\ncase \"$1\" in\n value) printf 'hello';;\n environment) env | cut -d= -f1 | sort;;\n \
                 failing) printf 'partial'; exit 3;;\n flood) head -c 70000 /dev/zero | tr '\\0' 'x';;\n \
                 hang) sleep 30 & echo $! > '{}'; wait;;\nesac\n",
                marker.display()
            ),
        )
        .unwrap();
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o700)).unwrap();
        let env = env(&[
            ("PATH", &format!("{}:/usr/bin:/bin", dir.path().display())),
            ("HOME", "/nonexistent-home"),
            ("OPENAI_API_KEY", "must-not-pass"),
            ("HTTPS_PROXY", "http://must-not-pass.test"),
        ]);
        let mut reader = system::ProcessReader { env: &env };
        let soon = || Instant::now() + Duration::from_secs(10);
        assert!(reader.available("tool") && !reader.available("absent"));
        assert_eq!(
            reader.read("tool", &["value"], soon()).as_deref(),
            Some("hello")
        );
        let names = reader.read("tool", &["environment"], soon()).unwrap();
        assert!(names.lines().any(|line| line == "HOME"));
        assert!(!names.contains("OPENAI_API_KEY") && !names.contains("HTTPS_PROXY"));
        assert!(reader.read("tool", &["failing"], soon()).is_none());
        assert!(reader.read("tool", &["flood"], soon()).is_none());
        let started = Instant::now();
        assert!(
            reader
                .read("tool", &["hang"], Instant::now() + Duration::from_secs(2))
                .is_none()
        );
        assert!(started.elapsed() < Duration::from_secs(8));
        let pid: i32 = std::fs::read_to_string(&marker)
            .expect("the hanging tool started within its deadline")
            .trim()
            .parse()
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(5);
        while unsafe { libc::kill(pid, 0) } == 0 {
            assert!(
                Instant::now() < deadline,
                "the reader's process group survived"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(reader.read("tool", &["value"], Instant::now()).is_none());
    }
}
