//! What `markitai serve` tells the person who started it. The lines are built by
//! pure functions so wording, language and warnings can be tested.
//!
//! Every line follows the terminal language and the CLI's terminal style: the
//! title carries the listening address, the items below it the token, where to
//! open the workbench, where data lives and how to stop. Scripts read the
//! title, which ends with `http://ADDRESS`, and the token line, which ends with
//! the token, in every language. Each line has a [`Kind`] that decides how the
//! terminal shows it: an address to open sits on a line of its own, never
//! wrapped, so it copies back whole.
use super::launch;
use crate::app::i18n::{Lang, text};
use std::{
    io::ErrorKind,
    net::{IpAddr, SocketAddr},
    path::Path,
};

pub(super) struct Startup<'a> {
    /// The address the listener actually bound.
    pub address: SocketAddr,
    /// `None` with `--no-auth`.
    pub token: Option<&'a str>,
    /// Where jobs and their history live.
    pub data: &'a Path,
    /// This computer's address on the local network, when it could be found.
    pub network: Option<IpAddr>,
    /// A newer release the update check found: (latest, current, command).
    pub newer: Option<(String, String, String)>,
    /// The version an update tried and undid.
    pub failed_update: Option<String>,
    /// The version running.
    pub current: String,
}

/// Whether other machines can reach the listener. Loopback peers only are the
/// default; anything else publishes the service.
pub(super) fn exposed(address: SocketAddr) -> bool {
    !address.ip().to_canonical().is_loopback()
}

/// Best effort: the address a route to the outside would use. Connecting a UDP
/// socket only selects the interface; nothing is sent.
pub(super) fn network_address() -> Option<IpAddr> {
    let socket = std::net::UdpSocket::bind(("0.0.0.0", 0)).ok()?;
    socket.connect(("192.0.2.1", 80)).ok()?;
    let ip = socket.local_addr().ok()?.ip();
    (!ip.is_loopback() && !ip.is_unspecified()).then_some(ip)
}

/// How a startup line is shown.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Kind {
    /// `◆ sentence http://ADDRESS`, never wrapped: scripts read the address.
    Title,
    /// `  • label: token`, never wrapped: scripts read the token.
    Token,
    /// `  • sentence`.
    Fact,
    /// `  → sentence`: what to do next.
    Action,
    /// `  ! sentence`.
    Warning,
    /// A further sentence of the warning above, under its text.
    More,
    /// An address to open, on its own line in the accent colour, never wrapped.
    Address,
    /// `  • New version 2.2.0 (current 2.1.0): markitai update`, one line.
    Notice,
}

pub(super) fn lines(lang: Lang, startup: &Startup<'_>) -> Vec<(Kind, String)> {
    let Startup {
        address,
        token,
        data,
        network,
        ..
    } = *startup;
    let mut lines = vec![(
        Kind::Title,
        text!(lang =>
            "Markitai server listening on http://{address}",
            "Markitai 服务正在监听 http://{address}"),
    )];
    if let Some(failed) = &startup.failed_update {
        let current = &startup.current;
        lines.push((
            Kind::Warning,
            text!(lang =>
                "The update to {failed} failed; still {current}",
                "更新到 {failed} 失败，仍是 {current}"),
        ));
    }
    if exposed(address) {
        if token.is_some() {
            lines.push((
                Kind::Warning,
                text!(lang =>
                "Binding to {address} makes this server reachable from your network.",
                "绑定到 {address} 会让网络中的其他设备能够访问这个服务。"),
            ));
            lines.push((Kind::More, text!(lang =>
                "Other machines must send the access token; treat it, and any address that contains it, like a password.",
                "其他设备必须提供访问令牌；请像对待密码一样对待令牌和包含令牌的地址。")));
        } else {
            lines.push((Kind::Warning, text!(lang =>
                "Binding to {address} WITHOUT authentication (--no-auth) lets anyone who can reach this address convert files and read, download or delete your whole conversion history.",
                "绑定到 {address} 且未启用认证（--no-auth），任何能访问这个地址的人都可以转换文件，并读取、下载或删除你的全部转换历史。")));
            lines.push((Kind::More, text!(lang =>
                "Drop --no-auth to require the access token, keep the default --host 127.0.0.1, or put the service behind a proxy that authenticates.",
                "请去掉 --no-auth 以要求访问令牌，保持默认的 --host 127.0.0.1，或把服务放在带认证的代理之后。")));
        }
    }
    if let Some(token) = token {
        lines.push((
            Kind::Token,
            text!(lang => "Access token: {token}", "访问令牌：{token}"),
        ));
    }
    // Every API client authenticates, including this computer's browser. A
    // fragment keeps the token out of the initial HTTP request and access log.
    let local = launch::browser_url(address, token);
    lines.push((
        Kind::Action,
        text!(lang => "Open in your browser:", "在浏览器中打开："),
    ));
    lines.push((Kind::Address, local));
    if address.ip().is_unspecified()
        && let Some(ip) = network
    {
        let remote = launch::address_url(SocketAddr::new(ip, address.port()), token);
        lines.push((
            Kind::Action,
            if token.is_some() {
                text!(lang =>
                    "From another device on your network (this address contains the access token):",
                    "从网络中的其他设备访问（该地址包含访问令牌）：")
            } else {
                text!(lang =>
                    "From another device on your network:",
                    "从网络中的其他设备访问：")
            },
        ));
        lines.push((Kind::Address, remote));
    }
    let data = data.display();
    lines.push((
        Kind::Fact,
        text!(lang =>
            "Jobs and history: {data}",
            "任务和历史：{data}"),
    ));
    if let Some((latest, current, command)) = &startup.newer {
        let separator = if lang == Lang::En { " " } else { "" };
        lines.push((
            Kind::Notice,
            format!(
                "  • {}{separator}{command}",
                text!(lang =>
                    "New version {latest} (current {current}):",
                    "有新版本 {latest}（当前 {current}）：")
            ),
        ));
    }
    lines.push((
        Kind::Fact,
        text!(lang =>
            "Press Ctrl-C to stop; running conversions finish and history is saved first.",
            "按 Ctrl-C 停止；正在转换的任务会先完成，历史也会先保存。"),
    ));
    lines
}

/// What a server that restarted into a new version prints instead of the
/// whole block: the address, and that the token stayed.
pub(super) fn print_restarted(lang: Lang, address: SocketAddr, token: bool, from: &str) {
    use crate::app::logging::{Shape, blank, console};
    use crate::term::{Mark, Style, Tone};
    let (version, _) = crate::update::current_identity();
    let title = text!(lang =>
        "Markitai {version} restarted, listening on",
        "Markitai {version} 已重启，监听");
    blank();
    console(
        Shape {
            mark: Some(Mark::Title),
            indent: 0,
            tone: None,
            wrap: false,
        },
        &format!(
            "{} http://{address}",
            Style::stderr().paint(Tone::Structure, &title)
        ),
    );
    blank();
    let line = match (token, lang) {
        (true, Lang::En) => format!("Updated from {from}; the access token is unchanged"),
        (true, Lang::Zh) => format!("已从 {from} 更新，访问令牌不变"),
        (false, Lang::En) => format!("Updated from {from}"),
        (false, Lang::Zh) => format!("已从 {from} 更新"),
    };
    console(Shape::item(Mark::Success), &line);
}

/// Why the listener could not be created, with what to try next.
pub(super) fn bind_error(lang: Lang, host: &str, port: u16, error: &std::io::Error) -> String {
    let target = if host.contains(':') && !host.starts_with('[') {
        format!("[{host}]:{port}")
    } else {
        format!("{host}:{port}")
    };
    match error.kind() {
        ErrorKind::AddrInUse => text!(lang =>
            "Cannot listen on {target}: the port is already in use. Stop the program using it, or choose another port with --port (--port 0 picks a free one).",
            "无法监听 {target}：端口已被占用。请停止占用它的程序，或用 --port 选择其他端口（--port 0 会自动选择空闲端口）。"),
        ErrorKind::AddrNotAvailable => text!(lang =>
            "Cannot listen on {target}: this computer has no such address. Use --host 127.0.0.1, 0.0.0.0 or the address of one of its network interfaces.",
            "无法监听 {target}：这台电脑没有这个地址。请使用 --host 127.0.0.1、0.0.0.0 或本机某个网卡的地址。"),
        ErrorKind::PermissionDenied => text!(lang =>
            "Cannot listen on {target}: permission denied. Ports below 1024 usually need extra privileges; choose a higher port with --port.",
            "无法监听 {target}：没有权限。1024 以下的端口通常需要额外权限，请用 --port 选择更大的端口。"),
        _ => text!(lang =>
            "Cannot listen on {target}: {error}",
            "无法监听 {target}：{error}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn startup<'a>(
        address: &str,
        token: Option<&'a str>,
        network: Option<&str>,
        data: &'a Path,
    ) -> Startup<'a> {
        Startup {
            address: address.parse().unwrap(),
            token,
            data,
            network: network.map(|ip| ip.parse().unwrap()),
            newer: None,
            failed_update: None,
            current: "2.1.0".into(),
        }
    }

    fn texts(lines: &[(Kind, String)]) -> Vec<&str> {
        lines.iter().map(|(_, text)| text.as_str()).collect()
    }

    #[test]
    fn a_loopback_listener_prints_where_to_go_where_data_lives_and_how_to_stop() {
        let data = Path::new("/tmp/home/serve/jobs");
        let lines = lines(
            Lang::En,
            &startup("127.0.0.1:3600", Some("secret"), None, data),
        );
        assert_eq!(
            lines,
            [
                (
                    Kind::Title,
                    "Markitai server listening on http://127.0.0.1:3600".into()
                ),
                (Kind::Token, "Access token: secret".into()),
                (Kind::Action, "Open in your browser:".into()),
                // The local browser receives a fragment token too.
                (Kind::Address, "http://127.0.0.1:3600/#token=secret".into()),
                (Kind::Fact, "Jobs and history: /tmp/home/serve/jobs".into()),
                (
                    Kind::Fact,
                    "Press Ctrl-C to stop; running conversions finish and history is saved first."
                        .into()
                ),
            ]
        );
        // No network warning.
        assert!(lines.iter().all(|(kind, _)| *kind != Kind::Warning));
    }

    #[test]
    fn every_line_follows_the_language_and_keeps_its_value_last() {
        let data = Path::new("/tmp/serve/jobs");
        let english = lines(
            Lang::En,
            &startup("127.0.0.1:3600", Some("secret"), None, data),
        );
        let chinese = lines(
            Lang::Zh,
            &startup("127.0.0.1:3600", Some("secret"), None, data),
        );
        assert_eq!(english.len(), chinese.len());
        assert_eq!(
            chinese,
            [
                (
                    Kind::Title,
                    "Markitai 服务正在监听 http://127.0.0.1:3600".into()
                ),
                (Kind::Token, "访问令牌：secret".into()),
                (Kind::Action, "在浏览器中打开：".into()),
                (Kind::Address, "http://127.0.0.1:3600/#token=secret".into()),
                (Kind::Fact, "任务和历史：/tmp/serve/jobs".into()),
                (
                    Kind::Fact,
                    "按 Ctrl-C 停止；正在转换的任务会先完成，历史也会先保存。".into()
                ),
            ]
        );
        // What scripts read ends both lines in either language.
        for lines in [&english, &chinese] {
            assert!(lines[0].1.ends_with(" http://127.0.0.1:3600"));
            assert!(lines[1].1.ends_with("secret"));
        }
    }

    #[test]
    fn a_network_listener_with_a_token_warns_and_offers_the_address_for_other_devices() {
        let data = Path::new("/d");
        let lines = lines(
            Lang::En,
            &startup("0.0.0.0:3600", Some("a b"), Some("192.168.1.20"), data),
        );
        let text = texts(&lines);
        assert_eq!(text[0], "Markitai server listening on http://0.0.0.0:3600");
        assert_eq!(lines[1].0, Kind::Warning);
        assert!(text[1].starts_with("Binding to 0.0.0.0:3600 makes this server reachable"));
        assert_eq!(lines[2].0, Kind::More);
        assert!(text[2].contains("like a password"));
        assert_eq!(lines[3], (Kind::Token, "Access token: a b".into()));
        assert_eq!(lines[4], (Kind::Action, "Open in your browser:".into()));
        assert_eq!(
            lines[5],
            (Kind::Address, "http://127.0.0.1:3600/#token=a+b".into())
        );
        assert!(text[6].contains("contains the access token"), "{}", text[6]);
        assert_eq!(
            lines[7],
            (Kind::Address, "http://192.168.1.20:3600/#token=a+b".into())
        );
        assert!(text.iter().all(|line| !line.contains("WITHOUT")));
    }

    #[test]
    fn a_network_listener_without_authentication_gets_the_stronger_warning() {
        let data = Path::new("/d");
        for lang in [Lang::En, Lang::Zh] {
            let lines = lines(lang, &startup("0.0.0.0:3600", None, Some("10.0.0.5"), data));
            assert!(lines.iter().all(|(kind, _)| *kind != Kind::Token));
            let (kind, warning) = lines
                .iter()
                .find(|(_, line)| line.contains("--no-auth"))
                .unwrap();
            assert_eq!(*kind, Kind::Warning);
            assert!(
                warning.contains("WITHOUT") || warning.contains("未启用认证"),
                "{warning}"
            );
            // The address for other devices carries no token because there is none.
            assert!(lines.contains(&(Kind::Address, "http://10.0.0.5:3600/".into())));
        }
        // A loopback listener needs no network-exposure warning; no-auth omits the token.
        let quiet = lines(
            Lang::En,
            &startup("[::1]:3600", None, None, Path::new("/d")),
        );
        assert!(quiet.iter().all(|(kind, _)| *kind != Kind::Warning));
        assert!(quiet.contains(&(Kind::Address, "http://[::1]:3600/".into())));
    }

    #[test]
    fn a_specific_network_address_is_opened_with_the_token_even_from_this_computer() {
        let data = Path::new("/d");
        let lines = lines(
            Lang::En,
            &startup("192.168.1.20:3600", Some("tok"), None, data),
        );
        assert_eq!(lines[1].0, Kind::Warning);
        assert!(lines[1].1.starts_with("Binding to 192.168.1.20:3600"));
        assert!(
            lines.contains(&(Kind::Address, "http://192.168.1.20:3600/#token=tok".into())),
            "{lines:?}"
        );
    }

    #[test]
    fn a_newer_release_sits_before_the_stop_line_and_a_failed_update_after_the_title() {
        let data = Path::new("/d");
        for (lang, notice, failed) in [
            (
                Lang::Zh,
                "  • 有新版本 2.2.0（当前 2.1.0）：markitai update",
                "更新到 2.2.0 失败，仍是 2.1.0",
            ),
            (
                Lang::En,
                "  • New version 2.2.0 (current 2.1.0): markitai update",
                "The update to 2.2.0 failed; still 2.1.0",
            ),
        ] {
            let mut with = startup("127.0.0.1:3600", Some("t"), None, data);
            with.newer = Some(("2.2.0".into(), "2.1.0".into(), "markitai update".into()));
            with.failed_update = Some("2.2.0".into());
            let lines = lines(lang, &with);
            let at = lines.len() - 2;
            assert_eq!(lines[at], (Kind::Notice, notice.into()));
            assert_eq!(lines[1], (Kind::Warning, failed.into()));
        }
    }

    #[test]
    fn bind_failures_name_the_address_and_say_what_to_try() {
        let in_use = std::io::Error::from(ErrorKind::AddrInUse);
        let english = bind_error(Lang::En, "127.0.0.1", 3600, &in_use);
        assert!(
            english.contains("127.0.0.1:3600") && english.contains("--port"),
            "{english}"
        );
        let chinese = bind_error(Lang::Zh, "::1", 3600, &in_use);
        assert!(
            chinese.contains("[::1]:3600") && chinese.contains("--port"),
            "{chinese}"
        );
        for kind in [ErrorKind::AddrNotAvailable, ErrorKind::PermissionDenied] {
            let text = bind_error(Lang::En, "10.9.9.9", 80, &std::io::Error::from(kind));
            assert!(
                text.contains("10.9.9.9:80") && text.contains("--"),
                "{text}"
            );
        }
        let other = bind_error(Lang::En, "x", 1, &std::io::Error::other("boom"));
        assert_eq!(other, "Cannot listen on x:1: boom");
    }
}
