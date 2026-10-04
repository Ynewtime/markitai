//! What `markitai serve` tells the person who started it. The lines are built by
//! pure functions so wording, language and warnings can be tested.
//!
//! The first line (`Markitai server listening on …`) and the token line
//! (`Remote access token: …`) are read by scripts and stay English in every
//! language; the sentences around them follow the terminal language.
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

pub(super) fn lines(lang: Lang, startup: &Startup<'_>) -> Vec<String> {
    let Startup {
        address,
        token,
        data,
        network,
    } = *startup;
    let mut lines = vec![format!("Markitai server listening on http://{address}")];
    if exposed(address) {
        if token.is_some() {
            lines.push(text!(lang =>
                "Warning: binding to {address} makes this server reachable from your network.",
                "警告：绑定到 {address} 会让网络中的其他设备能够访问这个服务。"));
            lines.push(text!(lang =>
                "         Other machines must send the access token; treat it, and any address that contains it, like a password.",
                "     其他设备必须提供访问令牌；请像对待密码一样对待令牌和包含令牌的地址。"));
        } else {
            lines.push(text!(lang =>
                "Warning: binding to {address} WITHOUT authentication (--no-auth) lets anyone who can reach this address convert files and read, download or delete your whole conversion history.",
                "警告：绑定到 {address} 且未启用认证（--no-auth），任何能访问这个地址的人都可以转换文件，并读取、下载或删除你的全部转换历史。"));
            lines.push(text!(lang =>
                "         Drop --no-auth to require the access token, keep the default --host 127.0.0.1, or put the service behind a proxy that authenticates.",
                "     请去掉 --no-auth 以要求访问令牌，保持默认的 --host 127.0.0.1，或把服务放在带认证的代理之后。"));
        }
    }
    if let Some(token) = token {
        lines.push(format!("Remote access token: {token}"));
    }
    // Every API client authenticates, including this computer's browser. A
    // fragment keeps the token out of the initial HTTP request and access log.
    let local = launch::browser_url(address, token);
    lines.push(text!(lang =>
        "Open in your browser: {local}",
        "在浏览器中打开：{local}"));
    if address.ip().is_unspecified()
        && let Some(ip) = network
    {
        let remote = launch::address_url(SocketAddr::new(ip, address.port()), token);
        lines.push(if token.is_some() {
            text!(lang =>
                "From another device on your network: {remote} (this address contains the access token)",
                "从网络中的其他设备访问：{remote}（该地址包含访问令牌）")
        } else {
            text!(lang =>
                "From another device on your network: {remote}",
                "从网络中的其他设备访问：{remote}")
        });
    }
    let data = data.display();
    lines.push(text!(lang =>
        "Jobs and history are stored in {data}",
        "任务和历史保存在 {data}"));
    lines.push(text!(lang =>
        "Press Ctrl-C to stop; running conversions finish and history is saved first.",
        "按 Ctrl-C 停止；正在转换的任务会先完成，历史也会先保存。"));
    lines
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
        }
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
                "Markitai server listening on http://127.0.0.1:3600",
                "Remote access token: secret",
                "Open in your browser: http://127.0.0.1:3600/#token=secret",
                "Jobs and history are stored in /tmp/home/serve/jobs",
                "Press Ctrl-C to stop; running conversions finish and history is saved first.",
            ]
        );
        // No network warning; the local browser receives a fragment token too.
        assert!(lines.iter().all(|line| !line.contains("Warning")));
        assert!(lines[2].contains("#token=secret"));
    }

    #[test]
    fn the_scripted_lines_do_not_change_with_the_language() {
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
        assert_eq!(english[..2], chinese[..2]);
        assert!(chinese[2].starts_with("在浏览器中打开：http://127.0.0.1:3600/"));
        assert!(chinese[3].contains("/tmp/serve/jobs"));
        assert!(chinese[4].contains("Ctrl-C"));
    }

    #[test]
    fn a_network_listener_with_a_token_warns_and_offers_the_address_for_other_devices() {
        let data = Path::new("/d");
        let lines = lines(
            Lang::En,
            &startup("0.0.0.0:3600", Some("a b"), Some("192.168.1.20"), data),
        );
        assert_eq!(lines[0], "Markitai server listening on http://0.0.0.0:3600");
        assert!(
            lines[1].starts_with("Warning: binding to 0.0.0.0:3600 makes this server reachable")
        );
        assert!(lines[2].contains("like a password"));
        assert_eq!(lines[3], "Remote access token: a b");
        assert_eq!(
            lines[4],
            "Open in your browser: http://127.0.0.1:3600/#token=a+b"
        );
        assert!(
            lines[5].contains("http://192.168.1.20:3600/#token=a+b"),
            "{}",
            lines[5]
        );
        assert!(lines[5].contains("contains the access token"));
        assert!(lines.iter().all(|line| !line.contains("WITHOUT")));
    }

    #[test]
    fn a_network_listener_without_authentication_gets_the_stronger_warning() {
        let data = Path::new("/d");
        for lang in [Lang::En, Lang::Zh] {
            let lines = lines(lang, &startup("0.0.0.0:3600", None, Some("10.0.0.5"), data));
            assert!(
                lines
                    .iter()
                    .all(|line| !line.contains("Remote access token"))
            );
            let warning = lines
                .iter()
                .find(|line| line.contains("--no-auth"))
                .unwrap();
            assert!(
                warning.contains("WITHOUT") || warning.contains("未启用认证"),
                "{warning}"
            );
            // The address for other devices carries no token because there is none.
            assert!(
                lines
                    .iter()
                    .any(|line| line.contains("http://10.0.0.5:3600/") && !line.contains("#token"))
            );
        }
        // A loopback listener needs no network-exposure warning; no-auth omits the token.
        let quiet = lines(
            Lang::En,
            &startup("[::1]:3600", None, None, Path::new("/d")),
        );
        assert!(quiet.iter().all(|line| !line.contains("Warning")));
        assert!(
            quiet
                .iter()
                .any(|line| line == "Open in your browser: http://[::1]:3600/")
        );
    }

    #[test]
    fn a_specific_network_address_is_opened_with_the_token_even_from_this_computer() {
        let data = Path::new("/d");
        let lines = lines(
            Lang::En,
            &startup("192.168.1.20:3600", Some("tok"), None, data),
        );
        assert!(lines[1].starts_with("Warning: binding to 192.168.1.20:3600"));
        assert!(
            lines
                .iter()
                .any(|line| line == "Open in your browser: http://192.168.1.20:3600/#token=tok"),
            "{lines:?}"
        );
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
