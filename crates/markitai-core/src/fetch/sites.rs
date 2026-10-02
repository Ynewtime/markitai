//! Sites that turn automated clients away, and what to tell the reader.
//!
//! Some sites answer a client they take for a program with a refusal: a
//! status 403, a JSON error, a verification page served with status 200, or a
//! script challenge that only a person's browser passes. Markitai does not
//! forge a browser, solve a challenge or sign requests, so such a site stays
//! refused; this module only recognizes the refusal, names the site, and says
//! what does work: converting the page the reader saved from their own
//! browser, or giving their own browser session's cookies to the local
//! browser (`fetch.playwright.cookies`, see `docs/fetch.md`).

use serde_json::Value;
use url::Url;

/// What to do instead, in every refusal.
const SAVE_PAGE: &str = "open the page in your browser and save it (File > Save Page As…, 'Webpage, HTML Only'), then convert the saved file";

struct Site {
    name: &'static str,
    /// Registrable domains the site's pages are served from.
    domains: &'static [&'static str],
    /// The cause, when it is known.
    cause: &'static str,
    /// Whether the reader's own cookies can help (a login the site expects).
    cookies: Option<&'static str>,
}

const SITES: &[Site] = &[
    Site {
        name: "Zhihu",
        domains: &["zhihu.com"],
        cause: "refuses automated clients",
        cookies: Some("zhihu.com"),
    },
    Site {
        name: "WeChat",
        domains: &["weixin.qq.com"],
        cause: "asks automated clients to complete a verification",
        cookies: None,
    },
    Site {
        name: "Douban",
        domains: &["douban.com"],
        cause: "shows a security check to automated clients",
        cookies: Some("douban.com"),
    },
    Site {
        name: "Weibo",
        domains: &["weibo.com", "weibo.cn"],
        cause: "asks automated clients for a login or a visitor check",
        cookies: Some("weibo.com"),
    },
    Site {
        name: "Toutiao",
        domains: &["toutiao.com"],
        cause: "serves its pages only to a browser that runs its scripts",
        cookies: None,
    },
    Site {
        name: "Reddit",
        domains: &["reddit.com"],
        cause: "asks automated clients to prove they are human",
        cookies: Some("reddit.com"),
    },
    Site {
        name: "Quora",
        domains: &["quora.com"],
        cause: "refuses automated clients behind a Cloudflare bot check",
        cookies: None,
    },
    Site {
        name: "Stack Overflow",
        domains: &["stackoverflow.com", "stackexchange.com"],
        cause: "refuses automated clients behind a Cloudflare bot check",
        cookies: None,
    },
    Site {
        name: "Medium",
        domains: &["medium.com"],
        cause: "refuses automated clients behind a Cloudflare bot check",
        cookies: None,
    },
    Site {
        name: "Hashnode",
        domains: &["hashnode.dev", "hashnode.com"],
        cause: "refuses automated clients behind a Cloudflare bot check",
        cookies: None,
    },
];

fn site_of(host: &str) -> Option<&'static Site> {
    let host = host.trim_end_matches('.').to_ascii_lowercase();
    SITES.iter().find(|site| {
        site.domains.iter().any(|domain| {
            host == *domain
                || host
                    .strip_suffix(domain)
                    .is_some_and(|rest| rest.ends_with('.'))
        })
    })
}

fn advice(site: &Site) -> String {
    let mut advice = format!("{} {}; {SAVE_PAGE}", site.name, site.cause);
    if let Some(domain) = site.cookies {
        advice.push_str(&format!(
            "; or give the local browser your own logged-in cookies for {domain} (fetch.playwright.cookies, see docs/fetch.md)"
        ));
    }
    advice
}

/// The hint for a refusal status (401, 403, 418, 429) from a site this module
/// knows, or from any site whose answer says it is a Cloudflare bot check.
pub(super) fn refusal_hint(url: &Url, cloudflare_challenge: bool) -> Option<String> {
    if let Some(site) = url.host_str().and_then(site_of) {
        return Some(advice(site));
    }
    cloudflare_challenge
        .then(|| format!("the site's Cloudflare bot check refused this client; {SAVE_PAGE}"))
}

/// What a JSON refusal body says (`{"error":{"message":"…","code":40362}}`),
/// cleaned for a one-line message: its own words, without control characters,
/// cut to 120 characters.
pub(super) fn json_refusal(body: &[u8]) -> Option<String> {
    let value: Value = serde_json::from_slice(body).ok()?;
    value.as_object()?;
    let text = |pointer: &str| {
        value
            .pointer(pointer)
            .and_then(Value::as_str)
            .map(|text| {
                text.chars()
                    .filter(|ch| !ch.is_control() || ch.is_whitespace())
                    .collect::<String>()
                    .split_whitespace()
                    .collect::<Vec<_>>()
                    .join(" ")
            })
            .filter(|text| !text.is_empty())
    };
    let message = [
        "/error/message",
        "/message",
        "/msg",
        "/errmsg",
        "/error_description",
        "/detail",
        "/error",
    ]
    .iter()
    .find_map(|pointer| text(pointer))?;
    let code = ["/error/code", "/code", "/errcode"]
        .iter()
        .find_map(|pointer| value.pointer(pointer)?.as_i64());
    let mut said: String = message.chars().take(120).collect();
    if said.chars().count() < message.chars().count() {
        said.push('…');
    }
    Some(match code {
        Some(code) => format!("{said} (code {code})"),
        None => said,
    })
}

/// The marker of failures a verification page produces; `auto` renders such a
/// page with the local browser, which an ordinary visitor's page needs.
pub(super) const VERIFICATION_PAGE: &str = "served a verification page";

/// A page that is a site's verification or security check rather than the
/// content, served with a success status: the site, its message and what to
/// do. `url` is the address the page was finally read from.
pub(super) fn verification_page(url: &Url, html: &str) -> Option<String> {
    let host = url.host_str()?.to_ascii_lowercase();
    let site = site_of(&host)?;
    let found = match site.name {
        "WeChat" => {
            url.path().contains("wappoc_appmsgcaptcha")
                || (host == "mp.weixin.qq.com"
                    && html.contains("环境异常")
                    && html.contains("完成验证"))
        }
        "Douban" => host == "sec.douban.com",
        "Weibo" => host == "passport.weibo.com" && url.path().starts_with("/visitor"),
        "Reddit" => html.contains("Prove your humanity"),
        // ByteDance's script-challenge page: an empty body and a bytecode VM.
        "Toutiao" => html.contains("_$jsvmprt"),
        // The interstitial of a client its script cannot run in, and the
        // security check that asks for a login before any page.
        "Zhihu" => url.path().starts_with("/account/unhuman") || html.contains("id=\"zh-zse-ck\""),
        _ => false,
    };
    found.then(|| {
        let mut message = format!(
            "{} {VERIFICATION_PAGE} instead of the content: it {}; {SAVE_PAGE}",
            site.name, site.cause
        );
        if let Some(domain) = site.cookies {
            message.push_str(&format!(
                "; or give the local browser your own logged-in cookies for {domain} (fetch.playwright.cookies, see docs/fetch.md)"
            ));
        }
        message
    })
}

/// A browser failure `Browser navigation returned HTTP <status>` as its status.
pub(super) fn browser_status(message: &str) -> Option<u16> {
    message
        .strip_prefix("Browser navigation returned HTTP ")?
        .split_whitespace()
        .next()?
        .parse()
        .ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_sites_name_themselves_and_what_works() {
        let zhihu = Url::parse("https://www.zhihu.com/question/1/answer/2").unwrap();
        let hint = refusal_hint(&zhihu, false).unwrap();
        assert!(
            hint.starts_with("Zhihu refuses automated clients; open the page in your browser"),
            "{hint}"
        );
        assert!(hint.contains("Webpage, HTML Only"), "{hint}");
        assert!(hint.contains("fetch.playwright.cookies"), "{hint}");
        assert!(hint.contains("zhihu.com"), "{hint}");
        for (url, name) in [
            ("https://zhuanlan.zhihu.com/p/1", "Zhihu"),
            ("https://mp.weixin.qq.com/s/x", "WeChat"),
            ("https://www.douban.com/note/1/", "Douban"),
            ("https://stackoverflow.com/questions/1", "Stack Overflow"),
            ("https://medium.com/@a/b", "Medium"),
            ("https://a.hashnode.dev/p", "Hashnode"),
        ] {
            let hint = refusal_hint(&Url::parse(url).unwrap(), false).unwrap();
            assert!(hint.starts_with(name), "{url}: {hint}");
        }
        // Sites that are not known get no site-aware text, except for a
        // Cloudflare bot check, which says so.
        let other = Url::parse("https://example.test/a").unwrap();
        assert!(refusal_hint(&other, false).is_none());
        assert!(refusal_hint(&other, true).unwrap().contains("Cloudflare"));
        assert!(refusal_hint(&Url::parse("https://notzhihu.com/").unwrap(), false).is_none());
        assert!(
            refusal_hint(&Url::parse("https://zhihu.com.evil.test/").unwrap(), false).is_none()
        );
    }

    #[test]
    fn a_json_refusal_is_quoted_cleanly() {
        let body = r#"{"error":{"message":"您当前请求存在异常，暂时限制本次访问。\n如有疑问，请联系客服。","code":40362}}"#;
        let said = json_refusal(body.as_bytes()).unwrap();
        assert_eq!(
            said,
            "您当前请求存在异常，暂时限制本次访问。 如有疑问，请联系客服。 (code 40362)"
        );
        assert_eq!(
            json_refusal(br#"{"message":"Forbidden"}"#).unwrap(),
            "Forbidden"
        );
        assert_eq!(
            json_refusal(br#"{"msg":"slow down","code":429}"#).unwrap(),
            "slow down (code 429)"
        );
        let long = format!(r#"{{"message":"{}"}}"#, "x".repeat(300));
        let said = json_refusal(long.as_bytes()).unwrap();
        assert_eq!(said.chars().count(), 121, "{said}");
        assert!(said.ends_with('…'));
        for not_json in [
            "no",
            "<html>blocked</html>",
            "[1,2]",
            "{}",
            r#"{"ok":true}"#,
        ] {
            assert!(json_refusal(not_json.as_bytes()).is_none(), "{not_json}");
        }
    }

    #[test]
    fn verification_pages_served_as_success_are_recognized_by_site() {
        let wechat = Url::parse("https://mp.weixin.qq.com/mp/wappoc_appmsgcaptcha?x=1").unwrap();
        let message =
            verification_page(&wechat, "<h2>环境异常</h2><p>完成验证后即可继续访问。</p>").unwrap();
        assert!(
            message.starts_with("WeChat served a verification page instead of the content"),
            "{message}"
        );
        assert!(message.contains("Webpage, HTML Only"), "{message}");
        let douban =
            Url::parse("https://sec.douban.com/c?r=https%3A%2F%2Fwww.douban.com%2F").unwrap();
        assert!(
            verification_page(&douban, "载入中 ...")
                .unwrap()
                .starts_with("Douban")
        );
        let reddit = Url::parse("https://www.reddit.com/r/a/comments/1/x/").unwrap();
        assert!(
            verification_page(&reddit, "<h1>Prove your humanity</h1>")
                .unwrap()
                .starts_with("Reddit")
        );
        assert!(verification_page(&reddit, "<h1>A thread about humanity</h1>").is_none());
        let toutiao = Url::parse("https://www.toutiao.com/article/1/").unwrap();
        assert!(
            verification_page(
                &toutiao,
                "<html><body></body><script>var glb;glb._$jsvmprt=function(){}</script>"
            )
            .unwrap()
            .starts_with("Toutiao served a verification page")
        );
        assert!(
            verification_page(&toutiao, "<article><p>An ordinary article.</p></article>").is_none()
        );
        let zhihu = Url::parse("https://www.zhihu.com/question/1").unwrap();
        assert!(verification_page(&zhihu, r#"<meta id="zh-zse-ck" content="x">"#).is_some());
        assert!(verification_page(&zhihu, "<p>an ordinary page</p>").is_none());
        let unhuman =
            Url::parse("https://www.zhihu.com/account/unhuman?type=U4E3Z1&need_login=true")
                .unwrap();
        assert!(
            verification_page(&unhuman, "登录知乎")
                .unwrap()
                .starts_with("Zhihu served a verification page")
        );
        // An ordinary WeChat article that mentions the words is not a check.
        let article = Url::parse("https://mp.weixin.qq.com/s/abc").unwrap();
        assert!(verification_page(&article, "<p>环境异常的处理方法</p>").is_none());
        let other = Url::parse("https://example.test/").unwrap();
        assert!(verification_page(&other, "Prove your humanity").is_none());
    }

    #[test]
    fn browser_failures_give_their_status() {
        assert_eq!(
            browser_status("Browser navigation returned HTTP 403"),
            Some(403)
        );
        assert_eq!(
            browser_status("Browser navigation returned HTTP 429 extra"),
            Some(429)
        );
        assert_eq!(browser_status("Browser navigation failed"), None);
    }
}
