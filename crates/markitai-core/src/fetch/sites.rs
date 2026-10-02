//! Sites that turn automated clients away, and what to tell the reader.
//!
//! Some sites answer a client they take for a program with a refusal: a
//! status 403, a JSON error, a verification page served with status 200, a
//! login page, or a script challenge that only a person's browser passes.
//! Markitai does not forge a browser, solve a challenge or sign requests, so
//! such a site stays refused; this module only recognizes the refusal, names
//! the site, and says what does work: converting the page the reader saved
//! from their own browser, or giving their own browser session's cookies to
//! the local browser (`fetch.playwright.cookies`, see `docs/fetch.md`).
//!
//! One judgement ([`Shown::refusal`]) serves every reader: the markup the
//! static client, the local browser and Cloudflare Browser Rendering return,
//! and the Markdown and title that defuddle and Jina Reader return. Markup
//! gives more evidence (a challenge widget, a site's script marker); Markdown
//! gives the visible text and the title, which is what a refusal page says.

use serde_json::Value;
use std::sync::LazyLock;
use url::Url;

/// What to do instead, in every refusal.
const SAVE_PAGE: &str = "open the page in your browser and save it (File > Save Page As…, 'Webpage, HTML Only'), then convert the saved file";

pub(super) struct Site {
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
/// The marker of failures a login page produces, routed like a verification.
pub(super) const LOGIN_PAGE: &str = "served a login page";
/// The words that end a remote service's failure whose reading was the
/// site's refusal rather than the page (see [`Refusal::read_by`]).
const INSTEAD: &str = "instead of the content";

/// Whether a failure is a site's verification or login page, which `auto`
/// gives to the local browser.
pub(super) fn refusal_page(message: &str) -> bool {
    message.contains(VERIFICATION_PAGE) || message.contains(LOGIN_PAGE)
}

/// Whether a remote service's failure is the site turning it away: a refusal
/// page or a refusal status the service read in place of the page.
pub(super) fn refused_reading(message: &str) -> bool {
    message.contains(INSTEAD)
}

/// Whether a message already says what works instead.
pub(super) fn says_what_works(message: &str) -> bool {
    message.contains(SAVE_PAGE)
}

/// What works instead, for a page of `url` that turned a reader away: the
/// site's own advice when it is known, otherwise saving the page.
pub(super) fn what_works(url: &Url) -> String {
    match url.host_str().and_then(site_of) {
        Some(site) => advice(site),
        None => format!("the site turns automated readers away; {SAVE_PAGE}"),
    }
}

/// A page that is a site's verification or security check rather than the
/// content, served with a success status: the site, its message and what to
/// do. `url` is the address the page was finally read from.
#[cfg(test)]
pub(super) fn verification_page(url: &Url, html: &str) -> Option<String> {
    Shown::html(url, html).site_refusal()?.message()
}

/// What a refusal page is.
#[derive(Clone, Copy)]
pub(super) enum Refusal {
    /// A known site's verification or security check.
    Verification(&'static Site),
    /// A known site's request to log in, with little else on the page.
    Login(&'static Site),
    /// A bot challenge: Cloudflare, reCAPTCHA, hCaptcha, Geetest.
    Challenge,
    /// A notice that the page needs JavaScript.
    JavaScript,
}

impl Refusal {
    /// The failure a local reader reports for a site's page: the site, what
    /// it served and what works instead. `None` for the other refusals, whose
    /// wording the caller owns.
    pub(super) fn message(self) -> Option<String> {
        let (site, served) = match self {
            Self::Verification(site) => (site, VERIFICATION_PAGE),
            Self::Login(site) => (site, LOGIN_PAGE),
            Self::Challenge | Self::JavaScript => return None,
        };
        let mut message = format!(
            "{} {served} instead of the content: it {}; {SAVE_PAGE}",
            site.name, site.cause
        );
        if let Some(domain) = site.cookies {
            message.push_str(&format!(
                "; or give the local browser your own logged-in cookies for {domain} (fetch.playwright.cookies, see docs/fetch.md)"
            ));
        }
        Some(message)
    }

    /// The failure of a remote service that was shown this page. It names
    /// the service and the page; what works instead is said once by whoever
    /// reports the whole failure ([`what_works`]).
    pub(super) fn read_by(self, service: &str) -> String {
        let shown = match self {
            Self::Verification(site) => format!("{}'s verification page", site.name),
            Self::Login(site) => format!("{}'s login page", site.name),
            Self::Challenge => "a challenge page".into(),
            Self::JavaScript => "a page that asks for JavaScript".into(),
        };
        format!("The {service} service was shown {shown} {INSTEAD}")
    }
}

/// The failure of a remote service that read a refusal status from the site
/// in place of the page.
pub(super) fn refused_status(service: &str, status: u16) -> String {
    format!("The {service} service received HTTP {status} from the site {INSTEAD}")
}

/// Text of a page shorter than this, in words, is "little else" beside a
/// login request or a challenge title.
const FEW_WORDS: usize = 60;

/// Visible text a challenge page starts with.
const CHALLENGE_TEXT: [&str; 6] = [
    "checking your browser before accessing",
    "verify you are human",
    "verify that you are human",
    "verifying you are human",
    "智能验证检测中",
    "由极验提供技术支持",
];
/// Titles of challenge pages.
const CHALLENGE_TITLES: [&str; 4] = [
    "just a moment",
    "attention required",
    "security verification",
    "verify you are human",
];
/// Visible text a page that only asks for JavaScript starts with.
const JAVASCRIPT_TEXT: [&str; 5] = [
    "please enable javascript",
    "javascript is disabled",
    "javascript is not available",
    "you need to enable javascript",
    "enable javascript to continue",
];
/// Requests to log in that a known site shows in place of a page (compared
/// in lower case).
const LOGIN_REQUESTS: [&str; 6] = [
    "请您登录后查看",
    "登录后查看更多",
    "登录后才能查看",
    "log in to continue",
    "sign in to continue",
    "log in to view",
];

/// A page as a reader returned it, reduced to what tells a refusal from the
/// content.
pub(super) struct Shown<'a> {
    /// The address the page was finally read from.
    url: &'a Url,
    /// The markup, when the reader had it; a remote service's Markdown has
    /// none.
    markup: Option<&'a str>,
    /// The title, whitespace collapsed.
    title: String,
    /// The text a reader sees, whitespace collapsed: the markup's text
    /// outside `head`, `script`, `style`, `pre`, `code` and `template`, or
    /// the Markdown without images, link destinations and markers.
    text: String,
    /// The markup carries a challenge widget (Cloudflare, reCAPTCHA, hCaptcha).
    widget: bool,
    /// The markup has an `<article>` with text of its own.
    article: bool,
}

fn collapsed(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

impl<'a> Shown<'a> {
    /// Markup from the static client, the local browser or Cloudflare.
    pub(super) fn html(url: &'a Url, html: &'a str) -> Self {
        // Vendor names in an article, literal code examples or a plain-text
        // document are not challenge evidence: only visible, non-code text
        // and the markup's own attributes are read.
        let tree = scraper::Html::parse_document(html);
        let mut text = String::new();
        let mut title = String::new();
        let mut widget = false;
        let mut article = false;
        for node in tree.tree.nodes() {
            if let Some(element) = scraper::ElementRef::wrap(node) {
                let name = element.value().name();
                if name == "article" {
                    article |= element.text().any(|value| !value.trim().is_empty());
                }
                if name == "title" {
                    title = element.text().collect();
                }
                for attr in ["id", "class", "src", "action"] {
                    if let Some(value) = element.value().attr(attr) {
                        let value = value.to_ascii_lowercase();
                        widget |= [
                            "cf-browser-verification",
                            "cf-chl-",
                            "cf_chl_",
                            "/cdn-cgi/challenge-platform/",
                            "google.com/recaptcha",
                            "hcaptcha.com",
                            "g-recaptcha",
                            "h-captcha",
                        ]
                        .iter()
                        .any(|marker| value.contains(marker));
                    }
                }
            } else if let scraper::Node::Text(value) = node.value() {
                let ignored =
                    node.ancestors()
                        .filter_map(scraper::ElementRef::wrap)
                        .any(|element| {
                            matches!(
                                element.value().name(),
                                "head" | "script" | "style" | "pre" | "code" | "template"
                            )
                        });
                if !ignored {
                    text.push_str(value);
                    text.push(' ');
                }
            }
        }
        Self {
            url,
            markup: Some(html),
            title: collapsed(&title),
            text: collapsed(&text),
            widget,
            article,
        }
    }

    /// Markdown from a remote service, with the title the service gave.
    pub(super) fn markdown(url: &'a Url, title: Option<&str>, markdown: &str) -> Self {
        static IMAGE: LazyLock<regex::Regex> =
            LazyLock::new(|| regex::Regex::new(r"!\[[^\]]*\]\([^)]*\)").unwrap());
        static LINK: LazyLock<regex::Regex> =
            LazyLock::new(|| regex::Regex::new(r"\[([^\]]*)\]\([^)]*\)").unwrap());
        let text = IMAGE.replace_all(markdown, " ");
        let text = LINK.replace_all(&text, "$1");
        let text: Vec<&str> = text
            .lines()
            .map(|line| {
                line.trim()
                    .trim_start_matches(['#', '>', '-', '*', '+', ' '])
            })
            .collect();
        let text = text.join(" ").replace(['*', '`'], "").replace("__", "");
        Self {
            url,
            markup: None,
            title: collapsed(title.unwrap_or_default()),
            text: collapsed(&text),
            widget: false,
            article: false,
        }
    }

    fn words(&self) -> usize {
        crate::formats::word_count(&self.text)
    }

    /// The refusal this page is, if any: the site's own page first, then a
    /// challenge or a JavaScript notice.
    pub(super) fn refusal(&self) -> Option<Refusal> {
        self.site_refusal().or_else(|| self.notice())
    }

    /// A known site's verification page or login page.
    pub(super) fn site_refusal(&self) -> Option<Refusal> {
        let host = self.url.host_str()?.to_ascii_lowercase();
        let site = site_of(&host)?;
        let markup = self.markup.unwrap_or_default();
        let says = |words: &str| {
            markup.contains(words) || self.text.contains(words) || self.title.contains(words)
        };
        let path = self.url.path();
        let verification = match site.name {
            "WeChat" => {
                path.contains("wappoc_appmsgcaptcha")
                    || (host == "mp.weixin.qq.com" && says("环境异常") && says("完成验证"))
            }
            "Douban" => host == "sec.douban.com",
            "Weibo" => {
                (host == "passport.weibo.com" && path.starts_with("/visitor"))
                    || self.title == "Sina Visitor System"
            }
            "Reddit" => says("Prove your humanity"),
            // ByteDance's script-challenge page: an empty body and a bytecode VM.
            "Toutiao" => markup.contains("_$jsvmprt"),
            // The interstitial of a client its script cannot run in, and the
            // security check (`安全验证 - 知乎`) that asks for a login first.
            "Zhihu" => {
                path.starts_with("/account/unhuman")
                    || markup.contains("id=\"zh-zse-ck\"")
                    || zhihu_check(&self.title)
            }
            _ => false,
        };
        if verification {
            return Some(Refusal::Verification(site));
        }
        // A site that expects a login and shows only the request for one.
        let folded = format!("{} {}", self.title, self.text).to_lowercase();
        (site.cookies.is_some()
            && LOGIN_REQUESTS
                .iter()
                .any(|request| folded.contains(request))
            && self.words() <= FEW_WORDS)
            .then_some(Refusal::Login(site))
    }

    /// A challenge page or a page that only asks for JavaScript, on any site.
    pub(super) fn notice(&self) -> Option<Refusal> {
        let text = self.text.to_lowercase();
        let title = self.title.to_lowercase();
        let short = text.len() <= 2000;
        let instruction = CHALLENGE_TEXT.iter().any(|phrase| text.starts_with(phrase));
        let challenge_title = CHALLENGE_TITLES
            .iter()
            .any(|phrase| title.starts_with(phrase));
        // Markup shows a challenge's widget; Markdown cannot, so a challenge
        // title over next to no text stands in for it there.
        let titled = challenge_title && self.markup.is_none() && self.words() <= FEW_WORDS;
        if short
            && ((!self.article && instruction)
                || (self.widget && (challenge_title || text.is_empty()))
                || titled)
        {
            return Some(Refusal::Challenge);
        }
        (short
            && !self.article
            && JAVASCRIPT_TEXT
                .iter()
                .any(|phrase| text.starts_with(phrase)))
        .then_some(Refusal::JavaScript)
    }
}

/// Zhihu's security check title, `安全验证 - 知乎`, and not a question that
/// starts with the same words.
fn zhihu_check(title: &str) -> bool {
    let bare = title
        .strip_suffix("知乎")
        .map(|rest| rest.trim_end().trim_end_matches(['-', '–', '|']).trim_end())
        .unwrap_or(title);
    bare == "安全验证"
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

    fn refused(url: &str, title: Option<&str>, markdown: &str) -> Option<String> {
        let url = Url::parse(url).unwrap();
        Shown::markdown(&url, title, markdown)
            .refusal()
            .map(|refusal| refusal.read_by("jina"))
    }

    #[test]
    fn remote_markdown_is_judged_like_markup() {
        let question = "https://www.zhihu.com/question/19550225";
        let wall = "![Image 1](https://www.zhihu.com/question/19550225)\n\n![Image 2: ZhiHu logo](https://static.zhihu.com/logo.png)\n\n请您登录后查看更多专业优质内容。";
        assert_eq!(
            refused(question, Some("安全验证 - 知乎"), wall).unwrap(),
            "The jina service was shown Zhihu's verification page instead of the content"
        );
        assert_eq!(
            refused(question, None, wall).unwrap(),
            "The jina service was shown Zhihu's login page instead of the content"
        );
        // A question that starts with the same words, and a page with the
        // login request among its own content, are pages.
        let answers = "很多人都遇到过这个问题，下面是几个回答。".repeat(10);
        assert!(refused(question, Some("安全验证码收不到怎么办？ - 知乎"), &answers).is_none());
        assert!(
            refused(
                question,
                None,
                &format!("{answers}\n\n请您登录后查看更多专业优质内容。")
            )
            .is_none()
        );
        // The same words on a site that is not known are a page.
        assert!(
            refused(
                "https://example.com/a",
                None,
                "请您登录后查看更多专业优质内容。"
            )
            .is_none()
        );
        // Other sites' pages, read from their Markdown.
        assert!(
            refused(
                "https://www.reddit.com/r/a/comments/1/x/",
                Some("Reddit"),
                "# Prove your humanity\n\nComplete the challenge below."
            )
            .unwrap()
            .contains("Reddit's verification page")
        );
        assert!(
            refused("https://weibo.com/u/1", Some("Sina Visitor System"), "")
                .unwrap()
                .contains("Weibo's verification page")
        );
        assert!(
            refused(
                "https://www.reddit.com/r/a/",
                Some("A thread"),
                "Log in to continue."
            )
            .unwrap()
            .contains("Reddit's login page")
        );
        // Markup-only markers do not fire on Markdown that quotes them.
        assert!(
            refused(
                "https://www.toutiao.com/article/1/",
                Some("头条"),
                &format!("`_$jsvmprt` {}", "正文内容".repeat(40))
            )
            .is_none()
        );
        // A challenge on any site: its title over next to no text, or its
        // instruction; an article with such a title is a page.
        let challenge = "## example.com\n\nVerifying you are human. This may take a few seconds.\n\nPerformance & security by Cloudflare";
        assert_eq!(
            refused("https://example.com/a", Some("Just a moment..."), challenge).unwrap(),
            "The jina service was shown a challenge page instead of the content"
        );
        assert!(
            refused(
                "https://example.com/a",
                None,
                "Verify you are human by completing the action below."
            )
            .is_some()
        );
        let essay = format!(
            "# Just a moment\n\n{}",
            "A long essay about waiting for things to happen. ".repeat(10)
        );
        assert!(refused("https://example.com/a", Some("Just a moment"), &essay).is_none());
        assert_eq!(
            refused(
                "https://example.com/a",
                None,
                "Please enable JavaScript to view this page."
            )
            .unwrap(),
            "The jina service was shown a page that asks for JavaScript instead of the content"
        );
        assert!(
            refused(
                "https://example.com/a",
                Some("Example"),
                "An ordinary short page."
            )
            .is_none()
        );
    }

    #[test]
    fn markup_is_judged_with_its_title_and_visible_text() {
        let question = Url::parse("https://www.zhihu.com/question/1").unwrap();
        let check = "<html><head><title>安全验证 - 知乎</title></head><body><p>请您登录后查看更多专业优质内容。</p></body></html>";
        let message = verification_page(&question, check).unwrap();
        assert!(
            message.starts_with("Zhihu served a verification page instead of the content: it refuses automated clients;"),
            "{message}"
        );
        let wall = "<html><head><title>知乎</title></head><body><p>请您登录后查看更多专业优质内容。</p></body></html>";
        let refusal = Shown::html(&question, wall).refusal().unwrap();
        let message = refusal.message().unwrap();
        assert!(
            message.starts_with("Zhihu served a login page instead of the content"),
            "{message}"
        );
        assert!(refusal_page(&message) && says_what_works(&message));
        assert!(!refused_reading("HTTP 503 from the jina service"));
        assert!(refused_reading(&refused_status("jina", 403)));
        // A widget with a challenge title, and none without the widget.
        let example = Url::parse("https://example.test/").unwrap();
        let widget = r#"<html><head><title>Just a moment...</title></head><body><div class="cf-chl-widget"></div></body></html>"#;
        assert!(matches!(
            Shown::html(&example, widget).notice(),
            Some(Refusal::Challenge)
        ));
        let titled =
            "<html><head><title>Just a moment...</title></head><body><p>Short.</p></body></html>";
        assert!(Shown::html(&example, titled).notice().is_none());
        // What works: the site's own advice, or saving the page.
        assert!(
            what_works(&question).starts_with("Zhihu refuses automated clients; open the page")
        );
        assert!(
            what_works(&example)
                .starts_with("the site turns automated readers away; open the page")
        );
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
