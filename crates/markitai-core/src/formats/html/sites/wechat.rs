//! WeChat official-account articles (`mp.weixin.qq.com`).
//!
//! The article is `#js_content`, which the page serves with inline
//! `visibility: hidden` until its script shows it, so a page saved as "HTML
//! only" reads as empty to a reader that honours the style. Images keep their
//! address in `data-src`, code lines are one `code` element each, and the
//! title, account, author and time sit in the page's own header.

use super::{Article, Dialect, clean_children, first, iso_from_text, iso_from_unix, meta_content};
use super::{show_time, substantial, text_at};
use scraper::Html;
use std::sync::OnceLock;
use url::Url;

/// The first group the pattern captures in an inline script of the page.
fn script_capture(document: &Html, pattern: &regex::Regex) -> Option<String> {
    document
        .select(&super::super::selector("script"))
        .filter(|script| script.value().attr("src").is_none())
        .find_map(|script| {
            let text = script.text().collect::<String>();
            let found = pattern.captures(&text)?;
            let value = found.get(1)?.as_str().trim();
            (!value.is_empty()).then(|| value.to_owned())
        })
}

/// The page's own title and publication time, which its script assigns
/// (`var msg_title = '…'`, `ct = "1531324000"`); the header's markup is empty
/// until the script fills it.
fn script_title(document: &Html) -> Option<String> {
    static PATTERN: OnceLock<regex::Regex> = OnceLock::new();
    script_capture(
        document,
        PATTERN.get_or_init(|| {
            regex::Regex::new(r#"\bmsg_title\s*=\s*['"]([^'"]+)['"]"#).expect("static pattern")
        }),
    )
}

fn script_time(document: &Html) -> Option<i64> {
    static PATTERN: OnceLock<regex::Regex> = OnceLock::new();
    script_capture(
        document,
        PATTERN.get_or_init(|| {
            regex::Regex::new(r#"\bct\s*=\s*["']?(\d{9,10})"#).expect("static pattern")
        }),
    )
    .and_then(|seconds| seconds.parse().ok())
}

pub(super) fn read(document: &Html, address: Option<&Url>) -> Option<Article> {
    let root = document.root_element();
    let content = first(root, "#js_content")?;
    let body = clean_children(content, Dialect::WeChat);
    if !substantial(&body) {
        return None;
    }
    let title = text_at(root, "#activity-name")
        .or_else(|| meta_content(document, "og:title"))
        .or_else(|| script_title(document))?;
    let account = text_at(root, "#js_name");
    let author = meta_content(document, "author").or_else(|| {
        // The header's plain text spans: the author, and the time (an `em`).
        root.select(&super::super::selector(".rich_media_meta_text"))
            .filter(|element| element.value().attr("id") != Some("publish_time"))
            .filter(|element| element.value().name() != "em")
            .map(super::text_of)
            .find(|text| !text.is_empty() && text != "原创" && iso_from_text(text).is_none())
    });
    let published = text_at(root, "#publish_time")
        .and_then(|text| iso_from_text(&text))
        .or_else(|| script_time(document).and_then(iso_from_unix));
    let canonical = address
        .filter(|url| url.path().starts_with("/s"))
        .map(|url| {
            let mut url = url.clone();
            url.set_fragment(None);
            url.to_string()
        });
    let mut byline = vec![author.clone().unwrap_or_default()];
    byline.extend(account.clone());
    byline.push(published.as_deref().map(show_time).unwrap_or_default());
    let mut article = Article {
        reader: "wechat",
        title,
        author,
        published,
        site: Some("微信公众平台".into()),
        canonical,
        byline,
        body,
        ..Article::default()
    };
    if let Some(account) = account {
        article.metadata.insert("account".into(), account.into());
    }
    Some(article)
}

#[cfg(test)]
mod tests {
    use crate::formats::html::extract_html;

    const ARTICLE: &str = r#"<!DOCTYPE html><html><head><title></title>
        <meta property="og:title" content="Fallback title">
        <meta name="author" content="Alice&nbsp;Author">
        <meta property="og:url" content="http://mp.weixin.qq.com/s?__biz=abc&amp;mid=1#rd">
        <script>var ct = "1531324000"; var nickname = "Account Name";</script></head>
        <body><div id="img-content" class="rich_media_wrp">
        <h1 class="rich_media_title" id="activity-name"> The real title </h1>
        <div id="meta_content" class="rich_media_meta_list">
          <span class="rich_media_meta rich_media_meta_text" id="copyright_logo">原创</span>
          <span class="rich_media_meta rich_media_meta_text">Alice Author</span>
          <span class="rich_media_meta rich_media_meta_nickname"><a id="js_name">Account Name</a></span>
          <em id="publish_time" class="rich_media_meta rich_media_meta_text">2018年7月11日 23:46</em>
        </div>
        <div class="rich_media_content" id="js_content" style="visibility: hidden; opacity: 0;">
          <section><p>First paragraph of the article.</p></section>
          <p><img class="js_img_placeholder wx_img_placeholder" data-src="https://mmbiz.qpic.cn/mmbiz_png/abc/640?wx_fmt=png#imgIndex=0" data-w="640" alt="Figure" src="data:image/svg+xml,%3Csvg%2F%3E"></p>
          <section class="code-snippet__fix"><ul class="code-snippet__line-index"><li></li><li></li></ul>
          <pre data-lang="rust"><code><span leaf="">fn main() {</span></code><code><span leaf="">}</span></code></pre></section>
          <mpvoice class="js_editor_audio"></mpvoice>
        </div></div>
        <div id="js_pc_qr_code">Scan to follow</div><div id="js_tags">tags chrome</div></body></html>"#;

    #[test]
    fn a_hidden_article_is_read_with_its_header() {
        for base in [Some("https://mp.weixin.qq.com/s/abc"), None] {
            let page = if base.is_none() {
                format!("<!-- saved from url=(0030)https://mp.weixin.qq.com/s/abc -->{ARTICLE}")
            } else {
                ARTICLE.to_owned()
            };
            let document = extract_html(&page, base).unwrap();
            let markdown = &document.markdown;
            assert!(markdown.starts_with("# The real title"), "{markdown}");
            assert!(
                markdown.contains("Alice Author · Account Name · 2018-07-11 23:46"),
                "{markdown}"
            );
            assert!(
                markdown.contains("First paragraph of the article."),
                "{markdown}"
            );
            assert!(
                markdown.contains("![Figure](https://mmbiz.qpic.cn/mmbiz_png/abc/640?wx_fmt=png)"),
                "{markdown}"
            );
            assert!(
                markdown.contains("```rust\nfn main() {\n}\n```"),
                "{markdown}"
            );
            assert!(!markdown.contains("Scan to follow"), "{markdown}");
            assert!(!markdown.contains("tags chrome"), "{markdown}");
            assert_eq!(document.metadata["title"], "The real title");
            assert_eq!(document.metadata["author"], "Alice Author");
            assert_eq!(document.metadata["account"], "Account Name");
            assert_eq!(document.metadata["published"], "2018-07-11T23:46:00+08:00");
        }
    }

    #[test]
    fn the_publish_time_falls_back_to_the_script_variable() {
        let page = ARTICLE.replace("2018年7月11日 23:46", "");
        let document = extract_html(&page, Some("https://mp.weixin.qq.com/s/abc")).unwrap();
        assert_eq!(document.metadata["published"], "2018-07-11T23:46:40+08:00");
    }

    #[test]
    fn a_page_without_the_article_is_left_to_the_generic_reader() {
        let page = r#"<html><head><title>Verify</title></head><body><div class="weui-msg"><h2>环境异常</h2><p>请完成验证后继续访问。这是一段足够长的提示文字，用来保证通用阅读器能够读到内容。</p></div></body></html>"#;
        let document = extract_html(page, Some("https://mp.weixin.qq.com/s/abc")).unwrap();
        assert!(document.markdown.contains("环境异常"));
        assert!(!document.metadata.contains_key("account"));
    }

    #[test]
    fn other_hosts_with_the_same_ids_are_not_read_as_wechat() {
        // The generic reader honours the inline style: nothing of the article shows.
        if let Ok(document) = extract_html(ARTICLE, Some("https://example.test/s/abc")) {
            assert!(
                !document.markdown.contains("First paragraph"),
                "{}",
                document.markdown
            );
            assert!(!document.metadata.contains_key("account"));
        }
    }
}
