//! Bilibili opus pages and columns (`bilibili.com/opus/<id>`, and the older
//! `read/cv<id>` addresses that lead to them).
//!
//! The page is rendered by scripts, so this reads the page a browser rendered
//! (the local browser's result, or a page saved from a browser). The article is
//! `.opus-module-content`, with its title, author and time in the modules above
//! it; the table of contents, the sidebar, the comments and the page's menu
//! are not part of it. An image is a `<picture>` whose address carries the
//! image service's size directive after an `@` (`….png@516w_494h.webp`); the
//! address without it is the image as uploaded.

use super::{Article, Dialect, clean_children, first, iso_from_text, show_time, substantial};
use super::{text_at, text_of};
use scraper::Html;

pub(super) fn read(document: &Html) -> Option<Article> {
    let root = document.root_element();
    let content = first(root, ".opus-module-content")?;
    let body = clean_children(content, Dialect::Bilibili);
    if !substantial(&body) {
        return None;
    }
    let title = text_at(root, ".opus-module-title__text")
        .or_else(|| text_at(root, ".opus-module-title"))?;
    let author = text_at(root, ".opus-module-author__name");
    let published = first(root, ".opus-module-author__pub__text")
        .map(text_of)
        .and_then(|text| iso_from_text(&text));
    let mut byline = vec![author.clone().unwrap_or_default()];
    byline.push(published.as_deref().map(show_time).unwrap_or_default());
    Some(Article {
        reader: "bilibili",
        title,
        byline,
        author,
        published,
        site: Some("哔哩哔哩".into()),
        body,
        ..Article::default()
    })
}

#[cfg(test)]
mod tests {
    use crate::formats::html::extract_html;

    const OPUS: &str = r#"<html><head><title>Riddle - bilibili</title></head><body>
        <div class="bili-header"><a href="/">首页</a><a href="/anime/">番剧</a></div>
        <div class="opus-detail"><div class="opus-toc"><div class="opus-toc__list">TOC chrome</div></div>
        <div class="bili-opus-view">
          <div class="opus-module-top"><div class="opus-module-top__album">album chrome</div></div>
          <div class="opus-module-title"><div class="opus-module-title__inner"><span class="opus-module-title__text">The Opus</span></div></div>
          <div class="opus-module-author"><div class="opus-module-author__name">Writer</div>
            <div class="opus-module-author__pub"><div class="opus-module-author__pub__text">2018年01月29日 09:38</div></div></div>
          <div class="opus-module-content opus-paragraph-children">
            <p><span>First paragraph of the opus.</span></p>
            <div class="opus-para-pic center"><div class="opus-pic-view"><div class="bili-dyn-pic"><div class="bili-dyn-pic__img"><div class="b-img"><picture class="b-img__inner">
              <source type="image/avif" srcset="//i0.hdslb.com/bfs/article/abc.png@516w_494h.avif">
              <img src="//i0.hdslb.com/bfs/article/abc.png@516w_494h.webp" loading="lazy"></picture></div></div></div></div></div>
            <h1><strong>CAST</strong></h1><ul><li><p>Actor one</p></li></ul>
          </div>
          <div class="opus-module-extend">tag chrome</div><div class="opus-module-bottom">share chrome</div>
        </div></div>
        <div class="bili-tabs opus-tabs">Comment chrome</div></body></html>"#;

    #[test]
    fn an_opus_is_read_without_the_page_around_it() {
        let document = extract_html(
            OPUS,
            Some("https://www.bilibili.com/opus/78819216888318323"),
        )
        .unwrap();
        let markdown = &document.markdown;
        assert!(markdown.starts_with("# The Opus"), "{markdown}");
        assert!(markdown.contains("Writer · 2018-01-29 09:38"), "{markdown}");
        assert!(
            markdown.contains("First paragraph of the opus."),
            "{markdown}"
        );
        assert!(
            markdown.contains("(https://i0.hdslb.com/bfs/article/abc.png)"),
            "{markdown}"
        );
        assert!(
            markdown.contains("# **CAST**") && markdown.contains("* Actor one"),
            "{markdown}"
        );
        for chrome in [
            "TOC chrome",
            "album chrome",
            "tag chrome",
            "share chrome",
            "Comment chrome",
            "番剧",
        ] {
            assert!(!markdown.contains(chrome), "{chrome}: {markdown}");
        }
        assert_eq!(document.metadata["author"], "Writer");
        assert_eq!(document.metadata["published"], "2018-01-29T09:38:00+08:00");
        // Another host's page with the same classes is read as any page.
        let other = extract_html(OPUS, Some("https://example.test/opus/1")).unwrap();
        assert!(
            other.markdown.contains("Comment chrome") || other.markdown.contains("TOC chrome"),
            "{}",
            other.markdown
        );
    }

    #[test]
    fn a_page_that_scripts_have_not_rendered_is_left_to_the_generic_reader() {
        let shell = r#"<html><head><title>哔哩哔哩专栏</title></head><body><div id="app"><p>首页 番剧 直播 游戏中心 会员购 漫画 赛事 下载客户端 登录 大会员 消息 动态 收藏 历史 创作中心 投稿</p></div></body></html>"#;
        let document = extract_html(shell, Some("https://www.bilibili.com/read/cv1/")).unwrap();
        assert!(document.markdown.contains("首页"), "{}", document.markdown);
        assert!(!document.metadata.contains_key("published"));
    }
}
