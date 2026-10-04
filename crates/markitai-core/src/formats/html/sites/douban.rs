//! Douban reviews (`book.douban.com/review/<id>`, and the movie and music
//! reviews that share their template).
//!
//! The review is `.review-content`, under an `h1` title and a header that
//! shows the reviewer's avatar and name, the work reviewed, the rating as
//! stars and the time, each on a line of its own; the work's card, the
//! reviewer's other reviews, the comments and the site's menus surround it.
//! The header becomes one line under the title (`reviewer · 评论《work》 ·
//! 力荐 · time`); a spoiler notice the site shows above the review stays.

use super::{Article, Dialect, clean_children, first, iso_from_text, show_time, substantial};
use super::{text_at, text_of};
use scraper::Html;

pub(super) fn read(document: &Html) -> Option<Article> {
    let root = document.root_element();
    let wrapper = first(root, ".review-wrapper")?;
    let content = first(wrapper, ".review-content")?;
    let mut body = clean_children(content, Dialect::Plain);
    if !substantial(&body) {
        return None;
    }
    let title = text_at(wrapper, "h1 [property='v:summary']").or_else(|| text_at(wrapper, "h1"))?;
    let header = first(wrapper, "header.main-hd");
    let author = header
        .and_then(|header| text_at(header, "a[href*='/people/']"))
        .or_else(|| {
            content
                .value()
                .attr("data-author")
                .map(str::trim)
                .filter(|name| !name.is_empty())
                .map(str::to_owned)
        });
    let work = header.and_then(|header| text_at(header, "a[href*='/subject/']"));
    let rating = header
        .and_then(|header| first(header, ".main-title-rating"))
        .and_then(|stars| stars.value().attr("title"))
        .map(str::trim)
        .filter(|rating| !rating.is_empty())
        .map(str::to_owned);
    let published = header
        .and_then(|header| first(header, ".main-meta"))
        .map(text_of)
        .and_then(|text| iso_from_text(&text));
    if let Some(notice) = text_at(wrapper, ".main-title-tip") {
        let mut marked = String::from("<p><em>");
        super::escaped(&notice, &mut marked);
        marked.push_str("</em></p>");
        body.insert_str(0, &marked);
    }
    let byline = vec![
        author.clone().unwrap_or_default(),
        work.map(|work| format!("评论《{work}》"))
            .unwrap_or_default(),
        rating.unwrap_or_default(),
        published.as_deref().map(show_time).unwrap_or_default(),
    ];
    Some(Article {
        reader: "douban",
        title,
        byline,
        author,
        published,
        site: Some("豆瓣".into()),
        body,
        ..Article::default()
    })
}

#[cfg(test)]
mod tests {
    use crate::formats::html::extract_html;

    const REVIEW: &str = r#"<html><head><title>认真你就赢了（卡徒）书评</title>
        <meta property="og:title" content="认真你就赢了"></head><body>
        <div id="db-global-nav"><a href="https://www.douban.com">豆瓣</a><a href="https://book.douban.com">读书</a></div>
        <div id="wrapper" class="book-content review-wrapper"><div id="content"><div class="grid-16-8 clearfix">
          <div class="article">
            <h1><span property="v:summary">认真你就赢了</span></h1>
            <div><div class="main" id="1">
              <a class="avatar author-avatar left" href="https://www.douban.com/people/someone/"><img src="https://img9.doubanio.com/icon/u1-44.jpg"></a>
              <header class="main-hd">
                <a href="https://www.douban.com/people/someone/"><span>本来老六</span></a>
                评论
                <a href="https://book.douban.com/subject/4880858/">卡徒</a>
                <span class="allstar50 main-title-rating" title="力荐"></span>
                <span class="main-title-hide">5</span>
                <div class="main-meta"><span content="2011-11-09">2011-11-09 15:53:00</span></div>
              </header>
              <div class="main-bd">
                <p class="main-title-tip">这篇书评可能有关键情节透露</p>
                <div id="link-report-1"><div class="review-content clearfix" data-author="本来老六">
                  据说人会自嘲是一种成熟的表现。<br><br>“点在哪里？”我有个好朋友。<br>第二行。
                </div></div>
                <div class="main-author">author card chrome</div>
              </div>
              <div class="main-ft">有用 24 没用 0</div>
            </div></div>
            <div id="comments">comment chrome</div>
          </div>
          <div class="aside"><div class="subject-card">作者: 方想 出版: 某出版社</div></div>
        </div></div></div></body></html>"#;

    #[test]
    fn a_review_is_read_with_its_header_as_one_line() {
        let document =
            extract_html(REVIEW, Some("https://book.douban.com/review/5161730/")).unwrap();
        let markdown = &document.markdown;
        assert!(markdown.starts_with("# 认真你就赢了\n"), "{markdown}");
        assert!(
            markdown.contains("本来老六 · 评论《卡徒》 · 力荐 · 2011-11-09 15:53"),
            "{markdown}"
        );
        assert!(
            markdown.contains("*这篇书评可能有关键情节透露*"),
            "{markdown}"
        );
        assert!(
            markdown.contains("据说人会自嘲是一种成熟的表现。\n\n“点在哪里？”"),
            "{markdown}"
        );
        for chrome in [
            "u1-44.jpg",
            "author card chrome",
            "有用",
            "comment chrome",
            "出版",
            "读书",
        ] {
            assert!(!markdown.contains(chrome), "{chrome}: {markdown}");
        }
        assert_eq!(document.metadata["title"], "认真你就赢了");
        assert_eq!(document.metadata["author"], "本来老六");
        assert_eq!(document.metadata["published"], "2011-11-09T15:53:00+08:00");
        assert_eq!(document.metadata["site"], "豆瓣");
    }

    #[test]
    fn a_page_without_a_review_is_left_to_the_generic_reader() {
        let page = r#"<html><head><title>豆瓣读书</title></head><body><div class="article"><h1>新书速递</h1><p>本周新书。</p></div></body></html>"#;
        let document = extract_html(page, Some("https://book.douban.com/")).unwrap();
        assert!(
            document.markdown.contains("本周新书。"),
            "{}",
            document.markdown
        );
    }
}
