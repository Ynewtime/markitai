//! Chinese blog platforms whose article sits among site chrome or whose
//! markup hides parts of it from the generic reader: cnblogs (博客园),
//! Jianshu (简书), OSCHINA (开源中国) and 36Kr (36氪).

use super::{Article, Dialect, Reading, clean_children, first, iso_from_text, iso_from_unix};
use super::{show_time, substantial, text_at};
use scraper::Html;
use serde_json::{Map, Value};

/// cnblogs: the post is `#cnblogs_post_body`, between a header with the blog's
/// own title and navigation and a footer of counters; the title, author and
/// time are `#cb_post_title_url`, the first link of `.postDesc` and
/// `#post-date`, whatever skin the blog uses.
pub(super) fn cnblogs(document: &Html) -> Option<Article> {
    let root = document.root_element();
    let content = first(root, "#cnblogs_post_body")?;
    let body = clean_children(content, Dialect::Cnblogs);
    if !substantial(&body) {
        return None;
    }
    let title = text_at(root, "#cb_post_title_url")
        .or_else(|| text_at(root, ".postTitle a"))
        .or_else(|| text_at(root, "h1"))?;
    let author = first(root, ".postDesc").and_then(|description| {
        description
            .select(&super::super::selector("a[href]"))
            .filter(|link| {
                link.value()
                    .attr("href")
                    .is_some_and(|href| href.starts_with("http"))
            })
            .map(super::text_of)
            .find(|text| !text.is_empty())
    });
    let published = text_at(root, "#post-date").and_then(|text| iso_from_text(&text));
    let canonical = first(root, "#cb_post_title_url")
        .and_then(|link| link.value().attr("href"))
        .filter(|href| href.starts_with("http"))
        .map(str::to_owned);
    let day = published.as_deref().map(show_time).unwrap_or_default();
    Some(Article {
        reader: "cnblogs",
        title,
        byline: vec![author.clone().unwrap_or_default(), day],
        author,
        published,
        site: Some("博客园".into()),
        canonical,
        body,
        ..Article::default()
    })
}

/// OSCHINA: the post is `.blog-content .editor`; around it the page has an AI
/// summary box, an advertisement, tags, comments and a list of recommended
/// posts. The author and date are in `.blog-content-info`.
pub(super) fn oschina(document: &Html) -> Option<Article> {
    let root = document.root_element();
    let content = first(root, ".blog-content .editor")?;
    let body = clean_children(content, Dialect::Plain);
    if !substantial(&body) {
        return None;
    }
    let title = text_at(root, "h1.blog-content-title")?;
    let info = first(root, ".blog-content-info");
    // The info items are the author, the date and the counters, in that order,
    // after a tag that says the post is original; a category link follows.
    let author = info.and_then(|info| text_at(info, ".blog-content-info-item"));
    let published = info
        .map(super::text_of)
        .and_then(|text| iso_from_text(&text));
    Some(Article {
        reader: "oschina",
        title,
        byline: vec![
            author.clone().unwrap_or_default(),
            published.as_deref().map(show_time).unwrap_or_default(),
        ],
        author,
        published,
        site: Some("开源中国".into()),
        body,
        ..Article::default()
    })
}

fn next_data(document: &Html) -> Option<Value> {
    let script = first(document.root_element(), "script#__NEXT_DATA__")?;
    let text: String = script.text().collect();
    serde_json::from_str(text.trim()).ok()
}

/// Jianshu: the post is the page's `article`. Its images keep their address
/// in `data-original-src` and carry the editor's default caption `image`;
/// the author and first publication time are in the page's `__NEXT_DATA__`.
pub(super) fn jianshu(document: &Html) -> Option<Article> {
    let root = document.root_element();
    let content = first(root, "article")?;
    let body = clean_children(content, Dialect::Jianshu);
    if !substantial(&body) {
        return None;
    }
    let note = next_data(document);
    let data = note
        .as_ref()
        .and_then(|data| data.pointer("/props/initialState/note/data"));
    let text = |key: &str| {
        data.and_then(|data| data.get(key)?.as_str())
            .map(|text| text.split_whitespace().collect::<Vec<_>>().join(" "))
            .filter(|text| !text.is_empty())
    };
    let title = text_at(root, "h1").or_else(|| text("public_title"))?;
    let author = data
        .and_then(|data| data.pointer("/user/nickname")?.as_str())
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .map(str::to_owned);
    let published = data
        .and_then(|data| data.get("first_shared_at")?.as_i64())
        .and_then(iso_from_unix);
    let mut byline = vec![author.clone().unwrap_or_default()];
    if let Some(published) = &published {
        byline.push(show_time(published));
    }
    let mut article = Article {
        reader: "jianshu",
        title,
        byline,
        author,
        published,
        site: Some("简书".into()),
        body,
        ..Article::default()
    };
    if let Some(views) = data.and_then(|data| data.get("views_count")?.as_i64()) {
        article.metadata.insert("views".into(), views.into());
    }
    Some(article)
}

/// 36Kr: the page's `article:published_time` is the moment the server wrote
/// the page, not the article's. The article's own time is the
/// `publishTime` of the state its script embeds, next to its content.
pub(super) fn kr36(document: &Html) -> Option<Reading> {
    static PUBLISHED: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    let pattern = PUBLISHED.get_or_init(|| {
        regex::Regex::new(r#""publishTime"\s*:\s*(\d{10,13})\s*,\s*"widgetContent""#)
            .expect("static pattern")
    });
    let published = document
        .select(&super::super::selector("script"))
        .filter(|script| script.value().attr("src").is_none())
        .find_map(|script| {
            let text = script.text().collect::<String>();
            let found = pattern.captures(&text)?;
            iso_from_unix(found[1].parse().ok()?)
        })?;
    let mut metadata = Map::new();
    metadata.insert("published".into(), published.into());
    Some(Reading {
        page: None,
        metadata,
    })
}

#[cfg(test)]
mod tests {
    use crate::formats::html::extract_html;

    const CNBLOGS: &str = r#"<!DOCTYPE html><html><head><title>A diary - Author - cnblogs</title>
        <script type="application/ld+json">{"@type":"Article","datePublished":"2020-03-27T17:38:00.0000000&#x2B;08:00"}</script></head>
        <body><div id="top_nav"><ul><li><a href="https://www.cnblogs.com/">cnblogs home</a></li></ul></div>
        <div id="header"><h1><a id="Header1_HeaderTitle" href="https://www.cnblogs.com/wkfvawl">Blog title</a></h1><p>Blog motto text</p></div>
        <div class="post"><h1 class="postTitle"><a id="cb_post_title_url" href="https://www.cnblogs.com/wkfvawl/p/1.html" title="posted"><span role="heading">The diary post</span></a></h1>
        <div class="postBody"><div id="cnblogs_post_body" class="blogpost-body">
        <h2>Background</h2><p>The post body text.</p><p><img src="https://img.cnblogs.com/a.png" alt=""></p>
        <div class="cnblogs_code"><img id="code_img_closed_1" class="code_img_closed" src="https://img.cnblogs.com/closed.gif"><pre><span style="color:#0000ff">int</span> x;<br>return x;</pre></div></div>
        <div id="blog_post_info_block"><div id="blog_post_info"></div></div></div>
        <div class="postDesc">posted @ <span id="post-date">2020-03-27 17:38</span>&nbsp;<a href="https://www.cnblogs.com/wkfvawl">Wang Lu</a>&nbsp;reads(<span>9</span>)</div></div>
        <div id="comment_form">Comment chrome</div></body></html>"#;

    #[test]
    fn a_cnblogs_post_is_read_without_the_blog_chrome() {
        let document =
            extract_html(CNBLOGS, Some("https://www.cnblogs.com/wkfvawl/p/1.html")).unwrap();
        let markdown = &document.markdown;
        assert!(markdown.starts_with("# The diary post"), "{markdown}");
        assert!(
            markdown.contains("Wang Lu · 2020-03-27 17:38"),
            "{markdown}"
        );
        assert!(
            markdown.contains("## Background") && markdown.contains("The post body text."),
            "{markdown}"
        );
        assert!(
            markdown.contains("![](https://img.cnblogs.com/a.png)"),
            "{markdown}"
        );
        assert!(markdown.contains("int x;\nreturn x;"), "{markdown}");
        for chrome in [
            "cnblogs home",
            "Blog title",
            "Blog motto",
            "Comment chrome",
            "closed.gif",
        ] {
            assert!(!markdown.contains(chrome), "{chrome}: {markdown}");
        }
        assert_eq!(document.metadata["title"], "The diary post");
        assert_eq!(document.metadata["author"], "Wang Lu");
        assert_eq!(document.metadata["published"], "2020-03-27T17:38:00+08:00");
        // A saved page is recognized by its canonical link.
        let saved = CNBLOGS.replace(
            "<title>",
            "<link rel=\"canonical\" href=\"https://www.cnblogs.com/wkfvawl/p/1.html\"><title>",
        );
        let document = extract_html(&saved, None).unwrap();
        assert!(
            document.markdown.starts_with("# The diary post"),
            "{}",
            document.markdown
        );
        // Another host's page with the same ids is not rewritten.
        let other = extract_html(CNBLOGS, Some("https://example.test/p/1.html")).unwrap();
        assert!(
            other.markdown.contains("cnblogs home"),
            "{}",
            other.markdown
        );
    }

    const JIANSHU: &str = r#"<!DOCTYPE html><html><head><title>Note - Jianshu</title>
        <meta property="og:url" content="https://www.jianshu.com/p/abc"></head>
        <body><div class="header">Chrome header</div><h1 title="Note">A note</h1>
        <article class="_2rhmJa"><p>See <a href="https://links.jianshu.com/go?to=https%3A%2F%2Fexample.test%2Fref">the reference</a>.</p>
        <div class="image-package"><div class="image-container"><div class="image-container-fill" style="padding-bottom: 54%;"></div>
        <div class="image-view" data-width="10"><img data-original-src="//upload-images.jianshu.io/upload_images/1-a.png" data-original-width="10"></div></div>
        <div class="image-caption">image</div></div>
        <div class="image-package"><div class="image-container"><div class="image-view"><img data-original-src="//upload-images.jianshu.io/upload_images/1-b.png"></div></div>
        <div class="image-caption">A real caption</div></div>
        <p>After the pictures.</p></article><div class="footer">Related chrome</div>
        <script id="__NEXT_DATA__" type="application/json">{"props":{"initialState":{"note":{"data":{"user":{"nickname":"Nemo"},"first_shared_at":1578136578,"views_count":4084}}}}}</script></body></html>"#;

    #[test]
    fn a_jianshu_note_keeps_its_images_and_drops_the_default_caption() {
        let document = extract_html(JIANSHU, Some("https://www.jianshu.com/p/abc")).unwrap();
        let markdown = &document.markdown;
        assert!(markdown.starts_with("# A note"), "{markdown}");
        assert!(markdown.contains("Nemo · 2020-01-04 19:16"), "{markdown}");
        assert!(
            markdown.contains("[the reference](https://example.test/ref)"),
            "{markdown}"
        );
        assert!(
            markdown.contains("(https://upload-images.jianshu.io/upload_images/1-a.png)"),
            "{markdown}"
        );
        assert!(
            markdown.contains(
                "![A real caption](https://upload-images.jianshu.io/upload_images/1-b.png)"
            ),
            "{markdown}"
        );
        assert!(!markdown.contains("\nimage\n"), "{markdown}");
        assert!(
            !markdown.contains("Chrome header") && !markdown.contains("Related chrome"),
            "{markdown}"
        );
        assert_eq!(document.metadata["author"], "Nemo");
        assert_eq!(document.metadata["published"], "2020-01-04T19:16:18+08:00");
        assert_eq!(document.metadata["views"], 4084);
    }

    #[test]
    fn an_oschina_post_is_read_without_its_summary_box_ads_and_recommendations() {
        let page = r#"<html><head><title>Post - OSCHINA</title></head><body>
            <section><main><div class="blog-content"><h1 class="blog-content-title">The post</h1>
            <div><div class="blog-content-info"><div class="blog-content-info-box"><span class="original">原创</span>
              <span class="blog-content-info-item">Huang <span></span></span><span class="blog-content-info-item">2013-11-29</span>
              <span class="blog-content-info-item">23,452</span></div><a href="/category/1">Category</a></div>
            <div class="blog-content-AiDesc">AI summary chrome</div></div>
            <div><div class="ad-container">ad chrome</div></div>
            <div class="editor heti"><h2>Install</h2><p>The body of the post.</p><pre><code>mvn -version</code></pre></div>
            <div class="blog-content-footer">tag chrome</div></div>
            <div id="comment" class="blog-content">comment chrome</div>
            <div class="blog-content">recommended chrome</div></main></section></body></html>"#;
        let document =
            extract_html(page, Some("https://my.oschina.net/huangyong/blog/180189")).unwrap();
        let markdown = &document.markdown;
        assert!(markdown.starts_with("# The post"), "{markdown}");
        assert!(markdown.contains("Huang · 2013-11-29"), "{markdown}");
        assert!(
            markdown.contains("## Install") && markdown.contains("mvn -version"),
            "{markdown}"
        );
        for chrome in [
            "AI summary",
            "ad chrome",
            "tag chrome",
            "comment chrome",
            "recommended chrome",
        ] {
            assert!(!markdown.contains(chrome), "{chrome}: {markdown}");
        }
        assert_eq!(document.metadata["author"], "Huang");
        // A page that is the site's 404 has no post: the generic reader reads it.
        let missing = r#"<html><body><main><h1>找不到您访问的页面</h1><p>抱歉，您访问的页面不存在，可能暂无权限或已删除。这里有足够多的文字内容。</p></main></body></html>"#;
        let document = extract_html(missing, Some("https://my.oschina.net/x/blog/1")).unwrap();
        assert!(document.markdown.contains("找不到您访问的页面"));
        assert!(!document.metadata.contains_key("published"));
    }

    #[test]
    fn a_36kr_page_gets_its_own_publication_time() {
        let page = r#"<html><head><title>Story-36Kr</title>
            <meta property="article:published_time" content="2026-10-02T15:17:51+08:00"></head>
            <body><article><h1>Story</h1><p>The story text is long enough to be an article on its own.</p></article>
            <script>window.initialState={"articleDetail":{"articleDetailData":{"data":{"widgetTitle":"Story","publishTime":1788516721882,"widgetContent":"<p>x</p>"}}}}</script></body></html>"#;
        let document = extract_html(page, Some("https://www.36kr.com/p/1")).unwrap();
        assert_eq!(document.metadata["published"], "2026-09-04T18:12:01+08:00");
        assert!(
            document.markdown.starts_with("# Story"),
            "{}",
            document.markdown
        );
        // Without that state the page's own metadata stays.
        let without = page.replace("publishTime", "other");
        let document = extract_html(&without, Some("https://www.36kr.com/p/1")).unwrap();
        assert_eq!(document.metadata["published"], "2026-10-02T15:17:51+08:00");
    }
}
