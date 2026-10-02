//! Zhihu answers, column articles and questions (`zhihu.com`,
//! `zhuanlan.zhihu.com`).
//!
//! The page embeds the data it renders from as JSON in
//! `<script id="js-initialData">`: the answer or article as an HTML string,
//! with its author, votes and dates. That is read first, because it is the
//! same text the page shows, free of the page's lazy-loading placeholders and
//! of the other answers it lists. A page without the script (or without the
//! entity its address names) is read from its rendered markup
//! (`.RichContent-inner`, `.Post-RichText`).
//!
//! Zhihu answers a client it takes for automated with a refusal; this reader
//! only reads what a page saved from a browser, or served to an ordinary
//! one, contains.

use super::{Article, Dialect, clean_children, clean_fragment, first, iso_from_unix, substantial};
use super::{text_at, text_of};
use scraper::{ElementRef, Html};
use serde_json::Value;
use url::Url;

/// What an address names.
#[derive(Debug, PartialEq, Eq)]
enum Target {
    Answer {
        question: Option<String>,
        answer: String,
    },
    Article(String),
    Question(String),
    Unknown,
}

fn target(address: Option<&Url>) -> Target {
    let Some(segments) = address.and_then(Url::path_segments) else {
        return Target::Unknown;
    };
    let segments: Vec<&str> = segments.filter(|segment| !segment.is_empty()).collect();
    let digits = |value: &str| value.bytes().all(|byte| byte.is_ascii_digit());
    match segments.as_slice() {
        ["question", question, "answer", answer, ..] if digits(question) && digits(answer) => {
            Target::Answer {
                question: Some((*question).to_owned()),
                answer: (*answer).to_owned(),
            }
        }
        ["answer", answer, ..] if digits(answer) => Target::Answer {
            question: None,
            answer: (*answer).to_owned(),
        },
        ["question", question] if digits(question) => Target::Question((*question).to_owned()),
        ["p", id, ..] if digits(id) => Target::Article((*id).to_owned()),
        _ => Target::Unknown,
    }
}

fn initial_data(document: &Html) -> Option<Value> {
    let script = first(document.root_element(), "script#js-initialData")?;
    let text: String = script.text().collect();
    serde_json::from_str(text.trim()).ok()
}

/// A JSON string or number as text.
fn text(value: &Value, key: &str) -> Option<String> {
    match value.get(key)? {
        Value::String(text) => {
            let text = text.split_whitespace().collect::<Vec<_>>().join(" ");
            (!text.is_empty()).then_some(text)
        }
        Value::Number(number) => Some(number.to_string()),
        _ => None,
    }
}

fn number(value: &Value, keys: &[&str]) -> Option<i64> {
    keys.iter().find_map(|key| value.get(*key)?.as_i64())
}

fn date(value: &Value, keys: &[&str]) -> Option<String> {
    number(value, keys).and_then(iso_from_unix)
}

/// The day of an ISO time.
fn day(iso: &str) -> &str {
    iso.split('T').next().unwrap_or(iso)
}

fn entity<'a>(entities: &'a Value, kind: &str, id: &str) -> Option<&'a Value> {
    entities.get(kind)?.get(id)
}

fn author_line(entity: &Value) -> (Option<String>, Option<String>) {
    let author = entity.get("author");
    let name = author
        .and_then(|author| text(author, "name"))
        .filter(|name| name != "匿名用户" || author.is_some());
    let headline = author.and_then(|author| text(author, "headline"));
    (name, headline)
}

fn answer_page(
    entities: &Value,
    answer: &Value,
    question_id: Option<&str>,
    answer_id: &str,
) -> Option<Article> {
    let content = answer.get("content")?.as_str()?;
    let body = clean_fragment(content, Dialect::Zhihu);
    if !substantial(&body) {
        return None;
    }
    let question_id = question_id
        .map(str::to_owned)
        .or_else(|| text(answer.get("question")?, "id"));
    let title = answer
        .get("question")
        .and_then(|question| text(question, "title"))
        .or_else(|| {
            let question = entity(entities, "questions", question_id.as_deref()?)?;
            text(question, "title")
        })?;
    let (name, headline) = author_line(answer);
    let published = date(answer, &["createdTime", "created_time", "created"]);
    let mut byline = vec![
        name.clone().unwrap_or_default(),
        headline.unwrap_or_default(),
    ];
    let votes = number(answer, &["voteupCount", "voteup_count"]);
    if let Some(votes) = votes {
        byline.push(format!("赞同 {votes}"));
    }
    if let Some(published) = &published {
        byline.push(format!("发布于 {}", day(published)));
    }
    let mut page = Article {
        reader: "zhihu",
        title,
        author: name,
        published,
        site: Some("知乎".into()),
        canonical: question_id.map(|question| {
            format!("https://www.zhihu.com/question/{question}/answer/{answer_id}")
        }),
        byline,
        body,
        ..Article::default()
    };
    if let Some(votes) = votes {
        page.metadata.insert("upvotes".into(), votes.into());
    }
    Some(page)
}

fn column_page(article: &Value, id: &str) -> Option<Article> {
    let content = article.get("content")?.as_str()?;
    let mut body = String::new();
    if let Some(cover) = text(article, "titleImage").or_else(|| text(article, "imageUrl")) {
        body.push_str(&clean_fragment(
            &format!(
                "<p><img src=\"{}\" alt=\"\"></p>",
                cover.replace('"', "%22")
            ),
            Dialect::Zhihu,
        ));
    }
    body.push_str(&clean_fragment(content, Dialect::Zhihu));
    if !substantial(&body) {
        return None;
    }
    let title = text(article, "title")?;
    let (name, headline) = author_line(article);
    let published = date(article, &["created", "createdTime", "created_time"]);
    let mut byline = vec![
        name.clone().unwrap_or_default(),
        headline.unwrap_or_default(),
    ];
    let votes = number(article, &["voteupCount", "voteup_count"]);
    if let Some(votes) = votes {
        byline.push(format!("赞同 {votes}"));
    }
    if let Some(published) = &published {
        byline.push(format!("发布于 {}", day(published)));
    }
    let mut page = Article {
        reader: "zhihu",
        title,
        author: name,
        published,
        site: Some("知乎".into()),
        canonical: Some(format!("https://zhuanlan.zhihu.com/p/{id}")),
        byline,
        body,
        ..Article::default()
    };
    if let Some(votes) = votes {
        page.metadata.insert("upvotes".into(), votes.into());
    }
    Some(page)
}

/// A question with the answers the page carries, the most upvoted first.
fn question_page(entities: &Value, question_id: &str) -> Option<Article> {
    let question = entity(entities, "questions", question_id);
    let mut answers: Vec<&Value> = entities
        .get("answers")?
        .as_object()?
        .values()
        .filter(|answer| {
            answer
                .get("question")
                .and_then(|question| text(question, "id"))
                .is_some_and(|id| id == question_id)
        })
        .collect();
    answers.sort_by_key(|answer| {
        std::cmp::Reverse(number(answer, &["voteupCount", "voteup_count"]).unwrap_or(0))
    });
    let title = question
        .and_then(|question| text(question, "title"))
        .or_else(|| {
            answers
                .iter()
                .find_map(|answer| text(answer.get("question")?, "title"))
        })?;
    let mut body = String::new();
    if let Some(detail) = question
        .and_then(|question| question.get("detail"))
        .and_then(Value::as_str)
    {
        body.push_str(&clean_fragment(detail, Dialect::Zhihu));
    }
    let mut written = 0;
    for answer in answers.into_iter().take(20) {
        let Some(content) = answer.get("content").and_then(Value::as_str) else {
            continue;
        };
        let markup = clean_fragment(content, Dialect::Zhihu);
        if !substantial(&markup) {
            continue;
        }
        let (name, headline) = author_line(answer);
        let mut heading = name.unwrap_or_else(|| "匿名用户".into());
        if let Some(headline) = headline {
            heading.push_str(" · ");
            heading.push_str(&headline);
        }
        if let Some(votes) = number(answer, &["voteupCount", "voteup_count"]) {
            heading.push_str(&format!(" · 赞同 {votes}"));
        }
        body.push_str("<h2>");
        super::super::escaped(&heading, &mut body);
        body.push_str("</h2>");
        body.push_str(&markup);
        written += 1;
    }
    if written == 0 && !substantial(&body) {
        return None;
    }
    Some(Article {
        reader: "zhihu",
        title,
        site: Some("知乎".into()),
        canonical: Some(format!("https://www.zhihu.com/question/{question_id}")),
        body,
        ..Article::default()
    })
}

fn from_data(data: &Value, wanted: &Target) -> Option<Article> {
    let entities = data.pointer("/initialState/entities")?;
    let answers = entities.get("answers").and_then(Value::as_object);
    let articles = entities.get("articles").and_then(Value::as_object);
    match wanted {
        Target::Answer { question, answer } => {
            let found = answers?.get(answer)?;
            answer_page(entities, found, question.as_deref(), answer)
        }
        Target::Article(id) => column_page(articles?.get(id)?, id),
        Target::Question(id) => question_page(entities, id),
        Target::Unknown => {
            // A saved page that names no address: the one answer or article it has.
            match (
                answers.filter(|map| !map.is_empty()),
                articles.filter(|map| !map.is_empty()),
            ) {
                (Some(answers), None) if answers.len() == 1 => {
                    let (id, answer) = answers.iter().next()?;
                    answer_page(entities, answer, None, id)
                }
                (None, Some(articles)) if articles.len() == 1 => {
                    let (id, article) = articles.iter().next()?;
                    column_page(article, id)
                }
                _ => None,
            }
        }
    }
}

/// The page's rendered markup: an answer (the one the address names, else the
/// first) or a column article.
fn from_markup(document: &Html, wanted: &Target) -> Option<Article> {
    let root = document.root_element();
    fn rich<'a>(scope: ElementRef<'a>) -> Option<ElementRef<'a>> {
        first(
            scope,
            ".RichContent-inner .RichText, .Post-RichText .RichText, .RichText",
        )
    }
    match wanted {
        Target::Article(id) => {
            let content = first(root, ".Post-RichTextContainer, .Post-RichText")?;
            let content = rich(content).unwrap_or(content);
            let body = clean_children(content, Dialect::Zhihu);
            if !substantial(&body) {
                return None;
            }
            let title = text_at(root, ".Post-Title")?;
            let author = text_at(root, ".AuthorInfo-name");
            Some(Article {
                reader: "zhihu",
                title,
                byline: vec![author.clone().unwrap_or_default()],
                author,
                site: Some("知乎".into()),
                canonical: Some(format!("https://zhuanlan.zhihu.com/p/{id}")),
                body,
                ..Article::default()
            })
        }
        Target::Answer { .. } | Target::Unknown => {
            let wanted_id = match wanted {
                Target::Answer { answer, .. } => Some(answer.as_str()),
                _ => None,
            };
            let items: Vec<ElementRef<'_>> = root
                .select(&super::super::selector(".AnswerItem"))
                .collect();
            let item = items
                .iter()
                .find(|item| wanted_id.is_some_and(|id| item.value().attr("name") == Some(id)))
                .or_else(|| items.first())?;
            let content = rich(*item)?;
            let body = clean_children(content, Dialect::Zhihu);
            if !substantial(&body) {
                return None;
            }
            let title = text_at(root, ".QuestionHeader-title")
                .or_else(|| first(root, "h1").map(text_of).filter(|t| !t.is_empty()))?;
            let author = text_at(*item, ".AuthorInfo-name");
            Some(Article {
                reader: "zhihu",
                title,
                byline: vec![author.clone().unwrap_or_default()],
                author,
                site: Some("知乎".into()),
                body,
                ..Article::default()
            })
        }
        Target::Question(_) => None,
    }
}

pub(super) fn read(document: &Html, address: Option<&Url>) -> Option<Article> {
    let wanted = target(address);
    initial_data(document)
        .and_then(|data| from_data(&data, &wanted))
        .or_else(|| from_markup(document, &wanted))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formats::html::extract_html;
    use serde_json::json;

    fn page(entities: Value, body: &str) -> String {
        let data = json!({"initialState": {"entities": entities}});
        format!(
            r#"<!doctype html><html><head><title>Q - 知乎</title><meta charset="utf-8"></head>
            <body><div id="root">{body}</div>
            <script id="js-initialData" type="text/json">{data}</script></body></html>"#
        )
    }

    fn answer_entities() -> Value {
        json!({
            "answers": {
                "2176548792": {
                    "id": 2176548792u64,
                    "type": "answer",
                    "question": {"id": 48728378u64, "title": "What is a good question?"},
                    "author": {"name": "Ann", "headline": "Writes answers"},
                    "createdTime": 1650000000,
                    "voteupCount": 1234,
                    "content": "<p data-pid=\"a\">First paragraph with <a href=\"https://link.zhihu.com/?target=https%3A//example.test/ref\">a link</a>.</p><figure><noscript><img src=\"https://pic.test/a_720w.jpg\"></noscript><img src=\"data:image/svg+xml;utf8,&lt;svg/&gt;\" data-original=\"https://pic.test/a_r.jpg\" data-actualsrc=\"https://pic.test/a_720w.jpg\"><figcaption>A caption</figcaption></figure><pre lang=\"python\"><code>print(1 &lt; 2)</code></pre><p>Energy <img src=\"https://www.zhihu.com/equation?tex=E%3Dmc%5E2\" alt=\"E=mc^2\" eeimg=\"1\"> follows.</p>"
                },
                "1": {
                    "id": 1,
                    "question": {"id": 48728378u64, "title": "What is a good question?"},
                    "author": {"name": "Bob"},
                    "voteupCount": 5,
                    "content": "<p>Another answer that is not the one asked for.</p>"
                }
            },
            "questions": {"48728378": {"id": 48728378u64, "title": "What is a good question?", "detail": "<p>Question details here.</p>"}}
        })
    }

    #[test]
    fn an_answer_is_read_from_the_embedded_data() {
        let html = page(
            answer_entities(),
            "<p>rendered chrome and other answers</p>",
        );
        let url = "https://www.zhihu.com/question/48728378/answer/2176548792";
        let document = extract_html(&html, Some(url)).unwrap();
        let markdown = &document.markdown;
        assert!(
            markdown.starts_with("# What is a good question?"),
            "{markdown}"
        );
        assert!(
            markdown.contains("Ann · Writes answers · 赞同 1234 · 发布于 2022-04-15"),
            "{markdown}"
        );
        assert!(
            markdown.contains("First paragraph with [a link](https://example.test/ref)."),
            "{markdown}"
        );
        assert!(
            markdown.contains("![A caption](https://pic.test/a_r.jpg)")
                || markdown.contains("(https://pic.test/a_r.jpg)"),
            "{markdown}"
        );
        assert!(
            !markdown.contains("a_720w") && !markdown.contains("svg"),
            "{markdown}"
        );
        assert!(
            markdown.contains("```python\nprint(1 < 2)\n```"),
            "{markdown}"
        );
        assert!(markdown.contains("Energy $E=mc^2$ follows."), "{markdown}");
        assert!(!markdown.contains("Another answer"), "{markdown}");
        assert!(!markdown.contains("rendered chrome"), "{markdown}");
        assert_eq!(document.metadata["author"], "Ann");
        assert_eq!(document.metadata["published"], "2022-04-15T13:20:00+08:00");
        assert_eq!(document.metadata["upvotes"], 1234);
        assert_eq!(document.metadata["site"], "知乎");
    }

    #[test]
    fn a_saved_answer_is_recognized_by_its_comment_or_its_canonical_link() {
        let html = page(answer_entities(), "<p>chrome</p>");
        let url = "https://www.zhihu.com/question/48728378/answer/2176548792";
        let commented = format!("<!-- saved from url=(0070){url} -->{html}");
        let linked = html.replace(
            "<title>",
            &format!("<link rel=\"canonical\" href=\"{url}\"><title>"),
        );
        for saved in [commented, linked] {
            let document = extract_html(&saved, None).unwrap();
            assert!(
                document.markdown.contains("First paragraph"),
                "{}",
                document.markdown
            );
            assert!(
                !document.markdown.contains("Another answer"),
                "{}",
                document.markdown
            );
        }
        // Without an address, a page with two answers is not guessed at.
        let document = extract_html(&html, None).unwrap();
        assert!(
            !document.markdown.contains("First paragraph with"),
            "{}",
            document.markdown
        );
    }

    #[test]
    fn a_column_article_and_a_question_are_read() {
        let article = json!({"articles": {"123": {
            "title": "A column post", "content": "<h2>Part</h2><p>Body of the post.</p>",
            "author": {"name": "Cy"}, "created": 1580000000, "voteupCount": 9,
            "titleImage": "https://pic.test/cover.jpg"}}});
        let document =
            extract_html(&page(article, ""), Some("https://zhuanlan.zhihu.com/p/123")).unwrap();
        let markdown = &document.markdown;
        assert!(markdown.starts_with("# A column post"), "{markdown}");
        assert!(
            markdown.contains("![](https://pic.test/cover.jpg)"),
            "{markdown}"
        );
        assert!(
            markdown.contains("## Part") && markdown.contains("Body of the post."),
            "{markdown}"
        );
        assert_eq!(document.metadata["author"], "Cy");
        let document = extract_html(
            &page(answer_entities(), ""),
            Some("https://www.zhihu.com/question/48728378"),
        )
        .unwrap();
        let markdown = &document.markdown;
        assert!(
            markdown.starts_with("# What is a good question?"),
            "{markdown}"
        );
        assert!(markdown.contains("Question details here."), "{markdown}");
        let ann = markdown.find("Ann · Writes answers · 赞同 1234").unwrap();
        let bob = markdown.find("Bob · 赞同 5").unwrap();
        assert!(ann < bob, "{markdown}");
    }

    #[test]
    fn rendered_markup_is_read_when_the_data_is_missing() {
        let html = r#"<html><body><h1 class="QuestionHeader-title">Rendered question</h1>
            <div class="AnswerItem" name="7"><div class="AuthorInfo-name">Dee</div>
              <div class="RichContent-inner"><div class="RichText ztext"><p>Wanted answer text.</p></div></div></div>
            <div class="AnswerItem" name="8"><div class="RichContent-inner"><div class="RichText"><p>Other answer.</p></div></div></div>
            </body></html>"#;
        let document =
            extract_html(html, Some("https://www.zhihu.com/question/1/answer/8")).unwrap();
        assert!(
            document.markdown.starts_with("# Rendered question"),
            "{}",
            document.markdown
        );
        assert!(document.markdown.contains("Other answer."));
        assert!(!document.markdown.contains("Wanted answer"));
        let document =
            extract_html(html, Some("https://www.zhihu.com/question/1/answer/7")).unwrap();
        assert!(document.markdown.contains("Dee"), "{}", document.markdown);
        assert!(document.markdown.contains("Wanted answer text."));
    }

    #[test]
    fn a_page_without_an_answer_is_left_to_the_generic_reader() {
        let html = r#"<html><head><title>知乎</title></head><body><main><h1>首页</h1><p>这是一个没有回答的页面，但是它有足够多的文字内容可以被通用阅读器读到并输出。</p></main></body></html>"#;
        let document = extract_html(html, Some("https://www.zhihu.com/")).unwrap();
        assert!(document.markdown.contains("没有回答的页面"));
        assert!(!document.metadata.contains_key("upvotes"));
        // Malformed data is no reason to fail.
        let broken = html.replace(
            "</body>",
            "<script id=\"js-initialData\" type=\"text/json\">{not json</script></body>",
        );
        assert!(extract_html(&broken, Some("https://www.zhihu.com/question/1/answer/2")).is_ok());
    }

    #[test]
    fn addresses_name_what_they_point_at() {
        let url = |value: &str| target(Some(&Url::parse(value).unwrap()));
        assert_eq!(
            url("https://www.zhihu.com/question/1/answer/2?utm=x"),
            Target::Answer {
                question: Some("1".into()),
                answer: "2".into()
            }
        );
        assert_eq!(
            url("https://www.zhihu.com/answer/2"),
            Target::Answer {
                question: None,
                answer: "2".into()
            }
        );
        assert_eq!(
            url("https://zhuanlan.zhihu.com/p/33/"),
            Target::Article("33".into())
        );
        assert_eq!(
            url("https://www.zhihu.com/question/9"),
            Target::Question("9".into())
        );
        assert_eq!(url("https://www.zhihu.com/people/x"), Target::Unknown);
    }
}
