//! Page furniture beside an article's body.
//!
//! [`beside_body`] reads the content region of a full page as running text: the
//! body is the innermost element holding three fifths of the region's text
//! (words outside links and headings), and what stands beside it without saying
//! much is page furniture. Nothing here reads a class name: the evidence is
//! how much text a block holds, how it is made (links, headings, pictures,
//! code) and where it stands.

use super::article::{discarded, heading, note, structured_page};
use scraper::{ElementRef, Node};

/// The body must have this many words of running text ...
const MIN_BODY: usize = 70;
/// ... in at least this many runs as long as a paragraph.
const MIN_PARAGRAPHS: usize = 2;
const MIN_PARAGRAPH: usize = 14;
/// Blocks after the body are furniture below this many words together ...
const MAX_AFTER: usize = 50;
/// ... and when they hold under a third of the body's.
const AFTER_SHARE: usize = 3;
/// A leading block says no more than a label ...
const MAX_LABEL: usize = 2;
/// ... and a table of data more than a row of links.
const MAX_NAV_TABLE: usize = 4;
/// A short label above the title, in words.
const MAX_EYEBROW: usize = 4;
/// A row of links after the body's last text.
const MIN_LINKS: usize = 2;
const MIN_TAIL: usize = 25;

/// Whether a heading's text labels the article's own supporting matter: notes,
/// sources, an appendix. Such a section is content wherever it stands.
pub(super) fn scholarly_label(text: &str) -> bool {
    let label = text.trim().trim_end_matches(':').to_lowercase();
    [
        "references",
        "notes",
        "footnotes",
        "endnotes",
        "bibliography",
        "sources",
        "citations",
        "works cited",
        "acknowledgements",
        "acknowledgments",
        "appendix",
    ]
    .contains(&label.as_str())
}

/// One element of the region with what lies beneath it.
struct Mass<'a> {
    element: ElementRef<'a>,
    parent: Option<usize>,
    /// Words of running text, outside links and headings.
    text: usize,
    /// Words of all text.
    total: usize,
    /// Links.
    links: usize,
    /// Runs of running text as long as a paragraph.
    paragraphs: usize,
    /// Code, math, a note or a table of data: content, never furniture.
    keep: bool,
    /// A table read as data; it is content only when it says more than a row of
    /// navigation links.
    grid: bool,
    /// A heading not inside a link: the title of what follows.
    title: bool,
    /// A picture, a video or a date: something to see, not only to read.
    media: bool,
}

struct Region<'a> {
    nodes: Vec<Mass<'a>>,
    children: Vec<Vec<usize>>,
}

impl<'a> Region<'a> {
    /// Measure every element of `root` that is not itself left out of a page.
    fn scan(root: ElementRef<'a>) -> Self {
        let mut region = Self {
            nodes: Vec::new(),
            children: Vec::new(),
        };
        let mut stack = vec![(root, None, true)];
        while let Some((element, parent, plain)) = stack.pop() {
            if discarded(element) && parent.is_some() {
                continue;
            }
            let index = region.nodes.len();
            let name = element.value().name();
            let title = heading(element) && plain;
            let plain = plain
                && !heading(element)
                && !(name == "a" && element.value().attr("href").is_some());
            let (mut text, mut total, mut paragraphs) = (0, 0, 0);
            for child in element.children() {
                if let Node::Text(run) = child.value() {
                    let words = super::count_words(run);
                    total += words;
                    if plain {
                        text += words;
                        paragraphs += usize::from(words >= MIN_PARAGRAPH);
                    }
                }
            }
            region.nodes.push(Mass {
                element,
                parent,
                text,
                total,
                links: usize::from(name == "a" && element.value().attr("href").is_some()),
                paragraphs,
                keep: matches!(name, "pre" | "math")
                    || note(element)
                    || (heading(element) && scholarly_label(&super::plain(element))),
                grid: name == "table" && {
                    let (_, rows) = super::table_rows(element);
                    super::table_grid(element, &rows, true)
                },
                title,
                media: matches!(
                    name,
                    "time" | "img" | "picture" | "figure" | "video" | "audio" | "canvas"
                ),
            });
            region.children.push(Vec::new());
            if let Some(parent) = parent {
                region.children[parent].push(index);
            }
            stack.extend(
                element
                    .children()
                    .rev()
                    .filter_map(ElementRef::wrap)
                    .map(|child| (child, Some(index), plain)),
            );
        }
        // Children follow their parent, so each subtree is complete on its turn.
        for index in (1..region.nodes.len()).rev() {
            if region.nodes[index].grid && region.nodes[index].text > MAX_NAV_TABLE {
                region.nodes[index].keep = true;
            }
            let Some(parent) = region.nodes[index].parent else {
                continue;
            };
            let (text, total, links) = {
                let node = &region.nodes[index];
                (node.text, node.total, node.links)
            };
            let (keep, paragraphs, title, media) = {
                let node = &region.nodes[index];
                (node.keep, node.paragraphs, node.title, node.media)
            };
            let parent = &mut region.nodes[parent];
            parent.text += text;
            parent.total += total;
            parent.links += links;
            parent.paragraphs += paragraphs;
            parent.keep |= keep;
            parent.title |= title;
            parent.media |= media;
        }
        region
    }

    /// From the region's root down to the body: each element holds three fifths
    /// of its parent's running text.
    fn chain(&self) -> Vec<usize> {
        let mut chain = vec![0];
        let mut current = 0;
        while let Some(next) = self.children[current]
            .iter()
            .copied()
            .find(|child| self.nodes[*child].text * 5 >= self.nodes[current].text * 3)
        {
            chain.push(next);
            current = next;
        }
        chain
    }

    /// Siblings of the elements on the way to the body: before it, a block that
    /// is only a link or a label (a banner, a back link, a breadcrumb trail)
    /// and no title, date or picture; after it, what holds few words together.
    fn around(&self, chain: &[usize], found: &mut Vec<ElementRef<'a>>) {
        let body = self.nodes[*chain.last().unwrap_or(&0)].text;
        for pair in chain.windows(2) {
            let siblings = &self.children[pair[0]];
            let at = siblings
                .iter()
                .position(|child| *child == pair[1])
                .unwrap_or_default();
            found.extend(
                siblings[..at]
                    .iter()
                    .map(|child| &self.nodes[*child])
                    .filter(|node| {
                        node.links > 0
                            && node.text <= MAX_LABEL
                            && !node.keep
                            && !node.title
                            && !node.media
                    })
                    .map(|node| node.element),
            );
            let after = &siblings[at + 1..];
            let words: usize = after.iter().map(|child| self.nodes[*child].text).sum();
            if words <= MAX_AFTER && words * AFTER_SHARE <= body {
                found.extend(
                    after
                        .iter()
                        .map(|child| &self.nodes[*child])
                        .filter(|node| !node.keep)
                        .map(|node| node.element),
                );
            }
        }
    }

    /// A label above the page's title (a category, a kicker): a short block of
    /// words without digits or punctuation that makes a sentence, right before
    /// the first `h1` in its container.
    fn eyebrow(&self, found: &mut Vec<ElementRef<'a>>) {
        let Some(title) = self
            .nodes
            .iter()
            .position(|node| node.element.value().name() == "h1")
            .filter(|title| self.nodes[*title].title)
        else {
            return;
        };
        let Some(parent) = self.nodes[title].parent else {
            return;
        };
        for &sibling in self.children[parent]
            .iter()
            .rev()
            .skip_while(|child| **child != title)
            .skip(1)
        {
            let node = &self.nodes[sibling];
            let label = super::plain(node.element);
            if node.total == 0
                || node.total > MAX_EYEBROW
                || node.keep
                || node.title
                || node.media
                || label.chars().any(|ch| ch.is_ascii_digit())
                || label.contains(['.', '?', '!', ':', ';', ','])
            {
                break;
            }
            found.push(node.element);
        }
    }

    /// Inside the body, what follows its last running text, code or picture and
    /// is a row of links with at most a label: tags, related posts, previous
    /// and next. A paragraph or a list is text however short, and a heading
    /// starts a section that stays.
    fn tail(&self, body: usize, found: &mut Vec<ElementRef<'a>>) {
        let node = &self.nodes[body];
        if node.text < MIN_TAIL
            || node.paragraphs < MIN_PARAGRAPHS
            || !matches!(
                node.element.value().name(),
                "article" | "main" | "section" | "div" | "body"
            )
        {
            return;
        }
        let mut tail = Vec::new();
        let mut sealed = false;
        let mut cursor = 0;
        for child in node.element.children() {
            match child.value() {
                Node::Text(text) if text.chars().any(|ch| !ch.is_whitespace()) => {
                    tail.clear();
                    sealed = false;
                }
                Node::Element(_) => {
                    let Some(&index) = self.children[body].get(cursor) else {
                        continue;
                    };
                    if self.nodes[index].element.id() != child.id() {
                        continue;
                    }
                    cursor += 1;
                    let block = &self.nodes[index];
                    let sentence = matches!(
                        block.element.value().name(),
                        "p" | "ul" | "ol" | "dl" | "blockquote"
                    );
                    if block.text > MAX_LABEL
                        || (sentence && block.text > 0)
                        || block.keep
                        || block.media
                    {
                        tail.clear();
                        sealed = false;
                    } else if block.title {
                        tail.clear();
                        sealed = true;
                    } else if !sealed && !sentence && block.links >= MIN_LINKS {
                        tail.push(block.element);
                    }
                }
                _ => {}
            }
        }
        found.extend(tail);
    }
}

/// Blocks of a full page's content region that are furniture beside the body.
///
/// The body is the innermost element holding three fifths of the region's
/// running text, and it must read as an article (enough words, at least two
/// paragraph-long runs). Then, on the way from the region's root to the body:
///
/// - a leading block that is only a link or a two-word label, with no title,
///   date or picture, is a banner or a breadcrumb and goes; a lede or a byline
///   stays;
/// - the siblings after it go when together they hold under fifty words and a
///   third of the body's: a subscribe box, a call to action, related-post
///   cards, an author bio.
///
/// A block with code, math, a note, a section of notes or sources, or a table
/// of data (more than a row of links) always stays. A short label right above
/// the first `h1`, and a row of links after the body's last text (see
/// [`Region::tail`]), go too. A region without such a body keeps everything.
pub(super) fn beside_body(root: ElementRef<'_>) -> Vec<ElementRef<'_>> {
    if structured_page(root) {
        return Vec::new();
    }
    let region = Region::scan(root);
    let chain = region.chain();
    let body = chain.last().copied().unwrap_or(0);
    let mut found = Vec::new();
    if region.nodes[body].paragraphs >= MIN_PARAGRAPHS && region.nodes[body].text >= MIN_BODY {
        region.around(&chain, &mut found);
    }
    if region.nodes[0].paragraphs >= MIN_PARAGRAPHS {
        region.eyebrow(&mut found);
    }
    region.tail(body, &mut found);
    found
}

#[cfg(test)]
mod tests {
    use super::super::article::select;
    use super::*;
    use scraper::Html;

    /// `words` words of running text.
    fn prose(words: usize) -> String {
        (0..words)
            .map(|index| format!("word{index}"))
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Three paragraphs of running text: an article's body.
    fn body() -> String {
        format!(
            "<p>{}</p><p>{}</p><p>{}</p>",
            prose(40),
            prose(40),
            prose(40)
        )
    }

    /// The ids of the elements left out of the page's content region.
    fn left_out(html: &str) -> Vec<String> {
        let document = Html::parse_document(html);
        let mut ids = beside_body(select(&document))
            .into_iter()
            .map(|element| {
                element
                    .value()
                    .attr("id")
                    .unwrap_or(element.value().name())
                    .to_owned()
            })
            .collect::<Vec<_>>();
        ids.sort();
        ids
    }

    #[test]
    fn blocks_after_the_body_that_say_little_are_furniture() {
        let ids = left_out(&format!(
            r##"<body><div><div class="post"><h1>Title</h1>{}</div>
            <div id="bio"><h3>About the author</h3><p>Jane writes about databases.</p><a href="/jane">More</a></div>
            <div id="signup"><p>Subscribe to our newsletter for updates.</p></div>
            <div id="cards"><article><h3><a href="/a">Other post</a></h3></article><article><h3><a href="/b">Another post</a></h3></article></div></div></body>"##,
            body()
        ));
        assert_eq!(ids, ["bio", "cards", "signup"]);
    }

    #[test]
    fn what_follows_the_body_stays_when_it_says_much() {
        // Over fifty words together, though a small share of a long body ...
        let long = left_out(&format!(
            r#"<body><div><div class="post">{}</div><div id="more"><p>{}</p></div></div></body>"#,
            body().repeat(2),
            prose(60)
        ));
        assert!(long.is_empty(), "{long:?}");
        // ... or over a third of a short body's.
        let short = |after: usize| {
            format!(
                r#"<body><div><div class="post"><p>{}</p><p>{}</p></div><div id="more"><p>{}</p></div></div></body>"#,
                prose(40),
                prose(40),
                prose(after)
            )
        };
        assert!(left_out(&short(30)).is_empty());
        assert_eq!(left_out(&short(20)), ["more"]);
    }

    #[test]
    fn code_math_notes_and_data_tables_after_the_body_stay() {
        let table = "<table><tr><th>Year</th><th>Total</th></tr><tr><td>2024</td><td>10 apples and pears</td></tr><tr><td>2025</td><td>12 apples and pears</td></tr></table>";
        for kept in [
            "<pre>let x = 1;</pre>",
            r#"<div class="footnotes"><p>Note one.</p></div>"#,
            "<section><h2>References</h2><p>Smith 2020.</p></section>",
            "<section><h2>Notes:</h2><p>An aside.</p></section>",
            table,
        ] {
            let ids = left_out(&format!(
                r#"<body><div><div class="post">{}</div><div id="rest">{kept}</div></div></body>"#,
                body()
            ));
            assert!(ids.is_empty(), "{kept}: {ids:?}");
        }
        let ids = left_out(&format!(
            r#"<body><div><div class="post">{}</div><div id="rest"><p>Written by Jane.</p></div></div></body>"#,
            body()
        ));
        assert_eq!(ids, ["rest"]);
    }

    #[test]
    fn a_table_of_links_is_navigation_but_a_table_of_data_is_content() {
        let nav = r#"<table><tr><td>Previous:</td><td><a href="/a">The last essay about things</a></td></tr><tr><td>Next:</td><td><a href="/b">The next essay about things</a></td></tr></table>"#;
        let ids = left_out(&format!(
            r#"<body><div><div class="post">{}</div><div id="nav">{nav}</div></div></body>"#,
            body()
        ));
        assert_eq!(ids, ["nav"]);
        let data = r#"<table><tr><td>Region</td><td>Sales</td></tr><tr><td>North America</td><td>Grew strongly</td></tr><tr><td>Europe</td><td>Held steady</td></tr></table>"#;
        let ids = left_out(&format!(
            r#"<body><div><div class="post">{}</div><div id="data">{data}</div></div></body>"#,
            body()
        ));
        assert!(ids.is_empty(), "{ids:?}");
    }

    #[test]
    fn before_the_body_only_a_banner_or_label_goes_and_the_title_and_lede_stay() {
        let ids = left_out(&format!(
            r##"<body><div><a id="banner" href="/event"><div>You are invited to the launch event</div></a>
            <div id="crumbs"><a href="/">Home</a> / <a href="/blog">Blog</a></div>
            <div id="title"><h1>Title</h1></div>
            <div id="series"><a href="/s">Series</a><h2>Part one</h2></div>
            <div id="read">Read <a href="/x">this</a> before you start the article</div>
            <div id="sponsor">Sponsored</div>
            <div id="hero"><img src="hero.jpg" alt=""></div>
            <div id="date"><a href="/d"><time>May 1</time></a></div>
            <p id="lede">A short lede that sums up the article in one line.</p>
            <div class="post">{}</div></div></body>"##,
            body()
        ));
        assert_eq!(ids, ["banner", "crumbs"]);
    }

    #[test]
    fn a_short_label_of_words_above_the_first_title_is_an_eyebrow() {
        let page = |before: &str| {
            format!(
                r#"<body><div>{before}<h1>Title</h1>{}</div></body>"#,
                body()
            )
        };
        assert_eq!(
            left_out(&page(r#"<p id="kicker">Blog post</p>"#)),
            ["kicker"]
        );
        assert_eq!(
            left_out(&page(
                r#"<div id="tag"><span>Product</span><span>News</span></div>"#
            )),
            ["tag"]
        );
        // The run of labels right above the title, no further.
        assert_eq!(
            left_out(&page(
                r#"<p>Earlier paragraph that is not a label at all, so it stays.</p><p id="a">Opinion</p><p id="b">Science</p>"#
            )),
            ["a", "b"]
        );
        // A date, a sentence or a long label is content.
        for kept in [
            "<p>May 22, 2026</p>",
            "<p>22 May 2026</p>",
            "<p>Part 2</p>",
            "<p>Breaking news.</p>",
            "<p>Update: what happened today</p>",
            "<p>A rather long introduction above</p>",
        ] {
            assert!(left_out(&page(kept)).is_empty(), "{kept}");
        }
    }

    #[test]
    fn a_row_of_links_after_the_last_text_of_the_body_is_furniture() {
        let page = |tail: &str| {
            format!(
                r#"<html><body><article><h1>Title</h1>{}{tail}</article></body></html>"#,
                body()
            )
        };
        let tags = r#"<div id="tags"><a href="/t/a">a</a> <a href="/t/b">b</a> <a href="/t/c">c</a></div>"#;
        assert_eq!(left_out(&page(tags)), ["tags"]);
        // A row of related-post paragraphs with dots between.
        let related = r#"<section id="rel"><p><a href="/x">First</a> · <a href="/t/x">x</a></p><p><a href="/y">Second</a> · <a href="/t/y">y</a></p></section>"#;
        assert_eq!(left_out(&page(related)), ["rel"]);
        // A labelled section, a paragraph, a list, one link and mid-body links stay.
        for kept in [
            format!("<h2>Links</h2>{tags}"),
            r#"<p>See also: <a href="/a">a</a>, <a href="/b">b</a>, <a href="/c">c</a></p>"#
                .to_owned(),
            r#"<ul><li><a href="/a">a</a></li><li><a href="/b">b</a></li></ul>"#.to_owned(),
            r#"<div><a href="/a">Download the full report</a></div>"#.to_owned(),
            format!("{tags}<p>{}</p>", prose(20)),
            format!("{tags} and the text goes on here"),
        ] {
            assert!(left_out(&page(&kept)).is_empty(), "{kept}");
        }
    }

    #[test]
    fn without_a_dominant_article_body_everything_stays() {
        // Sibling articles of equal weight, a short body, an introduction with
        // a list of links, and a page of one long paragraph.
        let equal = format!(
            r#"<body><main><article>{}</article><article>{}</article><div id="x"><p>Short.</p></div></main></body>"#,
            body(),
            body()
        );
        assert!(left_out(&equal).is_empty());
        let short = r#"<body><div><div><p>Two short lines.</p><p>Nothing more.</p></div><div id="x"><a href="/a">a</a> <a href="/b">b</a></div></div></body>"#;
        assert!(left_out(short).is_empty());
        // Two paragraphs that are too few words, one paragraph that is enough.
        let links = r#"<div id="x"><a href="/a">a</a> <a href="/b">b</a></div>"#;
        let thin = format!(
            "<body><div><div><p>{}</p><p>{}</p></div>{links}</div></body>",
            prose(20),
            prose(20)
        );
        assert!(left_out(&thin).is_empty());
        let single = format!(
            "<body><div><div><p>{}</p></div>{links}</div></body>",
            prose(100)
        );
        assert!(left_out(&single).is_empty());
        let intro = format!(
            r#"<body><div><div><p>{}</p></div><div id="x"><ul><li><a href="/a">a</a></li><li><a href="/b">b</a></li></ul></div></div></body>"#,
            prose(60)
        );
        assert!(left_out(&intro).is_empty());
    }

    #[test]
    fn repository_readmes_and_issue_pages_have_their_own_boundary() {
        let issue = format!(
            r#"<body><div data-testid="issue-viewer-container"><div class="post">{}</div><div id="bio"><p>Little.</p></div></div></body>"#,
            body()
        );
        assert!(left_out(&issue).is_empty());
    }

    #[test]
    fn labels_of_notes_and_sources_are_scholarly() {
        for label in ["References", " Notes: ", "WORKS CITED", "Appendix"] {
            assert!(scholarly_label(label), "{label}");
        }
        for other in ["Related articles", "See also", "Notes on style"] {
            assert!(!scholarly_label(other), "{other}");
        }
    }
}
