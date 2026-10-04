//! Page furniture beside an article's body.
//!
//! [`beside_body`] reads the content region of a full page as running text: the
//! body is the innermost element holding three fifths of the region's text
//! (words outside links and headings), and what stands beside it without saying
//! much is page furniture. The evidence is how much text a block holds, how it
//! is made (links, headings, pictures, code, dates) and where it stands; short
//! marks (a reading time, a count of likes) are read from their words, and only
//! a featured comment inside the body is known by its class name.

use super::Attribute;
use super::article::{
    discarded, featured_comment, furniture, heading, note, structured_page, teaser,
};
use super::facts::Facts;
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
/// A mark of the page (a reading time, a counter) in words ...
const MAX_MARK: usize = 6;
/// ... and in characters.
const MAX_MARK_CHARS: usize = 48;
/// A breadcrumb trail has fewer links than this, and the page it ends on at
/// most this many words.
const MAX_CRUMBS: usize = 6;
const MAX_CRUMB_PAGE: usize = 12;
/// An author's card, in words.
const MAX_CARD: usize = 80;
/// A sentence introducing what follows it with a colon, in words.
const MIN_INTRO: usize = 5;

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
    /// Words of running text in the element's own text, outside its children.
    own: usize,
    /// The index after the element's last descendant.
    end: usize,
    /// A `time` element.
    dated: bool,
    /// A picture.
    picture: bool,
    /// A link to the article's author (`rel=author`, `itemprop=author`).
    author: bool,
    /// Inside a paragraph, a list item, a table, a heading, a quote or code,
    /// where a short text is part of the sentence or cell around it.
    inline: bool,
}

/// The elements of a region in document order, each before its descendants.
struct Region<'a> {
    nodes: Vec<Mass<'a>>,
}

/// Elements whose text is a sentence, a cell or code: what stands inside them
/// is part of that text, never a mark of the page by itself.
fn sentence_or_cell(element: ElementRef<'_>) -> bool {
    heading(element)
        || matches!(
            element.value().name(),
            "p" | "li"
                | "dt"
                | "dd"
                | "table"
                | "caption"
                | "figcaption"
                | "blockquote"
                | "pre"
                | "code"
        )
}

impl<'a> Region<'a> {
    /// Measure every element of `root` that is not itself left out of a page.
    fn scan(facts: &Facts, root: ElementRef<'a>) -> Self {
        let mut region = Self { nodes: Vec::new() };
        let mut stack = vec![(root, None, true, false)];
        while let Some((element, parent, plain, inline)) = stack.pop() {
            if discarded(facts, element) && parent.is_some() {
                continue;
            }
            let index = region.nodes.len();
            let value = element.value();
            let name = value.name();
            let title = heading(element) && plain;
            let plain =
                plain && !heading(element) && !(name == "a" && value.attribute("href").is_some());
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
                links: usize::from(name == "a" && value.attribute("href").is_some()),
                paragraphs,
                keep: matches!(name, "pre" | "math")
                    || note(facts, element)
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
                own: text,
                end: index + 1,
                dated: name == "time",
                picture: matches!(name, "img" | "picture"),
                author: value.attribute("itemprop") == Some("author")
                    || (name == "a"
                        && value.attribute("rel").is_some_and(|rel| {
                            rel.split_ascii_whitespace()
                                .any(|token| token.eq_ignore_ascii_case("author"))
                        })),
                inline,
            });
            let inline = inline || sentence_or_cell(element);
            stack.extend(
                element
                    .children()
                    .rev()
                    .filter_map(ElementRef::wrap)
                    .map(|child| (child, Some(index), plain, inline)),
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
            let (text, total, links, paragraphs, end) = {
                let node = &region.nodes[index];
                (node.text, node.total, node.links, node.paragraphs, node.end)
            };
            let (keep, title, media, dated, picture, author) = {
                let node = &region.nodes[index];
                (
                    node.keep,
                    node.title,
                    node.media,
                    node.dated,
                    node.picture,
                    node.author,
                )
            };
            let parent = &mut region.nodes[parent];
            parent.text += text;
            parent.total += total;
            parent.links += links;
            parent.paragraphs += paragraphs;
            parent.end = parent.end.max(end);
            parent.keep |= keep;
            parent.title |= title;
            parent.media |= media;
            parent.dated |= dated;
            parent.picture |= picture;
            parent.author |= author;
        }
        region
    }

    /// An element's children: the first follows it, and each next one starts
    /// where the subtree of the one before ends.
    fn children(&self, parent: usize) -> impl Iterator<Item = usize> + '_ {
        let end = self.nodes[parent].end;
        let mut next = parent + 1;
        std::iter::from_fn(move || {
            (next < end).then(|| {
                let child = next;
                next = self.nodes[child].end;
                child
            })
        })
    }

    /// From the region's root down to the body: each element holds three fifths
    /// of its parent's running text.
    fn chain(&self) -> Vec<usize> {
        let mut chain = vec![0];
        let mut current = 0;
        while let Some(next) = self
            .children(current)
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
    /// Replies that repeat the post they follow stay, and so does the block
    /// right after the body when the body introduces it with a colon or when
    /// it is a conclusion of plain prose.
    fn around(&self, facts: &Facts, chain: &[usize], found: &mut Vec<ElementRef<'a>>) {
        let last = *chain.last().unwrap_or(&0);
        let body = self.nodes[last].text;
        // The first block after the body, from its own level outwards.
        let next = chain.windows(2).rev().find_map(|pair| {
            let siblings: Vec<usize> = self.children(pair[0]).collect();
            let at = siblings.iter().position(|child| *child == pair[1])?;
            siblings[at + 1..].iter().copied().find(|child| {
                let node = &self.nodes[*child];
                node.total > 0 || node.links > 0 || node.media
            })
        });
        let introduced = next.is_some() && introduces(facts, self.nodes[last].element);
        for pair in chain.windows(2) {
            let siblings: Vec<usize> = self.children(pair[0]).collect();
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
            let (named, after): (Vec<usize>, Vec<usize>) = siblings[at + 1..]
                .iter()
                .copied()
                .filter(|child| {
                    !(self.repeats(pair[1], *child)
                        || (Some(*child) == next
                            && (introduced
                                || (self.conclusion(*child)
                                    && !furniture(self.nodes[*child].element)))))
                })
                .partition(|child| furniture(self.nodes[*child].element));
            // A block named as furniture (`article-footer`, `share-bar`,
            // `newsletter`) goes however much it says.
            found.extend(
                named
                    .iter()
                    .map(|child| &self.nodes[*child])
                    .filter(|node| !node.keep)
                    .map(|node| node.element),
            );
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

    /// A reply after the post it follows in a thread: the same element with
    /// the same first class (or a plain `article`, the element of a comment),
    /// both dated, the reply with words of its own besides its date and not a
    /// teaser of another page.
    fn repeats(&self, post: usize, reply: usize) -> bool {
        fn class<'a>(node: &Mass<'a>) -> Option<&'a str> {
            node.element
                .value()
                .attribute("class")
                .and_then(|class| class.split_ascii_whitespace().next())
        }
        let (first, next) = (&self.nodes[post], &self.nodes[reply]);
        first.element.value().name() == next.element.value().name()
            && class(first) == class(next)
            && (class(first).is_some() || first.element.value().name() == "article")
            && first.dated
            && next.dated
            && !teaser(next.element)
            && self.says_more_than_a_date(reply)
    }

    /// Whether a block has running text of its own outside its `time`s.
    fn says_more_than_a_date(&self, block: usize) -> bool {
        let mut index = block;
        while index < self.nodes[block].end {
            let node = &self.nodes[index];
            if node.element.value().name() == "time" {
                index = node.end;
            } else if node.own > 0 {
                return true;
            } else {
                index += 1;
            }
        }
        false
    }

    /// A conclusion: plain prose with a paragraph-long run, with or without a
    /// heading, and without the links, pictures or dates that a bio, a call
    /// to action or a related post would show.
    fn conclusion(&self, block: usize) -> bool {
        let node = &self.nodes[block];
        node.paragraphs > 0 && node.links == 0 && !node.media
    }

    /// Marks of the page beside the article's prose:
    ///
    /// - anywhere off the way to the body, a block whose whole text is a
    ///   reading time or a count of likes, views, shares or comments, unless
    ///   it is a list item, a cell or part of a sentence;
    /// - before the body's first paragraph, a breadcrumb trail (a list of
    ///   links ending on the page's own name) and a block that only repeats an
    ///   earlier date;
    /// - outside the body, an author's card: a picture, a link to the author
    ///   and a sentence about them, without a heading or a date.
    fn marks(&self, chain: &[usize], found: &mut Vec<ElementRef<'a>>) {
        let body = *chain.last().unwrap_or(&0);
        // The body's lede ends at its first paragraph-long run of prose.
        let lede = (body..self.nodes[body].end)
            .find(|index| self.nodes[*index].own >= MIN_PARAGRAPH)
            .unwrap_or(body);
        let mut dates: Vec<&str> = Vec::new();
        let mut index = 1;
        while index < self.nodes.len() {
            let node = &self.nodes[index];
            if chain.contains(&index) {
                index += 1;
                continue;
            }
            let outside = index < body || index >= self.nodes[body].end;
            let mark = if self.mark(index) || (index < lede && self.crumbs(index)) {
                Some(index)
            } else if index < lede
                && node.element.value().name() == "time"
                && let Some(date) = node
                    .element
                    .value()
                    .attribute("datetime")
                    .and_then(|stamp| stamp.trim().get(..10))
            {
                if dates.contains(&date) {
                    self.date_line(index, chain)
                } else {
                    dates.push(date);
                    None
                }
            } else if outside && self.card(index) {
                Some(self.smallest_card(index))
            } else {
                None
            };
            match mark {
                Some(mark) => {
                    found.push(self.nodes[mark].element);
                    index = self.nodes[mark].end.max(index + 1);
                }
                None => index += 1,
            }
        }
    }

    /// A block whose whole text is a reading time or a counter.
    fn mark(&self, index: usize) -> bool {
        let node = &self.nodes[index];
        if node.total == 0
            || node.total > MAX_MARK
            || node.inline
            || node.keep
            || node.title
            || node.media
            || (sentence_or_cell(node.element) && node.element.value().name() != "p")
            // Every reading time and counter has a number.
            || !node
                .element
                .text()
                .any(|text| text.bytes().any(|byte| byte.is_ascii_digit()))
        {
            return false;
        }
        let text = super::plain(node.element);
        text.len() <= MAX_MARK_CHARS && {
            let words = label_words(&text);
            let words: Vec<&str> = words.iter().map(String::as_str).collect();
            reading_time(&words) || counter(&words)
        }
    }

    /// A breadcrumb trail: a list of two to five steps, each a single link,
    /// that ends on the page's own name as plain text.
    fn crumbs(&self, list: usize) -> bool {
        let node = &self.nodes[list];
        if !matches!(node.element.value().name(), "ul" | "ol") || node.inline || node.media {
            return false;
        }
        let steps: Vec<usize> = self.children(list).collect();
        let Some((page, links)) = steps.split_last() else {
            return false;
        };
        let page = &self.nodes[*page];
        (MIN_LINKS..MAX_CRUMBS).contains(&links.len())
            && steps
                .iter()
                .all(|step| self.nodes[*step].element.value().name() == "li")
            && links.iter().all(|step| {
                let step = &self.nodes[*step];
                step.links == 1 && step.text == 0
            })
            && page.links == 0
            && page.text > 0
            && page.total <= MAX_CRUMB_PAGE
    }

    /// The block that holds nothing but a repeated date, when it is a block
    /// of its own and not part of a byline.
    fn date_line(&self, time: usize, chain: &[usize]) -> Option<usize> {
        let words = self.nodes[time].total;
        let mut line = time;
        while let Some(parent) = self.nodes[line].parent
            && self.nodes[parent].total == words
            && !chain.contains(&parent)
        {
            line = parent;
        }
        super::block_tag(self.nodes[line].element.value().name()).then_some(line)
    }

    /// An author's card: a picture, a link to the author and a sentence about
    /// them, in few words, without a heading, a date or kept content.
    fn card(&self, index: usize) -> bool {
        let node = &self.nodes[index];
        node.author
            && node.picture
            && node.paragraphs > 0
            && node.total <= MAX_CARD
            && !node.title
            && !node.dated
            && !node.keep
    }

    /// The innermost card inside a card (the card, not the wrapper of the
    /// page's front matter around it).
    fn smallest_card(&self, mut index: usize) -> usize {
        while let Some(inner) = self.children(index).find(|child| self.card(*child)) {
            index = inner;
        }
        index
    }

    /// Inside the body, a featured comment (a block named as a top, featured,
    /// hot, best or pinned comment) with the article's own prose both before
    /// and after it: a reader's comment set into the article, not the page's
    /// discussion or a page about comments.
    fn featured(&self, body: usize, found: &mut Vec<ElementRef<'a>>) {
        let end = self.nodes[body].end;
        let mut blocks: Vec<usize> = Vec::new();
        let mut index = body + 1;
        while index < end {
            if featured_comment(self.nodes[index].element) {
                blocks.push(index);
                index = self.nodes[index].end;
            } else {
                index += 1;
            }
        }
        if blocks.is_empty() {
            return;
        }
        let prose = |index: usize| {
            self.nodes[index].own > 0
                && !blocks
                    .iter()
                    .any(|block| (*block..self.nodes[*block].end).contains(&index))
        };
        for &block in &blocks {
            if (body + 1..block).any(prose) && (self.nodes[block].end..end).any(prose) {
                found.push(self.nodes[block].element);
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
        let siblings: Vec<usize> = self.children(parent).collect();
        for &sibling in siblings
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
    /// and next. A paragraph or a list is text however short, a heading
    /// starts a section that stays, and a row that a sentence ending with a
    /// colon introduces is the article's.
    fn tail(&self, facts: &Facts, body: usize, found: &mut Vec<ElementRef<'a>>) {
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
        let mut blocks = self.children(body).peekable();
        // The block of text right before, which may introduce a row.
        let mut text = None;
        for child in node.element.children() {
            match child.value() {
                Node::Text(run) if run.chars().any(|ch| !ch.is_whitespace()) => {
                    tail.clear();
                    sealed = false;
                    text = None;
                }
                Node::Element(_) => {
                    let Some(&index) = blocks.peek() else {
                        continue;
                    };
                    if self.nodes[index].element.id() != child.id() {
                        continue;
                    }
                    blocks.next();
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
                        text = (!block.keep).then_some(block.element);
                    } else if block.title {
                        tail.clear();
                        sealed = true;
                        text = None;
                    } else if !sealed && !sentence && block.links >= MIN_LINKS {
                        if text.take().is_some_and(|text| introduces(facts, text)) {
                            tail.clear();
                        } else {
                            tail.push(block.element);
                        }
                    }
                }
                _ => {}
            }
        }
        found.extend(tail);
    }
}

/// Whether an element ends with a sentence that ends with a colon: an
/// introduction to what follows it. A label ("Related:", "Tags:") is not one.
fn introduces(facts: &Facts, element: ElementRef<'_>) -> bool {
    let mut stack: Vec<_> = element.children().collect();
    while let Some(node) = stack.pop() {
        match node.value() {
            Node::Text(text) => {
                let text = text.trim_end();
                if text.is_empty() {
                    continue;
                }
                if !text.ends_with([':', '：']) {
                    return false;
                }
                // The words of the line that ends there: back to a line break
                // or the start of its block.
                let mut words = super::count_words(text);
                let mut at = node;
                loop {
                    for sibling in at.prev_siblings() {
                        match sibling.value() {
                            Node::Text(text) => words += super::count_words(text),
                            Node::Element(value)
                                if value.name() == "br" || super::block_tag(value.name()) =>
                            {
                                return words >= MIN_INTRO;
                            }
                            Node::Element(_) => {
                                words += ElementRef::wrap(sibling)
                                    .map_or(0, |inline| super::count_words(&super::plain(inline)));
                            }
                            _ => {}
                        }
                    }
                    match at.parent() {
                        Some(parent)
                            if parent.id() != element.id()
                                && ElementRef::wrap(parent).is_some_and(|parent| {
                                    !super::block_tag(parent.value().name())
                                }) =>
                        {
                            at = parent;
                        }
                        _ => return words >= MIN_INTRO,
                    }
                }
            }
            Node::Element(_) => {
                if let Some(child) = ElementRef::wrap(node)
                    && !discarded(facts, child)
                {
                    stack.extend(child.children());
                }
            }
            _ => {}
        }
    }
    false
}

/// The words of a short label, lowercased: letters and digits split apart
/// (`8min` → `8`, `min`), a number keeping its separators (`1,234`, `1.2`).
fn label_words(text: &str) -> Vec<String> {
    let mut words: Vec<String> = Vec::new();
    let mut digits = false;
    for ch in text.chars().flat_map(char::to_lowercase) {
        let numeric = ch.is_ascii_digit() || (digits && matches!(ch, '.' | ','));
        if !(numeric || ch.is_alphanumeric()) {
            digits = false;
            words.push(String::new());
            continue;
        }
        match words.last_mut() {
            Some(word) if !word.is_empty() && numeric == digits => word.push(ch),
            _ => words.push(ch.to_string()),
        }
        digits = numeric;
    }
    words
        .into_iter()
        .map(|word| word.trim_end_matches(['.', ',']).to_owned())
        .filter(|word| !word.is_empty())
        .collect()
}

/// A reading time: "8 min read", "5-minute read", "Reading time: 3 minutes".
fn reading_time(words: &[&str]) -> bool {
    let number = |word: &str| word.len() <= 3 && word.bytes().all(|byte| byte.is_ascii_digit());
    let unit = |word: &str| matches!(word, "min" | "mins" | "minute" | "minutes");
    match words {
        [count, per, read] => number(count) && unit(per) && matches!(*read, "read" | "reading"),
        ["reading" | "read", "time", count, per] => number(count) && unit(per),
        _ => false,
    }
}

/// A counter: "9 Likes", "1.2K views", "3,456 shares", "12 comments".
fn counter(words: &[&str]) -> bool {
    // A number keeps its separators only after a digit (see `label_words`).
    let number = |word: &str| {
        word.bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'.' | b','))
    };
    let (count, what) = match words {
        [count, what] => (*count, *what),
        [count, "k" | "m", what] => (*count, *what),
        _ => return false,
    };
    number(count)
        && matches!(
            what,
            "like" | "likes" | "view" | "views" | "share" | "shares" | "comment" | "comments"
        )
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
///   cards, an author bio; replies repeating the post they follow, and a
///   conclusion or a list the body introduces right after it, stay.
///
/// Marks of the page (see [`Region::marks`]) and a featured comment set into
/// the body (see [`Region::featured`]) go as well. A block with code, math, a
/// note, a section of notes or sources, or a table of data (more than a row of
/// links) always stays. A short label right above the first `h1`, and a row of
/// links after the body's last text (see [`Region::tail`]), go too. A region
/// without such a body keeps everything.
pub(super) fn beside_body<'a>(facts: &Facts, root: ElementRef<'a>) -> Vec<ElementRef<'a>> {
    if structured_page(root) {
        return Vec::new();
    }
    let region = Region::scan(facts, root);
    let chain = region.chain();
    let body = chain.last().copied().unwrap_or(0);
    let mut found = Vec::new();
    if region.nodes[body].paragraphs >= MIN_PARAGRAPHS && region.nodes[body].text >= MIN_BODY {
        region.around(facts, &chain, &mut found);
        region.marks(&chain, &mut found);
        region.featured(body, &mut found);
    }
    if region.nodes[0].paragraphs >= MIN_PARAGRAPHS {
        region.eyebrow(&mut found);
    }
    region.tail(facts, body, &mut found);
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
        let facts = Facts::new(document.root_element());
        let mut ids = beside_body(&facts, select(&document, &facts))
            .into_iter()
            .map(|element| {
                element
                    .value()
                    .attribute("id")
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
    fn blocks_after_the_body_named_as_furniture_go_however_much_they_say() {
        let ids = left_out(&format!(
            r##"<body><div><div class="post"><h1>Title</h1>{}</div>
            <section id="footer" class="article-footer"><h2>Help improve the docs</h2><p>{}</p></section>
            <div id="comments"><p>{}</p></div>
            <div id="more"><p>{}</p></div>
            <div class="footnotes" id="notes"><p>{}</p></div></div></body>"##,
            body().repeat(3),
            prose(40),
            prose(80),
            prose(60),
            prose(30)
        ));
        assert_eq!(ids, ["comments", "footer"]);
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
        // ... or over a third of a short body's (a link keeps the block from
        // reading as a conclusion).
        let short = |after: usize| {
            format!(
                r#"<body><div><div class="post"><p>{}</p><p>{}</p></div><div id="more"><p>{} <a href="/x">x</a></p></div></div></body>"#,
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
        // A label is not an introduction, but a sentence ending with a colon is.
        assert_eq!(left_out(&page(&format!("<p>Tags:</p>{tags}"))), ["tags"]);
        let formats = format!("<p>Download the report in these formats:</p>{tags}");
        assert!(left_out(&page(&formats)).is_empty());
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
    fn replies_repeating_the_post_they_follow_stay() {
        let post = |class: &str, text: &str| {
            format!(
                r#"<div class="{class}"><p>by <a href="/u">ann</a> <time datetime="2026-03-01">Mar 1</time></p><div>{text}</div></div>"#
            )
        };
        let thread = |replies: &str| {
            format!(
                r#"<body><main><div class="topic">{}{replies}</div></main></body>"#,
                post("post bg2", &body())
            )
        };
        // Short replies after a long first post, alternating classes.
        let replies = format!(
            "{}{}",
            post("post bg1", "Thanks, the first bus helps."),
            post("post bg2", "Does route 7 still stop here?")
        );
        assert!(left_out(&thread(&replies)).is_empty());
        // Plain comment articles after the article.
        let comments = format!(
            r#"<body><main><article><time datetime="2026-01-01">Jan 1</time>{}</article><article><time datetime="2026-01-02">Jan 2</time><p>Nice post.</p></article></main></body>"#,
            body()
        );
        assert!(left_out(&comments).is_empty());
        // Without dates, a block of the same class is a call to action, and a
        // dated teaser of another page is a related post.
        let cta = format!(
            r#"<body><main><div class="mt-8">{}</div><div class="mt-8" id="cta"><p>Subscribe to our newsletter.</p></div></main></body>"#,
            body()
        );
        assert_eq!(left_out(&cta), ["cta"]);
        let teaser = format!(
            r#"<body><main><div class="topic">{}<div class="post" id="t"><h3><a href="/other">Other post</a></h3><time datetime="2026-01-02">Jan 2</time><p>A summary.</p></div></div></main></body>"#,
            post("post", &body())
        );
        assert_eq!(left_out(&teaser), ["t"]);
        // Another element, another first class, or no words of its own.
        for other in [
            r#"<section class="post" id="o"><time datetime="2026-01-02">Jan 2</time><p>Reply.</p></section>"#,
            r#"<div class="note" id="o"><time datetime="2026-01-02">Jan 2</time><p>Reply.</p></div>"#,
            r#"<div class="post" id="o"><time datetime="2026-01-02">Jan 2</time><a href="/x">Link</a></div>"#,
        ] {
            assert_eq!(left_out(&thread(other)), ["o"], "{other}");
        }
        // Blocks without a class are not posts (only plain `article`s are).
        let plain = format!(
            r#"<body><main><div><time datetime="2026-01-01">Jan 1</time>{}</div><div id="o"><time datetime="2026-01-02">Jan 2</time><p>Subscribe to our newsletter.</p></div></main></body>"#,
            body()
        );
        assert_eq!(left_out(&plain), ["o"]);
    }

    #[test]
    fn a_conclusion_or_a_list_the_body_introduces_stays() {
        let page = |intro: &str, after: &str| {
            format!(
                r#"<body><main><div class="intro">{}{intro}</div>{after}<div id="share"><a href="/s">Share</a> <a href="/t">Post</a></div></main></body>"#,
                body()
            )
        };
        let list = r#"<ul id="list"><li><a href="/a">On schedules</a></li><li><a href="/b">On maps</a></li></ul>"#;
        // A sentence ending with a colon introduces the list after it.
        assert_eq!(
            left_out(&page("<p>Here are the notes I wrote this year:</p>", list)),
            ["share"]
        );
        // A label is not a sentence; a list after other text is not introduced.
        for intro in [
            "<p><b>Related:</b></p>",
            "<p>Related posts:</p>",
            "<p>Here are my notes.</p>",
        ] {
            assert_eq!(left_out(&page(intro, list)), ["list", "share"], "{intro}");
        }
        // A line of its own after a break is a label too.
        assert_eq!(
            left_out(&page(
                "<p>The essay ends here.<br><br><b>Related:</b></p>",
                list
            )),
            ["list", "share"]
        );
        // A conclusion of plain prose right after the body stays, under a
        // heading or not ...
        for conclusion in [
            format!(
                r#"<div id="end"><h2>Conclusion</h2><p>{}</p></div>"#,
                prose(20)
            ),
            format!(r#"<p id="end">{}</p>"#, prose(16)),
        ] {
            assert_eq!(left_out(&page("", &conclusion)), ["share"], "{conclusion}");
        }
        // ... but not a short line, a bio with a link or picture, or prose
        // that does not follow the body directly.
        for (after, gone) in [
            (
                "<p id=\"end\">Thanks for reading.</p>".to_owned(),
                vec!["end", "share"],
            ),
            (
                format!(
                    r#"<div id="end"><p>{}</p><a href="/jane">Jane</a></div>"#,
                    prose(20)
                ),
                vec!["end", "share"],
            ),
            (
                format!(
                    r#"<div id="end"><img src="jane.jpg"><p>{}</p></div>"#,
                    prose(20)
                ),
                vec!["end", "share"],
            ),
        ] {
            assert_eq!(left_out(&page("", &after)), gone, "{after}");
        }
        let late = format!(
            r#"<body><main><div>{}</div><div id="share"><a href="/s">Share</a> <a href="/t">Post</a></div><p id="end">{}</p></main></body>"#,
            body(),
            prose(16)
        );
        assert_eq!(left_out(&late), ["end", "share"]);
    }

    #[test]
    fn marks_of_the_page_are_furniture_and_text_about_them_is_not() {
        let page = |marks: &str| {
            format!(
                r#"<body><article><h1>Title</h1>{marks}{}</article></body>"#,
                body()
            )
        };
        let gone = left_out(&page(
            r#"<div id="read">8 min read</div><div id="likes"><div><img src="a.png"></div><div id="count"><a>9 Likes</a></div></div>
            <div class="meta"><span>By Jane</span> <span id="views">1.2K views</span></div><p id="time">Reading time: 3 minutes</p>"#,
        ));
        assert_eq!(gone, ["count", "read", "time", "views"]);
        // A list item, a cell, a heading, a sentence and other durations stay.
        for kept in [
            "<ul><li>13 likes</li><li>4 shares</li></ul>",
            "<table><tr><td>Views</td><td>12 views</td></tr><tr><td>Likes</td><td>9 likes</td></tr></table>",
            "<h2>5 min read</h2>",
            "<div><h2>5 min read</h2></div>",
            "<p>It is an 8 min read at most.</p>",
            "<p>The phrase <span>8 min read</span> appears in headers.</p>",
            "<div>Prep time: 5 minutes</div>",
            "<div>9 Lives</div>",
        ] {
            assert!(left_out(&page(kept)).is_empty(), "{kept}");
        }
        // Without an article body nothing is a mark: an index lists read times.
        let index = r#"<body><main><div><h3><a href="/a">A</a></h3><span>5 min read</span></div><div><h3><a href="/b">B</a></h3><span>8 min read</span></div></main></body>"#;
        assert!(left_out(index).is_empty());
    }

    #[test]
    fn short_labels_are_read_as_words() {
        assert_eq!(label_words("· 8min read"), ["8", "min", "read"]);
        assert_eq!(label_words("1.2K Views"), ["1.2", "k", "views"]);
        assert_eq!(
            label_words("Reading time: 3 minutes."),
            ["reading", "time", "3", "minutes"]
        );
        let words = |text: &str| label_words(text);
        let read = |text: &str| {
            let words = words(text);
            reading_time(&words.iter().map(String::as_str).collect::<Vec<_>>())
        };
        let count = |text: &str| {
            let words = words(text);
            counter(&words.iter().map(String::as_str).collect::<Vec<_>>())
        };
        for yes in [
            "8 min read",
            "5-minute read",
            "12 mins read",
            "Read time: 4 min",
        ] {
            assert!(read(yes), "{yes}");
        }
        for no in [
            "8 min",
            "read 8 min",
            "1000 min read",
            "3 pages read",
            "Prep time: 5 minutes",
        ] {
            assert!(!read(no), "{no}");
        }
        for yes in [
            "9 Likes",
            "1 comment",
            "3,456 shares",
            "12k views",
            "1.2M Views",
        ] {
            assert!(count(yes), "{yes}");
        }
        for no in [
            "Likes",
            "9 lives",
            "nine likes",
            "9 likes today",
            "v2 likes",
        ] {
            assert!(!count(no), "{no}");
        }
    }

    #[test]
    fn a_breadcrumb_trail_above_the_text_is_furniture() {
        let trail = r#"<ul id="trail"><li><a href="/">Home</a></li><li><a href="/archive">Posts</a></li><li>I took a job in an unexpected industry</li></ul>"#;
        let page = |before: &str, after: &str| {
            format!(r#"<body><main>{before}{}{after}</main></body>"#, body())
        };
        assert_eq!(
            left_out(&page(
                &format!(r#"<div data-block="nav">{trail}</div>"#),
                ""
            )),
            ["trail"]
        );
        for kept in [
            // Every step a link, a step with words of its own, a long last step.
            r#"<ul><li><a href="/">Home</a></li><li><a href="/p">Posts</a></li><li><a href="/p/x">This</a></li></ul>"#,
            r#"<ul><li><a href="/">Home</a></li><li><a href="/p">Posts</a></li><li><a href="/p/x">This</a> page</li></ul>"#,
            r#"<ul><li><a href="/">Home</a> and more</li><li><a href="/p">Posts</a></li><li>This</li></ul>"#,
            r#"<ul><li><a href="/">Home</a></li><li><a href="/p">Posts</a></li><li>A step that goes on for far too many words to be the name of a page</li></ul>"#,
            // One link only.
            r#"<ul><li><a href="/">Home</a></li><li>This</li></ul>"#,
        ] {
            assert!(left_out(&page(kept, "")).is_empty(), "{kept}");
        }
        // After the first paragraph a list of steps is the article's.
        assert!(left_out(&page("", trail)).is_empty());
    }

    #[test]
    fn a_date_repeated_around_the_title_is_written_once() {
        let page = |lede: &str| {
            format!(
                r#"<body><article><time datetime="2025-01-15">January 15, 2025</time><h1>Title</h1>{lede}{}</article></body>"#,
                body()
            )
        };
        assert_eq!(
            left_out(&page(
                r#"<p id="again"><i><time datetime="2025-01-15T09:00">15 Jan, 2025</time></i></p>"#
            )),
            ["again"]
        );
        for kept in [
            // Another date, a date in a byline, a date after the first paragraph.
            r#"<p><time datetime="2025-01-16">Updated January 16, 2025</time></p>"#,
            r#"<p>By Jane, <time datetime="2025-01-15">15 Jan</time></p>"#,
        ] {
            assert!(left_out(&page(kept)).is_empty(), "{kept}");
        }
        let late = format!(
            r#"<body><article><time datetime="2025-01-15">January 15, 2025</time><h1>Title</h1>{}<p><time datetime="2025-01-15">15 Jan</time></p></article></body>"#,
            body()
        );
        assert!(left_out(&late).is_empty());
    }

    #[test]
    fn an_authors_card_beside_the_body_is_furniture() {
        let bio = "I write about distributed systems, storage and performance, and I like to share what I learn.";
        let page = |front: &str| {
            format!(
                r#"<body><main><h1>Title</h1><section><p>A short abstract.</p><p>07 April 2026</p>{front}</section><section>{}</section></main></body>"#,
                body()
            )
        };
        let card = format!(
            r#"<div class="grid"><div id="card"><div><a href="/jane"><img src="jane.jpg" alt="Jane"></a></div><address><a href="/jane" rel="author">Jane Doe</a></address><div><p>{bio}</p></div></div></div>"#
        );
        assert_eq!(left_out(&page(&card)), ["card"]);
        for kept in [
            // A byline (with an avatar or not), a dek beside a byline, a card
            // with a heading or a date.
            r#"<p>By <a href="/jane" rel="author">Jane Doe</a></p>"#.to_owned(),
            r#"<div><img src="jane.jpg"><a href="/jane" rel="author">Jane Doe</a></div>"#
                .to_owned(),
            format!(
                r#"<div><p>{bio}</p><p>By <a href="/jane" rel="author">Jane Doe</a></p></div>"#
            ),
            format!(
                r#"<div><h3>About Jane</h3><img src="jane.jpg"><a href="/jane" rel="author">Jane Doe</a><p>{bio}</p></div>"#
            ),
            format!(
                r#"<div><img src="jane.jpg"><a href="/jane" rel="author">Jane Doe</a> <time datetime="2026-04-07">Apr 7</time><p>{bio}</p></div>"#
            ),
            // A lead picture with a long caption names no author.
            format!(r#"<div><img src="hero.jpg"><p>{bio}</p></div>"#),
        ] {
            assert!(left_out(&page(&kept)).is_empty(), "{kept}");
        }
        // Inside the body a card is the article's.
        let inside = format!(
            r#"<body><main><h1>Title</h1><div><p>{}</p>{card}<p>{}</p><p>{}</p></div></main></body>"#,
            prose(40),
            prose(40),
            prose(40)
        );
        assert!(left_out(&inside).is_empty());
    }

    #[test]
    fn a_featured_comment_set_into_the_body_is_furniture() {
        let comment = |class: &str| {
            format!(
                r##"<div class="{class}"><h2 class="{class}__heading">Top comment by</h2><span>Liked by 28 people</span><p>I can't believe they kept it secret!</p><a href="#comments">View all comments</a></div>"##
            )
        };
        let page = |inner: &str| {
            format!(
                r#"<body><article><h1>Title</h1><p>{}</p>{inner}<p>{}</p><p>{}</p></article></body>"#,
                prose(40),
                prose(40),
                prose(40)
            )
        };
        for class in [
            "top-comment",
            "hotComment",
            "featured_comments",
            "pinned-comment",
            "featured-comments__list",
        ] {
            assert_eq!(left_out(&page(&comment(class))), ["div"], "{class}");
        }
        // An ordinary comment block, and a featured one opening or ending the
        // article.
        assert!(left_out(&page(&comment("comment"))).is_empty());
        for (before, after) in [(String::new(), body()), (body(), String::new())] {
            let edge = format!(
                r#"<body><article><h1>Title</h1>{before}{}{after}</article></body>"#,
                comment("top-comment")
            );
            assert!(left_out(&edge).is_empty(), "{edge}");
        }
        // A page of featured comments is about them.
        let pick = format!(r#"<div class="top-comment"><p>{}</p></div>"#, prose(40));
        let picks = format!(
            r#"<body><main><h1>Top comments</h1><p>Our editors picked these.</p>{}</main></body>"#,
            pick.repeat(3)
        );
        assert!(left_out(&picks).is_empty());
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
