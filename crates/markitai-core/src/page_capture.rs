//! What loading a URL's page left for showing it beside its Markdown: the
//! page's HTML, the browser's screenshots of it and where its blocks were
//! drawn, and how usable each view was when it was captured (content, a
//! blank page, or a verification, captcha or login page in its place).
use serde::{Deserialize, Serialize};
use url::Url;

/// Characters of visible text counted at most.
pub const MAX_TEXT_LENGTH: u64 = 1_000_000;
/// Characters of a title kept at most.
const MAX_TITLE: usize = 200;
/// Visible text shorter than this, in characters, is a blank page unless
/// the page shows a picture or a media player (a picture post's caption).
const BLANK_BELOW: u64 = 80;
/// Bytes of block boxes kept at most.
pub const MAX_BLOCKS_BYTES: usize = 4 * 1024 * 1024;

/// What a view of the page showed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CaptureState {
    /// The page's content.
    Ok,
    /// A blank page, or one that only asks for JavaScript.
    Empty,
    /// A verification, captcha, bot check or login page instead of the content.
    Challenge,
}

/// How usable a kept page or its screenshot was when it was captured.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CaptureQuality {
    pub state: CaptureState,
    /// Unicode scalar values of the visible text, white space runs counted
    /// as one and the ends trimmed, at most [`MAX_TEXT_LENGTH`].
    pub text_length: u64,
    /// The document's title, white space collapsed, at most 200 characters.
    pub title: Option<String>,
    /// Why it is not `ok`: `challenge`, `verification`, `login`,
    /// `javascript` or `blank`.
    pub reason: Option<String>,
}

/// What a URL conversion's page load left, filled by
/// [`crate::convert_with_page_capture`]. Everything is empty for a file, a
/// PDF, a remote service's Markdown or a page-cache hit.
#[derive(Debug, Default)]
pub struct PageCapture {
    /// The HTML the page content was converted from, or the DOM of the
    /// browser that took the page's screenshots when no other HTML was kept
    /// (at most 10 MiB).
    pub html: Option<String>,
    /// How [`Self::html`] reads, as a frame without scripts shows it.
    pub page: Option<CaptureQuality>,
    /// How the page read in the browser when its screenshots were taken;
    /// `None` when the browser took none.
    pub screenshot: Option<CaptureQuality>,
    /// The page cut into JPEG tiles from the top (quality 70, at most 20
    /// tiles), taken because the browser runtime keeps compare screenshots
    /// and the conversion asked for no screenshots of its own. They are not
    /// conversion output and are never sent to a model or read by OCR.
    pub compare_shots: Vec<Vec<u8>>,
    /// The page's blocks where the screenshots (the conversion's own, else
    /// the compare screenshots) show them: `{version:1,width,height,blocks}`
    /// JSON, at most 4 MiB, with the CSS size the screenshots cover.
    pub blocks: Option<Vec<u8>>,
}

/// Whitespace runs collapsed to one space, the ends trimmed.
fn collapsed(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Unicode scalar values of collapsed text, at most [`MAX_TEXT_LENGTH`].
fn length(text: &str) -> u64 {
    (collapsed(text).chars().count() as u64).min(MAX_TEXT_LENGTH)
}

/// A title as reported: collapsed, at most 200 characters, `None` when empty.
fn shown_title(title: &str) -> Option<String> {
    let title: String = collapsed(title).chars().take(MAX_TITLE).collect();
    (!title.is_empty()).then_some(title)
}

/// Judge a view: `markup` is the HTML it showed (the kept page, or the
/// browser's DOM), read from `url`; `text_length` and `title` are what it
/// showed a reader.
pub(crate) fn classify(
    url: Option<&Url>,
    markup: &str,
    text_length: u64,
    title: &str,
) -> CaptureQuality {
    static BLANK: std::sync::LazyLock<Url> =
        std::sync::LazyLock::new(|| Url::parse("about:blank").expect("a valid URL"));
    let refusal = crate::fetch::shown_instead(url.unwrap_or(&BLANK), markup);
    let text_length = text_length.min(MAX_TEXT_LENGTH);
    let (state, reason) = match refusal {
        Some(reason) if reason != "javascript" => (CaptureState::Challenge, Some(reason)),
        Some(reason) => (CaptureState::Empty, Some(reason)),
        None if text_length < BLANK_BELOW && !shows_media(markup) => {
            (CaptureState::Empty, Some("blank"))
        }
        None => (CaptureState::Ok, None),
    };
    CaptureQuality {
        state,
        text_length,
        title: shown_title(title),
        reason: reason.map(str::to_owned),
    }
}

/// How a kept page reads in a frame that runs no script: its text outside
/// `head`, `script`, `style` and `template`, `<noscript>` text included,
/// and its first `<title>`. `url` is the address the page was read from.
pub fn page_quality(url: &str, html: &str) -> CaptureQuality {
    let (text, title) = framed_text(html);
    let url = Url::parse(url).ok();
    classify(url.as_ref(), html, length(&text), &title)
}

/// The visible text and the title of markup shown without scripts.
fn framed_text(html: &str) -> (String, String) {
    let tree = crate::formats::parse_html(html);
    let mut text = String::new();
    let mut title = None;
    collect(&tree, &mut text, &mut title, true);
    (text, title.unwrap_or_default())
}

/// Text of a parsed tree into `text` (its first title into `title`). A
/// parser that runs scripts keeps `<noscript>` content as raw text; a frame
/// without scripts shows it as markup, so it is parsed again once.
fn collect(tree: &scraper::Html, text: &mut String, title: &mut Option<String>, nested: bool) {
    for node in tree.tree.nodes() {
        if let Some(element) = scraper::ElementRef::wrap(node) {
            if title.is_none() && element.value().name() == "title" {
                *title = Some(element.text().collect());
            }
            continue;
        }
        let scraper::Node::Text(value) = node.value() else {
            continue;
        };
        let mut hidden = false;
        let mut noscript = false;
        for element in node.ancestors().filter_map(scraper::ElementRef::wrap) {
            match element.value().name() {
                "head" | "script" | "style" | "template" | "title" => hidden = true,
                "noscript" => noscript = true,
                _ => {}
            }
        }
        if hidden {
            continue;
        }
        if noscript && nested {
            let fragment = crate::formats::parse_html_fragment(value);
            collect(&fragment, text, &mut Some(String::new()), false);
        } else {
            text.push_str(value);
        }
        if text.len() > 8 * MAX_TEXT_LENGTH as usize {
            return;
        }
    }
}

/// Whether markup shows a picture or a media player: an `<img>` with a
/// `src` or `srcset` that is not a `data:` placeholder, a `<video>`, or an
/// `<audio controls>`, outside `head`, `template` and `noscript`, and not
/// hidden (a `hidden` attribute or an inline `display: none` or
/// `visibility: hidden` on it or an ancestor, a `width` or `height` of 0 or
/// 1 on it, as tracking pixels have).
fn shows_media(markup: &str) -> bool {
    let tree = crate::formats::parse_html(markup);
    tree.tree
        .nodes()
        .filter_map(scraper::ElementRef::wrap)
        .any(|element| {
            let value = element.value();
            let media = match value.name() {
                "img" => ["src", "srcset"].into_iter().any(|name| {
                    value.attr(name).is_some_and(|address| {
                        let address = address.trim();
                        !address.is_empty()
                            && !address
                                .get(..5)
                                .is_some_and(|scheme| scheme.eq_ignore_ascii_case("data:"))
                    })
                }),
                "video" => true,
                "audio" => value.attr("controls").is_some(),
                _ => false,
            };
            let pixel = ["width", "height"].into_iter().any(|name| {
                value
                    .attr(name)
                    .is_some_and(|size| matches!(size.trim(), "0" | "1"))
            });
            media
                && !pixel
                && !hidden(&element)
                && element
                    .ancestors()
                    .filter_map(scraper::ElementRef::wrap)
                    .all(|parent| {
                        !matches!(parent.value().name(), "head" | "template" | "noscript")
                            && !hidden(&parent)
                    })
        })
}

/// A `hidden` attribute, or an inline style that hides the element.
fn hidden(element: &scraper::ElementRef) -> bool {
    let value = element.value();
    value.attr("hidden").is_some()
        || value.attr("style").is_some_and(|style| {
            let style: String = style
                .chars()
                .filter(|character| !character.is_whitespace())
                .collect::<String>()
                .to_ascii_lowercase();
            style.contains("display:none") || style.contains("visibility:hidden")
        })
}

/// `page_blocks.js`'s result for screenshots that cover `width` × `height`
/// CSS pixels from the page's top: that size in place of the page's own, no
/// block that starts below it, serialized; `None` when it is not the
/// script's shape or is larger than 4 MiB.
pub(crate) fn sized_blocks(
    mut value: serde_json::Value,
    width: u64,
    height: u64,
) -> Option<Vec<u8>> {
    let object = value.as_object_mut()?;
    if object.get("version").and_then(serde_json::Value::as_u64) != Some(1) {
        return None;
    }
    let blocks = object.get_mut("blocks")?.as_array_mut()?;
    blocks.retain(|block| {
        block
            .pointer("/bbox/1")
            .and_then(serde_json::Value::as_f64)
            .is_some_and(|top| top < height as f64)
    });
    object.insert("width".into(), width.into());
    object.insert("height".into(), height.into());
    let bytes = serde_json::to_vec(&value).ok()?;
    (bytes.len() <= MAX_BLOCKS_BYTES).then_some(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn article(words: usize) -> String {
        format!(
            "<html><head><title>  An   invented\n article </title><style>p{{}}</style></head><body><article><h1>Heading</h1><p>{}</p></article><script>var hidden = 'not text';</script></body></html>",
            "Plain words of an invented article. ".repeat(words)
        )
    }

    #[test]
    fn an_article_is_ok_with_its_collapsed_text_and_title() {
        let quality = page_quality("https://example.com/post", &article(4));
        let expected = format!(
            "Heading{}",
            "Plain words of an invented article. ".repeat(4).trim_end()
        );
        assert_eq!(
            quality,
            CaptureQuality {
                state: CaptureState::Ok,
                text_length: expected.chars().count() as u64,
                title: Some("An invented article".into()),
                reason: None,
            }
        );
        assert_eq!(
            serde_json::to_value(&quality).unwrap()["state"],
            json!("ok")
        );
    }

    #[test]
    fn a_challenge_widget_a_site_captcha_and_a_login_wall_are_challenges() {
        let cloudflare = "<html><head><title>Just a moment...</title></head><body><div id=\"cf-chl-widget\"></div><noscript>Enable JavaScript and cookies to continue</noscript><script src=\"/cdn-cgi/challenge-platform/h/b/orchestrate/chl_page/v1\"></script></body></html>";
        let quality = page_quality("https://example.org/a", cloudflare);
        assert_eq!(quality.state, CaptureState::Challenge);
        assert_eq!(quality.reason.as_deref(), Some("challenge"));
        assert_eq!(quality.title.as_deref(), Some("Just a moment..."));
        // The frame shows the noscript text.
        assert_eq!(
            quality.text_length,
            "Enable JavaScript and cookies to continue".len() as u64
        );
        let captcha = "<html><head><title>Video</title></head><body><div id=\"risk-captcha-app\"></div></body></html>";
        let quality = page_quality("https://www.bilibili.com/video/BV0000000000", captcha);
        assert_eq!(
            (quality.state, quality.reason.as_deref()),
            (CaptureState::Challenge, Some("verification"))
        );
        // The same markup elsewhere is only a blank page.
        let quality = page_quality("https://example.com/video", captcha);
        assert_eq!(
            (quality.state, quality.reason.as_deref()),
            (CaptureState::Empty, Some("blank"))
        );
        let login = "<html><head><title>Example</title></head><body><p>Log in to continue reading this answer.</p></body></html>";
        let quality = page_quality("https://www.zhihu.com/question/1", login);
        assert_eq!(
            (quality.state, quality.reason.as_deref()),
            (CaptureState::Challenge, Some("login"))
        );
    }

    #[test]
    fn a_javascript_notice_and_an_empty_body_are_empty() {
        let notice = "<html><head><title>Example</title></head><body><div><p>JavaScript is not available.</p><p>We've detected that JavaScript is disabled in this browser.</p></div></body></html>";
        let quality = page_quality("https://example.com/status/1", notice);
        assert_eq!(
            (quality.state, quality.reason.as_deref()),
            (CaptureState::Empty, Some("javascript"))
        );
        for empty in [
            "",
            "<html><head><title> </title></head><body>\n \t</body></html>",
        ] {
            let quality = page_quality("https://example.com/", empty);
            assert_eq!(
                quality,
                CaptureQuality {
                    state: CaptureState::Empty,
                    text_length: 0,
                    title: None,
                    reason: Some("blank".into()),
                },
                "{empty:?}"
            );
        }
        // 79 characters is blank, 80 is content.
        let text = |count: usize| format!("<body><p>{}</p></body>", "x".repeat(count));
        assert_eq!(
            page_quality("https://example.com/", &text(79)).state,
            CaptureState::Empty
        );
        assert_eq!(
            page_quality("https://example.com/", &text(80)).state,
            CaptureState::Ok
        );
    }

    #[test]
    fn a_short_caption_under_a_picture_or_a_player_is_ok() {
        let page = |body: &str| {
            page_quality(
                "https://example.com/p/1",
                &format!(
                    "<html><head><title>Post</title><link rel=\"icon\" href=\"/i.png\"></head><body>{body}</body></html>"
                ),
            )
        };
        for shown in [
            "<article><img src=\"https://pictures.example.com/river.jpg\" alt=\"A river\"><p>Autumn light on the river.</p></article>",
            "<picture><source srcset=\"/river.avif\"><img srcset=\"/river.jpg 1x\"></picture><p>Autumn light.</p>",
            "<video controls src=\"https://media.example.com/clip.mp4\"></video><p>A short clip.</p>",
            "<video><source src=\"/clip.webm\"></video>",
            "<audio controls src=\"/song.ogg\"></audio><p>Track 1</p>",
        ] {
            let quality = page(shown);
            assert_eq!(
                (quality.state, quality.reason),
                (CaptureState::Ok, None),
                "{shown}"
            );
        }
        // Nothing a reader would see: still blank.
        for unseen in [
            "<div id=\"app\"></div><noscript><img height=\"1\" width=\"1\" style=\"display:none\" src=\"https://pixel.example.com/t\"></noscript>",
            "<img src=\"https://pixel.example.com/t\" width=\"1\" height=\"1\" alt=\"\"><div id=\"root\"></div>",
            "<img src=\"data:image/gif;base64,R0lGODlhAQABAAAAACw=\" data-src=\"/river.jpg\">",
            "<img src=\"  \"><img alt=\"No address\">",
            "<img hidden src=\"/river.jpg\"><div style=\"Display : None\"><img src=\"/river.jpg\"></div>",
            "<div style=\"visibility:hidden\"><video src=\"/clip.mp4\"></video></div>",
            "<audio src=\"/chime.ogg\" autoplay></audio>",
            "<template><img src=\"/river.jpg\"></template><svg viewBox=\"0 0 10 10\"><circle r=\"4\"/></svg>",
        ] {
            let quality = page(unseen);
            assert_eq!(
                (quality.state, quality.reason.as_deref()),
                (CaptureState::Empty, Some("blank")),
                "{unseen}"
            );
        }
        // Only the blank rule looks at pictures: a JavaScript notice stays one.
        let quality = page("<img src=\"/logo.png\"><p>Please enable JavaScript to continue.</p>");
        assert_eq!(
            (quality.state, quality.reason.as_deref()),
            (CaptureState::Empty, Some("javascript"))
        );
    }

    #[test]
    fn text_is_counted_in_characters_and_capped_and_titles_are_bounded() {
        let wide = format!("<body><p>{}</p></body>", "字  \n ".repeat(100));
        // "字 " a hundred times, trimmed: 199 characters.
        assert_eq!(page_quality("https://example.com/", &wide).text_length, 199);
        let long = format!(
            "<head><title>{}</title></head><body><p>{}</p></body>",
            "t".repeat(300),
            "word ".repeat(300_000)
        );
        let quality = page_quality("https://example.com/", &long);
        assert_eq!(quality.text_length, MAX_TEXT_LENGTH);
        assert_eq!(quality.title.unwrap().chars().count(), 200);
        // The first title counts; template and script text never does.
        let quality = page_quality(
            "https://example.com/",
            "<title>First</title><svg><title>Second</title></svg><template><p>hidden</p></template>",
        );
        assert_eq!(
            (quality.title.as_deref(), quality.text_length),
            (Some("First"), 0)
        );
        assert_eq!(
            classify(None, "", 5_000_000, "").text_length,
            MAX_TEXT_LENGTH
        );
    }

    #[test]
    fn blocks_take_the_screenshots_size_and_lose_what_is_below_it() {
        let value = json!({"version":1,"width":1300,"height":9000,"blocks":[
            {"tag":"p","text":"top","bbox":[0,10,100,30]},
            {"tag":"p","text":"cut","bbox":[0,4000,100,4020]},
            {"tag":"img","text":"","src":"https://example.com/a.png","bbox":[0,3990,100,4100]}]});
        let sized: serde_json::Value =
            serde_json::from_slice(&sized_blocks(value, 1280, 4000).unwrap()).unwrap();
        assert_eq!(
            (sized["width"].clone(), sized["height"].clone()),
            (json!(1280), json!(4000))
        );
        let texts: Vec<_> = sized["blocks"]
            .as_array()
            .unwrap()
            .iter()
            .map(|block| block["tag"].as_str().unwrap())
            .collect();
        assert_eq!(texts, ["p", "img"]);
        assert!(sized_blocks(json!({"version":2,"blocks":[]}), 1, 1).is_none());
        assert!(sized_blocks(json!(null), 1, 1).is_none());
        let huge = json!({"version":1,"blocks":(0..5000).map(|_| json!({"tag":"p","text":"x".repeat(1000),"bbox":[0,0,1,1]})).collect::<Vec<_>>()});
        assert!(sized_blocks(huge, 1, 1).is_none());
    }
}
