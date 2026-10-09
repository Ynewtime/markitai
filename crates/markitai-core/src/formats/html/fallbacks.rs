//! Images a page shows only once its scripts run. A lazy-loading page leaves
//! an empty box or an inline placeholder where the image goes and puts the
//! real `<img>` in a `<noscript>` fallback; a `<picture>` may carry its
//! addresses only on its `<source>` sets, with an inline placeholder as its
//! `<img>`. A reading without scripts takes the fallback, as the reference
//! does, instead of dropping the image or linking the placeholder.

use super::{Attribute, escaped, is_hidden, srcset};
use scraper::{ElementRef, Html, Node};

/// Fallbacks recovered per page; a page with more is read as it is.
const MAX_FALLBACKS: usize = 1_000;

fn placeholder(url: &str) -> bool {
    url.trim()
        .get(..5)
        .is_some_and(|scheme| scheme.eq_ignore_ascii_case("data:"))
}

/// The address an `<img>` shows, unless it is none or an inline placeholder.
fn shown(element: &scraper::node::Element) -> Option<&str> {
    srcset::image(element).filter(|url| !placeholder(url))
}

/// A tracking pixel or a hidden image is no content.
fn content_image(image: ElementRef<'_>) -> bool {
    let tiny = |name: &str| {
        image
            .value()
            .attribute(name)
            .is_some_and(|value| value.trim().parse::<f64>().is_ok_and(|v| v <= 1.0))
    };
    shown(image.value()).is_some() && !is_hidden(image) && !tiny("width") && !tiny("height")
}

/// The images a `<noscript>` holds when it holds nothing else: no text, and
/// only `img`, `picture` and `source` elements.
fn fallback_images(noscript: ElementRef<'_>) -> Vec<scraper::node::Element> {
    // Parsed with scripting on, the fallback is the element's raw text;
    // otherwise it is already a tree.
    let markup = if noscript.children().all(|child| child.value().is_text()) {
        noscript.text().collect::<String>()
    } else {
        noscript.inner_html()
    };
    if !markup.to_ascii_lowercase().contains("<img") {
        return Vec::new();
    }
    let fragment = Html::parse_fragment(&markup);
    let root = fragment.root_element();
    if root.text().any(|text| !text.trim().is_empty()) {
        return Vec::new();
    }
    let mut images = Vec::new();
    for element in root.descendants().skip(1).filter_map(ElementRef::wrap) {
        match element.value().name() {
            "img" if content_image(element) => images.push(element.value().clone()),
            "img" | "picture" | "source" => {}
            _ => return Vec::new(),
        }
    }
    images
}

/// Put each `<noscript>` image fallback in place of its `<noscript>` where the
/// box around it shows no image of its own, dropping the inline placeholder
/// images beside it; and give a `<picture>` whose `<img>` shows only an inline
/// placeholder the largest candidate of its first `<source>` set.
pub(super) fn recover(document: &mut Html) {
    let mut fallbacks = Vec::new();
    let mut pictures = Vec::new();
    for element in document
        .root_element()
        .descendants()
        .filter_map(ElementRef::wrap)
    {
        if fallbacks.len() + pictures.len() >= MAX_FALLBACKS {
            break;
        }
        match element.value().name() {
            "noscript" => {
                let in_body = element
                    .ancestors()
                    .filter_map(ElementRef::wrap)
                    .any(|ancestor| ancestor.value().name() == "body");
                let Some(parent) = element.parent().and_then(ElementRef::wrap) else {
                    continue;
                };
                // A box that already shows an image (a lazy `data-src` the
                // reader follows) needs no fallback.
                let shows_image = parent
                    .descendants()
                    .filter_map(ElementRef::wrap)
                    .any(|image| image.value().name() == "img" && shown(image.value()).is_some());
                if !in_body || is_hidden(element) || shows_image {
                    continue;
                }
                let images = fallback_images(element);
                if images.is_empty() {
                    continue;
                }
                let placeholders: Vec<_> = parent
                    .children()
                    .filter_map(ElementRef::wrap)
                    .filter(|sibling| sibling.value().name() == "img")
                    .map(|sibling| sibling.id())
                    .collect();
                fallbacks.push((element.id(), images, placeholders));
            }
            "img" if shown(element.value()).is_none() => {
                let Some(picture) = element
                    .parent()
                    .and_then(ElementRef::wrap)
                    .filter(|parent| parent.value().name() == "picture")
                else {
                    continue;
                };
                let source = picture
                    .children()
                    .filter_map(ElementRef::wrap)
                    .filter(|child| child.value().name() == "source")
                    .find_map(|source| {
                        ["data-srcset", "srcset"]
                            .iter()
                            .filter_map(|name| source.value().attribute(name))
                            .find_map(srcset::best)
                            .filter(|url| !placeholder(url))
                    });
                if let Some(source) = source {
                    // The source set stands in for the placeholder `src`.
                    let mut markup = String::from("<img src=\"");
                    escaped(source, &mut markup);
                    for name in ["alt", "title"] {
                        if let Some(value) = element.value().attribute(name) {
                            markup.push_str(&format!("\" {name}=\""));
                            escaped(value, &mut markup);
                        }
                    }
                    markup.push_str("\">");
                    let fragment = Html::parse_fragment(&markup);
                    let image = fragment
                        .root_element()
                        .descendants()
                        .filter_map(ElementRef::wrap)
                        .find(|image| image.value().name() == "img")
                        .map(|image| image.value().clone());
                    if let Some(image) = image {
                        pictures.push((element.id(), image));
                    }
                }
            }
            _ => {}
        }
    }
    for (noscript, images, placeholders) in fallbacks {
        let Some(mut node) = document.tree.get_mut(noscript) else {
            continue;
        };
        for image in images {
            node.insert_before(Node::Element(image));
        }
        node.detach();
        for placeholder in placeholders {
            if let Some(mut node) = document.tree.get_mut(placeholder) {
                node.detach();
            }
        }
    }
    for (placeholder, image) in pictures {
        if let Some(mut node) = document.tree.get_mut(placeholder) {
            node.insert_before(Node::Element(image));
            node.detach();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn recovered(body: &str) -> String {
        let mut document = Html::parse_document(&format!("<html><body>{body}</body></html>"));
        recover(&mut document);
        super::super::render_clean(document.root_element(), None).unwrap()
    }

    #[test]
    fn a_noscript_image_stands_for_its_lazy_box() {
        let markdown = recovered(
            r#"<figure><div class="lazy"></div><noscript><img src="/g/1.jpg" alt="First"></noscript>
            <figcaption>First caption</figcaption></figure>
            <p><img src="data:image/gif;base64,R0lGOD" alt="Second"><noscript><img src="/g/2.jpg" alt="Second"></noscript></p>"#,
        );
        assert!(markdown.contains("![First](/g/1.jpg)"), "{markdown}");
        assert!(markdown.contains("![Second](/g/2.jpg)"), "{markdown}");
        // The placeholder beside the fallback is gone.
        assert!(!markdown.contains("data:"), "{markdown}");
    }

    #[test]
    fn a_shown_image_text_or_pixel_keeps_its_noscript_out() {
        let markdown = recovered(
            r#"<p><img src="data:x" data-src="/lazy.jpg" alt="Lazy"><noscript><img src="/lazy.jpg" alt="Again"></noscript></p>
            <div><noscript>Please enable JavaScript <img src="/js.png" alt="Js"></noscript></div>
            <div><noscript><img src="/tr?id=1" width="1" height="1" alt="Pixel"></noscript></div>
            <div><noscript><iframe src="/frame"></iframe><img src="/f.png" alt="Frame"></noscript></div>"#,
        );
        assert_eq!(markdown.matches("![").count(), 1, "{markdown}");
        assert!(markdown.contains("![Lazy](/lazy.jpg)"), "{markdown}");
    }

    #[test]
    fn a_picture_placeholder_takes_its_first_source_set() {
        let markdown = recovered(
            r#"<picture><source type="image/webp" srcset="/h.webp 2x, /h-small.webp 1x "><source srcset="/h.png 2x">
            <img src="data:image/gif;base64,R0lGOD" alt="Hero"></picture>
            <picture><source srcset="/other.webp 2x"><img src="/real.jpg" alt="Real"></picture>"#,
        );
        assert!(markdown.contains("![Hero](/h.webp)"), "{markdown}");
        assert!(markdown.contains("![Real](/real.jpg)"), "{markdown}");
    }
}
