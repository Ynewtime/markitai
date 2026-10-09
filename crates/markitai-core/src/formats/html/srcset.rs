//! The image an `<img>` stands for: its largest `srcset` candidate where that
//! is no downgrade, else its lazy-loading or plain `src`. A page's `src` is
//! often the smallest size of the set (a thumbnail kept for old browsers), and
//! the Markdown links one address.

use super::Attribute;

/// One candidate of a `srcset` attribute: an address and the size it names.
#[derive(Debug, PartialEq)]
struct Candidate<'a> {
    url: &'a str,
    /// The `w` descriptor, in pixels.
    width: Option<f64>,
    /// The `x` descriptor; `1` when the candidate has no descriptor.
    density: f64,
}

/// The candidates of a `srcset` value, read as the HTML standard reads it: a
/// candidate's address runs to the next space, so a comma inside it (a CDN's
/// `w_728,c_limit`) is part of it, and only a comma after the address (or after
/// a descriptor) ends the candidate. A candidate with a descriptor this reader
/// does not know (`future(a,b)`, a height without a width) is dropped.
fn candidates(value: &str) -> Vec<Candidate<'_>> {
    let bytes = value.as_bytes();
    let space = |byte: u8| matches!(byte, b' ' | b'\t' | b'\n' | b'\r' | 0x0c);
    let mut found = Vec::new();
    let mut at = 0;
    loop {
        while at < bytes.len() && (space(bytes[at]) || bytes[at] == b',') {
            at += 1;
        }
        if at >= bytes.len() {
            return found;
        }
        let start = at;
        while at < bytes.len() && !space(bytes[at]) {
            at += 1;
        }
        let mut url = &value[start..at];
        let mut descriptors = Vec::new();
        if url.ends_with(',') {
            // The commas end the candidate; it has no descriptor.
            url = url.trim_end_matches(',');
        } else {
            // Descriptors, up to a comma outside parentheses.
            let mut token = None::<usize>;
            let mut parentheses = false;
            while at < bytes.len() {
                let byte = bytes[at];
                if !parentheses && (space(byte) || byte == b',') {
                    if let Some(from) = token.take() {
                        descriptors.push(&value[from..at]);
                    }
                    at += 1;
                    if byte == b',' {
                        break;
                    }
                    continue;
                }
                match byte {
                    b'(' => parentheses = true,
                    b')' => parentheses = false,
                    _ => {}
                }
                token.get_or_insert(at);
                at += 1;
            }
            if let Some(from) = token {
                descriptors.push(&value[from..at.min(bytes.len())]);
            }
        }
        if url.is_empty() {
            continue;
        }
        let mut width = None;
        let mut density = None;
        let mut known = true;
        for descriptor in descriptors {
            let Some((cut, kind)) = descriptor.char_indices().next_back() else {
                known = false;
                continue;
            };
            let number = &descriptor[..cut];
            let digits = !number.is_empty() && number.bytes().all(|byte| byte.is_ascii_digit());
            match kind {
                'w' if digits && width.is_none() => width = number.parse::<f64>().ok(),
                'h' if digits => {}
                'x' if density.is_none() => {
                    density = positive_ratio(number);
                    known &= density.is_some();
                }
                _ => known = false,
            }
        }
        // A width and a density together are not a valid candidate.
        known &= !(width.is_some() && density.is_some());
        known &= width.is_none_or(|pixels| pixels > 0.0);
        if known {
            found.push(Candidate {
                url,
                width,
                density: density.unwrap_or(1.0),
            });
        }
    }
}

/// A density descriptor's number: digits with at most one dot, above zero.
fn positive_ratio(number: &str) -> Option<f64> {
    let valid = number
        .bytes()
        .all(|byte| byte.is_ascii_digit() || byte == b'.')
        && number.bytes().filter(|byte| *byte == b'.').count() <= 1
        && number.bytes().any(|byte| byte.is_ascii_digit());
    valid
        .then(|| number.parse::<f64>().ok())
        .flatten()
        .filter(|ratio| *ratio > 0.0)
}

/// The largest candidate: the widest `w` descriptor, else the densest `x` one
/// (a candidate without a descriptor counts as `1x`); the first of equals.
fn largest<'a, 'b>(all: &'b [Candidate<'a>]) -> Option<&'b Candidate<'a>> {
    let mut widest: Option<&Candidate<'_>> = None;
    let mut densest: Option<&Candidate<'_>> = None;
    for candidate in all {
        if let Some(width) = candidate.width {
            if widest.is_none_or(|best| width > best.width.unwrap_or(0.0)) {
                widest = Some(candidate);
            }
        } else if densest.is_none_or(|best| candidate.density > best.density) {
            densest = Some(candidate);
        }
    }
    widest.or(densest)
}

/// The address of the largest candidate of a `srcset` value.
pub(super) fn best(value: &str) -> Option<&str> {
    largest(&candidates(value)).map(|candidate| candidate.url)
}

fn placeholder(url: &str) -> bool {
    url.trim()
        .get(..5)
        .is_some_and(|scheme| scheme.eq_ignore_ascii_case("data:"))
}

/// A `width` attribute in pixels (a percentage names no size).
fn pixels(value: &str) -> Option<f64> {
    let value = value.trim();
    let number = value.strip_suffix("px").unwrap_or(value).trim();
    number
        .parse::<f64>()
        .ok()
        .filter(|pixels| *pixels > 0.0 && pixels.is_finite())
}

/// Whether the largest candidate may stand for the image in place of its plain
/// address (`src`, or the lazy `data-src`); an image is never downgraded:
///
/// - with no plain address, or an inline placeholder, the candidate stands;
/// - a plain address that is itself a candidate is one size of the set (a
///   thumbnail kept as `src`), so the largest candidate stands;
/// - otherwise the plain address is a size the set does not list. Where the set
///   has widths, the candidate stands only when it is at least the image's
///   `width` attribute, which says how wide the plain address is (BBC's `src`
///   is 2560 pixels wide, its set ends at 1920), and without that attribute the
///   plain address stays. Where the set has only densities, the plain address
///   is the `1x` one, and a denser candidate stands.
fn replaces(
    plain: Option<&str>,
    largest: &Candidate<'_>,
    all: &[Candidate<'_>],
    element: &scraper::node::Element,
) -> bool {
    let Some(plain) = plain.map(str::trim) else {
        return true;
    };
    if placeholder(plain) || all.iter().any(|candidate| candidate.url == plain) {
        return true;
    }
    if all.iter().any(|candidate| candidate.width.is_some()) {
        let declared = element.attribute("width").and_then(pixels);
        return declared
            .zip(largest.width)
            .is_some_and(|(declared, widest)| widest >= declared);
    }
    largest.density > 1.0
}

/// The address an `<img>` shows: its plain address (the lazy `data-src` or
/// `data-original`, else `src`) or, where [`replaces`] allows, the largest
/// candidate of `data-srcset` or `srcset`. An inline placeholder candidate
/// (`data:`) is never chosen.
pub(super) fn image(element: &scraper::node::Element) -> Option<&str> {
    let attribute = |name: &str| {
        element
            .attribute(name)
            .filter(|value| !value.trim().is_empty())
    };
    let plain = attribute("data-src")
        .or_else(|| attribute("data-original"))
        .or_else(|| attribute("data-original-src"))
        .or_else(|| attribute("data-actualsrc"))
        .or_else(|| attribute("src"));
    for name in ["data-srcset", "srcset"] {
        let Some(set) = attribute(name) else {
            continue;
        };
        let all = candidates(set);
        let Some(top) = largest(&all).filter(|candidate| !placeholder(candidate.url)) else {
            continue;
        };
        return if replaces(plain, top, &all, element) {
            Some(top.url)
        } else {
            plain
        };
    }
    match plain {
        Some(url) if !placeholder(url) => Some(url),
        _ => lazy_address(element).or(plain),
    }
}

/// A lazy-loading address under an attribute name this reader does not know
/// (`data-image-loader`, `data-lazy-url`): the first `data-` attribute whose
/// value is one absolute or root-relative address of an image file. Read
/// only for an image with no address, or an inline placeholder.
fn lazy_address(element: &scraper::node::Element) -> Option<&str> {
    element.attrs.iter().find_map(|(name, value)| {
        let value = value.trim();
        let located = value.starts_with("https://")
            || value.starts_with("http://")
            || value.starts_with("//")
            || (value.starts_with('/') && !value.starts_with("//"));
        let path = value.split(['?', '#']).next().unwrap_or_default();
        let image = path.rsplit_once('.').is_some_and(|(_, extension)| {
            matches!(
                extension.to_ascii_lowercase().as_str(),
                "png" | "jpg" | "jpeg" | "gif" | "webp" | "avif" | "svg" | "bmp"
            )
        });
        (name.local.starts_with("data-")
            && located
            && image
            && !value.contains(char::is_whitespace))
        .then_some(value)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_unknown_lazy_attribute_stands_in_for_a_placeholder_only() {
        let image = |markup: &str| {
            let fragment = scraper::Html::parse_fragment(markup);
            let element = fragment
                .root_element()
                .descendants()
                .filter_map(scraper::ElementRef::wrap)
                .find(|element| element.value().name() == "img")
                .unwrap()
                .value()
                .clone();
            image(&element).map(str::to_owned)
        };
        let placeholder = "data:image/svg+xml,%3Csvg%3E%3C/svg%3E";
        assert_eq!(
            image(&format!(
                r#"<img src="{placeholder}" data-image-loader="https://i.test/a-17.png" data-image-path="reviews/a-17.png">"#
            ))
            .as_deref(),
            Some("https://i.test/a-17.png")
        );
        assert_eq!(
            image(r#"<img data-lazy-url="/media/b.webp?w=800">"#).as_deref(),
            Some("/media/b.webp?w=800")
        );
        // A real address stays; a relative path, a page or two words are no image address.
        assert_eq!(
            image(r#"<img src="real.jpg" data-loader="https://i.test/other.png">"#).as_deref(),
            Some("real.jpg")
        );
        assert_eq!(
            image(&format!(
                r#"<img src="{placeholder}" data-path="reviews/c.png" data-href="https://i.test/page.html" data-x="/a.png /b.png">"#
            ))
            .as_deref(),
            Some(placeholder)
        );
    }

    #[test]
    fn the_widest_candidate_wins_then_the_densest() {
        assert_eq!(
            best("a-150.webp 150w, a-1200.webp 1200w, a-600.webp 600w"),
            Some("a-1200.webp")
        );
        assert_eq!(best("a.jpg 1x, a-2x.jpg 2x, a-3x.jpg 3x"), Some("a-3x.jpg"));
        // Widths outrank densities; a bare address is a `1x` candidate.
        assert_eq!(best("one.jpg, two.jpg 640w"), Some("two.jpg"));
        assert_eq!(best("only.jpg"), Some("only.jpg"));
        assert_eq!(best("a.jpg 100w, b.jpg 100w"), Some("a.jpg"));
        assert_eq!(best("  ,, "), None);
        assert_eq!(best(""), None);
    }

    #[test]
    fn commas_inside_an_address_do_not_split_it() {
        let cdn = "https://cdn.test/image/fetch/$s_!x!,w_424,c_limit,f_auto/https%3A%2F%2Fm.test%2Fa.png 424w, \
                   https://cdn.test/image/fetch/$s_!x!,w_1456,c_limit,f_auto/https%3A%2F%2Fm.test%2Fa.png 1456w";
        assert_eq!(
            best(cdn),
            Some(
                "https://cdn.test/image/fetch/$s_!x!,w_1456,c_limit,f_auto/https%3A%2F%2Fm.test%2Fa.png"
            )
        );
        // Only a comma that ends the address or follows a descriptor ends a
        // candidate: `a.jpg,b.jpg` is one address.
        assert_eq!(best("a.jpg,b.jpg 2x"), Some("a.jpg,b.jpg"));
        assert_eq!(best("a.jpg, b.jpg 2x"), Some("b.jpg"));
        assert_eq!(best("a.jpg 1x,b.jpg 2x"), Some("b.jpg"));
        // An inline image's comma is part of its data.
        assert_eq!(
            best("data:image/gif;base64,R0lGOD 1x, big.png 2x"),
            Some("big.png")
        );
    }

    #[test]
    fn candidates_with_unknown_descriptors_are_dropped() {
        assert_eq!(best("a.png future(a,b), b.png 2x"), Some("b.png"));
        assert_eq!(best("a.png 0w, b.png -1x, c.png 2x"), Some("c.png"));
        assert_eq!(best("a.png 100w 2x, b.png 1x"), Some("b.png"));
    }

    fn shown(html: &str) -> String {
        let document = scraper::Html::parse_fragment(html);
        let node = document
            .select(&scraper::Selector::parse("img").unwrap())
            .next()
            .unwrap();
        image(node.value()).unwrap_or_default().to_owned()
    }

    #[test]
    fn an_image_prefers_its_largest_candidate_over_its_thumbnail_src() {
        // `src` is one size of the set: the largest candidate stands.
        assert_eq!(
            shown(
                r#"<img src="t150.webp" srcset="t150.webp 150w, t300.webp 300w, t1200.webp 1200w" width="1920">"#
            ),
            "t1200.webp"
        );
        // So does the lazy address, when the set lists it.
        assert_eq!(
            shown(
                r#"<img src="p.gif" data-src="t150.webp" srcset="t150.webp 150w, t1200.webp 1200w">"#
            ),
            "t1200.webp"
        );
    }

    #[test]
    fn an_image_is_never_downgraded_to_a_smaller_candidate() {
        let set = r#"srcset="https://m.test/400/a.jpg 400w, https://m.test/1920/a.jpg 1920w""#;
        // BBC: `src` is 2560 wide and the set ends at 1920; `src` stays.
        let bbc = format!(r#"<img src="https://m.test/2560/a.jpg" {set} width="2560">"#);
        assert_eq!(shown(&bbc), "https://m.test/2560/a.jpg");
        // Without a declared width nothing says `src` is smaller: it stays.
        let bare = format!(r#"<img src="https://m.test/2560/a.jpg" {set}>"#);
        assert_eq!(shown(&bare), "https://m.test/2560/a.jpg");
        // A width in percent names no size.
        let percent = format!(r#"<img src="https://m.test/2560/a.jpg" {set} width="100%">"#);
        assert_eq!(shown(&percent), "https://m.test/2560/a.jpg");
        // A declared width the largest candidate reaches lets it stand.
        for width in ["1920", "1200px", "800"] {
            let reached = format!(r#"<img src="https://m.test/2560/a.jpg" {set} width="{width}">"#);
            assert_eq!(shown(&reached), "https://m.test/1920/a.jpg", "{width}");
        }
        // One candidate, far smaller than `src` (a 160w portrait of 640).
        assert_eq!(
            shown(r#"<img src="p640.png" srcset="p160.png 160w" width="640" height="640">"#),
            "p640.png"
        );
        // Densities: `src` is the `1x` image of a set that lists the others.
        assert_eq!(
            shown(r#"<img src="250px.png" srcset="330px.png 1.5x, 500px.png 2x" width="250">"#),
            "500px.png"
        );
        assert_eq!(
            shown(r#"<img src="250px.png" srcset="125px.png 0.5x">"#),
            "250px.png"
        );
    }

    #[test]
    fn lazy_loading_addresses_stand_in_for_placeholder_candidates() {
        for (html, expected) in [
            (
                r#"<img src="data:image/gif;base64,R0lGOD" data-src="real.jpg">"#,
                "real.jpg",
            ),
            (
                r#"<img srcset="data:image/gif;base64,R0lGOD 1x" data-src="real.jpg" src="p.gif">"#,
                "real.jpg",
            ),
            (
                r#"<img srcset="p.gif 1x" data-srcset="small.jpg 1x, big.jpg 2x" src="p.gif">"#,
                "big.jpg",
            ),
            (r#"<img data-original="o.jpg" src="p.gif">"#, "o.jpg"),
            (r#"<img src="plain.jpg">"#, "plain.jpg"),
        ] {
            assert_eq!(shown(html), expected, "{html}");
        }
    }
}
