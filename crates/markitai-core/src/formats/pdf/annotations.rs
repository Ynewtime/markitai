//! Review comments.
//!
//! A reviewer's sticky note (a Text annotation) and the comment on a
//! highlight, an underline, a strike-out, a shape, a stamp or another
//! markup annotation (its `/Contents`, by its author, `/T`) are no part of
//! the page's content, which the page reader reads: each page's comments
//! follow it as a "Comments" section, one quotation each, a text markup
//! annotation's with the text it marks (`/QuadPoints`). A FreeText
//! annotation shows its text on the page, and the page reader reads it
//! there; links, form widgets and pop-ups (which show another annotation's
//! comment) are no comments, nor is an annotation the page hides.
use super::geometry;
use lopdf::{Dictionary, Document, Object, ObjectId};
use pdf_inspector::{TextItem, types::ItemType};
use std::collections::{BTreeMap, HashSet};

/// The annotations whose `/Contents` is a reviewer's comment: ISO 32000-1's
/// markup annotations less FreeText (page text) and those that carry media
/// (Sound, Redact).
const COMMENTED: &[&[u8]] = &[
    b"Text",
    b"Highlight",
    b"Underline",
    b"Squiggly",
    b"StrikeOut",
    b"Caret",
    b"Square",
    b"Circle",
    b"Line",
    b"Polygon",
    b"PolyLine",
    b"Ink",
    b"Stamp",
    b"FileAttachment",
];

/// The text markup annotations, whose `/QuadPoints` mark text.
const MARKUP: &[&[u8]] = &[b"Highlight", b"Underline", b"Squiggly", b"StrikeOut"];

/// The longest quotation of marked text, in characters, before it is cut
/// short with an ellipsis.
const QUOTE_CHARS: usize = 200;

/// The most comments read from one page.
const PAGE_COMMENTS: usize = 1_000;

/// The most bytes a comment held in a stream expands to.
const STREAM_BYTES: usize = 1 << 20;

#[derive(Debug, PartialEq)]
pub(super) struct Comment {
    id: Option<ObjectId>,
    /// The annotation this one replies to (`/IRT`, unless it is a group).
    reply_to: Option<ObjectId>,
    stamp: bool,
    author: Option<String>,
    text: String,
    /// The boxes a text markup annotation marks, in user space, as
    /// `[x0, y0, x1, y1]`.
    marked: Vec<[f32; 4]>,
}

fn resolve<'a>(pdf: &'a Document, object: &'a Object) -> Option<&'a Object> {
    match object {
        Object::Reference(id) => pdf.get_object(*id).ok(),
        direct => Some(direct),
    }
}

/// A text string: UTF-16 or UTF-8 after its byte order mark, else
/// PDFDocEncoding, whose table in lopdf has no line breaks or tabs: a line
/// break (`CR`, `LF` or both) is kept as `\n` and a tab read as a space.
fn decode(bytes: &[u8]) -> Option<String> {
    if bytes.starts_with(b"\xFE\xFF") || bytes.starts_with(b"\xEF\xBB\xBF") {
        return lopdf::decode_text_string(&Object::string_literal(bytes.to_vec())).ok();
    }
    let mut out = String::new();
    let mut rest = bytes;
    while !rest.is_empty() {
        let end = rest
            .iter()
            .position(|b| matches!(b, b'\r' | b'\n' | b'\t'))
            .unwrap_or(rest.len());
        let part = Object::string_literal(rest[..end].to_vec());
        out.push_str(&lopdf::decode_text_string(&part).ok()?);
        rest = &rest[end..];
        match rest {
            [b'\r', b'\n', tail @ ..] | [b'\r' | b'\n', tail @ ..] => {
                out.push('\n');
                rest = tail;
            }
            [b'\t', tail @ ..] => {
                out.push(' ');
                rest = tail;
            }
            _ => {}
        }
    }
    Some(out)
}

/// The text string `dict` holds under `key` (a string or a stream),
/// decoded and trimmed; `None` when there is none.
fn text(pdf: &Document, dict: &Dictionary, key: &[u8]) -> Option<String> {
    let text = match resolve(pdf, dict.get(key).ok()?)? {
        // A comment held in a stream is expanded within a bound of its own.
        Object::Stream(stream) if stream.dict.has(b"Filter") => {
            decode(&stream.decompressed_content_with_limit(STREAM_BYTES).ok()?)?
        }
        Object::Stream(stream) => decode(&stream.content)?,
        other => decode(other.as_str().ok()?)?,
    };
    let text = text.trim();
    (!text.is_empty()).then(|| text.to_owned())
}

/// The plain text of rich text (`/RC`, an XHTML body): its tags dropped, a
/// closing paragraph or a break ending a line, and XML's entities decoded.
fn rich_text(pdf: &Document, dict: &Dictionary) -> Option<String> {
    let markup = text(pdf, dict, b"RC")?;
    let mut plain = String::new();
    let mut rest = markup.as_str();
    while let Some(open) = rest.find('<') {
        plain.push_str(&rest[..open]);
        let Some(close) = rest[open..].find('>') else {
            rest = "";
            break;
        };
        let tag = rest[open + 1..open + close].trim().to_ascii_lowercase();
        if tag.starts_with("/p") || tag.starts_with("br") || tag.starts_with("/div") {
            plain.push('\n');
        }
        rest = &rest[open + close + 1..];
    }
    plain.push_str(rest);
    let plain = plain
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
        .replace("&#13;", "\n")
        .replace("&amp;", "&");
    let plain = plain.trim();
    (!plain.is_empty()).then(|| plain.to_owned())
}

/// `/QuadPoints`: each group of eight numbers is a box's four corners.
fn quad_boxes(pdf: &Document, dict: &Dictionary) -> Vec<[f32; 4]> {
    let Some(points) = dict
        .get(b"QuadPoints")
        .ok()
        .and_then(|points| resolve(pdf, points))
        .and_then(|points| points.as_array().ok())
    else {
        return Vec::new();
    };
    points
        .as_chunks::<8>()
        .0
        .iter()
        .take(PAGE_COMMENTS)
        .filter_map(|quad| {
            let numbers: Vec<f32> = quad.iter().filter_map(|n| n.as_float().ok()).collect();
            if numbers.len() != 8 || !numbers.iter().all(|n| n.is_finite()) {
                return None;
            }
            let xs = [numbers[0], numbers[2], numbers[4], numbers[6]];
            let ys = [numbers[1], numbers[3], numbers[5], numbers[7]];
            let low = |v: [f32; 4]| v.into_iter().fold(f32::INFINITY, f32::min);
            let high = |v: [f32; 4]| v.into_iter().fold(f32::NEG_INFINITY, f32::max);
            Some([low(xs), low(ys), high(xs), high(ys)])
        })
        .collect()
}

/// The comments on page `page`, in the order of its `/Annots`, each reply
/// after the comment it answers.
pub(super) fn page_comments(pdf: &Document, page: ObjectId) -> Vec<Comment> {
    let Some(annotations) = pdf
        .get_dictionary(page)
        .ok()
        .and_then(|page| page.get(b"Annots").ok())
        .and_then(|annotations| resolve(pdf, annotations))
        .and_then(|annotations| annotations.as_array().ok())
    else {
        return Vec::new();
    };
    let mut comments = Vec::new();
    for annotation in annotations.iter().take(PAGE_COMMENTS) {
        let id = annotation.as_reference().ok();
        let Some(dict) = resolve(pdf, annotation).and_then(|a| a.as_dict().ok()) else {
            continue;
        };
        let Some(subtype) = dict
            .get(b"Subtype")
            .ok()
            .and_then(|subtype| subtype.as_name().ok())
            .filter(|subtype| COMMENTED.contains(subtype))
        else {
            continue;
        };
        // Hidden (2) or NoView (32): the page does not show it.
        if dict
            .get(b"F")
            .ok()
            .and_then(|flags| flags.as_i64().ok())
            .is_some_and(|flags| flags & (2 | 32) != 0)
        {
            continue;
        }
        let Some(body) = text(pdf, dict, b"Contents").or_else(|| rich_text(pdf, dict)) else {
            continue;
        };
        let group = dict
            .get(b"RT")
            .ok()
            .and_then(|rt| rt.as_name().ok())
            .is_some_and(|rt| rt == b"Group");
        comments.push(Comment {
            id,
            reply_to: dict
                .get(b"IRT")
                .ok()
                .and_then(|irt| irt.as_reference().ok())
                .filter(|_| !group),
            stamp: subtype == b"Stamp",
            author: text(pdf, dict, b"T"),
            text: body,
            marked: if MARKUP.contains(&subtype) {
                quad_boxes(pdf, dict)
            } else {
                Vec::new()
            },
        });
    }
    threaded(comments)
}

/// Comments in their order, each reply moved after the comment it answers
/// and that comment's earlier replies.
fn threaded(comments: Vec<Comment>) -> Vec<Comment> {
    let ids: HashSet<ObjectId> = comments.iter().filter_map(|c| c.id).collect();
    let mut slots: Vec<Option<Comment>> = comments.into_iter().map(Some).collect();
    let mut ordered = Vec::with_capacity(slots.len());
    for index in 0..slots.len() {
        let top = slots[index]
            .as_ref()
            .is_some_and(|c| c.reply_to.is_none_or(|parent| !ids.contains(&parent)));
        if !top {
            continue;
        }
        let mut stack = vec![index];
        while let Some(at) = stack.pop() {
            let Some(comment) = slots[at].take() else {
                continue;
            };
            let id = comment.id;
            ordered.push(comment);
            if let Some(id) = id {
                let replies: Vec<usize> = (0..slots.len())
                    .filter(|&reply| {
                        slots[reply]
                            .as_ref()
                            .is_some_and(|c| c.reply_to == Some(id))
                    })
                    .collect();
                stack.extend(replies.into_iter().rev());
            }
        }
    }
    // A reply in a cycle of replies keeps its place at the end.
    ordered.extend(slots.into_iter().flatten());
    ordered
}

/// A character's advance in Helvetica, in thousandths of an em: the shares
/// of a run's width its characters take, which locate the words a quad
/// covers. Wide East Asian characters take an em.
fn advance(c: char) -> f32 {
    const ASCII: [u16; 95] = [
        278, 278, 355, 556, 556, 889, 667, 191, 333, 333, 389, 584, 278, 333, 278, 278, 556, 556,
        556, 556, 556, 556, 556, 556, 556, 556, 278, 278, 584, 584, 584, 556, 1015, 667, 667, 722,
        722, 667, 611, 778, 722, 278, 500, 667, 556, 833, 722, 778, 667, 778, 722, 667, 611, 722,
        667, 944, 667, 667, 611, 278, 278, 278, 469, 556, 333, 556, 556, 500, 556, 556, 278, 556,
        556, 222, 222, 500, 222, 833, 556, 556, 556, 556, 333, 500, 278, 556, 500, 722, 500, 500,
        500, 334, 260, 334, 584,
    ];
    match c as u32 {
        code @ 0x20..=0x7e => f32::from(ASCII[(code - 0x20) as usize]),
        0x1100..=0x11ff | 0x2e80..=0xa4cf | 0xac00..=0xd7a3 | 0xf900..=0xfaff | 0xff00..=0xff60 => {
            1000.
        }
        _ => 556.,
    }
}

/// The words of `run` a box covers: a word whose characters (placed by
/// their [`advance`] across the run's width) are half or more inside the
/// box's width.
fn covered_words(run: &TextItem, x0: f32, x1: f32) -> Vec<&str> {
    let text = run.text.as_str();
    let total: f32 = text.chars().map(advance).sum();
    if total <= 0. || run.width <= 0. {
        return Vec::new();
    }
    let scale = run.width / total;
    let mut words = Vec::new();
    let mut offset = 0.;
    let mut word: Option<(usize, usize, usize)> = None; // start byte, chars, covered
    let mut close = |word: &mut Option<(usize, usize, usize)>, end: usize| {
        if let Some((start, chars, covered)) = word.take()
            && covered > 0
            && covered * 2 >= chars
        {
            words.push(&text[start..end]);
        }
    };
    for (at, c) in text.char_indices() {
        let width = advance(c) * scale;
        let centre = run.x + offset + width / 2.;
        offset += width;
        if c.is_whitespace() {
            close(&mut word, at);
            continue;
        }
        let inside = usize::from(centre >= x0 && centre <= x1);
        word = Some(match word {
            Some((start, chars, covered)) => (start, chars + 1, covered + inside),
            None => (at, 1, inside),
        });
    }
    close(&mut word, text.len());
    words
}

/// The text the boxes of a text markup annotation mark: on each box, the
/// words of the upright runs whose middle height lies inside it, left to
/// right. `items` are the page's runs in its visible box, whose lower left
/// corner is at `frame`'s origin.
fn quote(marked: &[[f32; 4]], items: &[TextItem], frame: geometry::Frame) -> Option<String> {
    let mut words = Vec::new();
    for &[x0, y0, x1, y1] in marked {
        let (x0, y0, x1, y1) = (x0 - frame.x, y0 - frame.y, x1 - frame.x, y1 - frame.y);
        let mut runs: Vec<&TextItem> = items
            .iter()
            .filter(|run| {
                let middle = run.y + run.font_size * 0.35;
                matches!(run.item_type, ItemType::Text)
                    && run.rotation == 0.
                    && middle >= y0
                    && middle <= y1
                    && run.x < x1
                    && run.x + run.width > x0
            })
            .collect();
        runs.sort_by(|a, b| a.x.total_cmp(&b.x));
        for run in runs {
            words.extend(covered_words(run, x0, x1));
        }
    }
    let quoted = words.join(" ");
    if quoted.is_empty() {
        return None;
    }
    Some(if quoted.chars().count() > QUOTE_CHARS {
        let cut: String = quoted.chars().take(QUOTE_CHARS).collect();
        format!("{}…", cut.trim_end())
    } else {
        quoted
    })
}

/// A page's "Comments" section: a quotation for each comment, the text it
/// marks given (`quoted`) where it is known.
fn section(comments: &[(Comment, Option<String>)]) -> String {
    let mut out = String::from("**Comments**");
    for (comment, quoted) in comments {
        let kind = if comment.stamp {
            "Stamp"
        } else if comment.reply_to.is_some() {
            "Reply"
        } else {
            "Comment"
        };
        let mut head = kind.to_owned();
        if let Some(author) = &comment.author {
            head.push_str(&format!(" ({})", super::super::escape(author)));
        }
        if let Some(quoted) = quoted {
            head.push_str(&format!(" on \"{}\"", super::super::escape(quoted)));
        }
        // A line of the comment that would open a block of its own reads
        // as written.
        let body = super::super::literal_heading_marks(&super::super::escape(&comment.text));
        out.push_str("\n\n> ");
        out.push_str(&head);
        out.push_str(": ");
        for (index, line) in body.split('\n').enumerate() {
            if index > 0 {
                out.push_str("\n>");
                if !line.trim().is_empty() {
                    out.push(' ');
                }
            }
            out.push_str(line.trim_end());
        }
    }
    out
}

/// Reads the positioned runs of the given pages in their visible box, and
/// the pages among them whose text the reader turned.
pub(super) type Positions<'a> = dyn Fn(&HashSet<u32>) -> (Vec<TextItem>, HashSet<u32>) + 'a;

/// Each page's "Comments" section, by page number, for the pages that have
/// comments. `position` reads the positioned runs of pages in their
/// visible box, where a text markup annotation's marked text is found; a
/// page whose box is turned or not known is quoted without it.
pub(super) fn sections(
    pdf: &Document,
    pages: &BTreeMap<u32, ObjectId>,
    position: &Positions<'_>,
) -> BTreeMap<usize, String> {
    let comments: Vec<(u32, ObjectId, Vec<Comment>)> = pages
        .iter()
        .map(|(&number, &id)| (number, id, page_comments(pdf, id)))
        .filter(|(_, _, comments)| !comments.is_empty())
        .collect();
    let marked: HashSet<u32> = comments
        .iter()
        .filter(|(_, _, comments)| comments.iter().any(|c| !c.marked.is_empty()))
        .map(|(number, _, _)| *number)
        .collect();
    let (items, turned) = if marked.is_empty() {
        (Vec::new(), HashSet::new())
    } else {
        position(&marked)
    };
    comments
        .into_iter()
        .map(|(number, id, comments)| {
            let frame = geometry::frame(pdf, id).filter(|_| !turned.contains(&number));
            let page_items: Vec<TextItem> = items
                .iter()
                .filter(|item| item.page == number)
                .cloned()
                .collect();
            let quoted: Vec<(Comment, Option<String>)> = comments
                .into_iter()
                .map(|comment| {
                    let quoted = frame.and_then(|frame| quote(&comment.marked, &page_items, frame));
                    (comment, quoted)
                })
                .collect();
            (number as usize, section(&quoted))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use lopdf::dictionary;

    fn run(text: &str, x: f32, y: f32, width: f32) -> TextItem {
        let mut item = helvetica();
        item.text = text.into();
        item.x = x;
        item.y = y;
        item.width = width;
        item
    }

    /// A 12pt Helvetica run on page 1, with every style unset.
    fn helvetica() -> TextItem {
        TextItem {
            text: String::new(),
            x: 0.,
            y: 0.,
            width: 0.,
            height: 12.,
            font: "Helvetica".into(),
            font_tag: String::new(),
            legacy_symbol_rewrite: false,
            font_size: 12.,
            page: 1,
            is_bold: false,
            is_italic: false,
            font_weight: None,
            bold_source: None,
            fixed_pitch: None,
            fill_color: None,
            stroke_color: None,
            render_mode: None,
            is_underline: false,
            is_strikeout: false,
            rotation: 0.,
            advance_known: true,
            item_type: ItemType::Text,
            mcid: None,
            baseline_shift: 0.,
        }
    }

    #[test]
    fn a_highlight_quotes_the_words_its_boxes_cover() {
        // PyMuPDF's highlight of "thirty days" in 12pt Helvetica, its box
        // reaching a little past both words.
        let line = run(
            "The contract states that payment is due within thirty days of the invoice date. Late",
            60.,
            749.1,
            432.9,
        );
        let frame = geometry::Frame {
            x: 0.,
            y: 0.,
            width: 595.,
            height: 842.,
        };
        let marked = [[307.45, 745.51, 362.14, 762.]];
        assert_eq!(
            quote(&marked, std::slice::from_ref(&line), frame).as_deref(),
            Some("thirty days")
        );
        // A box over the next line marks nothing on this one.
        assert_eq!(quote(&[[60., 700., 300., 720.]], &[line], frame), None);
    }

    #[test]
    fn replies_follow_the_comment_they_answer() {
        let comment = |id: u32, reply_to: Option<u32>| Comment {
            id: Some((id, 0)),
            reply_to: reply_to.map(|id| (id, 0)),
            stamp: false,
            author: None,
            text: id.to_string(),
            marked: Vec::new(),
        };
        let order: Vec<String> = threaded(vec![
            comment(1, None),
            comment(2, None),
            comment(3, Some(1)),
            comment(4, Some(3)),
            comment(5, Some(9)),
        ])
        .into_iter()
        .map(|c| c.text)
        .collect();
        assert_eq!(order, ["1", "3", "4", "2", "5"]);
    }

    #[test]
    fn comments_are_read_from_notes_and_markup_but_not_from_page_text() {
        let mut pdf = Document::with_version("1.7");
        let annotation =
            |pdf: &mut Document, dict: Dictionary| Object::Reference(pdf.add_object(dict));
        let note = annotation(
            &mut pdf,
            dictionary! { "Type" => "Annot", "Subtype" => "Text", "Rect" => vec![0.into(), 0.into(), 16.into(), 16.into()],
            "Contents" => Object::string_literal("Check *this* with legal\r\nsecond line"), "T" => Object::string_literal("Ann") },
        );
        let reply = annotation(
            &mut pdf,
            dictionary! { "Type" => "Annot", "Subtype" => "Text", "Rect" => vec![0.into(), 0.into(), 16.into(), 16.into()],
            "Contents" => Object::string_literal("Done"), "IRT" => note.clone() },
        );
        let hidden = annotation(
            &mut pdf,
            dictionary! { "Type" => "Annot", "Subtype" => "Text", "F" => 2, "Contents" => Object::string_literal("Hidden") },
        );
        let free_text = annotation(
            &mut pdf,
            dictionary! { "Type" => "Annot", "Subtype" => "FreeText", "Contents" => Object::string_literal("On the page") },
        );
        let popup = annotation(
            &mut pdf,
            dictionary! { "Type" => "Annot", "Subtype" => "Popup", "Contents" => Object::string_literal("Popup") },
        );
        let stamp = annotation(
            &mut pdf,
            dictionary! { "Type" => "Annot", "Subtype" => "Stamp", "Contents" => Object::string_literal("Approved") },
        );
        let empty = annotation(
            &mut pdf,
            dictionary! { "Type" => "Annot", "Subtype" => "Highlight", "QuadPoints" => vec![0.into(); 8] },
        );
        let page = pdf.add_object(dictionary! {
            "Type" => "Page",
            "Annots" => vec![reply, note, hidden, free_text, popup, stamp, empty],
        });
        let comments = page_comments(&pdf, page);
        let quoted: Vec<(Comment, Option<String>)> =
            comments.into_iter().map(|c| (c, None)).collect();
        assert_eq!(
            section(&quoted),
            "**Comments**\n\n> Comment (Ann): Check \\*this\\* with legal\n> second line\n\n> Reply: Done\n\n> Stamp: Approved"
        );
    }

    /// A page of text with a highlight's comment and a sticky note, read
    /// end to end: the page's text, then its comments, the highlight's with
    /// the words it marks.
    #[test]
    fn a_page_is_followed_by_its_comments() {
        const LINE: &str = "Payment is due within thirty days of the invoice date.";
        let mut pdf = Document::with_version("1.7");
        let tree = pdf.new_object_id();
        let font = pdf.add_object(dictionary! {
            "Type" => "Font", "Subtype" => "Type1", "BaseFont" => "Helvetica",
            "Encoding" => "WinAnsiEncoding",
        });
        let content = pdf.add_object(lopdf::Stream::new(
            Dictionary::new(),
            format!("BT /F1 12 Tf 72 700 Td ({LINE}) Tj ET").into_bytes(),
        ));
        // The marked words' box, from Helvetica's advances at 12pt.
        let width = |text: &str| text.chars().map(advance).sum::<f32>() * 12. / 1000.;
        let x0 = 72. + width("Payment is due within ");
        let x1 = x0 + width("thirty days");
        let quad: Vec<Object> = [x0, 712., x1, 712., x0, 697., x1, 697.]
            .into_iter()
            .map(Object::Real)
            .collect();
        let highlight = pdf.add_object(dictionary! {
            "Type" => "Annot", "Subtype" => "Highlight", "QuadPoints" => quad,
            "Rect" => vec![x0.into(), 697.into(), x1.into(), 712.into()],
            "Contents" => Object::string_literal("Should this be 45 days?"),
            "T" => Object::string_literal("Reviewer"),
        });
        let note = pdf.add_object(dictionary! {
            "Type" => "Annot", "Subtype" => "Text",
            "Rect" => vec![500.into(), 700.into(), 516.into(), 716.into()],
            "Contents" => Object::string_literal("Check with legal"),
        });
        let page = pdf.add_object(dictionary! {
            "Type" => "Page", "Parent" => tree, "Contents" => content,
            "Resources" => dictionary! { "Font" => dictionary! { "F1" => font } },
            "Annots" => vec![highlight.into(), note.into()],
        });
        pdf.objects.insert(
            tree,
            dictionary! {
                "Type" => "Pages", "Count" => 1, "Kids" => vec![page.into()],
                "MediaBox" => vec![0.into(), 0.into(), 612.into(), 792.into()]
            }
            .into(),
        );
        let catalog = pdf.add_object(dictionary! { "Type" => "Catalog", "Pages" => tree });
        pdf.trailer.set("Root", catalog);
        let mut bytes = Vec::new();
        pdf.save_to(&mut bytes).unwrap();
        let document = super::super::extract(&bytes).unwrap();
        assert_eq!(
            document.markdown,
            format!(
                "<!-- Page number: 1 -->\n\n{LINE}\n\n**Comments**\n\n\
                 > Comment (Reviewer) on \"thirty days\": Should this be 45 days?\n\n\
                 > Comment: Check with legal"
            )
        );
    }
}
