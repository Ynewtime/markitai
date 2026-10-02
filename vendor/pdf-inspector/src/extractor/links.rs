//! Hyperlink, FreeText annotation and AcroForm field extraction.

use crate::types::{ItemType, TextItem};
use lopdf::{Dictionary, Document, Object, ObjectId};
use std::collections::{HashMap, HashSet};

use super::fonts::{resolve_array, resolve_dict};
use super::get_number;

/// Upper bound on the number of form-field nodes visited during a single
/// `extract_form_fields` pass. A crafted PDF can chain thousands of distinct
/// `/Kids` fields to blow the stack even without an outright reference cycle,
/// so we cap total traversal work in addition to detecting cycles.
const MAX_FORM_FIELD_NODES: usize = 100_000;

/// Upper bound on `/Kids` recursion depth. Real AcroForm hierarchies are only
/// a few levels deep (fields → child fields → widgets); a crafted PDF can chain
/// tens of thousands of distinct fields into a linear `/Kids` list that would
/// overflow the stack via depth-first recursion long before the node budget is
/// reached. This depth cap bounds the stack independently of total node count.
const MAX_FORM_FIELD_DEPTH: usize = 100;

/// Traversal budget for the AcroForm field walk. Bounds both the number of
/// distinct nodes visited *and* the total number of `/Fields`/`/Kids` entries
/// examined.
///
/// Counting `visited` alone is not enough: invalid entries (non-references) and
/// duplicate references never grow `visited`, so an oversized array full of them
/// would iterate to completion no matter how large. Charging every examined
/// entry against the same budget makes it a real cap on traversal work.
pub(crate) struct FieldWalkBudget {
    visited: HashSet<ObjectId>,
    examined: usize,
}

impl FieldWalkBudget {
    fn new() -> Self {
        Self {
            visited: HashSet::new(),
            examined: 0,
        }
    }

    /// True once the budget is spent; callers must stop iterating and recursing.
    fn exhausted(&self) -> bool {
        self.visited.len() >= MAX_FORM_FIELD_NODES || self.examined >= MAX_FORM_FIELD_NODES
    }
}

/// An item of the given type with every style unset.
fn plain_item(
    text: String,
    (x, y, width, height): (f32, f32, f32, f32),
    font_size: f32,
    page: u32,
    item_type: ItemType,
) -> TextItem {
    TextItem {
        text,
        x,
        y,
        width,
        height,
        font: String::new(),
        font_tag: String::new(),
        legacy_symbol_rewrite: false,
        font_size,
        page,
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
        rotation: 0.0,
        advance_known: true,
        item_type,
        mcid: None,
        baseline_shift: 0.0,
    }
}

/// An annotation's `/Rect` as `(x, y, width, height)`, its corners in any
/// order.
fn annotation_rect(dict: &Dictionary) -> Option<(f32, f32, f32, f32)> {
    let rect = dict.get(b"Rect").ok()?.as_array().ok()?;
    if rect.len() < 4 {
        return None;
    }
    let [x1, y1, x2, y2] = [0, 1, 2, 3].map(|i| get_number(&rect[i]).unwrap_or(0.0));
    Some((x1.min(x2), y1.min(y2), (x2 - x1).abs(), (y2 - y1).abs()))
}

/// markitai: whether an annotation's `/F` flags keep it off the page
/// (Hidden, 2, or NoView, 32).
fn annotation_hidden(dict: &Dictionary) -> bool {
    dict.get(b"F")
        .ok()
        .and_then(|flags| flags.as_i64().ok())
        .is_some_and(|flags| flags & (2 | 32) != 0)
}

/// markitai: the text string `dict` holds under `key`, resolved and decoded
/// (UTF-16, UTF-8 or PDFDocEncoding, line breaks kept), with surrounding
/// white space trimmed; `None` when there is none.
fn text_entry(doc: &Document, dict: &Dictionary, key: &[u8]) -> Option<String> {
    let object = match dict.get(key).ok()? {
        Object::Reference(id) => doc.get_object(*id).ok()?,
        direct => direct,
    };
    let text = crate::text_utils::decode_pdf_text_string(object.as_str().ok()?);
    let text = text.trim();
    (!text.is_empty()).then(|| text.to_string())
}

/// markitai: a stream's bytes, expanded within the loader's bound
/// ([`crate::MAX_STREAM_DECOMPRESSED_BYTES`]); `None` when its filters fail
/// or would exceed it.
fn stream_bytes(stream: &lopdf::Stream) -> Option<Vec<u8>> {
    if stream.dict.has(b"Filter") {
        stream
            .decompressed_content_with_limit(crate::MAX_STREAM_DECOMPRESSED_BYTES)
            .ok()
    } else {
        Some(stream.content.clone())
    }
}

/// markitai: the plain text of rich text (`/RC`, an XHTML body): its tags
/// dropped, a closing paragraph or a break ending a line, and the five XML
/// entities and numeric references decoded.
fn rich_text_plain(doc: &Document, dict: &Dictionary) -> Option<String> {
    let markup = match dict.get(b"RC").ok()? {
        Object::Reference(id) => doc.get_object(*id).ok()?,
        direct => direct,
    };
    let markup = match markup {
        Object::Stream(stream) => crate::text_utils::decode_pdf_text_string(&stream_bytes(stream)?),
        other => crate::text_utils::decode_pdf_text_string(other.as_str().ok()?),
    };
    let mut text = String::new();
    let mut rest = markup.as_str();
    while let Some(open) = rest.find('<') {
        text.push_str(&rest[..open]);
        let Some(close) = rest[open..].find('>') else {
            break;
        };
        let tag = rest[open + 1..open + close].trim().to_ascii_lowercase();
        if tag.starts_with("/p") || tag.starts_with("br") || tag.starts_with("/div") {
            text.push('\n');
        }
        rest = &rest[open + close + 1..];
    }
    if !rest.contains('<') {
        text.push_str(rest);
    }
    let mut decoded = String::with_capacity(text.len());
    let mut rest = text.as_str();
    while let Some(at) = rest.find('&') {
        decoded.push_str(&rest[..at]);
        let entity = rest[at + 1..].split(';').next().unwrap_or("");
        let character = match entity {
            "amp" => Some('&'),
            "lt" => Some('<'),
            "gt" => Some('>'),
            "quot" => Some('"'),
            "apos" => Some('\''),
            _ => entity
                .strip_prefix("#x")
                .or_else(|| entity.strip_prefix("#X"))
                .and_then(|hex| u32::from_str_radix(hex, 16).ok())
                .or_else(|| entity.strip_prefix('#').and_then(|n| n.parse().ok()))
                .and_then(char::from_u32),
        };
        match character {
            Some(character) if rest[at + 1..].contains(';') => {
                decoded.push(character);
                rest = &rest[at + entity.len() + 2..];
            }
            _ => {
                decoded.push('&');
                rest = &rest[at + 1..];
            }
        }
    }
    decoded.push_str(rest);
    let decoded = decoded.trim();
    (!decoded.is_empty()).then(|| decoded.to_string())
}

/// markitai: the font size a default appearance string (`/DA`, such as
/// `0 g /Helv 12 Tf`) sets: the number before its last `Tf`. Zero asks for
/// automatic sizing.
fn appearance_font_size(doc: &Document, dict: &Dictionary) -> Option<f32> {
    let appearance = text_entry(doc, dict, b"DA")?;
    let tokens: Vec<&str> = appearance.split_whitespace().collect();
    let at = tokens.iter().rposition(|token| *token == "Tf")?;
    tokens.get(at.checked_sub(1)?)?.parse().ok()
}

/// markitai: a FreeText annotation's text, as page text: the annotation
/// draws it on the page from its own appearance, outside the page's
/// content, so the content stream does not hold it. Each line of its
/// `/Contents` (or of its rich text) becomes one item, set in the
/// annotation's text box (`/Rect` less `/RD`) at its `/DA` size and `/Q`
/// alignment, its width estimated at half an em per character.
fn free_text_items(doc: &Document, dict: &Dictionary, page: u32) -> Vec<TextItem> {
    if annotation_hidden(dict) {
        return Vec::new();
    }
    let Some(text) = text_entry(doc, dict, b"Contents").or_else(|| rich_text_plain(doc, dict))
    else {
        return Vec::new();
    };
    let Some((mut x, mut y, mut width, mut height)) = annotation_rect(dict) else {
        return Vec::new();
    };
    if let Some(inset) = dict
        .get(b"RD")
        .ok()
        .and_then(|rd| rd.as_array().ok())
        .filter(|rd| rd.len() == 4)
        .map(|rd| [0, 1, 2, 3].map(|i| get_number(&rd[i]).unwrap_or(0.0).max(0.0)))
        .filter(|[left, top, right, bottom]| left + right < width && top + bottom < height)
    {
        let [left, top, right, bottom] = inset;
        x += left;
        y += bottom;
        width -= left + right;
        height -= top + bottom;
    }
    let size = match appearance_font_size(doc, dict) {
        Some(size) if size > 1.0 && size < 200.0 => size,
        _ => 12f32.min(height * 0.8).max(4.0),
    };
    let alignment = dict
        .get(b"Q")
        .ok()
        .and_then(|q| q.as_i64().ok())
        .unwrap_or(0);
    let text = text.replace("\r\n", "\n").replace('\r', "\n");
    let top = y + height;
    text.split('\n')
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .enumerate()
        .map(|(index, line)| {
            let line_width = (line.chars().count() as f32 * size * 0.5).min((width - 4.0).max(1.0));
            let left = match alignment {
                1 => x + (width - line_width) / 2.0,
                2 => x + width - 2.0 - line_width,
                _ => x + 2.0,
            };
            let baseline = top - 2.0 - size * (0.8 + 1.2 * index as f32);
            plain_item(
                line.to_string(),
                (left, baseline, line_width, size),
                size,
                page,
                ItemType::Text,
            )
        })
        .collect()
}

/// A page's link annotations, as items carrying their target, and its
/// FreeText annotations, as the text they show (see [`free_text_items`]).
pub fn extract_page_links(doc: &Document, page_id: ObjectId, page_num: u32) -> Vec<TextItem> {
    let mut links = Vec::new();

    // Try to get the page dictionary
    if let Ok(page_dict) = doc.get_dictionary(page_id) {
        // Get Annots array
        let annots = if let Ok(annots_ref) = page_dict.get(b"Annots") {
            if let Ok(obj_ref) = annots_ref.as_reference() {
                doc.get_object(obj_ref)
                    .ok()
                    .and_then(|o| o.as_array().ok().cloned())
            } else {
                annots_ref.as_array().ok().cloned()
            }
        } else {
            None
        };

        if let Some(annots) = annots {
            for annot_ref in annots {
                // Get annotation dictionary
                let annot_dict = if let Ok(obj_ref) = annot_ref.as_reference() {
                    doc.get_dictionary(obj_ref).ok()
                } else {
                    annot_ref.as_dict().ok()
                };

                if let Some(annot_dict) = annot_dict {
                    // Check if this is a Link annotation
                    if let Ok(subtype) = annot_dict.get(b"Subtype") {
                        if let Ok(subtype_name) = subtype.as_name() {
                            if subtype_name == b"FreeText" {
                                links.extend(free_text_items(doc, annot_dict, page_num));
                                continue;
                            }
                            if subtype_name != b"Link" {
                                continue;
                            }
                        }
                    }

                    // Get the Rect (position)
                    let rect = if let Ok(rect_obj) = annot_dict.get(b"Rect") {
                        if let Ok(rect_array) = rect_obj.as_array() {
                            if rect_array.len() >= 4 {
                                let x1 = get_number(&rect_array[0]).unwrap_or(0.0);
                                let y1 = get_number(&rect_array[1]).unwrap_or(0.0);
                                let x2 = get_number(&rect_array[2]).unwrap_or(0.0);
                                let y2 = get_number(&rect_array[3]).unwrap_or(0.0);
                                Some((x1, y1, x2 - x1, y2 - y1))
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    } else {
                        None
                    };

                    // Get the action (A dictionary) or Dest
                    let uri = extract_link_uri(doc, annot_dict);

                    if let (Some(rect), Some(url)) = (rect, uri) {
                        links.push(plain_item(
                            url.clone(),
                            rect,
                            0.0,
                            page_num,
                            ItemType::Link(url),
                        ));
                    }
                }
            }
        }
    }

    links
}

/// Extract URI from a link annotation
pub(crate) fn extract_link_uri(doc: &Document, annot_dict: &lopdf::Dictionary) -> Option<String> {
    // Try to get the A (Action) dictionary
    if let Ok(action_ref) = annot_dict.get(b"A") {
        let action_dict = if let Ok(obj_ref) = action_ref.as_reference() {
            doc.get_dictionary(obj_ref).ok()
        } else {
            action_ref.as_dict().ok()
        };

        if let Some(action_dict) = action_dict {
            // Check for URI action. markitai: the string may be an indirect
            // object (Quartz writes `/URI 17 0 R`); it was read as no URI.
            if let Ok(uri_obj) = action_dict.get(b"URI") {
                let uri_obj = match uri_obj {
                    Object::Reference(id) => doc.get_object(*id).ok(),
                    direct => Some(direct),
                };
                if let Some(Ok(uri_str)) = uri_obj.map(Object::as_str) {
                    return Some(String::from_utf8_lossy(uri_str).to_string());
                }
            }
        }
    }

    // Try Dest (named destination) - less common for external links
    // We'll skip this for now as it requires looking up named destinations

    None
}

/// The check box state a form shows: `☒` checked, `☐` clear, as markitai
/// writes the check boxes of Word's and RTF's legacy form fields.
const CHECKED: &str = "\u{2612}";
const CLEAR: &str = "\u{2610}";

/// markitai: what one widget of a form field shows, before it is placed
/// among its page's text (see [`place_form_values`]).
pub(crate) struct FormValue {
    /// A [`ItemType::FormField`] item over the widget's rectangle, its
    /// text the value under the field's own label (`label: value`): its
    /// tooltip (`/TU`), else the last part of its name. That is how a
    /// value no page text labels is read.
    pub(crate) item: TextItem,
    /// The value alone: a text field's text, a choice field's choices, or
    /// a check box's or radio button's state (see [`CHECKED`]).
    value: String,
    /// Whether the widget is a check box or a radio button, labelled by
    /// the text beside it, usually on its right.
    check: bool,
}

/// The inheritable attributes of a field (ISO 32000-1 §12.7.3.1), and the
/// nearest name and tooltip, which belong to the field a widget is part of.
#[derive(Clone, Copy, Default)]
struct Inherited<'a> {
    kind: Option<&'a [u8]>,
    flags: i64,
    value: Option<&'a Object>,
    name: Option<&'a Object>,
    tooltip: Option<&'a Object>,
    options: Option<&'a Object>,
}

impl<'a> Inherited<'a> {
    fn under(self, field: &'a Dictionary) -> Self {
        Self {
            kind: field
                .get(b"FT")
                .ok()
                .and_then(|o| o.as_name().ok())
                .or(self.kind),
            flags: field
                .get(b"Ff")
                .ok()
                .and_then(|o| o.as_i64().ok())
                .unwrap_or(self.flags),
            value: field.get(b"V").ok().or(self.value),
            name: field.get(b"T").ok().or(self.name),
            tooltip: field.get(b"TU").ok().or(self.tooltip),
            options: field.get(b"Opt").ok().or(self.options),
        }
    }
}

/// Field flags (`/Ff`): a text field that hides what is typed, and a
/// button that is a push button (radio buttons and check boxes both show
/// their state in their appearance state, `/AS`).
const PASSWORD: i64 = 1 << 13;
const PUSH_BUTTON: i64 = 1 << 16;

fn decoded(doc: &Document, object: &Object) -> Option<String> {
    let object = match object {
        Object::Reference(id) => doc.get_object(*id).ok()?,
        direct => direct,
    };
    match object {
        Object::Stream(stream) => Some(crate::text_utils::decode_pdf_text_string(&stream_bytes(
            stream,
        )?)),
        other => other
            .as_str()
            .ok()
            .map(crate::text_utils::decode_pdf_text_string),
    }
}

/// The text a field's name ends with: the last part of a dotted name,
/// without an XFA index (`topmostSubform[0].Page1[0].f1_01[0]` is
/// `f1_01`).
fn name_tail(name: &str) -> &str {
    let tail = name.rsplit('.').next().unwrap_or(name).trim();
    match tail.strip_suffix(']').and_then(|t| t.rsplit_once('[')) {
        Some((stem, index)) if index.chars().all(|c| c.is_ascii_digit()) => stem,
        _ => tail,
    }
}

/// What a choice field shows for `value`: the display text `/Opt` pairs
/// with it as its export value, else the value itself.
fn choice_text(doc: &Document, options: Option<&Object>, value: String) -> String {
    let Some(options) = options.and_then(|o| resolve_array(doc, o)) else {
        return value;
    };
    for option in options {
        if let Some(pair) = resolve_array(doc, option).filter(|pair| pair.len() == 2) {
            if decoded(doc, &pair[0]).as_deref() == Some(value.as_str()) {
                if let Some(shown) = decoded(doc, &pair[1]).filter(|s| !s.trim().is_empty()) {
                    return shown;
                }
            }
        }
    }
    value
}

/// What a widget shows, and whether it is a check box or radio button;
/// `None` for a signature, a push button, a password, and an empty text
/// or choice field.
fn widget_value(
    doc: &Document,
    widget: &Dictionary,
    field: Inherited<'_>,
) -> Option<(String, bool)> {
    let value = field.value;
    match field.kind? {
        // A multi-line value reads on its label's line, its lines run on.
        b"Tx" if field.flags & PASSWORD == 0 => {
            let text = decoded(doc, value?)?;
            let text = text.split_whitespace().collect::<Vec<_>>().join(" ");
            (!text.is_empty()).then_some((text, false))
        }
        b"Ch" => {
            let shown: Vec<String> = match value? {
                Object::Array(choices) => choices
                    .iter()
                    .filter_map(|choice| decoded(doc, choice))
                    .collect(),
                Object::Reference(id) => match doc.get_object(*id).ok()? {
                    Object::Array(choices) => choices
                        .iter()
                        .filter_map(|choice| decoded(doc, choice))
                        .collect(),
                    single => decoded(doc, single).into_iter().collect(),
                },
                single => decoded(doc, single).into_iter().collect(),
            };
            let shown: Vec<String> = shown
                .into_iter()
                .map(|choice| choice.trim().to_string())
                .filter(|choice| !choice.is_empty())
                .map(|choice| choice_text(doc, field.options, choice))
                .collect();
            (!shown.is_empty()).then(|| (shown.join(", "), false))
        }
        b"Btn" if field.flags & PUSH_BUTTON == 0 => {
            // A widget's appearance state is its own: a radio button's names
            // the option it stands for when chosen, `Off` otherwise.
            let state = widget
                .get(b"AS")
                .ok()
                .and_then(|state| state.as_name().ok());
            let on = match state {
                Some(state) => state != b"Off",
                None => value
                    .and_then(|v| v.as_name().ok())
                    .is_some_and(|v| v != b"Off"),
            };
            Some((if on { CHECKED } else { CLEAR }.to_string(), true))
        }
        _ => None,
    }
}

/// Extract form field values from AcroForm dictionary: one
/// [`FormValue`] for each widget that shows one, on the page its `/P`
/// names, else the page whose `/Annots` holds it, else page 1.
pub(crate) fn extract_form_fields(
    doc: &Document,
    page_map: &HashMap<ObjectId, u32>,
) -> Vec<FormValue> {
    let mut items = Vec::new();

    // Navigate: trailer -> /Root -> /AcroForm -> /Fields
    let root = match doc.trailer.get(b"Root") {
        Ok(root_ref) => match root_ref.as_reference() {
            Ok(r) => match doc.get_dictionary(r) {
                Ok(d) => d,
                Err(_) => return items,
            },
            Err(_) => return items,
        },
        Err(_) => return items,
    };

    let acroform = match root.get(b"AcroForm") {
        Ok(obj) => match resolve_dict(doc, obj) {
            Some(d) => d,
            None => return items,
        },
        Err(_) => return items,
    };

    // Borrow the array rather than cloning it: a crafted `/Fields` can be huge,
    // and cloning would pay an O(n) allocation/copy before the budget check
    // below can stop the work.
    let fields = match acroform.get(b"Fields") {
        Ok(obj) => match resolve_array(doc, obj) {
            Some(arr) => arr,
            None => return items,
        },
        Err(_) => return items,
    };
    if fields.is_empty() {
        return items;
    }
    let annotation_pages = annotation_page_map(doc, page_map);

    // Bound the walk so a crafted PDF cannot send us into unbounded recursion
    // via a `/Kids` cycle, a deep chain, or an oversized array of invalid or
    // duplicate entries.
    let mut budget = FieldWalkBudget::new();

    for field_obj in fields {
        // Stop once the budget is spent so a `/Fields` array wider than the
        // budget can't burn CPU iterating entries whose walk would no-op. Charge
        // every entry (including invalid ones) against the budget.
        if budget.exhausted() {
            break;
        }
        budget.examined += 1;
        if let Ok(field_ref) = field_obj.as_reference() {
            walk_form_fields(
                doc,
                field_ref,
                Inherited::default(),
                page_map,
                &annotation_pages,
                &mut items,
                &mut budget,
                0,
            );
        }
    }

    items
}

/// Map widget annotation objects back to the page whose `/Annots` array owns
/// them. Some valid widgets omit `/P`, so the page tree is the only reliable
/// ownership signal available for page-filtered extraction.
fn annotation_page_map(
    doc: &Document,
    page_map: &HashMap<ObjectId, u32>,
) -> HashMap<ObjectId, u32> {
    let mut annotation_pages = HashMap::new();
    for (&page_id, &page_num) in page_map {
        let Some(annotations) = doc
            .get_dictionary(page_id)
            .ok()
            .and_then(|page| page.get(b"Annots").ok())
            .and_then(|annotations| resolve_array(doc, annotations))
        else {
            continue;
        };
        for annotation in annotations {
            if let Ok(annotation_id) = annotation.as_reference() {
                annotation_pages.insert(annotation_id, page_num);
            }
        }
    }
    annotation_pages
}

/// Recursively walk the form field tree, extracting what each widget shows.
/// A field's kind, flags, value and options pass down to its kids, as do
/// its name and tooltip to the widgets that have none of their own.
#[allow(clippy::too_many_arguments)]
fn walk_form_fields<'a>(
    doc: &'a Document,
    field_id: ObjectId,
    inherited: Inherited<'a>,
    page_map: &HashMap<ObjectId, u32>,
    annotation_pages: &HashMap<ObjectId, u32>,
    items: &mut Vec<FormValue>,
    budget: &mut FieldWalkBudget,
    depth: usize,
) {
    // Guard against `/Kids` cycles and pathologically large field trees.
    // Exceeding the depth cap means the chain is too deep to be a legitimate
    // form (and would overflow the stack); an exhausted budget means the tree is
    // too large. Both checks run *before* inserting so the visited set can never
    // grow past the budget.
    if depth > MAX_FORM_FIELD_DEPTH || budget.exhausted() {
        return;
    }
    // Revisiting an object ID means we hit a `/Kids` cycle.
    if !budget.visited.insert(field_id) {
        return;
    }

    let field_dict = match doc.get_dictionary(field_id) {
        Ok(d) => d,
        Err(_) => return,
    };
    let field = inherited.under(field_dict);

    // Check for /Kids — if present, recurse into children
    if let Ok(kids_obj) = field_dict.get(b"Kids") {
        // Iterate the borrowed array directly — cloning a crafted, oversized
        // `/Kids` would allocate and copy every entry before the budget check
        // below could stop the work.
        if let Some(kids) = resolve_array(doc, kids_obj) {
            for kid in kids {
                // Stop once the budget is spent so a `/Kids` array wider than the
                // budget can't burn CPU iterating entries whose walk would no-op.
                // Charge every entry (including invalid/duplicate ones) against
                // the budget so this is a true traversal-work cap.
                if budget.exhausted() {
                    break;
                }
                budget.examined += 1;
                if let Ok(kid_ref) = kid.as_reference() {
                    walk_form_fields(
                        doc,
                        kid_ref,
                        field,
                        page_map,
                        annotation_pages,
                        items,
                        budget,
                        depth + 1,
                    );
                }
            }
            return;
        }
    }

    // Leaf: a widget (or a field merged with its only widget).
    if annotation_hidden(field_dict) {
        return;
    }
    let Some((value, check)) = widget_value(doc, field_dict, field) else {
        return;
    };

    // Get Rect for positioning
    let rect = annotation_rect(field_dict).unwrap_or((0.0, 0.0, 0.0, 0.0));

    // Determine page number from /P reference
    let page_num = field_dict
        .get(b"P")
        .ok()
        .and_then(|o| o.as_reference().ok())
        .and_then(|p| page_map.get(&p).copied())
        .or_else(|| annotation_pages.get(&field_id).copied())
        .unwrap_or(1);

    // markitai: the field's own label is its tooltip, else the last part of
    // its name; a full name (`topmostSubform[0].Page1[0].f1_01[0]`) is no
    // label a reader knows.
    let label = field
        .tooltip
        .and_then(|tooltip| decoded(doc, tooltip))
        .map(|tooltip| tooltip.trim().to_string())
        .filter(|tooltip| !tooltip.is_empty())
        .or_else(|| {
            field
                .name
                .and_then(|name| decoded(doc, name))
                .map(|name| name_tail(&name).to_string())
        })
        .unwrap_or_default();
    let text = if label.is_empty() {
        value.clone()
    } else {
        format!("{label}: {value}")
    };

    items.push(FormValue {
        item: plain_item(text, rect, 0.0, page_num, ItemType::FormField),
        value,
        check,
    });
}

/// Whether `c` is a CJK character, after which a label takes a full-width
/// colon.
fn wide(c: char) -> bool {
    matches!(c as u32, 0x3040..=0x30FF | 0x3400..=0x9FFF | 0xAC00..=0xD7AF | 0xF900..=0xFAFF)
}

/// markitai: each form value placed among its page's text beside the text
/// that labels its widget, so it reads as `label: value` where the page
/// shows it. A widget's label is the nearest upright run on its row to its
/// left, with no other widget between, else the nearest run just above it
/// across its width; a check box or radio button is labelled first by the
/// run just to its right. The value continues the label's line: after a
/// colon, which a label ending in a letter, digit or bracket is given, or
/// for a check box after a space; a check box labelled on its right stands
/// at its own place before the label. Lines are read in the order of the
/// items, so a value is put among `items` right after its label (before
/// it, for a label on the right). A widget nothing labels keeps its own
/// rectangle and its field's own label (see [`FormValue::item`]), after
/// all the items.
pub(crate) fn place_form_values(values: Vec<FormValue>, items: &mut Vec<TextItem>) {
    if values.is_empty() {
        return;
    }
    let pages: HashSet<u32> = values.iter().map(|value| value.item.page).collect();
    let mut runs: HashMap<u32, Vec<usize>> = HashMap::new();
    for (index, item) in items.iter().enumerate() {
        if pages.contains(&item.page)
            && matches!(item.item_type, ItemType::Text)
            && item.rotation == 0.0
            && item.font_size > 1.0
            && item.width > 0.0
            && !item.text.trim().is_empty()
            && [item.x, item.y, item.width, item.font_size]
                .iter()
                .all(|n| n.is_finite())
        {
            runs.entry(item.page).or_default().push(index);
        }
    }
    // Each widget's rectangle, by page: another widget between a label and
    // a widget keeps that label to itself.
    let mut widgets: HashMap<u32, Vec<[f32; 4]>> = HashMap::new();
    for value in &values {
        let item = &value.item;
        widgets.entry(item.page).or_default().push(rect_of(item));
    }
    let mut before: HashMap<usize, Vec<TextItem>> = HashMap::new();
    let mut after: HashMap<usize, Vec<TextItem>> = HashMap::new();
    let mut unlabelled = Vec::new();
    for value in values {
        let page = value.item.page;
        let page_widgets = widgets.get(&page).map_or(&[][..], Vec::as_slice);
        let page_runs = runs.get(&page).map_or(&[][..], Vec::as_slice);
        // A page of more widgets or runs than any form has is read as it
        // always was, without the search for labels.
        let found = (page_widgets.len() <= MAX_PLACED_WIDGETS && page_runs.len() <= MAX_LABEL_RUNS)
            .then(|| label(&value, items, page_runs, page_widgets))
            .flatten();
        match found {
            Some((anchor, side)) => {
                let placed = place(value, &items[anchor], &side);
                match side {
                    Side::Right => before.entry(anchor).or_default().push(placed),
                    Side::Left | Side::Above => after.entry(anchor).or_default().push(placed),
                }
            }
            None => unlabelled.push(value.item),
        }
    }
    if !before.is_empty() || !after.is_empty() {
        let read = std::mem::take(items);
        items.reserve(read.len() + before.len() + after.len());
        for (index, item) in read.into_iter().enumerate() {
            if let Some(mut placed) = before.remove(&index) {
                crate::sort::stable(&mut placed, &mut |a, b| a.x.total_cmp(&b.x));
                items.extend(placed);
            }
            items.push(item);
            if let Some(mut placed) = after.remove(&index) {
                crate::sort::stable(&mut placed, &mut |a, b| a.x.total_cmp(&b.x));
                items.extend(placed);
            }
        }
    }
    items.extend(unlabelled);
}

/// The most widgets on a page whose values are placed by their labels, and
/// the most runs searched for those labels: the search compares every
/// widget with every run and every other widget.
const MAX_PLACED_WIDGETS: usize = 2_000;
const MAX_LABEL_RUNS: usize = 20_000;

/// An item's box as `[x0, y0, x1, y1]`.
fn rect_of(item: &TextItem) -> [f32; 4] {
    [item.x, item.y, item.x + item.width, item.y + item.height]
}

/// Where the label of a widget stands relative to it.
enum Side {
    Left,
    Above,
    Right,
}

/// The index among `items` of the run labelling a value's widget (see
/// [`place_form_values`]), one of `runs`, and on which side of it it
/// stands; `widgets` are the rectangles of the page's widgets.
fn label(
    value: &FormValue,
    items: &[TextItem],
    runs: &[usize],
    widgets: &[[f32; 4]],
) -> Option<(usize, Side)> {
    let rect = rect_of(&value.item);
    let [x0, y0, x1, y1] = rect;
    if !(x1 > x0 && y1 > y0) {
        return None;
    }
    // A run is on the widget's row when its baseline lies within the
    // widget's height, allowing for a box drawn on the baseline.
    let on_row = |run: &TextItem| {
        run.y >= y0 - run.font_size * 0.3 && run.y <= y1 - run.font_size.min(y1 - y0) * 0.3
    };
    let between = |from: f32, to: f32| {
        widgets.iter().any(|&other| {
            let [ox0, oy0, ox1, oy1] = other;
            other != rect && ox0 >= from - 1.0 && ox1 <= to + 1.0 && oy0 < y1 && oy1 > y0
        })
    };
    let candidates = || runs.iter().map(|&index| (index, &items[index]));
    let right = |run: &TextItem| run.x + run.width;
    let left = candidates()
        .filter(|(_, run)| on_row(run) && right(run) <= x0 + 1.0 && x0 - right(run) <= 300.0)
        .max_by(|(_, a), (_, b)| right(a).total_cmp(&right(b)))
        .filter(|(_, run)| !between(right(run), x0));
    let beside = || {
        candidates()
            .filter(|(_, run)| {
                on_row(run) && run.x >= x1 - 1.0 && run.x - x1 <= ((x1 - x0) * 2.5).max(24.0)
            })
            .min_by(|(_, a), (_, b)| a.x.total_cmp(&b.x))
            .filter(|(_, run)| !between(x1, run.x))
    };
    let above = || {
        candidates()
            .filter(|(_, run)| {
                run.y >= y1 - run.font_size * 0.3
                    && run.y - y1 <= run.font_size * 1.6
                    && run.x < x1
                    && right(run) > x0
            })
            .min_by(|(_, a), (_, b)| {
                a.y.total_cmp(&b.y)
                    .then((a.x - x0).abs().total_cmp(&(b.x - x0).abs()))
            })
    };
    if value.check {
        if let Some((index, _)) = beside() {
            return Some((index, Side::Right));
        }
    }
    left.map(|(index, _)| (index, Side::Left))
        .or_else(|| above().map(|(index, _)| (index, Side::Above)))
}

/// A value set on its label's line (see [`place_form_values`]).
fn place(value: FormValue, run: &TextItem, side: &Side) -> TextItem {
    let FormValue {
        mut item,
        value,
        check,
    } = value;
    let size = run.font_size;
    let em = |text: &str| text.chars().count() as f32 * size * 0.5;
    let end = run.x + run.width;
    let (text, x, width) = match side {
        // The box's own place, short of its label by a space.
        Side::Right => {
            let width = (run.x - size * 0.3 - item.x).clamp(1.0, item.width.max(1.0));
            (value, item.x, width)
        }
        Side::Left | Side::Above => match run.text.trim_end().chars().last() {
            Some(c) if !check && (c.is_alphanumeric() || matches!(c, ')' | ']')) => {
                let colon = if wide(c) { "\u{ff1a}" } else { ":" };
                let text = format!("{colon} {value}");
                let width = em(&text);
                (text, end, width)
            }
            _ => {
                let width = em(&value);
                (value, end + size * 0.3, width)
            }
        },
    };
    item.text = text;
    item.x = x;
    item.width = width;
    item.y = run.y;
    item.height = run.height;
    item.font_size = size;
    item
}

/// markitai: a page's annotation items added after its text, which starts
/// at `start` in `items`: link items at the end, as they always were, and
/// the text of its FreeText annotations (see [`free_text_items`]) where it
/// reads. Lines are read in the order of the items, so each line goes
/// before the first of the page's runs set below it across its width (else
/// below it anywhere, else after them all). Lines past the first
/// [`MAX_PLACED_ANNOTATION_LINES`] of a page follow its text.
pub(crate) fn add_page_annotations(
    items: &mut Vec<TextItem>,
    start: usize,
    annotations: Vec<TextItem>,
) {
    let (text, links): (Vec<TextItem>, Vec<TextItem>) = annotations
        .into_iter()
        .partition(|item| matches!(item.item_type, ItemType::Text));
    if text.is_empty() {
        items.extend(links);
        return;
    }
    let page = items.split_off(start.min(items.len()));
    let mut placed: Vec<(usize, TextItem)> = text
        .into_iter()
        .enumerate()
        .map(|(index, line)| {
            if index >= MAX_PLACED_ANNOTATION_LINES {
                return (page.len(), line);
            }
            let below =
                |item: &TextItem| matches!(item.item_type, ItemType::Text) && item.y < line.y - 1.0;
            let at = page
                .iter()
                .position(|item| {
                    below(item) && item.x < line.x + line.width && item.x + item.width > line.x
                })
                .or_else(|| page.iter().position(below))
                .unwrap_or(page.len());
            (at, line)
        })
        .collect();
    // Lines at one place keep their order.
    crate::sort::total(&mut placed, &mut |a, b| a.0.cmp(&b.0));
    let mut placed = placed.into_iter().peekable();
    items.reserve(page.len() + placed.len() + links.len());
    for (offset, item) in page.into_iter().enumerate() {
        while let Some((_, line)) = placed.next_if(|(at, _)| *at <= offset) {
            items.push(line);
        }
        items.push(item);
    }
    items.extend(placed.map(|(_, line)| line));
    items.extend(links);
}

/// The most FreeText lines of a page set among its text: each is compared
/// with every run of the page.
const MAX_PLACED_ANNOTATION_LINES: usize = 1_000;

#[cfg(test)]
mod tests {
    use super::*;
    use lopdf::{dictionary, Object};

    #[test]
    fn widget_without_page_reference_uses_owning_page_annotation() {
        let mut doc = Document::new();
        let widget_id = doc.add_object(dictionary! {
            "Type" => "Annot",
            "Subtype" => "Widget",
            "FT" => "Tx",
            "T" => Object::string_literal("customer"),
            "V" => Object::string_literal("Alice"),
            "Rect" => vec![10.into(), 20.into(), 110.into(), 40.into()],
        });
        let page_one_id = doc.add_object(dictionary! {
            "Type" => "Page",
        });
        let page_two_id = doc.add_object(dictionary! {
            "Type" => "Page",
            "Annots" => vec![Object::Reference(widget_id)],
        });
        let catalog_id = doc.add_object(dictionary! {
            "Type" => "Catalog",
            "AcroForm" => dictionary! {
                "Fields" => vec![Object::Reference(widget_id)],
            },
        });
        doc.trailer.set("Root", Object::Reference(catalog_id));

        let page_map = HashMap::from([(page_one_id, 1), (page_two_id, 2)]);
        let items = extract_form_fields(&doc, &page_map);

        assert_eq!(items.len(), 1);
        assert_eq!(items[0].item.page, 2);
        assert_eq!(items[0].item.text, "customer: Alice");
    }

    #[test]
    fn kids_self_cycle_does_not_overflow_stack() {
        // A crafted AcroForm field that lists itself in `/Kids` must not send
        // the traversal into unbounded recursion.
        let mut doc = Document::new();
        let field_id = doc.new_object_id();
        doc.set_object(
            field_id,
            dictionary! {
                "FT" => "Tx",
                "T" => Object::string_literal("loop"),
                "Kids" => vec![Object::Reference(field_id)],
            },
        );
        let catalog_id = doc.add_object(dictionary! {
            "Type" => "Catalog",
            "AcroForm" => dictionary! {
                "Fields" => vec![Object::Reference(field_id)],
            },
        });
        doc.trailer.set("Root", Object::Reference(catalog_id));

        let page_map = HashMap::new();
        // Completes (rather than overflowing the stack) and yields no items.
        let items = extract_form_fields(&doc, &page_map);
        assert!(items.is_empty());
    }

    #[test]
    fn kids_mutual_cycle_terminates() {
        // Two fields that reference each other via `/Kids` form a cycle that
        // must also terminate.
        let mut doc = Document::new();
        let field_a = doc.new_object_id();
        let field_b = doc.new_object_id();
        doc.set_object(
            field_a,
            dictionary! {
                "T" => Object::string_literal("a"),
                "Kids" => vec![Object::Reference(field_b)],
            },
        );
        doc.set_object(
            field_b,
            dictionary! {
                "T" => Object::string_literal("b"),
                "Kids" => vec![Object::Reference(field_a)],
            },
        );
        let catalog_id = doc.add_object(dictionary! {
            "Type" => "Catalog",
            "AcroForm" => dictionary! {
                "Fields" => vec![Object::Reference(field_a)],
            },
        });
        doc.trailer.set("Root", Object::Reference(catalog_id));

        let page_map = HashMap::new();
        let items = extract_form_fields(&doc, &page_map);
        assert!(items.is_empty());
    }

    #[test]
    fn deep_acyclic_kids_chain_does_not_overflow_stack() {
        // A long chain of *distinct* fields (no cycle) must also terminate:
        // the visited set alone would still recurse to the chain length, so
        // the depth cap is what prevents a stack overflow here.
        let mut doc = Document::new();
        let n = MAX_FORM_FIELD_DEPTH * 500;
        let ids: Vec<ObjectId> = (0..=n).map(|_| doc.new_object_id()).collect();
        for i in 0..n {
            doc.set_object(
                ids[i],
                dictionary! {
                    "FT" => "Tx",
                    "Kids" => vec![Object::Reference(ids[i + 1])],
                },
            );
        }
        // Leaf carries a value; it sits far below the depth cap so it is never
        // reached, proving traversal stops early rather than crashing.
        doc.set_object(
            ids[n],
            dictionary! {
                "FT" => "Tx",
                "T" => Object::string_literal("leaf"),
                "V" => Object::string_literal("x"),
                "Rect" => vec![10.into(), 20.into(), 110.into(), 40.into()],
            },
        );
        let catalog_id = doc.add_object(dictionary! {
            "Type" => "Catalog",
            "AcroForm" => dictionary! {
                "Fields" => vec![Object::Reference(ids[0])],
            },
        });
        doc.trailer.set("Root", Object::Reference(catalog_id));

        let page_map = HashMap::new();
        let items = extract_form_fields(&doc, &page_map);
        assert!(items.is_empty());
    }

    #[test]
    fn wide_tree_traversal_stops_at_node_budget() {
        // A single field with a `/Kids` array wider than the node budget must
        // stop traversal at the cap rather than growing `visited` (and the work)
        // without bound. Each processed leaf emits one item, so the item count
        // is bounded by the budget and reaches right up to it (a couple of
        // slots go to the root and the boundary node charged against the cap).
        let mut doc = Document::new();
        let fanout = MAX_FORM_FIELD_NODES + 50;
        let leaf_ids: Vec<ObjectId> = (0..fanout).map(|_| doc.new_object_id()).collect();
        for &leaf in &leaf_ids {
            doc.set_object(
                leaf,
                dictionary! {
                    "FT" => "Tx",
                    "V" => Object::string_literal("v"),
                    "Rect" => vec![10.into(), 20.into(), 110.into(), 40.into()],
                },
            );
        }
        let kids: Vec<Object> = leaf_ids.iter().map(|&id| Object::Reference(id)).collect();
        let root_id = doc.add_object(dictionary! {
            "T" => Object::string_literal("root"),
            "Kids" => kids,
        });
        let catalog_id = doc.add_object(dictionary! {
            "Type" => "Catalog",
            "AcroForm" => dictionary! {
                "Fields" => vec![Object::Reference(root_id)],
            },
        });
        doc.trailer.set("Root", Object::Reference(catalog_id));

        let page_map = HashMap::new();
        let items = extract_form_fields(&doc, &page_map);
        // Extraction stops at the budget: bounded above by the cap, and it gets
        // right up to it (allowing a small delta for the root/boundary nodes
        // charged against the budget).
        assert!(items.len() <= MAX_FORM_FIELD_NODES);
        assert!(items.len() >= MAX_FORM_FIELD_NODES - 3);
    }

    #[test]
    fn wide_top_level_fields_stop_at_node_budget() {
        // A top-level `/Fields` array wider than the budget must also stop at
        // the cap: the item count is bounded by the budget and reaches right up
        // to it.
        let mut doc = Document::new();
        let fanout = MAX_FORM_FIELD_NODES + 50;
        let leaf_ids: Vec<ObjectId> = (0..fanout).map(|_| doc.new_object_id()).collect();
        for &leaf in &leaf_ids {
            doc.set_object(
                leaf,
                dictionary! {
                    "FT" => "Tx",
                    "V" => Object::string_literal("v"),
                    "Rect" => vec![10.into(), 20.into(), 110.into(), 40.into()],
                },
            );
        }
        let fields: Vec<Object> = leaf_ids.iter().map(|&id| Object::Reference(id)).collect();
        let catalog_id = doc.add_object(dictionary! {
            "Type" => "Catalog",
            "AcroForm" => dictionary! {
                "Fields" => fields,
            },
        });
        doc.trailer.set("Root", Object::Reference(catalog_id));

        let page_map = HashMap::new();
        let items = extract_form_fields(&doc, &page_map);
        assert!(items.len() <= MAX_FORM_FIELD_NODES);
        assert!(items.len() >= MAX_FORM_FIELD_NODES - 3);
    }

    #[test]
    fn duplicate_and_invalid_kids_entries_stop_at_budget() {
        // Duplicate references and non-reference junk never grow `visited`, so
        // without charging examined entries against the budget an oversized
        // array of them would iterate to completion. The walk must still
        // terminate and extract the single real leaf exactly once.
        let mut doc = Document::new();
        let leaf_id = doc.new_object_id();
        doc.set_object(
            leaf_id,
            dictionary! {
                "FT" => "Tx",
                "V" => Object::string_literal("v"),
                "Rect" => vec![10.into(), 20.into(), 110.into(), 40.into()],
            },
        );
        // A `/Kids` array far wider than the budget: half duplicate references
        // to the same leaf, half invalid (null) entries.
        let mut kids: Vec<Object> = Vec::new();
        for i in 0..(MAX_FORM_FIELD_NODES * 2) {
            if i % 2 == 0 {
                kids.push(Object::Reference(leaf_id));
            } else {
                kids.push(Object::Null);
            }
        }
        let root_id = doc.add_object(dictionary! {
            "T" => Object::string_literal("root"),
            "Kids" => kids,
        });
        let catalog_id = doc.add_object(dictionary! {
            "Type" => "Catalog",
            "AcroForm" => dictionary! {
                "Fields" => vec![Object::Reference(root_id)],
            },
        });
        doc.trailer.set("Root", Object::Reference(catalog_id));

        let page_map = HashMap::new();
        let items = extract_form_fields(&doc, &page_map);
        assert_eq!(items.len(), 1);
    }

    /// markitai: one Letter page showing `lines` of 11pt Helvetica, each
    /// `(x, baseline, text)`, with `annotations` on it and those of them
    /// that are fields in the AcroForm; returns the saved file.
    fn page_with(
        lines: &[(i64, i64, &str)],
        annotations: Vec<Dictionary>,
        fields: &[usize],
    ) -> Vec<u8> {
        let mut doc = Document::with_version("1.7");
        let pages_id = doc.new_object_id();
        let page_id = doc.new_object_id();
        let font = doc.add_object(dictionary! {
            "Type" => "Font", "Subtype" => "Type1", "BaseFont" => "Helvetica",
            "Encoding" => "WinAnsiEncoding",
        });
        let mut content = String::new();
        for (x, y, text) in lines {
            content.push_str(&format!("BT /F1 11 Tf {x} {y} Td ({text}) Tj ET\n"));
        }
        let content = doc.add_object(lopdf::Stream::new(dictionary! {}, content.into_bytes()));
        let ids: Vec<ObjectId> = annotations
            .into_iter()
            .map(|mut annotation| {
                annotation.set("P", page_id);
                doc.add_object(annotation)
            })
            .collect();
        doc.objects.insert(
            page_id,
            dictionary! {
                "Type" => "Page", "Parent" => pages_id, "Contents" => content,
                "MediaBox" => vec![0.into(), 0.into(), 612.into(), 792.into()],
                "Resources" => dictionary! { "Font" => dictionary! { "F1" => font } },
                "Annots" => ids.iter().map(|&id| Object::Reference(id)).collect::<Vec<_>>(),
            }
            .into(),
        );
        doc.objects.insert(
            pages_id,
            dictionary! { "Type" => "Pages", "Kids" => vec![page_id.into()], "Count" => 1 }.into(),
        );
        let fields: Vec<Object> = fields.iter().map(|&i| Object::Reference(ids[i])).collect();
        let catalog = doc.add_object(dictionary! {
            "Type" => "Catalog", "Pages" => pages_id,
            "AcroForm" => dictionary! { "Fields" => fields },
        });
        doc.trailer.set("Root", catalog);
        let mut bytes = Vec::new();
        doc.save_to(&mut bytes).unwrap();
        bytes
    }

    fn markdown(bytes: &[u8]) -> String {
        crate::extract_pages_markdown_mem(bytes, None)
            .unwrap()
            .pages[0]
            .markdown
            .clone()
    }

    fn widget(rect: [i64; 4], entries: Dictionary) -> Dictionary {
        let mut widget = dictionary! {
            "Type" => "Annot", "Subtype" => "Widget",
            "Rect" => rect.iter().map(|&n| Object::Integer(n)).collect::<Vec<_>>(),
        };
        widget.extend(&entries);
        widget
    }

    #[test]
    fn a_form_value_reads_beside_the_text_that_labels_its_widget() {
        let bytes = page_with(
            &[
                (60, 712, "1. First name"),
                (60, 676, "Email:"),
                (220, 640, "I agree to the terms"),
                (60, 600, "Home phone"),
            ],
            vec![
                // Labelled on its left, under an internal name.
                widget(
                    [220, 706, 500, 726],
                    dictionary! {
                        "FT" => "Tx", "T" => Object::string_literal("topmostSubform[0].Page1[0].f1_01[0]"),
                        "TU" => Object::string_literal("Full legal first name"), "V" => Object::string_literal("Maria"),
                    },
                ),
                // A label with its own colon, and a UTF-16 value.
                widget(
                    [220, 670, 500, 690],
                    dictionary! {
                        "FT" => "Tx", "T" => Object::string_literal("f2"),
                        "V" => Object::String(b"\xFE\xFF\x00m\x00@\x00x\x00.\x00o\x00r\x00g".to_vec(), lopdf::StringFormat::Hexadecimal),
                    },
                ),
                // A check box labelled on its right.
                widget(
                    [200, 637, 214, 651],
                    dictionary! { "FT" => "Btn", "T" => Object::string_literal("cb"), "V" => "Yes", "AS" => "Yes" },
                ),
                // Labelled above, set apart by more than a line.
                widget(
                    [60, 560, 300, 590],
                    dictionary! { "FT" => "Tx", "T" => Object::string_literal("phone"), "V" => Object::string_literal("555 0100") },
                ),
            ],
            &[0, 1, 2, 3],
        );
        let markdown = markdown(&bytes);
        for expected in [
            "1. First name: Maria",
            "Email: m@x.org",
            "\u{2612} I agree to the terms",
            "Home phone: 555 0100",
        ] {
            assert!(markdown.contains(expected), "{expected:?} in {markdown:?}");
        }
        assert!(!markdown.contains("topmostSubform"), "{markdown}");
        // A label and its answer set apart from other lines are no heading.
        assert!(!markdown.contains('#'), "{markdown}");
    }

    #[test]
    fn a_widget_no_text_labels_keeps_its_field_label() {
        let doc_bytes = page_with(
            &[(60, 712, "A page heading set far from the field")],
            vec![
                widget(
                    [300, 300, 500, 320],
                    dictionary! {
                        "FT" => "Tx", "T" => Object::string_literal("topmostSubform[0].Page1[0].f1_02[0]"),
                        "V" => Object::string_literal("Garcia"),
                    },
                ),
                widget(
                    [300, 200, 500, 220],
                    dictionary! {
                        "FT" => "Tx", "T" => Object::string_literal("f3"), "TU" => Object::string_literal("Department"),
                        "V" => Object::string_literal("Engineering"),
                    },
                ),
            ],
            &[0, 1],
        );
        let markdown = markdown(&doc_bytes);
        assert!(markdown.contains("f1_02: Garcia"), "{markdown}");
        assert!(markdown.contains("Department: Engineering"), "{markdown}");
        assert!(!markdown.contains("topmostSubform"), "{markdown}");
    }

    #[test]
    fn field_values_pass_to_their_widgets_and_hidden_values_stay_hidden() {
        let mut doc = Document::with_version("1.7");
        let page = doc.add_object(dictionary! { "Type" => "Page" });
        let kid = |doc: &mut Document, state: &str, x: i64| {
            doc.add_object(dictionary! {
                "Type" => "Annot", "Subtype" => "Widget", "AS" => state, "P" => page,
                "Rect" => vec![x.into(), 100.into(), (x + 12).into(), 112.into()],
            })
        };
        let first = kid(&mut doc, "Off", 100);
        let second = kid(&mut doc, "Choice2", 200);
        let radio = doc.add_object(dictionary! {
            "FT" => "Btn", "Ff" => 1 << 15, "T" => Object::string_literal("employment"),
            "V" => "Choice2", "Kids" => vec![first.into(), second.into()],
        });
        let shared = doc.add_object(dictionary! {
            "Type" => "Annot", "Subtype" => "Widget", "P" => page,
            "Rect" => vec![100.into(), 300.into(), 200.into(), 320.into()],
        });
        let text = doc.add_object(dictionary! {
            "FT" => "Tx", "T" => Object::string_literal("name"), "V" => Object::string_literal("Ada"),
            "Kids" => vec![shared.into()],
        });
        let password = doc.add_object(dictionary! {
            "Type" => "Annot", "Subtype" => "Widget", "FT" => "Tx", "Ff" => 1 << 13, "P" => page,
            "T" => Object::string_literal("pin"), "V" => Object::string_literal("1234"),
            "Rect" => vec![100.into(), 400.into(), 200.into(), 420.into()],
        });
        let push = doc.add_object(dictionary! {
            "Type" => "Annot", "Subtype" => "Widget", "FT" => "Btn", "Ff" => 1 << 16, "P" => page,
            "T" => Object::string_literal("submit"), "Rect" => vec![100.into(), 500.into(), 200.into(), 520.into()],
        });
        let choice = doc.add_object(dictionary! {
            "Type" => "Annot", "Subtype" => "Widget", "FT" => "Ch", "P" => page,
            "T" => Object::string_literal("country"), "V" => Object::string_literal("DE"),
            "Opt" => vec![
                Object::Array(vec![Object::string_literal("FR"), Object::string_literal("France")]),
                Object::Array(vec![Object::string_literal("DE"), Object::string_literal("Germany")]),
            ],
            "Rect" => vec![100.into(), 600.into(), 200.into(), 620.into()],
        });
        let catalog = doc.add_object(dictionary! {
            "Type" => "Catalog",
            "AcroForm" => dictionary! {
                "Fields" => vec![radio.into(), text.into(), password.into(), push.into(), choice.into()],
            },
        });
        doc.trailer.set("Root", catalog);
        let values = extract_form_fields(&doc, &HashMap::from([(page, 1)]));
        let shown: Vec<&str> = values
            .iter()
            .map(|value| value.item.text.as_str())
            .collect();
        assert_eq!(
            shown,
            [
                "employment: \u{2610}",
                "employment: \u{2612}",
                "name: Ada",
                "country: Germany"
            ]
        );
        assert!(values[0].check && !values[2].check);
        assert_eq!(values[1].item.x, 200.0);
    }

    #[test]
    fn a_field_name_ends_in_its_last_part() {
        assert_eq!(name_tail("topmostSubform[0].Page1[0].f1_01[0]"), "f1_01");
        assert_eq!(name_tail("customer"), "customer");
        assert_eq!(name_tail("list[a]"), "list[a]");
    }

    #[test]
    fn free_text_annotations_are_read_as_page_text_where_they_stand() {
        let bytes = page_with(
            &[
                (60, 700, "The first paragraph of the page."),
                (60, 400, "The last paragraph of the page."),
            ],
            vec![
                dictionary! {
                    "Type" => "Annot", "Subtype" => "FreeText", "DA" => Object::string_literal("0 g /Helv 11 Tf"),
                    "Rect" => vec![60.into(), 520.into(), 400.into(), 580.into()],
                    "Contents" => Object::string_literal("Typed on the page\rin two lines"),
                },
                dictionary! {
                    "Type" => "Annot", "Subtype" => "FreeText", "F" => 2,
                    "Rect" => vec![60.into(), 300.into(), 400.into(), 340.into()],
                    "Contents" => Object::string_literal("A hidden box"),
                },
                dictionary! {
                    "Type" => "Annot", "Subtype" => "Text",
                    "Rect" => vec![500.into(), 700.into(), 516.into(), 716.into()],
                    "Contents" => Object::string_literal("A reviewer's note"),
                },
            ],
            &[],
        );
        let markdown = markdown(&bytes);
        let first = markdown.find("The first paragraph").unwrap();
        let typed = markdown.find("Typed on the page").expect(&markdown);
        let last = markdown.find("The last paragraph").unwrap();
        assert!(first < typed && typed < last, "{markdown}");
        assert!(markdown.contains("in two lines"), "{markdown}");
        assert!(!markdown.contains("hidden"), "{markdown}");
        assert!(!markdown.contains("reviewer"), "{markdown}");
    }

    #[test]
    fn rich_text_is_read_without_its_markup() {
        let doc = Document::with_version("1.7");
        let annotation = dictionary! {
            "RC" => Object::string_literal("<body><p>Fish &amp; chips</p><p>&#169; 2026</p></body>"),
        };
        assert_eq!(
            rich_text_plain(&doc, &annotation).as_deref(),
            Some("Fish & chips\n\u{a9} 2026")
        );
    }
}
