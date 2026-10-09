//! Deliberate import policy, not an assertion that OOXML mandates missing color = black.
//! Only simple, theme-free models qualify; unknown inheritance keeps the original.
use super::{Result, check_deadline, failure, package::Package, xml};
use std::time::Instant;

const SHEET: &str = "http://schemas.openxmlformats.org/spreadsheetml/2006/main";
const REL: &str = "http://schemas.openxmlformats.org/officeDocument/2006/relationships";
const PACKAGE: &str = "http://schemas.openxmlformats.org/package/2006/relationships";
const COLOR: &str =
    "<color xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\" rgb=\"FF000000\"/>";

#[cfg(test)]
mod tests;

fn descendants<'a>(nodes: &'a [xml::Node], parent: &xml::Node) -> &'a [xml::Node] {
    let start = nodes.partition_point(|n| n.start < parent.open_end);
    let end = nodes.partition_point(|n| n.start < parent.close_start);
    &nodes[start..end]
}
fn children<'a>(nodes: &'a [xml::Node], parent: &xml::Node) -> Vec<&'a xml::Node> {
    descendants(nodes, parent)
        .iter()
        .filter(|n| n.depth == parent.depth + 1)
        .collect()
}
fn count_matches(node: &xml::Node, count: usize) -> bool {
    node.attr("count")
        .is_none_or(|value| value.parse::<usize>().ok() == Some(count))
}
fn container<'a>(nodes: &'a [xml::Node], name: &str) -> Option<&'a xml::Node> {
    let mut matching = nodes.iter().filter(|n| n.depth == 1 && n.local == name);
    let first = matching.next()?;
    matching.next().is_none().then_some(first)
}
fn index(node: &xml::Node, name: &str, default: usize, count: usize) -> Option<usize> {
    let value = node
        .attr(name)
        .map_or(Some(default), |value| value.parse::<usize>().ok())?;
    (value < count).then_some(value)
}
fn unsupported(nodes: &[xml::Node]) -> bool {
    nodes.iter().any(|n| {
        n.local == "AlternateContent"
            || (n.namespace == SHEET
                && (matches!(
                    n.local.as_str(),
                    "conditionalFormatting" | "dxf" | "r" | "extLst" | "tableParts" | "tableStyle"
                ) || n.attr("theme").is_some()))
    })
}

// Encoding support is a policy eligibility check, not a new workbook validity rule.
// Do not transcode: the installed renderer still receives unsupported encodings unchanged.
fn unsupported_encoding(bytes: &[u8], deadline: Instant) -> Result<bool> {
    if bytes.starts_with(&[0xff, 0xfe])
        || bytes.starts_with(&[0xfe, 0xff])
        || bytes.starts_with(&[0x00, 0x00, 0xfe, 0xff])
        || matches!(
            bytes.get(..4),
            Some(b"<\0?\0" | b"\0<\0?" | b"<\0\0\0" | b"\0\0\0<")
        )
    {
        return Ok(true);
    }
    let bytes = bytes.strip_prefix(b"\xef\xbb\xbf").unwrap_or(bytes);
    if !bytes.starts_with(b"<?xml") {
        return Ok(false);
    }
    let mut reader = quick_xml::Reader::from_reader(bytes);
    let event = reader
        .read_event()
        .map_err(|_| failure("invalid workbook XML declaration"))?;
    let quick_xml::events::Event::Decl(declaration) = event else {
        return Ok(false);
    };
    declaration
        .xml_version()
        .map_err(|_| failure("invalid workbook XML declaration"))?;
    let content: &str = &declaration;
    let declaration = quick_xml::events::BytesStart::from_content(content, 3);
    let mut unsupported = false;
    for attribute in declaration.attributes() {
        check_deadline(deadline)?;
        let attribute = attribute.map_err(|_| failure("invalid workbook XML declaration"))?;
        if attribute.key.into_inner() == "encoding" {
            let value = attribute.value.as_bytes();
            if !value.first().is_some_and(u8::is_ascii_alphabetic)
                || !value
                    .iter()
                    .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
            {
                return Err(failure("invalid workbook XML encoding declaration"));
            }
            unsupported |= !value.eq_ignore_ascii_case(b"UTF-8");
        }
    }
    Ok(unsupported)
}

// Unknown XML parts may legally use encodings outside this UTF-8 index's contract.
// Decline the optional policy before parsing them; malformed known roles stay errors.
fn known_xml_role(name: &str) -> bool {
    if matches!(
        name,
        "[Content_Types].xml"
            | "_rels/.rels"
            | "xl/workbook.xml"
            | "xl/_rels/workbook.xml.rels"
            | "xl/styles.xml"
            | "xl/sharedStrings.xml"
    ) {
        return true;
    }
    ["xl/worksheets/", "xl/charts/", "xl/drawings/"]
        .iter()
        .any(|prefix| {
            name.strip_prefix(prefix).is_some_and(|rest| {
                if let Some(relation) = rest.strip_prefix("_rels/") {
                    !relation.contains('/') && relation.ends_with(".xml.rels")
                } else {
                    !rest.contains('/') && rest.ends_with(".xml")
                }
            })
        })
}
fn xml_part(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower.ends_with(".xml") || lower.ends_with(".rels")
}

pub(super) fn normalize(bytes: &[u8], deadline: Instant, limit: u64) -> Result<Option<Vec<u8>>> {
    check_deadline(deadline)?;
    let mut package = Package::open(bytes, deadline, limit)?;
    if !package.names.contains("xl/styles.xml")
        || package.names.iter().any(|name| {
            let lower = name.to_ascii_lowercase();
            lower.contains("/theme") || lower.starts_with("_xmlsignatures/")
        })
    {
        return Ok(None);
    }
    if package
        .names
        .iter()
        .any(|name| xml_part(name) && !known_xml_role(name))
    {
        return Ok(None);
    }
    let styles = package.read("xl/styles.xml")?;
    if unsupported_encoding(&styles, deadline)? {
        return Ok(None);
    }
    let nodes = xml::index(&styles, deadline)?;
    if nodes[0].namespace != SHEET
        || nodes[0].local != "styleSheet"
        || nodes.iter().any(|n| n.namespace != SHEET)
        || unsupported(&nodes)
    {
        return Ok(None);
    }
    let Some(fonts_node) = container(&nodes, "fonts") else {
        return Ok(None);
    };
    if children(&nodes, &nodes[0]).iter().any(|n| {
        !matches!(
            n.local.as_str(),
            "numFmts"
                | "fonts"
                | "fills"
                | "borders"
                | "cellStyleXfs"
                | "cellXfs"
                | "cellStyles"
                | "dxfs"
                | "tableStyles"
                | "colors"
        )
    }) {
        return Ok(None);
    }
    let fonts = children(&nodes, fonts_node);
    if fonts.is_empty()
        || fonts.len() > 4096
        || !count_matches(fonts_node, fonts.len())
        || fonts.iter().any(|n| n.local != "font")
    {
        return Ok(None);
    }
    // Parse simple font declarations; presence of *any* color element protects it.
    let mut missing = Vec::new();
    for font in &fonts {
        check_deadline(deadline)?;
        let parts = children(&nodes, font);
        if parts.iter().any(|n| {
            !matches!(
                n.local.as_str(),
                "name"
                    | "charset"
                    | "family"
                    | "sz"
                    | "b"
                    | "i"
                    | "strike"
                    | "outline"
                    | "shadow"
                    | "condense"
                    | "extend"
                    | "u"
                    | "vertAlign"
                    | "color"
            )
        }) || descendants(&nodes, font)
            .iter()
            .any(|n| n.depth > font.depth + 1)
        {
            return Ok(None);
        }
        if !parts.iter().any(|n| n.local == "color") {
            missing.push(*font);
        }
    }
    if missing.is_empty() {
        return Ok(None);
    }
    let Some(parent_node) = container(&nodes, "cellStyleXfs") else {
        return Ok(None);
    };
    let Some(cell_node) = container(&nodes, "cellXfs") else {
        return Ok(None);
    };
    let parents = children(&nodes, parent_node);
    let cells = children(&nodes, cell_node);
    if parents.is_empty()
        || cells.is_empty()
        || !count_matches(parent_node, parents.len())
        || !count_matches(cell_node, cells.len())
        || parents.iter().chain(&cells).any(|n| n.local != "xf")
    {
        return Ok(None);
    }
    if parents.iter().chain(&cells).any(|xf| {
        children(&nodes, xf)
            .iter()
            .any(|n| !matches!(n.local.as_str(), "alignment" | "protection"))
    }) {
        return Ok(None);
    }
    let Some(parent_fonts) = parents
        .iter()
        .map(|n| index(n, "fontId", 0, fonts.len()))
        .collect::<Option<Vec<_>>>()
    else {
        return Ok(None);
    };
    for cell in &cells {
        let (Some(font), Some(parent)) = (
            index(cell, "fontId", 0, fonts.len()),
            index(cell, "xfId", 0, parents.len()),
        ) else {
            return Ok(None);
        };
        if !matches!(
            cell.attr("applyFont"),
            None | Some("0" | "1" | "false" | "true")
        ) || (font != parent_fonts[parent]
            && !matches!(cell.attr("applyFont"), Some("1" | "true")))
        {
            return Ok(None);
        }
    }
    if nodes
        .iter()
        .filter(|n| n.local == "cellStyle")
        .any(|n| index(n, "xfId", 0, parents.len()).is_none())
    {
        return Ok(None);
    }
    // Validate only the recognized XML roles with existing part/index/deadline bounds.
    // Rich shared strings, worksheet defaults, and theme relationships remain inspected.
    let names: Vec<_> = package
        .names
        .iter()
        .filter(|name| name.as_str() != "xl/styles.xml" && xml_part(name))
        .cloned()
        .collect();
    let mut canonical_styles = false;
    for name in names {
        check_deadline(deadline)?;
        let data = package.read(&name)?;
        if unsupported_encoding(&data, deadline)? {
            return Ok(None);
        }
        let part = xml::index(&data, deadline)?;
        if unsupported(&part) {
            return Ok(None);
        }
        let lower = name.to_ascii_lowercase();
        if (lower == "xl/workbook.xml"
            || (lower.starts_with("xl/worksheets/") && lower.ends_with(".xml"))
            || lower == "xl/sharedstrings.xml")
            && part.iter().any(|n| n.namespace != SHEET)
        {
            return Ok(None);
        }
        if lower.ends_with(".rels")
            && (part[0].namespace != PACKAGE || part[0].local != "Relationships")
        {
            return Ok(None);
        }
        for n in &part {
            if n.namespace == PACKAGE && n.local == "Relationship" {
                let Some(kind) = n.attr("Type") else {
                    return Ok(None);
                };
                if kind.ends_with("/theme") || kind.ends_with("/themeOverride") {
                    return Ok(None);
                }
                if kind == format!("{REL}/styles") {
                    if canonical_styles
                        || name != "xl/_rels/workbook.xml.rels"
                        || !matches!(n.attr("Target"), Some("styles.xml" | "/xl/styles.xml"))
                        || !matches!(n.attr("TargetMode"), None | Some("Internal"))
                    {
                        return Ok(None);
                    }
                    canonical_styles = true;
                }
            }
            if n.namespace == SHEET {
                let attribute = match n.local.as_str() {
                    "c" | "row" => Some("s"),
                    "col" => Some("style"),
                    _ => None,
                };
                if let Some(attribute) = attribute
                    && n.attr(attribute).is_some()
                    && index(n, attribute, 0, cells.len()).is_none()
                {
                    return Ok(None);
                }
            }
        }
    }
    if !canonical_styles {
        return Ok(None);
    }
    // Bound the repeated closing QName needed only for empty font elements.
    if missing.iter().any(|font| {
        font.empty
            && styles[font.start..font.open_end]
                .iter()
                .take_while(|b| !b.is_ascii_whitespace() && !matches!(b, b'/' | b'>'))
                .count()
                > 128
    }) {
        return Ok(None);
    }
    // Single pass insertion, avoiding font-count × XML-size allocation/copy amplification.
    let capacity = styles
        .len()
        .checked_add(missing.len() * (COLOR.len() + 256))
        .filter(|size| *size <= xml::LIMIT)
        .ok_or_else(|| failure("normalized workbook styles exceed XML byte budget"))?;
    let mut changed = Vec::with_capacity(capacity);
    let mut from = 0;
    for font in missing {
        check_deadline(deadline)?;
        if font.empty {
            let opening = &styles[font.start..font.open_end];
            let name_end = opening[1..]
                .iter()
                .position(|b| b.is_ascii_whitespace() || matches!(b, b'/' | b'>'))
                .map(|n| n + 1)
                .ok_or_else(|| failure("invalid workbook font element"))?;
            changed.extend_from_slice(&styles[from..font.open_end - 2]);
            changed.push(b'>');
            changed.extend_from_slice(COLOR.as_bytes());
            changed.extend_from_slice(b"</");
            changed.extend_from_slice(&opening[1..name_end]);
            changed.push(b'>');
            from = font.end;
        } else {
            changed.extend_from_slice(&styles[from..font.close_start]);
            changed.extend_from_slice(COLOR.as_bytes());
            from = font.close_start;
        }
    }
    changed.extend_from_slice(&styles[from..]);
    if changed.len() > xml::LIMIT {
        return Err(failure("normalized workbook styles exceed XML byte budget"));
    }
    package.store("xl/styles.xml".into(), changed);
    Ok(Some(package.finish(deadline, limit)?))
}
