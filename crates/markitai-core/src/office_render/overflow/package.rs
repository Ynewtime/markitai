//! Preserve source bytes/relationships while adding an inert viewport carrier.
use super::{Extension, Result, anchor, check_deadline, failure, xml};
use std::{
    collections::{BTreeMap, BTreeSet},
    io::{Cursor, Read, Write},
    time::Instant,
};
const SHEET: &str = "http://schemas.openxmlformats.org/spreadsheetml/2006/main";
const REL: &str = "http://schemas.openxmlformats.org/officeDocument/2006/relationships";
const PACKAGE: &str = "http://schemas.openxmlformats.org/package/2006/relationships";
const DRAW: &str = "http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing";
const CONTENT: &str = "http://schemas.openxmlformats.org/package/2006/content-types";
const MAX_UNPACKED: u64 = 256 * 1024 * 1024;

pub(super) struct Package<'a> {
    archive: zip::ZipArchive<Cursor<&'a [u8]>>,
    pub(super) names: BTreeSet<String>,
    changed: BTreeMap<String, Vec<u8>>,
}
impl<'a> Package<'a> {
    pub(super) fn open(bytes: &'a [u8], deadline: Instant, limit: u64) -> Result<Self> {
        if bytes.len() as u64 > limit {
            return Err(failure("workbook repair input exceeds byte budget"));
        }
        let mut archive = zip::ZipArchive::new(Cursor::new(bytes))
            .map_err(|_| failure("invalid workbook ZIP package"))?;
        if archive.len() > 16_384 {
            return Err(failure("workbook ZIP exceeds entry limit"));
        }
        let mut names = BTreeSet::new();
        let mut total = 0u64;
        for i in 0..archive.len() {
            check_deadline(deadline)?;
            let file = archive
                .by_index(i)
                .map_err(|_| failure("invalid workbook ZIP entry"))?;
            let name = file.name();
            if name.is_empty()
                || name.starts_with('/')
                || name.contains(['\\', '\0', ':'])
                || name.split('/').any(|p| p == ".." || p == ".")
                || !names.insert(name.into())
            {
                return Err(failure("ambiguous or invalid workbook ZIP part name"));
            }
            if file
                .unix_mode()
                .is_some_and(|mode| !matches!(mode & 0o170000, 0 | 0o100000 | 0o040000))
            {
                return Err(failure("workbook ZIP contains a special file"));
            }
            total = total
                .checked_add(file.size())
                .filter(|v| *v <= MAX_UNPACKED)
                .ok_or_else(|| failure("workbook ZIP exceeds unpacked byte budget"))?;
        }
        Ok(Self {
            archive,
            names,
            changed: BTreeMap::new(),
        })
    }
    pub(super) fn read(&mut self, name: &str) -> Result<Vec<u8>> {
        if let Some(bytes) = self.changed.get(name) {
            return Ok(bytes.clone());
        }
        let file = self
            .archive
            .by_name(name)
            .map_err(|_| failure("required workbook XML part is missing"))?;
        if file.is_dir() || file.size() > xml::LIMIT as u64 {
            return Err(failure("workbook XML part exceeds limit"));
        }
        let size = file.size();
        let mut bytes = Vec::new();
        file.take(xml::LIMIT as u64 + 1).read_to_end(&mut bytes)?;
        if bytes.len() as u64 != size || bytes.len() > xml::LIMIT {
            return Err(failure("workbook XML part size mismatch"));
        }
        Ok(bytes)
    }
    pub(super) fn store(&mut self, name: String, bytes: Vec<u8>) {
        self.names.insert(name.clone());
        self.changed.insert(name, bytes);
    }
    pub(super) fn finish(mut self, deadline: Instant, limit: u64) -> Result<Vec<u8>> {
        let mut writer = zip::ZipWriter::new(Cursor::new(Vec::new()));
        for i in 0..self.archive.len() {
            check_deadline(deadline)?;
            let file = self
                .archive
                .by_index(i)
                .map_err(|_| failure("workbook ZIP entry cannot be copied"))?;
            if !self.changed.contains_key(file.name()) {
                writer
                    .raw_copy_file(file)
                    .map_err(|_| failure("workbook ZIP part could not be preserved"))?;
            }
        }
        for (name, bytes) in self.changed {
            check_deadline(deadline)?;
            writer
                .start_file(
                    name,
                    zip::write::SimpleFileOptions::default()
                        .compression_method(zip::CompressionMethod::Deflated),
                )
                .map_err(|_| failure("workbook ZIP part could not be added"))?;
            writer.write_all(&bytes)?;
        }
        let bytes = writer
            .finish()
            .map_err(|_| failure("workbook ZIP could not be completed"))?
            .into_inner();
        if bytes.len() as u64 > limit {
            return Err(failure("repaired workbook exceeds remaining byte budget"));
        }
        Ok(bytes)
    }
}
struct Relationship {
    target: String,
    kind: String,
    external: bool,
}
fn relationships(bytes: &[u8], deadline: Instant) -> Result<BTreeMap<String, Relationship>> {
    let nodes = xml::index(bytes, deadline)?;
    if nodes[0].namespace != PACKAGE || nodes[0].local != "Relationships" {
        return Err(failure("invalid workbook relationship root"));
    }
    let mut map = BTreeMap::new();
    for n in nodes.iter().filter(|n| n.depth == 1) {
        if n.namespace != PACKAGE || n.local != "Relationship" {
            continue;
        }
        let id = n
            .attr("Id")
            .filter(|id| !id.is_empty())
            .ok_or_else(|| failure("workbook relationship has no id"))?;
        let relation = Relationship {
            target: n
                .attr("Target")
                .ok_or_else(|| failure("workbook relationship has no target"))?
                .into(),
            kind: n
                .attr("Type")
                .ok_or_else(|| failure("workbook relationship has no type"))?
                .into(),
            external: n.attr("TargetMode").is_some_and(|mode| mode != "Internal"),
        };
        if map.insert(id.into(), relation).is_some() {
            return Err(failure("duplicate workbook relationship id"));
        }
    }
    Ok(map)
}
fn target(source: &str, relation: &Relationship, kind: &str) -> Result<String> {
    if relation.external || relation.kind != format!("{REL}/{kind}") {
        return Err(failure(
            "unsupported or external workbook layout relationship",
        ));
    }
    let mut decoded = Vec::new();
    let bytes = relation.target.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            let pair = bytes
                .get(i + 1..i + 3)
                .ok_or_else(|| failure("invalid encoded workbook relationship"))?;
            let value = std::str::from_utf8(pair)
                .ok()
                .and_then(|s| u8::from_str_radix(s, 16).ok())
                .ok_or_else(|| failure("invalid encoded workbook relationship"))?;
            decoded.push(value);
            i += 3;
        } else {
            decoded.push(bytes[i]);
            i += 1;
        }
    }
    let value = std::str::from_utf8(&decoded)
        .map_err(|_| failure("invalid workbook relationship target"))?;
    if value.is_empty() || value.contains(['\\', '\0', ':', '?', '#']) {
        return Err(failure("invalid workbook relationship target"));
    }
    let mut path = if value.starts_with('/') {
        Vec::new()
    } else {
        source
            .rsplit_once('/')
            .map_or(Vec::new(), |(dir, _)| dir.split('/').collect::<Vec<_>>())
    };
    for component in value.trim_start_matches('/').split('/') {
        match component {
            "" | "." => return Err(failure("ambiguous workbook relationship target")),
            ".." => {
                path.pop()
                    .ok_or_else(|| failure("workbook relationship leaves package"))?;
            }
            _ => path.push(component),
        }
    }
    Ok(path.join("/"))
}
fn rel_name(part: &str) -> String {
    part.rsplit_once('/').map_or_else(
        || format!("_rels/{part}.rels"),
        |(dir, name)| format!("{dir}/_rels/{name}.rels"),
    )
}
fn unique(prefix: &str, suffix: &str, used: &BTreeSet<String>) -> Result<String> {
    (1..=16_384)
        .map(|n| format!("{prefix}{n}{suffix}"))
        .find(|s| !used.contains(s))
        .ok_or_else(|| failure("workbook has no free layout part name"))
}
fn shape(id: u32, name: &str, right: f64, anchor: anchor::Anchor) -> Result<String> {
    let extent = ((right - 1.) * 12700.).ceil();
    if !extent.is_finite() || !(12700. ..=i32::MAX as f64).contains(&extent) {
        return Err(failure("workbook drawing coordinate exceeds bounded range"));
    }
    let width = extent as i64;
    let (col, row) = (anchor.col, anchor.row);
    // Small offsets stay inside the existing visible cell. The extent carries
    // the required width; Calc clamps large cell offsets to the next boundary.
    // If an earlier chart defines the canvas origin this adds bounded whitespace.
    Ok(format!(
        r#"<mt:oneCellAnchor xmlns:mt="{DRAW}" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><mt:from><mt:col>{col}</mt:col><mt:colOff>12700</mt:colOff><mt:row>{row}</mt:row><mt:rowOff>12700</mt:rowOff></mt:from><mt:ext cx="{width}" cy="12700"/><mt:sp><mt:nvSpPr><mt:cNvPr id="{id}" name="{name}" hidden="0"/><mt:cNvSpPr/></mt:nvSpPr><mt:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{width}" cy="12700"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/><a:ln><a:noFill/></a:ln><a:effectLst/></mt:spPr></mt:sp><mt:clientData fPrintsWithSheet="1"/></mt:oneCellAnchor>"#
    ))
}
pub(super) fn rewrite(
    bytes: &[u8],
    extensions: &[Extension],
    deadline: Instant,
    limit: u64,
) -> Result<Vec<u8>> {
    check_deadline(deadline)?;
    if extensions.is_empty() {
        return Ok(bytes.to_vec());
    }
    let mut package = Package::open(bytes, deadline, limit)?;
    let book = package.read("xl/workbook.xml")?;
    let book_nodes = xml::index(&book, deadline)?;
    if book_nodes[0].namespace != SHEET || book_nodes[0].local != "workbook" {
        return Err(failure(
            "only transitional OOXML workbooks support overflow repair",
        ));
    }
    let relations = relationships(&package.read("xl/_rels/workbook.xml.rels")?, deadline)?;
    let containers = book_nodes
        .iter()
        .filter(|n| n.namespace == SHEET && n.local == "sheets" && n.depth == 1)
        .collect::<Vec<_>>();
    if containers.len() != 1 {
        return Err(failure("invalid workbook sheet container"));
    }
    let container = containers[0];
    let sheets = book_nodes
        .iter()
        .filter(|n| {
            n.namespace == SHEET
                && n.local == "sheet"
                && n.depth == 2
                && n.start >= container.open_end
                && n.end <= container.close_start
        })
        .collect::<Vec<_>>();
    if sheets.is_empty() || sheets.len() > super::super::MAX_PAGES {
        return Err(failure("invalid workbook repair sheet count"));
    }
    // Detect aliases across every sheet, including those not being repaired.
    // Editing a shared drawing must not silently change a second worksheet.
    let mut seen_sheets = BTreeSet::new();
    let mut all_drawings = BTreeSet::new();
    for sheet in &sheets {
        let id = sheet
            .attribute(REL, "id")
            .ok_or_else(|| failure("workbook sheet relationship is missing"))?;
        let part = target(
            "xl/workbook.xml",
            relations
                .get(id)
                .ok_or_else(|| failure("workbook sheet relationship is missing"))?,
            "worksheet",
        )?;
        if !seen_sheets.insert(part.clone()) {
            return Err(failure("shared worksheet part cannot be safely expanded"));
        }
        let nodes = xml::index(&package.read(&part)?, deadline)?;
        for drawing in nodes
            .iter()
            .filter(|n| n.depth == 1 && n.namespace == SHEET && n.local == "drawing")
        {
            let id = drawing
                .attribute(REL, "id")
                .ok_or_else(|| failure("worksheet drawing relationship is missing"))?;
            let relations = relationships(&package.read(&rel_name(&part))?, deadline)?;
            let drawing_part = target(
                &part,
                relations
                    .get(id)
                    .ok_or_else(|| failure("worksheet drawing relationship is missing"))?,
                "drawing",
            )?;
            if !all_drawings.insert(drawing_part) {
                return Err(failure(
                    "shared worksheet drawing cannot be safely expanded",
                ));
            }
        }
    }
    let mut shared = None;
    let mut planned = BTreeSet::new();
    let mut used_drawings = BTreeSet::new();
    for extension in extensions {
        check_deadline(deadline)?;
        if !planned.insert(extension.page) || extension.page == 0 || extension.page > sheets.len() {
            return Err(failure("duplicate or invalid workbook repair page"));
        }
        super::dimensions(extension.right, extension.height)?;
        let sheet = sheets[extension.page - 1];
        let id = sheet
            .attribute(REL, "id")
            .ok_or_else(|| failure("workbook sheet relationship is missing"))?;
        let part = target(
            "xl/workbook.xml",
            relations
                .get(id)
                .ok_or_else(|| failure("workbook sheet relationship is missing"))?,
            "worksheet",
        )?;
        let sheet_bytes = package.read(&part)?;
        let nodes = xml::index(&sheet_bytes, deadline)?;
        if shared.is_none()
            && nodes
                .iter()
                .any(|n| n.namespace == SHEET && n.local == "c" && n.attr("t") == Some("s"))
        {
            let mut sources = relations
                .values()
                .filter(|r| r.kind == format!("{REL}/sharedStrings"));
            let relation = sources
                .next()
                .ok_or_else(|| failure("shared string relationship missing"))?;
            if sources.next().is_some() {
                return Err(failure("ambiguous shared string relationship"));
            }
            let part = target("xl/workbook.xml", relation, "sharedStrings")?;
            shared = Some(anchor::shared_nonempty(&package.read(&part)?, deadline)?);
        }
        let anchor = anchor::select(
            &nodes,
            &sheet_bytes,
            shared.as_deref().unwrap_or(&[]),
            deadline,
        )?;
        let drawings = nodes
            .iter()
            .filter(|n| n.depth == 1 && n.namespace == SHEET && n.local == "drawing")
            .collect::<Vec<_>>();
        if drawings.len() > 1 {
            return Err(failure("worksheet has multiple drawing references"));
        }
        let relation_part = rel_name(&part);
        if let Some(drawing) = drawings.first() {
            let id = drawing
                .attribute(REL, "id")
                .ok_or_else(|| failure("worksheet drawing relationship is missing"))?;
            let relations = relationships(&package.read(&relation_part)?, deadline)?;
            let drawing_part = target(
                &part,
                relations
                    .get(id)
                    .ok_or_else(|| failure("worksheet drawing relationship is missing"))?,
                "drawing",
            )?;
            if !used_drawings.insert(drawing_part.clone()) {
                return Err(failure(
                    "shared worksheet drawing cannot be safely expanded",
                ));
            }
            let original = package.read(&drawing_part)?;
            let nodes = xml::index(&original, deadline)?;
            if nodes[0].namespace != DRAW || nodes[0].local != "wsDr" {
                return Err(failure("unsupported spreadsheet drawing namespace"));
            }
            let mut ids = BTreeSet::new();
            let mut names = BTreeSet::new();
            for n in nodes
                .iter()
                .filter(|n| n.namespace == DRAW && n.local == "cNvPr")
            {
                if let Some(name) = n.attr("name") {
                    names.insert(name.to_owned());
                }
                let id = n
                    .attr("id")
                    .and_then(|s| s.parse::<u32>().ok())
                    .ok_or_else(|| failure("invalid drawing object id"))?;
                if !ids.insert(id) {
                    return Err(failure("duplicate drawing object id"));
                }
            }
            let id = ids
                .last()
                .copied()
                .unwrap_or(0)
                .checked_add(1)
                .ok_or_else(|| failure("drawing object id exhausted"))?;
            let name = unique("Markitai layout bounds ", "", &names)?;
            let changed = xml::append(
                &original,
                &nodes[0],
                &shape(id, &name, extension.right, anchor)?,
            )?;
            package.store(drawing_part, changed);
        } else {
            let drawing_part = unique("xl/drawings/markitai-bounds-", ".xml", &package.names)?;
            let original = if package.names.contains(&relation_part) {
                package.read(&relation_part)?
            } else {
                format!(r#"<Relationships xmlns="{PACKAGE}"/>"#).into_bytes()
            };
            let relations = relationships(&original, deadline)?;
            let used = relations.keys().cloned().collect();
            let id = unique("rIdMarkitaiBounds", "", &used)?;
            let relation = format!(
                r#"<Relationship xmlns="{PACKAGE}" Id="{id}" Type="{REL}/drawing" Target="/{drawing_part}"/>"#
            );
            let index = xml::index(&original, deadline)?;
            package.store(relation_part, xml::append(&original, &index[0], &relation)?);
            let marker = format!(r#"<drawing xmlns="{SHEET}" xmlns:r="{REL}" r:id="{id}"/>"#);
            // OOXML drawing precedes these optional trailing worksheet children.
            let insertion = nodes
                .iter()
                .find(|n| {
                    n.depth == 1
                        && n.namespace == SHEET
                        && matches!(
                            n.local.as_str(),
                            "legacyDrawing"
                                | "legacyDrawingHF"
                                | "drawingHF"
                                | "picture"
                                | "oleObjects"
                                | "controls"
                                | "webPublishItems"
                                | "tableParts"
                                | "extLst"
                        )
                })
                .map_or(nodes[0].close_start, |n| n.start);
            let changed = if nodes[0].empty {
                xml::append(&sheet_bytes, &nodes[0], &marker)?
            } else {
                xml::insert(&sheet_bytes, insertion, &marker)?
            };
            package.store(part, changed);
            package.store(
                drawing_part.clone(),
                format!(
                    r#"<mt:wsDr xmlns:mt="{DRAW}">{}</mt:wsDr>"#,
                    shape(1, "Markitai layout bounds 1", extension.right, anchor)?
                )
                .into_bytes(),
            );
            let original = package.read("[Content_Types].xml")?;
            let index = xml::index(&original, deadline)?;
            if index[0].namespace != CONTENT || index[0].local != "Types" {
                return Err(failure("invalid workbook content-type root"));
            }
            if index.iter().any(|n| {
                n.namespace == CONTENT && n.attr("PartName") == Some(&format!("/{drawing_part}"))
            }) {
                return Err(failure("ambiguous workbook drawing content type"));
            }
            let addition = format!(
                r#"<Override xmlns="{CONTENT}" PartName="/{drawing_part}" ContentType="application/vnd.openxmlformats-officedocument.drawing+xml"/>"#
            );
            package.store(
                "[Content_Types].xml".into(),
                xml::append(&original, &index[0], &addition)?,
            );
        }
    }
    package.finish(deadline, limit)
}
