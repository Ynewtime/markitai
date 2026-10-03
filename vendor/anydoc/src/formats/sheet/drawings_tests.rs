use super::*;
use crate::package::xml::parse_xml;
use std::io::{Cursor, Write};

fn xml_figures(
    anchor: &str,
    rel_type: &str,
    rel_mode: &str,
    content: SheetContent,
) -> (Vec<Inline>, AssetSink) {
    let drawing = format!(
        r#"<xdr:wsDr xmlns:xdr="{XDR}" xmlns:a="{}" xmlns:r="{}">{anchor}</xdr:wsDr>"#,
        ns::A,
        ns::R
    );
    let rels = format!(
        r#"<Relationships xmlns="{}"><Relationship Id="image" Type="{rel_type}" Target="../media/diagram.png" {rel_mode}/></Relationships>"#,
        ns::PKG_RELS
    );
    let mut z = zip::ZipWriter::new(Cursor::new(Vec::new()));
    for (p, b) in [
        ("xl/drawings/drawing.xml", drawing.as_bytes()),
        ("xl/drawings/_rels/drawing.xml.rels", rels.as_bytes()),
        ("xl/media/diagram.png", b"raw image bytes".as_slice()),
    ] {
        z.start_file(p, zip::write::SimpleFileOptions::default()).unwrap();
        z.write_all(b).unwrap();
    }
    let bytes = z.finish().unwrap().into_inner();
    let mut pkg = Package::open(&bytes).unwrap();
    let sheet = format!(
        r#"<worksheet xmlns="{}" xmlns:r="{}"><drawing r:id="draw"/></worksheet>"#,
        ns::SML,
        ns::R
    );
    let worksheet = parse_xml(sheet.as_bytes()).unwrap();
    let worksheet = worksheet.find(ns::SML, "worksheet").unwrap();
    // Use the normal relationships parser rather than constructing private maps.
    let relationships = format!(
        r#"<Relationships xmlns="{}"><Relationship Id="draw" Type="{DRAWING}" Target="../drawings/drawing.xml"/></Relationships>"#,
        ns::PKG_RELS
    );
    let mut z = zip::ZipWriter::new(Cursor::new(Vec::new()));
    z.start_file("r.xml", zip::write::SimpleFileOptions::default()).unwrap();
    z.write_all(relationships.as_bytes()).unwrap();
    let rel_bytes = z.finish().unwrap().into_inner();
    let mut rel_pkg = Package::open(&rel_bytes).unwrap();
    let rels = read_rels(&mut rel_pkg, "r.xml").unwrap();
    let mut assets = AssetSink::new();
    let images = xml(
        &mut pkg,
        worksheet,
        "xl/worksheets/sheet.xml",
        &rels,
        &content,
        &mut assets,
        &mut Vec::new(),
    )
    .unwrap();
    (images, assets)
}
fn pic(hidden: bool) -> String {
    format!(
        r#"<xdr:pic><xdr:nvPicPr><xdr:cNvPr id="1" name="Diagram" hidden="{}"/></xdr:nvPicPr><xdr:blipFill><a:blip r:embed="image"/></xdr:blipFill></xdr:pic>"#,
        u8::from(hidden)
    )
}
fn anchor(pictures: &str) -> String {
    format!(
        r#"<xdr:oneCellAnchor><xdr:from><xdr:col>3</xdr:col><xdr:row>1</xdr:row></xdr:from>{pictures}</xdr:oneCellAnchor>"#
    )
}
#[test]
fn typed_visible_sheet_picture_keeps_original_bytes_and_each_reference() {
    let (images, assets) = xml_figures(
        &anchor(&[pic(false), pic(false)].concat()),
        IMAGE,
        "",
        SheetContent::default(),
    );
    assert_eq!(images.len(), 2);
    assert_eq!(assets.assets.len(), 1);
    assert_eq!(assets.assets[0].bytes, b"raw image bytes");
}
#[test]
fn hidden_shape_anchor_or_group_does_not_load_image_bytes() {
    let (images, assets) = xml_figures(&anchor(&pic(true)), IMAGE, "", SheetContent::default());
    assert!(images.is_empty() && assets.assets.is_empty());
    let mut content = SheetContent::default();
    content.hidden_rows.insert(1);
    assert!(xml_figures(&anchor(&pic(false)), IMAGE, "", content).0.is_empty());
    let mut content = SheetContent::default();
    content.hidden_cols.push((3, 3));
    assert!(xml_figures(&anchor(&pic(false)), IMAGE, "", content).0.is_empty());
    let group = format!(
        r#"<xdr:grpSp><xdr:nvGrpSpPr><xdr:cNvPr hidden="true"/></xdr:nvGrpSpPr>{}</xdr:grpSp>"#,
        pic(false)
    );
    assert!(xml_figures(&anchor(&group), IMAGE, "", SheetContent::default()).0.is_empty());
}
#[test]
fn wrong_relationship_type_and_out_of_grid_anchor_cannot_publish_orphans() {
    assert!(xml_figures(&anchor(&pic(false)), DRAWING, "", SheetContent::default()).0.is_empty());
    let bad = anchor(&pic(false)).replace("<xdr:col>3</xdr:col>", "<xdr:col>16384</xdr:col>");
    assert!(xml_figures(&bad, IMAGE, "", SheetContent::default()).0.is_empty());
}
#[test]
fn external_image_is_a_link_and_never_a_downloaded_asset() {
    let (images, assets) = xml_figures(
        &anchor(&pic(false)),
        IMAGE,
        r#"TargetMode="External""#,
        SheetContent::default(),
    );
    assert!(matches!(&images[..], [Inline::Image { source: ImageSource::External(_), .. }]));
    assert!(assets.assets.is_empty());
}
#[test]
fn absolute_anchor_and_image_only_sheet_are_kept_without_an_empty_grid() {
    let a = format!("<xdr:absoluteAnchor>{}</xdr:absoluteAnchor>", pic(false));
    let (images, _) = xml_figures(&a, IMAGE, "", SheetContent::default());
    let mut doc = crate::model::Document::default();
    super::super::xlsx::push_sheet_images(
        &mut doc,
        "Figures",
        true,
        super::super::xlsx::Built::default(),
        images,
    );
    assert_eq!(doc.blocks.len(), 2);
    assert!(matches!(doc.blocks[0], crate::model::Block::Heading { .. }));
    assert!(matches!(doc.blocks[1], crate::model::Block::Paragraph(_)));
}

fn odf(body: &str, styles: &str) -> crate::model::Document {
    odf_with_layers(body, styles, "")
}
fn odf_with_layers(body: &str, styles: &str, layers: &str) -> crate::model::Document {
    let xml = format!(
        r#"<office:document-content xmlns:office="{}" xmlns:table="{}" xmlns:draw="{}" xmlns:text="{}" xmlns:xlink="{}" xmlns:style="{}"><office:automatic-styles>{styles}</office:automatic-styles><office:body><office:spreadsheet>{body}</office:spreadsheet></office:body></office:document-content>"#,
        ns::OFFICE,
        ns::TABLE,
        ns::DRAW,
        ns::TEXT,
        ns::XLINK,
        ns::STYLE
    );
    let layer_xml = format!(
        r#"<office:document-styles xmlns:office="{}" xmlns:draw="{}" xmlns:style="{}" office:version="1.3"><office:master-styles>{layers}</office:master-styles></office:document-styles>"#,
        ns::OFFICE,
        ns::DRAW,
        ns::STYLE
    );
    let mut z = zip::ZipWriter::new(Cursor::new(Vec::new()));
    for (p, b) in [
        ("content.xml", xml.as_bytes()),
        ("styles.xml", layer_xml.as_bytes()),
        ("Pictures/diagram.png", b"original PNG".as_slice()),
    ] {
        z.start_file(p, zip::write::SimpleFileOptions::default()).unwrap();
        z.write_all(b).unwrap();
    }
    crate::formats::odf::parse(&z.finish().unwrap().into_inner()).unwrap()
}
const FRAME: &str = r#"<draw:frame><draw:image xlink:href="Pictures/diagram.png"/></draw:frame>"#;
fn table_image(style: &str, shapes: &str) -> String {
    format!(
        r#"<table:table table:name="Budget" table:style-name="{style}"><table:shapes>{shapes}</table:shapes><table:table-row><table:table-cell><text:p>2150.75</text:p></table:table-cell></table:table-row></table:table>"#
    )
}
#[test]
fn standard_odf_table_shapes_are_referenced_once_after_the_semantic_grid() {
    let d = odf(&table_image("", FRAME), "");
    assert_eq!(d.assets.len(), 1);
    assert_eq!(d.assets[0].bytes, b"original PNG");
    assert!(
        matches!(&d.blocks[..],[crate::model::Block::Table(_),crate::model::Block::Paragraph(p)] if matches!(&p[..],[Inline::Image{..}]))
    );
}
#[test]
fn inherited_odf_hidden_sheet_and_hidden_group_do_not_load_new_floating_figures() {
    let styles = r#"<style:style style:name="hidden" style:family="table"><style:table-properties table:display="false"/></style:style><style:style style:name="child" style:family="table" style:parent-style-name="hidden"/>"#;
    let d = odf(&table_image("child", FRAME), styles);
    assert!(d.assets.is_empty());
    assert!(
        matches!(d.blocks[0], crate::model::Block::Table(_)),
        "existing text policy stays unchanged"
    );
    let group = format!(r#"<draw:g draw:layer="secret">{FRAME}</draw:g>"#);
    let layers =
        r#"<draw:layer-set><draw:layer draw:name="secret" draw:display="none"/></draw:layer-set>"#;
    assert!(odf_with_layers(&table_image("", &group), styles, layers).assets.is_empty());
}
#[test]
fn odf_compatibility_body_and_repeated_standard_references_share_only_asset_bytes() {
    let body = format!("{}{}{}", table_image("", FRAME), FRAME, FRAME);
    let d = odf(&body, "");
    assert_eq!(d.assets.len(), 1);
    assert_eq!(d.blocks.len(), 4);
    assert_eq!(d.assets[0].bytes, b"original PNG");
}

fn oa(kind: u16, ver: u16, instance: u16, body: &[u8]) -> Vec<u8> {
    let mut out = (ver | instance << 4).to_le_bytes().to_vec();
    out.extend(kind.to_le_bytes());
    out.extend((body.len() as u32).to_le_bytes());
    out.extend(body);
    out
}
fn binary_parts(hidden_group: bool) -> (Vec<u8>, Vec<u8>) {
    let mut png = vec![0; 17];
    png.extend(b"binary diagram");
    let blip = oa(0xF01E, 0, 0x6E0, &png);
    let mut fbse = vec![0; 36];
    fbse[0] = 6;
    fbse[1] = 6;
    fbse[20..24].copy_from_slice(&(blip.len() as u32).to_le_bytes());
    fbse[24..28].copy_from_slice(&1u32.to_le_bytes());
    fbse.extend(blip);
    let bank = oa(0xF000, 15, 0, &oa(0xF001, 15, 1, &oa(0xF007, 2, 6, &fbse)));
    let mut group = 1u32.to_le_bytes().to_vec();
    group.extend(5u32.to_le_bytes());
    let mut group = oa(0xF00A, 2, 0, &group);
    if hidden_group {
        group.extend(oa(0xF00B, 3, 1, &[0xBF, 0x03, 2, 0, 2, 0]));
    }
    let mut shape = 2u32.to_le_bytes().to_vec();
    shape.extend(2560u32.to_le_bytes());
    let mut shape = oa(0xF00A, 2, 75, &shape);
    shape.extend(oa(0xF00B, 3, 1, &[0x04, 0x41, 1, 0, 0, 0]));
    let mut anchor = vec![0; 18];
    anchor[2..4].copy_from_slice(&3u16.to_le_bytes());
    anchor[6..8].copy_from_slice(&1u16.to_le_bytes());
    shape.extend(oa(0xF010, 0, 0, &anchor));
    let drawing = oa(
        0xF002,
        15,
        0,
        &oa(0xF003, 15, 0, &[oa(0xF004, 15, 0, &group), oa(0xF004, 15, 0, &shape)].concat()),
    );
    (bank, drawing)
}
#[test]
fn binary_figure_is_anchored_and_hidden_group_or_cell_suppresses_the_actual_bank_load() {
    let (group, drawing) = binary_parts(false);
    let mut bank = Bank::read(&group, &[]).unwrap();
    assert_eq!(binary(&drawing, &SheetContent::default(), &mut bank).unwrap().len(), 1);
    assert_eq!(bank.assets.assets[0].bytes, b"binary diagram");
    let (group, drawing) = binary_parts(true);
    let mut bank = Bank::read(&group, &[]).unwrap();
    assert!(binary(&drawing, &SheetContent::default(), &mut bank).unwrap().is_empty());
    assert!(bank.assets.assets.is_empty());
    let (group, drawing) = binary_parts(false);
    let mut bank = Bank::read(&group, &[]).unwrap();
    let mut content = SheetContent::default();
    content.hidden_rows.insert(1);
    assert!(binary(&drawing, &content, &mut bank).unwrap().is_empty());
    assert!(bank.assets.assets.is_empty());
}

#[test]
fn odf_layer_none_suppresses_floating_frames_but_view_specific_modes_are_retained() {
    for display in ["none", "screen", "printer", "always"] {
        let styles = format!(
            r#"<draw:layer-set><draw:layer draw:name="controlled" draw:display="{display}"/></draw:layer-set>"#
        );
        let frame = FRAME.replace("<draw:frame>", r#"<draw:frame draw:layer="controlled">"#);
        let d = odf_with_layers(&table_image("", &frame), "", &styles);
        assert_eq!(d.assets.is_empty(), display == "none");
    }
    // draw:display does not belong to style:graphic-properties. It must
    // not turn an ordinary figure into a standard hidden-layer case.
    let nonstandard = r#"<style:style style:name="misplaced" style:family="graphic"><style:graphic-properties draw:display="none"/></style:style>"#;
    let frame = FRAME.replace("<draw:frame>", r#"<draw:frame draw:style-name="misplaced">"#);
    assert_eq!(odf(&table_image("", &frame), nonstandard).assets.len(), 1);
}

#[test]
fn page_scoped_odf_layer_does_not_hide_an_unrelated_sheet_figure() {
    let local = r#"<style:master-page style:name="Local"><draw:layer-set><draw:layer draw:name="secret" draw:display="none"/></draw:layer-set></style:master-page>"#;
    let frame = FRAME.replace("<draw:frame>", r#"<draw:frame draw:layer="secret">"#);
    assert_eq!(odf_with_layers(&table_image("", &frame), "", local).assets.len(), 1);
}
