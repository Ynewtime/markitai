use super::*;
use iwork::table::{CellValue, Format};

fn bytes(doc: &iwork::Document) -> Vec<u8> {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("authored.numbers");
    doc.save(&path).unwrap();
    std::fs::read(path).unwrap()
}

#[test]
fn independent_numbers_parser_fixture_preserves_sheet_table_order_and_values() {
    let result = extract(include_bytes!("fixtures/test-1.numbers")).unwrap();
    assert_eq!(result.metadata["sheet_count"], 2);
    assert_eq!(result.metadata["table_count"], 3);
    // Expected names and cells are independently recorded by the fixture's
    // upstream test_tables.py, not generated from iwork's own output.
    let ordered = [
        "# ZZZ\\_Sheet\\_1",
        "## ZZZ\\_Table\\_1",
        "## ZZZ\\_Table\\_2",
        "# ZZZ\\_Sheet\\_2",
        "## XXX\\_Table\\_1",
    ];
    let positions: Vec<_> = ordered
        .iter()
        .map(|s| result.markdown.find(s).unwrap())
        .collect();
    assert!(positions.windows(2).all(|p| p[0] < p[1]));
    for text in ["YYY\\_ROW\\_4", "YYY\\_4\\_2", "ZZZ\\_3\\_3", "XXX\\_3\\_5"] {
        assert!(result.markdown.contains(text), "missing {text}");
    }
    let sizes: Vec<_> = result.metadata["numbers_tables"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| (v["rows"].as_u64().unwrap(), v["columns"].as_u64().unwrap()))
        .collect();
    assert_eq!(sizes, [(5, 3), (4, 4), (4, 6)]);
}

#[test]
fn independent_format_fixture_is_not_reported_as_exact_display_formatting() {
    let result = extract(include_bytes!("fixtures/test-formats.numbers")).unwrap();
    assert!(result.metadata["table_count"].as_u64().unwrap() > 0);
    assert!(
        result
            .warnings
            .iter()
            .any(|w| w.contains("display formatting is normalized"))
    );
    for text in [
        "star-bullet-1",
        "dash-bullet",
        "the quick brown fox jumped",
        "blue",
        "apple",
    ] {
        assert!(
            result.markdown.contains(text),
            "missing upstream rich-text value: {text}"
        );
    }
}

#[test]
fn authored_unicode_decimal_percent_currency_and_cached_formula_values() {
    let mut doc = iwork::Document::new_spreadsheet("工作表", "销售", 4, 3).unwrap();
    {
        let mut table = doc.table_mut("销售").unwrap();
        table
            .set_block("A1", &[vec!["项目", "金额", "比例"]])
            .unwrap();
        table.set("A2", "雪 ❄ | <script>").unwrap();
        table
            .set("B2", CellValue::Currency(Decimal::parse("12.50").unwrap()))
            .unwrap();
        table
            .set("C2", CellValue::Number(Decimal::parse("0.125").unwrap()))
            .unwrap();
        table.formula("B3", "=B2*2", 25).unwrap();
        table
            .format(
                "B2",
                &Format::Currency {
                    code: "USD".into(),
                    decimals: Some(2),
                },
            )
            .unwrap();
        table
            .format("C2", &Format::Percent { decimals: Some(1) })
            .unwrap();
    }
    let result = extract(&bytes(&doc)).unwrap();
    assert!(result.markdown.contains("# 工作表\n\n## 销售\n"));
    assert!(result.markdown.contains("雪 ❄ \\| &lt;script&gt;"));
    assert!(result.markdown.contains("USD 12.50"));
    assert!(result.markdown.contains("12.5%"));
    assert!(result.markdown.contains("25"));
    assert!(!result.markdown.contains("=B2"));
    assert_eq!(
        result.metadata["numbers_tables"][0]["cached_formula_cells"],
        1
    );
    assert!(
        result
            .warnings
            .iter()
            .any(|w| w.contains("not recalculated"))
    );
}

#[test]
fn duplicate_table_names_across_sheets_and_merges_keep_identity() {
    let mut doc = iwork::Document::new_spreadsheet("第一", "相同", 3, 3).unwrap();
    doc.set_cell("相同", 1, 1, CellValue::Text("合并 <&>".into()))
        .unwrap();
    doc.merge_cells("相同", 1, 1, 2, 2).unwrap();
    // The writer forbids ambiguous names, while the reader uses object IDs.
    // Rename only the model's label after constructing the distinct table.
    doc.add_sheet("第二", "另一表", 2, 2).unwrap();
    let second = doc
        .tables()
        .into_iter()
        .find(|t| t.sheet.as_deref() == Some("第二"))
        .unwrap();
    doc.set_cell(
        &second.identifier.to_string(),
        1,
        1,
        CellValue::Text("独立值".into()),
    )
    .unwrap();
    let mut model = doc.archive(second.model).unwrap();
    model.set_in_order(8, iwork::pb::Value::Bytes("相同".as_bytes().to_vec()));
    doc.set_archive_for(second.model, &model).unwrap();
    let result = extract(&bytes(&doc)).unwrap();
    assert_eq!(result.markdown.matches("## 相同").count(), 2);
    assert!(
        result
            .markdown
            .contains("rowspan=\"2\" colspan=\"2\">合并 &lt;&amp;&gt;")
    );
    assert_eq!(result.markdown.matches("合并").count(), 1);
    assert!(result.markdown.contains("独立值"));
    assert_eq!(
        result.metadata["numbers_tables"][0]["merges"][0],
        json!({"row":2,"column":2,"rows":2,"columns":2})
    );
}

#[test]
fn fixed_decimal_rounding_does_not_pass_through_binary_float() {
    for (value, places, expected) in [
        ("1.005", 2, "1.01"),
        ("-1.005", 2, "-1.01"),
        ("0.0001", 2, "0.00"),
        ("9.995", 2, "10.00"),
        (
            "99999999999999999999999999",
            2,
            "99999999999999999999999999.00",
        ),
    ] {
        assert_eq!(
            decimal(Decimal::parse(value).unwrap(), Some(places)),
            expected
        );
    }
    assert_eq!(
        decimal(
            Decimal {
                mantissa: 123,
                exponent: 400
            },
            None
        ),
        "123e400"
    );
}

#[test]
fn output_limit_is_checked_inside_a_wide_row() {
    let mut doc = iwork::Document::new_spreadsheet("S", "T", 2, 20).unwrap();
    doc.set_cell("T", 1, 0, CellValue::Text("&".repeat(MAX_TEXT)))
        .unwrap();
    let table = doc.tables().remove(0);
    let mut output = "x".repeat(MAX_OUTPUT - 100);
    assert!(append_table(&mut output, &table, &mut Flags::default()).is_err());
    assert!(output.len() <= MAX_OUTPUT + 6 * MAX_TEXT);
}

#[test]
fn duplicate_table_model_reference_cannot_multiply_decode_allocations() {
    let doc = iwork::Document::new_spreadsheet("S", "T", 2, 2).unwrap();
    let mut package = doc.package().clone();
    let mut duplicate = doc
        .objects()
        .find(|(_, object)| object.message_type() == 6000)
        .unwrap()
        .1
        .clone();
    duplicate.identifier = u64::MAX - 1;
    package.entries.push((
        "Index/duplicate.iwa".into(),
        iwork::iwa::serialize(&[duplicate]),
    ));
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("duplicate-model.numbers");
    package.write(&path).unwrap();
    assert!(
        extract(&std::fs::read(path).unwrap())
            .unwrap_err()
            .to_string()
            .contains("same model")
    );
}

#[test]
fn malformed_model_and_duplicate_object_ids_fail_before_high_level_decode() {
    use iwork::iwa::{ArchiveMessage, ArchiveObject};
    use iwork::pb::{Message, Value};
    use std::io::{Cursor, Write};
    let mut payload = Message::default();
    payload.set_in_order(6, Value::Varint(u64::MAX));
    payload.set_in_order(7, Value::Varint(3));
    let object = ArchiveObject {
        identifier: 1,
        messages: vec![ArchiveMessage {
            message_type: 6001,
            version: vec![],
            extra: vec![],
            payload: payload.encode(),
        }],
        extra: vec![],
    };
    let mut writer = zip::ZipWriter::new(Cursor::new(Vec::new()));
    writer
        .start_file(
            "Index/Document.iwa",
            zip::write::SimpleFileOptions::default(),
        )
        .unwrap();
    writer.write_all(&iwork::iwa::serialize(&[object])).unwrap();
    let data = writer.finish().unwrap().into_inner();
    assert!(
        extract(&data)
            .unwrap_err()
            .to_string()
            .contains("dimensions")
    );

    let doc = iwork::Document::new_spreadsheet("S", "T", 2, 2).unwrap();
    let mut package = doc.package().clone();
    let first = package
        .entries
        .iter()
        .find(|(name, _)| name.ends_with(".iwa"))
        .unwrap()
        .1
        .clone();
    package.entries.push(("Index/duplicate.iwa".into(), first));
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("duplicate.numbers");
    package.write(&path).unwrap();
    assert!(
        extract(&std::fs::read(path).unwrap())
            .unwrap_err()
            .to_string()
            .contains("duplicate IWA")
    );
}

#[test]
fn a_cell_s_line_breaks_are_one_br_each_and_none_at_its_edges() {
    let mut markdown = String::new();
    escaped(
        &mut markdown,
        "\nBring gloves\r\n\r\nand a trowel\u{2028}| <br>\n",
        false,
    );
    assert_eq!(markdown, "Bring gloves<br>and a trowel<br>\\| &lt;br&gt;");
    let mut html = String::new();
    escaped(&mut html, "a\u{2029}\u{0b}b", true);
    assert_eq!(html, "a<br>b");
}
