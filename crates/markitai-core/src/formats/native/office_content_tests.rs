//! End-to-end semantic regressions for ordinary-user DOC and ODP/OTP tasks.
//! Expected facts come from the source tables, not renderer output tokens.

use super::*;
use std::io::{Cursor, Read, Write};

#[test]
fn legacy_doc_embedded_chart_keeps_all_source_rows_and_the_existing_body() {
    let doc = extract(include_bytes!("fixtures/legacy-doc-chart.doc"), "doc").unwrap();
    let expected = [
        "|  | Column 1 | Column 2 | Column 3 |",
        "| Row 1 | 9.1 | 3.2 | 4.54 |",
        "| Row 2 | 2.4 | 8.8 | 9.65 |",
        "| Row 3 | 3.1 | 1.5 | 3.7 |",
        "| Row 4 | 4.3 | 9.02 | 6.2 |",
    ];
    let mut previous = 0;
    for row in expected {
        assert_eq!(
            doc.markdown.matches(row).count(),
            1,
            "{row}\n{}",
            doc.markdown
        );
        let at = doc.markdown.find(row).unwrap();
        assert!(at > previous);
        previous = at;
    }
    // FTXBXS.lid=1026 binds this chart to the main-story CP1280 anchor,
    // just after the list. Its textbox story is stored after the main story.
    assert!(doc.markdown.starts_with("**Lorem ipsum**\n\n# Lorem ipsum"));
    assert!(doc.markdown.contains("- *Nulla facilisi.*"));
    assert!(
        doc.markdown
            .contains("[Mauris id ex erat. ](https://products.office.com/en-us/word)"),
        "{}",
        doc.markdown
    );
    assert!(
        doc.markdown
            .contains("| 5 | Etiam vehicula luctus fermentum. | Ipsum |  |")
    );
    assert!(
        doc.markdown
            .find("Maecenas tincidunt est efficitur ligula euismod, sit amet ornare est vulputate.")
            .unwrap()
            < doc.markdown.find("Column 1").unwrap()
    );
    assert!(
        doc.markdown.find("Column 1").unwrap() < doc.markdown.find("In non mauris justo.").unwrap()
    );
    assert!(!doc.markdown.contains("EMBED LibreOffice"));
    // The actual textbox PLC is complete. Missing-index recovery is tested
    // separately with authored missing metadata, never inferred from this DOC.
    assert!(
        !doc.warnings
            .iter()
            .any(|warning| warning.contains("field index omits"))
    );
    let chart = format!(
        "{}\n| --- | --- | --- | --- |\n{}",
        expected[0],
        expected[1..].join("\n")
    );
    let body = doc.markdown.replace(&chart, "");
    assert_eq!(
        body.split_whitespace().collect::<Vec<_>>(),
        EXISTING_DOC_BODY.split_whitespace().collect::<Vec<_>>()
    );
}

#[test]
fn modern_word_reserved3_cannot_move_an_existing_textbox_story() {
    let original = include_bytes!("fixtures/legacy-doc-chart.doc");
    let expected = extract(original, "doc").unwrap();
    // Mutate only the WordDocument copy. The authorized source and its OLE
    // package, PLC metadata, chart facts and existing body remain unchanged.
    for reserved in [1u32, u32::MAX] {
        let mut ole = cfb::CompoundFile::open(Cursor::new(original.to_vec())).unwrap();
        let mut word = Vec::new();
        ole.open_stream("/WordDocument")
            .unwrap()
            .read_to_end(&mut word)
            .unwrap();
        assert_eq!(&word[0x58..0x5C], &[0, 0, 0, 0]);
        word[0x58..0x5C].copy_from_slice(&reserved.to_le_bytes());
        ole.open_stream("/WordDocument")
            .unwrap()
            .write_all(&word)
            .unwrap();
        let altered = ole.into_inner().into_inner();
        let actual = extract(&altered, "doc").unwrap();
        assert_eq!(actual.markdown, expected.markdown, "reserved3={reserved}");
        assert_eq!(actual.warnings, expected.warnings, "reserved3={reserved}");
        assert!(actual.markdown.contains("| Row 4 | 4.3 | 9.02 | 6.2 |"));
    }
}

const NAMESPACES: &str = r#"xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
 xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"
 xmlns:draw="urn:oasis:names:tc:opendocument:xmlns:drawing:1.0"
 xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0"
 xmlns:xlink="http://www.w3.org/1999/xlink""#;

fn table(name: &str) -> String {
    let rows = [
        ["Region", "Amount"],
        ["North", "1250.50"],
        ["South", "900.25"],
        ["Grand total", "2150.75"],
    ];
    let rows = rows
        .into_iter()
        .map(|row| {
            format!(
                "<table:table-row>{}</table:table-row>",
                row.into_iter()
                    .map(|text| format!(
                        "<table:table-cell><text:p>{text}</text:p></table:table-cell>"
                    ))
                    .collect::<String>()
            )
        })
        .collect::<String>();
    format!("<table:table table:name=\"{name}\">{rows}</table:table>")
}

fn presentation(pages: &str, template: bool) -> Vec<u8> {
    let content = format!(
        "<office:document-content {NAMESPACES}><office:body><office:presentation>{pages}</office:presentation></office:body></office:document-content>"
    );
    let mut zip = zip::ZipWriter::new(Cursor::new(Vec::new()));
    zip.start_file("mimetype", zip::write::SimpleFileOptions::default())
        .unwrap();
    zip.write_all(if template {
        b"application/vnd.oasis.opendocument.presentation-template".as_slice()
    } else {
        b"application/vnd.oasis.opendocument.presentation".as_slice()
    })
    .unwrap();
    zip.start_file("content.xml", zip::write::SimpleFileOptions::default())
        .unwrap();
    zip.write_all(content.as_bytes()).unwrap();
    zip.finish().unwrap().into_inner()
}

#[test]
fn odp_and_otp_budget_tables_stay_on_the_second_page_and_before_its_image() {
    let pages = format!(
        r#"<draw:page><draw:frame><draw:text-box><text:p>Overview</text:p></draw:text-box></draw:frame></draw:page>
      <draw:page>{}<draw:frame><draw:image xlink:href="https://example.invalid/diagram.png"/></draw:frame></draw:page>
      <draw:page/>"#,
        table("Budget")
    );
    for (extension, template) in [("odp", false), ("otp", true)] {
        let doc = extract(&presentation(&pages, template), extension).unwrap();
        let pages: Vec<_> = doc.markdown.split("<!-- Slide number: ").collect();
        assert_eq!(pages.len(), 4, "{}", doc.markdown);
        assert!(pages[1].contains("Overview"));
        assert!(!pages[1].contains("1250.50"));
        for row in [
            "| Region | Amount |",
            "| North | 1250.50 |",
            "| South | 900.25 |",
            "| Grand total | 2150.75 |",
        ] {
            assert_eq!(
                pages[2].matches(row).count(),
                1,
                "{extension}: {row}\n{}",
                doc.markdown
            );
        }
        assert!(pages[2].find("Grand total").unwrap() < pages[2].find("diagram.png").unwrap());
        assert!(!pages[3].contains("1250.50"));
    }
}

#[test]
fn tables_on_a_page_in_a_group_and_in_a_frame_are_read_once_in_source_order() {
    let pages = format!(
        r#"<draw:page><draw:frame><draw:text-box><text:p>Before</text:p></draw:text-box></draw:frame>
      {}<draw:g>{}</draw:g><draw:frame>{}</draw:frame>
      <draw:frame><draw:text-box><text:p>After</text:p></draw:text-box></draw:frame></draw:page>"#,
        table("Direct").replace("North", "Direct North"),
        table("Grouped").replace("North", "Grouped North"),
        table("Framed").replace("North", "Framed North")
    );
    let doc = extract(&presentation(&pages, false), "odp").unwrap();
    let positions: Vec<_> = [
        "Before",
        "Direct North",
        "Grouped North",
        "Framed North",
        "After",
    ]
    .iter()
    .map(|text| {
        assert_eq!(doc.markdown.matches(text).count(), 1, "{}", doc.markdown);
        doc.markdown.find(text).unwrap()
    })
    .collect();
    assert!(positions.windows(2).all(|pair| pair[0] < pair[1]));
    assert_eq!(doc.markdown.matches("| Grand total | 2150.75 |").count(), 3);
}

// Frozen pre-fix public CLI body: a regression for all preexisting words,
// formatting and their order, independently of the new chart source facts.
const EXISTING_DOC_BODY: &str = r###"**Lorem ipsum**

# Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nunc ac faucibus odio.

Vestibulum neque massa, scelerisque sit amet ligula eu, congue molestie mi. Praesent ut varius sem. Nullam at porttitor arcu, nec lacinia nisi. Ut ac dolor vitae odio interdum condimentum. **Vivamus dapibus sodales ex, vitae malesuada ipsum cursus convallis. Maecenas sed egestas nulla, ac condimentum orci.** Mauris diam felis, vulputate ac suscipit et, iaculis non est. Curabitur semper arcu ac ligula semper, nec luctus nisl blandit. Integer lacinia ante ac libero lobortis imperdiet. *Nullam mollis convallis ipsum, ac accumsan nunc vehicula vitae.* Nulla eget justo in felis tristique fringilla. Morbi sit amet tortor quis risus auctor condimentum. Morbi in ullamcorper elit. Nulla iaculis tellus sit amet mauris tempus fringilla.

Maecenas mauris lectus, lobortis et purus mattis, blandit dictum tellus.

- **Maecenas non lorem quis tellus placerat varius.**
- *Nulla facilisi.*
- <u>Aenean congue fringilla justo ut aliquam.</u>
- [Mauris id ex erat. ](https://products.office.com/en-us/word)Nunc vulputate neque vitae justo facilisis, non condimentum ante sagittis.
- Morbi viverra semper lorem nec molestie.
- Maecenas tincidunt est efficitur ligula euismod, sit amet ornare est vulputate.

![](.markitai/assets/asset-1.emf)

In non mauris justo. Duis vehicula mi vel mi pretium, a viverra erat efficitur. Cras aliquam est ac eros varius, id iaculis dui auctor. Duis pretium neque ligula, et pulvinar mi placerat et. Nulla nec nunc sit amet nunc posuere vestibulum. Ut id neque eget tortor mattis tristique. Donec ante est, blandit sit amet tristique vel, lacinia pulvinar arcu. Pellentesque scelerisque fermentum erat, id posuere justo pulvinar ut. Cras id eros sed enim aliquam lobortis. Sed lobortis nisl ut eros efficitur tincidunt. Cras justo mi, porttitor quis mattis vel, ultricies ut purus. Ut facilisis et lacus eu cursus.

In eleifend velit vitae libero sollicitudin euismod. Fusce vitae vestibulum velit. Pellentesque vulputate lectus quis pellentesque commodo. Aliquam erat volutpat. Vestibulum in egestas velit. Pellentesque fermentum nisl vitae fringilla venenatis. Etiam id mauris vitae orci maximus ultricies.

# Cras fringilla ipsum magna, in fringilla dui commodo a.

|  |  |  |  |
| --- | --- | --- | --- |
|  | Lorem ipsum | Lorem ipsum | Lorem ipsum |
| 1 | In eleifend velit vitae libero sollicitudin euismod. | Lorem |  |
| 2 | Cras fringilla ipsum magna, in fringilla dui commodo a. | Ipsum |  |
| 3 | Aliquam erat volutpat. | Lorem |  |
| 4 | Fusce vitae vestibulum velit. | Lorem |  |
| 5 | Etiam vehicula luctus fermentum. | Ipsum |  |

Etiam vehicula luctus fermentum. In vel metus congue, pulvinar lectus vel, fermentum dui. Maecenas ante orci, egestas ut aliquet sit amet, sagittis a magna. Aliquam ante quam, pellentesque ut dignissim quis, laoreet eget est. Aliquam erat volutpat. Class aptent taciti sociosqu ad litora torquent per conubia nostra, per inceptos himenaeos. Ut ullamcorper justo sapien, in cursus libero viverra eget. Vivamus auctor imperdiet urna, at pulvinar leo posuere laoreet. Suspendisse neque nisl, fringilla at iaculis scelerisque, ornare vel dolor. Ut et pulvinar nunc. Pellentesque fringilla mollis efficitur. Nullam venenatis commodo imperdiet. Morbi velit neque, semper quis lorem quis, efficitur dignissim ipsum. Ut ac lorem sed turpis imperdiet eleifend sit amet id sapien.

# Lorem ipsum dolor sit amet, consectetur adipiscing elit.

Nunc ac faucibus odio. Vestibulum neque massa, scelerisque sit amet ligula eu, congue molestie mi. Praesent ut varius sem. Nullam at porttitor arcu, nec lacinia nisi. Ut ac dolor vitae odio interdum condimentum. Vivamus dapibus sodales ex, vitae malesuada ipsum cursus convallis. Maecenas sed egestas nulla, ac condimentum orci. Mauris diam felis, vulputate ac suscipit et, iaculis non est. Curabitur semper arcu ac ligula semper, nec luctus nisl blandit. Integer lacinia ante ac libero lobortis imperdiet. Nullam mollis convallis ipsum, ac accumsan nunc vehicula vitae. Nulla eget justo in felis tristique fringilla. Morbi sit amet tortor quis risus auctor condimentum. Morbi in ullamcorper elit. Nulla iaculis tellus sit amet mauris tempus fringilla.

## Maecenas mauris lectus, lobortis et purus mattis, blandit dictum tellus.

Maecenas non lorem quis tellus placerat varius. Nulla facilisi. Aenean congue fringilla justo ut aliquam. Mauris id ex erat. Nunc vulputate neque vitae justo facilisis, non condimentum ante sagittis. Morbi viverra semper lorem nec molestie. Maecenas tincidunt est efficitur ligula euismod, sit amet ornare est vulputate.

In non mauris justo. Duis vehicula mi vel mi pretium, a viverra erat efficitur. Cras aliquam est ac eros varius, id iaculis dui auctor. Duis pretium neque ligula, et pulvinar mi placerat et. Nulla nec nunc sit amet nunc posuere vestibulum. Ut id neque eget tortor mattis tristique. Donec ante est, blandit sit amet tristique vel, lacinia pulvinar arcu. Pellentesque scelerisque fermentum erat, id posuere justo pulvinar ut. Cras id eros sed enim aliquam lobortis. Sed lobortis nisl ut eros efficitur tincidunt. Cras justo mi, porttitor quis mattis vel, ultricies ut purus. Ut facilisis et lacus eu cursus.

## In eleifend velit vitae libero sollicitudin euismod.

Fusce vitae vestibulum velit. Pellentesque vulputate lectus quis pellentesque commodo. Aliquam erat volutpat. Vestibulum in egestas velit. Pellentesque fermentum nisl vitae fringilla venenatis. Etiam id mauris vitae orci maximus ultricies. Cras fringilla ipsum magna, in fringilla dui commodo a.

Etiam vehicula luctus fermentum. In vel metus congue, pulvinar lectus vel, fermentum dui. Maecenas ante orci, egestas ut aliquet sit amet, sagittis a magna. Aliquam ante quam, pellentesque ut dignissim quis, laoreet eget est. Aliquam erat volutpat. Class aptent taciti sociosqu ad litora torquent per conubia nostra, per inceptos himenaeos. Ut ullamcorper justo sapien, in cursus libero viverra eget. Vivamus auctor imperdiet urna, at pulvinar leo posuere laoreet. Suspendisse neque nisl, fringilla at iaculis scelerisque, ornare vel dolor. Ut et pulvinar nunc. Pellentesque fringilla mollis efficitur. Nullam venenatis commodo imperdiet. Morbi velit neque, semper quis lorem quis, efficitur dignissim ipsum. Ut ac lorem sed turpis imperdiet eleifend sit amet id sapien.

![](.markitai/assets/asset-2.jpg)
"###;
