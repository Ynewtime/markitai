//! Legacy PowerPoint embedded objects (charts and worksheets) read as their
//! data, end to end. The fixture and its generator are in
//! `tests/fixtures/legacy-ppt`.

use super::*;

const PPT: &[u8] = include_bytes!("../fixtures/legacy-ppt/embedded-objects.ppt");

const EXPECTED: &str = "\
<!-- Slide number: 1 -->
## Revenue

Quarterly revenue

|  | North | South |
| --- | --- | --- |
| Q1 | 4.5 | 3.25 |
| Q2 | 5.1 | 3.9 |
| Q3 | 6 | 4.4 |

<!-- Slide number: 2 -->
## Budget

| Item | Q1 | Q2 |
| --- | --- | --- |
| Rent | $1,200 | $1,250 |
| Travel | $430 | $515 |
| Share | 35% | 40% |

<!-- Slide number: 3 -->
## Visitors

Weekly visitors

|  | Web | Store |
| --- | --- | --- |
| Mon | 1,250 | 310 |
| Tue | 1,430 | 275 |
| Wed | 980 | 402 |

<!-- Slide number: 4 -->
## Sales

|  | 2025 | 2026 |
| --- | --- | --- |
| Jan | 12 | 14 |
| Feb | 15 | 19 |

<!-- Slide number: 5 -->
## Formula";

#[test]
fn legacy_ppt_embedded_charts_and_worksheets_read_as_tables() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("objects.ppt");
    std::fs::write(&source, PPT).unwrap();
    let result = convert(
        source.to_str().unwrap(),
        ConvertOptions {
            config: Some(json!({
                "llm":{"enabled":false,"pure":true},"ocr":{"enabled":false},
                "cache":{"enabled":false},"history":{"record":false}
            })),
            ..Default::default()
        },
    )
    .unwrap();
    // A LibreOffice chart (ODF package), an Excel worksheet, an MS Graph
    // chart and an Excel chart each read as their data; the chart's data
    // sheet cell it does not plot stays out, and the equation adds nothing.
    assert_eq!(result.markdown.trim_end(), EXPECTED);
    assert!(!result.markdown.contains("Worksheet only"));
}
