//! Fonts the portable renderer draws with when a PDF does not embed its own.
//!
//! hayro asks in two cases. For one of the 14 standard fonts, and for every
//! non-embedded simple font (which hayro maps onto them), it gets hayro's
//! embedded Foxit substitutes from PDFium: the same faces on every host,
//! including containers without fonts, but Latin only. For a non-embedded
//! composite (CID) font, the usual case in Chinese, Japanese and Korean PDFs,
//! the host's fonts are searched by the PDF's font name and then by the
//! font's character collection; hayro maps each character through Unicode
//! into the face found. With no suitable host face the Foxit face draws the
//! Latin it can and the other glyphs stay blank.
//!
//! The host font list is read once per process, only when a document first
//! needs it. Face data is read from disk once and kept for the process.

use hayro::hayro_interpret::font::{FallbackFontQuery, FontData, FontQuery};
use hayro::hayro_interpret::hayro_cmap::CidFamily;
use resvg::usvg::fontdb;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

pub(super) fn resolve(query: &FontQuery) -> Option<(FontData, u32)> {
    match query {
        FontQuery::Standard(font) => Some(font.get_font_data()),
        FontQuery::Fallback(query) => {
            host_face(query).or_else(|| Some(query.pick_standard_font().get_font_data()))
        }
    }
}

struct HostFonts {
    database: fontdb::Database,
    /// Faces by normalized PostScript and family name (every language).
    by_name: HashMap<String, Vec<fontdb::ID>>,
}

fn host() -> &'static HostFonts {
    static HOST: OnceLock<HostFonts> = OnceLock::new();
    HOST.get_or_init(|| {
        let mut database = fontdb::Database::new();
        database.load_system_fonts();
        let mut by_name: HashMap<String, Vec<fontdb::ID>> = HashMap::new();
        for face in database.faces() {
            let mut keys = vec![key(&face.post_script_name)];
            keys.extend(face.families.iter().map(|(family, _)| key(family)));
            keys.sort();
            keys.dedup();
            for name in keys.into_iter().filter(|name| !name.is_empty()) {
                by_name.entry(name).or_default().push(face.id);
            }
        }
        HostFonts { database, by_name }
    })
}

/// Name comparison ignores case, spaces and punctuation:
/// `Arial,Bold`, `MS-Mincho` and `Noto Sans CJK SC` become
/// `arialbold`, `msmincho` and `notosanscjksc`.
fn key(name: &str) -> String {
    name.chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

/// The PDF's names for the font, whole first, then without a style suffix
/// (`SimSun,Bold` -> `SimSun`, `Arial-BoldMT` -> `Arial`).
fn requested_names(query: &FallbackFontQuery) -> Vec<String> {
    let names = [
        &query.post_script_name,
        &query.font_name,
        &query.font_family,
    ];
    let names = names.into_iter().flatten().collect::<Vec<_>>();
    let whole = names.iter().map(|name| key(name));
    let bases = names.iter().filter_map(|name| {
        let base = name.split([',', '-']).next()?;
        (base.len() < name.len()).then(|| key(base))
    });
    let mut keys = Vec::new();
    for name in whole.chain(bases) {
        // A short base such as `MS` from `MS-Mincho` names no real family.
        if name.chars().count() >= 3 && !keys.contains(&name) {
            keys.push(name);
        }
    }
    keys
}

/// Host families for one character collection, serif (Song/Ming/Mincho/
/// Batang) and sans-serif (Hei/Gothic) faces apart; Windows, macOS and common
/// Linux names.
fn collection_families(
    family: &CidFamily,
) -> Option<(&'static [&'static str], &'static [&'static str])> {
    Some(match family {
        CidFamily::AdobeGB1 => (
            &[
                "SimSun",
                "NSimSun",
                "Songti SC",
                "STSong",
                "Noto Serif CJK SC",
                "Noto Serif SC",
                "Source Han Serif SC",
                "Source Han Serif CN",
                "AR PL UMing CN",
                "FangSong",
                "KaiTi",
            ],
            &[
                "Microsoft YaHei",
                "DengXian",
                "SimHei",
                "PingFang SC",
                "Heiti SC",
                "STHeiti",
                "Hiragino Sans GB",
                "Noto Sans CJK SC",
                "Noto Sans SC",
                "Source Han Sans SC",
                "Source Han Sans CN",
                "WenQuanYi Zen Hei",
                "WenQuanYi Micro Hei",
                "Droid Sans Fallback",
            ],
        ),
        CidFamily::AdobeCNS1 => (
            &[
                "PMingLiU",
                "MingLiU",
                "Songti TC",
                "LiSong Pro",
                "Noto Serif CJK TC",
                "Noto Serif TC",
                "Source Han Serif TC",
                "Source Han Serif TW",
                "AR PL UMing TW",
                "AR PL UMing HK",
            ],
            &[
                "Microsoft JhengHei",
                "PingFang TC",
                "Heiti TC",
                "Noto Sans CJK TC",
                "Noto Sans TC",
                "Source Han Sans TC",
                "Source Han Sans TW",
            ],
        ),
        CidFamily::AdobeJapan1 => (
            &[
                "MS Mincho",
                "MS PMincho",
                "Yu Mincho",
                "Hiragino Mincho ProN",
                "Hiragino Mincho Pro",
                "Noto Serif CJK JP",
                "Noto Serif JP",
                "Source Han Serif JP",
                "Source Han Serif",
                "IPAexMincho",
                "IPAMincho",
                "TakaoMincho",
            ],
            &[
                "Yu Gothic",
                "Meiryo",
                "MS Gothic",
                "MS PGothic",
                "Hiragino Sans",
                "Hiragino Kaku Gothic ProN",
                "Noto Sans CJK JP",
                "Noto Sans JP",
                "Source Han Sans JP",
                "Source Han Sans",
                "IPAexGothic",
                "IPAGothic",
                "TakaoGothic",
            ],
        ),
        CidFamily::AdobeKorea1 => (
            &[
                "Batang",
                "BatangChe",
                "AppleMyungjo",
                "Noto Serif CJK KR",
                "Noto Serif KR",
                "Source Han Serif K",
                "Source Han Serif KR",
                "UnBatang",
                "NanumMyeongjo",
            ],
            &[
                "Malgun Gothic",
                "Gulim",
                "Dotum",
                "Apple SD Gothic Neo",
                "AppleGothic",
                "Noto Sans CJK KR",
                "Noto Sans KR",
                "Source Han Sans K",
                "Source Han Sans KR",
                "NanumGothic",
                "UnDotum",
            ],
        ),
        CidFamily::AdobeIdentity | CidFamily::Custom { .. } => return None,
    })
}

const COLLECTIONS: [CidFamily; 4] = [
    CidFamily::AdobeGB1,
    CidFamily::AdobeCNS1,
    CidFamily::AdobeJapan1,
    CidFamily::AdobeKorea1,
];

/// Words in a font name that mark a serif CJK design.
const SERIF_HINTS: [&str; 7] = [
    "song", "ming", "mincho", "batang", "myungjo", "serif", "kai",
];

/// Families to try after the PDF's own names: the font's collection first
/// (its serif or sans-serif list first, by the descriptor flag or a name
/// hint), then the other collections, which share most Han characters. A font
/// without a known collection (Identity encodings) gets them only when its
/// name is CJK-looking.
fn fallback_families(query: &FallbackFontQuery, names: &[String]) -> Vec<&'static str> {
    let serif = query.is_serif
        || names
            .iter()
            .any(|name| SERIF_HINTS.iter().any(|hint| name.contains(hint)));
    let own = query.character_collection.as_ref().and_then(|collection| {
        collection_families(&collection.family).map(|lists| (collection.family.clone(), lists))
    });
    let looks_cjk = [
        &query.post_script_name,
        &query.font_name,
        &query.font_family,
    ]
    .into_iter()
    .flatten()
    .any(|name| !name.is_ascii())
        || names.iter().any(|name| {
            [
                "sim", "song", "hei", "ming", "kai", "gothic", "mincho", "yahei", "batang",
                "gulim", "dotum",
            ]
            .iter()
            .any(|hint| name.contains(hint))
        });
    if own.is_none() && !looks_cjk {
        return Vec::new();
    }
    let mut families = Vec::new();
    let mut push = |(serifs, sans): (&'static [&'static str], &'static [&'static str])| {
        let (first, second) = if serif {
            (serifs, sans)
        } else {
            (sans, serifs)
        };
        families.extend(first.iter().chain(second));
    };
    if let Some((_, lists)) = &own {
        push(*lists);
    }
    for family in COLLECTIONS {
        if own.as_ref().is_none_or(|(own, _)| *own != family) {
            push(collection_families(&family).expect("predefined collections have families"));
        }
    }
    families
}

/// The face among `ids` closest to the requested weight and slant.
fn closest(
    database: &fontdb::Database,
    ids: &[fontdb::ID],
    query: &FallbackFontQuery,
) -> Option<fontdb::ID> {
    let weight = if query.is_bold {
        query.font_weight.max(700)
    } else {
        query.font_weight
    };
    ids.iter().copied().min_by_key(|id| {
        let Some(face) = database.face(*id) else {
            return u32::MAX;
        };
        let italic = face.style != fontdb::Style::Normal;
        u32::from(face.weight.0).abs_diff(weight)
            + if italic == query.is_italic { 0 } else { 1_000 }
    })
}

fn host_face(query: &FallbackFontQuery) -> Option<(FontData, u32)> {
    let host = host();
    let names = requested_names(query);
    let found = names
        .iter()
        .find_map(|name| host.by_name.get(name))
        .or_else(|| {
            fallback_families(query, &names)
                .into_iter()
                .find_map(|family| host.by_name.get(&key(family)))
        })
        .and_then(|ids| closest(&host.database, ids, query))?;
    face_data(&host.database, found)
}

fn face_data(database: &fontdb::Database, id: fontdb::ID) -> Option<(FontData, u32)> {
    static LOADED: OnceLock<Mutex<HashMap<fontdb::ID, (FontData, u32)>>> = OnceLock::new();
    let loaded = LOADED.get_or_init(Mutex::default);
    if let Some(face) = loaded.lock().ok()?.get(&id) {
        return Some(face.clone());
    }
    let face = database.with_face_data(id, |data, index| {
        (Arc::new(data.to_vec()) as FontData, index)
    })?;
    loaded.lock().ok()?.insert(id, face.clone());
    Some(face)
}

/// Whether a Chinese face from the lists is installed, for tests that need one.
#[cfg(test)]
pub(crate) fn host_has_cjk_face() -> bool {
    let host = host();
    let (serif, sans) = collection_families(&CidFamily::AdobeGB1).expect("GB1 has families");
    serif
        .iter()
        .chain(sans)
        .any(|family| host.by_name.contains_key(&key(family)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use hayro::hayro_interpret::hayro_cmap::CharacterCollection;

    fn query(name: &str, family: Option<CidFamily>) -> FallbackFontQuery {
        FallbackFontQuery {
            post_script_name: Some(name.into()),
            character_collection: family.map(|family| CharacterCollection {
                family,
                supplement: 0,
            }),
            ..FallbackFontQuery::default()
        }
    }

    #[test]
    fn names_are_compared_without_case_spacing_or_style_suffixes() {
        assert_eq!(key("Noto Sans CJK SC"), "notosanscjksc");
        assert_eq!(key("MS-Mincho"), key("MS Mincho"));
        assert_eq!(
            requested_names(&query("SimSun,Bold", None)),
            ["simsunbold", "simsun"]
        );
        assert_eq!(
            requested_names(&query("Arial-BoldMT", None)),
            ["arialboldmt", "arial"]
        );
        // `MS` alone is too short to name a family.
        assert_eq!(requested_names(&query("MS-Mincho", None)), ["msmincho"]);
    }

    #[test]
    fn a_collection_orders_its_own_families_first_by_design() {
        let sans = fallback_families(&query("UnknownGothic", Some(CidFamily::AdobeJapan1)), &[]);
        assert_eq!(sans.first(), Some(&"Yu Gothic"));
        let names = requested_names(&query("STSongStd-Light", Some(CidFamily::AdobeGB1)));
        let serif = fallback_families(&query("STSongStd-Light", Some(CidFamily::AdobeGB1)), &names);
        assert_eq!(serif.first(), Some(&"SimSun"));
        // Other collections follow, so a Japanese file can still draw its Han
        // characters with a Chinese face.
        assert!(serif.contains(&"MS Mincho") && serif.contains(&"Batang"));
        // An Identity-encoded Latin font is not sent to CJK families.
        assert!(
            fallback_families(
                &query("Calibri", None),
                &requested_names(&query("Calibri", None))
            )
            .is_empty()
        );
        let identity = query("SimHei", None);
        assert!(!fallback_families(&identity, &requested_names(&identity)).is_empty());
    }

    #[test]
    fn the_standard_fonts_are_embedded() {
        let (data, index) = resolve(&FontQuery::Standard(
            hayro::hayro_interpret::font::StandardFont::Helvetica,
        ))
        .unwrap();
        assert_eq!(index, 0);
        assert!((*data).as_ref().len() > 10_000);
    }
}
