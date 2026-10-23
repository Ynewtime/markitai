//! CJK radical forms in a PDF's text, read as the ideographs they stand for.
//!
//! A PDF's ToUnicode map sometimes names a radical code point instead of the
//! ideograph its glyph draws: Chrome's "Save as PDF" writes `使⽤` (U+2F64
//! KANGXI RADICAL USE) and `右下⻆` (U+2EC6, a CJK Radicals Supplement form)
//! for 使用 and 右下角, which look alike but do not search, sort or match.
//! The mapping is the shared workbench's (`web/src/lib/compare.ts`, `norm`),
//! as Python markitai 1.3.0 applies it to PDF output: NFKC for the Kangxi
//! Radicals (U+2F00–U+2FD5, all of which it maps) and the two supplement
//! forms it maps (U+2E9F, U+2EF3), then compare.ts's `RADICALS` for the
//! supplement forms NFKC leaves alone. Other supplement forms (component
//! shapes such as ⺁ and ⺌) stay.

use std::borrow::Cow;

/// NFKC of U+2F00..=U+2FD5, in code point order.
const KANGXI: [char; 214] = [
    '一', '丨', '丶', '丿', '乙', '亅', '二', '亠', '人', '儿', '入', '八', '冂', '冖', '冫', '几',
    '凵', '刀', '力', '勹', '匕', '匚', '匸', '十', '卜', '卩', '厂', '厶', '又', '口', '囗', '土',
    '士', '夂', '夊', '夕', '大', '女', '子', '宀', '寸', '小', '尢', '尸', '屮', '山', '巛', '工',
    '己', '巾', '干', '幺', '广', '廴', '廾', '弋', '弓', '彐', '彡', '彳', '心', '戈', '戶', '手',
    '支', '攴', '文', '斗', '斤', '方', '无', '日', '曰', '月', '木', '欠', '止', '歹', '殳', '毋',
    '比', '毛', '氏', '气', '水', '火', '爪', '父', '爻', '爿', '片', '牙', '牛', '犬', '玄', '玉',
    '瓜', '瓦', '甘', '生', '用', '田', '疋', '疒', '癶', '白', '皮', '皿', '目', '矛', '矢', '石',
    '示', '禸', '禾', '穴', '立', '竹', '米', '糸', '缶', '网', '羊', '羽', '老', '而', '耒', '耳',
    '聿', '肉', '臣', '自', '至', '臼', '舌', '舛', '舟', '艮', '色', '艸', '虍', '虫', '血', '行',
    '衣', '襾', '見', '角', '言', '谷', '豆', '豕', '豸', '貝', '赤', '走', '足', '身', '車', '辛',
    '辰', '辵', '邑', '酉', '釆', '里', '金', '長', '門', '阜', '隶', '隹', '雨', '靑', '非', '面',
    '革', '韋', '韭', '音', '頁', '風', '飛', '食', '首', '香', '馬', '骨', '高', '髟', '鬥', '鬯',
    '鬲', '鬼', '魚', '鳥', '鹵', '鹿', '麥', '麻', '黃', '黍', '黑', '黹', '黽', '鼎', '鼓', '鼠',
    '鼻', '齊', '齒', '龍', '龜', '龠',
];

/// The supplement forms that stand for one ideograph: NFKC's two, then
/// compare.ts's `RADICALS`, entry for entry; sorted by code point.
const SUPPLEMENT: [(char, char); 30] = [
    ('\u{2E9F}', '母'),
    ('\u{2EA0}', '民'),
    ('\u{2EB0}', '纟'),
    ('\u{2EC4}', '西'),
    ('\u{2EC5}', '见'),
    ('\u{2EC6}', '角'),
    ('\u{2EC8}', '讠'),
    ('\u{2EC9}', '贝'),
    ('\u{2ECB}', '车'),
    ('\u{2ED0}', '钅'),
    ('\u{2ED3}', '长'),
    ('\u{2ED4}', '门'),
    ('\u{2ED8}', '青'),
    ('\u{2ED9}', '韦'),
    ('\u{2EDA}', '页'),
    ('\u{2EDB}', '风'),
    ('\u{2EDC}', '飞'),
    ('\u{2EDD}', '食'),
    ('\u{2EE0}', '饣'),
    ('\u{2EE2}', '马'),
    ('\u{2EE5}', '鱼'),
    ('\u{2EE6}', '鸟'),
    ('\u{2EE7}', '卤'),
    ('\u{2EE8}', '麦'),
    ('\u{2EE9}', '黄'),
    ('\u{2EEA}', '黾'),
    ('\u{2EEC}', '齐'),
    ('\u{2EEE}', '齿'),
    ('\u{2EF0}', '龙'),
    ('\u{2EF3}', '龟'),
];

/// The ideograph a radical form stands for, if it stands for one.
pub(super) fn ideograph(c: char) -> Option<char> {
    match c {
        '\u{2F00}'..='\u{2FD5}' => Some(KANGXI[c as usize - 0x2F00]),
        '\u{2E80}'..='\u{2EFF}' => SUPPLEMENT
            .binary_search_by_key(&c, |&(radical, _)| radical)
            .ok()
            .map(|index| SUPPLEMENT[index].1),
        _ => None,
    }
}

/// `text` with each mappable radical form replaced by its ideograph.
fn fold(text: &str) -> Cow<'_, str> {
    if !text.chars().any(|c| ideograph(c).is_some()) {
        return Cow::Borrowed(text);
    }
    Cow::Owned(text.chars().map(|c| ideograph(c).unwrap_or(c)).collect())
}

/// Replace each mappable radical form in `text` by its ideograph.
pub(super) fn fold_in_place(text: &mut String) {
    if let Cow::Owned(folded) = fold(text) {
        *text = folded;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn folds_kangxi_and_supplement_forms() {
        assert_eq!(fold("使\u{2F64} 右下\u{2EC6}"), "使用 右下角");
        assert_eq!(
            fold("\u{2F00}\u{2FD5}\u{2E9F}\u{2EF3}\u{2EA0}\u{2EF0}"),
            "一龠母龟民龙"
        );
    }

    #[test]
    fn keeps_component_shapes_and_other_text() {
        // ⺁ (U+2E81) and ⺌ (U+2E8C) are shapes, not characters.
        assert!(matches!(
            fold("\u{2E81}\u{2E8C} plain 中文"),
            Cow::Borrowed(_)
        ));
        // Unassigned code points after the Kangxi block stay.
        assert_eq!(fold("\u{2FD6}"), "\u{2FD6}");
    }

    #[test]
    fn tables_are_sorted_and_complete() {
        assert!(SUPPLEMENT.windows(2).all(|pair| pair[0].0 < pair[1].0));
        assert_eq!(SUPPLEMENT.len(), 30);
        assert!(KANGXI.iter().all(|c| ('\u{4E00}'..='\u{9FFF}').contains(c)));
    }
}
