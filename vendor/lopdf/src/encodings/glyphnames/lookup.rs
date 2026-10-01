//! markitai: glyph name lookup by binary search.
//!
//! `sorted_names.rs`, generated from the `glyphs!` list by
//! `scripts/sorted_glyph_names.py`, holds every name with its code in byte
//! order. At compile time the list is checked to be strictly ascending and
//! packed into one byte string with 16-bit bounds and a parallel code array,
//! so the table holds no pointers for a position-independent executable to
//! relocate at load.

use core::cmp::Ordering;

use super::sorted_names::SORTED_GLYPH_NAMES as NAMES;

const COUNT: usize = NAMES.len();

/// The length of all names together.
const NAME_BYTES_LEN: usize = {
    let mut total = 0;
    let mut i = 0;
    while i < COUNT {
        total += NAMES[i].0.len();
        i += 1;
    }
    total
};

// `NAME_BOUNDS` holds 16-bit offsets.
const _: () = assert!(NAME_BYTES_LEN <= u16::MAX as usize);

// Binary search needs strictly ascending names, which also rules out a name
// listed twice.
const _: () = {
    let mut i = 1;
    while i < COUNT {
        assert!(
            precedes(NAMES[i - 1].0.as_bytes(), NAMES[i].0.as_bytes()),
            "glyph names out of order; run scripts/sorted_glyph_names.py"
        );
        i += 1;
    }
};

/// Every name, concatenated in table order.
static NAME_BYTES: [u8; NAME_BYTES_LEN] = {
    let mut bytes = [0; NAME_BYTES_LEN];
    let mut at = 0;
    let mut i = 0;
    while i < COUNT {
        let name = NAMES[i].0.as_bytes();
        let mut j = 0;
        while j < name.len() {
            bytes[at] = name[j];
            at += 1;
            j += 1;
        }
        i += 1;
    }
    bytes
};

/// Name `i` is `NAME_BYTES[NAME_BOUNDS[i]..NAME_BOUNDS[i + 1]]`.
static NAME_BOUNDS: [u16; COUNT + 1] = {
    let mut bounds = [0; COUNT + 1];
    let mut i = 0;
    while i < COUNT {
        bounds[i + 1] = bounds[i] + NAMES[i].0.len() as u16;
        i += 1;
    }
    bounds
};

/// The code of name `i`.
static CODES: [u16; COUNT] = {
    let mut codes = [0; COUNT];
    let mut i = 0;
    while i < COUNT {
        codes[i] = NAMES[i].1;
        i += 1;
    }
    codes
};

/// Whether `a` sorts strictly before `b`, comparing bytes.
const fn precedes(a: &[u8], b: &[u8]) -> bool {
    let mut i = 0;
    while i < a.len() && i < b.len() {
        if a[i] != b[i] {
            return a[i] < b[i];
        }
        i += 1;
    }
    a.len() < b.len()
}

/// The code of the glyph named `name`, if the list has that name.
pub(super) fn code_of(name: &[u8]) -> Option<u16> {
    let mut low = 0;
    let mut high = COUNT;
    while low < high {
        let mid = low + (high - low) / 2;
        let candidate = &NAME_BYTES[NAME_BOUNDS[mid] as usize..NAME_BOUNDS[mid + 1] as usize];
        match candidate.cmp(name) {
            Ordering::Less => low = mid + 1,
            Ordering::Greater => high = mid,
            Ordering::Equal => return Some(CODES[mid]),
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::super::{Glyph, LISTED_GLYPH_NAMES};
    use super::*;

    /// What the replaced `match` returned for each name: the code of the
    /// name's first listing.
    fn listed() -> HashMap<&'static [u8], u16> {
        let mut listed = HashMap::new();
        for &(name, code) in LISTED_GLYPH_NAMES {
            listed.entry(name).or_insert(code);
        }
        listed
    }

    #[test]
    fn every_listed_name_has_its_code() {
        assert_eq!(COUNT, LISTED_GLYPH_NAMES.len());
        assert_eq!(listed().len(), COUNT);
        for &(name, code) in LISTED_GLYPH_NAMES {
            assert_eq!(
                Glyph::from_name(name),
                Some(Glyph(code)),
                "{}",
                String::from_utf8_lossy(name)
            );
        }
    }

    /// Every prefix of each name, the name with a byte before or after it,
    /// with one byte changed by one, and with its letters' case flipped.
    #[test]
    fn changed_names_agree_with_the_listing() {
        let listed = listed();
        let mut probes: Vec<Vec<u8>> = vec![Vec::new(), b"\xff".to_vec(), b"\0".to_vec(), b"zzzzzzzz".to_vec()];
        for &(name, _) in LISTED_GLYPH_NAMES {
            for end in 0..name.len() {
                probes.push(name[..end].to_vec());
            }
            for byte in [0, b'.', b'0', b'A', b'Z', b'_', b'a', b'z', 0x7f, 0xff] {
                probes.push([name, &[byte]].concat());
                probes.push([&[byte], name].concat());
            }
            for i in 0..name.len() {
                for delta in [1, u8::MAX] {
                    let mut changed = name.to_vec();
                    changed[i] = changed[i].wrapping_add(delta);
                    probes.push(changed);
                }
            }
            probes.push(
                name.iter()
                    .map(|byte| if byte.is_ascii_alphabetic() { byte ^ 0x20 } else { *byte })
                    .collect(),
            );
        }
        let mut names = 0;
        for probe in &probes {
            let expected = listed.get(probe.as_slice()).copied();
            names += usize::from(expected.is_some());
            assert_eq!(
                Glyph::from_name(probe).map(|glyph| glyph.0),
                expected,
                "{}",
                String::from_utf8_lossy(probe)
            );
        }
        // Some changes give another name (`A` → `B`, `Acute` → `acute`); the
        // counts were also computed independently from the list.
        assert_eq!((probes.len(), names), (251_701, 8_207));
    }

    #[test]
    fn known_names() {
        assert_eq!(Glyph::from_name(b"space"), Some(Glyph::space));
        assert_eq!(Glyph::from_name(b"A"), Some(Glyph(0x0041)));
        assert_eq!(Glyph::from_name(b"AEacute"), Some(Glyph(0x01fc)));
        assert_eq!(Glyph::from_name(b"wonmonospace"), Some(Glyph(0xffe6)));
        assert_eq!(Glyph::from_name(b"Space"), None);
        assert_eq!(Glyph::from_name(b"uni0041"), None);
    }
}
