//! markitai: shared entry points for the crate's slice sorts.
//!
//! A closure is a type of its own, so every `sort_by` call with a closure
//! compiled another copy of the standard library's sort, a few kilobytes
//! each over well over a hundred call sites (and again for every caller of a
//! generic function that sorts). These entry points take the comparator as a
//! trait object, so the stable sort is compiled once per element type. A
//! caller keeps its element type and its comparator; the sort's steps depend
//! only on those and on the comparisons' results, so the order produced is
//! unchanged, also for a comparator that is not a total order. Every call
//! of this module is a markitai change (see `MARKITAI-PATCH.md`).

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

/// [`slice::sort_by`], compiled once per element type.
#[inline(never)]
pub(crate) fn stable<T>(v: &mut [T], compare: &mut dyn FnMut(&T, &T) -> Ordering) {
    v.sort_by(|a, b| compare(a, b));
}

/// [`slice::sort_by`] for a comparator that is a total order (integers,
/// `total_cmp`, `Ord` keys, or floats that cannot be NaN), with no compiled
/// sort of its own: the positions `0..len` are sorted through the `usize`
/// instance of [`stable`] and the elements are then moved into that order.
/// A total order has exactly one stable order, so the result is the one
/// `sort_by` gives whichever sort finds it; for any other comparator the
/// result could depend on the compiled sort, so such callers use [`stable`].
pub(crate) fn total<T>(v: &mut [T], compare: &mut dyn FnMut(&T, &T) -> Ordering) {
    if v.len() < 2 {
        return;
    }
    let mut order: Vec<usize> = (0..v.len()).collect();
    stable(&mut order, &mut |&a, &b| compare(&v[a], &v[b]));
    permute(v, order);
}

/// `v.sort()` for integers or tuples of integers. Equal values are
/// identical, so the unstable order is the stable one, and every sort of one
/// such type shares the unstable sort's compiled instance.
pub(crate) fn integers<T: Integer>(v: &mut [T]) {
    v.sort_unstable();
}

/// Types whose equal values are identical.
pub(crate) trait Integer: Ord {}
impl Integer for u16 {}
impl Integer for u32 {}
impl Integer for usize {}

/// `iter.collect::<BTreeMap<_, _>>()` without its sort (compiled again for
/// every iterator type), for keys whose equal values are identical: entry
/// by entry, keeping the last value of a repeated key as `collect` does.
pub(crate) fn map<K: Integer, V>(iter: impl IntoIterator<Item = (K, V)>) -> BTreeMap<K, V> {
    let mut map = BTreeMap::new();
    for (key, value) in iter {
        map.insert(key, value);
    }
    map
}

/// `iter.collect::<BTreeSet<_>>()` without its sort, for elements whose
/// equal values are identical.
pub(crate) fn set<T: Integer>(iter: impl IntoIterator<Item = T>) -> BTreeSet<T> {
    let mut set = BTreeSet::new();
    for item in iter {
        set.insert(item);
    }
    set
}

/// Moves the element at `order[i]` to position `i`, walking each cycle of
/// the permutation once and marking its positions done.
fn permute<T>(v: &mut [T], mut order: Vec<usize>) {
    for start in 0..order.len() {
        let mut current = start;
        loop {
            let next = std::mem::replace(&mut order[current], current);
            if next == start {
                break;
            }
            v.swap(current, next);
            current = next;
        }
    }
}

/// `v.sort_by(|a, b| a.total_cmp(b))`.
#[inline(never)]
pub(crate) fn f32_ascending(v: &mut [f32]) {
    v.sort_by(f32::total_cmp);
}

/// `v.sort_by(|a, b| b.total_cmp(a))`. Two floats `total_cmp` finds equal
/// have the same bits, so the reversed ascending order is the same sequence
/// and needs no second compiled sort.
#[inline(never)]
pub(crate) fn f32_descending(v: &mut [f32]) {
    v.sort_by(f32::total_cmp);
    v.reverse();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_orders_match_the_closure_sorts_they_replace() {
        let values = [
            3.5,
            -0.0,
            0.0,
            f32::NAN,
            -f32::NAN,
            1.0,
            -2.25,
            f32::INFINITY,
            0.0,
            f32::NEG_INFINITY,
            1.0,
            -0.0,
            f32::MIN_POSITIVE,
        ];
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        let mut expected = values.to_vec();
        expected.sort_by(|a, b| a.total_cmp(b));
        let mut got = values.to_vec();
        f32_ascending(&mut got);
        assert_eq!(bits(&got), bits(&expected));

        let mut expected = values.to_vec();
        expected.sort_by(|a, b| b.total_cmp(a));
        let mut got = values.to_vec();
        f32_descending(&mut got);
        assert_eq!(bits(&got), bits(&expected));
        assert!(got[0].is_nan() && got[0].is_sign_positive());
    }

    #[test]
    fn shared_sort_keeps_stability_and_the_callers_direction() {
        // Pairs equal by their first field keep their input order, and the
        // comparator's direction is the caller's.
        let pairs = [(2, 'a'), (1, 'b'), (2, 'c'), (0, 'd'), (1, 'e'), (2, 'f')];
        let mut expected = pairs.to_vec();
        expected.sort_by_key(|p| std::cmp::Reverse(p.0));
        let mut got = pairs.to_vec();
        stable(&mut got, &mut |a, b| b.0.cmp(&a.0));
        assert_eq!(got, expected);
        assert_eq!(got.iter().map(|p| p.1).collect::<String>(), "acfbed");

        // Past the insertion-sort length of either std sort, so an unstable
        // sort would reorder the ties.
        let many: Vec<(u32, usize)> = (0..64).map(|i| ((i * 7 % 5) as u32, i)).collect();
        let mut expected = many.clone();
        expected.sort_by_key(|p| std::cmp::Reverse(p.0));
        let mut got = many;
        stable(&mut got, &mut |a, b| b.0.cmp(&a.0));
        assert_eq!(got, expected);
        assert!(got.windows(2).all(|w| w[0].0 > w[1].0 || w[0].1 < w[1].1));
    }

    #[test]
    fn total_order_sort_matches_sort_by() {
        // Deterministic values with many ties, at lengths around the std
        // sorts' small-sort, eager and merge thresholds.
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for len in [0, 1, 2, 3, 8, 16, 20, 21, 33, 64, 65, 300, 4096] {
            for range in [2u64, 7, 1000, u64::MAX] {
                let input: Vec<(u64, f32, usize)> = (0..len)
                    .map(|i| (next() % range, (next() % 9) as f32 - 4.0, i))
                    .collect();
                let mut expected = input.clone();
                expected.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.total_cmp(&b.1)));
                let mut got = input.clone();
                total(&mut got, &mut |a, b| b.0.cmp(&a.0).then(a.1.total_cmp(&b.1)));
                assert_eq!(got, expected, "len {len}, range {range}");

                let mut expected = input.clone();
                expected.sort_by_key(|item| item.0 % 3);
                let mut got = input;
                total(&mut got, &mut |a, b| (a.0 % 3).cmp(&(b.0 % 3)));
                assert_eq!(got, expected, "len {len}, range {range}");
                // Ties keep their input order.
                assert!(got.windows(2).all(|w| w[0].0 % 3 < w[1].0 % 3 || w[0].2 < w[1].2));
            }
        }
        // Elements that own memory are moved, not copied.
        let mut words: Vec<String> = ["delta", "alpha", "charlie", "alpha", "bravo"]
            .map(String::from)
            .to_vec();
        total(&mut words, &mut |a, b| b.len().cmp(&a.len()));
        assert_eq!(words, ["charlie", "delta", "alpha", "alpha", "bravo"]);
    }

    #[test]
    fn integer_sorts_and_collections_match_the_std_ones() {
        let values: Vec<u32> = (0..300u32).map(|i| i.wrapping_mul(2_654_435_761) % 97).collect();
        let mut expected = values.clone();
        expected.sort();
        let mut got = values.clone();
        integers(&mut got);
        assert_eq!(got, expected);

        let pairs: Vec<(u32, usize)> = values.iter().enumerate().map(|(i, &v)| (v, i)).collect();
        let collected = pairs.iter().copied().collect::<BTreeMap<_, _>>();
        assert_eq!(map(pairs.iter().copied()), collected);
        // A repeated key keeps its last value, as `collect` does.
        assert_eq!(map([(1u32, 'a'), (2, 'b'), (1, 'c')])[&1], 'c');
        let codes: Vec<u16> = values.iter().map(|&v| v as u16).collect();
        assert_eq!(set(codes.iter().copied()), codes.into_iter().collect::<BTreeSet<_>>());
    }
}
