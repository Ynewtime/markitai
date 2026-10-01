//! One compiled sort for the command's slice sorts and ordered collections.
//!
//! A `sort_by` or `sort_by_key` call compiles the standard library's stable
//! sort again for its element type and comparator closure, 3 to 22 KB each
//! here, and collecting into a `BTreeMap` or `BTreeSet` sorts the collected
//! pairs the same way, once per source iterator type. [`by`] and [`by_key`]
//! sort the positions `0..len` through one compiled sort with the caller's
//! comparator and then move the elements into that order; [`map`] and [`set`]
//! insert one entry at a time. Every comparator given to them is a total order
//! (integers, strings, booleans, `Option`s and tuples of these), which has
//! exactly one stable order, so the order produced is the one `sort_by` gave.
//! Keys that compare equal are equal strings or integers here, so it does not
//! matter which of two equal keys a collection keeps.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

/// `v.sort_by(compare)`, for a `compare` that is a total order.
pub(crate) fn by<T>(v: &mut [T], mut compare: impl FnMut(&T, &T) -> Ordering) {
    let order = stable_order(v.len(), &mut |a, b| compare(&v[a], &v[b]));
    permute(v, order);
}

/// `v.sort_by_key(key)`; like `sort_by_key`, the key is computed for each
/// comparison.
pub(crate) fn by_key<T, K: Ord>(v: &mut [T], mut key: impl FnMut(&T) -> K) {
    by(v, |a, b| key(a).cmp(&key(b)));
}

/// `iter.collect::<BTreeMap<_, _>>()` for keys whose equal values are equal
/// (the last value of a repeated key is kept, as there).
pub(crate) fn map<K: Ord, V>(iter: impl IntoIterator<Item = (K, V)>) -> BTreeMap<K, V> {
    let mut map = BTreeMap::new();
    for (key, value) in iter {
        map.insert(key, value);
    }
    map
}

/// `iter.collect::<BTreeSet<_>>()` for elements whose equal values are equal.
pub(crate) fn set<T: Ord>(iter: impl IntoIterator<Item = T>) -> BTreeSet<T> {
    let mut set = BTreeSet::new();
    for item in iter {
        set.insert(item);
    }
    set
}

/// The stable order of the positions `0..len` under `compare`.
#[inline(never)]
fn stable_order(len: usize, compare: &mut dyn FnMut(usize, usize) -> Ordering) -> Vec<usize> {
    let mut order: Vec<usize> = (0..len).collect();
    order.sort_by(|&a, &b| compare(a, b));
    order
}

/// Moves the element at `order[i]` to position `i`, walking each cycle of the
/// permutation once and marking its positions done.
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random values with many ties.
    fn values(len: usize, seed: u64, range: u64) -> Vec<(u64, usize)> {
        let mut state = seed | 1;
        (0..len)
            .map(|i| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state % range, i)
            })
            .collect()
    }

    #[test]
    fn matches_the_standard_stable_sort() {
        for len in [0, 1, 2, 3, 7, 8, 16, 20, 21, 33, 64, 65, 200, 1000, 5000] {
            for (seed, range) in [(1, 2), (7, 5), (42, 1000), (9, u64::MAX)] {
                let input = values(len, seed + len as u64, range);
                let mut expected = input.clone();
                expected.sort_by_key(|a| std::cmp::Reverse(a.0));
                let mut got = input.clone();
                by(&mut got, |a, b| b.0.cmp(&a.0));
                assert_eq!(got, expected, "descending, len {len}, seed {seed}");

                let mut expected = input.clone();
                expected.sort_by_key(|item| item.0 % 3);
                let mut got = input;
                by_key(&mut got, |item| item.0 % 3);
                assert_eq!(got, expected, "by key, len {len}, seed {seed}");
                // Ties keep their input order.
                assert!(
                    got.windows(2)
                        .all(|w| w[0].0 % 3 < w[1].0 % 3 || w[0].1 < w[1].1)
                );
            }
        }
    }

    #[test]
    fn permutes_elements_that_are_not_copied() {
        let mut names: Vec<String> = ["delta", "alpha", "charlie", "alpha", "bravo"]
            .map(String::from)
            .to_vec();
        by_key(&mut names, |name| std::cmp::Reverse(name.len()));
        assert_eq!(names, ["charlie", "delta", "alpha", "alpha", "bravo"]);
        by(&mut names, String::cmp);
        assert_eq!(names, ["alpha", "alpha", "bravo", "charlie", "delta"]);
    }

    #[test]
    fn collections_match_collect() {
        let pairs = [("b", 1), ("a", 2), ("b", 3), ("c", 4), ("a", 5)];
        assert_eq!(map(pairs), pairs.into_iter().collect::<BTreeMap<_, _>>());
        assert_eq!(map(pairs)["b"], 3);
        let words = ["b", "a", "b", "c", "a"];
        assert_eq!(set(words), words.into_iter().collect::<BTreeSet<_>>());
    }
}
