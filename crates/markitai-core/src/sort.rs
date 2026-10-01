//! One compiled sort for the crate's total-order slice sorts.
//!
//! A `sort_by`, `sort_by_key` or `sort_unstable_by` call compiles the standard
//! library's sort again for its element type and comparator closure, 3 to 7 KB
//! each here. [`by`] and [`by_key`] sort the positions `0..len` through one
//! compiled sort with the caller's comparator and then move the elements into
//! that order. Every comparator given to them is a total order (integers,
//! strings, paths, booleans, `total_cmp` on floats, `Option`s and tuples of
//! these), which has exactly one stable order, so the order produced is the
//! one `sort_by` gave. A caller that replaced an unstable sort either sorts
//! elements whose equal values are identical or did not depend on the order
//! of equal elements.

use std::cmp::Ordering;

/// `v.sort_by(compare)`, for a `compare` that is a total order.
pub(crate) fn by<T>(v: &mut [T], mut compare: impl FnMut(&T, &T) -> Ordering) {
    if v.len() < 2 {
        return;
    }
    let order = stable_order(v.len(), &mut |a, b| compare(&v[a], &v[b]));
    permute(v, order);
}

/// `v.sort_by_key(key)`; like `sort_by_key`, the key is computed for each
/// comparison.
pub(crate) fn by_key<T, K: Ord>(v: &mut [T], mut key: impl FnMut(&T) -> K) {
    by(v, |a, b| key(a).cmp(&key(b)));
}

/// The stable order of the positions `0..len` under `compare`. The command
/// line's own sorts go through it too, so the program holds one copy.
#[doc(hidden)]
#[inline(never)]
pub fn stable_order(len: usize, compare: &mut dyn FnMut(usize, usize) -> Ordering) -> Vec<usize> {
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
    fn values(len: usize, seed: u64, range: u64) -> Vec<(u64, f32, usize)> {
        let mut state = seed | 1;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        (0..len)
            .map(|i| (next() % range, (next() % 9) as f32 - 4.0, i))
            .collect()
    }

    #[test]
    fn matches_the_standard_stable_sort() {
        // Lengths around the standard sorts' small-sort, eager and merge
        // thresholds.
        for len in [0, 1, 2, 3, 7, 8, 16, 20, 21, 33, 64, 65, 300, 4096] {
            for (seed, range) in [(1, 2), (7, 5), (42, 1000), (9, u64::MAX)] {
                let input = values(len, seed + len as u64, range);
                let compare = |a: &(u64, f32, usize), b: &(u64, f32, usize)| {
                    b.0.cmp(&a.0).then(a.1.total_cmp(&b.1))
                };
                let mut expected = input.clone();
                expected.sort_by(compare);
                let mut got = input.clone();
                by(&mut got, compare);
                assert_eq!(got, expected, "descending, len {len}, seed {seed}");

                let mut expected = input.clone();
                expected.sort_by_key(|item| item.0 % 3);
                let mut got = input;
                by_key(&mut got, |item| item.0 % 3);
                assert_eq!(got, expected, "by key, len {len}, seed {seed}");
                // Ties keep their input order.
                assert!(
                    got.windows(2)
                        .all(|w| w[0].0 % 3 < w[1].0 % 3 || w[0].2 < w[1].2)
                );
            }
        }
    }

    #[test]
    fn sorts_a_part_and_moves_elements_that_are_not_copied() {
        let mut names: Vec<String> = ["delta", "alpha", "charlie", "alpha", "bravo"]
            .map(String::from)
            .to_vec();
        by_key(&mut names, |name| std::cmp::Reverse(name.len()));
        assert_eq!(names, ["charlie", "delta", "alpha", "alpha", "bravo"]);
        by(&mut names[1..4], String::cmp);
        assert_eq!(names, ["charlie", "alpha", "alpha", "delta", "bravo"]);
        by(&mut names, String::cmp);
        assert_eq!(names, ["alpha", "alpha", "bravo", "charlie", "delta"]);
    }
}
