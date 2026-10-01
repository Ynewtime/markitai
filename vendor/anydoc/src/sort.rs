//! markitai: one compiled sort for the crate's slice sorts by key.
//!
//! A `sort_by_key` call compiles the standard library's stable sort again for
//! its element type and key closure, 3 to 5 KB each. [`by_key`] sorts the
//! positions `0..len` through one compiled sort with the caller's key and then
//! moves the elements into that order. Every key given to it is an integer, a
//! total order whose equal keys keep their input order here as in
//! `sort_by_key`: a total order has exactly one stable order, so the order
//! produced is the one `sort_by_key` gave.

use std::cmp::Ordering;

/// `v.sort_by_key(key)` for an integer key; like `sort_by_key`, the key is
/// computed for each comparison.
pub(crate) fn by_key<T, K: Ord>(v: &mut [T], mut key: impl FnMut(&T) -> K) {
    if v.len() < 2 {
        return;
    }
    let order = stable_order(v.len(), &mut |a, b| key(&v[a]).cmp(&key(&v[b])));
    permute(v, order);
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

    #[test]
    fn matches_the_standard_stable_sort() {
        // Deterministic values with many ties, at lengths around the standard
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
                let input: Vec<(u64, usize)> = (0..len).map(|i| (next() % range, i)).collect();
                let mut expected = input.clone();
                expected.sort_by_key(|item| std::cmp::Reverse(item.0));
                let mut got = input.clone();
                by_key(&mut got, |item| std::cmp::Reverse(item.0));
                assert_eq!(got, expected, "len {len}, range {range}");

                let mut expected = input.clone();
                expected.sort_by_key(|item| item.0 % 3);
                let mut got = input;
                by_key(&mut got, |item| item.0 % 3);
                assert_eq!(got, expected, "len {len}, range {range}");
                // Ties keep their input order.
                assert!(got.windows(2).all(|w| w[0].0 % 3 < w[1].0 % 3 || w[0].1 < w[1].1));
            }
        }
    }

    #[test]
    fn moves_elements_that_are_not_copied() {
        let mut words: Vec<(u32, String)> =
            [(3, "delta"), (1, "alpha"), (2, "charlie"), (1, "echo"), (0, "bravo")]
                .map(|(rank, word)| (rank, word.to_string()))
                .to_vec();
        by_key(&mut words, |item| item.0);
        let words: Vec<&str> = words.iter().map(|item| item.1.as_str()).collect();
        assert_eq!(words, ["bravo", "alpha", "echo", "charlie", "delta"]);
    }
}
