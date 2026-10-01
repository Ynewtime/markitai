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

/// [`slice::sort_by`], compiled once per element type.
#[inline(never)]
pub(crate) fn stable<T>(v: &mut [T], compare: &mut dyn FnMut(&T, &T) -> Ordering) {
    v.sort_by(|a, b| compare(a, b));
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
}
