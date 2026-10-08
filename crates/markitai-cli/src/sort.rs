//! The core's slice sorts, which share one compiled stable index sort instead
//! of specializing the full sorting algorithm for every element and comparator
//! type. Ordered collections likewise insert entries individually to avoid
//! specializing bulk collection.
//!
//! Comparators must be total orders. For collections, equal keys must also be
//! interchangeable: maps retain the last value for a repeated key.

pub(crate) use markitai_core::sort::{by, by_key};
use std::collections::{BTreeMap, BTreeSet};

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collections_match_collect() {
        let pairs = [("b", 1), ("a", 2), ("b", 3), ("c", 4), ("a", 5)];
        assert_eq!(map(pairs), pairs.into_iter().collect::<BTreeMap<_, _>>());
        assert_eq!(map(pairs)["b"], 3);
        let words = ["b", "a", "b", "c", "a"];
        assert_eq!(set(words), words.into_iter().collect::<BTreeSet<_>>());
    }
}
