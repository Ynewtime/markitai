//! Facts about the elements of one parsed page that several passes ask for.
//!
//! The choice of the content region, the reading of its furniture, footnote
//! recovery and the cleaner each ask the elements of a page whether they are
//! hidden, page chrome or notes, and some ask again for every ancestor of an
//! element. Each fact is computed the first time it is asked about an element
//! and kept, in a table with a cell per node of the page's tree, for the rest
//! of the conversion.

use scraper::ElementRef;
use std::cell::Cell;
use std::hash::{Hash, Hasher};

/// What is kept of an element: whether it is hidden ([`super::is_hidden`]),
/// full-page chrome ([`super::article::excluded`]), a note
/// ([`super::article::note`]) or a note's context ([`super::note_context`]).
#[derive(Clone, Copy)]
pub(super) enum Fact {
    Hidden,
    Excluded,
    Note,
    NoteContext,
}

/// A node's position among the nodes of its tree. ego_tree keeps a tree's
/// nodes in one vector and numbers each by its place in it (a `NodeId` is
/// that place plus one), and an id hashes as that number alone, which is all
/// this hasher keeps; any other way of hashing an id fails loudly instead of
/// giving a wrong position.
pub(super) fn position(id: impl Hash) -> usize {
    struct Number(usize);
    impl Hasher for Number {
        fn finish(&self) -> u64 {
            self.0 as u64
        }
        fn write(&mut self, _: &[u8]) {
            unreachable!("a node id hashes as one number");
        }
        fn write_usize(&mut self, number: usize) {
            self.0 = number;
        }
    }
    let mut number = Number(0);
    id.hash(&mut number);
    number.0 - 1
}

/// The facts known about the elements of one tree: two bits per fact and
/// node, whether it is known and whether it holds.
pub(super) struct Facts {
    cells: Vec<Cell<u8>>,
    /// The tree's address, to check that every element asked about is one of
    /// its own.
    #[cfg(debug_assertions)]
    tree: usize,
}

impl Facts {
    /// No facts yet about the elements of `node`'s tree.
    pub(super) fn new(node: ElementRef<'_>) -> Self {
        let tree = node.tree();
        #[cfg(debug_assertions)]
        for (index, node) in tree.nodes().enumerate() {
            assert_eq!(position(node.id()), index, "node positions");
        }
        Self {
            cells: vec![Cell::new(0); tree.values().len()],
            #[cfg(debug_assertions)]
            tree: std::ptr::from_ref(tree).addr(),
        }
    }

    /// Whether `fact` holds for `element`; `compute` says so the first time.
    pub(super) fn remember(
        &self,
        element: ElementRef<'_>,
        fact: Fact,
        compute: impl FnOnce() -> bool,
    ) -> bool {
        #[cfg(debug_assertions)]
        assert_eq!(
            std::ptr::from_ref(element.tree()).addr(),
            self.tree,
            "an element of another tree"
        );
        let cell = &self.cells[position(element.id())];
        let known = 1 << (2 * fact as u8);
        let holds = known << 1;
        if cell.get() & known != 0 {
            return cell.get() & holds != 0;
        }
        let value = compute();
        // `compute` may have learned other facts about the element meanwhile.
        cell.set(cell.get() | known | if value { holds } else { 0 });
        value
    }

    /// [`super::is_hidden`], kept.
    pub(super) fn hidden(&self, element: ElementRef<'_>) -> bool {
        self.remember(element, Fact::Hidden, || super::is_hidden(element))
    }
}
