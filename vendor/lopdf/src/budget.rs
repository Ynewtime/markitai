//! A decompression budget shared by every stream decoded for one caller
//! (markitai).
//!
//! A limit per stream does not bound a document: a small file can hold
//! thousands of streams, each inflating up to the limit, and a stream of
//! nested filters inflates by each filter's ratio in turn. Inside
//! [`with_decompress_budget`], every stream decoded on this thread, and by
//! the loader's workers for this thread, draws from one budget, and a stream
//! that would pass what remains fails with
//! [`crate::DecompressError::MemoryLimitExceeded`] naming the budget. Each
//! decoder stops at what remains, so no single stream allocates past it.

use std::cell::RefCell;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Bytes the streams of one caller may still decompress to.
#[derive(Debug)]
pub struct DecompressBudget {
    limit: usize,
    remaining: AtomicUsize,
}

impl DecompressBudget {
    /// The budget's size in bytes.
    pub fn limit(&self) -> usize {
        self.limit
    }

    /// Bytes still allowed.
    pub fn remaining(&self) -> usize {
        self.remaining.load(Ordering::Relaxed)
    }

    /// Takes `bytes`; false, taking nothing, when fewer remain.
    pub(crate) fn charge(&self, bytes: usize) -> bool {
        let mut left = self.remaining.load(Ordering::Relaxed);
        loop {
            let Some(next) = left.checked_sub(bytes) else {
                return false;
            };
            match self
                .remaining
                .compare_exchange_weak(left, next, Ordering::Relaxed, Ordering::Relaxed)
            {
                Ok(_) => return true,
                Err(actual) => left = actual,
            }
        }
    }
}

thread_local! {
    static CURRENT: RefCell<Option<Arc<DecompressBudget>>> = const { RefCell::new(None) };
}

/// Runs `work` with every stream decoded on this thread drawing from one
/// budget of `limit` bytes. Inside a budget already in force, `work` draws
/// from that one instead, so nested calls never add to a document's total.
pub fn with_decompress_budget<T>(limit: usize, work: impl FnOnce() -> T) -> T {
    if current_decompress_budget().is_some() {
        return work();
    }
    let budget = Arc::new(DecompressBudget {
        limit,
        remaining: AtomicUsize::new(limit),
    });
    with_current_decompress_budget(Some(budget), work)
}

/// The budget in force on this thread, to hand to work moved to another.
pub fn current_decompress_budget() -> Option<Arc<DecompressBudget>> {
    CURRENT.with(|current| current.borrow().clone())
}

/// Runs `work` under `budget` (one taken from [`current_decompress_budget`]
/// on another thread), restoring this thread's own afterwards.
pub fn with_current_decompress_budget<T>(budget: Option<Arc<DecompressBudget>>, work: impl FnOnce() -> T) -> T {
    struct Restore(Option<Arc<DecompressBudget>>);
    impl Drop for Restore {
        fn drop(&mut self) {
            let previous = self.0.take();
            CURRENT.with(|current| *current.borrow_mut() = previous);
        }
    }
    let _restore = Restore(CURRENT.with(|current| current.replace(budget)));
    work()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_budget_is_shared_nested_and_restored() {
        assert!(current_decompress_budget().is_none());
        with_decompress_budget(100, || {
            let budget = current_decompress_budget().unwrap();
            assert!(budget.charge(60));
            assert!(!budget.charge(41));
            assert_eq!(budget.remaining(), 40);
            // A nested call keeps the budget in force.
            with_decompress_budget(1_000, || {
                assert_eq!(current_decompress_budget().unwrap().remaining(), 40);
            });
            // Another thread draws from it once handed over.
            let handed = current_decompress_budget();
            std::thread::spawn(move || {
                with_current_decompress_budget(handed, || {
                    assert!(current_decompress_budget().unwrap().charge(40));
                });
                assert!(current_decompress_budget().is_none());
            })
            .join()
            .unwrap();
            assert_eq!(budget.remaining(), 0);
        });
        assert!(current_decompress_budget().is_none());
    }
}
