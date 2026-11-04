//! Conversions running on worker threads, shared with Quit (DESIGN.md 7.3).
//! A run that ends before Quit keeps its folder and reaches History; one
//! that ends after it is discarded.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard};

use uuid::Uuid;

use super::outcome::Outcome;

#[derive(Clone, Default)]
pub struct Runs(Arc<Mutex<Ended>>);

#[derive(Default)]
struct Ended {
    quitting: bool,
    runs: HashMap<Uuid, (PathBuf, Outcome)>,
}

impl Runs {
    fn lock(&self) -> MutexGuard<'_, Ended> {
        self.0.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// On the worker: the run ended, and its outcome waits for the main
    /// loop. False once Quit was confirmed; the worker then deletes `folder`.
    pub fn end(&self, id: Uuid, folder: PathBuf, outcome: Outcome) -> bool {
        let mut ended = self.lock();
        if !ended.quitting {
            ended.runs.insert(id, (folder, outcome));
        }
        !ended.quitting
    }

    /// On the main loop: the outcome `end` kept, unless Quit took it.
    pub fn take(&self, id: Uuid) -> Option<(PathBuf, Outcome)> {
        self.lock().runs.remove(&id)
    }

    /// Quit confirmed: no run ends after this. The runs that ended before
    /// it and have not reached the main loop yet.
    pub fn quit(&self) -> Vec<(Uuid, PathBuf, Outcome)> {
        let mut ended = self.lock();
        ended.quitting = true;
        ended
            .runs
            .drain()
            .map(|(id, (folder, outcome))| (id, folder, outcome))
            .collect()
    }

    pub fn quitting(&self) -> bool {
        self.lock().quitting
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::outcome::Failure;

    fn outcome() -> Outcome {
        Outcome::Failed(Failure::new("x", "y"))
    }

    #[test]
    fn a_run_that_ended_before_quit_is_kept_and_one_after_is_not() {
        let runs = Runs::default();
        let (before, after) = (Uuid::new_v4(), Uuid::new_v4());
        assert!(runs.end(before, "/out/before".into(), outcome()));
        assert!(!runs.quitting());

        let kept = runs.quit();
        assert_eq!(kept.len(), 1);
        assert_eq!((kept[0].0, kept[0].1.clone()), (before, PathBuf::from("/out/before")));
        assert!(runs.take(before).is_none(), "Quit finished it; the main loop must not again");

        assert!(!runs.end(after, "/out/after".into(), outcome()), "its worker deletes the folder");
        assert!(runs.take(after).is_none());
        assert!(runs.quit().is_empty());
        assert!(runs.quitting());
    }

    #[test]
    fn the_main_loop_takes_each_outcome_once() {
        let runs = Runs::default();
        let id = Uuid::new_v4();
        assert!(runs.end(id, "/out/a".into(), outcome()));
        assert_eq!(runs.take(id).map(|(folder, _)| folder), Some(PathBuf::from("/out/a")));
        assert!(runs.take(id).is_none());
        assert!(runs.quit().is_empty());
    }
}
