//! One compiled task type for the command's own runtime tasks.
//!
//! `tokio::task::spawn_blocking` and `tokio::spawn` compile the runtime's task
//! code (polling, joining, cancelling and freeing a task: about 6 KB per
//! closure here, and again per scheduler for `tokio::spawn`) for every
//! closure or future type they are given. [`blocking`] and [`spawn`] give them
//! one boxed type instead, so that code is compiled once. A task starts when
//! it is spawned, as before. [`Blocking`] stands for the `JoinHandle`: it
//! resolves to the closure's value or to the same `JoinError` (panic or
//! cancellation), and dropping it detaches the task. [`spawn`] still returns
//! the task's `JoinHandle`.

use std::any::Any;
use std::future::Future;
use std::marker::PhantomData;
use std::pin::Pin;
use std::task::{Context, Poll};
use tokio::task::{JoinError, JoinHandle};

/// The boxed value of every blocking task.
type Value = Box<dyn Any + Send>;

/// `tokio::task::spawn_blocking(work)`.
pub(crate) fn blocking<T: Send + 'static>(
    work: impl FnOnce() -> T + Send + 'static,
) -> Blocking<T> {
    let work: Box<dyn FnOnce() -> Value + Send> = Box::new(move || Box::new(work()));
    Blocking {
        handle: tokio::task::spawn_blocking(work),
        value: PhantomData,
    }
}

/// The handle of a [`blocking`] task, awaited for the closure's value.
pub(crate) struct Blocking<T> {
    handle: JoinHandle<Value>,
    value: PhantomData<fn() -> T>,
}

impl<T> Blocking<T> {
    /// `JoinHandle::is_finished`.
    pub(crate) fn is_finished(&self) -> bool {
        self.handle.is_finished()
    }
}

impl<T: 'static> Future for Blocking<T> {
    type Output = Result<T, JoinError>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        Pin::new(&mut self.handle).poll(cx).map(|done| {
            done.map(|value| {
                *value
                    .downcast::<T>()
                    .expect("a blocking task returns its closure's value")
            })
        })
    }
}

/// `tokio::spawn(future)` for a future without a value.
pub(crate) fn spawn(future: impl Future<Output = ()> + Send + 'static) -> JoinHandle<()> {
    let future: Pin<Box<dyn Future<Output = ()> + Send>> = Box::pin(future);
    tokio::spawn(future)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn blocking_returns_the_value_starts_at_once_and_reports_panics() {
        assert_eq!(blocking(|| 40 + 2).await.unwrap(), 42);
        assert_eq!(blocking(|| String::from("text")).await.unwrap(), "text");

        // The work runs before the handle is first polled, and the handle
        // tells whether it has finished.
        let (send, receive) = std::sync::mpsc::channel();
        let (release, wait) = std::sync::mpsc::channel::<()>();
        let pending = blocking(move || {
            send.send(7).unwrap();
            wait.recv().unwrap();
        });
        let ten_seconds = std::time::Duration::from_secs(10);
        assert_eq!(receive.recv_timeout(ten_seconds).unwrap(), 7);
        assert!(!pending.is_finished());
        release.send(()).unwrap();
        let started = std::time::Instant::now();
        while !pending.is_finished() {
            assert!(started.elapsed() < ten_seconds);
            tokio::task::yield_now().await;
        }
        pending.await.unwrap();

        let error = blocking(|| -> u8 { panic!("boom") }).await.unwrap_err();
        assert!(error.is_panic());
        assert_eq!(error.into_panic().downcast_ref::<&str>(), Some(&"boom"));
    }

    #[tokio::test]
    async fn spawn_runs_the_future_to_its_end() {
        let (send, receive) = tokio::sync::oneshot::channel();
        let handle = spawn(async move { send.send(5).unwrap() });
        handle.await.unwrap();
        assert_eq!(receive.await.unwrap(), 5);
        let error = spawn(async { panic!("boom") }).await.unwrap_err();
        assert!(error.is_panic());
    }
}
