//! Original, deliberately narrow compatibility adapter for objc2-foundation 0.2.
//! Implements only the main-queue synchronous API used by this pinned GUI tree.
//! No code from dispatch 0.2.0 is included. Not a general dispatch replacement.
#![cfg(target_vendor = "apple")]
use dispatch2::DispatchQueue;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};

pub struct Queue(&'static DispatchQueue);
impl Queue {
    pub fn main() -> Self {
        Self(DispatchQueue::main())
    }
    /// Blocks until work finishes. Do not call from the main queue itself.
    /// objc2-foundation checks its main-thread marker before calling this.
    pub fn exec_sync<T: Send, F: Send + FnOnce() -> T>(&self, work: F) -> T {
        execute(self.0, work)
    }
}
fn execute<T: Send, F: Send + FnOnce() -> T>(queue: &DispatchQueue, work: F) -> T {
    let mut result = None;
    // Catch on the callback side: Rust panic must never cross the C ABI.
    queue.exec_sync(|| result = Some(catch_unwind(AssertUnwindSafe(work))));
    match result.expect("synchronous dispatch did not execute callback") {
        Ok(value) => value,
        Err(panic) => resume_unwind(panic),
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn synchronous_callback_borrows_returns_and_propagates_panic() {
        let queue = DispatchQueue::global_queue(dispatch2::GlobalQueueIdentifier::Priority(
            dispatch2::DispatchQueueGlobalPriority::Default,
        ));
        let mut value = String::from("borrowed");
        let len = execute(&queue, || {
            value.push_str(" callback");
            value.len()
        });
        assert_eq!(value, "borrowed callback");
        assert_eq!(len, value.len());
        assert!(
            catch_unwind(AssertUnwindSafe(|| execute(&queue, || panic!(
                "callback failure"
            ))))
            .is_err()
        );
    }
}
