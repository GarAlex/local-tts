# Main-queue adapter

Original project code, MIT (see the project LICENSE). This crate supplies exactly
`Queue::main().exec_sync` for objc2-foundation 0.2.2, using dispatch2 0.3.1.
It does not copy dispatch 0.2.0 code or claim compatibility with its entire API.
Cargo's root patch selects it; a new transitive use must compile against the
limited interface or fail. The pinned enabled dependency tree uses only this API.

Synchronous work may borrow caller data and return a Send value. Callback panics
are captured before crossing C and resumed on the calling thread. Calling the main
queue synchronously from the main thread is invalid; objc2-foundation's existing
main-thread check prevents that. Tests use a global queue to exercise callback
semantics without needing a macOS application run loop.
