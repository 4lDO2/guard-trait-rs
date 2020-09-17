# `guard-trait-rs`

Provides safe abstractions for working with memory that can potentially be
shared with another process or kernel (especially `io_uring`), by enforcing
certain restrictions on how the memory can be used after it has been protected
by a guard.
