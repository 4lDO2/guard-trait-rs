# `guard-trait-rs`
[![Crates.io](https://img.shields.io/crates/v/guard-trait.svg)](https://crates.io/crates/guard-trait)
[![Documentation](https://docs.rs/guard-trait/badge.svg)](https://docs.rs/guard-trait/)

Provides safe abstractions for working with memory that can potentially be
shared with another process or kernel (especially `io_uring`), by enforcing
certain restrictions on how the memory can be used after it has been protected
by a guard.
