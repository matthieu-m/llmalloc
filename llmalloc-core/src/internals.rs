//! The internals of llmalloc-core.
//!
//! The internals provide all the heavy-lifting.

pub mod blocks;
pub mod huge_allocator;
pub mod huge_page;
pub mod large_page;
pub mod socket_local;
pub mod thread_local;

mod atomic;
