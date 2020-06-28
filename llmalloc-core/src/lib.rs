#![no_std]

#![deny(missing_docs)]

//! Building blocks for a low-latency allocator.
//!
//! llmalloc-core is a set of building blocks to build a custom low-latency malloc replacement with ease. It contains:
//! -   A platform trait, used to allocate large raw blocks of memory to be carved up.
//! -   A handful of user-facing types representing NUMA node and thread data, leaving it up to the user to arrange
//!     those as desired in memory.

mod api;
mod internals;
mod utils;

pub use api::*;
