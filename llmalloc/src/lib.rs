#![no_std]
#![deny(missing_docs)]

//! A Low-Latency Memory Allocator library.
//!
//! The type `LLAllocator` provides a low-latency memory allocator, as a drop-in replacement for regular allocators.
//!
//! #   Warning
//!
//! This low-latency memory allocator is not suitable for all applications.
//!
//! See the README.md file for the limitations and trade-offs made.

mod allocator;
mod platform;

pub use allocator::LLAllocator;

use platform::{LLConfiguration, Platform, LLPlatform, ThreadLocal, LLThreadLocal};
