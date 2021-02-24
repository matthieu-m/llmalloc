//! Platform
//!
//! The Platform trait is used to request memory directly from the Platform. By abstracting the underlying platform,
//! it becomes possible to easily port the code to a different OS, or even to a bare-metal target.

use core::{
    alloc::Layout,
    ptr::NonNull,
};

/// Abstraction of platform specific memory allocation and deallocation.
pub trait Platform {
    /// Allocates a fresh block of memory as per the specified layout.
    ///
    /// May return a null pointer if the allocation request cannot be satisfied.
    ///
    /// #   Safety
    ///
    /// The caller may assume that if the returned pointer is not null then:
    /// -   The number of usable bytes is _at greater than or equal_ to `layout.size()`.
    /// -   The pointer is _at least_ aligned to `layout.align()`.
    ///
    /// `allocate` assumes that:
    /// -   `layout.size()` is a multiple of `layout.align()`.
    /// -   `layout.align()` is non-zero, and is a power of 2.
    unsafe fn allocate(&self, layout: Layout) -> Option<NonNull<u8>>;

    /// Deallocates the supplied block of memory.
    ///
    /// #   Safety
    ///
    /// The caller should no longer reference the memory after calling this function.
    ///
    /// `deallocate` assumes that:
    /// -   `pointer` was allocated by this instance of `Platform`, with `layout` as argument.
    /// -   `pointer` is the value returned by `Plaform`, and not an interior pointer.
    unsafe fn deallocate(&self, pointer: NonNull<u8>, layout: Layout);
}
