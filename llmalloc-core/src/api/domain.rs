//! Domain Handle.
//!
//! An instance of the Domain Handle defines an allocation Domain:
//!
//! -   A single allocation Domain is connected to 1 to N Sockets.
//! -   A single Socket is connected to 1 to N Threads.
//!
//! The allocation Domain is its own island, memory wise:
//!
//! -   Any piece of memory allocated by a connected Socket (and Thread) is served by the Domain.
//! -   In exchange, pieces of memory MUST be deallocated by a Socket (and Thread) connected to its original Domain.
//!
//! Typically, applications will use a global Domain.

use crate::internals::huge_allocator::HugeAllocator;

/// Domain Handle.
///
/// By design, `free` only takes a pointer. At the same time, the `Platform` abstraction requires that the layout of
/// the pointer to be deallocated is passed.
///
/// The `DomainHandle` bridges the gap by recording the original layout on allocation and providing it back on deallocation.
///
/// #   Limitation
///
/// A single `DomainHandle` is limited to 128 allocations above C::HUGE_PAGE_SIZE.
pub struct DomainHandle<C, P>(HugeAllocator<C, P>);

impl<C, P> DomainHandle<C, P> {
    /// Creates a Domain.
    ///
    /// The Domain created will allocate memory from the `platform`, and return it to the `platform`.
    pub fn new(platform: P) -> Self { Self(HugeAllocator::new(platform)) }

    pub(crate) fn as_raw(&self) -> &HugeAllocator<C, P> { &self.0 }
}

impl<C, P> Default for DomainHandle<C, P>
    where
        P: Default
{
    fn default() -> Self { Self::new(P::default()) }
}
