//! Thread Handle
//!
//! The Thread Handle is a thread-local cache; the user is expected to allocate one for each of the threads they use,
//! and on each allocation to refer to thread-local Thread Handle.
//!
//! The lack of `AtomicThreadHandle` reflects the fact that no two threads should ever contend for a single handle.
//!
//! #   Safety
//!
//! A Thread Handle _assumes_ it is only used from a single thread, and makes no attempt at synchronizing memory
//! accesses. It is Undefined Behavior to use a Thread Handle from multiple threads without external synchronization.

use core::ptr;

use crate::Configuration;
use crate::internals::thread_local::ThreadLocal;

/// Handle to thread-local cache.
pub struct ThreadHandle<C>(ptr::NonNull<ThreadLocal<C>>);

impl<C> ThreadHandle<C>
    where
        C: Configuration
{
    /// Creates an instance.
    pub(crate) fn new(value: ptr::NonNull<ThreadLocal<C>>) -> Self { Self(value) }

    /// Back to raw.
    pub(crate) fn into_raw(self) -> ptr::NonNull<ThreadLocal<C>> { self.0 }

    /// Yield the underlying reference.
    pub(crate) unsafe fn as_ref(&self) -> &ThreadLocal<C> { self.0.as_ref() }
}
