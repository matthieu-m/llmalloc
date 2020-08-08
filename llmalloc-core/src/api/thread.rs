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

use crate::{Configuration, SocketHandle};
use crate::internals::thread_local::ThreadLocal;

/// Handle to thread-local cache.
pub struct ThreadHandle<C>(ptr::NonNull<ThreadLocal<C>>);

impl<C> ThreadHandle<C>
    where
        C: Configuration
{
    /// Rematerialize from raw pointer.
    ///
    /// #   Safety
    ///
    /// -   Assumes `pointer` points to a valid instance of `ThreadHandle<C>`
    pub unsafe fn from_pointer(pointer: *mut u8) -> ThreadHandle<C> {
        debug_assert!(!pointer.is_null());

        Self::new(ptr::NonNull::new_unchecked(pointer).cast())
    }

    /// Turn into raw pointer.
    pub fn into_pointer(self) -> *mut u8 { self.0.as_ptr() as *mut u8 }

    /// Get associated SocketHandle.
    ///
    /// #   Safety
    ///
    /// -   Assumes that the lifetime and platform are correct.
    pub unsafe fn socket<'a, P>(&self) -> SocketHandle<'a, C, P> {
        let socket = self.0.as_ref().owner();
        debug_assert!(!socket.is_null());

        SocketHandle::from(ptr::NonNull::new_unchecked(socket).cast())
    }

    /// Creates an instance.
    pub(crate) fn new(value: ptr::NonNull<ThreadLocal<C>>) -> Self { Self(value) }

    /// Back to raw.
    pub(crate) fn into_raw(self) -> ptr::NonNull<ThreadLocal<C>> { self.0 }

    /// Yield the underlying reference.
    pub(crate) unsafe fn as_ref(&self) -> &ThreadLocal<C> { self.0.as_ref() }
}
