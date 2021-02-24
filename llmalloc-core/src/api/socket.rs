//! Socket Handle.
//!
//! All Thread Handles instances sharing a given Socket Handle instance will exchange memory between themselves.
//!
//! For a simple allocator, a simple Socket Handle instance is sufficient.
//!
//! Using multiple Socket Handle instances allows:
//!
//! -   Better performance when a given Socket Handle instance is local to a NUMA node.
//! -   Less contention, by reducing the number of ThreadLocal instances contending over it.
//!
//! The name comes from the socket in which a CPU is plugged in, as a recommendation to use one instance of Socket Handle
//! for each socket.

use core::{
    alloc::Layout,
    ptr::{self, NonNull},
    sync::atomic::{AtomicPtr, Ordering},
};

use crate::{Configuration, DomainHandle, Platform, ThreadHandle};
use crate::internals::socket_local::SocketLocal;

/// A handle to socket-local memory structures.
///
/// The socket-local memory structures are thread-safe, but the handle itself is not. For a thread-safe handle, please
/// see `AtomicSocketHandle`.
pub struct SocketHandle<'a, C, P>(NonNull<SocketLocal<'a, C, P>>);

impl<'a, C, P> SocketHandle<'a, C, P>
where
    C: Configuration,
    P: Platform,
{
    /// Creates a new instance of SocketHandle.
    pub fn new(domain: &'a DomainHandle<C, P>) -> Option<Self> {
        SocketLocal::bootstrap(domain.as_raw()).map(SocketHandle)
    }

    /// Returns whether the layout is valid, or not, for use with `SocketLocal`.
    pub fn is_valid_layout(layout: Layout) -> bool { SocketLocal::<C, P>::is_valid_layout(layout) }

    /// Attempts to acquire a `ThreadHandle` from within the buffer area of the first HugePage.
    ///
    /// Returns a valid pointer to `ThreadHandle` if successful, and None otherwise.
    pub fn acquire_thread_handle(&self) -> Option<ThreadHandle<C>> {
        //  Safety:
        //  -   Local lifetime.
        let socket_local = unsafe { self.0.as_ref() };

        socket_local.acquire_thread_local().map(ThreadHandle::new)
    }

    /// Releases a `ThreadHandle`.
    ///
    /// #   Safety
    ///
    /// -   Assumes that the `ThreadHandle` came from `self`.
    pub unsafe fn release_thread_handle(&self, handle: ThreadHandle<C>) {
        //  Safety:
        //  -   Local lifetime.
        let socket_local = self.0.as_ref();

        //  Safety:
        //  -   `handle` is assumed to come from `socket`.
        socket_local.release_thread_local(handle.into_raw());
    }

    /// Attempts to ensure that at least `target` `HugePage` are allocated on the socket.
    ///
    /// Returns the minimum of the currently allocated number of pages and `target`.
    ///
    /// Failure to meet the `target` may occur if:
    ///
    /// -   The maximum number of `HugePage` a `socket` may contain has been reached.
    /// -   The underlying `Platform` is failing to allocate more `HugePage`.
    pub fn reserve(&self, target: usize) -> usize {
        //  Safety:
        //  -   Local lifetime.
        let socket_local = unsafe { self.0.as_ref() };

        socket_local.reserve(target)
    }

    /// Deallocates all HugePages allocated by the socket.
    ///
    /// This may involve deallocating the memory used by the socket itself, after which it can no longer be used.
    ///
    /// #   Safety
    ///
    /// -   Assumes that none of the memory allocated by the socket is still in use, with the possible exception of the
    ///     memory used by `self`.
    pub unsafe fn close(self) {
        //  Safety:
        //  -   Local lifetime.
        let socket_local = self.0.as_ref();

        //  Safety:
        //  -   Assumes that none of the memory allocated by the socket is still in use.
        //  -   Assumes that `P` can deallocate itself.
        socket_local.close()
    }

    /// Allocates a fresh block of memory as per the specified layout.
    ///
    /// May return a null pointer if the allocation request cannot be satisfied.
    ///
    /// #   Safety
    ///
    /// The caller may assume that if the returned pointer is not null then:
    /// -   The number of usable bytes is _greater than or equal_ to `layout.size()`.
    /// -   The pointer is _at least_ aligned to `layout.align()`.
    ///
    /// `allocate` assumes that:
    /// -   `thread_handle` is not concurrently accessed by another thread.
    /// -   `thread_handle` belongs to this socket.
    /// -   `layout` is valid, as per `Self::is_valid_layout`.
    #[inline(always)]
    pub unsafe fn allocate(&self, thread_handle: &ThreadHandle<C>, layout: Layout) -> Option<NonNull<u8>> {
        //  Safety:
        //  -   Local lifetime.
        let socket_local = self.0.as_ref();
        let thread_local = thread_handle.as_ref();

        socket_local.allocate(thread_local, layout)
    }

    /// Deallocates the supplied block of memory.
    ///
    /// #   Safety
    ///
    /// The caller should no longer reference the memory after calling this function.
    ///
    /// `deallocate` assumes that:
    /// -   `thread_handle` is not concurrently accessed by another thread.
    /// -   `thread_handle` belongs to this socket.
    /// -   `ptr` is a value allocated by an instance of `Self`, and the same underlying `Platform`.
    #[inline(always)]
    pub unsafe fn deallocate(&self, thread_handle: &ThreadHandle<C>, ptr: NonNull<u8>) {
        //  Safety:
        //  -   Local lifetime.
        let socket_local = self.0.as_ref();
        let thread_local = thread_handle.as_ref();

        socket_local.deallocate(thread_local, ptr)
    }

    /// Deallocates the supplied block of memory.
    ///
    /// Unlike `deallocate`, the pointer is not cached for reuse on the local thread; as a result, this call may be
    /// slightly more costly.
    ///
    /// #   Safety
    ///
    /// The caller should no longer reference the memory after calling this function.
    ///
    /// `deallocate` assumes that:
    /// -   `thread_handle` is not concurrently accessed by another thread.
    /// -   `thread_handle` belongs to this socket.
    /// -   `ptr` is a value allocated by an instance of `Self`, and the same underlying `Platform`.
    #[inline(always)]
    pub unsafe fn deallocate_uncached(&self, ptr: NonNull<u8>) {
        //  Safety:
        //  -   Local lifetime.
        let socket_local = self.0.as_ref();

        socket_local.deallocate_uncached(ptr)
    }
}

impl<'a, C, P> SocketHandle<'a, C, P> {
    /// Creates a new instance from its content.
    pub(crate) fn from(socket: NonNull<SocketLocal<'a, C, P>>) -> Self { SocketHandle(socket) }
}

impl<'a, C, P> Clone for SocketHandle<'a, C, P> {
    fn clone(&self) -> Self { *self }
}

impl<'a, C, P> Copy for SocketHandle<'a, C, P> {}

/// A thread-safe handle to socket-local memory structures.
///
/// #   Recommendation
///
/// There is a slight potential cost to using `AtomicSocketHandle` instead of `SocketHandle`, hence it is recommended
/// to keep a global array of `AtomicSocketHandle` indexed by socket _and_ a thread-local `SocketHandle` on each
/// thread:
///
/// -   The global array to avoid allocating more than one `SocketHandle` per socket.
/// -   The thread-local `SocketHandle` is used to speed-up allocation and deallocation.
pub struct AtomicSocketHandle<'a, C, P>(AtomicPtr<SocketLocal<'a, C, P>>);

impl <'a, C, P> AtomicSocketHandle<'a, C, P> {
    /// Creates a null instance.
    pub const fn new() -> Self { Self(AtomicPtr::new(ptr::null_mut())) }

    /// Initializes the instance with the given handle.
    ///
    /// If `self` is NOT currently None, then the initialization fails and the `handle` is returned.
    pub fn initialize(&self, handle: SocketHandle<'a, C, P>) -> Result<(), SocketHandle<'a, C, P>> {
        self.0.compare_exchange(ptr::null_mut(), handle.0.as_ptr(), Ordering::Relaxed, Ordering::Relaxed)
            .and(Ok(()))
            .or(Err(handle))
    }

    /// Loads the value of the handle, it may be None.
    pub fn load(&self) -> Option<SocketHandle<'a, C, P>> {
        NonNull::new(self.0.load(Ordering::Relaxed)).map(SocketHandle)
    }

    /// Stores a handle, discards the previous handle if any.
    ///
    /// This method should generally be used only at start-up, to initialize the instance.
    pub fn store(&self, handle: SocketHandle<'a, C, P>) {
        self.0.store(handle.0.as_ptr(), Ordering::Relaxed);
    }
}

impl<'a, C, P> Default for AtomicSocketHandle<'a, C, P> {
    fn default() -> Self { Self(AtomicPtr::new(ptr::null_mut())) }
}
