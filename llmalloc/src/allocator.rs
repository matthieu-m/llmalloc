//! Allocator

use core::{alloc, ptr};

use llmalloc_core::{self, Layout};

use crate::{LLConfiguration, Platform, LLPlatform, ThreadLocal, LLThreadLocal};

/// Low-Latency Allocator.
#[derive(Default)]
pub struct LLAllocator;

impl LLAllocator {
    /// Creates an instance.
    pub const fn new() -> Self { Self }

    /// Prepares the socket-local and thread-local structures for allocation.
    ///
    /// Returns Ok if the attempt succeeded, Err otherwise.
    ///
    /// Failure to warm up the current thread may occur if:
    ///
    /// -   The socket-local structure is not ready, and the underlying `Platform` cannot allocate one.
    /// -   The socket-local structure cannot allocate a thread-local structure.
    #[cold]
    pub fn warm_up(&self) -> Result<(), ()> {
        Thread::get().or_else(Thread::initialize).map(|_| ()).ok_or(())
    }

    /// Ensures that at least `target` `HugePage` are allocated on the socket.
    ///
    /// Returns the minimum of the currently allocated number of pages and `target`.
    ///
    /// Failure to meet the `target` may occur if:
    ///
    /// -   The maximum number of `HugePage` a `socket` may contain has been reached.
    /// -   The underlying `Platform` is failing to allocate more `HugePage`.
    #[cold]
    pub fn reserve(&self, target: usize) -> usize {
        if let Some(socket) = Sockets::socket_handle() {
            socket.reserve(target)
        } else {
            0
        }
    }

    /// Allocates `size` bytes of memory, aligned on at least an `alignment` boundary.
    ///
    /// If allocation fails, the returned pointer may be NULL.
    pub fn allocate(&self, layout: Layout) -> *mut u8 {
        if let Some(thread_local) = Thread::get().or_else(Thread::initialize) {
            return thread_local.allocate(layout);
        }

        ptr::null_mut()
    }

    /// Deallocates the memory located at `pointer`.
    ///
    /// #   Safety
    ///
    /// -   Assumes `pointer` has been returned by a prior call to `allocate`.
    /// -   Assumes `pointer` has not been deallocated since its allocation.
    /// -   Assumes the memory pointed by `pointer` is no longer in use.
    pub unsafe fn deallocate(&self, pointer: *mut u8) {
        if pointer.is_null() {
            return;
        }

        if let Some(thread_local) = Thread::get().or_else(Thread::initialize) {
            return thread_local.deallocate(pointer);
        }

        //  If a non-null pointer exists, it _must_ have been allocated, and therefore there should be at least one
        //  non-null socket-handle, somewhere, through which the memory can be returned.
        Sockets::any_socket_handle().deallocate_uncached(pointer);
    }
}

unsafe impl alloc::GlobalAlloc for LLAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 { self.allocate(layout) }

    unsafe fn dealloc(&self, ptr: *mut u8, _: Layout) { self.deallocate(ptr) }
}

//
//  Implementation
//

type AtomicSocketHandle = llmalloc_core::AtomicSocketHandle<'static, LLConfiguration, LLPlatform>;
type DomainHandle = llmalloc_core::DomainHandle<LLConfiguration, LLPlatform>;
type SocketHandle = llmalloc_core::SocketHandle<'static, LLConfiguration, LLPlatform>;
type ThreadHandle = llmalloc_core::ThreadHandle<LLConfiguration>;

//  Domain Handle.
static DOMAIN: DomainHandle = DomainHandle::new(LLPlatform::new());

//  Storage for up to 64 NUMA nodes; it should be vastly overkill.
static SOCKETS: Sockets = Sockets::new();

//  Thread-local.
static THREAD_LOCAL: LLThreadLocal<u8> = LLThreadLocal::new(drop_handle as *const u8);

#[cold]
unsafe fn drop_handle(handle: *mut u8) {
    let thread = ThreadHandle::from_pointer(handle);
    let socket: SocketHandle = thread.socket();
    socket.release_thread_handle(thread);
}

struct Thread(ThreadHandle);

impl Thread {
    //  Returns a pointer to the thread-local instance, if initialized.
    #[inline(always)]
    fn get() -> Option<Thread> {
        let pointer = THREAD_LOCAL.get();
        if pointer.is_null() { None } else { unsafe { Some(Self(ThreadHandle::from_pointer(pointer))) } }
    }

    //  Initializes the thread-local instance and attempts to return a reference to it.
    //
    //  Initialization may fail for any reason, in which case None is returned.
    #[cold]
    #[inline(never)]
    fn initialize() -> Option<Thread> {
        //  Get the handles, can't do anything without both!
        let socket = Sockets::socket_handle()?;
        let thread = socket.acquire_thread_handle()?;

        THREAD_LOCAL.set(thread.into_pointer());

        Self::get()
    }

    //  Allocates `size` bytes of memory, aligned on at least an `alignment` boundary.
    //
    //  If allocation fails, the returned pointer may be NULL.
    #[inline(always)]
    fn allocate(&self, layout: Layout) -> *mut u8 {
        //  Safety:
        //  -   Only uses SocketHandle type.
        let socket: SocketHandle = unsafe { self.0.socket() };

        //  Safety: TODO
        unsafe { socket.allocate(&self.0, layout) }
    }

    //  Deallocates the memory located at `pointer`.
    //
    //  #   Safety
    //
    //  -   Assumes `pointer` has been returned by a prior call to `allocate`.
    //  -   Assumes `pointer` has not been deallocated since its allocation.
    //  -   Assumes the memory pointed by `pointer` is no longer in use.
    #[inline(always)]
    unsafe fn deallocate(&self, pointer: *mut u8) {
        debug_assert!(!pointer.is_null());

        //  Safety:
        //  -   Only uses SocketHandle type.
        let socket: SocketHandle = self.0.socket();

        //  Safety: TODO
        socket.deallocate(&self.0, pointer)
    }
}

struct Sockets([AtomicSocketHandle; 64]);

impl Sockets {
    //  Creates an instance.
    #[cold]
    const fn new() -> Self {
        const fn ash() -> AtomicSocketHandle { AtomicSocketHandle::new() }

        Self([
            //  Line 0: up to 16 instances.
            ash(), ash(), ash(), ash(), ash(), ash(), ash(), ash(), ash(), ash(), ash(), ash(), ash(), ash(), ash(), ash(),
            //  Line 1: up to 32 instances.
            ash(), ash(), ash(), ash(), ash(), ash(), ash(), ash(), ash(), ash(), ash(), ash(), ash(), ash(), ash(), ash(),
            //  Line 2: up to 48 instances.
            ash(), ash(), ash(), ash(), ash(), ash(), ash(), ash(), ash(), ash(), ash(), ash(), ash(), ash(), ash(), ash(),
            //  Line 3: up to 64 instances.
            ash(), ash(), ash(), ash(), ash(), ash(), ash(), ash(), ash(), ash(), ash(), ash(), ash(), ash(), ash(), ash(),
        ])
    }

    //  Returns a SocketHandle for this particular NUMA Node.
    #[cold]
    #[inline(never)]
    fn socket_handle() -> Option<SocketHandle> { SOCKETS.socket_handle_impl() }

    //  Returns the first SocketHandle it finds.
    //
    //  #   Panics
    //
    //  If no handle has been allocated.
    #[cold]
    #[inline(never)]
    fn any_socket_handle() -> SocketHandle { SOCKETS.any_socket_handle_impl() }

    //  Internal; returns a SocketHandle, initialized if need be.
    #[cold]
    fn socket_handle_impl(&self) -> Option<SocketHandle> {
        let index = Self::current_node();
        let atomic_handle = &self.0[index];

        if let Some(socket_handle) = atomic_handle.load() {
            return Some(socket_handle);
        }

        //  There may not be enough memory to allocate a new handle.
        let socket_handle = SocketHandle::new(&DOMAIN)?;

        //  Let's race to see who gets to initialize the handle.
        //
        //  If this thread loses, free the superfluous handle.
        if let Err(socket_handle) = atomic_handle.initialize(socket_handle) {
            //  Safety:
            //  -   No alias exists, the handle has never been shared yet.
            unsafe { socket_handle.close() };
        }

        //  If the race was won, it's initialized, otherwise, it's initialized!
        atomic_handle.load()
    }

    //  Internal; returns the first SocketHandle it finds, or panics if it finds none.
    #[cold]
    fn any_socket_handle_impl(&self) -> SocketHandle {
        for atomic_handle in &self.0[..] {
            if let Some(socket_handle) = atomic_handle.load() {
                return socket_handle;
            }
        }

        unreachable!("How can memory need be deallocated, if no socket handle was ever allocated?");
    }

    #[cold]
    fn current_node() -> usize { DOMAIN.platform().current_node().value() as usize }
}
