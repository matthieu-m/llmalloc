//! SocketLocal
//!
//! All ThreadLocal instances sharing a given SocketLocal instance will exchange memory between themselves.
//!
//! For a simple allocator, a simple SocketLocal instance is sufficient.
//!
//! Using multiple SocketLocal instances allows:
//!
//! -   Better performance when a given SocketLocal instance is local to a NUMA node.
//! -   Less contention, by reducing the number of ThreadLocal instances contending over it.
//!
//! The name comes from the socket in which a CPU is plugged in, as a recommendation to use one instance of SocketLocal
//! for each socket.

use core::{alloc::Layout, cmp, marker, mem, num, ptr, slice, sync::atomic};

use crate::{Category, ClassSize, Configuration, Platform, PowerOf2, Properties};
use crate::internals::{
    cells::{CellAtomicForeign, CellForeign, CellForeignList, CellAtomicForeignStack},
    huge_allocator::HugeAllocator,
    huge_page::HugePage,
    large_page::{LargePage, LargePageStack},
    thread_local::{ThreadLocal},
};
use crate::utils;

/// SocketLocal.
#[repr(align(128))]
#[repr(C)]
pub(crate) struct SocketLocal<'a, C, P> {
    //  Linked-lists of already allocated LargePages.
    large_pages: [LargePageStack; 64],
    //  Huge Pages allocated, to be carved into Large allocations.
    huge_pages: HugePagesManager<C, P>,
    //  Management of buffer area for ThreadLocals.
    thread_locals: ThreadLocalsManager<C>,
    //  Huge (unmanaged) allocations, directly allocated/deallocated by the OS.
    huge_allocator: &'a HugeAllocator<C, P>,
}

impl<'a, C, P> SocketLocal<'a, C, P>
where
    C: Configuration + 'a,
    P: Platform + 'a,
{
    /// Attempts to allocate a `HugePage` and creates an instance of SocketLocal within it.
    ///
    /// Returns a valid pointer to Self if the bootstrap is successful, and None otherwise.
    pub(crate) fn bootstrap(huge_allocator: &'a HugeAllocator<C, P>) -> Option<ptr::NonNull<Self>> {
        let page = HugePagesManager::<C, P>::allocate_huge_page(huge_allocator.platform(), ptr::null_mut());

        if page.is_null() {
            return None;
        }

        //  Safety:
        //  -   `page` is not null.
        //  -   `page` is assumed to be suitably aligned.
        //  -   `page` points to an exclusive memory area.
        let huge_page = unsafe { &mut *page };

        let place = huge_page.buffer_mut();

        //  Align start of slice.
        let place = {
            let before = place.as_mut_ptr() as usize;
            let after = PowerOf2::align_of::<Self>().round_up(before);
            let increment = after - before;

            if increment == 0 {
                place
            } else if increment >= place.len() {
                &mut []
            } else {
                //  Safety:
                //  -   `increment` is less than `place.len()`.
                unsafe { place.get_unchecked_mut(increment..) }
            }
        };

        let bootstrap_size = place.len();

        if bootstrap_size < mem::size_of::<Self>() {
            //  Safety:
            //  -   The `page` is unused.
            unsafe { HugePagesManager::<C, P>::deallocate_huge_page(huge_allocator.platform(), page) };
            return None;
        }

        let (socket_place, thread_locals_place) = place.split_at_mut(mem::size_of::<Self>());

        let thread_locals = ThreadLocalsManager::new(socket_place.as_mut_ptr() as *mut (), thread_locals_place);

        //  Safety:
        //  -   `socket_place` is sufficiently sized.
        //  -   `socket_place` is sufficiently aligned.
        let mut result = unsafe { Self::initialize(socket_place, huge_allocator) };

        huge_page.set_owner(result.as_ptr() as *mut ());

        //  Safety:
        //  -   `result` is not null.
        //  -   `result` is exclusive.
        let socket = unsafe { result.as_mut() };
        socket.huge_pages.0[0].store(huge_page);
        socket.thread_locals = thread_locals;

        Some(result)
    }

    /// Returns whether the layout is valid, or not, for use with `SocketLocal`.
    pub(crate) fn is_valid_layout(layout: Layout) -> bool {
        layout.size() != 0 &&
            layout.align().count_ones() == 1 &&
            layout.align() <= C::HUGE_PAGE_SIZE.value() &&
            layout.size() % unsafe { PowerOf2::new_unchecked(layout.align()) } == 0
    }

    /// Attempts to ensure that at least `target` `HugePage` are allocated on the socket.
    ///
    /// Returns the minimum of the currently allocated number of pages and `target`.
    pub(crate) fn reserve(&self, target: usize) -> usize {
        self.huge_pages.reserve(target, self.as_owner(), self.platform())
    }

    /// Deallocates all HugePagesManager allocated by the socket.
    ///
    /// This may involve deallocating the memory used by the socket itself, after which it can no longer be used.
    ///
    /// #   Safety
    ///
    /// -   Assumes that none of the memory allocated by the socket is still in use, with the possible exception of the
    ///     memory used by `self`.
    pub(crate) unsafe fn close(&self) {
        self.huge_pages.close(self.as_owner(), self.platform());
    }

    /// Attempts to acquire a `ThreadLocal` from within the buffer area of the first HugePage.
    ///
    /// Returns a valid pointer to `ThreadLocal` if successful, and None otherwise.
    pub(crate) fn acquire_thread_local(&self) -> Option<ptr::NonNull<ThreadLocal<C>>> {
        self.thread_locals.acquire()
    }

    /// Releases a `ThreadLocal`.
    ///
    /// #   Safety
    ///
    /// -   Assumes that the `ThreadLocal` comes from `self`.
    pub(crate) unsafe fn release_thread_local(&self, thread_local: ptr::NonNull<ThreadLocal<C>>) {
        debug_assert!((self.as_owner() as usize) < (thread_local.as_ptr() as usize));
        debug_assert!((thread_local.as_ptr() as usize) < (self.thread_locals.end as usize));

        //  Safety:
        //  -   `thread_local` is not null.
        thread_local.as_ref().flush(|page| Self::catch_large_page(page));

        self.thread_locals.release(thread_local)
    }

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
    /// -   `thread_local` is not concurrently accessed by another thread.
    /// -   `layout` is valid, as per `Self::is_valid_layout`.
    #[inline(always)]
    pub(crate) unsafe fn allocate(&self, thread_local: &ThreadLocal<C>, layout: Layout) -> *mut u8 {
        debug_assert!(Self::is_valid_layout(layout));

        match Properties::<C>::category_of_size(layout.size()) {
            Category::Normal => self.allocate_normal(thread_local, layout),
            Category::Large => self.allocate_large(layout),
            Category::Huge => self.allocate_huge(layout),
        }
    }

    /// Deallocates the supplied block of memory.
    ///
    /// #   Safety
    ///
    /// The caller should no longer reference the memory after calling this function.
    ///
    /// `deallocate` assumes that:
    /// -   `thread_local` is not concurrently accessed by another thread.
    /// -   `ptr` is a value allocated by an instance of `Self`, and the same underlying `Platform`.
    #[inline(always)]
    pub(crate) unsafe fn deallocate(&self, thread_local: &ThreadLocal<C>, ptr: *mut u8) {
        match Properties::<C>::category_of_pointer(ptr) {
            Category::Normal => self.deallocate_normal(thread_local, ptr),
            Category::Large => self.deallocate_large(ptr),
            Category::Huge => self.deallocate_huge(ptr),
        }
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
    /// -   `thread_local` is not concurrently accessed by another thread.
    /// -   `ptr` is a value allocated by an instance of `Self`, and the same underlying `Platform`.
    #[inline(always)]
    pub(crate) unsafe fn deallocate_uncached(&self, ptr: *mut u8) {
        match Properties::<C>::category_of_pointer(ptr) {
            Category::Normal => self.deallocate_normal_uncached(ptr),
            Category::Large => self.deallocate_large(ptr),
            Category::Huge => self.deallocate_huge(ptr),
        }
    }

    //  Internal; Creates a new instance of SocketLocal.
    fn new(huge_allocator: &'a HugeAllocator<C, P>) -> Self {
        let large_pages = unsafe { mem::zeroed() };
        let huge_pages = HugePagesManager::new();
        let thread_locals = ThreadLocalsManager::default();

        SocketLocal { large_pages, huge_pages, huge_allocator, thread_locals, }
    }

    //  Internal; Returns a reference to the Platform.
    fn platform(&self) -> &P { self.huge_allocator.platform() }

    //  Internal; In-place constructs an instance of SocketLocal.
    // 
    //  #   Safety
    // 
    //  -   Assumes that `place` is of sufficient size and alignment.
    unsafe fn initialize( place: &mut [u8], huge_allocator: &'a HugeAllocator<C, P>) -> ptr::NonNull<Self> {
        debug_assert!(place.len() >= mem::size_of::<Self>());

        let at = place.as_mut_ptr();

        debug_assert!(!at.is_null());
        debug_assert!(utils::is_sufficiently_aligned_for(at, PowerOf2::align_of::<Self>()));

        //  Safety:
        //  -   Sufficient size is assumed.
        //  -   Sufficient alignment is assumed.
        #[allow(clippy::cast_ptr_alignment)]
        let result: *mut Self = at as *mut Self;

        //  Safety:
        //  -   `result` is valid for writes.
        ptr::write(result, Self::new(huge_allocator));

        //  Safety:
        //  -   `result` is not null.
        ptr::NonNull::new_unchecked(result)
    }

    //  Internal; Allocates a Normal allocation.
    //
    //  #   Safety
    //
    //  -   Assumes `thread_local` is not concurrently accessed by another thread.
    //  -   Assumes that `layout` is valid, as per `Self::is_valid_layout`.
    #[inline(always)]
    unsafe fn allocate_normal(&self, thread_local: &ThreadLocal<C>, layout: Layout) -> *mut u8 {
        debug_assert!(Self::is_valid_layout(layout));

        //  Safety:
        //  -   `layout.size()` is assumed not to be zero.
        let size = num::NonZeroUsize::new_unchecked(layout.size());

        let class_size = ClassSize::from_size(size);

        //  Safety:
        //  -   `thread_local` is assumed not be accessed concurrently from another thread.
        thread_local.allocate(class_size, |class_size| self.allocate_large_page(class_size))
    }

    //  Internal; Deallocates a Large allocation.
    //
    //  #   Safety
    //
    //  -   Assumes `thread_local` is not concurrently accessed by another thread.
    //  -   Assumes that `ptr` is a Normal allocation allocated by an instance of `Self`.
    #[inline(never)]
    unsafe fn deallocate_normal(&self, thread_local: &ThreadLocal<C>, ptr: *mut u8) {
        debug_assert!((ptr as usize) % C::LARGE_PAGE_SIZE != 0);

        //  Safety:
        //  -   `thread_local` is assumed not be accessed concurrently from another thread.
        thread_local.deallocate(ptr, |page| Self::catch_large_page(page));
    }

    //  Internal; Deallocates a Large allocation.
    //
    //  #   Safety
    //
    //  -   Assumes that `ptr` is a Normal allocation allocated by an instance of `Self`.
    #[inline(never)]
    unsafe fn deallocate_normal_uncached(&self, ptr: *mut u8) {
        debug_assert!((ptr as usize) % C::LARGE_PAGE_SIZE != 0);
        debug_assert!(!ptr.is_null());

        //  Safety:
        //  -   `ptr` is assumed to point to memory that is no longer in use.
        //  -   `ptr` is assumed to point to a sufficiently large memory area.
        //  -   `ptr` is assumed to be correctly aligned.
        let cell = CellForeign::initialize(ptr);

        let foreign_list = CellForeignList::default();
        foreign_list.push(cell);

        //  Safety:
        //  -   `ptr` is assumed to belong to a `LargePage`.
        let page = LargePage::from_raw::<C>(ptr);
        debug_assert!(!page.is_null());

        //  Safety:
        //  -   `page` is not null.
        let large_page = &*page;

        large_page.refill_foreign(&foreign_list, |page| Self::catch_large_page(page));
        debug_assert!(foreign_list.is_empty());
    }

    //  Internal; Allocates a Large allocation.
    //
    //  #   Safety
    //
    //  -   Assumes that `layout` is valid, as per `Self::is_valid_layout`.
    unsafe fn allocate_large(&self, layout: Layout) -> *mut u8 {
        self.huge_pages.allocate_large(layout, self.as_owner(), self.platform())
    }

    //  Internal; Deallocates a Large allocation.
    //
    //  #   Safety
    //
    //  -   Assumes that `ptr` is a Large allocation allocated by an instance of `Self`.
    unsafe fn deallocate_large(&self, ptr: *mut u8) { self.huge_pages.deallocate_large(ptr); }

    // Internal;  Allocates a Huge allocation.
    //
    //  #   Safety
    //
    //  -   Assumes that `layout` is valid, as per `Self::is_valid_layout`.
    unsafe fn allocate_huge(&self, layout: Layout) -> *mut u8 {
        debug_assert!(Self::is_valid_layout(layout));

        self.huge_allocator.allocate_huge(layout)
    }

    //  Internal; Deallocates a Huge allocation.
    //
    //  #   Safety
    //
    //  -   Assumes that `ptr` is a Huge allocation allocated by an instance of `Self` sharing the same manager.
    unsafe fn deallocate_huge(&self, ptr: *mut u8) { self.huge_allocator.deallocate_huge(ptr); }

    //  Internal; Returns the address of `self`.
    fn as_owner(&self) -> *mut () { self as *const Self as *mut Self as *mut () }

    //  Internal; Allocates a LargePage, as defined by C.
    //
    //  #   Safety
    //
    //  -   Assumes that `class_size` is in bounds.
    //  -   Assumes that `C::LARGE_PAGE_SIZE` is large enough for a `LargePage`.
    #[inline(never)]
    unsafe fn allocate_large_page(&self, class_size: ClassSize) -> *mut LargePage {
        debug_assert!(class_size.value() < self.large_pages.len());

        //  Fast Path: locate an existing one!

        //  Safety:
        //  -   `class_size` is in bounds.
        let large_page = self.large_pages.get_unchecked(class_size.value()).pop();

        if !large_page.is_null() {
            return large_page;
        }

        //  Slow Path: allocate a fresh one!
        let size = C::LARGE_PAGE_SIZE.value();

        //  Safety:
        //  -   `size` is not zero.
        //  -   `size` is a power of 2.
        let layout = Layout::from_size_align_unchecked(size, size);
        debug_assert!(Self::is_valid_layout(layout));

        //  Safety:
        //  -   `layout` is valid.
        let large_page = self.allocate_large(layout);

        if large_page.is_null() {
            return ptr::null_mut();
        }

        //  Safety:
        //  -   `large_page` is not null.
        //  -   `size` is assumed to be the size of the memory allocation.
        let place = slice::from_raw_parts_mut(large_page, size);

        //  Safety:
        //  -   `place` is assumed to be sufficiently sized.
        //  -   `place` is assumed to be sufficiently aligned.
        LargePage::initialize::<C>(place, self.as_owner(), class_size)
    }

    //  Internal; Catch a LargePage, and store it locally.
    #[inline(never)]
    unsafe fn catch_large_page(page: *mut LargePage) {
        debug_assert!(!page.is_null());

        //  Safety:
        //  -   `page` is not null.
        let large_page = &*page;

        let class_size = large_page.class_size();
        let owner = large_page.owner();
        debug_assert!(!owner.is_null());

        //  Safety:
        //  -   `owner` is not null.
        //  -   `owner` points to an instance of `Self`.
        let socket = &*(owner as *mut Self);
        debug_assert!(class_size.value() < socket.large_pages.len());

        //  Safety:
        //  -   `class_size` is within bounds.
        socket.large_pages.get_unchecked(class_size.value()).push(page);
    }
}

//
//  Implementation Details
//

//  Manager of Huge Pages.
struct HugePagesManager<C, P>([HugePagePtr; 64], marker::PhantomData<*const C>, marker::PhantomData<*const P>);

impl<C, P> HugePagesManager<C, P> {
    //  Creates a new instance.
    fn new() -> Self { unsafe { mem::zeroed() } }
}

impl<C, P> HugePagesManager<C, P>
    where
        C: Configuration,
        P: Platform,
{
    //  Safety:
    //  -   `C::HUGE_PAGE_SIZE` is not zero.
    //  -   `C::HUGE_PAGE_SIZE` is a power of 2.
    //  -   `C::HUGE_PAGE_SIZE` is a multiple of itself.
    const HUGE_PAGE_LAYOUT: Layout =
        unsafe { Layout::from_size_align_unchecked(C::HUGE_PAGE_SIZE.value(), C::HUGE_PAGE_SIZE.value()) };

    //  Deallocates all HugePagesManager allocated by the socket.
    // 
    //  This may involve deallocating the memory used by the socket itself, after which it can no longer be used.
    // 
    //  #   Safety
    // 
    //  -   Assumes that none of the memory allocated by the socket is still in use, with the possible exception of the
    //      memory used by `self`.
    unsafe fn close(&self, owner: *mut (), platform: &P) {
        let mut self_page: *mut HugePage = ptr::null_mut();

        for huge_page in &self.0[..] {
            let huge_page = huge_page.steal();

            if huge_page.is_null() {
                break;
            }

            debug_assert!(owner == (*huge_page).owner());

            if C::HUGE_PAGE_SIZE.round_down(owner as usize) == (huge_page as usize) {
                self_page = huge_page;
                continue;
            }

            Self::deallocate_huge_page(platform, huge_page);
        }

        if !self_page.is_null() {
            Self::deallocate_huge_page(platform, self_page);
        }
    }

    //  Attempts to ensure that at least `target` `HugePage` are allocated on the socket.
    //
    //  Returns the minimum of the currently allocated number of pages and `target`.
    fn reserve(&self, target: usize, owner: *mut (), platform: &P) -> usize {
        let mut fresh_page: *mut HugePage = ptr::null_mut();

        for (index, huge_page) in self.0.iter().enumerate() {
            //  Filled from the start, reaching this point means that the `target` is achieved.
            if index >= target {
                break;
            }

            if !huge_page.load().is_null() {
                continue;
            }

            //  Allocate fresh-page if none currently ready.
            if fresh_page.is_null() {
                fresh_page = Self::allocate_huge_page(platform, owner);
            }

            //  If allocation fails, there's no reason it would succeed later, just stop.
            if fresh_page.is_null () {
                return index;
            }

            //  If the replacement is successful, null `fresh_page` to indicate it should not be deallocated or reused.
            if huge_page.replace_null(fresh_page) {
                fresh_page = ptr::null_mut();
            }
        }

        if !fresh_page.is_null() {
            //  Safety:
            //  -   `fresh_page` was allocated by this platform.
            //  -   `fresh_page` is not referenced by anything.
            unsafe{ Self::deallocate_huge_page(platform, fresh_page) };
        }

        cmp::min(target, self.0.len())
    }

    //  Allocates a Large allocation.
    //
    //  #   Safety
    //
    //  -   Assumes that `layout` is valid, as per `Self::is_valid_layout`.
    #[inline(never)]
    unsafe fn allocate_large(&self, layout: Layout, owner: *mut (), platform: &P) -> *mut u8 {
        let mut first_null = self.0.len();

        //  Check if any existing page can accomodate the request.
        for (index, huge_page) in self.0.iter().enumerate() {
            let huge_page = huge_page.load();

            //  The array is filled in order, there's none after that.
            if huge_page.is_null() {
                first_null = index;
                break;
            }

            //  Safety:
            //  -   `huge_page` is not null.
            let huge_page = &*huge_page;

            let result = huge_page.allocate(layout);

            if !result.is_null() {
                return result;
            }
        }

        //  This is typically where a lock would be acquired to avoid over-acquiring memory from the system.
        //
        //  The chances of over-acquiring are slim, though, so instead a lock-free algorithm is used, betting on the
        //  fact that no two threads will concurrently attempt to allocate a new HugePage.

        //  None of the non-null pages could accomodate the request, so allocate a fresh one.
        let fresh_page = Self::allocate_huge_page(platform, owner);

        let result = if !fresh_page.is_null() {
            //  Safety:
            //  -   `fresh_page` is not null.
            (&*fresh_page).allocate(layout)
        } else {
            ptr::null_mut()
        };

        for huge_page in &self.0[first_null..] {
            //  Spot claimed!
            if !fresh_page.is_null() && huge_page.replace_null(fresh_page) {
                //  Exclusive access to `fresh_page` at the moment `result` was allocated should guarantee that the
                //  allocation succeeded.
                debug_assert!(!result.is_null());
                return result;
            }

            //  Someone claimed the spot first, or `fresh_page` is null...
            let huge_page = huge_page.load();

            //  Nobody claimed the spot, so `fresh_page` is null, no allocation today!
            if huge_page.is_null() {
                debug_assert!(fresh_page.is_null());
                return ptr::null_mut();
            }

            //  Someone claimed the spot, maybe there's room!

            //  Safety:
            //  -   `huge_page` is not null.
            let huge_page = &*huge_page;

            let result = huge_page.allocate(layout);

            //  There is room! Release the newly acquired `fresh_page`, for now.
            if !result.is_null() {
                platform.deallocate(fresh_page as *mut u8, Self::HUGE_PAGE_LAYOUT);
                return result;
            }
        }

        //  The array of huge pages is full. Unexpected, but not a reason to leak!
        platform.deallocate(fresh_page as *mut u8, Self::HUGE_PAGE_LAYOUT);

        ptr::null_mut()
    }

    //  Deallocates a Large allocation.
    //
    //  #   Safety
    //
    //  -   Assumes that `ptr` is a Large allocation allocated by an instance of `Self`.
    #[inline(never)]
    unsafe fn deallocate_large(&self, ptr: *mut u8) {
        debug_assert!((ptr as usize) % C::LARGE_PAGE_SIZE == 0);
        debug_assert!((ptr as usize) % C::HUGE_PAGE_SIZE != 0);

        //  Safety:
        //  -   `ptr` is assumed to be a large allocation within a HugePage.
        let huge_page = HugePage::from_raw::<C>(ptr);
        debug_assert!(!huge_page.is_null());

        //  Safety:
        //  -   `huge_page` is not null.
        let huge_page = &*huge_page;

        huge_page.deallocate(ptr);
    }

    //  Internal; Allocates a HugePage, as defined by C.
    fn allocate_huge_page(platform: &P, owner: *mut ()) -> *mut HugePage {
        //  Safety:
        //  -   Layout is correctly formed.
        let ptr = unsafe { platform.allocate(Self::HUGE_PAGE_LAYOUT) };

        if ptr.is_null() {
            return ptr::null_mut();
        }

        debug_assert!(utils::is_sufficiently_aligned_for(ptr, C::HUGE_PAGE_SIZE));

        //  Safety:
        //  -   `ptr` is not null.
        //  -   `ptr` is sufficiently aligned.
        //  -   `C::HUGE_PAGE_SIZE` bytes are assumed to have been allocated.
        let slice = unsafe { slice::from_raw_parts_mut(ptr, C::HUGE_PAGE_SIZE.value()) };
    
        //  Safety:
        //  -   The slice is sufficiently large.
        //  -   The slice is sufficiently aligned.
        unsafe { HugePage::initialize::<C>(slice, owner) }
    }

    //  Internal; Deallocates a HugePage, as defined by C.
    //
    //  Accept null arguments.
    //
    //  #   Safety
    //
    //  -   Assumes that `page` was allocated by this platform.
    unsafe fn deallocate_huge_page(platform: &P, page: *mut HugePage) {
        if page.is_null() {
            return;
        }

        platform.deallocate(page as *mut u8, Self::HUGE_PAGE_LAYOUT);
    }
}

impl<C, P> Default for HugePagesManager<C, P> {
    fn default() -> Self { Self::new() }
}

//  Manager of Thread Locals.
struct ThreadLocalsManager<C> {
    //  Owner.
    owner: *mut (),
    //  Stack of available thread-locals.
    stack: CellAtomicForeignStack,
    //  Current watermark for fresh allocations into the buffer area.
    watermark: atomic::AtomicPtr<u8>,
    //  End of buffer area.
    end: *mut u8,
    //  Marker...
    _configuration: marker::PhantomData<*const C>,
}

impl<C> ThreadLocalsManager<C> 
    where
        C: Configuration,
{
    const THREAD_LOCAL_SIZE: usize = mem::size_of::<GuardedThreadLocal<C>>();

    //  Creates an instance which will carve-up the memory of `place` into `ThreadLocals`.
    fn new(owner: *mut (), buffer: &mut [u8]) -> Self {
        let _configuration = marker::PhantomData;

        //  Safety:
        //  -   `buffer.len()` is valid if `buffer` is valid.
        let end = unsafe { buffer.as_mut_ptr().add(buffer.len()) };
        debug_assert!(utils::is_sufficiently_aligned_for(end, PowerOf2::align_of::<GuardedThreadLocal<C>>()));

        let nb_thread_locals = buffer.len() / Self::THREAD_LOCAL_SIZE;
        debug_assert!(nb_thread_locals * Self::THREAD_LOCAL_SIZE <= buffer.len());

        //  Safety:
        //  -   `watermark` still points inside `buffer`, as `x / y * y <= x`.
        let watermark = unsafe { end.sub(nb_thread_locals * Self::THREAD_LOCAL_SIZE) };
        let watermark = atomic::AtomicPtr::new(watermark);

        let stack = CellAtomicForeignStack::default();

        Self { owner, stack, watermark, end, _configuration, }
    }

    //  Acquires a ThreadLocal, if possible.
    fn acquire(&self) -> Option<ptr::NonNull<ThreadLocal<C>>> {
        const RELAXED: atomic::Ordering = atomic::Ordering::Relaxed;

        //  Pick one from stack, if any.
        if let Some(thread_local) = self.pop() {
            return Some(thread_local);
        }

        let mut current = self.watermark.load(RELAXED);
        let end = self.end;

        while current < end {
            //  Safety:
            //  -   `next` still within original `buffer`, as `current < end`.
            let next = unsafe { current.add(Self::THREAD_LOCAL_SIZE) };

            match self.watermark.compare_exchange(current, next, RELAXED, RELAXED) {
                Ok(_) => break,
                Err(previous) => current = previous,
            }
        }

        //  Acquisition failed.
        if current == end {
            return None;
        }

        debug_assert!(utils::is_sufficiently_aligned_for(current, PowerOf2::align_of::<GuardedThreadLocal<C>>()));

        #[allow(clippy::cast_ptr_alignment)]
        let current = current as *mut GuardedThreadLocal<C>;

        //  Safety:
        //  -   `current` is valid for writes.
        //  -   `current` is properly aligned.
        unsafe { ptr::write(current, GuardedThreadLocal::new(self.owner)) };

        //  Safety:
        //  -   `current` is exclusive.
        let guarded = unsafe { &mut *current };

        let thread_local = &mut guarded.thread_local;

        //  Safety:
        //  -   `thread_local` is not null.
        Some(unsafe { ptr::NonNull::new_unchecked(thread_local as *mut _) })
    }

    //  Releases a ThreadLocal, after use.
    fn release(&self, thread_local: ptr::NonNull<ThreadLocal<C>>) { self.push(thread_local); }

    //  Internal; Pops a ThreadLocal off the stack, if any.
    fn pop(&self) -> Option<ptr::NonNull<ThreadLocal<C>>> {
        self.stack.pop().map(|cell| {
            let thread_local = cell.cast();
            //  Safety:
            //  -   `thread_local` is valid for writes.
            //  -   `thread_local` is properly aligned.
            unsafe { ptr::write(thread_local.as_ptr(), ThreadLocal::new(self.owner)) };
            thread_local
        })
    }

    //  Internal; Pushes a ThreadLocal onto the stack.
    fn push(&self, thread_local: ptr::NonNull<ThreadLocal<C>>) {
        let cell = thread_local.cast();
        unsafe { ptr::write(cell.as_ptr(), CellAtomicForeign::default()) };

        self.stack.push(cell);
    }
}

impl<C> Default for ThreadLocalsManager<C> {
    fn default() -> Self {
        Self {
            owner: ptr::null_mut(),
            stack: CellAtomicForeignStack::default(),
            watermark: atomic::AtomicPtr::new(ptr::null_mut()),
            end: ptr::null_mut(),
            _configuration: marker::PhantomData,
        }
    }
}

//  A simple atomic pointer to a `HugePage`.
struct HugePagePtr(atomic::AtomicPtr<HugePage>);

impl HugePagePtr {
    fn load(&self) -> *mut HugePage { self.0.load(atomic::Ordering::Acquire) }

    fn replace_null(&self, ptr: *mut HugePage) -> bool {
        self.0.compare_exchange(ptr::null_mut(), ptr, atomic::Ordering::AcqRel, atomic::Ordering::Relaxed).is_ok()
    }

    fn steal(&self) -> *mut HugePage { self.0.swap(ptr::null_mut(), atomic::Ordering::AcqRel) }

    fn store(&self, ptr: *mut HugePage) { self.0.store(ptr, atomic::Ordering::Release) }
}

impl Default for HugePagePtr {
    fn default() -> Self { Self(atomic::AtomicPtr::new(ptr::null_mut())) }
}

//  A simple padding wrapper to avoid false-sharing between `ThreadLocal`.
#[repr(C)]
struct GuardedThreadLocal<C>{
    _guard: utils::PrefetchGuard,
    thread_local: ThreadLocal<C>,
}

impl<C> GuardedThreadLocal<C>
    where
        C: Configuration
{
    fn new(owner: *mut ()) -> Self {
        GuardedThreadLocal { _guard: Default::default(), thread_local: ThreadLocal::new(owner) }
    }
}

impl<C> Default for GuardedThreadLocal<C>
    where
        C: Configuration
{
    fn default() -> Self { Self::new(ptr::null_mut()) }
}

#[cfg(test)]
mod tests {

use core::cell;

use crate::{PowerOf2, Properties};

use super::*;

type TestThreadLocalsManager = ThreadLocalsManager<TestConfiguration>;

type TestHugePagesManager = HugePagesManager<TestConfiguration, TestPlatform>;

type TestHugeAllocator = HugeAllocator<TestConfiguration, TestPlatform>;

type TestSocketLocal<'a> = SocketLocal<'a, TestConfiguration, TestPlatform>;

struct TestConfiguration;

impl Configuration for TestConfiguration {
    const LARGE_PAGE_SIZE: PowerOf2 = unsafe { PowerOf2::new_unchecked(4 * 1024) };
    const HUGE_PAGE_SIZE: PowerOf2 = unsafe { PowerOf2::new_unchecked(8 * 1024) };
}

const LARGE_PAGE_SIZE: usize = TestConfiguration::LARGE_PAGE_SIZE.value();
const HUGE_PAGE_SIZE: usize = TestConfiguration::HUGE_PAGE_SIZE.value();

const LARGE_PAGE_LAYOUT: Layout = unsafe { Layout::from_size_align_unchecked(LARGE_PAGE_SIZE, LARGE_PAGE_SIZE) };
const HUGE_PAGE_LAYOUT: Layout = unsafe { Layout::from_size_align_unchecked(HUGE_PAGE_SIZE, HUGE_PAGE_SIZE) };

struct TestPlatform([cell::Cell<*mut u8>; 32]);

impl TestPlatform {
    const HUGE_PAGE_SIZE: usize = TestConfiguration::HUGE_PAGE_SIZE.value();

    //  Stores have to be split to avoid stack overflow in Debug mode.
    unsafe fn new(first: &HugePageStore, second: &HugePageStore) -> TestPlatform {
        let stores: [cell::Cell<*mut u8>; 32] = mem::zeroed();

        for i in 0..16 {
            stores[i].set(first.as_ptr().add(i * Self::HUGE_PAGE_SIZE));
        }

        for i in 0..16 {
            stores[16 + i].set(second.as_ptr().add(i * Self::HUGE_PAGE_SIZE));
        }

        TestPlatform(stores)
    }

    //  Creates a TestHugeAllocator.
    unsafe fn allocator(first: &HugePageStore, second: &HugePageStore) -> TestHugeAllocator {
        TestHugeAllocator::new(Self::new(first, second))
    }

    //  Shrink the number of allocations to at most n.
    fn shrink(&self, n: usize) {
        for ptr in &self.0[n..] {
            ptr.set(ptr::null_mut());
        }
    }

    //  Exhausts the HugePagesManager.
    fn exhaust(&self, manager: &TestHugePagesManager) {
        let owner = 0x1234 as *mut ();
        let platform = TestPlatform::default();

        loop {
            let large = unsafe { manager.allocate_large(LARGE_PAGE_LAYOUT, owner, &platform) };

            if large.is_null() { break; }
        }
    }

    //  Returns the number of allocated pages.
    fn allocated(&self) -> usize { self.0.len() - self.available() }

    //  Returns the number of available pages.
    fn available(&self) -> usize { self.0.iter().filter(|p| !p.get().is_null()).count() }
}

impl Platform for TestPlatform {
    unsafe fn allocate(&self, layout: Layout) -> *mut u8 {
        assert_eq!(Self::HUGE_PAGE_SIZE, layout.size());
        assert_eq!(Self::HUGE_PAGE_SIZE, layout.align());

        for ptr in &self.0[..] {
            if ptr.get().is_null() {
                continue;
            }

            return ptr.replace(ptr::null_mut());
        }

        ptr::null_mut()
    }

    unsafe fn deallocate(&self, pointer: *mut u8, layout: Layout) {
        assert_eq!(Self::HUGE_PAGE_SIZE, layout.size());
        assert_eq!(Self::HUGE_PAGE_SIZE, layout.align());

        for ptr in &self.0[..] {
            if !ptr.get().is_null() {
                continue;
            }

            ptr.set(pointer);
            return;
        }
    }
}

impl Default for TestPlatform {
    fn default() -> Self { unsafe { mem::zeroed() } }
}

#[repr(align(128))]
struct ThreadLocalsStore([u8; 8192]);

impl ThreadLocalsStore {
    const THREAD_LOCAL_SIZE: usize = TestThreadLocalsManager::THREAD_LOCAL_SIZE;

    //  Creates a ThreadLocalsManager.
    //
    //  #   Safety
    //
    //  -   Takes over the memory.
    unsafe fn create(&self) -> TestThreadLocalsManager {
        let buffer = slice::from_raw_parts_mut(self.0.as_ptr() as *mut _, self.0.len());

        ThreadLocalsManager::new(ptr::null_mut(), buffer)
    }
}

impl Default for ThreadLocalsStore {
    fn default() -> Self { unsafe { mem::zeroed() } }
}

#[repr(align(8192))]
struct HugePageStore([u8; 131072]); // 128K

impl HugePageStore {
    fn as_ptr(&self) -> *mut u8 { self.0[..].as_ptr() as *mut _ }
}

impl Default for HugePageStore {
    fn default() -> Self { unsafe { mem::zeroed() } }
}

#[test]
fn thread_locals_manager_new() {
    let store = ThreadLocalsStore::default();
    let manager = unsafe { store.create() };

    let watermark = manager.watermark.load(atomic::Ordering::Relaxed);

    assert_eq!(None, manager.stack.pop());
    assert_ne!(ptr::null_mut(), watermark);
    assert_ne!(ptr::null_mut(), manager.end);

    let bytes = manager.end as usize - watermark as usize;

    assert_eq!(7680, bytes);
    assert_eq!(0, bytes % ThreadLocalsStore::THREAD_LOCAL_SIZE);
    assert_eq!(10, bytes / ThreadLocalsStore::THREAD_LOCAL_SIZE);
}

#[test]
fn thread_locals_acquire_release() {
    let store = ThreadLocalsStore::default();
    let manager = unsafe { store.create() };

    //  Acquire fresh pointers, by bumping the watermark.
    let mut thread_locals = [ptr::null_mut(); 10];

    for ptr in &mut thread_locals {
        *ptr = manager.acquire().unwrap().as_ptr();
    }

    //  Watermark bumped all the way through.
    let watermark = manager.watermark.load(atomic::Ordering::Relaxed);
    assert_eq!(manager.end, watermark);

    //  No more!
    assert_eq!(None, manager.acquire());

    //  Release thread-locals.
    for ptr in &thread_locals {
        manager.release(ptr::NonNull::new(*ptr).unwrap());
    }

    //  Acquire them again, in reverse order.
    for ptr in thread_locals.iter().rev() {
        assert_eq!(*ptr, manager.acquire().unwrap().as_ptr());
    }
}

#[test]
fn huge_pages_reserve_full() {
    let owner = 0x1234 as *mut ();

    let (first, second) = (HugePageStore::default(), HugePageStore::default());
    let platform = unsafe { TestPlatform::new(&first, &second) };

    //  Created empty.
    let manager = TestHugePagesManager::new();

    for page in &manager.0[..] {
        assert_eq!(ptr::null_mut(), page.load());
    }

    //  Reserve a few pages.
    let reserved = manager.reserve(3, owner, &platform);

    assert_eq!(3, reserved);
    assert_eq!(3, platform.allocated());

    for page in &manager.0[..3] {
        assert_ne!(ptr::null_mut(), page.load());
    }

    for page in &manager.0[3..] {
        assert_eq!(ptr::null_mut(), page.load());
    }

    //  Reserve a few more pages.
    let reserved = manager.reserve(5, owner, &platform);

    assert_eq!(5, reserved);
    assert_eq!(5, platform.allocated());

    for page in &manager.0[..5] {
        assert_ne!(ptr::null_mut(), page.load());
    }

    for page in &manager.0[5..] {
        assert_eq!(ptr::null_mut(), page.load());
    }

    //  Reserve a few _less_ pages.
    let reserved = manager.reserve(4, owner, &platform);

    assert_eq!(4, reserved);
    assert_eq!(5, platform.allocated());

    for page in &manager.0[..5] {
        assert_ne!(ptr::null_mut(), page.load());
    }

    for page in &manager.0[5..] {
        assert_eq!(ptr::null_mut(), page.load());
    }
}

#[test]
fn huge_pages_reserve_partial() {
    let owner = 0x1234 as *mut ();

    let (first, second) = (HugePageStore::default(), HugePageStore::default());
    let platform = unsafe { TestPlatform::new(&first, &second) };

    //  Created empty.
    let manager = TestHugePagesManager::new();

    for page in &manager.0[..] {
        assert_eq!(ptr::null_mut(), page.load());
    }

    //  Reserve a few pages.
    let reserved = manager.reserve(3, owner, &platform);

    assert_eq!(3, reserved);
    assert_eq!(3, platform.allocated());

    for page in &manager.0[..3] {
        assert_ne!(ptr::null_mut(), page.load());
    }

    for page in &manager.0[3..] {
        assert_eq!(ptr::null_mut(), page.load());
    }

    //  Clear platform.
    platform.shrink(4);
    assert_eq!(31, platform.allocated());

    //  Reserve a few more pages, failing to allocate part-way through.
    let reserved = manager.reserve(5, owner, &platform);

    assert_eq!(4, reserved);
    assert_eq!(32, platform.allocated());

    for page in &manager.0[..4] {
        assert_ne!(ptr::null_mut(), page.load());
    }

    for page in &manager.0[4..] {
        assert_eq!(ptr::null_mut(), page.load());
    }
}

#[test]
fn huge_pages_close() {
    let owner = 0x1234 as *mut ();

    let (first, second) = (HugePageStore::default(), HugePageStore::default());
    let platform = unsafe { TestPlatform::new(&first, &second) };

    let manager = TestHugePagesManager::new();

    //  Reserve a few pages.
    let reserved = manager.reserve(31, owner, &platform);

    assert_eq!(31, reserved);
    assert_eq!(31, platform.allocated());

    //  Release the reserved pages.
    unsafe { manager.close(owner, &platform) };

    assert_eq!(0, platform.allocated());
}

#[test]
fn huge_pages_allocate_initial_fresh() {
    let owner = 0x1234 as *mut ();

    let (first, second) = (HugePageStore::default(), HugePageStore::default());
    let platform = unsafe { TestPlatform::new(&first, &second) };

    let manager = TestHugePagesManager::new();
    let large = unsafe { manager.allocate_large(LARGE_PAGE_LAYOUT, owner, &platform) };

    assert_eq!(1, platform.allocated());

    assert_ne!(ptr::null_mut(), manager.0[0].load());
    assert_ne!(ptr::null_mut(), large);
}

#[test]
fn huge_pages_allocate_initial_out_of_memory() {
    let owner = 0x1234 as *mut ();

    //  Empty!
    let platform = TestPlatform::default();

    assert_eq!(0, platform.available());

    let manager = TestHugePagesManager::new();
    let large = unsafe { manager.allocate_large(LARGE_PAGE_LAYOUT, owner, &platform) };

    assert_eq!(ptr::null_mut(), manager.0[0].load());
    assert_eq!(ptr::null_mut(), large);
}

#[test]
fn huge_pages_allocate_primed_reuse() {
    let owner = 0x1234 as *mut ();

    let (first, second) = (HugePageStore::default(), HugePageStore::default());
    let platform = unsafe { TestPlatform::new(&first, &second) };

    //  Create manager with existing pages.
    let manager = TestHugePagesManager::new();
    let reserved = manager.reserve(3, owner, &platform);

    assert_eq!(3, reserved);
    assert_eq!(3, platform.allocated());

    //  Allocate from existing pages.
    for _ in 0..reserved {
        let large = unsafe { manager.allocate_large(LARGE_PAGE_LAYOUT, owner, &platform) };

        assert_ne!(ptr::null_mut(), large);
    }

    //  Without any more allocations.
    assert_eq!(3, platform.allocated());
}

#[test]
fn huge_pages_allocate_primed_fresh() {
    let owner = 0x1234 as *mut ();

    let (first, second) = (HugePageStore::default(), HugePageStore::default());
    let platform = unsafe { TestPlatform::new(&first, &second) };

    //  Create manager with existing pages.
    let manager = TestHugePagesManager::new();
    let reserved = manager.reserve(3, owner, &platform);

    assert_eq!(3, reserved);
    assert_eq!(3, platform.allocated());

    //  Exhaust manager.
    platform.exhaust(&manager);

    //  Allocate one more Large Page by creating a new HugePage.
    let large = unsafe { manager.allocate_large(LARGE_PAGE_LAYOUT, owner, &platform) };

    assert_ne!(ptr::null_mut(), large);
    assert_eq!(4, platform.allocated());
}

#[test]
fn huge_pages_allocate_primed_out_of_memory() {
    let owner = 0x1234 as *mut ();

    let (first, second) = (HugePageStore::default(), HugePageStore::default());
    let platform = unsafe { TestPlatform::new(&first, &second) };

    //  Create manager with existing pages.
    let manager = TestHugePagesManager::new();
    let reserved = manager.reserve(3, owner, &platform);

    assert_eq!(3, reserved);
    assert_eq!(3, platform.allocated());

    //  Exhaust manager and platform.
    platform.exhaust(&manager);
    platform.shrink(0);

    //  Fail to allocate any further.
    let large = unsafe { manager.allocate_large(LARGE_PAGE_LAYOUT, owner, &platform) };

    assert_eq!(ptr::null_mut(), large);
}

#[test]
fn huge_pages_allocate_full() {
    let owner = 0x1234 as *mut ();

    let (first, second) = (HugePageStore::default(), HugePageStore::default());
    let platform = unsafe { TestPlatform::new(&first, &second) };

    //  Create manager with existing pages.
    let manager = TestHugePagesManager::new();
    let reserved = manager.reserve(1, owner, &platform);

    assert_eq!(1, reserved);
    assert_eq!(1, platform.allocated());

    //  Exhaust manager.
    platform.exhaust(&manager);

    //  Mimic a manager with all its HugePages allocated, by setting the pointer to the same (exhausted) page.
    let initial = manager.0[0].load();
    for page in &manager.0[..] {
        page.store(initial);
    }

    //  Fail to allocate any further.
    let large = unsafe { manager.allocate_large(LARGE_PAGE_LAYOUT, owner, &platform) };

    assert_eq!(ptr::null_mut(), large);

    //  The HugePage was returned to the platform.
    assert_eq!(1, platform.allocated());
}

#[test]
fn huge_pages_deallocate() {
    let owner = 0x1234 as *mut ();

    let (first, second) = (HugePageStore::default(), HugePageStore::default());
    let platform = unsafe { TestPlatform::new(&first, &second) };

    //  Create manager with existing pages.
    let manager = TestHugePagesManager::new();

    //  Allocate a page.
    let large = unsafe { manager.allocate_large(LARGE_PAGE_LAYOUT, owner, &platform) };

    assert_ne!(ptr::null_mut(), large);
    assert_eq!(1, platform.allocated());

    //  Deallocate it.
    unsafe { manager.deallocate_large(large) };

    //  Allocate a page again, it's the same one!
    let other = unsafe { manager.allocate_large(LARGE_PAGE_LAYOUT, owner, &platform) };

    assert_eq!(large, other);
    assert_eq!(1, platform.allocated());
}

#[test]
fn socket_local_size() {
    assert_eq!(1152, mem::size_of::<TestSocketLocal<'static>>());
}

#[test]
fn socket_local_boostrap_success() {
    let (first, second) = (HugePageStore::default(), HugePageStore::default());
    let allocator = unsafe { TestPlatform::allocator(&first, &second) };

    let socket = TestSocketLocal::bootstrap(&allocator);

    assert_eq!(1, allocator.platform().allocated());
    assert_ne!(None, socket);
}

#[test]
fn socket_local_bootstrap_out_of_memory() {
    let allocator = TestHugeAllocator::default();

    let socket = TestSocketLocal::bootstrap(&allocator);

    assert_eq!(None, socket);
}

#[test]
fn socket_local_is_valid_layout() {
    fn is_valid_layout(size: usize, align: usize) -> bool {
        //  Actually, the assumptions do not hold; the test is all about verifying them.
        let layout = unsafe { Layout::from_size_align_unchecked(size, align) };
        TestSocketLocal::<'static>::is_valid_layout(layout)
    }

    //  Cannot handle 0-sized or 0-aligned allocations.
    assert!(!is_valid_layout(0, 0));
    assert!(!is_valid_layout(0, 1));
    assert!(!is_valid_layout(1, 0));

    //  Cannot handle non-power of 2 alignments.
    assert!(!is_valid_layout(3, 3));

    //  Cannot handle alignment above the HUGE_PAGE_SIZE.
    assert!(!is_valid_layout(HUGE_PAGE_SIZE * 2, HUGE_PAGE_SIZE * 2));

    //  Cannot handle a size that is not a multiple of the alignment.
    assert!(!is_valid_layout(3, 4));
    assert!(!is_valid_layout(5, 4));
    assert!(!is_valid_layout(6, 4));
    assert!(!is_valid_layout(7, 4));
    assert!(!is_valid_layout(9, 4));

    //  Layout with non-zero size that is a multiple of a power-of-2 alignment are valid!
    assert!(is_valid_layout(1, 1));
    assert!(is_valid_layout(2, 1));
    assert!(is_valid_layout(3, 1));

    assert!(is_valid_layout(2, 2));
    assert!(is_valid_layout(4, 2));
    assert!(is_valid_layout(6, 2));

    assert!(is_valid_layout(HUGE_PAGE_SIZE, HUGE_PAGE_SIZE));
    assert!(is_valid_layout(HUGE_PAGE_SIZE * 2, HUGE_PAGE_SIZE));
    assert!(is_valid_layout(HUGE_PAGE_SIZE * 3, HUGE_PAGE_SIZE));
}

#[test]
fn socket_local_reserve() {
    let (first, second) = (HugePageStore::default(), HugePageStore::default());
    let allocator = unsafe { TestPlatform::allocator(&first, &second) };
    let socket = TestSocketLocal::bootstrap(&allocator).unwrap();
    let socket = unsafe { socket.as_ref() };

    //  Reserve a few: it works!
    let reserved = socket.reserve(3);

    assert_eq!(3, reserved);
    assert_eq!(3, allocator.platform().allocated());

    //  Attempt to reserve too many: it saturates!
    let available = allocator.platform().available();

    let reserved = socket.reserve(3 + available + 1);

    assert_eq!(3 + available, reserved);
    assert_eq!(0, allocator.platform().available());
}

#[test]
fn socket_local_close() {
    let (first, second) = (HugePageStore::default(), HugePageStore::default());
    let allocator = unsafe { TestPlatform::allocator(&first, &second) };
    let socket = TestSocketLocal::bootstrap(&allocator).unwrap();
    let socket = unsafe { socket.as_ref() };

    //  Reserve a few: it works!
    let reserved = socket.reserve(3);

    assert_eq!(3, reserved);
    assert_eq!(3, allocator.platform().allocated());

    //  Close the socket.
    unsafe { socket.close() };

    assert_eq!(0, allocator.platform().allocated());
}

#[test]
fn socket_local_acquire_release_thread_local() {
    let (first, second) = (HugePageStore::default(), HugePageStore::default());
    let allocator = unsafe { TestPlatform::allocator(&first, &second) };
    let socket = TestSocketLocal::bootstrap(&allocator).unwrap();
    let socket = unsafe { socket.as_ref() };

    //  Acquire a thread-local.
    let thread_local = socket.acquire_thread_local();

    assert_ne!(None, thread_local);

    //  Release the thread-local.
    unsafe { socket.release_thread_local(thread_local.unwrap()) };
}

#[test]
fn socket_local_allocate_deallocate_huge() {
    let (first, second) = (HugePageStore::default(), HugePageStore::default());
    let allocator = unsafe { TestPlatform::allocator(&first, &second) };

    let socket = TestSocketLocal::bootstrap(&allocator).unwrap();
    let socket = unsafe { socket.as_ref() };

    let thread_local = socket.acquire_thread_local().unwrap();
    let thread_local = unsafe { thread_local.as_ref() };

    //  Allocate a huge page.
    let allocation = unsafe { socket.allocate(thread_local, HUGE_PAGE_LAYOUT) };

    assert_ne!(ptr::null_mut(), allocation);
    assert_eq!(2, allocator.platform().allocated());

    //  Deallocate the huge page.
    unsafe { socket.deallocate(thread_local, allocation) };
}

#[test]
fn socket_local_allocate_huge_failure() {
    let (first, second) = (HugePageStore::default(), HugePageStore::default());
    let allocator = unsafe { TestPlatform::allocator(&first, &second) };

    let socket = TestSocketLocal::bootstrap(&allocator).unwrap();
    let socket = unsafe { socket.as_ref() };

    let thread_local = socket.acquire_thread_local().unwrap();
    let thread_local = unsafe { thread_local.as_ref() };

    //  Exhaust platform.
    allocator.platform().shrink(0);
    assert_eq!(0, allocator.platform().available());

    //  Allocate a huge page.
    let allocation = unsafe { socket.allocate(thread_local, HUGE_PAGE_LAYOUT) };
    assert_eq!(ptr::null_mut(), allocation);
}

#[test]
fn socket_local_allocate_deallocate_large() {
    let (first, second) = (HugePageStore::default(), HugePageStore::default());
    let allocator = unsafe { TestPlatform::allocator(&first, &second) };

    let socket = TestSocketLocal::bootstrap(&allocator).unwrap();
    let socket = unsafe { socket.as_ref() };

    let thread_local = socket.acquire_thread_local().unwrap();
    let thread_local = unsafe { thread_local.as_ref() };

    //  Allocate a large page.
    let allocation = unsafe { socket.allocate(thread_local, LARGE_PAGE_LAYOUT) };

    assert_ne!(ptr::null_mut(), allocation);
    assert_eq!(1, allocator.platform().allocated());

    //  Deallocate the large page.
    unsafe { socket.deallocate(thread_local, allocation) };
}

#[test]
fn socket_local_allocate_large_failure() {
    let (first, second) = (HugePageStore::default(), HugePageStore::default());
    let allocator = unsafe { TestPlatform::allocator(&first, &second) };

    let socket = TestSocketLocal::bootstrap(&allocator).unwrap();
    let socket = unsafe { socket.as_ref() };

    let thread_local = socket.acquire_thread_local().unwrap();
    let thread_local = unsafe { thread_local.as_ref() };

    //  Exhaust huge pages and platform.
    allocator.platform().exhaust(&socket.huge_pages);
    allocator.platform().shrink(0);

    assert_eq!(0, allocator.platform().available());

    //  Allocate a large page.
    let allocation = unsafe { socket.allocate(thread_local, LARGE_PAGE_LAYOUT) };

    assert_eq!(ptr::null_mut(), allocation);
}

#[test]
fn socket_local_allocate_deallocate_normal() {
    let (first, second) = (HugePageStore::default(), HugePageStore::default());
    let allocator = unsafe { TestPlatform::allocator(&first, &second) };

    let socket = TestSocketLocal::bootstrap(&allocator).unwrap();
    let socket = unsafe { socket.as_ref() };

    let thread_local = socket.acquire_thread_local().unwrap();
    let thread_local = unsafe { thread_local.as_ref() };

    //  Allocate the smallest possible piece.
    let layout = Layout::from_size_align(1, 1).unwrap();
    let allocation = unsafe { socket.allocate(thread_local, layout) };

    assert_ne!(ptr::null_mut(), allocation);
    assert_eq!(1, allocator.platform().allocated());

    //  Deallocate the piece.
    unsafe { socket.deallocate(thread_local, allocation) };
}

#[test]
fn socket_local_allocate_normal_failure() {
    let (first, second) = (HugePageStore::default(), HugePageStore::default());
    let allocator = unsafe { TestPlatform::allocator(&first, &second) };

    let socket = TestSocketLocal::bootstrap(&allocator).unwrap();
    let socket = unsafe { socket.as_ref() };

    let thread_local = socket.acquire_thread_local().unwrap();
    let thread_local = unsafe { thread_local.as_ref() };

    //  Exhaust manager platform.
    allocator.platform().exhaust(&socket.huge_pages);
    allocator.platform().shrink(0);

    assert_eq!(0, allocator.platform().available());

    //  Allocate the smallest possible piece.
    let layout = Layout::from_size_align(1, 1).unwrap();
    let allocation = unsafe { socket.allocate(thread_local, layout) };

    assert_eq!(ptr::null_mut(), allocation);
}

#[test]
fn socket_local_allocate_deallocate_normal_catch() {
    let (first, second) = (HugePageStore::default(), HugePageStore::default());
    let allocator = unsafe { TestPlatform::allocator(&first, &second) };

    let socket = TestSocketLocal::bootstrap(&allocator).unwrap();
    let socket = unsafe { socket.as_ref() };

    let thread_local = socket.acquire_thread_local().unwrap();
    let thread_local = unsafe { thread_local.as_ref() };

    //  Exhaust platform.
    allocator.platform().shrink(0);

    //  Determine the largest allocation size still normal.
    let size = Properties::<TestConfiguration>::normal_threshold().value();
    let class_size = ClassSize::from_size(num::NonZeroUsize::new(size).unwrap());
    let layout = Layout::from_size_align(size, 1).unwrap();

    //  There are only 2 allocations on a given LargePage, so exhaust it.
    let allocations = [
        unsafe { socket.allocate(thread_local, layout) },
        unsafe { socket.allocate(thread_local, layout) },
    ];

    assert_ne!(ptr::null_mut(), allocations[0]);
    assert_ne!(ptr::null_mut(), allocations[1]);

    //  No further allocation is possible.
    let further = unsafe { socket.allocate(thread_local, layout) };
    assert_eq!(ptr::null_mut(), further);

    //  Deallocate 1 of the two allocations, the LargePage should be caught.
    unsafe { socket.deallocate(thread_local, allocations[0]) };

    assert_ne!(ptr::null_mut(), unsafe { socket.large_pages[class_size.value()].peek() });

    //  Further allocation is now possible!
    let further = unsafe { socket.allocate(thread_local, layout) };
    assert_ne!(ptr::null_mut(), further);
}

}
