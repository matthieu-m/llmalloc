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

mod huge_pages_manager;
mod thread_locals_manager;

#[cfg(test)]
mod test;

use core::{
    alloc::Layout,
    mem,
    num,
    ptr::{self, NonNull},
    slice,
};

use crate::{Category, ClassSize, Configuration, Platform, PowerOf2, Properties};
use crate::{
    internals::{
        atomic_stack::AtomicStack,
        blocks::{BlockForeign, BlockForeignList},
        huge_allocator::HugeAllocator,
        huge_page::HugePage,
        large_page::LargePage,
        thread_local::{ThreadLocal},
    },
    utils,
};

use huge_pages_manager::HugePagesManager;
use thread_locals_manager::ThreadLocalsManager;

/// SocketLocal.
#[repr(align(128))]
#[repr(C)]
pub(crate) struct SocketLocal<'a, C, P> {
    //  Linked-lists of already allocated LargePages.
    //
    //  AtomicStack is not entirely ABA-proof in case of concurrent pop vs pop+re-push. Here, it should work fine as
    //  the delay between pop and re-push is large, considering that it involves:
    //
    //  -   Draining the page, and casting it adrift.
    //  -   Freeing a large number of memory blocks, and catching it.
    //
    //  In a low-latency scenario, with uninterrupted threads, the chances of _that_ happening within the time another
    //  thread takes to just execute pop are exceedingly low, and thus _hopefully_ the merkle-chain of AtomicStack will
    //  be sufficient to guard against those rare cases.
    large_pages: [AtomicStack<LargePage>; 64],
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
    pub(crate) fn bootstrap(huge_allocator: &'a HugeAllocator<C, P>) -> Option<NonNull<Self>> {
        let page = HugePagesManager::<C, P>::allocate_huge_page(huge_allocator.platform(), ptr::null_mut())?;

        //  Safety:
        //  -   `page` is not null.
        //  -   `page` is assumed to be suitably aligned.
        //  -   `page` points to an exclusive memory area.
        let huge_page = unsafe { &mut *page.as_ptr() };

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
        let result = unsafe { Self::initialize(page, socket_place, huge_allocator, thread_locals) };

        huge_page.set_owner(result.as_ptr() as *mut ());

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
    pub(crate) fn acquire_thread_local(&self) -> Option<NonNull<ThreadLocal<C>>> {
        self.thread_locals.acquire()
    }

    /// Releases a `ThreadLocal`.
    ///
    /// #   Safety
    ///
    /// -   Assumes that the `ThreadLocal` comes from `self`.
    pub(crate) unsafe fn release_thread_local(&self, thread_local: NonNull<ThreadLocal<C>>) {
        //  Safety:
        //  -   `thread_local` is not null.
        thread_local.as_ref().flush(|page| Self::catch_large_page(page));

        //  Safety:
        //  -   `thread_local` points to valid memory.
        //  -   `thread_local` is the exclusive point of access to that memory.
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
    pub(crate) unsafe fn allocate(&self, thread_local: &ThreadLocal<C>, layout: Layout) -> Option<NonNull<u8>> {
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
    pub(crate) unsafe fn deallocate(&self, thread_local: &ThreadLocal<C>, ptr: NonNull<u8>) {
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
    pub(crate) unsafe fn deallocate_uncached(&self, ptr: NonNull<u8>) {
        match Properties::<C>::category_of_pointer(ptr) {
            Category::Normal => self.deallocate_normal_uncached(ptr),
            Category::Large => self.deallocate_large(ptr),
            Category::Huge => self.deallocate_huge(ptr),
        }
    }

    //  Internal; Creates a new instance of SocketLocal.
    fn new(
        page: NonNull<HugePage>,
        huge_allocator: &'a HugeAllocator<C, P>,
        thread_locals: ThreadLocalsManager<C>,
    )
        -> Self
    {
        let large_pages = unsafe { mem::zeroed() };
        let huge_pages = HugePagesManager::new(Some(page));

        SocketLocal { large_pages, huge_pages, huge_allocator, thread_locals, }
    }

    //  Internal; Returns a reference to the Platform.
    fn platform(&self) -> &P { self.huge_allocator.platform() }

    //  Internal; In-place constructs an instance of SocketLocal.
    // 
    //  #   Safety
    // 
    //  -   Assumes that `place` is of sufficient size and alignment.
    unsafe fn initialize(
        page: NonNull<HugePage>,
        place: &mut [u8],
        huge_allocator: &'a HugeAllocator<C, P>,
        thread_locals: ThreadLocalsManager<C>,
    )
        -> NonNull<Self>
    {
        debug_assert!(place.len() >= mem::size_of::<Self>());

        //  Safety:
        //  -   `place` is not a null slice.
        let at = NonNull::new_unchecked(place.as_mut_ptr());

        debug_assert!(utils::is_sufficiently_aligned_for(at, PowerOf2::align_of::<Self>()));

        //  Safety:
        //  -   Sufficient size is assumed.
        //  -   Sufficient alignment is assumed.
        let result: NonNull<Self> = at.cast();

        //  Safety:
        //  -   `result` is valid for writes.
        ptr::write(result.as_ptr(), Self::new(page, huge_allocator, thread_locals));

        result
    }

    //  Internal; Allocates a Normal allocation.
    //
    //  #   Safety
    //
    //  -   Assumes `thread_local` is not concurrently accessed by another thread.
    //  -   Assumes that `layout` is valid, as per `Self::is_valid_layout`.
    #[inline(always)]
    unsafe fn allocate_normal(&self, thread_local: &ThreadLocal<C>, layout: Layout) -> Option<NonNull<u8>> {
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
    unsafe fn deallocate_normal(&self, thread_local: &ThreadLocal<C>, ptr: NonNull<u8>) {
        debug_assert!((ptr.as_ptr() as usize) % C::LARGE_PAGE_SIZE != 0);

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
    unsafe fn deallocate_normal_uncached(&self, ptr: NonNull<u8>) {
        debug_assert!((ptr.as_ptr() as usize) % C::LARGE_PAGE_SIZE != 0);

        //  Safety:
        //  -   `ptr` is assumed to point to memory that is no longer in use.
        //  -   `ptr` is assumed to point to a sufficiently large memory area.
        //  -   `ptr` is assumed to be correctly aligned.
        let cell = BlockForeign::initialize(ptr);

        let foreign_list = BlockForeignList::default();
        foreign_list.push(cell);

        //  Safety:
        //  -   `ptr` is assumed to belong to a `LargePage`.
        let page = LargePage::from_raw::<C>(ptr);

        //  Safety:
        //  -   `page` is not null.
        let large_page = page.as_ref();

        large_page.refill_foreign(&foreign_list, |page| Self::catch_large_page(page));
        debug_assert!(foreign_list.is_empty());
    }

    //  Internal; Allocates a Large allocation.
    //
    //  #   Safety
    //
    //  -   Assumes that `layout` is valid, as per `Self::is_valid_layout`.
    unsafe fn allocate_large(&self, layout: Layout) -> Option<NonNull<u8>> {
        self.huge_pages.allocate_large(layout, self.as_owner(), self.platform())
    }

    //  Internal; Deallocates a Large allocation.
    //
    //  #   Safety
    //
    //  -   Assumes that `ptr` is a Large allocation allocated by an instance of `Self`.
    unsafe fn deallocate_large(&self, ptr: NonNull<u8>) { self.huge_pages.deallocate_large(ptr); }

    // Internal;  Allocates a Huge allocation.
    //
    //  #   Safety
    //
    //  -   Assumes that `layout` is valid, as per `Self::is_valid_layout`.
    unsafe fn allocate_huge(&self, layout: Layout) -> Option<NonNull<u8>> {
        debug_assert!(Self::is_valid_layout(layout));

        self.huge_allocator.allocate_huge(layout)
    }

    //  Internal; Deallocates a Huge allocation.
    //
    //  #   Safety
    //
    //  -   Assumes that `ptr` is a Huge allocation allocated by an instance of `Self` sharing the same manager.
    unsafe fn deallocate_huge(&self, ptr: NonNull<u8>) { self.huge_allocator.deallocate_huge(ptr); }

    //  Internal; Returns the address of `self`.
    fn as_owner(&self) -> *mut () { self as *const Self as *mut Self as *mut () }

    //  Internal; Allocates a LargePage, as defined by C.
    //
    //  #   Safety
    //
    //  -   Assumes that `class_size` is in bounds.
    //  -   Assumes that `C::LARGE_PAGE_SIZE` is large enough for a `LargePage`.
    #[inline(never)]
    unsafe fn allocate_large_page(&self, class_size: ClassSize) -> Option<NonNull<LargePage>> {
        debug_assert!(class_size.value() < self.large_pages.len());

        //  Fast Path: locate an existing one!

        //  Safety:
        //  -   `class_size` is in bounds.
        let large_page = self.large_pages.get_unchecked(class_size.value()).pop();

        if large_page.is_some() {
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
        let large_page = self.allocate_large(layout)?;

        //  Safety:
        //  -   `large_page` is not null.
        //  -   `size` is assumed to be the size of the memory allocation.
        let place = slice::from_raw_parts_mut(large_page.as_ptr(), size);

        //  Safety:
        //  -   `place` is assumed to be sufficiently sized.
        //  -   `place` is assumed to be sufficiently aligned.
        Some(LargePage::initialize::<C>(place, self.as_owner(), class_size))
    }

    //  Internal; Catch a LargePage, and store it locally.
    #[inline(never)]
    unsafe fn catch_large_page(page: NonNull<LargePage>) {
        //  Safety:
        //  -   `page` is not null.
        let large_page = page.as_ref();

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
        socket.large_pages.get_unchecked(class_size.value()).push(&mut *page.as_ptr());
    }
}

#[cfg(test)]
mod tests {

use crate::Properties;

use super::*;
use super::test::{HugePageStore, TestConfiguration, TestHugeAllocator, TestPlatform};

type TestSocketLocal<'a> = SocketLocal<'a, TestConfiguration, TestPlatform>;

const LARGE_PAGE_SIZE: usize = TestConfiguration::LARGE_PAGE_SIZE.value();
const HUGE_PAGE_SIZE: usize = TestConfiguration::HUGE_PAGE_SIZE.value();

const LARGE_PAGE_LAYOUT: Layout = unsafe { Layout::from_size_align_unchecked(LARGE_PAGE_SIZE, LARGE_PAGE_SIZE) };
const HUGE_PAGE_LAYOUT: Layout = unsafe { Layout::from_size_align_unchecked(HUGE_PAGE_SIZE, HUGE_PAGE_SIZE) };

#[test]
fn socket_local_size() {
    assert_eq!(1152, mem::size_of::<TestSocketLocal<'static>>());
}

#[test]
fn socket_local_boostrap_success() {
    let store = HugePageStore::default();
    let allocator = unsafe { TestPlatform::allocator(&store) };

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
    let store = HugePageStore::default();
    let allocator = unsafe { TestPlatform::allocator(&store) };
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
    let store = HugePageStore::default();
    let allocator = unsafe { TestPlatform::allocator(&store) };
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
    let store = HugePageStore::default();
    let allocator = unsafe { TestPlatform::allocator(&store) };
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
    let store = HugePageStore::default();
    let allocator = unsafe { TestPlatform::allocator(&store) };

    let socket = TestSocketLocal::bootstrap(&allocator).unwrap();
    let socket = unsafe { socket.as_ref() };

    let thread_local = socket.acquire_thread_local().unwrap();
    let thread_local = unsafe { thread_local.as_ref() };

    //  Allocate a huge page.
    let allocation = unsafe { socket.allocate(thread_local, HUGE_PAGE_LAYOUT) };

    assert_ne!(None, allocation);
    assert_eq!(2, allocator.platform().allocated());

    //  Deallocate the huge page.
    unsafe { socket.deallocate(thread_local, allocation.unwrap()) };
}

#[test]
fn socket_local_allocate_huge_failure() {
    let store = HugePageStore::default();
    let allocator = unsafe { TestPlatform::allocator(&store) };

    let socket = TestSocketLocal::bootstrap(&allocator).unwrap();
    let socket = unsafe { socket.as_ref() };

    let thread_local = socket.acquire_thread_local().unwrap();
    let thread_local = unsafe { thread_local.as_ref() };

    //  Exhaust platform.
    allocator.platform().shrink(0);
    assert_eq!(0, allocator.platform().available());

    //  Allocate a huge page.
    let allocation = unsafe { socket.allocate(thread_local, HUGE_PAGE_LAYOUT) };
    assert_eq!(None, allocation);
}

#[test]
fn socket_local_allocate_deallocate_large() {
    let store = HugePageStore::default();
    let allocator = unsafe { TestPlatform::allocator(&store) };

    let socket = TestSocketLocal::bootstrap(&allocator).unwrap();
    let socket = unsafe { socket.as_ref() };

    let thread_local = socket.acquire_thread_local().unwrap();
    let thread_local = unsafe { thread_local.as_ref() };

    //  Allocate a large page.
    let allocation = unsafe { socket.allocate(thread_local, LARGE_PAGE_LAYOUT) };

    assert_ne!(None, allocation);
    assert_eq!(1, allocator.platform().allocated());

    //  Deallocate the large page.
    unsafe { socket.deallocate(thread_local, allocation.unwrap()) };
}

#[test]
fn socket_local_allocate_large_failure() {
    let store = HugePageStore::default();
    let allocator = unsafe { TestPlatform::allocator(&store) };

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

    assert_eq!(None, allocation);
}

#[test]
fn socket_local_allocate_deallocate_normal() {
    let store = HugePageStore::default();
    let allocator = unsafe { TestPlatform::allocator(&store) };

    let socket = TestSocketLocal::bootstrap(&allocator).unwrap();
    let socket = unsafe { socket.as_ref() };

    let thread_local = socket.acquire_thread_local().unwrap();
    let thread_local = unsafe { thread_local.as_ref() };

    //  Allocate the smallest possible piece.
    let layout = Layout::from_size_align(1, 1).unwrap();
    let allocation = unsafe { socket.allocate(thread_local, layout) };

    assert_ne!(None, allocation);
    assert_eq!(1, allocator.platform().allocated());

    //  Deallocate the piece.
    unsafe { socket.deallocate(thread_local, allocation.unwrap()) };
}

#[test]
fn socket_local_allocate_normal_failure() {
    let store = HugePageStore::default();
    let allocator = unsafe { TestPlatform::allocator(&store) };

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

    assert_eq!(None, allocation);
}

#[test]
fn socket_local_allocate_deallocate_normal_catch() {
    let store = HugePageStore::default();
    let allocator = unsafe { TestPlatform::allocator(&store) };

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

    assert_ne!(None, allocations[0]);
    assert_ne!(None, allocations[1]);

    //  No further allocation is possible.
    let further = unsafe { socket.allocate(thread_local, layout) };
    assert_eq!(None, further);

    //  Deallocate 1 of the two allocations, the LargePage should be caught.
    unsafe { socket.deallocate(thread_local, allocations[0].unwrap()) };

    assert!(!socket.large_pages[class_size.value()].is_empty());

    //  Further allocation is now possible!
    let further = unsafe { socket.allocate(thread_local, layout) };
    assert_ne!(None, further);
}

} // mod tests
