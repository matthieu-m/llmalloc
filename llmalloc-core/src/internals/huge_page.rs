//! Huge Page
//!
//! A Huge Page is a polymorphic allocator of `Configuration::HUGE_PAGE_SIZE` bytes, which fulfills allocations of the
//! Large category.
//!
//! A `SocketLocal` may own multiple 

mod atomic_bit_mask;
mod foreign;
mod number_pages;
mod page_index;
mod page_sizes;
mod page_tokens;

use core::{
    alloc::Layout,
    cmp,
    mem,
    ptr::{self, NonNull},
    slice,
    sync::atomic::{self, Ordering},
};

use crate::{Configuration, PowerOf2};
use crate::utils;

use atomic_bit_mask::AtomicBitMask;
use foreign::Foreign;
use number_pages::NumberPages;
use page_index::PageIndex;

#[repr(C)]
pub(crate) struct HugePage {
    //  Guard against pre-fetching on previous page.
    _prefetch: utils::PrefetchGuard,
    //  Common elements, immutable.
    common: Common,
    //  Foreign elements, mutable and contended.
    foreign: Foreign,
    //  Guard against pre-fetching on start of buffer zone.
    _postfetch: utils::PrefetchGuard,
}

impl HugePage {
    /// In-place constructs a `HugePage`.
    ///
    /// #   Safety
    ///
    /// -   Assumes that there is sufficient memory available.
    /// -   Assumes that the pointer is correctly aligned.
    pub(crate) unsafe fn initialize<C>(place: &mut [u8], owner: *mut ()) -> NonNull<Self>
        where
            C: Configuration,
    {
        debug_assert!(place.len() >= C::HUGE_PAGE_SIZE.value());

        //  Safety:
        //  -   `place` is not a null size.
        let at = NonNull::new_unchecked(place.as_mut_ptr());

        debug_assert!(utils::is_sufficiently_aligned_for(at, C::HUGE_PAGE_SIZE));
        debug_assert!(mem::size_of::<Self>() <= C::LARGE_PAGE_SIZE.value());

        //  Safety:
        //  -   `at` is assumed to be sufficiently sized.
        //  -   `at` is assumed to be sufficiently aligned.
        #[allow(clippy::cast_ptr_alignment)]
        let huge_page = at.as_ptr() as *mut Self;

        ptr::write(huge_page, HugePage::new::<C>(owner));

        //  Enforce memory ordering, later Acquire need to see those 0s and 1s.
        atomic::fence(Ordering::Release);

        at.cast()
    }

    /// Obtain the huge page associated to a given allocation.
    ///
    /// #   Safety
    ///
    /// -   Assumes that the pointer is pointing strictly _inside_ a HugePage.
    pub(crate) unsafe fn from_raw<C>(ptr: NonNull<u8>) -> NonNull<HugePage>
        where
            C: Configuration,
    {
        debug_assert!(!utils::is_sufficiently_aligned_for(ptr, C::HUGE_PAGE_SIZE));

        let address = ptr.as_ptr() as usize;
        let huge_page = C::HUGE_PAGE_SIZE.round_down(address);

        //  Safety:
        //  -   `ptr` was not null
        NonNull::new_unchecked(huge_page as *mut HugePage)
    }

    /// Allocate one or more LargePages from this page, if any.
    ///
    /// Returns a null pointer if the allocation cannot be fulfilled.
    pub(crate) unsafe fn allocate(&self, layout: Layout) -> Option<NonNull<u8>> {
        debug_assert!(layout.align().count_ones() == 1, "{} is not a power of 2", layout.align());

        let large_page_size = self.common.page_size;

        let number_pages = large_page_size.round_up(layout.size()) / large_page_size;
        debug_assert!(number_pages > 0);
        debug_assert!(number_pages <= self.common.number_pages.0);

        //  Safety:
        //  -   `layout.align()` is a power of 2.
        let align_pages = PowerOf2::new_unchecked(cmp::max(layout.align() / large_page_size, 1));
        debug_assert!(align_pages.value() > 0);

        if let Some(index) = self.foreign.allocate(NumberPages(number_pages), align_pages) {
            NonNull::new(self.address().add(index.value() * large_page_size))
        } else {
            None
        }
    }

    /// Deallocates one or multiple pages from this page.
    ///
    /// #   Safety
    ///
    /// -   Assumes that the pointer is pointing to a `LargePage` inside _this_ `HugePage`.
    /// -   Assumes that the pointed page is no longer in use.
    pub(crate) unsafe fn deallocate(&self, ptr: NonNull<u8>) {
        debug_assert!(utils::is_sufficiently_aligned_for(ptr, self.common.page_size));

        let index = (ptr.as_ptr() as usize - self.address() as usize) / self.common.page_size;
        debug_assert!(index > 0 && index <= self.common.number_pages.0);

        //  Safety:
        //  -   `index` is assumed not to be 0.
        //  -   `index` is assumed to point to pages no longer in use.
        self.foreign.deallocate(PageIndex::new_unchecked(index));
    }

    /// Returns the owner of the page.
    pub(crate) fn owner(&self) -> *mut () { self.common.owner }

    /// Sets the owner of the page.
    pub(crate) fn set_owner(&mut self, owner: *mut ()) {
        debug_assert!(self.common.owner.is_null());
        self.common.owner = owner;
    }

    /// Returns a mutable slice to the free-space area between the end of the `HugePage` header and the first
    /// `LargePage`.
    ///
    /// Note that this buffer area may be empty, if the HugePage header just fits within a Large page size.
    pub(crate) fn buffer_mut(&mut self) -> &mut [u8] {
        //  Safety:
        //  -   The pointer is non-null and suitably aligned.
        //  -   The length correctly represents the available memory, and fits within `isize`.
        unsafe { slice::from_raw_parts_mut(self.buffer_ptr(), self.buffer_len()) }
    }

    fn new<C>(owner: *mut ()) -> Self
        where
            C: Configuration,
    {
        let large_page_size = C::LARGE_PAGE_SIZE;
        let huge_page_size = C::HUGE_PAGE_SIZE;

        let number_pages = NumberPages(huge_page_size / large_page_size - 1);

        let _prefetch = utils::PrefetchGuard::default();
        let common = Common::new(owner, large_page_size, number_pages);
        let foreign = Foreign::new(number_pages);
        let _postfetch = utils::PrefetchGuard::default();

        Self { _prefetch, common, foreign, _postfetch, }
    }

    fn address(&self) -> *mut u8 { self as *const _ as *const u8 as *mut u8 }

    fn buffer_ptr(&self) -> *mut u8 {
        //  Safety:
        //  -   The size of Self is small enough that the resulting pointer fits within the same block of memory.
        unsafe { self.address().add(mem::size_of::<Self>()) }
    }

    fn buffer_len(&self) -> usize {
        //  Account for a pre-fetch guard at end of buffer area.
        let reserved = mem::size_of::<Self>() + 128;

        self.common.page_size.value().saturating_sub(reserved)
    }
}

//
//  Implementation Details
//
//  A 128-bytes alignment is used as Intel CPUs prefetch data 2 cache lines (64 bytes) at a time, which to the best of
//  my knowledge is the greatest prefetching among mainstream CPUs.
//

//  Common data. Read-only, accessible from both the local thread and foreign threads without synchronization.
#[repr(align(128))]
struct Common {
    //  A pointer to the owner of the LargePage.
    //
    //  Outside tests, this should point to the SocketLocal from which the page was allocated.
    owner: *mut (),
    //  The size of an individual Large Page.
    page_size: PowerOf2,
    //  The number of Large Pages.
    number_pages: NumberPages,
}

impl Common {
    /// Creates a new instance of `Common`.
    fn new(owner: *mut (), page_size: PowerOf2, number_pages: NumberPages) -> Self  {
        debug_assert!(number_pages.0 >= 1);

        Self { owner, page_size, number_pages, }
    }
}

#[cfg(test)]
mod tests {

use super::*;

#[test]
fn huge_page_smoke_test() {
    const HUGE_PAGE_SIZE: usize = 128 * 1024;
    const LARGE_PAGE_SIZE: usize = 8 * 1024;
    const HUGE_HEADER_SIZE: usize = mem::size_of::<HugePage>();

    struct TestConfiguration;

    impl Configuration for TestConfiguration {
        const LARGE_PAGE_SIZE: PowerOf2 = unsafe { PowerOf2::new_unchecked(LARGE_PAGE_SIZE) };
        const HUGE_PAGE_SIZE: PowerOf2 = unsafe { PowerOf2::new_unchecked(HUGE_PAGE_SIZE) };
    }

    #[repr(align(131072))]
    struct AlignedPage(u8);

    let owner = 1234usize as *mut ();

    //  Use PrefetchGuard to guarantee correct alignment.
    let mut raw: mem::MaybeUninit<AlignedPage> = mem::MaybeUninit::uninit();
    let slice = unsafe { slice::from_raw_parts_mut(raw.as_mut_ptr() as *mut u8, mem::size_of::<AlignedPage>()) };

    let mut huge_page = unsafe { HugePage::initialize::<TestConfiguration>(slice, ptr::null_mut()) };
    let huge_page_ptr = huge_page.as_ptr() as *mut u8;
    assert_eq!(slice.as_mut_ptr(), huge_page_ptr);

    let huge_page = unsafe { huge_page.as_mut() };
    assert_eq!(huge_page_ptr as usize, huge_page.address() as usize);

    assert_eq!(ptr::null_mut(), huge_page.owner());

    huge_page.set_owner(owner);
    assert_eq!(owner, huge_page.owner());

    {
        let buffer = huge_page.buffer_mut();

        let start_ptr = huge_page_ptr as usize;
        let buffer_ptr = buffer.as_mut_ptr() as usize;

        assert_eq!(HUGE_HEADER_SIZE, buffer_ptr - start_ptr);
        assert_eq!(LARGE_PAGE_SIZE - HUGE_HEADER_SIZE - 128, buffer.len());
    }

    let layout = Layout::from_size_align(LARGE_PAGE_SIZE + 1, 1).expect("Proper layout");
    let allocated = unsafe { huge_page.allocate(layout) };
    assert_ne!(None, allocated);

    let retrieved = unsafe { HugePage::from_raw::<TestConfiguration>(allocated.unwrap()) };
    assert_eq!(huge_page_ptr, retrieved.as_ptr() as *mut u8);

    let layout = Layout::from_size_align(LARGE_PAGE_SIZE * 14, 1).expect("Proper layout");
    let failed = unsafe { huge_page.allocate(layout) };
    assert_eq!(None, failed);

    unsafe { huge_page.deallocate(allocated.unwrap()) };
}

} // mod tests
