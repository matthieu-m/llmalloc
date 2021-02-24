//! Manager of Huge Allocations.
//!
//! By design, `free` only takes a pointer. At the same time, the `Platform` abstraction requires that the layout of
//! the pointer to be deallocated is passed.
//!
//! The Huge Allocations Manager bridges the gap by recording the original layout on allocation and providing back on
//! deallocation.

use core::{
    alloc::Layout,
    cmp,
    marker::PhantomData,
    ptr::NonNull,
    sync::atomic::{AtomicUsize, Ordering},
};

use crate::{Configuration, Platform, PowerOf2};
use crate::utils;

/// Manager of Huge Allocations (ie, 1 or more HugePages)
pub(crate) struct HugeAllocator<C, P> {
    //  Array of allocation ptr+size.
    //
    //  As an optimization, allocations for which `size == C::HUGE_PAGE_SIZE` are not recorded.
    allocations: [AtomicHugeAllocation<C>; 128],
    platform: P,
    _configuration: PhantomData<*const C>,
}

impl<C, P> HugeAllocator<C, P> {
    /// Creates a new instance.
    pub(crate) const fn new(platform: P) -> Self {
        const fn aha<C>() -> AtomicHugeAllocation<C> { AtomicHugeAllocation::new() }

        //  Replace with mem::zeroed once const on stable.
        let allocations = [
            //  Line 0: up to 16 instances.
            aha(), aha(), aha(), aha(), aha(), aha(), aha(), aha(), aha(), aha(), aha(), aha(), aha(), aha(), aha(), aha(),
            //  Line 1: up to 32 instances.
            aha(), aha(), aha(), aha(), aha(), aha(), aha(), aha(), aha(), aha(), aha(), aha(), aha(), aha(), aha(), aha(),
            //  Line 2: up to 48 instances.
            aha(), aha(), aha(), aha(), aha(), aha(), aha(), aha(), aha(), aha(), aha(), aha(), aha(), aha(), aha(), aha(),
            //  Line 4: up to 64 instances.
            aha(), aha(), aha(), aha(), aha(), aha(), aha(), aha(), aha(), aha(), aha(), aha(), aha(), aha(), aha(), aha(),
            //  Line 5: up to 80 instances.
            aha(), aha(), aha(), aha(), aha(), aha(), aha(), aha(), aha(), aha(), aha(), aha(), aha(), aha(), aha(), aha(),
            //  Line 6: up to 96 instances.
            aha(), aha(), aha(), aha(), aha(), aha(), aha(), aha(), aha(), aha(), aha(), aha(), aha(), aha(), aha(), aha(),
            //  Line 7: up to 112 instances.
            aha(), aha(), aha(), aha(), aha(), aha(), aha(), aha(), aha(), aha(), aha(), aha(), aha(), aha(), aha(), aha(),
            //  Line 8: up to 128 instances.
            aha(), aha(), aha(), aha(), aha(), aha(), aha(), aha(), aha(), aha(), aha(), aha(), aha(), aha(), aha(), aha(),
        ];

        let _configuration = PhantomData;

        Self { allocations, platform, _configuration, }
    }

    /// Returns a reference to the platform.
    pub(crate) fn platform(&self) -> &P { &self.platform }
}

impl<C, P> HugeAllocator<C, P>
    where
        C: Configuration,
        P: Platform,
{
    /// Allocates a Huge allocation.
    #[inline(never)]
    pub(crate) fn allocate_huge(&self, layout: Layout) -> Option<NonNull<u8>> {
        debug_assert!(layout.align().count_ones() == 1, "Invalid layout!");

        let align = cmp::max(C::HUGE_PAGE_SIZE.value(), layout.align());

        //  Safety:
        //  -   `align` is a power of 2.
        let size = unsafe { PowerOf2::new_unchecked(align) }.round_up(layout.size());

        debug_assert!(size >= C::HUGE_PAGE_SIZE.value());

        if size > HugeAllocation::<C>::MAX_SIZE {
            return None;
        }

        //  Safety:
        //  -   `align` is not zero.
        //  -   `align` is a power of 2.
        //  -   `size` is a multiple of `align`.
        let layout = unsafe { Layout::from_size_align_unchecked(size, align) };

        let result = unsafe { self.platform.allocate(layout) };

        //  If null, no need to memorize it, there's nothing to deallocate.
        let result = result?;

        //  Safety:
        //  -   `size` is <= `HugeAllocation::<C>::MAX_SIZE`.
        //  -   `size` is >= `C::HUGE_PAGE_SIZE`.
        if unsafe { self.push_allocation(result, size) } {
            return Some(result);
        }

        //  Oopsie, no place to register it!
        unsafe { self.platform.deallocate(result, layout) };

        None
    }

    /// Deallocates a Huge allocation.
    ///
    /// #   Safety
    ///
    /// -   Assumes that `ptr` was allocated by `self`.
    /// -   Assumes that `ptr` points to the start of the allocation.
    #[inline(never)]
    pub(crate) unsafe fn deallocate_huge(&self, ptr: NonNull<u8>) {
        debug_assert!(utils::is_sufficiently_aligned_for(ptr, C::HUGE_PAGE_SIZE));

        let size = self.pop_allocation(ptr);

        debug_assert!(size % C::HUGE_PAGE_SIZE == 0);
        debug_assert!(size >= C::HUGE_PAGE_SIZE.value());
        debug_assert!(size <= HugeAllocation::<C>::MAX_SIZE);

        let align = C::HUGE_PAGE_SIZE.value();

        //  Safety:
        //  -   `align` is not zero.
        //  -   `align` is a power of 2.
        //  -   `size` is a multiple of `align`.
        let layout = Layout::from_size_align_unchecked(size, align);

        self.platform.deallocate(ptr, layout);
    }

    //  Internal; Pushes a new HugeAllocation into the array.
    //
    //  Returns true on success, false on failure.
    //
    //  #   Safety
    //
    //  -   Assumes that `size` is less than or equal to `MAX_SIZE`.
    //  -   Assumes that `size` is strictly greater than `C::HUGE_PAGE_SIZE`.
    #[must_use]
    unsafe fn push_allocation(&self, ptr: NonNull<u8>, size: usize) -> bool {
        //  Optimize storage of single huge pages.
        if size == C::HUGE_PAGE_SIZE.value() {
            return true;
        }

        //  Safety:
        //  -   `size` is less than or equal to `HugeAllocation::MAX_SIZE`.
        //  -   `size` is greater than or equal to `C::HUGE_PAGE_SIZE`.
        let allocation = HugeAllocation::new(ptr, size);

        let null = HugeAllocation::default();

        for huge in &self.allocations[..] {
            if huge.replace(null, allocation) {
                return true;
            }
        }

        false
    }

    //  Internal; Pops a previous HugeAllocation from the array.
    //
    //  #   Safety
    //
    //  -   Assumes that `ptr` was allocated by `self`.
    unsafe fn pop_allocation(&self, ptr: NonNull<u8>) -> usize {
        for huge in &self.allocations[..] {
            let allocation = huge.load();
            let (p, size) = allocation.inflate();

            if p != Some(ptr) {
                continue;
            }

            let _ = huge.replace(allocation, HugeAllocation::default());
            return size;
        }

        C::HUGE_PAGE_SIZE.value()
    }
}

impl<C, P> Default for HugeAllocator<C, P>
    where
        P: Default,
{
    fn default() -> Self { Self::new(P::default()) }
}

unsafe impl<C, P> Send for HugeAllocator<C, P> {}
unsafe impl<C, P> Sync for HugeAllocator<C, P> {}

//
//  Implementation Details
//

//  A compressed representation of a pointer to a HugePage and the number of HugePages.
struct AtomicHugeAllocation<C>(AtomicUsize, PhantomData<*const C>);

impl<C> AtomicHugeAllocation<C> {
    /// Returns an instance.
    const fn new() -> Self { Self(AtomicUsize::new(0), PhantomData) }
}

impl<C> AtomicHugeAllocation<C>
    where
        C: Configuration,
{
    /// Returns the `HugeAllocation`.
    fn load(&self) -> HugeAllocation<C> {
        let huge = self.0.load(Ordering::Relaxed);
        HugeAllocation(huge, PhantomData)
    }

    /// Sets to `HugeAllocation` if equal to `current`; returns true on success, false on failure.
    #[must_use]
    fn replace(&self, current: HugeAllocation<C>, new: HugeAllocation<C>) -> bool {
        self.0.compare_exchange(current.0, new.0, Ordering::Relaxed, Ordering::Relaxed).is_ok()
    }
}

impl<C> Default for AtomicHugeAllocation<C> {
    fn default() -> Self { Self(AtomicUsize::new(0), PhantomData) }
}

//  A compressed representation of a pointer to a HugePage and the number of HugePages.
struct HugeAllocation<C>(usize, PhantomData<*const C>);

impl<C> HugeAllocation<C>
    where
        C: Configuration,
{
    /// Maximum size which can be encoded in HugeAllocation.
    const MAX_SIZE: usize = (C::HUGE_PAGE_SIZE.value() - 1) * C::HUGE_PAGE_SIZE.value();

    /// Creates a new instance.
    ///
    /// #   Safety
    ///
    /// -   Assumes that `size` is less than or equal to `MAX_SIZE`.
    /// -   Assumes that `size` is zero or greater than or equal to `C::HUGE_PAGE_SIZE`.
    unsafe fn new(ptr: NonNull<u8>, size: usize) -> Self {
        debug_assert!(utils::is_sufficiently_aligned_for(ptr, C::HUGE_PAGE_SIZE),
            "{} not aligned on {}", ptr.as_ptr() as usize, C::HUGE_PAGE_SIZE.value());
        debug_assert!(size % C::HUGE_PAGE_SIZE == 0, "{} not a multiple of {}", size, C::HUGE_PAGE_SIZE.value());
        debug_assert!(size <= Self::MAX_SIZE, "{} greater than {}", size, Self::MAX_SIZE);

        let ptr = ptr.as_ptr() as usize;
        let compressed_size = size / C::HUGE_PAGE_SIZE;

        Self(ptr + compressed_size, PhantomData)
    }

    /// Returns the uncompressed pointer and size.
    fn inflate(&self) -> (Option<NonNull<u8>>, usize) {
        let compressed_ptr = self.0 / C::HUGE_PAGE_SIZE;
        let compressed_size = self.0 % C::HUGE_PAGE_SIZE;

        let ptr = NonNull::new((compressed_ptr * C::HUGE_PAGE_SIZE) as *mut u8);
        let size = compressed_size * C::HUGE_PAGE_SIZE;

        (ptr, size)
    }
}

impl<C> Clone for HugeAllocation<C> {
    fn clone(&self) -> Self { *self }
}

impl<C> Copy for HugeAllocation<C> {}

impl<C> Default for HugeAllocation<C> {
    fn default() -> Self { Self(0, PhantomData) }
}

#[cfg(test)]
mod tests {

use core::{
    cell::UnsafeCell,
    mem,
};

use crate::PowerOf2;

use super::*;

type Allocator = HugeAllocator<TestConfiguration, TestPlatform>;
type Allocation = HugeAllocation<TestConfiguration>;
type AtomicAllocation = AtomicHugeAllocation<TestConfiguration>;

struct TestConfiguration;

impl Configuration for TestConfiguration {
    const LARGE_PAGE_SIZE: PowerOf2 = unsafe { PowerOf2::new_unchecked(1 << 6) };
    const HUGE_PAGE_SIZE: PowerOf2 = unsafe { PowerOf2::new_unchecked(1 << 8) };
}

#[repr(align(1024))]
struct TestPlatform {
    pool: UnsafeCell<[u8; 1024]>,
}

impl TestPlatform {
    fn new() -> Self { unsafe { mem::zeroed() } }

    fn occupied(&self) -> [bool; 4] {
        let starters = self.starters();
        unsafe { [*starters[0] == 1, *starters[1] == 1, *starters[2] == 1, *starters[3] == 1] }
    }

    fn starters(&self) -> [*mut u8; 4] {
        let huge = TestConfiguration::HUGE_PAGE_SIZE.value();

        let ptr = self.pool.get();

        assert_eq!(huge * 4, unsafe { (*ptr).len() });

        let ptr = ptr as *mut u8;
        unsafe { [ptr, ptr.add(huge), ptr.add(huge * 2), ptr.add(huge * 3)] }
    }
}

impl Platform for TestPlatform {
    unsafe fn allocate(&self, layout: Layout) -> Option<NonNull<u8>> {
        let huge = TestConfiguration::HUGE_PAGE_SIZE.value();

        assert!(layout.align() >= huge);
        assert!(layout.size() % huge == 0);
        assert!(layout.size() > 0);

        let starters = self.starters();
        let number_pages = layout.size() / huge;

        for slice in starters.windows(number_pages) {
            if slice[0] as usize % layout.align() != 0 {
                continue;
            }

            if slice.iter().all(|x| **x == 0) {
                slice.iter().for_each(|x| **x = 1);
                return NonNull::new(slice[0]);
            }
        }

        None
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        let huge = TestConfiguration::HUGE_PAGE_SIZE.value();

        assert_eq!(layout.align(), huge);
        assert!(layout.size() % huge == 0);
        assert!(layout.size() > 0);

        let starters = self.starters();
        let number_pages = layout.size() / huge;

        assert!(number_pages <= starters.len());

        for slice in starters.windows(number_pages) {
            if slice[0] == ptr.as_ptr() {
                assert!(slice.iter().all(|x| **x == 1));

                slice.iter().for_each(|x| **x = 0);
                return;
            }
        }

        unreachable!();
    }
}

impl Default for TestPlatform {
    fn default() -> Self { Self::new() }
}

#[test]
fn huge_allocation_new_inflate() {
    fn new_inflate(ptr: usize, size: usize) -> (usize, usize) {
        type C = TestConfiguration;

        let page_size = C::HUGE_PAGE_SIZE;

        let ptr = ptr * page_size;
        let huge = unsafe { Allocation::new(NonNull::new(ptr as *mut u8).unwrap(), size * page_size) };

        let (ptr, size) = huge.inflate();
        let ptr = ptr.map(|ptr| ptr.as_ptr() as usize).unwrap_or(0);

        (ptr / page_size, size / page_size)
    }

    let huge = TestConfiguration::HUGE_PAGE_SIZE.value() - 1;

    assert_eq!((  7,    1), new_inflate(  7,    1));
    assert_eq!(( 42,   23), new_inflate( 42,   23));
    assert_eq!((245, huge), new_inflate(245, huge));
}

#[test]
fn atomic_huge_allocation_load_replace() {
    fn huge_allocation(ptr: usize, size: usize) -> Allocation {
        type C = TestConfiguration;

        if ptr == 0 {
            return Allocation::default();
        }

        let page_size = C::HUGE_PAGE_SIZE;

        let ptr = ptr * page_size;
        unsafe { Allocation::new(NonNull::new(ptr as *mut u8).unwrap(), size * page_size) }
    }

    fn load(atomic: &AtomicAllocation) -> (usize, usize) {
        type C = TestConfiguration;

        let page_size = C::HUGE_PAGE_SIZE;

        let (ptr, size) = atomic.load().inflate();
        let ptr = ptr.map(|ptr| ptr.as_ptr() as usize).unwrap_or(0);

        (ptr / page_size, size / page_size)
    }

    let atomic = AtomicAllocation::default();

    let zero = huge_allocation(0, 0);
    let tracker = huge_allocation(42, 21);
    let (a, b) = (huge_allocation(23, 46), huge_allocation(14, 27));

    assert_eq!((0, 0), load(&atomic));

    assert!(atomic.replace(zero, tracker));
    assert_eq!((42, 21), load(&atomic));

    assert!(!atomic.replace(a, b));
    assert_eq!((42, 21), load(&atomic));

    assert!(atomic.replace(tracker, b));
    assert_eq!((14, 27), load(&atomic));
}

#[test]
fn huge_allocator_allocate_underaligned() {
    fn layout(size: usize) -> Layout { Layout::from_size_align(size, 1).unwrap() }

    let huge = TestConfiguration::HUGE_PAGE_SIZE.value();

    let allocator = Allocator::default();
    let platform = allocator.platform();
    let starters = platform.starters();

    let ptr = allocator.allocate_huge(layout(huge));
    assert!(ptr.is_some());
    assert_eq!(starters[0], ptr.unwrap().as_ptr());
    assert_eq!([true, false, false, false], platform.occupied());

    let ptr = allocator.allocate_huge(layout(huge * 2));
    assert!(ptr.is_some());
    assert_eq!(starters[1], ptr.unwrap().as_ptr());
    assert_eq!([true, true, true, false], platform.occupied());

    let ptr = allocator.allocate_huge(layout(huge * 2));
    assert!(ptr.is_none());
}

#[test]
fn huge_allocator_allocate_overaligned() {
    let huge = TestConfiguration::HUGE_PAGE_SIZE.value();

    let allocator = Allocator::default();
    let platform = allocator.platform();
    let starters = platform.starters();

    let ptr = allocator.allocate_huge(Layout::from_size_align(huge, 4 * huge).unwrap());
    assert!(ptr.is_some());
    assert_eq!(starters[0], ptr.unwrap().as_ptr());
    assert_eq!([true, true, true, true], platform.occupied());
}

#[test]
fn huge_allocator_deallocate() {
    fn layout(size: usize) -> Layout { Layout::from_size_align(size, 1).unwrap() }

    let huge = TestConfiguration::HUGE_PAGE_SIZE.value();

    let allocator = Allocator::default();
    let platform = allocator.platform();

    let one = allocator.allocate_huge(layout(huge * 3));
    let two = allocator.allocate_huge(layout(huge));

    assert_eq!([true, true, true, true], platform.occupied());

    unsafe { allocator.deallocate_huge(two.unwrap()) };
    assert_eq!([true, true, true, false], platform.occupied());

    unsafe { allocator.deallocate_huge(one.unwrap()) };
    assert_eq!([false, false, false, false], platform.occupied());
}

}
