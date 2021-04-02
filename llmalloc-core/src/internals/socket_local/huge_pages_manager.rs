//! Manager of Huge pages.

use core::{
    alloc::Layout,
    cmp,
    marker::PhantomData,
    mem,
    ptr::NonNull,
    slice,
};

use crate::{Configuration, Platform};
use crate::{
    internals::{
        atomic::AtomicPtr,
        huge_page::HugePage,
    },
    utils,
};

//  Manager of Huge Pages.
pub(crate) struct HugePagesManager<C, P>([HugePagePtr; 64], PhantomData<*const C>, PhantomData<*const P>);

impl<C, P> HugePagesManager<C, P> {
    //  Creates a new instance.
    pub(crate) fn new(page: Option<NonNull<HugePage>>) -> Self {
        let huge_pages: [HugePagePtr; 64] = unsafe { mem::zeroed() };
        let _configuration = PhantomData;
        let _platform = PhantomData;

        huge_pages[0].store(page);

        HugePagesManager(huge_pages, _configuration, _platform)
    }
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
    pub(crate) unsafe fn close(&self, owner: *mut (), platform: &P) {
        let mut self_page: Option<NonNull<HugePage>> = None;

        for huge_page in &self.0[..] {
            let huge_page = match huge_page.exchange(None) {
                None => break,
                Some(page) => page,
            };

            debug_assert!(owner == huge_page.as_ref().owner());

            if C::HUGE_PAGE_SIZE.round_down(owner as usize) == (huge_page.as_ptr() as usize) {
                self_page = Some(huge_page);
                continue;
            }

            Self::deallocate_huge_page(platform, huge_page);
        }

        if let Some(self_page) = self_page {
            Self::deallocate_huge_page(platform, self_page);
        }
    }

    //  Attempts to ensure that at least `target` `HugePage` are allocated on the socket.
    //
    //  Returns the minimum of the currently allocated number of pages and `target`.
    pub(crate) fn reserve(&self, target: usize, owner: *mut (), platform: &P) -> usize {
        let mut fresh_page: Option<NonNull<HugePage>> = None;

        for (index, huge_page) in self.0.iter().enumerate() {
            //  Filled from the start, reaching this point means that the `target` is achieved.
            if index >= target {
                break;
            }

            if !huge_page.load().is_none() {
                continue;
            }

            //  Allocate fresh-page if none currently ready.
            if fresh_page.is_none() {
                fresh_page = Self::allocate_huge_page(platform, owner);
            }

            //  If allocation fails, there's no reason it would succeed later, just stop.
            if fresh_page.is_none() {
                return index;
            }

            //  If the replacement is successful, null `fresh_page` to indicate it should not be deallocated or reused.
            if let Ok(_) = huge_page.compare_exchange(None, fresh_page) {
                fresh_page = None;
            }
        }

        if let Some(fresh_page) = fresh_page {
            //  Safety:
            //  -   `fresh_page` was allocated by this platform.
            //  -   `fresh_page` is not referenced by anything.
            unsafe { Self::deallocate_huge_page(platform, fresh_page) };
        }

        cmp::min(target, self.0.len())
    }

    //  Allocates a Large allocation.
    //
    //  #   Safety
    //
    //  -   Assumes that `layout` is valid, as per `Self::is_valid_layout`.
    #[inline(never)]
    pub(crate) unsafe fn allocate_large(&self, layout: Layout, owner: *mut (), platform: &P) -> Option<NonNull<u8>> {
        let mut first_null = self.0.len();

        //  Check if any existing page can accomodate the request.
        for (index, huge_page) in self.0.iter().enumerate() {
            let huge_page = match huge_page.load() {
                None => {
                    //  The array is filled in order, there's none after that.
                    first_null = index;
                    break;
                },
                Some(page) => page,
            };

            //  Safety:
            //  -   `huge_page` is not null.
            let huge_page = huge_page.as_ref();

            if let Some(result) = huge_page.allocate(layout) {
                return Some(result);
            }
        }

        //  This is typically where a lock would be acquired to avoid over-acquiring memory from the system.
        //
        //  The chances of over-acquiring are slim, though, so instead a lock-free algorithm is used, betting on the
        //  fact that no two threads will concurrently attempt to allocate a new HugePage.

        //  None of the non-null pages could accomodate the request, so allocate a fresh one.
        let fresh_page = Self::allocate_huge_page(platform, owner);

        let result = fresh_page.and_then(|page| page.as_ref().allocate(layout));

        for huge_page in &self.0[first_null..] {
            //  Spot claimed!
            if fresh_page.is_some() && huge_page.compare_exchange(None, fresh_page).is_ok() {
                //  Exclusive access to `fresh_page` at the moment `result` was allocated should guarantee that the
                //  allocation succeeded.
                debug_assert!(result.is_some());
                return result;
            }

            //  Someone claimed the spot first, or `fresh_page` is null...
            let huge_page = huge_page.load();

            let huge_page = if let Some(huge_page) = huge_page {
                huge_page
            } else {
                //  Nobody claimed the spot, so `fresh_page` is null, no allocation today!
                debug_assert!(fresh_page.is_none());
                return None;
            };

            //  Someone claimed the spot, maybe there's room!

            //  Safety:
            //  -   `huge_page` is not null.
            let huge_page = huge_page.as_ref();

            //  There is room! Release the newly acquired `fresh_page`, for now.
            if let Some(result) = huge_page.allocate(layout) {
                if let Some(fresh_page) = fresh_page {
                    platform.deallocate(fresh_page.cast(), Self::HUGE_PAGE_LAYOUT);
                }
                return Some(result);
            }
        }

        //  The array of huge pages is full. Unexpected, but not a reason to leak!
        if let Some(fresh_page) = fresh_page {
            platform.deallocate(fresh_page.cast(), Self::HUGE_PAGE_LAYOUT);
        }

        None
    }

    //  Deallocates a Large allocation.
    //
    //  #   Safety
    //
    //  -   Assumes that `ptr` is a Large allocation allocated by an instance of `Self`.
    #[inline(never)]
    pub(crate) unsafe fn deallocate_large(&self, ptr: NonNull<u8>) {
        debug_assert!((ptr.as_ptr() as usize) % C::LARGE_PAGE_SIZE == 0);
        debug_assert!((ptr.as_ptr() as usize) % C::HUGE_PAGE_SIZE != 0);

        //  Safety:
        //  -   `ptr` is assumed to be a large allocation within a HugePage.
        let huge_page = HugePage::from_raw::<C>(ptr);

        //  Safety:
        //  -   `huge_page` is not null.
        let huge_page = huge_page.as_ref();

        huge_page.deallocate(ptr);
    }

    //  Allocates a HugePage, as defined by C.
    //
    //  Used by socket_local for its initialization.
    pub(crate) fn allocate_huge_page(platform: &P, owner: *mut ()) -> Option<NonNull<HugePage>> {
        //  Safety:
        //  -   Layout is correctly formed.
        let ptr = unsafe { platform.allocate(Self::HUGE_PAGE_LAYOUT) }?;

        debug_assert!(utils::is_sufficiently_aligned_for(ptr, C::HUGE_PAGE_SIZE));

        //  Safety:
        //  -   `ptr` is not null.
        //  -   `ptr` is sufficiently aligned.
        //  -   `C::HUGE_PAGE_SIZE` bytes are assumed to have been allocated.
        let slice = unsafe { slice::from_raw_parts_mut(ptr.as_ptr(), C::HUGE_PAGE_SIZE.value()) };
    
        //  Safety:
        //  -   The slice is sufficiently large.
        //  -   The slice is sufficiently aligned.
        Some(unsafe { HugePage::initialize::<C>(slice, owner) })
    }

    //  Deallocates a HugePage, as defined by C.
    //
    //  Used by socket_local for its initialization.
    //
    //  #   Safety
    //
    //  -   Assumes that `page` was allocated by this platform.
    pub(crate) unsafe fn deallocate_huge_page(platform: &P, page: NonNull<HugePage>) {
        platform.deallocate(page.cast(), Self::HUGE_PAGE_LAYOUT);
    }
}

impl<C, P> Default for HugePagesManager<C, P> {
    fn default() -> Self { Self::new(None) }
}

//
//  Implementation
//

//  A simple atomic pointer to a `HugePage`.
type HugePagePtr = AtomicPtr<HugePage>;

#[cfg(test)]
mod tests {

use std::{
    alloc,
    cell::RefCell,
    ops::Range,
    ptr,
    sync::atomic::{AtomicPtr, Ordering},
};

use llmalloc_test::BurstyBuilder;

use super::*;
use super::super::test::{HugePageStore, TestConfiguration, TestHugePagesManager, TestPlatform};

const LARGE_PAGE_SIZE: usize = TestConfiguration::LARGE_PAGE_SIZE.value();
const LARGE_PAGE_LAYOUT: Layout = unsafe { Layout::from_size_align_unchecked(LARGE_PAGE_SIZE, LARGE_PAGE_SIZE) };

const HUGE_PAGE_SIZE: usize = TestConfiguration::HUGE_PAGE_SIZE.value();
const HUGE_PAGE_LAYOUT: Layout = unsafe { Layout::from_size_align_unchecked(HUGE_PAGE_SIZE, HUGE_PAGE_SIZE) };

#[test]
fn huge_pages_reserve_full() {
    let owner = 0x1234 as *mut ();

    let store = HugePageStore::default();
    let platform = unsafe { TestPlatform::new(&store) };

    //  Created empty.
    let manager = TestHugePagesManager::default();

    for page in &manager.0[..] {
        assert_eq!(None, page.load());
    }

    //  Reserve a few pages.
    let reserved = manager.reserve(3, owner, &platform);

    assert_eq!(3, reserved);
    assert_eq!(3, platform.allocated());

    for page in &manager.0[..3] {
        assert_ne!(None, page.load());
    }

    for page in &manager.0[3..] {
        assert_eq!(None, page.load());
    }

    //  Reserve a few more pages.
    let reserved = manager.reserve(5, owner, &platform);

    assert_eq!(5, reserved);
    assert_eq!(5, platform.allocated());

    for page in &manager.0[..5] {
        assert_ne!(None, page.load());
    }

    for page in &manager.0[5..] {
        assert_eq!(None, page.load());
    }

    //  Reserve a few _less_ pages.
    let reserved = manager.reserve(4, owner, &platform);

    assert_eq!(4, reserved);
    assert_eq!(5, platform.allocated());

    for page in &manager.0[..5] {
        assert_ne!(None, page.load());
    }

    for page in &manager.0[5..] {
        assert_eq!(None, page.load());
    }
}

#[test]
fn huge_pages_reserve_partial() {
    let owner = 0x1234 as *mut ();

    let store = HugePageStore::default();
    let platform = unsafe { TestPlatform::new(&store) };

    //  Created empty.
    let manager = TestHugePagesManager::default();

    for page in &manager.0[..] {
        assert_eq!(None, page.load());
    }

    //  Reserve a few pages.
    let reserved = manager.reserve(3, owner, &platform);

    assert_eq!(3, reserved);
    assert_eq!(3, platform.allocated());

    for page in &manager.0[..3] {
        assert_ne!(None, page.load());
    }

    for page in &manager.0[3..] {
        assert_eq!(None, page.load());
    }

    //  Clear platform.
    platform.shrink(4);
    assert_eq!(31, platform.allocated());

    //  Reserve a few more pages, failing to allocate part-way through.
    let reserved = manager.reserve(5, owner, &platform);

    assert_eq!(4, reserved);
    assert_eq!(32, platform.allocated());

    for page in &manager.0[..4] {
        assert_ne!(None, page.load());
    }

    for page in &manager.0[4..] {
        assert_eq!(None, page.load());
    }
}

#[test]
fn huge_pages_close() {
    let owner = 0x1234 as *mut ();

    let store = HugePageStore::default();
    let platform = unsafe { TestPlatform::new(&store) };

    let manager = TestHugePagesManager::default();

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

    let store = HugePageStore::default();
    let platform = unsafe { TestPlatform::new(&store) };

    let manager = TestHugePagesManager::default();
    let large = unsafe { manager.allocate_large(LARGE_PAGE_LAYOUT, owner, &platform) };

    assert_eq!(1, platform.allocated());

    assert_ne!(None, manager.0[0].load());
    assert_ne!(None, large);
}

#[test]
fn huge_pages_allocate_initial_out_of_memory() {
    let owner = 0x1234 as *mut ();

    //  Empty!
    let platform = TestPlatform::default();

    assert_eq!(0, platform.available());

    let manager = TestHugePagesManager::default();
    let large = unsafe { manager.allocate_large(LARGE_PAGE_LAYOUT, owner, &platform) };

    assert_eq!(None, manager.0[0].load());
    assert_eq!(None, large);
}

#[test]
fn huge_pages_allocate_primed_reuse() {
    let owner = 0x1234 as *mut ();

    let store = HugePageStore::default();
    let platform = unsafe { TestPlatform::new(&store) };

    //  Create manager with existing pages.
    let manager = TestHugePagesManager::default();
    let reserved = manager.reserve(3, owner, &platform);

    assert_eq!(3, reserved);
    assert_eq!(3, platform.allocated());

    //  Allocate from existing pages.
    for _ in 0..reserved {
        let large = unsafe { manager.allocate_large(LARGE_PAGE_LAYOUT, owner, &platform) };

        assert_ne!(None, large);
    }

    //  Without any more allocations.
    assert_eq!(3, platform.allocated());
}

#[test]
fn huge_pages_allocate_primed_fresh() {
    let owner = 0x1234 as *mut ();

    let store = HugePageStore::default();
    let platform = unsafe { TestPlatform::new(&store) };

    //  Create manager with existing pages.
    let manager = TestHugePagesManager::default();
    let reserved = manager.reserve(3, owner, &platform);

    assert_eq!(3, reserved);
    assert_eq!(3, platform.allocated());

    //  Exhaust manager.
    platform.exhaust(&manager);

    //  Allocate one more Large Page by creating a new HugePage.
    let large = unsafe { manager.allocate_large(LARGE_PAGE_LAYOUT, owner, &platform) };

    assert_ne!(None, large);
    assert_eq!(4, platform.allocated());
}

#[test]
fn huge_pages_allocate_primed_out_of_memory() {
    let owner = 0x1234 as *mut ();

    let store = HugePageStore::default();
    let platform = unsafe { TestPlatform::new(&store) };

    //  Create manager with existing pages.
    let manager = TestHugePagesManager::default();
    let reserved = manager.reserve(3, owner, &platform);

    assert_eq!(3, reserved);
    assert_eq!(3, platform.allocated());

    //  Exhaust manager and platform.
    platform.exhaust(&manager);
    platform.shrink(0);

    //  Fail to allocate any further.
    let large = unsafe { manager.allocate_large(LARGE_PAGE_LAYOUT, owner, &platform) };

    assert_eq!(None, large);
}

#[test]
fn huge_pages_allocate_full() {
    let owner = 0x1234 as *mut ();

    let store = HugePageStore::default();
    let platform = unsafe { TestPlatform::new(&store) };

    //  Create manager with existing pages.
    let manager = TestHugePagesManager::default();
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

    assert_eq!(None, large);

    //  The HugePage was returned to the platform.
    assert_eq!(1, platform.allocated());
}

#[test]
fn huge_pages_deallocate() {
    let owner = 0x1234 as *mut ();

    let store = HugePageStore::default();
    let platform = unsafe { TestPlatform::new(&store) };

    //  Create manager with existing pages.
    let manager = TestHugePagesManager::default();

    //  Allocate a page.
    let large = unsafe { manager.allocate_large(LARGE_PAGE_LAYOUT, owner, &platform) };

    assert_ne!(None, large);
    assert_eq!(1, platform.allocated());

    //  Deallocate it.
    unsafe { manager.deallocate_large(large.unwrap()) };

    //  Allocate a page again, it's the same one!
    let other = unsafe { manager.allocate_large(LARGE_PAGE_LAYOUT, owner, &platform) };

    assert_eq!(large, other);
    assert_eq!(1, platform.allocated());
}

struct PlatformStore(Vec<AtomicPtr<u8>>);

impl PlatformStore {
    fn new(n: usize) -> PlatformStore {
        let mut v = vec!();
        for _ in 0..n {
            let pointer = Self::allocate_huge_page().expect("Successful allocation").cast();
            v.push(AtomicPtr::new(pointer.as_ptr()));
        }
        PlatformStore(v)
    }

    fn empty(n: usize) -> PlatformStore {
        let mut v = vec!();
        for _ in 0..n {
            v.push(AtomicPtr::new(ptr::null_mut()));
        }
        PlatformStore(v)
    }

    fn pop(&self, indexes: Range<usize>) -> LocalPlatform {
        let mut v = vec!();
        for i in indexes {
            let cell = &self.0[i];
            let previous = cell.swap(ptr::null_mut(), Ordering::Relaxed);
            if let Some(previous) = NonNull::new(previous) {
                v.push(previous);
            }
        }
        LocalPlatform(RefCell::new(v))
    }

    fn push(&self, indexes: Range<usize>, platform: LocalPlatform) {
        let mut v = platform.0.into_inner();
        let mut indexes = indexes.rev();

        while let Some(last) = v.pop() {
            let i = indexes.next().unwrap();

            let cell = &self.0[i];
            let previous = cell.swap(last.as_ptr(), Ordering::Relaxed);
            assert_eq!(ptr::null_mut(), previous);
        }
    }

    fn clear(&self) {
        for cell in &self.0 {
            let pointer = cell.swap(ptr::null_mut(), Ordering::Relaxed);
            if !pointer.is_null() {
                unsafe { alloc::dealloc(pointer, HUGE_PAGE_LAYOUT) };
            }
        }
    }

    fn allocate_huge_page() -> Option<NonNull<HugePage>> {
        //  Safety:
        //  -   Layout is correctly formed.
        let ptr = NonNull::new(unsafe { alloc::alloc(HUGE_PAGE_LAYOUT) })?;

        debug_assert!(utils::is_sufficiently_aligned_for(ptr, TestConfiguration::HUGE_PAGE_SIZE));

        //  Safety:
        //  -   `ptr` is not null.
        //  -   `ptr` is sufficiently aligned.
        //  -   `C::HUGE_PAGE_SIZE` bytes are assumed to have been allocated.
        let slice = unsafe { slice::from_raw_parts_mut(ptr.as_ptr(), HUGE_PAGE_SIZE) };

        //  Safety:
        //  -   The slice is sufficiently large.
        //  -   The slice is sufficiently aligned.
        Some(unsafe { HugePage::initialize::<TestConfiguration>(slice, ptr::null_mut()) })
    }
}

impl Drop for PlatformStore {
    fn drop(&mut self) {
        self.clear();
    }
}

#[derive(Default)]
struct LocalPlatform(RefCell<Vec<NonNull<u8>>>);

impl Platform for LocalPlatform {
    unsafe fn allocate(&self, layout: Layout) -> Option<NonNull<u8>> {
        debug_assert_eq!(HUGE_PAGE_LAYOUT, layout);

        self.0.borrow_mut().pop()
    }

    unsafe fn deallocate(&self, pointer: NonNull<u8>, layout: Layout) {
        debug_assert_eq!(HUGE_PAGE_LAYOUT, layout);

        self.0.borrow_mut().push(pointer)
    }
}

struct Global {
    victim: HugePagesManager<TestConfiguration, LocalPlatform>,
    store: PlatformStore,
    large_pages: Vec<AtomicPtr<u8>>,
}

impl Global {
    fn new(n: usize) -> Global {
        let victim = HugePagesManager::new(None);
        let store = PlatformStore::new(n);
        let large_pages = Self::create_large_pages(n);

        Global { victim, store, large_pages, }
    }

    fn empty(n: usize) -> Global {
        let victim = HugePagesManager::new(None);
        let store = PlatformStore::empty(n);
        let large_pages = Self::create_large_pages(n);

        Global { victim, store, large_pages, }
    }

    fn clear(&self, indexes: Range<usize>) {
        self.drain_victim(indexes.clone());
        self.store.clear();

        for cell in &self.large_pages[indexes] {
            cell.store(ptr::null_mut(), Ordering::Relaxed);
        }
    }

    fn collect_allocated(&self) -> Vec<usize> {
        self.store.0.iter()
            .enumerate()
            .filter(|(_, cell)| cell.load(Ordering::Relaxed).is_null())
            .map(|(index, _)| index)
            .collect()
    }

    fn collect_large_pages(&self) -> Vec<usize> {
        self.large_pages.iter()
            .enumerate()
            .filter(|(_, cell)| !cell.load(Ordering::Relaxed).is_null())
            .map(|(index, _)| index)
            .collect()
    }

    fn drain_victim(&self, indexes: Range<usize>) {
        if indexes.start != 0 {
            return;
        }

        let mut slice = &self.store.0[..];

        for huge_page in &self.victim.0 {
            let huge_page = huge_page.exchange(None);

            if huge_page.is_none() {
                continue;
            }

            let mut pointer = huge_page.unwrap().as_ptr() as *mut u8;

            while let Some((head, tail)) = slice.split_first() {
                slice = tail;

                if head.load(Ordering::Relaxed) != ptr::null_mut() {
                    continue;
                }

                head.store(pointer, Ordering::Relaxed);
                pointer = ptr::null_mut();
                break;
            }

            assert_eq!(ptr::null_mut(), pointer);
        }
    }

    #[track_caller]
    fn verify_complete(&self, indexes: Range<usize>) {
        for cell in &self.store.0[indexes] {
            assert_ne!(ptr::null_mut(), cell.load(Ordering::Relaxed));
        }
    }

    fn create_large_pages(n: usize) -> Vec<AtomicPtr<u8>> {
        let mut v = vec!();
        for _ in 0..n {
            v.push(AtomicPtr::new(ptr::null_mut()));
        }
        v
    }
}

unsafe impl Send for Global {}
unsafe impl Sync for Global {}

#[test]
fn huge_pages_reserve_fuzzing() {
    //  This test aims at testing that multiple threads can call reserve in parallel.
    //
    //  Notably, it verifies that:
    //  -   The number of reserved pages is no more than the target.
    //  -   Any superfluously allocated page is returned to the platform, avoiding over-reservation.
    const THREADS: usize = 4;
    const TARGET: usize = 4;

    let mut builder = BurstyBuilder::new(Global::new(THREADS * TARGET), vec!(0usize..4, 4..8, 8..12, 12..16));

    //  Step 1: Verify store is complete.
    builder.add_simple_step(|| |global: &Global, local: &mut Range<usize>| {
        //  Ensure that a single (local) platform can accommodate all reservations.
        assert_eq!(TARGET, local.len(), "{} != {}", TARGET, local.len());

        global.verify_complete(local.clone());
    });

    //  Step 2: Reserve if you can.
    builder.add_complex_step(|| {
        let prep = |global: &Global, local: &mut Range<usize>| -> LocalPlatform {
            global.store.pop(local.clone())
        };
        let step = |global: &Global, local: &mut Range<usize>, platform: LocalPlatform| {
            let reserved = global.victim.reserve(TARGET, ptr::null_mut(), &platform);
            assert_eq!(TARGET, reserved);

            global.store.push(local.clone(), platform)
        };

        (prep, step)
    });

    //  Step 3: Verify only TARGET pages allocated.
    builder.add_simple_step(|| |global: &Global, _: &mut Range<usize>| {
        let missing = global.collect_allocated();
        assert_eq!(TARGET, missing.len(), "{:?}", missing);
    });

    //  Step 4: Return allocations.
    builder.add_simple_step(|| |global: &Global, local: &mut Range<usize>| {
        global.drain_victim(local.clone());
    });

    builder.launch(100);
}

#[test]
fn huge_pages_allocate_large_multiple_huge_additions_fuzzing() {
    //  This test aims at testing that multiple threads can add huge pages in parallel while allocating.
    const THREADS: usize = 4;

    let mut builder = BurstyBuilder::new(Global::new(THREADS), vec!(0usize, 1, 2, 3));

    //  Step 1: Verify store is complete.
    builder.add_simple_step(|| |global: &Global, local: &mut usize| {
        global.verify_complete(*local..(*local + 1));
    });

    //  Step 2: Allocate, it should succeed (and empty the platform)
    builder.add_complex_step(|| {
        let prep = |global: &Global, local: &mut usize| -> LocalPlatform {
            global.store.pop(*local..(*local + 1))
        };
        let step = |global: &Global, _: &mut usize, platform: LocalPlatform| {
            const EMPTY: &'static [NonNull<u8>] = &[];

            let allocated = unsafe { global.victim.allocate_large(LARGE_PAGE_LAYOUT, ptr::null_mut(), &platform) };
            assert_ne!(None, allocated);

            let guard = platform.0.borrow();
            assert_eq!(EMPTY, &guard[..]);
        };

        (prep, step)
    });

    //  Step 3: Return allocations.
    builder.add_simple_step(|| |global: &Global, local: &mut usize| {
        global.drain_victim(*local..(*local + 1));
    });

    builder.launch(100);
}

#[test]
fn huge_pages_allocate_large_race_fuzzing() {
    //  This test aims at testing that a single thread can acquire a large page in case of memory exhaustion.
    const THREADS: usize = 4;

    let mut builder = BurstyBuilder::new(Global::empty(THREADS), vec!(0usize, 1, 2, 3));

    //  Step 1: Prepare single huge page.
    builder.add_simple_step(|| |global: &Global, local: &mut usize| {
        //  Ensure that a single (local) platform can accommodate its reservation.
        if *local != 0 {
            return;
        }

        global.victim.0[0].store(PlatformStore::allocate_huge_page());
    });

    //  Step 2: Allocate, it should succeed for a single thread.
    builder.add_simple_step(|| |global: &Global, local: &mut usize| {
        let platform = LocalPlatform::default();

        let allocated = unsafe { global.victim.allocate_large(LARGE_PAGE_LAYOUT, ptr::null_mut(), &platform) };

        if let Some(allocated) = allocated {
            global.large_pages[*local].store(allocated.as_ptr(), Ordering::Relaxed);
        }
    });

    //  Step 3: Check that a single thread succeeded in allocating.
    builder.add_simple_step(|| |global: &Global, _: &mut usize| {
        let pages = global.collect_large_pages();
        assert_eq!(1, pages.len(), "{:?}", pages);
    });

    //  Step 4: Return allocations.
    builder.add_simple_step(|| |global: &Global, local: &mut usize| {
        global.clear(*local..(*local + 1));
    });

    builder.launch(100);

}

} // mod tests
