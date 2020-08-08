//! ThreadLocal
//!
//! A ThreadLocal instance is a thread-local cache used to speed up Normal allocations.

use core::{marker, mem, ptr};

use crate::{ClassSize, Configuration};
use crate::internals::{cells, large_page::LargePage};

/// ThreadLocal
///
/// Thread-local caching to speed up Normal allocations.
#[repr(C)]
pub(crate) struct ThreadLocal<C> {
    //  Owner (socket).
    owner: *mut (),
    //  Locally cached pages, 1 per class-size.
    local_pages: [LargePagePtr; 63],
    //  Foreign allocations, temporarily stored here to minimize touching another thread's cache lines.
    foreign_allocations: [cells::CellForeignList; 8],
    _configuration: marker::PhantomData<C>,
}

impl<C> ThreadLocal<C>
    where
        C: Configuration
{
    /// Creates an instance.
    pub(crate) fn new(owner: *mut ()) -> Self {
        //  Safety:
        //  -   Pointers can safely be zeroed.
        let local_pages: [LargePagePtr; 63] = unsafe { mem::zeroed() };
        let foreign_allocations = Default::default();
        let _configuration = marker::PhantomData;

        assert!(local_pages.len() >= ClassSize::number_classes(C::LARGE_PAGE_SIZE));

        Self { owner, local_pages, foreign_allocations, _configuration, }
    }

    /// Returns the owner.
    pub(crate) fn owner(&self) -> *mut () { self.owner }

    /// Flushes all the memory retained by the current instance.
    pub(crate) fn flush<F>(&self, mut recycler: F)
        where
            F: FnMut(*mut LargePage)
    {
        //  The order in which the pages and foreign allocations are returned is inconsequential as it is guaranteed
        //  that the foreign allocations do not belong to the local pages.

        for local_page in &self.local_pages[..] {
            let local_page = local_page.replace_with_null();

            if local_page.is_null() {
                continue;
            }

            recycler(local_page);
        }

        for foreign_list in &self.foreign_allocations {
            if foreign_list.is_empty() {
                continue;
            }

            let head = foreign_list.head();
            debug_assert!(!head.is_null());

            //  Safety:
            //  -   `head` is assumed to belong to a `LargePage`.
            let page = unsafe { LargePage::from_raw::<C>(head as *mut u8) };
            debug_assert!(!page.is_null());

            //  Safety:
            //  -   `page` is not null.
            let large_page = unsafe { &*page };

            //  Safety:
            //  -   The linked-cells belong to the page.
            //  -   The access to `foreign_list` is exclusive.
            unsafe { large_page.refill_foreign(foreign_list, &mut recycler) };
        }
    }

    /// Allocates a cell of the specified size, if available.
    ///
    /// If necessary, queries the provided function to require a new LargePage of the appropriate class-size.
    ///
    /// #   Safety
    ///
    /// -   Assumes that `self` is not concurrently accessed by another thread.
    /// -   Assumes that `class_size` is within bounds.
    pub(crate) unsafe fn allocate<F>(&self, class_size: ClassSize, provider: F) -> *mut u8
        where
            F: FnOnce(ClassSize) -> *mut LargePage
    {
        debug_assert!(class_size.value() < self.local_pages.len());

        //  Safety:
        //  -   `class_size` is assumed to be within bounds.
        let page = self.local_pages.get_unchecked(class_size.value());

        //  Fast Path.
        if !page.get().is_null() {
            //  Safety:
            //  -   `page` is not null.
            let large_page = &*page.get();

            //  Safety:
            //  -   It is assumed that this function is never called from multiple threads concurrently.
            let result = large_page.allocate();

            if !result.is_null() {
                return result;
            }

            //  The current large page is empty.
            page.replace_with_null();
        }

        //  Slow Path.
        self.slow_allocate(page, class_size, provider)
    }

    /// Deallocates a cell.
    ///
    /// Calls `recycler` with any `LargePage` that was adrift and was caught.
    ///
    /// #   Safety
    ///
    /// -   Assumes that `self` is not concurrently accessed by another thread.
    /// -   Assumes that the `ptr` points to memory that is no longer in use.
    /// -   Assumes that `ptr` belongs to a `LargePage`.
    pub(crate) unsafe fn deallocate<F>(&self, ptr: *mut u8, recycler: F)
        where
            F: FnMut(*mut LargePage)
    {
        debug_assert!(!ptr.is_null());

        //  Safety:
        //  -   `ptr` is assumed to belong to a `LargePage`.
        let page = LargePage::from_raw::<C>(ptr);
        debug_assert!(!page.is_null());

        //  Safety:
        //  -   `page` is not null.
        let large_page = &*page;

        let class_size = large_page.class_size();

        //  Safety:
        //  -   `class_size` is assumed to be within bounds.
        let local_page = self.local_pages.get_unchecked(class_size.value());

        //  Fast Path.
        if local_page.get() == page {
            //  Safety:
            //  -   A single thread is assumed to be calling `deallocate` at a time.
            //  -   `ptr` is assumed to point to memory that is no longer in use.
            //  -   `ptr` is assumed to belong to `page`.
            large_page.deallocate(ptr)
        } else {
            //  Safety:
            //  -   A single thread is assumed to be calling `deallocate` at a time.
            //  -   `ptr` is assumed to point to memory that is no longer in use.
            //  -   `ptr` is assumed to belong to `page`.
            self.foreign_deallocate(ptr, page, recycler)
        }
    }

    //  Internal; Attempts to allocate a new LargePage, and if successful allocates memory from it.
    // 
    //  #   Safety
    // 
    /// -   Assumes that `self` is not concurrently accessed by another thread.
    //  -   Assumes that `class_size` is within bounds.
    #[inline(never)]
    unsafe fn slow_allocate<F>(&self, page: &LargePagePtr, class_size: ClassSize, provider: F) -> *mut u8
        where
            F: FnOnce(ClassSize) -> *mut LargePage
    {
        page.set(provider(class_size));

        //  The provider could not provide a page, it is now someone else's problem.
        if page.get().is_null() {
            return ptr::null_mut();
        }

        //  Safety:
        //  -   `page` is not null.
        let large_page = &*page.get();

        //  Some of the so-called `foreign_allocations` may actually be local allocations now!
        for foreign_list in &self.foreign_allocations {
            if foreign_list.is_empty() {
                continue;
            }

            //  Safety:
            //  -   It is assumed that this function is never called from multiple threads concurrently.
            large_page.refill_local(foreign_list);
        }

        //  Safety:
        //  -   It is assumed that this function is never called from multiple threads concurrently.
        large_page.allocate()
    }

    //  Internal; Deallocates a cell from the specified foreign page.
    //
    //  Calls `recycler` with any `LargePage` that was adrift and was caught.
    // 
    //  #   Safety
    // 
    /// -   Assumes that `self` is not concurrently accessed by another thread.
    //  -   Assumes that the `ptr` points to memory that is no longer in use.
    //  -   Assumes that `ptr` belongs to `page`.
    unsafe fn foreign_deallocate<F>(&self, ptr: *mut u8, page: *mut LargePage, mut recycler: F)
        where
            F: FnMut(*mut LargePage)
    {
        debug_assert!(!ptr.is_null());
        debug_assert!(!page.is_null());
        debug_assert!(C::LARGE_PAGE_SIZE.round_down(ptr as usize) == C::LARGE_PAGE_SIZE.round_down(page as usize));

        //  Safety:
        //  -   `page` is not null.
        let large_page = &*page;

        //  Safety:
        //  -   `ptr` is assumed to point to memory that is no longer in use.
        //  -   `ptr` is assumed to point to a sufficiently large memory area.
        //  -   `ptr` is assumed to be correctly aligned.
        let cell = cells::CellForeign::initialize(ptr);

        //  Short-circuit implementation.
        //
        //  A flush_threshold of 1 means immediate flush, so there is no need to:
        //  -   Look for a list to the same page, there will be none.
        //  -   Evict a list to another page prematurely, it'll be a waste.
        if large_page.flush_threshold() == 1 {
            let foreign_list = cells::CellForeignList::default();
            Self::push(cell, &foreign_list, large_page, &mut recycler);
            return;
        }

        //  Scan for existing list.
        //
        //  There may be "holes", due to flushes, but it still is desirable to aggregate all cells of a given page
        //  together in a single list.
        for foreign_list in &self.foreign_allocations {
            if foreign_list.is_empty() {
                continue;
            }

            if !foreign_list.is_compatible::<C>(cell) {
                continue;
            }

            Self::push(cell, foreign_list, large_page, &mut recycler);
            return;
        }

        //  Scan for either empty list or largest list.
        let mut selected_index = 0;
        let mut selected_score = 0;

        for (index, foreign_list) in self.foreign_allocations.iter().enumerate() {
            //  Little trick: now "empty" is the best score.
            let score = foreign_list.len().wrapping_sub(1);

            if score > selected_score {
                selected_index = index;
                selected_score = score;
            }
        }

        debug_assert!(selected_index < self.foreign_allocations.len());

        //  Safety:
        //  -   `selected_index` is within bounds, as per `enumerate()`.
        let foreign_list = self.foreign_allocations.get_unchecked(selected_index);

        //  Non-empty list selected, and therefore incompatible; flush it.
        if selected_score != usize::MAX {
            let head = foreign_list.head();
            debug_assert!(!head.is_null());

            //  Safety:
            //  -   `head` is assumed to belong to a `LargePage`.
            let page = LargePage::from_raw::<C>(head as *mut u8);
            debug_assert!(!page.is_null());

            //  Safety:
            //  -   `page` is not null.
            (*page).refill_foreign(foreign_list, &mut recycler);
        }

        Self::push(cell, foreign_list, large_page, &mut recycler);
    }

    //  Internal; Pushes a cell into a foreign-list, possibly refilling the page.
    // 
    //  Returns the page if it was adrift and has been caught, null otherwise.
    // 
    //  #   Safety
    // 
    /// -   Assumes that `self` is not concurrently accessed by another thread.
    //  -   Assumes that `cell` points to memory that is no longer in use.
    //  -   Assumes that `cell` is compatible with `foreign_list`.
    //  -   Assumes that `cell` belongs to `large_page`.
    unsafe fn push<F>(
        cell: ptr::NonNull<cells::CellForeign>,
        foreign_list: &cells::CellForeignList,
        large_page: &LargePage,
        recycler: F,
    )
        where
            F: FnMut(*mut LargePage)
    {
        debug_assert!(foreign_list.is_compatible::<C>(cell));
        debug_assert!(C::LARGE_PAGE_SIZE.round_down(cell.as_ptr() as usize)
            == C::LARGE_PAGE_SIZE.round_down(large_page as *const _ as usize));

        let flush_threshold = large_page.flush_threshold();

        let length = foreign_list.push(cell);

        if length < flush_threshold {
            return;
        }

        large_page.refill_foreign(foreign_list, recycler);
    }
}

impl<C> Default for ThreadLocal<C>
    where
        C: Configuration
{
    fn default() -> Self { Self::new(ptr::null_mut()) }
}


//
//  Implementation Details
//

type LargePagePtr = cells::CellPtr<LargePage>;

#[cfg(test)]
mod tests {

use core::{ops, slice};

use crate::PowerOf2;

use super::*;

struct TestConfiguration;

impl Configuration for TestConfiguration {
    const LARGE_PAGE_SIZE: PowerOf2 = unsafe { PowerOf2::new_unchecked(16 * 1024) };
    const HUGE_PAGE_SIZE: PowerOf2 = unsafe { PowerOf2::new_unchecked(128 * 1024) };
}

type TestThreadLocal = ThreadLocal<TestConfiguration>;

#[derive(Clone, Copy)]
#[repr(align(131072))]
struct HugePageStore([usize; 16384]);

impl HugePageStore {
    /// Creates a Recycler, which will memorize the recycled pages.
    fn recycler<'a>(&'a self, recycled: &'a mut [usize]) -> impl FnMut(*mut LargePage) + 'a {
        let mut i = 0;
        move |large_page| {
            let r = unsafe { self.recycle(large_page) };
            recycled[i] = r;
            i += 1;
        }
    }

    /// Allocates a LargePage at the specified index and with the specified ClassSize.
    ///
    /// #   Safety
    ///
    /// -   `i` is assumed to be unoccupied.
    unsafe fn provide(&self, i: usize, class_size: ClassSize) -> *mut LargePage {
        let owner = self.address();
        let place = self.place(i);
        LargePage::initialize::<TestConfiguration>(place, owner, class_size)
    }

    /// Deallocates a LargePage.
    ///
    /// #   Safety
    ///
    /// -   `page` is assumed to be unused after this call.
    unsafe fn recycle(&self, page: *mut LargePage) -> usize {
        let address = self.address() as usize;

        assert!(address < page as usize, "{} >= {}", address, page as usize);
        assert!((page as usize) < address + Self::huge_page_size(),
            "{} >= {}", page as usize, address + Self::huge_page_size());

        let offset = page as usize - self.address() as usize;
        assert!(offset % Self::large_page_size() == 0, "{} % {} != 0", offset, Self::large_page_size());

        let owner = (*page).owner() as usize;
        assert!(owner == self.address() as usize, "{} != {}", owner, self.address() as usize);

        //  Erase any information, which should trip up any test where the page is used afterwards.
        //  It should also catch any case of the same page being returned multiple times, by modifying the owner field.
        ptr::write_bytes(page as *mut u8, 0xfe, Self::large_page_size());

        offset / Self::large_page_size()
    }

    /// Exhausts a page, using another page as scratch pad.
    unsafe fn exhaust(&self, page: &LargePage, scratch: usize) {
        let class_size = page.class_size();

        let scratch = self.provide(scratch, class_size);

        //  Count how many cells can be allocated by a given page.
        let number_cells = self.cast_adrift(&*scratch);

        //  Exaust `second_page`, stopping short of triggering the null allocation.
        for _ in 0..number_cells {
            assert_ne!(ptr::null_mut(), page.allocate());
        }
    }

    /// Casts a page adrift, by exhausting it. Returns the number of allocations performed.
    unsafe fn cast_adrift(&self, page: &LargePage) -> usize {
        let mut number_cells = 0;
        while !page.allocate().is_null() { number_cells += 1 }
        number_cells
    }

    /// Fills the `CellForeignList` at the specified index, from the specified page, with the specified number of cells.
    ///
    /// Creates a page on the fly for it.
    unsafe fn materialize_foreign_list(
        &self,
        thread_local: &mut TestThreadLocal,
        class_size: ClassSize,
        index: ops::Range<usize>,
        page: ops::Range<usize>,
        n: usize
    )
    {
        assert_eq!(index.end - index.start, page.end - page.start, "Lengths differ between {:?} and {:?}", index, page);

        for (index, page) in index.zip(page) {
            self.provide(page, class_size);
            thread_local.foreign_allocations[index] = self.create_foreign_list(page, n);
        }
    }

    /// Creates a `CellForeignList` containing the specified number of cells from the specified `LargePage` index.
    ///
    /// #   Safety
    ///
    /// -   A `LargePage` is assumed to exist at this index.
    unsafe fn create_foreign_list(&self, i: usize, n: usize) -> cells::CellForeignList {
        let page = &*self.get_large_page(i);

        assert_eq!(page.owner() as usize, self.address() as usize,
            "{} != {}: no LargePage at {}", page.owner() as usize, self.address() as usize, i);

        let list = cells::CellForeignList::default();

        for _ in 0..n {
            let cell_address = page.allocate();
            let cell = ptr::NonNull::new(cell_address as *mut cells::CellForeign).unwrap();
            list.push(cell);
        }

        list
    }

    /// Returns a pointer to the LargePage at the specified index, it may not be initialized.
    fn get_large_page(&self, index: usize) -> *mut LargePage {
        assert!(index != 0);
        assert!(index < Self::number_large_pages(), "{} is too large", index);

        let base = self.address() as *mut u8;
        let place = unsafe { base.add(index * Self::large_page_size()) };

        place as *mut LargePage
    }

    //  Internal; creates a `place` to initialize a `LargePage` in.
    unsafe fn place(&self, index: usize) -> &mut [u8] {
        let place = self.get_large_page(index) as *mut u8;

        slice::from_raw_parts_mut(place, Self::large_page_size())
    }

    fn address(&self) -> *mut () { self as *const Self as *mut Self as *mut () }

    fn huge_page_size() -> usize { TestConfiguration::HUGE_PAGE_SIZE.value() }

    fn large_page_size() -> usize { TestConfiguration::LARGE_PAGE_SIZE.value() }

    fn number_large_pages() -> usize { Self::huge_page_size() / Self::large_page_size() }
}

impl Default for HugePageStore {
    fn default() -> Self {
        assert_eq!(mem::size_of::<Self>(), TestConfiguration::HUGE_PAGE_SIZE.value());

        unsafe { mem::zeroed() }
    }
}

#[cfg(target_pointer_width = "64")]
#[test]
fn size() {
    const CACHE_LINE_SIZE: usize = 64;

    assert_eq!(9 * CACHE_LINE_SIZE, mem::size_of::<ThreadLocal<TestConfiguration>>());
}

#[test]
fn new() {
    TestThreadLocal::default();
}

#[test]
fn flush() {
    const LOCAL_PAGE: usize = 1;
    const FOREIGN_PAGE: usize = 2;
    const CLASS_SIZE: ClassSize = ClassSize::new(3);

    let store = HugePageStore::default();

    let mut thread_local = TestThreadLocal::default();

    thread_local.local_pages[CLASS_SIZE.value()] = LargePagePtr::new(unsafe { store.provide(LOCAL_PAGE, CLASS_SIZE) });

    unsafe { store.provide(FOREIGN_PAGE, ClassSize::new(5)) };
    thread_local.foreign_allocations[2] = unsafe { store.create_foreign_list(FOREIGN_PAGE, 3) };

    let mut recycled = [0; 1];
    thread_local.flush(store.recycler(&mut recycled));

    assert_eq!(1, recycled[0]);

    for (index, local_page) in thread_local.local_pages.iter().enumerate() {
        let local_page = local_page.get();
        assert!(local_page.is_null(), "Page at {} is not null!", index);
    }

    for (index, foreign_list) in thread_local.foreign_allocations.iter().enumerate() {
        assert!(foreign_list.is_empty(), "Foreign list at {} is not empty!", index);
    }
}

#[test]
fn allocate_fast() {
    const LOCAL_PAGE: usize = 1;
    const CLASS_SIZE: ClassSize = ClassSize::new(3);

    let store = HugePageStore::default();
    let local_page = unsafe { store.provide(LOCAL_PAGE, CLASS_SIZE) };

    let mut thread_local = TestThreadLocal::default();

    thread_local.local_pages[CLASS_SIZE.value()] = LargePagePtr::new(local_page);

    let p = unsafe { thread_local.allocate(CLASS_SIZE, |_| panic!("No provider!")) };
    assert_ne!(ptr::null_mut(), p);

    //  In debug, throws if `p` doesn't belong to `local_page`.
    unsafe { (*local_page).deallocate(p) };
}

#[test]
fn allocate_slow_initial() {
    const LOCAL_PAGE: usize = 1;
    const CLASS_SIZE: ClassSize = ClassSize::new(3);

    let store = HugePageStore::default();
    let local_page = unsafe { store.provide(LOCAL_PAGE, CLASS_SIZE) };

    let thread_local = TestThreadLocal::default();

    let p = unsafe {
        thread_local.allocate(CLASS_SIZE, |class_size| {
            assert_eq!(class_size, CLASS_SIZE);
            local_page
        }) 
    };
    assert_ne!(ptr::null_mut(), p);

    //  In debug, throws if `p` doesn't belong to `local_page`.
    unsafe { (*local_page).deallocate(p) };

    for (index, page) in thread_local.local_pages.iter().enumerate() {
        let page = page.get();

        if index == CLASS_SIZE.value() {
            assert!(!page.is_null(), "Page at {} is null!", index);
            assert_eq!(local_page as usize, page as usize);
        } else {
            assert!(page.is_null(), "Page at {} is not null!", index);
        }
    }
}

#[test]
fn allocate_slow_initial_provider_exhausted() {
    const CLASS_SIZE: ClassSize = ClassSize::new(3);

    let thread_local = TestThreadLocal::default();

    //  Attempt to allocate from a null page, triggering a change of page.
    let p = unsafe { thread_local.allocate(CLASS_SIZE, |_| { ptr::null_mut() }) };
    assert_eq!(ptr::null_mut(), p);

    for (index, page) in thread_local.local_pages.iter().enumerate() {
        let page = page.get();

        assert!(page.is_null(), "Page at {} is not null!", index);
    }
}

#[test]
fn allocate_slow_exhausted() {
    const FIRST_PAGE: usize = 1;
    const SECOND_PAGE: usize = 2;
    const THIRD_PAGE: usize = 3;
    const CLASS_SIZE: ClassSize = ClassSize::new(3);

    let store = HugePageStore::default();
    let second_page = unsafe { store.provide(SECOND_PAGE, CLASS_SIZE) };
    let third_page = unsafe { store.provide(THIRD_PAGE, CLASS_SIZE) };

    //  Exaused `second_page`, stopping short of triggering the null allocation.
    unsafe { store.exhaust(&*second_page, FIRST_PAGE) };

    let mut thread_local = TestThreadLocal::default();
    thread_local.local_pages[CLASS_SIZE.value()] = LargePagePtr::new(second_page);

    //  Attempt to allocate from an empty page, triggering a change of page.
    let p = unsafe {
        thread_local.allocate(CLASS_SIZE, |class_size| {
            assert_eq!(class_size, CLASS_SIZE);
            third_page
        })
    };
    assert_ne!(ptr::null_mut(), p);

    //  In debug, throws if `p` doesn't belong to `third_page`.
    unsafe { (*third_page).deallocate(p) };

    for (index, page) in thread_local.local_pages.iter().enumerate() {
        let page = page.get();

        if index == CLASS_SIZE.value() {
            assert!(!page.is_null(), "Page at {} is null!", index);
            assert_eq!(third_page as usize, page as usize);
        } else {
            assert!(page.is_null(), "Page at {} is not null!", index);
        }
    }
}

#[test]
fn allocate_slow_exhausted_provider_exhausted() {
    const FIRST_PAGE: usize = 1;
    const SECOND_PAGE: usize = 2;
    const CLASS_SIZE: ClassSize = ClassSize::new(3);

    let store = HugePageStore::default();
    let second_page = unsafe { store.provide(SECOND_PAGE, CLASS_SIZE) };

    //  Exaused `second_page`, stopping short of triggering the null allocation.
    unsafe { store.exhaust(&*second_page, FIRST_PAGE) };

    let mut thread_local = TestThreadLocal::default();
    thread_local.local_pages[CLASS_SIZE.value()] = LargePagePtr::new(second_page);

    //  Attempt to allocate from an empty page, triggering a change of page.
    let p = unsafe { thread_local.allocate(CLASS_SIZE, |_| { ptr::null_mut() }) };
    assert_eq!(ptr::null_mut(), p);

    for (index, page) in thread_local.local_pages.iter().enumerate() {
        let page = page.get();

        assert!(page.is_null(), "Page at {} is not null!", index);
    }
}

#[test]
fn allocate_slow_exhausted_recover_foreign() {
    const FIRST_PAGE: usize = 1;
    const SECOND_PAGE: usize = 2;
    const THIRD_PAGE: usize = 3;
    const FOREIGN_LIST: usize = 2;
    const CLASS_SIZE: ClassSize = ClassSize::new(3);

    let store = HugePageStore::default();
    let other = HugePageStore::default();
    let second_page = unsafe { store.provide(SECOND_PAGE, CLASS_SIZE) };
    let third_page = unsafe { store.provide(THIRD_PAGE, CLASS_SIZE) };

    //  Exaused `second_page`, stopping short of triggering the null allocation.
    unsafe { store.exhaust(&*second_page, FIRST_PAGE) };

    let mut thread_local = TestThreadLocal::default();
    thread_local.local_pages[CLASS_SIZE.value()] = LargePagePtr::new(second_page);

    thread_local.foreign_allocations[FOREIGN_LIST] = unsafe { store.create_foreign_list(THIRD_PAGE, 3) };
    let head = thread_local.foreign_allocations[FOREIGN_LIST].head() as *mut u8;

    unsafe {
        let bound = FOREIGN_LIST + 1;

        other.materialize_foreign_list(&mut thread_local, CLASS_SIZE, 0..FOREIGN_LIST, 1..bound, 3);
        other.materialize_foreign_list(&mut thread_local, CLASS_SIZE, bound..8, bound..8, 3);
    }

    //  Attempt to allocate from an empty page, triggering a change of page.
    //
    //  The foreign allocations at `FOREIGN_LIST` should be immediately recovered.
    let p = unsafe {
        thread_local.allocate(CLASS_SIZE, |class_size| {
            assert_eq!(class_size, CLASS_SIZE);
            third_page
        })
    };
    assert_eq!(head, p);

    //  In debug, throws if `p` doesn't belong to `third_page`.
    unsafe { (*third_page).deallocate(p) };

    for (index, page) in thread_local.local_pages.iter().enumerate() {
        let page = page.get();

        if index == CLASS_SIZE.value() {
            assert!(!page.is_null(), "Page at {} is null!", index);
            assert_eq!(third_page as usize, page as usize);
        } else {
            assert!(page.is_null(), "Page at {} is not null!", index);
        }
    }

    for (index, foreign_list) in thread_local.foreign_allocations.iter().enumerate() {
        if index == FOREIGN_LIST {
            assert!(foreign_list.is_empty(), "Foreign list at {} is not empty!", index);
        } else {
            assert!(!foreign_list.is_empty(), "Foreign list at {} is empty!", index);
        }
    }
}

#[test]
fn deallocate_local() {
    const LOCAL_PAGE: usize = 1;
    const CLASS_SIZE: ClassSize = ClassSize::new(3);

    let store = HugePageStore::default();
    let local_page = unsafe { store.provide(LOCAL_PAGE, CLASS_SIZE) };

    let mut thread_local = TestThreadLocal::default();
    thread_local.local_pages[CLASS_SIZE.value()] = LargePagePtr::new(local_page);

    let p = unsafe { thread_local.allocate(CLASS_SIZE, |_| panic!("No provider!")) };
    assert_ne!(ptr::null_mut(), p);

    unsafe { thread_local.deallocate(p, |_| panic!("No recycler!")) };

    //  Immediately reusable!
    let q = unsafe { thread_local.allocate(CLASS_SIZE, |_| panic!("No provider!")) };
    assert_eq!(p, q);
}

#[test]
fn deallocate_foreign_no_local() {
    const FOREIGN_PAGE: usize = 1;
    const CLASS_SIZE: ClassSize = ClassSize::new(0);

    let store = HugePageStore::default();
    let foreign_page = unsafe { store.provide(FOREIGN_PAGE, CLASS_SIZE) };

    let flush_threshold = unsafe { (*foreign_page).flush_threshold() };
    assert!(flush_threshold > 5, "{} <= 5", flush_threshold);

    let thread_local = TestThreadLocal::default();

    let mut allocated = [ptr::null_mut(); 5];
    for p in &mut allocated {
        *p = unsafe { (*foreign_page).allocate() };
    }

    for p in &allocated {
        unsafe { thread_local.deallocate(*p, |_| panic!("No recycler!")) };
    }

    assert_eq!(allocated.len(), thread_local.foreign_allocations[0].len());
    assert_eq!(allocated[4], thread_local.foreign_allocations[0].head() as *mut u8);
}

#[test]
fn deallocate_foreign_with_local() {
    const LOCAL_PAGE: usize = 1;
    const FOREIGN_PAGE: usize = 2;
    const CLASS_SIZE: ClassSize = ClassSize::new(0);

    let store = HugePageStore::default();
    let local_page = unsafe { store.provide(LOCAL_PAGE, CLASS_SIZE) };
    let foreign_page = unsafe { store.provide(FOREIGN_PAGE, CLASS_SIZE) };

    let mut thread_local = TestThreadLocal::default();
    thread_local.local_pages[CLASS_SIZE.value()] = LargePagePtr::new(local_page);

    let mut allocated = [ptr::null_mut(); 5];
    for p in &mut allocated {
        *p = unsafe { (*foreign_page).allocate() };
    }

    for p in &allocated {
        unsafe { thread_local.deallocate(*p, |_| panic!("No recycler!")) };
    }

    assert_eq!(allocated.len(), thread_local.foreign_allocations[0].len());
    assert_eq!(allocated[4], thread_local.foreign_allocations[0].head() as *mut u8);
}

#[test]
fn deallocate_foreign_recycle_immediate() {
    const FLUSH_TRESHOLD: usize = 1;
    const FOREIGN_PAGE: usize = 1;
    const CLASS_SIZE: ClassSize = ClassSize::new(8);

    let store = HugePageStore::default();
    let foreign_page = unsafe { store.provide(FOREIGN_PAGE, CLASS_SIZE) };

    assert_eq!(FLUSH_TRESHOLD, unsafe { (*foreign_page).flush_threshold() });

    let mut allocated = [ptr::null_mut(); FLUSH_TRESHOLD];
    for p in &mut allocated {
        *p = unsafe { (*foreign_page).allocate() };
        assert_ne!(ptr::null_mut(), *p);
    }

    //  Cast the page adrift, then refill it just below the catch threshold.
    unsafe {
        let catch_threshold = (*foreign_page).catch_threshold();

        let foreign_list = store.create_foreign_list(FOREIGN_PAGE, catch_threshold - FLUSH_TRESHOLD);

        store.cast_adrift(&*foreign_page);

        (*foreign_page).refill_foreign(&foreign_list, |_| panic!("No recycler!"));
    }

    let thread_local = TestThreadLocal::default();

    //  Provokes flush, which provokes a catch!
    let mut recycled = [0; 1];

    unsafe { thread_local.deallocate(allocated[FLUSH_TRESHOLD - 1], store.recycler(&mut recycled[..])) };

    assert_eq!(FOREIGN_PAGE, recycled[0]);

    for (index, foreign_list) in thread_local.foreign_allocations.iter().enumerate() {
        assert!(foreign_list.is_empty(), "Foreign list at {} is not empty!", index);
    }
}

#[test]
fn deallocate_foreign_recycle_foreign() {
    const FLUSH_TRESHOLD: usize = 7;
    const FOREIGN_PAGE: usize = 1;
    const CLASS_SIZE: ClassSize = ClassSize::new(0);

    let store = HugePageStore::default();
    let foreign_page = unsafe { store.provide(FOREIGN_PAGE, CLASS_SIZE) };

    assert_eq!(FLUSH_TRESHOLD, unsafe { (*foreign_page).flush_threshold() });

    let mut allocated = [ptr::null_mut(); FLUSH_TRESHOLD];
    for p in &mut allocated {
        *p = unsafe { (*foreign_page).allocate() };
        assert_ne!(ptr::null_mut(), *p);
    }

    //  Cast the page adrift, then refill it just below the catch threshold.
    unsafe {
        let catch_threshold = (*foreign_page).catch_threshold();

        let foreign_list = store.create_foreign_list(FOREIGN_PAGE, catch_threshold - FLUSH_TRESHOLD);

        store.cast_adrift(&*foreign_page);

        (*foreign_page).refill_foreign(&foreign_list, |_| panic!("No recycler!"));
    }

    let thread_local = TestThreadLocal::default();

    for p in &allocated[..(FLUSH_TRESHOLD - 1)] {
        unsafe { thread_local.deallocate(*p, |_| panic!("No recycler!")) };
    }

    //  Provokes flush, which provokes a catch!
    let mut recycled = [0; 1];

    unsafe { thread_local.deallocate(allocated[FLUSH_TRESHOLD - 1], store.recycler(&mut recycled[..])) };

    assert_eq!(FOREIGN_PAGE, recycled[0]);

    for (index, foreign_list) in thread_local.foreign_allocations.iter().enumerate() {
        assert!(foreign_list.is_empty(), "Foreign list at {} is not empty!", index);
    }
}

#[test]
fn deallocate_foreign_last_list() {
    const FOREIGN_PAGE: usize = 1;
    const SECOND_PAGE: usize = 2;
    const THIRD_PAGE: usize = 3;
    const FOURTH_PAGE: usize = 4;
    const CLASS_SIZE: ClassSize = ClassSize::new(0);

    let store = HugePageStore::default();
    let foreign_page = unsafe { store.provide(FOREIGN_PAGE, CLASS_SIZE) };
    unsafe { store.provide(SECOND_PAGE, CLASS_SIZE) };
    unsafe { store.provide(THIRD_PAGE, CLASS_SIZE) };
    unsafe { store.provide(FOURTH_PAGE, CLASS_SIZE) };

    let mut thread_local = TestThreadLocal::default();
    thread_local.foreign_allocations[0] = unsafe { store.create_foreign_list(SECOND_PAGE, 3) };
    thread_local.foreign_allocations[1] = unsafe { store.create_foreign_list(THIRD_PAGE, 3) };
    thread_local.foreign_allocations[3] = unsafe { store.create_foreign_list(FOURTH_PAGE, 3) };

    let mut allocated = [ptr::null_mut(); 5];
    for p in &mut allocated {
        *p = unsafe { (*foreign_page).allocate() };
    }

    for p in &allocated {
        unsafe { thread_local.deallocate(*p, |_| panic!("No recycler!")) };
    }

    assert_eq!(allocated.len(), thread_local.foreign_allocations[2].len());
    assert_eq!(allocated[4], thread_local.foreign_allocations[2].head() as *mut u8);
}

#[test]
fn deallocate_foreign_kick_out_longest_list() {
    const FOREIGN_PAGE: usize = 1;
    const KICKED_PAGE: usize = 2;
    const LONGEST_LIST: usize = 2;
    const CLASS_SIZE: ClassSize = ClassSize::new(0);

    let store = HugePageStore::default();
    let other = HugePageStore::default();
    let foreign_page = unsafe { store.provide(FOREIGN_PAGE, CLASS_SIZE) };

    let mut thread_local = TestThreadLocal::default();

    unsafe {
        let bound = LONGEST_LIST + 1;

        store.materialize_foreign_list(&mut thread_local, CLASS_SIZE, LONGEST_LIST..bound, KICKED_PAGE..(KICKED_PAGE + 1), 9);

        other.materialize_foreign_list(&mut thread_local, CLASS_SIZE, 0..LONGEST_LIST, 1..bound, 3);
        other.materialize_foreign_list(&mut thread_local, CLASS_SIZE, bound..8, bound..8, 3);
    }

    let mut allocated = [ptr::null_mut(); 5];
    for p in &mut allocated {
        *p = unsafe { (*foreign_page).allocate() };
    }

    //  Kicks out the longest list (at 2), to store those cells instead.
    for p in &allocated {
        unsafe { thread_local.deallocate(*p, |_| panic!("No recycler!")) };
    }

    assert_eq!(allocated.len(), thread_local.foreign_allocations[LONGEST_LIST].len());
    assert_eq!(allocated[4], thread_local.foreign_allocations[LONGEST_LIST].head() as *mut u8);
}

#[test]
fn deallocate_foreign_recycle_longest_list() {
    const FOREIGN_PAGE: usize = 1;
    const KICKED_PAGE: usize = 2;
    const LONGEST_LIST: usize = 2;
    const CLASS_SIZE: ClassSize = ClassSize::new(0);

    let store = HugePageStore::default();
    let other = HugePageStore::default();
    let foreign_page = unsafe { store.provide(FOREIGN_PAGE, CLASS_SIZE) };

    let catch_threshold = unsafe { (*foreign_page).catch_threshold() };
    assert!(catch_threshold > 3, "{} <= 3", catch_threshold);

    let mut thread_local = TestThreadLocal::default();

    unsafe {
        let bound = LONGEST_LIST + 1;

        store.materialize_foreign_list(&mut thread_local, CLASS_SIZE, LONGEST_LIST..bound, KICKED_PAGE..(KICKED_PAGE + 1), catch_threshold);

        other.materialize_foreign_list(&mut thread_local, CLASS_SIZE, 0..LONGEST_LIST, 1..bound, 3);
        other.materialize_foreign_list(&mut thread_local, CLASS_SIZE, bound..8, bound..8, 3);
    }

    //  Cast kicked_page adrift.
    let kicked_page = store.get_large_page(KICKED_PAGE);
    unsafe { store.cast_adrift(&*kicked_page) };

    let mut allocated = [ptr::null_mut(); 5];
    for p in &mut allocated {
        *p = unsafe { (*foreign_page).allocate() };
    }

    //  Kicks out the longest list (at 2), recycling its page, to store those cells instead.
    let mut recycled = [0; 1];
    for p in &allocated {
        unsafe { thread_local.deallocate(*p, store.recycler(&mut recycled[..])) };
    }

    assert_eq!(KICKED_PAGE, recycled[0]);

    assert_eq!(allocated.len(), thread_local.foreign_allocations[LONGEST_LIST].len());
    assert_eq!(allocated[4], thread_local.foreign_allocations[LONGEST_LIST].head() as *mut u8);
}

}
