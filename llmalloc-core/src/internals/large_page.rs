//! Large Page
//!
//! A Large Page is a slab of `Configuration::LARGE_PAGE_SIZE` bytes, which fulfills allocations of the Normal category.
//!
//! Each instance of a Large Page can only fulfill allocations of a specific ClassSize.

use core::{cell, cmp, mem, ptr, sync::atomic};

use crate::{ClassSize, Configuration, PowerOf2};
use crate::internals::cells::{CellLocal, CellLocalPtr, CellForeignList, CellAtomicForeignPtr};
use crate::utils;

/// The header of a Large Page, for normal allocations.
#[repr(C)]
pub(crate) struct LargePage {
    _prefetch: utils::PrefetchGuard,
    common: Common,
    local: Local,
    foreign: Foreign,
}

impl LargePage {
    /// In-place constructs a `LargePage`.
    ///
    /// #   Safety
    ///
    /// -   Assumes that there is sufficient memory available.
    /// -   Assumes that the pointer is correctly aligned.
    pub(crate) unsafe fn initialize<C>(place: &mut [u8], owner: *mut (), class_size: ClassSize) -> *mut Self
        where
            C: Configuration,
    {
        debug_assert!(place.len() >= C::LARGE_PAGE_SIZE.value());

        let at = place.as_mut_ptr();

        debug_assert!(!at.is_null());
        debug_assert!(utils::is_sufficiently_aligned_for(at, C::LARGE_PAGE_SIZE));
        debug_assert!(!utils::is_sufficiently_aligned_for(at, C::HUGE_PAGE_SIZE));

        //  Safety:
        //  -   `at` is assumed to be sufficiently sized.
        //  -   `at` is assumed to be sufficiently aligned.
        #[allow(clippy::cast_ptr_alignment)]
        let large_page = at as *mut Self;

        //  Safety:
        //  -   `large_page` is accessed exclusively from this thread.
        ptr::write(large_page, Self::new::<C>(at, owner, class_size));

        //  Enforce memory ordering, later Acquire need to see those 0s and 1s.
        atomic::fence(atomic::Ordering::Release);

        large_page
    }

    /// Obtain the large page associated to a given allocation.
    ///
    /// #   Safety
    ///
    /// -   Assumes that the pointer is pointing strictly _inside_ a LargePage.
    pub(crate) unsafe fn from_raw<C>(ptr: *mut u8) -> *mut LargePage
        where
            C: Configuration,
    {
        debug_assert!(!utils::is_sufficiently_aligned_for(ptr, C::LARGE_PAGE_SIZE));

        let address = ptr as usize;
        let large_page = C::LARGE_PAGE_SIZE.round_down(address);

        large_page as *mut LargePage
    }

    /// Returns the owner of the page.
    pub(crate) fn owner(&self) -> *mut () { self.common.owner }

    /// Returns the class size of the page.
    pub(crate) fn class_size(&self) -> ClassSize { self.common.class_size }

    /// Returns the flush threshold of the page.
    pub(crate) fn flush_threshold(&self) -> usize { self.common.flush_threshold }

    /// Returns the catch threshold of the page.
    #[cfg(test)]
    pub(crate) fn catch_threshold(&self) -> usize { self.foreign.catch_threshold }

    /// Allocate one cell from the page, if any.
    ///
    /// Returns a null pointer if no cell is available.
    ///
    /// #   Safety
    ///
    /// -   Assumes that a single thread calls `allocate` at a time.
    pub(crate) unsafe fn allocate(&self) -> *mut u8 {
        let ptr = self.local.allocate();

        if !ptr.is_null() {
            ptr
        } else {
            self.foreign.allocate(&self.local)
        }
    }

    /// Deallocate a cell from the local thread.
    ///
    /// #   Safety
    ///
    /// -   Assumes that a single thread calls `deallocate` at a time.
    /// -   Assumes that the pointer is pointing to a cell inside _this_ LargePage.
    /// -   Assumes that the pointed cell is no longer in use.
    pub(crate) unsafe fn deallocate(&self, ptr: *mut u8) {
        debug_assert!(self.common.is_local_cell_pointer(ptr));

        self.local.deallocate(ptr);
    }

    /// Returns the cells to the page "in batch" from the local thread.
    ///
    /// The page will actually check if the foreign list belongs, and leave it untouched if it doesn't.
    ///
    /// #   Safety
    ///
    /// -   Assumes that a single thread calls `deallocate` at a time.
    /// -   Assumes that the access to the linked cells, is exclusive.
    pub(crate) unsafe fn refill_local(&self, list: &CellForeignList) {
        debug_assert!(!list.is_empty());

        if !self.common.is_local_cell_pointer(list.head() as *mut u8) {
            return;
        }

        self.local.extend(list)
    }

    /// Returns the cells to the page "in batch" from a foreign thread.
    ///
    /// Calls `recycler` with the `LargePage` if it was adrift and was caught.
    ///
    /// #   Safety
    ///
    /// -   Assumes that the linked cells actually belong to the page!
    /// -   Assumes that the access to the linked cells, is exclusive.
    #[inline(never)]
    pub(crate) unsafe fn refill_foreign<F>(&self, list: &CellForeignList, mut recycler: F)
        where
            F: FnMut(*mut LargePage),
    {
        debug_assert!(!list.is_empty());
        debug_assert!(self.common.is_local_cell_pointer(list.head() as *mut u8));

        if self.foreign.refill(list) {
            recycler(self as *const _ as *mut _);
        }
    }

    //  Internal: Creates an instance of `LargePage` at the designated address.
    unsafe fn new<C>(at: *mut u8, owner: *mut (), class_size: ClassSize) -> Self
        where
            C: Configuration,
    {
        let large_page_size = C::LARGE_PAGE_SIZE;

        let layout = class_size.layout();
        let cell_size = layout.size();

        let reserved = cmp::max(mem::size_of::<Self>(), layout.align());
        let number_cells = class_size.number_elements(large_page_size.value() - reserved);
        debug_assert!(number_cells >= 1);

        let end = at.add(large_page_size.value());
        let begin = end.sub(number_cells * cell_size);
    
        let flush_threshold = cmp::max(number_cells / 64, 1);
        let catch_threshold = cmp::max(number_cells / 4, 1);

        let _prefetch = utils::PrefetchGuard::default();
        let common = Common::new(owner, class_size, flush_threshold, begin, end);
        let local = Local::new(cell_size, begin, end);
        let foreign = Foreign::new(catch_threshold);

        Self { _prefetch, common, local, foreign, }
    }
}

/// A linked-list of LargePages.
#[derive(Default)]
pub(crate) struct LargePageStack(LargePagePtr);

impl LargePageStack {
    /// Pops the head of the stack, it may be null.
    pub(crate) unsafe fn pop(&self) -> *mut LargePage {
        let mut head = self.0.load();

        loop {
            if head.is_null() {
                return head;
            }

            //  Safety:
            //  -   `head` is not null.
            let next = (*head).foreign.next.load();

            if let Some(new_head) = self.0.exchange(head, next) {
                head = new_head;
                continue;
            }

            return head;
        }
    }

    /// Pushes the page at the head of the stack.
    ///
    /// #   Safety
    ///
    /// -   `page` must not be null.
    /// -   `page.foreign.next` must be null.
    pub(crate) unsafe fn push(&self, page: *mut LargePage) {
        debug_assert!(!page.is_null());

        //  Safety:
        //  -   `page` is not null.
        let large_page = &*page;

        debug_assert!(large_page.foreign.next.is_null());

        let mut head = self.0.load();

        loop {
            large_page.foreign.next.store(head);

            if let Some(new_head) = self.0.exchange(head, page) {
                head = new_head;
                continue;
            }

            break;
        }
    }

    #[cfg(test)]
    pub(crate) unsafe fn peek(&self) -> *mut LargePage {
        self.0.load()
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
    //  The Class Size of the page, and thus all the properties of its cells.
    class_size: ClassSize,
    //  When the number of foreign cells in the ThreadLocal local list exceeds the flush threshold, then the list
    //  should be returned to this page instead.
    flush_threshold: usize,
    //  Pointer to the beginning of the page, the initial value of `self.next`.
    begin: *mut u8,
    //  Pointer to the end of the page; when `watermark == end`, the entire page has been carved.
    //
    //  When the entire page has been carved, acquiring new cells from the page is only possible through `foreign.freed`.
    end: *mut u8,
}

impl Common {
    /// Creates a new instance of `Common`.
    fn new(owner: *mut (), class_size: ClassSize, flush_threshold: usize, begin: *mut u8, end: *mut u8) -> Self {
        debug_assert!(flush_threshold >= 1);

        Self { owner, class_size, flush_threshold, begin, end, }
    }

    /// Checks whether the pointer points to a Cell within this page.
    fn is_local_cell_pointer(&self, ptr: *mut u8) -> bool {
        let (ptr, begin, end) = (ptr as usize, self.begin as usize, self.end as usize);

        begin <= ptr && ptr < end
    }
}

//  Local data. Only accessible from the local thread.
#[repr(align(128))]
struct Local {
    //  Pointer to the next free cell, if any.
    next: CellLocalPtr,
    //  Pointer to the beginning of the uncarved area of the page.
    //
    //  Linking all the cells together when creating the page would take too long, so instead only the first cell is
    //  prepared, and the `watermark` and `end` are initialized to denote the area of the page which can freely be
    //  carved into further cells.
    watermark: cell::Cell<*mut u8>,
    //  Pointer to the end of the page; when `watermark == end`, the entire page has been carved.
    //
    //  When the entire page has been carved, acquiring new cells from the page is only possible through `foreign.freed`.
    end: *mut u8,
    //  Size, in bytes, of the cells.
    cell_size: usize,
}

impl Local {
    /// Creates a new instance of `Local`.
    ///
    /// #   Safety
    ///
    /// -   `begin` and `end` are assumed to be correctly aligned and sized for a `CellForeign` pointer.
    /// -   `end - begin` is assumed to be a multiple of `cell_size`.
    unsafe fn new(cell_size: usize, begin: *mut u8, end: *mut u8) -> Self {
        debug_assert!(cell_size >= 1);
        debug_assert!((end as usize - begin as usize) % cell_size == 0,
            "cell_size: {}, begin: {:x}, end: {:x}", cell_size, begin as usize, end as usize);

        let next = CellLocalPtr::from_raw(begin);
        let watermark = cell::Cell::new(begin.add(cell_size));

        Self { next, watermark, end, cell_size, }
    }

    /// Allocates one cell from the page, if any.
    ///
    /// Returns a null pointer is no cell is available.
    fn allocate(&self) -> *mut u8 {
        //  Fast Path.
        if let Some(cell) = self.next.pop() {
            return cell.as_ptr() as *mut u8;
        }

        //  Cruise path.
        if self.watermark.get() == self.end {
            return ptr::null_mut();
        }

        //  Expansion path.
        let result = self.watermark.get();

        //  Safety:
        //  -   `self.cell_size` matches the size of the cells.
        //  -   `self.watermark` is still within bounds.
        unsafe { self.watermark.set(result.add(self.cell_size)) };

        result as *mut u8
    }

    /// Deallocates one cell from the page.
    ///
    /// #   Safety
    ///
    /// -   Assumes that `ptr` is sufficiently sized.
    /// -   Assumes that `ptr` is sufficiently aligned.
    unsafe fn deallocate(&self, ptr: *mut u8) {
        debug_assert!(utils::is_sufficiently_aligned_for(ptr, PowerOf2::align_of::<CellLocal>()));

        //  Safety:
        //  -   `ptr` is assumed to be non-null.
        //  -   `ptr` is assumed to be sufficiently sized.
        //  -   `ptr` is assumed to be sufficiently aligned.
        #[allow(clippy::cast_ptr_alignment)]
        let cell = ptr::NonNull::new_unchecked(ptr as *mut CellLocal);

        self.next.push(cell);
    }

    /// Extends the local list from a foreign list.
    ///
    /// #   Safety
    ///
    /// -   Assumes that the access to the linked cells, is exclusive.
    unsafe fn extend(&self, list: &CellForeignList) {
        debug_assert!(!list.is_empty());

        //  Safety:
        //  -   It is assumed that access to the cell, and all linked cells, is exclusive.
        self.next.extend(list);
    }

    /// Refills the local list from a foreign list.
    ///
    /// #   Safety
    ///
    /// -   Assumes that access to the cell, and all linked cells, is exclusive.
    unsafe fn refill(&self, list: ptr::NonNull<CellLocal>) {
        //  Safety:
        //  -   It is assumed that access to the cell, and all linked cells, is exclusive.
        self.next.refill(list);
    }
}

//  Foreign data. Accessible both from the local thread and foreign threads, at the cost of synchronization.
#[repr(align(128))]
struct Foreign {
    //  Pointer to the next LargePage.
    next: LargePagePtr,
    //  List of cells returned by other threads.
    freed: CellAtomicForeignPtr,
    //  The adrift marker.
    //
    //  A page that is too full is marked as "adrift", and cast away. As cells are freed -- as denoted by next.length
    //  -- the adrift page will be caught, ready to be used for allocations again.
    adrift: Adrift,
    //  When the number of freed cells exceeds this threshold, the page should be caught.
    catch_threshold: usize,
}

impl Foreign {
    /// Creates a new instance of `Foreign`.
    fn new(catch_threshold: usize) -> Self {
        debug_assert!(catch_threshold >= 1);

        let next = LargePagePtr::default();
        let freed = CellAtomicForeignPtr::default();
        let adrift = Adrift::default();

        Self { next, freed, adrift, catch_threshold, }
    }

    /// Attempts to refill `local` from foreign, and allocate.
    ///
    /// In case of failure, the page is cast adrift and null is returned.
    unsafe fn allocate(&self, local: &Local) -> *mut u8 {
        //  If the number of freed cells is sufficient to catch the page, immediately recycle them.
        if self.freed.len() >= self.catch_threshold {
            return self.recycle_allocate(local);
        }

        //  Cast the page adrift.
        let generation = self.adrift.cast_adrift();
        debug_assert!(generation % 2 != 0);

        //  There is an inherent race-condition, above, as a foreign thread may have extended the freed list beyond
        //  the catch threshold and yet seen a non-adrift page. The current thread therefore needs to check again.
        if self.freed.len() < self.catch_threshold {
            return ptr::null_mut();
        }

        //  A race-condition DID occur! Let's attempt to catch the adrift page then!
        if self.adrift.catch(generation) {
            //  The page was successfully caught, it is the current thread's unique property again.
            self.recycle_allocate(local)
        } else {
            //  The page was caught by another thread, it can no longer be used by this thread.
            ptr::null_mut()
        }
    }

    /// Refills foreign.
    ///
    /// Returns true if the page was adrift and has been caught, false otherwise.
    ///
    /// #   Safety
    ///
    /// -   Assumes that the linked cells actually belong to the page!
    unsafe fn refill(&self, list: &CellForeignList) -> bool {
        let len = self.freed.extend(list);

        if len < self.catch_threshold {
            return false;
        }

        if let Some(adrift) = self.adrift.is_adrift() {
            self.adrift.catch(adrift)
        } else {
            false
        }
    }

    //  Internal: Recycles the freed cells and immediately allocate.
    //
    //  #   Safety
    //
    //  -   Assumes that the current LargePage is owned by the current thread.
    unsafe fn recycle_allocate(&self, local: &Local) -> *mut u8 {
        debug_assert!(self.freed.len() >= self.catch_threshold);

        //  Safety:
        //  -   It is assumed that the current thread owns the LargePage, otherwise access to `local` is racy.
        let list = ptr::NonNull::new_unchecked(self.freed.steal());

        //  Safety:
        //  -   Access to `list` is exclusive.
        local.refill(CellLocal::from_atomic(list));

        local.allocate()
    }
}

//  Adrift "boolean"
// 
//  A memory allocator needs to track pages that are empty, partially used, or completely used.
// 
//  Tracking the latter is really unfortunate, moving them and out of concurrent lists means contention with other
//  other threads, for pages that are completely unusable to fulfill allocation requests.
// 
//  Enters the Anchored/Adrift mechanism!
// 
//  Rather than keeping track of full pages within a special list, they are instead "cast adrift".
// 
//  The trick is to catch and anchor them back when the user deallocates memory, for unlike a leaked page, a page that
//  is cast adrift is still pointed to: by the user's allocations.
// 
//  Essentially, thus, the page is cast adrift when full, and caught back when "sufficiently" empty.
// 
//  To avoid ABA issues with a boolean, a counter is used instead:
//  -   An even value means "anchored".
//  -   An odd value means "adrift".
struct Adrift(atomic::AtomicU64);

impl Adrift {
    //  Creates an anchored instance.
    fn new() -> Self { Self(atomic::AtomicU64::new(0)) }

    //  Checks whether the value is adrift.
    //
    //  If adrift, returns the current value, otherwise returns None.
    fn is_adrift(&self) -> Option<u64> {
        let current = self.load();
        if current % 2 != 0 { Some(current) } else { None }
    }

    //  Casts the value adrift, incrementing the counter.
    //
    //  Returns the (new) current value.
    fn cast_adrift(&self) -> u64 {
        let before = self.0.fetch_add(1, atomic::Ordering::AcqRel);
        debug_assert!(before % 2 == 0, "before: {}", before);

        before + 1
    }

    //  Attempts to catch the value, returns true if it succeeds.
    fn catch(&self, current: u64) -> bool { 
        debug_assert!(current % 2 != 0);

        self.0.compare_exchange(current, current + 1, atomic::Ordering::AcqRel, atomic::Ordering::Relaxed).is_ok()
    }

    //  Internal: load.
    fn load(&self) -> u64 { self.0.load(atomic::Ordering::Acquire) }
}

impl Default for Adrift {
    fn default() -> Self { Self::new() }
}

//  largePagePtr
struct LargePagePtr(atomic::AtomicPtr<LargePage>);

impl LargePagePtr {
    fn is_null(&self) -> bool { self.load().is_null() }

    fn load(&self) -> *mut LargePage { self.0.load(atomic::Ordering::Acquire) }

    fn store(&self, page: *mut LargePage) { self.0.store(page, atomic::Ordering::Release); }

    //  Returns None if the exchange succeed, or the new current value otherwise.
    fn exchange(&self, current: *mut LargePage, new: *mut LargePage) -> Option<*mut LargePage> {
        self.0.compare_exchange(current, new, atomic::Ordering::AcqRel, atomic::Ordering::Acquire).err()
    }
}

impl Default for LargePagePtr {
    fn default() -> Self { Self(atomic::AtomicPtr::new(ptr::null_mut())) }
}

#[cfg(test)]
mod tests {

use core::{mem, ops, slice};

use super::*;
use crate::internals::cells::{CellForeign, CellAtomicForeign};

const CELL_SIZE: usize = 32;

struct CellStore([usize; 256]);

impl CellStore {
    const CAPACITY: usize = 256;

    fn get(&self, index: usize) -> *mut usize { &self.0[index] as *const _ as *mut _ }

    /// Borrows self, outside of the compiler's overview.
    unsafe fn create_local(&mut self, cell_size: usize) -> Local {
        assert!(cell_size >= mem::size_of::<CellForeign>());

        let (begin, end) = self.begin_end(cell_size);

        Local::new(cell_size, begin, end)
    }

    /// Creates a `CellForeignList` containing the specified range of cells.
    ///
    /// #   Safety
    ///
    /// -   `local` should have been created from this instance.
    /// -   The cells should not _also_ be available through `local.next`.
    unsafe fn create_foreign_list(&self, local: &Local, cells: ops::Range<usize>) -> CellForeignList {
        assert!(cells.start <= cells.end);

        let cell_size = local.cell_size;

        let (begin, end) = self.begin_end(cell_size);
        assert_eq!(end, local.end);
        assert!(cells.end <= (end as usize - begin as usize) / cell_size);

        let list = CellForeignList::default();

        for index in cells.rev() {
            let cell_address = begin.add(index * cell_size);
            let cell = ptr::NonNull::new(cell_address as *mut CellForeign).unwrap();
            list.push(cell);
        }

        list
    }

    /// Creates a stack of `CellForeign` containing the specified range of cells.
    ///
    /// #   Safety
    ///
    /// -   `local` should have been created from this instance.
    /// -   The cells should not _also_ be available through `local.next`.
    unsafe fn create_foreign_stack(&self, local: &Local, cells: ops::Range<usize>) -> ptr::NonNull<CellAtomicForeign> {
        let list = self.create_foreign_list(local, cells);

        let cell = CellAtomicForeignPtr::default();
        cell.extend(&list);

        ptr::NonNull::new(cell.steal()).unwrap()
    }

    //  Internal: Compute begin and end for a given `cell_size`.
    unsafe fn begin_end(&self, cell_size: usize) -> (*mut u8, *mut u8) {
        let begin = self as *const Self as *mut Self as *mut u8;
        let end = begin.add(mem::size_of::<Self>());

        let number_elements = (end as usize - begin as usize) / cell_size;
        let begin = end.sub(number_elements * cell_size);

        (begin, end)
    }
}

impl Default for CellStore {
    fn default() -> Self {
        let result: Self = unsafe { mem::zeroed() };
        assert_eq!(Self::CAPACITY, result.0.len());

        result
    }
}

#[test]
fn implementation_sizes() {
    assert_eq!(128, mem::size_of::<utils::PrefetchGuard>());
    assert_eq!(128, mem::size_of::<Common>());
    assert_eq!(128, mem::size_of::<Local>());
    assert_eq!(128, mem::size_of::<Foreign>());
    assert_eq!(4 * 128, mem::size_of::<LargePage>());
}

#[test]
fn large_page_ptr() {
    let null: *mut LargePage = ptr::null_mut();
    let other: *mut LargePage = 1234usize as *mut _;

    let large_page = LargePagePtr::default();
    assert!(large_page.is_null());
    assert_eq!(null, large_page.load());

    large_page.store(other);
    assert!(!large_page.is_null());
    assert_eq!(other, large_page.load());

    assert_eq!(Some(other), large_page.exchange(null, other));
    assert_eq!(other, large_page.load());

    assert_eq!(None, large_page.exchange(other, null));
    assert_eq!(null, large_page.load());
}

#[test]
fn adrift() {
    let adrift = Adrift::default();
    assert_eq!(None, adrift.is_adrift());

    assert_eq!(1, adrift.cast_adrift());
    assert_eq!(Some(1), adrift.is_adrift());

    assert!(!adrift.catch(3));
    assert_eq!(Some(1), adrift.is_adrift());

    assert!(adrift.catch(1));
    assert_eq!(None, adrift.is_adrift());

    assert_eq!(3, adrift.cast_adrift());
    assert_eq!(Some(3), adrift.is_adrift());
}

#[test]
fn common_is_local_cell_pointer() {
    let common = Common::new(ptr::null_mut(), ClassSize::new(0), 1, 0x40 as *mut u8, 0x60 as *mut u8);

    assert!(!common.is_local_cell_pointer(0x3f as *mut u8));
    
    assert!(common.is_local_cell_pointer(0x40 as *mut u8));
    assert!(common.is_local_cell_pointer(0x48 as *mut u8));
    assert!(common.is_local_cell_pointer(0x50 as *mut u8));
    assert!(common.is_local_cell_pointer(0x58 as *mut u8));
    assert!(common.is_local_cell_pointer(0x5f as *mut u8));

    assert!(!common.is_local_cell_pointer(0x60 as *mut u8));
}

#[test]
fn local_new() {
    //  This test actually tests `cell_store.create_local` more than anything.
    //  Since further tests will depend on it correctly initializing `Local`, it is better to validate it early.
    let mut cell_store = CellStore::default();
    let end_store = cell_store.get(0) as usize + CELL_SIZE / 4 * CellStore::CAPACITY;

    {
        let local = unsafe { cell_store.create_local(CELL_SIZE * 2) };
        assert_eq!(cell_store.get(0) as usize, local.next.peek().unwrap().as_ptr() as usize);
        assert_eq!(cell_store.get(8) as usize, local.watermark.get() as usize);
        assert_eq!(end_store, local.end as usize);
    }

    {
        let local = unsafe { cell_store.create_local(CELL_SIZE * 2 + CELL_SIZE / 2) };
        assert_eq!(cell_store.get(6) as usize, local.next.peek().unwrap().as_ptr() as usize);
        assert_eq!(cell_store.get(16) as usize, local.watermark.get() as usize);
        assert_eq!(end_store, local.end as usize);
    }
}

#[test]
fn local_allocate_expansion() {
    let mut cell_store = CellStore::default();
    let local = unsafe { cell_store.create_local(CELL_SIZE) };

    //  Bump watermark until it is no longer possible.
    for i in 0..64 {
        assert_eq!(cell_store.get(4 * i) as usize, local.allocate() as usize);
    }

    assert_eq!(local.end, local.watermark.get());

    assert_eq!(ptr::null_mut(), local.allocate());
}

#[test]
fn local_allocate_deallocate_ping_pong() {
    let mut cell_store = CellStore::default();
    let local = unsafe { cell_store.create_local(CELL_SIZE) };

    let ptr = local.allocate();

    for _ in 0..10 {
        unsafe { local.deallocate(ptr) };
        assert_eq!(ptr, local.allocate());
    }

    assert_eq!(cell_store.get(4) as usize, local.watermark.get() as usize);
}

#[test]
fn local_extend() {
    let mut cell_store = CellStore::default();
    let local = unsafe { cell_store.create_local(CELL_SIZE) };

    //  Allocate all.
    while !local.allocate().is_null() {}

    let foreign_list = unsafe { cell_store.create_foreign_list(&local, 3..7) };

    unsafe { local.extend(&foreign_list) };
    assert!(foreign_list.is_empty());

    for i in 3..7 {
        assert_eq!(cell_store.get(4 * i) as usize, local.allocate() as usize);
    }
}

#[test]
fn local_refill() {
    let mut cell_store = CellStore::default();
    let local = unsafe { cell_store.create_local(CELL_SIZE) };

    //  Allocate all.
    while !local.allocate().is_null() {}

    let foreign = unsafe { cell_store.create_foreign_stack(&local, 3..7) };

    unsafe { local.refill(CellLocal::from_atomic(foreign)) };

    for i in 3..7 {
        assert_eq!(cell_store.get(4 * i) as usize, local.allocate() as usize);
    }
}

#[test]
fn foreign_refill() {
    let mut cell_store = CellStore::default();
    let local = unsafe { cell_store.create_local(CELL_SIZE) };

    //  Allocate all.
    while !local.allocate().is_null() {}

    let foreign = Foreign::new(16);

    //  Insufficient number of elements.
    let list = unsafe { cell_store.create_foreign_list(&local, 3..7) };
    assert!(unsafe { !foreign.refill(&list) });

    //  Sufficient number, but not adrift.
    let list = unsafe { cell_store.create_foreign_list(&local, 7..32) };
    assert!(unsafe { !foreign.refill(&list) });

    //  Number already sufficient, and now was adrift.
    foreign.adrift.cast_adrift();
    assert_eq!(Some(1), foreign.adrift.is_adrift());

    let list = unsafe { cell_store.create_foreign_list(&local, 0..3) };
    assert!(unsafe { foreign.refill(&list) });
}

#[test]
fn foreign_allocate() {
    let mut cell_store = CellStore::default();
    let local = unsafe { cell_store.create_local(CELL_SIZE) };

    //  Allocate all.
    while !local.allocate().is_null() {}

    let foreign = Foreign::new(16);

    //  Enough elements, immediate recycling.
    let list = unsafe { cell_store.create_foreign_list(&local, 0..32) };
    assert!(unsafe { !foreign.refill(&list) });

    assert_eq!(cell_store.get(0) as usize, unsafe { foreign.allocate(&local) } as usize);

    //  Local was refilled with 32 elements (one already consumed above).
    for i in 1..32 {
        assert_eq!(cell_store.get(4 * i) as usize, local.allocate() as usize);
    }

    assert_eq!(ptr::null_mut(), local.allocate());

    //  Not enough elements, cast the page adrift.
    let list = unsafe { cell_store.create_foreign_list(&local, 32..40) };
    assert!(unsafe { !foreign.refill(&list) });

    assert_eq!(ptr::null_mut(), unsafe { foreign.allocate(&local) });
    assert_eq!(Some(1), foreign.adrift.is_adrift());

    //  The race-condition cannot be tested in single-threaded code.
}

struct TestConfiguration;

impl Configuration for TestConfiguration {
    const LARGE_PAGE_SIZE: PowerOf2 = unsafe { PowerOf2::new_unchecked(2048) };
    const HUGE_PAGE_SIZE: PowerOf2 = unsafe { PowerOf2::new_unchecked(8 * 1024 * 1024) };
}

#[derive(Clone, Copy)]
#[repr(align(2048))]
struct LargePageStore([usize; 256]);

impl LargePageStore {
    unsafe fn initialize(&mut self, class_size: ClassSize) -> *mut LargePage {
        let start = self.address() as *mut u8;
        let size = mem::size_of::<Self>();

        assert_eq!(size, TestConfiguration::LARGE_PAGE_SIZE.value());

        let place = slice::from_raw_parts_mut(start, size);

        LargePage::initialize::<TestConfiguration>(place, self.address(), class_size)
    }

    fn create_foreign_list(&self, large_page: &LargePage, cells: ops::Range<usize>) -> CellForeignList {
        assert_eq!(self.address(), large_page.owner());

        let cell_size = large_page.local.cell_size;
        let (begin, end) = (large_page.common.begin, large_page.common.end);

        assert!(cells.end * cell_size <= (end as usize - begin as usize));

        let list = CellForeignList::default();

        for index in cells.rev() {
            let cell_address = unsafe { begin.add(index * cell_size) };
            let cell = ptr::NonNull::new(cell_address as *mut CellForeign).unwrap();
            list.push(cell);
        }

        list
    }

    fn address(&self) -> *mut () { self as *const Self as *mut Self as *mut () }

    fn inner_pointer(&self) -> *mut u8 {
        let base_ptr = self.address() as *mut u8;
        unsafe { base_ptr.add(mem::size_of::<Self>() / 2) }
    }
}

impl Default for LargePageStore {
    fn default() -> Self { unsafe { mem::zeroed() } }
}

#[test]
fn large_page_initialize_from_raw() {
    let mut store = LargePageStore::default();

    let class_size = ClassSize::new(4);
    let large_page_ptr = unsafe { store.initialize(class_size) };
    let large_page = unsafe { &*large_page_ptr };

    assert_eq!(store.address(), large_page.owner());
    assert_eq!(class_size, large_page.class_size());
    assert!(large_page.flush_threshold() >= 1);

    let random_ptr = store.inner_pointer();
    let other_page_ptr = unsafe { LargePage::from_raw::<TestConfiguration>(random_ptr) };
    assert_eq!(large_page_ptr, other_page_ptr);
}

#[test]
fn large_page_allocate_deallocate_local() {
    let mut store = LargePageStore::default();

    let large_page_ptr = unsafe { store.initialize(ClassSize::new(4)) };
    let large_page = unsafe { &*large_page_ptr };

    let cell_size = large_page.local.cell_size;
    assert_eq!(64, cell_size);

    let mut last = 0usize;
    let mut counter = 0;

    loop {
        let new = unsafe { large_page.allocate() };

        if new.is_null() {
            break;
        }

        let new = new as usize;

        if last != 0 {
            assert_eq!(last + cell_size, new);
        }

        last = new;
        counter += 1;
    }

    assert_eq!(24, counter);
    assert_eq!(Some(1), large_page.foreign.adrift.is_adrift());

    for _ in 0..counter {
        unsafe { large_page.deallocate(last as *mut u8) };
        last -= cell_size;
    }

    assert!(large_page.foreign.adrift.catch(1));

    for _ in 0..counter {
        let new = unsafe { large_page.allocate() };

        assert_ne!(ptr::null_mut(), new);
        assert_eq!(last + cell_size, new as usize);

        last += cell_size;
    }

    assert_eq!(ptr::null_mut(), unsafe { large_page.allocate() });
    assert_eq!(Some(3), large_page.foreign.adrift.is_adrift());
}

#[test]
fn large_page_refill_local() {
    let mut store = LargePageStore::default();

    let large_page_ptr = unsafe { store.initialize(ClassSize::new(4)) };
    let large_page = unsafe { &*large_page_ptr };

    //  Allocate a few from local.
    let mut allocated: [usize; 12] = Default::default();

    for i in 0..12 {
        allocated[i] = unsafe { large_page.allocate() } as usize;
    }

    //  Refill.
    let list = store.create_foreign_list(large_page, 6..12);
    unsafe { large_page.refill_local(&list) };

    //  The list was consumed.
    assert_eq!(0, list.len());

    //  New allocated pointers match previous allocated ones.
    for i in 6..12 {
        let new = unsafe { large_page.allocate() } as usize;
        assert_eq!(allocated[i], new);
    }
}

#[test]
fn large_page_refill_local_skip() {
    let mut store = LargePageStore::default();
    let mut other_store = LargePageStore::default();

    let large_page_ptr = unsafe { store.initialize(ClassSize::new(4)) };
    let large_page = unsafe { &*large_page_ptr };

    let other_page_ptr = unsafe { other_store.initialize(ClassSize::new(4)) };
    let other_page = unsafe { &*other_page_ptr };

    //  Allocate one from local as a reference.
    let allocated = unsafe { large_page.allocate() } as usize;

    //  Attempt refill from cells from another LargePage, it should be skipped.
    let other_list = other_store.create_foreign_list(other_page, 0..6);
    unsafe { large_page.refill_local(&other_list) };

    //  The list was not consumed.
    assert_eq!(6, other_list.len());

    //  The next allocated pointer is as expected.
    assert_eq!(allocated + large_page.local.cell_size, unsafe { large_page.allocate() } as usize);
}

#[test]
fn large_page_refill_foreign_not_adrift() {
    let mut store = LargePageStore::default();

    let large_page_ptr = unsafe { store.initialize(ClassSize::new(4)) };
    let large_page = unsafe { &*large_page_ptr };

    //  Allocate a few from local.
    let mut allocated: [usize; 12] = Default::default();

    for i in 0..12 {
        allocated[i] = unsafe { large_page.allocate() } as usize;
    }

    //  Refill.
    let list = store.create_foreign_list(large_page, 6..12);
    unsafe { large_page.refill_foreign(&list, |_| unreachable!()) };

    //  The list was consumed.
    assert_eq!(0, list.len());

    //  Still use local pointers for a while.
    for _ in 12..24 {
        unsafe { large_page.allocate() };
    }

    //  Then switch to refilled pointers.
    for i in 6..12 {
        let new = unsafe { large_page.allocate() } as usize;
        assert_eq!(allocated[i], new);
    }
}

#[test]
fn large_page_refill_foreign_adrift() {
    let mut store = LargePageStore::default();

    let large_page_ptr = unsafe { store.initialize(ClassSize::new(4)) };
    let large_page = unsafe { &*large_page_ptr };
    assert_eq!(6, large_page.foreign.catch_threshold);

    //  Empty local.
    let mut allocated: [usize; 24] = Default::default();

    for i in 0..24 {
        allocated[i] = unsafe { large_page.allocate() } as usize;
    }

    assert_eq!(ptr::null_mut(), unsafe { large_page.allocate() });
    assert_eq!(Some(1), large_page.foreign.adrift.is_adrift());

    //  Refill.
    let list = store.create_foreign_list(large_page, 3..6);
    unsafe { large_page.refill_foreign(&list, |_| unreachable!()) };

    //  The list was consumed.
    assert_eq!(0, list.len());
    assert_eq!(3, large_page.foreign.freed.len());

    //  The page is still adrift.
    assert_eq!(Some(1), large_page.foreign.adrift.is_adrift());

    //  Refill with enough supplementary elements.
    let mut caught: *mut LargePage = ptr::null_mut();
    let caught = &mut caught;

    let list = store.create_foreign_list(large_page, 0..3);
    unsafe { large_page.refill_foreign(&list, |c| *caught = c) };
    
    //  The list was consumed.
    assert_eq!(0, list.len());
    assert_eq!(6, large_page.foreign.freed.len());

    //  The page was recycled.
    assert_eq!(large_page_ptr, *caught);

    //  The page is no longer adrift.
    assert_eq!(None, large_page.foreign.adrift.is_adrift());

    //  Allocating works again!
    for i in 0..6 {
        let new = unsafe { large_page.allocate() } as usize;
        assert_eq!(allocated[i], new);
    }
}

#[test]
fn large_page_stack() {
    let class_size = ClassSize::new(2);

    let mut stores = [LargePageStore::default(); 6];

    let pointers = {
        let mut result: [*mut LargePage; 6] = unsafe { mem::zeroed() };

        for i in 0..6 {
            result[i] = unsafe { stores[i].initialize(class_size) };
        }

        result
    };

    let stack = LargePageStack::default();
    assert_eq!(ptr::null_mut(), unsafe { stack.pop() });

    for large_page in pointers.iter().rev() {
        unsafe { stack.push(*large_page) };
    }

    for large_page in pointers.iter() {
        assert_eq!(*large_page, unsafe { stack.pop() });
    }

    assert_eq!(ptr::null_mut(), unsafe { stack.pop() });
}

}
