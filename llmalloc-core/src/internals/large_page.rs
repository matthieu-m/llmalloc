//! Large Page
//!
//! A Large Page is a slab of `Configuration::LARGE_PAGE_SIZE` bytes, which fulfills allocations of the Normal category.
//!
//! Each instance of a Large Page can only fulfill allocations of a specific ClassSize.

mod adrift;
mod foreign;
mod local;

#[cfg(test)]
mod test;

use core::{
    cmp,
    mem,
    ptr::{self, NonNull},
    sync::atomic::{self, Ordering},
};

use crate::{ClassSize, Configuration};
use crate::{
    internals::{
        atomic::AtomicPtr,
        blocks::BlockForeignList,
    },
    utils,
};

use foreign::Foreign;
use local::Local;

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
    pub(crate) unsafe fn initialize<C>(place: &mut [u8], owner: *mut (), class_size: ClassSize) -> NonNull<Self>
        where
            C: Configuration,
    {
        debug_assert!(place.len() >= C::LARGE_PAGE_SIZE.value());

        //  Safety:
        //  -   `place` is not a null slice.
        let at = NonNull::new_unchecked(place.as_mut_ptr());

        debug_assert!(utils::is_sufficiently_aligned_for(at, C::LARGE_PAGE_SIZE));
        debug_assert!(!utils::is_sufficiently_aligned_for(at, C::HUGE_PAGE_SIZE));

        //  Safety:
        //  -   `at` is assumed to be sufficiently sized.
        //  -   `at` is assumed to be sufficiently aligned.
        #[allow(clippy::cast_ptr_alignment)]
        let large_page = at.as_ptr() as *mut Self;

        //  Safety:
        //  -   `large_page` is accessed exclusively from this thread.
        ptr::write(large_page, Self::new::<C>(at, owner, class_size));

        //  Enforce memory ordering, later Acquire need to see those 0s and 1s.
        atomic::fence(Ordering::Release);

        at.cast()
    }

    /// Obtain the large page associated to a given allocation.
    ///
    /// #   Safety
    ///
    /// -   Assumes that the pointer is pointing strictly _inside_ a LargePage.
    pub(crate) unsafe fn from_raw<C>(ptr: NonNull<u8>) -> NonNull<Self>
        where
            C: Configuration,
    {
        debug_assert!(!utils::is_sufficiently_aligned_for(ptr, C::LARGE_PAGE_SIZE));

        let address = ptr.as_ptr() as usize;
        let large_page = C::LARGE_PAGE_SIZE.round_down(address);

        //  Safety:
        //  -   Not null, by assumption.
        NonNull::new_unchecked(large_page as *mut Self)
    }

    /// Returns the owner of the page.
    pub(crate) fn owner(&self) -> *mut () { self.common.owner }

    /// Returns the class size of the page.
    pub(crate) fn class_size(&self) -> ClassSize { self.common.class_size }

    /// Returns the flush threshold of the page.
    pub(crate) fn flush_threshold(&self) -> usize { self.common.flush_threshold }

    /// Returns the catch threshold of the page.
    #[cfg(test)]
    pub(crate) fn catch_threshold(&self) -> usize { self.foreign.catch_threshold() }

    /// Allocate one block from the page, if any.
    ///
    /// Returns a null pointer if no block is available.
    ///
    /// #   Safety
    ///
    /// -   Assumes that a single thread calls `allocate` at a time.
    pub(crate) unsafe fn allocate(&self) -> Option<NonNull<u8>> {
        self.local.allocate()
            .or_else(|| self.foreign.allocate(&self.local))
    }

    /// Deallocate a block from the local thread.
    ///
    /// #   Safety
    ///
    /// -   Assumes that a single thread calls `deallocate` at a time.
    /// -   Assumes that the pointer is pointing to a block inside _this_ LargePage.
    /// -   Assumes that the pointed block is no longer in use.
    pub(crate) unsafe fn deallocate(&self, ptr: NonNull<u8>) {
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
    pub(crate) unsafe fn refill_local(&self, list: &BlockForeignList) {
        debug_assert!(!list.is_empty());

        let head = match list.head() {
            None => return,
            Some(head) => head,
        };

        if !self.common.is_local_cell_pointer(head.cast()) {
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
    pub(crate) unsafe fn refill_foreign<F>(&self, list: &BlockForeignList, mut recycler: F)
        where
            F: FnMut(NonNull<LargePage>),
    {
        debug_assert!(!list.is_empty());
        debug_assert!(self.common.is_local_cell_pointer(list.head().expect("Not empty").cast()));

        if self.foreign.refill(list) {
            recycler(NonNull::from(self));
        }
    }

    //  Internal: Creates an instance of `LargePage` at the designated address.
    unsafe fn new<C>(at: NonNull<u8>, owner: *mut (), class_size: ClassSize) -> Self
        where
            C: Configuration,
    {
        let large_page_size = C::LARGE_PAGE_SIZE;

        let layout = class_size.layout();
        let block_size = layout.size();

        let reserved = cmp::max(mem::size_of::<Self>(), layout.align());
        let number_cells = class_size.number_elements(large_page_size.value() - reserved);
        debug_assert!(number_cells >= 1);

        let end = NonNull::new_unchecked(at.as_ptr().add(large_page_size.value()));
        let begin = NonNull::new_unchecked(end.as_ptr().sub(number_cells * block_size));
    
        let flush_threshold = cmp::max(number_cells / 64, 1);
        let catch_threshold = cmp::max(number_cells / 4, 1);

        let _prefetch = utils::PrefetchGuard::default();
        let common = Common::new(owner, class_size, flush_threshold, begin, end);
        let local = Local::new(block_size, begin, end);
        let foreign = Foreign::new(catch_threshold);

        Self { _prefetch, common, local, foreign, }
    }
}

/// A linked-list of LargePages.
#[derive(Default)]
pub(crate) struct LargePageStack(AtomicPtr<LargePage>);

impl LargePageStack {
    /// Pops the head of the stack, it may be null.
    pub(crate) fn pop(&self) -> Option<NonNull<LargePage>> {
        let mut head = self.0.load();

        loop {
            let head_ptr = head?;

            //  Safety:
            //  -   `head_ptr` points to a valid instance.
            //  -   The lifetime is bounded.
            let next = unsafe { head_ptr.as_ref() }.foreign.next.load().map(NonNull::cast);

            if let Err(new_head) = self.0.compare_exchange(head, next) {
                head = new_head;
                continue;
            }

            debug_assert!(!head.is_none());

            //  Safety:
            //  -   `head_ptr` points to a valid instance.
            //  -   The lifetime is bounded.
            //
            //  Exclusive access to LargePage, so no data-race on store.
            unsafe { head_ptr.as_ref() }.foreign.next.store(None);

            return head;
        }
    }

    /// Pushes the page at the head of the stack.
    ///
    /// #   Safety
    ///
    /// -   `page` must not be null.
    /// -   `page.foreign.next` must be null.
    pub(crate) fn push(&self, page: NonNull<LargePage>) {
        //  Safety:
        //  -   `page` points to a valid instance.
        //  -   The lifetime is bounded.
        let large_page = unsafe { page.as_ref() };

        debug_assert!(large_page.foreign.next.load().is_none());

        let mut head = self.0.load().map(NonNull::cast);

        loop {
            //  Exclusive access to LargePage, so no data-race on store.
            large_page.foreign.next.store(head.map(NonNull::cast));

            if let Err(new_head) = self.0.compare_exchange(head, Some(page)) {
                head = new_head;
                continue;
            }

            break;
        }
    }

    #[cfg(test)]
    pub(crate) unsafe fn peek(&self) -> Option<NonNull<LargePage>> {
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
    begin: NonNull<u8>,
    //  Pointer to the end of the page; when `watermark == end`, the entire page has been carved.
    //
    //  When the entire page has been carved, acquiring new cells from the page is only possible through `foreign.freed`.
    end: NonNull<u8>,
}

impl Common {
    /// Creates a new instance of `Common`.
    fn new(
        owner: *mut (),
        class_size: ClassSize,
        flush_threshold: usize,
        begin: NonNull<u8>,
        end: NonNull<u8>
    )
        -> Self
    {
        debug_assert!(flush_threshold >= 1);

        Self { owner, class_size, flush_threshold, begin, end, }
    }

    /// Checks whether the pointer points to a Block within this page.
    fn is_local_cell_pointer(&self, ptr: NonNull<u8>) -> bool {
        let (ptr, begin, end) = (ptr.as_ptr() as usize, self.begin.as_ptr() as usize, self.end.as_ptr() as usize);

        begin <= ptr && ptr < end
    }
}

#[cfg(test)]
mod tests {

use core::{
    mem::{self, MaybeUninit},
    ops,
    slice,
};

use crate::{
    PowerOf2,
    internals::blocks::BlockForeign,
};

use super::*;

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
    let other = NonNull::new(1234usize as *mut LargePage).unwrap();

    let large_page = AtomicPtr::default();
    assert_eq!(None, large_page.load());

    large_page.store(Some(other));
    assert_eq!(Some(other), large_page.load());

    assert_eq!(Err(Some(other)), large_page.compare_exchange(None, Some(other)));
    assert_eq!(Some(other), large_page.load());

    assert_eq!(Ok(Some(other)), large_page.compare_exchange(Some(other), None));
    assert_eq!(None, large_page.load());
}

#[test]
fn common_is_local_cell_pointer() {
    fn pointer(x: usize) -> NonNull<u8> { NonNull::new(x as *mut u8).unwrap() }

    let common = Common::new(ptr::null_mut(), ClassSize::new(0), 1, pointer(0x40), pointer(0x60));

    assert!(!common.is_local_cell_pointer(pointer(0x3f)));
    
    assert!(common.is_local_cell_pointer(pointer(0x40)));
    assert!(common.is_local_cell_pointer(pointer(0x48)));
    assert!(common.is_local_cell_pointer(pointer(0x50)));
    assert!(common.is_local_cell_pointer(pointer(0x58)));
    assert!(common.is_local_cell_pointer(pointer(0x5f)));

    assert!(!common.is_local_cell_pointer(pointer(0x60)));
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
    unsafe fn initialize(&mut self, class_size: ClassSize) -> NonNull<LargePage> {
        let start = self.address() as *mut u8;
        let size = mem::size_of::<Self>();

        assert_eq!(size, TestConfiguration::LARGE_PAGE_SIZE.value());

        let place = slice::from_raw_parts_mut(start, size);

        LargePage::initialize::<TestConfiguration>(place, self.address(), class_size)
    }

    fn create_foreign_list(&self, large_page: &LargePage, blocks: ops::Range<usize>) -> BlockForeignList {
        assert_eq!(self.address(), large_page.owner());

        let block_size = large_page.local.block_size();
        let (begin, end) = (large_page.common.begin, large_page.common.end);

        assert!(blocks.end * block_size <= (end.as_ptr() as usize - begin.as_ptr() as usize));

        let list = BlockForeignList::default();

        for index in blocks.rev() {
            let block_address = unsafe { begin.as_ptr().add(index * block_size) };
            let block = NonNull::new(block_address as *mut BlockForeign).unwrap();
            list.push(block);
        }

        list
    }

    fn address(&self) -> *mut () { self as *const Self as *mut Self as *mut () }

    fn inner_pointer(&self) -> NonNull<u8> {
        let base_ptr = self.address() as *mut u8;
        NonNull::new(unsafe { base_ptr.add(mem::size_of::<Self>() / 2) }).unwrap()
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
    let large_page = unsafe { large_page_ptr.as_ref() };

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
    let large_page = unsafe { large_page_ptr.as_ref() };

    let block_size = large_page.local.block_size();
    assert_eq!(64, block_size);

    let mut last = 0usize;
    let mut counter = 0;

    loop {
        let new = unsafe { large_page.allocate() };

        if new.is_none() {
            break;
        }

        let new = new.unwrap().as_ptr() as usize;

        if last != 0 {
            assert_eq!(last + block_size, new);
        }

        last = new;
        counter += 1;
    }

    assert_eq!(24, counter);
    assert_eq!(Some(1), large_page.foreign.is_adrift());

    for _ in 0..counter {
        unsafe { large_page.deallocate(NonNull::new(last as *mut u8).unwrap()) };
        last -= block_size;
    }

    assert!(large_page.foreign.catch_adrift(1));

    for _ in 0..counter {
        let new = unsafe { large_page.allocate() };

        assert_ne!(None, new);
        assert_eq!(last + block_size, new.unwrap().as_ptr() as usize);

        last += block_size;
    }

    assert_eq!(None, unsafe { large_page.allocate() });
    assert_eq!(Some(3), large_page.foreign.is_adrift());
}

#[test]
fn large_page_refill_local() {
    let mut store = LargePageStore::default();

    let large_page_ptr = unsafe { store.initialize(ClassSize::new(4)) };
    let large_page = unsafe { large_page_ptr.as_ref() };

    //  Allocate a few from local.
    let mut allocated: [usize; 12] = Default::default();

    for i in 0..12 {
        allocated[i] = unsafe { large_page.allocate() }.unwrap().as_ptr() as usize;
    }

    //  Refill.
    let list = store.create_foreign_list(large_page, 6..12);
    unsafe { large_page.refill_local(&list) };

    //  The list was consumed.
    assert_eq!(0, list.len());

    //  New allocated pointers match previous allocated ones.
    for i in 6..12 {
        let new = unsafe { large_page.allocate() }.unwrap().as_ptr() as usize;
        assert_eq!(allocated[i], new);
    }
}

#[test]
fn large_page_refill_local_skip() {
    let mut store = LargePageStore::default();
    let mut other_store = LargePageStore::default();

    let large_page_ptr = unsafe { store.initialize(ClassSize::new(4)) };
    let large_page = unsafe { large_page_ptr.as_ref() };

    let other_page_ptr = unsafe { other_store.initialize(ClassSize::new(4)) };
    let other_page = unsafe { other_page_ptr.as_ref() };

    //  Allocate one from local as a reference.
    let allocated = unsafe { large_page.allocate() }.unwrap().as_ptr() as usize;

    //  Attempt refill from cells from another LargePage, it should be skipped.
    let other_list = other_store.create_foreign_list(other_page, 0..6);
    unsafe { large_page.refill_local(&other_list) };

    //  The list was not consumed.
    assert_eq!(6, other_list.len());

    //  The next allocated pointer is as expected.
    assert_eq!(allocated + large_page.local.block_size(), unsafe { large_page.allocate() }.unwrap().as_ptr() as usize);
}

#[test]
fn large_page_refill_foreign_not_adrift() {
    let mut store = LargePageStore::default();

    let large_page_ptr = unsafe { store.initialize(ClassSize::new(4)) };
    let large_page = unsafe { large_page_ptr.as_ref() };

    //  Allocate a few from local.
    let mut allocated: [usize; 12] = Default::default();

    for i in 0..12 {
        allocated[i] = unsafe { large_page.allocate() }.unwrap().as_ptr() as usize;
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
        let new = unsafe { large_page.allocate() }.unwrap().as_ptr() as usize;
        assert_eq!(allocated[i], new);
    }
}

#[test]
fn large_page_refill_foreign_adrift() {
    let mut store = LargePageStore::default();

    let large_page_ptr = unsafe { store.initialize(ClassSize::new(4)) };
    let large_page = unsafe { large_page_ptr.as_ref() };
    assert_eq!(6, large_page.catch_threshold());

    //  Empty local.
    let mut allocated: [usize; 24] = Default::default();

    for i in 0..24 {
        allocated[i] = unsafe { large_page.allocate() }.unwrap().as_ptr() as usize;
    }

    assert_eq!(None, unsafe { large_page.allocate() });
    assert_eq!(Some(1), large_page.foreign.is_adrift());

    //  Refill.
    let list = store.create_foreign_list(large_page, 3..6);
    unsafe { large_page.refill_foreign(&list, |_| unreachable!()) };

    //  The list was consumed.
    assert_eq!(0, list.len());
    assert_eq!(3, large_page.foreign.freed());

    //  The page is still adrift.
    assert_eq!(Some(1), large_page.foreign.is_adrift());

    //  Refill with enough supplementary elements.
    let mut caught: Option<NonNull<LargePage>> = None;
    let caught = &mut caught;

    let list = store.create_foreign_list(large_page, 0..3);
    unsafe { large_page.refill_foreign(&list, |c| *caught = Some(c)) };
    
    //  The list was consumed.
    assert_eq!(0, list.len());
    assert_eq!(6, large_page.foreign.freed());

    //  The page was recycled.
    assert_eq!(large_page_ptr, caught.unwrap());

    //  The page is no longer adrift.
    assert_eq!(None, large_page.foreign.is_adrift());

    //  Allocating works again!
    for i in 0..6 {
        let new = unsafe { large_page.allocate() }.unwrap().as_ptr() as usize;
        assert_eq!(allocated[i], new);
    }
}

#[test]
fn large_page_stack() {
    let class_size = ClassSize::new(2);

    let mut stores = [LargePageStore::default(); 6];

    let pointers = {
        let mut result: MaybeUninit<[NonNull<LargePage>; 6]> = MaybeUninit::uninit();

        for i in 0..6 {
            unsafe { (*result.as_mut_ptr())[i] = stores[i].initialize(class_size) };
        }

        unsafe { result.assume_init() }
    };

    let stack = LargePageStack::default();
    assert_eq!(None, stack.pop());

    for large_page in pointers.iter().rev() {
        stack.push(*large_page);
    }

    for large_page in pointers.iter() {
        assert_eq!(Some(*large_page), stack.pop());
    }

    assert_eq!(None, stack.pop());
}

} // mod tests
