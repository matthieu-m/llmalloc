//! Huge Page
//!
//! A Huge Page is a polymorphic allocator of `Configuration::HUGE_PAGE_SIZE` bytes, which fulfills allocations of the
//! Large category.
//!
//! A `SocketLocal` may own multiple 

use core::{alloc::Layout, cmp, mem, num, ptr, slice, sync::atomic};

use crate::{Configuration, PowerOf2};
use crate::utils;

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
    pub(crate) unsafe fn initialize<C>(place: &mut [u8], owner: *mut ()) -> *mut Self
        where
            C: Configuration,
    {
        debug_assert!(place.len() >= C::HUGE_PAGE_SIZE.value());

        let at = place.as_mut_ptr();

        debug_assert!(!at.is_null());
        debug_assert!(utils::is_sufficiently_aligned_for(at, C::HUGE_PAGE_SIZE));
        debug_assert!(mem::size_of::<Self>() <= C::LARGE_PAGE_SIZE.value());

        //  Safety:
        //  -   `at` is assumed to be sufficiently sized.
        //  -   `at` is assumed to be sufficiently aligned.
        #[allow(clippy::cast_ptr_alignment)]
        let huge_page = at as *mut Self;

        ptr::write(huge_page, HugePage::new::<C>(owner));

        //  Enforce memory ordering, later Acquire need to see those 0s and 1s.
        atomic::fence(atomic::Ordering::Release);

        huge_page
    }

    /// Obtain the huge page associated to a given allocation.
    ///
    /// #   Safety
    ///
    /// -   Assumes that the pointer is pointing strictly _inside_ a HugePage.
    pub(crate) unsafe fn from_raw<C>(ptr: *mut u8) -> *mut HugePage
        where
            C: Configuration,
    {
        debug_assert!(!utils::is_sufficiently_aligned_for(ptr, C::HUGE_PAGE_SIZE));

        let address = ptr as usize;
        let huge_page = C::HUGE_PAGE_SIZE.round_down(address);

        huge_page as *mut HugePage
    }

    /// Allocate one or more LargePages from this page, if any.
    ///
    /// Returns a null pointer if the allocation cannot be fulfilled.
    pub(crate) unsafe fn allocate(&self, layout: Layout) -> *mut u8 {
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
            self.address().add(index.value() * large_page_size)
        } else {
            ptr::null_mut()
        }
    }

    /// Deallocates one or multiple pages from this page.
    ///
    /// #   Safety
    ///
    /// -   Assumes that the pointer is pointing to a `LargePage` inside _this_ `HugePage`.
    /// -   Assumes that the pointed page is no longer in use.
    pub(crate) unsafe fn deallocate(&self, ptr: *mut u8) {
        debug_assert!(utils::is_sufficiently_aligned_for(ptr, self.common.page_size));

        let index = (ptr as usize - self.address() as usize) / self.common.page_size;
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

//  Foreign data. Accessible both from the local thread and foreign threads, at the cost of synchronization.
#[repr(align(128))]
struct Foreign {
    //  Bitmap of pages.
    pages: PageTokens,
    //  Sizes of allocations.
    sizes: PageSizes,
    //  Actual number of available pages.
    number_pages: NumberPages,
}

impl Foreign {
    /// Creates a new instance of `Foreign`.
    fn new(number_pages: NumberPages) -> Self {
        let pages = PageTokens::new(number_pages);
        let sizes = PageSizes::default();

        Self { pages, sizes, number_pages, }
    }

    /// Allocates `n` consecutive pages, returns their index.
    ///
    /// The index returned is a multiple of `align_pages`.
    ///
    /// Returns 0 if no allocation could be made.
    unsafe fn allocate(&self, number_pages: NumberPages, align_pages: PowerOf2) -> Option<PageIndex> {
        if number_pages.0 == 1 {
            self.fast_allocate()
        } else {
            self.flexible_allocate(number_pages, align_pages)
        }
    }

    /// Deallocates all cells allocated at the given index.
    unsafe fn deallocate(&self, index: PageIndex) {
        debug_assert!(index.value() > 0);
        debug_assert!(index.value() <= self.number_pages.0);

        //  Safety:
        //  -   `index` is assumed to be within bounds.
        let number_pages = self.sizes.get(index);

        if number_pages.0 == 1 {
            self.fast_deallocate(index);
        } else {
            self.flexible_deallocate(index, number_pages);
        }
    }

    //  Internal: fast-allocate a single page.
    fn fast_allocate(&self) -> Option<PageIndex> { self.pages.fast_allocate() }

    //  Internal: allocate multiple pages aligned on `align_pages`, if possible.
    fn flexible_allocate(&self, number_pages: NumberPages, align_pages: PowerOf2) -> Option<PageIndex> {
        let index = self.pages.flexible_allocate(number_pages, align_pages);

        if let Some(index) = index {
            //  Safety:
            //  -   `index` is within bounds.
            unsafe { self.sizes.set(index, number_pages) };
        }

        index
    }

    //  Internal.
    unsafe fn fast_deallocate(&self, index: PageIndex) { self.pages.fast_deallocate(index) }

    //  Internal.
    unsafe fn flexible_deallocate(&self, index: PageIndex, number_pages: NumberPages) {
        self.pages.flexible_deallocate(index, number_pages);
        self.sizes.unset(index, number_pages);
    }
}

//  Page Tokens.
//
//  A bitmap of which Large Pages are available, and which are not, where 0 means available and 1 occupied.
//
//  The first bit is always 1, as the first Large Page is always occupied by the Huge Page header itself.
struct PageTokens([BitMask; 8]);

impl PageTokens {
    /// Initialize the tokens based on the number of pages.
    fn new(number_pages: NumberPages) -> Self {
        let result = PageTokens(Default::default());

        debug_assert!(number_pages.0 < result.capacity());

        //  The first Large Page is always reserved:
        //  -   for the header of the Huge Page.
        //  -   and optionally to store data in the remaining buffer area.
        result.0[0].initialize(1);

        //  The trailing pages are also reserved, based on how many actual pages are available.
        let mut trailing_ones = result.capacity() - 1 - number_pages.0;
        let mut index = result.0.len() - 1;

        while trailing_ones >= 64 {
            result.0[index].initialize(u64::MAX);
            trailing_ones -= 64;
            index -= 1;
        }

        if trailing_ones > 0 {
            let zeroes = {
                let shift = 64 - trailing_ones;
                (1u64 << shift) - 1
            };

            let ones = u64::MAX - zeroes + if index > 0 { 0 } else { 1 };
            result.0[index].initialize(ones);
        }

        result
    }

    /// Allocates 1 Large Page, returns its index or none if it could not allocate.
    fn fast_allocate(&self) -> Option<PageIndex> {
        for (outer, bits) in self.0.iter().enumerate() {
            if let Some(inner) = bits.claim_single() {
                return PageIndex::new(outer * BitMask::CAPACITY + inner);
            }
        }

        None
    }

    /// Allocates multiple Large Pages, return the index of the first page, or none if it could not allocate.
    ///
    /// The index returned is a multiple of `align_pages`.
    fn flexible_allocate(&self, number_pages: NumberPages, align_pages: PowerOf2) -> Option<PageIndex> {
        const OUTER_1: &[usize] = &[7, 6, 5, 4, 3, 2];
        const OUTER_2: &[usize] = &[7, 5, 3];
        const OUTER_4: &[usize] = &[7];
        const OUTER: &[&[usize]] = &[&[], OUTER_1, OUTER_2, &[], OUTER_4];

        //  Safety:
        //  -   `align_pages / BitMask::CAPACITY` is either 0 or a power of 2.
        //  -   1 is a power of 2.
        //  -   Hence the maximum is a power of 2.
        let align_outer = align_pages / BitMask::CAPACITY;
        debug_assert!(align_outer == 0 || self.0.len() % align_outer == 0);
        debug_assert!(self.0.len() == 8, "{} != 8 => review `match`!", self.0.len());

        //  Do a single pass over the bits, attempting to find `number` consecutive ones.
        //
        //  Iterate in reverse order to avoid interfering with the fast single-page allocation path.
        let mut outer = self.0.len() as isize - 1;

        match align_outer {
            0 => while outer >= 0 {
                //  Safety:
                //  -   `outer` is within bounds.
                let index = unsafe { self.flexible_allocate_backward_from(outer as usize, number_pages, align_pages) };

                outer = match index {
                    Ok(index) => return Some(index),
                    Err(next) => next,
                };
            },
            _ => for outer in unsafe { *OUTER.get_unchecked(align_outer) } {
                //  Safety:
                //  -   `outer` is within bounds.
                let index = unsafe { self.flexible_allocate_backward_from(*outer, number_pages, BitMask::CAPACITY) };

                if let Ok(index) = index {
                    return Some(index);
                }
            },
        }

        None
    }

    /// Deallocates the large page at the specified `index`.
    ///
    /// #   Safety
    ///
    /// -   Assumes that `index` is within bounds.
    unsafe fn fast_deallocate(&self, index: PageIndex) {
        let (outer, inner) = (index.value() / BitMask::capacity(), index.value() % BitMask::capacity());
        debug_assert!(outer < self.0.len());

        self.0.get_unchecked(outer).release_single(inner);
    }

    /// Deallocates `number_pages` Large Pages starting from `index`.
    ///
    /// #   Safety
    ///
    /// -   Assumes that `index` is within bounds.
    /// -   Assumes that `index + number_pages` is within bounds.
    unsafe fn flexible_deallocate(&self, index: PageIndex, number_pages: NumberPages) {
        let (outer, inner) = (index.value() / BitMask::capacity(), index.value() % BitMask::capacity());
        debug_assert!(outer < self.0.len());

        let (head_bits, middle_atomics, tail_bits) = Self::split(inner, number_pages.0);
        debug_assert!(outer + middle_atomics < self.0.len());
        debug_assert!(tail_bits == 0 || outer + middle_atomics + 1 < self.0.len());

        self.0.get_unchecked(outer).release_multiple(inner, head_bits);

        for n in 0..middle_atomics {
            self.0.get_unchecked(outer + n + 1).release_multiple(0, BitMask::capacity());
        }

        if tail_bits != 0 {
            self.0.get_unchecked(outer + middle_atomics + 1).release_multiple(0, tail_bits);
        }
    }

    //  Internal: Returns the number of bits.
    fn capacity(&self) -> usize { self.0.len() * BitMask::capacity() }

    //  Internal: Allocates number_pages, starting from outer.
    //
    //  In case of failure, all internally claimed pages are released and the index of the next outer index to try is
    //  returned.
    //
    //  #   Safety
    //
    //  -   `outer` is within bounds.
    unsafe fn flexible_allocate_backward_from(&self, outer: usize, number_pages: NumberPages, align_pages: PowerOf2)
        -> Result<PageIndex, isize>
    {
        debug_assert!(number_pages.0 % align_pages == 0, "{} % {} != 0", number_pages.0, align_pages.value());

        //  The maximum capacity, accounting for the fact that `self.0` has 1 always reserved page.
        let maximum_capacity = outer * BitMask::capacity() + BitMask::capacity() - 1;

        let number_pages = number_pages.0;

        //  If the number of pages to allocate is greater than the maximum capacity, abandon.
        if number_pages > maximum_capacity {
            return Err(-1);
        }

        let (inner, tail) =
            match self.0.get_unchecked(outer).claim_multiple(cmp::min(number_pages, BitMask::capacity()), align_pages) {
                Some(tuple) => tuple,
                None => return Err(outer as isize - 1),
            };

        if tail == number_pages {
            return PageIndex::new(outer * BitMask::capacity() + inner).ok_or(-1);
        }

        debug_assert!(inner == 0, "{} != 0", inner);
        debug_assert!(tail % align_pages == 0, "{} % {} != 0", tail, align_pages.value());

        //  If outer is 0, it is not possible to go further back.
        if outer == 0 {
            self.flexible_allocate_rewind(outer, outer, tail);
            return Err(-1);
        }

        let remaining_capacity = maximum_capacity - BitMask::capacity();
        let remaining_pages = number_pages - tail;
        debug_assert!(remaining_pages % align_pages == 0, "{} % {} != 0", remaining_pages, align_pages.value());

        //  There may have been capacity if the BitMask at `outer` could allocate more, but it failed to.
        if remaining_pages > remaining_capacity {
            self.flexible_allocate_rewind(outer, outer, tail);
            return Err(-1);
        }

        let (head, middle) = (remaining_pages % BitMask::capacity(), remaining_pages / BitMask::capacity());
        debug_assert!(outer > middle);
        debug_assert!(head % align_pages == 0, "{} % {} != 0", head, align_pages.value());

        let middle_outer = outer - middle;
        debug_assert!(middle_outer > 0);

        for i in (middle_outer..outer).rev() {
            if !self.0.get_unchecked(i).claim_at(0, BitMask::capacity()) {
                //  The current BitMask `i` was released by claim_at, so only the other BitMasks need releasing.
                self.flexible_allocate_rewind(i + 1, outer, tail);
                return Err(i as isize - 1);
            }
        }

        if head == 0 {
            return PageIndex::new(middle_outer * BitMask::capacity()).ok_or(-1);
        }

        let head_outer = middle_outer - 1;

        if !self.0.get_unchecked(head_outer).claim_at(BitMask::capacity() - head, head) {
            //  The current BitMask `middle_outer - 1` was released by claim_at, so only the other BitMasks need
            //  releasing.
            self.flexible_allocate_rewind(middle_outer, outer, tail);
            return Err(head_outer as isize);
        }

        PageIndex::new(head_outer * BitMask::capacity() + BitMask::capacity() - head).ok_or(-1)
    }

    //  Internal: Releases the tentatively claimed pages.
    //
    //  -   [from, to): Completely claimed BitMask.
    //  -   to: first "tail" bits are claimed.
    unsafe fn flexible_allocate_rewind(&self, from: usize, to: usize, tail: usize) {
        debug_assert!(from <= to);
        debug_assert!(to < self.0.len());

        for i in from..to {
            self.0.get_unchecked(i).release_multiple(0, BitMask::capacity());
        }

        self.0.get_unchecked(to).release_multiple(0, tail);
    }

    //  Internal: Returns the number of head bits, middle atomics, tail bits.
    fn split(inner: usize, number_pages: usize) -> (usize, usize, usize) {
        let head_bits = BitMask::capacity() - inner;

        if head_bits >= number_pages {
            return (number_pages, 0, 0);
        }

        let number_pages = number_pages - head_bits;
        (head_bits, number_pages / BitMask::CAPACITY, number_pages % BitMask::CAPACITY)
    }
}

//  Page Sizes.
//
//  For a given allocation spanning pages [M, N), the size of the allocation is N - M, which is registered at index
//  M in the array below.
//
//  For performance reasons, the size is represented as with an implied +1, so that a value of 0 means a size of 1.
//
//  This has two benefits:
//  -   The fast case of single-page allocation need not register the size, as the array is initialized with 0s,
//      and deallocation restores 0.
//  -   Very large allocation sizes of 257 or more can be represented with only 2 u8s: 256 + 255 == 511.
struct PageSizes([atomic::AtomicU8; 512]);

impl PageSizes {
    /// Returns the number of pages from a particular index.
    ///
    /// #   Safety
    ///
    /// -   Assumes that `index` is within bounds.
    unsafe fn get(&self, index: PageIndex) -> NumberPages {
        let index = index.value();
        debug_assert!(index < self.0.len());

        let number_pages = self.0.get_unchecked(index).load(atomic::Ordering::Acquire) as usize;

        if number_pages == 255 {
            debug_assert!(index + 1 < self.0.len());
            //  No implicit +1 on overflow size.
            NumberPages(256 + self.0.get_unchecked(index + 1).load(atomic::Ordering::Acquire) as usize)
        } else {
            //  Implicit +1
            NumberPages(number_pages + 1)
        }
    }

    /// Sets the number of pages at a particular index.
    ///
    /// #   Safety
    ///
    /// -   Assumes that `index` is within bounds.
    /// -   Assumes that `number_pages` is less than or equal to 511.
    unsafe fn set(&self, index: PageIndex, number_pages: NumberPages) {
        debug_assert!(number_pages.0 >= 1 && number_pages.0 <= (u8::MAX as usize) * 2 + 1,
            "index: {}, number_pages: {}", index.value(), number_pages.0);

        let index = index.value();
        debug_assert!(index < self.0.len());

        if number_pages.0 <= 256 {
            let number_pages = (number_pages.0 - 1) as u8;
            self.0.get_unchecked(index).store(number_pages, atomic::Ordering::Release);
        } else {
            let overflow = number_pages.0 - 256;
            debug_assert!(overflow <= (u8::MAX as usize));

            self.0.get_unchecked(index).store(255, atomic::Ordering::Release);
            self.0.get_unchecked(index + 1).store(overflow as u8, atomic::Ordering::Release);
        }
    }

    /// Unsets the number of pages at a particular index.
    ///
    /// #   Safety
    ///
    /// -   Assumes that `index` is within bounds.
    /// -   Assumes that `number_pages` is less than or equal to 511.
    unsafe fn unset(&self, index: PageIndex, number_pages: NumberPages) {
        let index = index.value();
        debug_assert!(index < self.0.len());

        self.0.get_unchecked(index).store(0, atomic::Ordering::Release);

        if number_pages.0 > 256 {
            debug_assert!(self.0[index + 1].load(atomic::Ordering::Acquire) > 0);

            self.0.get_unchecked(index + 1).store(0, atomic::Ordering::Release);
        }
    }
}

impl Default for PageSizes {
    fn default() -> Self { unsafe { mem::zeroed() } }
}

struct BitMask(atomic::AtomicU64);

impl BitMask {
    //  Safety:
    //  -   64 is a power of 2.
    const CAPACITY: PowerOf2 = unsafe { PowerOf2::new_unchecked(64) };

    /// Initializes the BitMask with the given mask.
    fn initialize(&self, mask: u64) { self.0.store(mask, atomic::Ordering::Relaxed); }

    /// Claims a 0 bit, returns its index or None if all bits are claimed.
    fn claim_single(&self) -> Option<usize> {
        loop {
            //  Locate first 0 bit.
            let current_mask = self.0.load(atomic::Ordering::Acquire);
            let candidate = (!current_mask).trailing_zeros() as usize;

            //  All bits are ones, move on.
            if candidate == Self::capacity() { return None; }

            let candidate_mask = 1u64 << candidate;
            let before = self.0.fetch_or(candidate_mask, atomic::Ordering::AcqRel);

            //  This thread claimed the bit first, victory!
            if before & candidate_mask == 0 {
                return Some(candidate);
            }
        }
    }

    /// Claims up to `number` bits, returns the index of the lowest claimed and the number of bits claimed, or None.
    ///
    /// This functions scans the bits from highest to lowest.
    ///
    /// If the number of returned bits is less than `number`, then the index is 0, allowing extending the allocation
    /// to the correct number by claiming bits from the previous BitMask.
    fn claim_multiple(&self, number: usize, align: PowerOf2) -> Option<(usize, usize)> {
        //  Do a single pass over the bits, attempting to find `number` consecutive ones.
        let mut progress_mask = 0;

        loop {
            //  Locate last 0 bit.
            let current_mask = self.0.load(atomic::Ordering::Acquire) | progress_mask;

            //  Potential number of available bits.
            let potential = Self::capacity() - (!current_mask).leading_zeros() as usize;

            //  Not enough potential bits available, try to claim as many low-bits as possible.
            if potential < number { break; }

            //  Highest lower index which allows `number` bits.
            let candidate = align.round_down(potential - number);

            let candidate_mask = Self::low(number) << candidate;

            if self.claim(candidate_mask) {
                return Some((candidate, number));
            }

            //  Force searches to discard previously checked bits.
            //
            //  Otherwise, it could enter an infinite loop attempting to claim n bits when only 1 is available.
            progress_mask = Self::high(Self::capacity() - potential + 1);
        }

        //  Claim as many low-bits as possible, as long as they are aligned.
        {
            let current_mask = self.0.load(atomic::Ordering::Acquire);

            let potential = align.round_down(current_mask.trailing_zeros() as usize);

            if potential == 0 {
                return None;
            }

            let candidate_mask = Self::low(potential);

            if self.claim(candidate_mask) {
                Some((0, potential))
            } else {
                None
            }
        }
    }

    /// Claims the exact number of bits at the exact index.
    ///
    /// Returns true on success, false on failure.
    fn claim_at(&self, inner: usize, number: usize) -> bool {
        let mask = Self::low(number) << inner;

        self.claim(mask)
    }

    /// Releases bit at given index.
    fn release_single(&self, inner: usize) {
        debug_assert!(inner < Self::capacity());

        let inner_mask = 1u64 << inner;

        self.release(inner_mask);
    }

    /// Releases `number` bits starting at the given index.
    fn release_multiple(&self, inner: usize, number: usize) {
        debug_assert!(inner + number <= Self::capacity());

        let mask = BitMask::low(number) << inner;

        self.release(mask);
    }

    //  Internal: Returns the capacity as usize.
    fn capacity() -> usize { Self::CAPACITY.value() }

    //  Internal: Claims the bits, returns true on success, false on failure.
    //
    //  On failure, unclaims bits that were erroneously claimed.
    fn claim(&self, mask: u64) -> bool {
        let before = self.0.fetch_or(mask, atomic::Ordering::AcqRel);

        //  Success!
        if before & mask == 0 {
            return true;
        } 

        //  Not all bits were claimed before, release the ones that were not.
        //
        //  Example:
        //  -   mask:           0111.
        //  -   before:         1010.
        //  -   before & mask:  0010.
        //  -   !before & mask: 0101.
        if before & mask != mask {
            self.release(!before & mask)
        }

        false
    }

    //  Internal: Releases the bits.
    fn release(&self, mask: u64) {
        let _before = self.0.fetch_and(!mask, atomic::Ordering::AcqRel);
        debug_assert!(_before & mask == mask);
    }

    //  Internal: Computes a mask with the `number` high bits set, and all others unset.
    fn high(number: usize) -> u64 {
        debug_assert!(number <= Self::capacity());

        !Self::low(Self::capacity() - number)
    }

    //  Internal: Computes a mask with the `number` low bits set, and all others unset.
    fn low(number: usize) -> u64 {
        debug_assert!(number <= Self::capacity());

        if number == 64 {
            u64::MAX
        } else {
            (1u64 << number) - 1
        }
    }
}

#[cfg(test)]
impl Clone for BitMask {
    fn clone(&self) -> Self {
        let result = BitMask::default();
        result.initialize(self.0.load(atomic::Ordering::Relaxed));
        result
    }
}

impl Default for BitMask {
    fn default() -> Self { Self(atomic::AtomicU64::new(0)) }
}

#[derive(Clone, Copy, Default)]
struct NumberPages(usize);

#[derive(Clone, Copy)]
struct PageIndex(num::NonZeroUsize);

impl PageIndex {
    /// Creates an instance of PageIndex, or None if `index` is zero.
    fn new(index: usize) -> Option<PageIndex> { num::NonZeroUsize::new(index).map(PageIndex) }

    /// Creates an instance of PageIndex.
    ///
    /// #   Safety
    ///
    /// -   Assumes that `index` is non-zero.
    unsafe fn new_unchecked(index: usize) -> PageIndex {
        debug_assert!(index > 0);

        PageIndex(num::NonZeroUsize::new_unchecked(index))
    }

    /// Returns the inner value.
    fn value(&self) -> usize { self.0.get() }
}

#[cfg(test)]
mod tests {

use super::*;

#[test]
fn page_index_new() {
    fn new(index: usize) -> usize { PageIndex::new(index).unwrap().value() }

    assert!(PageIndex::new(0).is_none());

    assert_eq!(1, new(1));
    assert_eq!(3, new(3));
    assert_eq!(42, new(42));
    assert_eq!(99, new(99));
    assert_eq!(1023, new(1023));
}

#[test]
fn page_index_new_unchecked() {
    fn new(index: usize) -> usize {
        assert_ne!(0, index);

        //  Safety:
        //  -   `index` is not 0.
        unsafe { PageIndex::new_unchecked(index) }.value()
    }

    assert_eq!(1, new(1));
    assert_eq!(3, new(3));
    assert_eq!(42, new(42));
    assert_eq!(99, new(99));
    assert_eq!(1023, new(1023));
}

fn load_bitmask(bitmask: &BitMask) -> u64 { bitmask.0.load(atomic::Ordering::Relaxed) }

#[test]
fn bitmask_initialize() {
    fn initialize(mask: u64) -> u64 {
        let bitmask = BitMask::default();
        bitmask.initialize(mask);
        load_bitmask(&bitmask)
    }

    assert_eq!(0, initialize(0));
    assert_eq!(3, initialize(3));
    assert_eq!(42, initialize(42));
    assert_eq!(u64::MAX, initialize(u64::MAX));
}

#[test]
fn bitmask_claim_single() {
    //  Initializes bitmask with `mask`, call claim_single, return single + new state of bitmask
    fn claim_single(mask: u64) -> (Option<usize>, u64) {
        let bitmask = BitMask::default();
        bitmask.initialize(mask);

        let claimed = bitmask.claim_single();

        (claimed, load_bitmask(&bitmask))
    }

    assert_eq!((None, u64::MAX), claim_single(u64::MAX));

    //  Check the next is claimed with a contigous pattern starting from beginning
    for i in 0..63 {
        let before = (1u64 << i) - 1;
        let after = (1u64 << (i + 1)) - 1;

        assert_eq!((Some(i), after), claim_single(before));
    }

    //  Check the one available bit is claimed if a single is available.
    for i in 0..64 {
        let mask = u64::MAX - (1u64 << i);

        assert_eq!((Some(i), u64::MAX), claim_single(mask));
    }
}

#[test]
fn bitmask_claim_at() {
    //  Initializes bitmask with `mask`, call claim_at with `index` and `number`, return result + new state of bitmask
    fn claim_at(mask: u64, index: usize, number: usize) -> (bool, u64) {
        let bitmask = BitMask::default();
        bitmask.initialize(mask);

        let result = bitmask.claim_at(index, number);

        (result, load_bitmask(&bitmask))
    }

    for i in 0..64 {
        assert_eq!((true, 1u64 << i), claim_at(0, i, 1));
    }

    for i in 0..64 {
        assert_eq!((false, 1u64 << i), claim_at(1u64 << i, i, 1));
    }

    assert_eq!((true, u64::MAX), claim_at(0, 0, 64));
    assert_eq!((false, 1), claim_at(1, 0, 64));
}

//  Initializes bitmask with `mask`, call claim_multiple with `number`, return result + new state of bitmask
fn claim_multiple(mask: u64, number: usize) -> (Option<usize>, Option<usize>, u64) {
    let bitmask = BitMask::default();
    bitmask.initialize(mask);

    if let Some(claimed) = bitmask.claim_multiple(number, PowerOf2::ONE) {
        (Some(claimed.0), Some(claimed.1), load_bitmask(&bitmask))
    } else {
        (None, None, load_bitmask(&bitmask))
    }
}

#[test]
fn bitmask_claim_multiple_none() {
    assert_eq!((None, None, u64::MAX), claim_multiple(u64::MAX, 1));

    for i in 1..63 {
        let mask = BitMask::low(i);
        assert_eq!((None, None, mask), claim_multiple(mask, 65 - i));
    }
}

#[test]
fn bitmask_claim_multiple_complete() {
    //  Claim multiple always starts from the _high_ bits.
    for number in 1..=64 {
        let after = BitMask::low(number) << (64 - number);
        assert_eq!((Some(64 - number), Some(number), after), claim_multiple(0, number));
    }
}

#[test]
fn bitmask_claim_multiple_partial() {
    //  Claim multiple should be able to claim N (< number) low bits.
    for number in 1..=63 {
        let before = BitMask::high(number);
        assert_eq!((Some(0), Some(64 - number), u64::MAX), claim_multiple(before, 64));
    }
}

#[test]
fn bitmask_claim_multiple_pockmarked() {
    let mask = {
        BitMask::low(1) << 63 |
        // 7 0-bits
        BitMask::low(8) << 48 |
        // 8 0-bits
        BitMask::low(8) << 32 |
        // 9 0-bits
        BitMask::low(7) << 16 |
        // 10 0-bits
        BitMask::low(5) << 1
        // 1 0-bits
    };

    let after = mask | BitMask::low(7) << 56;
    assert_eq!((Some(56), Some(7), after), claim_multiple(mask, 7));

    let after = mask | BitMask::low(8) << 40;
    assert_eq!((Some(40), Some(8), after), claim_multiple(mask, 8));

    let after = mask | BitMask::low(9) << 23;
    assert_eq!((Some(23), Some(9), after), claim_multiple(mask, 9));

    let after = mask | BitMask::low(10) << 6;
    assert_eq!((Some(6), Some(10), after), claim_multiple(mask, 10));

    let after = mask | BitMask::low(1);
    assert_eq!((Some(0), Some(1), after), claim_multiple(mask, 11));
}

//  Initializes bitmask with `mask`, call claim_multiple with `number`, return result + new state of bitmask
fn claim_multiple_aligned(mask: u64, number: usize, align: usize) -> (Option<usize>, Option<usize>, u64) {
    let bitmask = BitMask::default();
    bitmask.initialize(mask);

    if let Some(claimed) = bitmask.claim_multiple(number, PowerOf2::new(align).expect("Power of 2")) {
        (Some(claimed.0), Some(claimed.1), load_bitmask(&bitmask))
    } else {
        (None, None, load_bitmask(&bitmask))
    }
}

#[test]
fn bitmask_claim_multiple_aligned_none() {
    let tests = [
        ( 2, 0b10011001_10011001_10011001_10011001_10011001_10011001_10011001_10011001u64),
        ( 4, 0b10000001_10000001_10000001_10000001_10000001_10000001_10000001_10000001u64),
        ( 8, 0b10000000_00000001_10000000_00000001_10000000_00000001_10000000_00000001u64),
        (16, 0b10000000_00000000_00000000_00000001_10000000_00000000_00000000_00000001u64),
        (32, 0b10000000_00000000_00000000_00000000_00000000_00000000_00000000_00000001u64),
        (64, BitMask::high(1)),
    ];

    for &(number, mask) in &tests {

        assert_eq!((None, None, mask), claim_multiple_aligned(mask, number, number),
            "mask: {:b}, number: {}", mask, number);
    }
}

#[test]
fn bitmask_claim_multiple_aligned_pockmarked() {
    let mask = {
        BitMask::low(1) << 63 |
        //  2 0-bits
        BitMask::low(1) << 60 |
        BitMask::low(1) << 59 |
        //  4 0-bits
        BitMask::low(1) << 54 |
        //  8 0-bits
        BitMask::low(1) << 45 |
        //  16 0-bits
        BitMask::low(1) << 28
        //  29 0-bits
    };

    let after = mask | BitMask::low(2) << 56;
    assert_eq!((Some(56), Some(2), after), claim_multiple_aligned(mask, 2, 2));

    let after = mask | BitMask::low(4) << 48;
    assert_eq!((Some(48), Some(4), after), claim_multiple_aligned(mask, 4, 4));

    let after = mask | BitMask::low(8) << 32;
    assert_eq!((Some(32), Some(8), after), claim_multiple_aligned(mask, 8, 8));

    let after = mask | BitMask::low(16);
    assert_eq!((Some(0), Some(16), after), claim_multiple_aligned(mask, 16, 16));
}

#[test]
fn bitmask_release_single() {
    //  Initializes bitmask with `mask`, call release_single with `index`, return new state of bitmask
    fn release_single(mask: u64, index: usize) -> u64 {
        let bitmask = BitMask::default();
        bitmask.initialize(mask);

        bitmask.release_single(index);

        load_bitmask(&bitmask)
    }

    for i in 0..64 {
        assert_eq!(0, release_single(1u64 << i, i));
    }

    for i in 0..64 {
        assert_eq!(u64::MAX - (1u64 << i), release_single(u64::MAX, i));
    }
}

#[test]
fn bitmask_release_multiple() {
    //  Initializes bitmask with `mask`, call release_multiple with `index` and `number`, return new state of bitmask
    fn release_multiple(mask: u64, index: usize, number: usize) -> u64 {
        let bitmask = BitMask::default();
        bitmask.initialize(mask);

        bitmask.release_multiple(index, number);

        load_bitmask(&bitmask)
    }

    assert_eq!(0, release_multiple(u64::MAX, 0, 64));

    for i in 0..64 {
        assert_eq!(0, release_multiple(1u64 << i, i, 1));
    }

    for i in 0..64 {
        assert_eq!(u64::MAX - (1u64 << i), release_multiple(u64::MAX, i, 1));
    }
}

#[test]
fn page_sizes_get_set() {
    //  Creates a PageSizes, call `set` then with `index` and `number`, call `get` with `index`, return its result.
    fn set_get(index: usize, number: usize) -> usize {
        let index = PageIndex::new(index).unwrap();

        let page_sizes = PageSizes::default();

        unsafe { page_sizes.set(index, NumberPages(number)) };

        unsafe { page_sizes.get(index).0 }
    }

    for index in 1..512 {
        assert_eq!(1, set_get(index, 1));
    }

    for number in 1..512 {
        assert_eq!(number, set_get(1, number));
    }
}

#[test]
fn page_sizes_unset() {
    //  Creates a PageSizes, call `set` then `unset` with `index` and `number`, call `get` with `index`, return its result.
    fn unset(index: usize, number: usize) -> usize {
        let index = PageIndex::new(index).unwrap();

        let page_sizes = PageSizes::default();

        unsafe { page_sizes.set(index, NumberPages(number)) };
        unsafe { page_sizes.unset(index, NumberPages(number)) };

        unsafe { page_sizes.get(index).0 }
    }

    for index in 1..512 {
        assert_eq!(1, unset(index, 1));
    }

    for number in 1..512 {
        assert_eq!(1, unset(1, number));
    }
}

type RawPageTokens = [u64; 8];

fn load_page_tokens(page_tokens: &PageTokens) -> RawPageTokens {
    assert_eq!(512, page_tokens.capacity());

    [
        load_bitmask(&page_tokens.0[0]),
        load_bitmask(&page_tokens.0[1]),
        load_bitmask(&page_tokens.0[2]),
        load_bitmask(&page_tokens.0[3]),
        load_bitmask(&page_tokens.0[4]),
        load_bitmask(&page_tokens.0[5]),
        load_bitmask(&page_tokens.0[6]),
        load_bitmask(&page_tokens.0[7]),
    ] 
}

fn create_page_tokens(tokens: RawPageTokens) -> PageTokens {
    assert_eq!(1, tokens[0] & 1);

    let page_tokens = PageTokens::new(NumberPages(511));

    for i in 0..tokens.len() {
        page_tokens.0[i].initialize(tokens[i]);
    }

    page_tokens
}

#[track_caller]
fn check_tokens(tokens: RawPageTokens, expected: RawPageTokens) {
    assert_eq!(tokens.len(), expected.len());

    for (index, (expected, token)) in expected.iter().zip(tokens.iter()).enumerate() {
        assert_eq!(expected, token,
            "Index {} - expected {:b} got {:b}", index, expected, token);
    }
}

#[test]
fn page_tokens_new() {
    fn new(number_pages: usize) -> RawPageTokens {
        let page_tokens = PageTokens::new(NumberPages(number_pages));
        load_page_tokens(&page_tokens)
    }

    const FULL: u64 = u64::MAX;

    check_tokens(new(511), [BitMask::low(1), 0, 0, 0, 0, 0, 0, 0]);

    for i in 1..=64 {
        check_tokens(new(511 - i), [BitMask::low(1), 0, 0, 0, 0, 0, 0, BitMask::high(i)]);
    }

    for i in 1..=64 {
        check_tokens(new(511 - 64 - i), [BitMask::low(1), 0, 0, 0, 0, 0, BitMask::high(i), FULL]);
    }

    for i in 1..=64 {
        check_tokens(new(127 - i), [BitMask::low(1), BitMask::high(i), FULL, FULL, FULL, FULL, FULL, FULL]);
    }

    for i in 1..=62 {
        check_tokens(new(63 - i), [BitMask::low(1) + BitMask::high(i), FULL, FULL, FULL, FULL, FULL, FULL, FULL]);
    }
}

#[test]
fn page_tokens_fast_allocate() {
    fn fast_allocate(initial: RawPageTokens) -> Option<usize> {
        let tokens = create_page_tokens(initial);

        tokens.fast_allocate().map(|x| x.0.get())
    }

    let full = u64::MAX;

    assert_eq!(Some(1), fast_allocate([1, 0, 0, 0, 0, 0, 0, 0]));
    assert_eq!(None, fast_allocate([full, full, full, full, full, full, full, full]));

    for i in 0..64 {
        assert_eq!(Some(64 + i), fast_allocate([full, BitMask::low(i), full, full, full, full, full, full]));
    }

    for i in 0..64 {
        assert_eq!(Some(64 * 7 + i), fast_allocate([full, full, full, full, full, full, full, BitMask::low(i)]));
    }
}

#[test]
fn page_tokens_fast_deallocate() {
    fn fast_deallocate(index: usize, initial: RawPageTokens) -> RawPageTokens {
        let page_tokens = create_page_tokens(initial);
        unsafe { page_tokens.fast_deallocate(PageIndex::new(index).unwrap()) };
        load_page_tokens(&page_tokens)
    }

    let full = u64::MAX;

    assert_eq!(
        [1, 0, 0, 0, 0, 0, 0, 0],
        fast_deallocate(1, [3, 0, 0, 0, 0, 0, 0, 0])
    );

    assert_eq!(
        [1, 0, 0, 0, 0, 0, 0, 0],
        fast_deallocate(64 * 4 + 5, [1, 0, 0, 0, 1u64 << 5, 0, 0, 0])
    );

    assert_eq!(
        [1 + BitMask::high(62), 0, 0, 0, 0, 0, 0, 0],
        fast_deallocate(1, [full, 0, 0, 0, 0, 0, 0, 0])
    );
}

fn flexible_allocate(number: usize, initial: RawPageTokens, expected: RawPageTokens) -> Option<usize> {
    let tokens = create_page_tokens(initial);
    let index = tokens.flexible_allocate(NumberPages(number), PowerOf2::ONE).map(|x| x.0.get());

    check_tokens(load_page_tokens(&tokens), expected);

    index
}

#[test]
fn page_tokens_flexible_allocate_success() {
    let full = u64::MAX;
    let all_empty = [1, 0, 0, 0, 0, 0, 0, 0];
    let all_full = [full, full, full, full, full, full, full, full];

    assert_eq!(Some(510), flexible_allocate(2, all_empty, [1, 0, 0, 0, 0, 0, 0, BitMask::high(2)]));
    assert_eq!(Some(1), flexible_allocate(511, all_empty, all_full));

    for number in 2..=64 {
        let expected = [1, 0, 0, 0, 0, 0, 0, BitMask::high(number)];
        assert_eq!(Some(512 - number), flexible_allocate(number, all_empty, expected));
    }

    for number in 1..=64 {
        let expected = [1, 0, 0, 0, 0, 0, BitMask::high(number), full];
        assert_eq!(Some(512 - 64 - number), flexible_allocate(64 + number, all_empty, expected));
    }
}

#[test]
fn page_tokens_flexible_allocate_failure() {
    let full = u64::MAX;
    let all_full = [full, full, full, full, full, full, full, full];

    assert_eq!(None, flexible_allocate(2, all_full, all_full));
    assert_eq!(None, flexible_allocate(511, all_full, all_full));

    for number in 2..=64 {
        let mask = BitMask::low(65 - number);
        let tokens = [mask, mask, mask, mask, mask, mask, mask, mask];
        assert_eq!(None, flexible_allocate(number, tokens, tokens));
    }

    for number in 1..=64 {
        let mask = BitMask::low(65 - number);
        let tokens = [mask, 0, mask, 0, mask, 0, mask, 0];
        assert_eq!(None, flexible_allocate(64 + number, tokens, tokens));
    }
}

#[test]
fn page_tokens_flexible_allocate_straddling() {
    let full = u64::MAX;

    for number in 2..=64 {
        let initial = [1, 0, 0, 0, 0, 0, 0, BitMask::high(63)];
        let expected = [1, 0, 0, 0, 0, 0, BitMask::high(number - 1), full];
        assert_eq!(Some(512 - 63 - number), flexible_allocate(number, initial, expected));
    }

    for number in 1..=64 {
        let initial = [1, 0, 0, 0, 0, 0, 0, BitMask::high(63)];
        let expected = [1, 0, 0, 0, 0, BitMask::high(number - 1), full, full];
        assert_eq!(Some(512 - 63 - 64 - number), flexible_allocate(64 + number, initial, expected));
    }
}

fn flexible_allocate_aligned(number: usize, alignment: usize, initial: RawPageTokens, expected: RawPageTokens)
    -> Option<usize>
{
    let alignment = PowerOf2::new(alignment).expect("Power of 2");
    let tokens = create_page_tokens(initial);
    let index = tokens.flexible_allocate(NumberPages(number), alignment).map(|x| x.0.get());

    check_tokens(load_page_tokens(&tokens), expected);

    index
}

#[test]
fn page_tokens_flexible_allocate_aligned_small_success() {
    let full = u64::MAX;
    let all_empty = [1, 0, 0, 0, 0, 0, 0, 0];

    assert_eq!(Some(510), flexible_allocate_aligned(2, 2, all_empty, [1, 0, 0, 0, 0, 0, 0, BitMask::high(2)]));
    assert_eq!(Some(512 - 64), flexible_allocate_aligned(64, 64, all_empty, [1, 0, 0, 0, 0, 0, 0, full]));

    let before = [1, 0, 0, 0, 0, 0, 0, BitMask::high(1)];
    let tests = [
        ( 2, [1, 0, 0, 0, 0, 0, 0, BitMask::high(1) + (BitMask::low(2) << 60)]),
        ( 4, [1, 0, 0, 0, 0, 0, 0, BitMask::high(1) + (BitMask::low(4) << 56)]),
        ( 8, [1, 0, 0, 0, 0, 0, 0, BitMask::high(1) + (BitMask::low(8) << 48)]),
        (16, [1, 0, 0, 0, 0, 0, 0, BitMask::high(1) + (BitMask::low(16) << 32)]),
        (32, [1, 0, 0, 0, 0, 0, 0, BitMask::high(1) + BitMask::low(32)]),
        (64, [1, 0, 0, 0, 0, 0, full, BitMask::high(1)]),
    ];

    for &(number, expected) in &tests {
        assert_eq!(Some(512 - number * 2), flexible_allocate_aligned(number, number, before, expected));
    }

    let before = [1, 0, 0, 0, 0, 0, BitMask::high(1), full];
    for number in 1..=31 {
        let number = number * 2;
        let expected = [1, 0, 0, 0, 0, 0, BitMask::high(number + 2) - (BitMask::high(1) >> 1), full];
        assert_eq!(Some(512 - 64 - number - 2), flexible_allocate_aligned(number, 2, before, expected));
    }
}

#[test]
fn page_tokens_flexible_allocate_aligned_small_failure() {
    let full = u64::MAX;
    let all_full = [full, full, full, full, full, full, full, full];

    assert_eq!(None, flexible_allocate_aligned(2, 2, all_full, all_full));
    assert_eq!(None, flexible_allocate_aligned(256, 256, all_full, all_full));
}

#[test]
fn page_tokens_flexible_allocate_aligned_128() {
    let full = u64::MAX;
    let one = BitMask::low(1) << 11;
    let two = BitMask::low(1) << 33;
    let all_empty = [1, 0, 0, 0, 0, 0, 0, 0];

    assert_eq!(Some(384), flexible_allocate_aligned(128, 128, all_empty, [1, 0, 0, 0, 0, 0, full, full]));
    assert_eq!(Some(256), flexible_allocate_aligned(256, 128, all_empty, [1, 0, 0, 0, full, full, full, full]));
    assert_eq!(Some(128), flexible_allocate_aligned(384, 128, all_empty, [1, 0, full, full, full, full, full, full]));

    let before = [1, 0, 0, 0, 0, two, one, 0];
    assert_eq!(Some(128), flexible_allocate_aligned(128, 128, before, [1, 0, full, full, 0, two, one, 0]));

    let before = [1, 0, 0, 0, 0, 0, 0, one];
    assert_eq!(Some(128), flexible_allocate_aligned(256, 128, before, [1, 0, full, full, full, full, 0, one]));
}

#[test]
fn page_tokens_flexible_allocate_aligned_256() {
    let full = u64::MAX;
    let all_empty = [1, 0, 0, 0, 0, 0, 0, 0];

    assert_eq!(Some(256), flexible_allocate_aligned(256, 256, all_empty, [1, 0, 0, 0, full, full, full, full]));

    for outer in 0..=3 {
        for inner in 0..=63 {
            let mut before = all_empty;
            let mut after = [1, 0, 0, 0, full, full, full, full];

            before[outer] |= BitMask::low(1) << inner;
            after[outer] |= BitMask::low(1) << inner;

            assert_eq!(Some(256), flexible_allocate_aligned(256, 256, before, after));
        }
    }

    for outer in 4..=7 {
        for inner in 0..=63 {
            let mut mask = all_empty;
            mask[outer] |= BitMask::low(1) << inner;

            assert_eq!(None, flexible_allocate_aligned(256, 256, mask, mask));
        }
    }
}

#[test]
fn page_tokens_flexible_deallocate() {
    fn flexible_deallocate(index: usize, number: usize, initial: RawPageTokens) -> RawPageTokens {
        let page_tokens = create_page_tokens(initial);
        unsafe { page_tokens.flexible_deallocate(PageIndex::new(index).unwrap(), NumberPages(number)) };

        load_page_tokens(&page_tokens)
    }

    let full = u64::MAX;

    assert_eq!(
        [1, 0, 0, 0, 0, 0, 0, 0],
        flexible_deallocate(510, 2, [1, 0, 0, 0, 0, 0, 0, BitMask::high(2)])
    );

    assert_eq!(
        [1, 0, 0, 0, 0, 0, 0, 0],
        flexible_deallocate(1, 511, [full, full, full, full, full, full, full, full])
    );

    assert_eq!(
        [1, 0, 0, 0, BitMask::low(3) + BitMask::high(40), 0, 0, 0],
        flexible_deallocate(64 * 4 + 3, 21, [1, 0, 0, 0, full, 0, 0, 0])
    );

    assert_eq!(
        [1, 0, 0, BitMask::low(37), BitMask::high(40), 0, 0, 0],
        flexible_deallocate(64 * 3 + 37, 51, [1, 0, 0, full, full, 0, 0, 0])
    );

    assert_eq!(
        [1, 0, BitMask::low(37), 0, 0, BitMask::high(40), 0, 0],
        flexible_deallocate(64 * 2 + 37, 179, [1, 0, full, full, full, full, 0, 0])
    );
}

#[test]
fn foreign_allocate_deallocate_fast() {
    fn allocate_fast(foreign: &Foreign) -> Option<usize> {
        unsafe { foreign.allocate(NumberPages(1), PowerOf2::ONE) }.map(|x| x.0.get())
    }

    let foreign = Foreign::new(NumberPages(511));

    for i in 0..95 {
        assert_eq!(Some(i + 1), allocate_fast(&foreign));
    }

    let reused = PageIndex::new(75).unwrap();

    unsafe { foreign.deallocate(reused) };

    assert_eq!(Some(reused.value()), allocate_fast(&foreign));
}

#[test]
fn foreign_allocate_deallocate_flexible() {
    fn allocate_flexible(foreign: &Foreign, number_pages: usize) -> Option<usize> {
        unsafe { foreign.allocate(NumberPages(number_pages), PowerOf2::ONE) }.map(|x| x.0.get())
    }

    let foreign = Foreign::new(NumberPages(511));

    assert_eq!(Some(64 * 7 + 38), allocate_flexible(&foreign, 26));
    assert_eq!(Some(64 * 7 + 13), allocate_flexible(&foreign, 25));
    assert_eq!(Some(64 * 6 + 51), allocate_flexible(&foreign, 26));
    assert_eq!(Some(64 * 6 + 26), allocate_flexible(&foreign, 25));

    unsafe { foreign.deallocate(PageIndex::new(64 * 6 + 51).unwrap()) };
    unsafe { foreign.deallocate(PageIndex::new(64 * 7 + 13).unwrap()) };

    assert_eq!(Some(64 * 7 + 11), allocate_flexible(&foreign, 27));
    assert_eq!(Some(64 * 6 + 51), allocate_flexible(&foreign, 24));
}

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

    let huge_page_ptr = unsafe { HugePage::initialize::<TestConfiguration>(slice, ptr::null_mut()) };
    assert_eq!(slice.as_mut_ptr(), huge_page_ptr as *mut u8);

    let huge_page = unsafe { &mut *huge_page_ptr };
    assert_eq!(huge_page_ptr as usize, huge_page.address() as usize);

    assert_eq!(ptr::null_mut(), huge_page.owner());

    huge_page.set_owner(owner);
    assert_eq!(owner, huge_page.owner());

    {
        let buffer = huge_page.buffer_mut();

        let start_ptr = huge_page_ptr as *mut u8 as usize;
        let buffer_ptr = buffer.as_mut_ptr() as usize;

        assert_eq!(HUGE_HEADER_SIZE, buffer_ptr - start_ptr);
        assert_eq!(LARGE_PAGE_SIZE - HUGE_HEADER_SIZE - 128, buffer.len());
    }

    let layout = Layout::from_size_align(LARGE_PAGE_SIZE + 1, 1).expect("Proper layout");
    let allocated = unsafe { huge_page.allocate(layout) };
    assert_ne!(ptr::null_mut(), allocated);

    let retrieved = unsafe { HugePage::from_raw::<TestConfiguration>(allocated) };
    assert_eq!(huge_page_ptr, retrieved);

    let layout = Layout::from_size_align(LARGE_PAGE_SIZE * 14, 1).expect("Proper layout");
    let failed = unsafe { huge_page.allocate(layout) };
    assert_eq!(ptr::null_mut(), failed);

    unsafe { huge_page.deallocate(allocated) };
}

}
