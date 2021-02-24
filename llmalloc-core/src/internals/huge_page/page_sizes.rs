//! A mapping of how many pages are allocated.

use core::{
    mem,
    sync::atomic::{AtomicU8, Ordering},
};

use super::{NumberPages, PageIndex};

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
pub(crate) struct PageSizes([AtomicU8; 512]);

impl PageSizes {
    /// Returns the number of pages from a particular index.
    ///
    /// #   Safety
    ///
    /// -   Assumes that `index` is within bounds.
    pub(crate) unsafe fn get(&self, index: PageIndex) -> NumberPages {
        let index = index.value();
        debug_assert!(index < self.0.len());

        let number_pages = self.0.get_unchecked(index).load(Ordering::Acquire) as usize;

        if number_pages == 255 {
            debug_assert!(index + 1 < self.0.len());
            //  No implicit +1 on overflow size.
            NumberPages(256 + self.0.get_unchecked(index + 1).load(Ordering::Acquire) as usize)
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
    pub(crate) unsafe fn set(&self, index: PageIndex, number_pages: NumberPages) {
        debug_assert!(number_pages.0 >= 1 && number_pages.0 <= (u8::MAX as usize) * 2 + 1,
            "index: {}, number_pages: {}", index.value(), number_pages.0);

        let index = index.value();
        debug_assert!(index < self.0.len());

        if number_pages.0 <= 256 {
            let number_pages = (number_pages.0 - 1) as u8;
            self.0.get_unchecked(index).store(number_pages, Ordering::Release);
        } else {
            let overflow = number_pages.0 - 256;
            debug_assert!(overflow <= (u8::MAX as usize));

            self.0.get_unchecked(index).store(255, Ordering::Release);
            self.0.get_unchecked(index + 1).store(overflow as u8, Ordering::Release);
        }
    }

    /// Unsets the number of pages at a particular index.
    ///
    /// #   Safety
    ///
    /// -   Assumes that `index` is within bounds.
    /// -   Assumes that `number_pages` is less than or equal to 511.
    pub(crate) unsafe fn unset(&self, index: PageIndex, number_pages: NumberPages) {
        let index = index.value();
        debug_assert!(index < self.0.len());

        self.0.get_unchecked(index).store(0, Ordering::Release);

        if number_pages.0 > 256 {
            debug_assert!(self.0[index + 1].load(Ordering::Acquire) > 0);

            self.0.get_unchecked(index + 1).store(0, Ordering::Release);
        }
    }
}

impl Default for PageSizes {
    fn default() -> Self { unsafe { mem::zeroed() } }
}

#[cfg(test)]
mod tests {

use super::*;

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

} // mod tests
