use crate::PowerOf2;

use super::{
    NumberPages,
    PageIndex,
    page_sizes::PageSizes,
    page_tokens::PageTokens,
};

//  Foreign data. Accessible both from the local thread and foreign threads, at the cost of synchronization.
#[repr(align(128))]
pub(crate) struct Foreign {
    //  Bitmap of pages.
    pages: PageTokens,
    //  Sizes of allocations.
    sizes: PageSizes,
    //  Actual number of available pages.
    number_pages: NumberPages,
}

impl Foreign {
    /// Creates a new instance of `Foreign`.
    pub(crate) fn new(number_pages: NumberPages) -> Self {
        let pages = PageTokens::new(number_pages);
        let sizes = PageSizes::default();

        Self { pages, sizes, number_pages, }
    }

    /// Allocates `n` consecutive pages, returns their index.
    ///
    /// The index returned is a multiple of `align_pages`.
    ///
    /// Returns 0 if no allocation could be made.
    pub(crate) unsafe fn allocate(&self, number_pages: NumberPages, align_pages: PowerOf2) -> Option<PageIndex> {
        if number_pages.0 == 1 {
            self.fast_allocate()
        } else {
            self.flexible_allocate(number_pages, align_pages)
        }
    }

    /// Deallocates all cells allocated at the given index.
    pub(crate) unsafe fn deallocate(&self, index: PageIndex) {
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

#[cfg(test)]
mod tests {

use super::*;

#[test]
fn foreign_allocate_deallocate_fast() {
    fn allocate_fast(foreign: &Foreign) -> Option<usize> {
        unsafe { foreign.allocate(NumberPages(1), PowerOf2::ONE) }.map(|x| x.value())
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
        unsafe { foreign.allocate(NumberPages(number_pages), PowerOf2::ONE) }.map(|x| x.value())
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

} // mod tests
