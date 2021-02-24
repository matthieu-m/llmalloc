//! The index of a Huge Page.

use core::num::NonZeroUsize;

#[derive(Clone, Copy)]
pub(crate) struct PageIndex(NonZeroUsize);

impl PageIndex {
    /// Creates an instance of PageIndex, or None if `index` is zero.
    pub(crate) fn new(index: usize) -> Option<PageIndex> { NonZeroUsize::new(index).map(PageIndex) }

    /// Creates an instance of PageIndex.
    ///
    /// #   Safety
    ///
    /// -   Assumes that `index` is non-zero.
    pub(crate) unsafe fn new_unchecked(index: usize) -> PageIndex {
        debug_assert!(index > 0);

        PageIndex(NonZeroUsize::new_unchecked(index))
    }

    /// Returns the inner value.
    pub(crate) fn value(&self) -> usize { self.0.get() }
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

} // mod tests
