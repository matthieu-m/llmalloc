//! The configuration of llmalloc-core.
//!
//! A single Configuration instance should be shared between all related SocketLocals and ThreadLocals.
//!
//! llmalloc features 3 allocation categories:
//!
//! -   Normal: fulfilled using slabs, cached in ThreadLocal, and lazily refilled.
//! -   Large: fulfilled using bitmaps, stored in SocketLocal, and eagerly refilled.
//! -   Huge: fullfilled directly by the Platform trait.
//!
//! The Configuration instance allows adjusting the thresholds of those categories to better match the underlying
//! platform native page sizes.

use core::{
    num,
    ptr::NonNull,
};

use super::{AllocationSize, Category, ClassSize, Layout, PowerOf2};

/// Configuration
///
/// The Configuration instance allows adjusting the thresholds of the allocation categories.
pub trait Configuration {
    /// The size of Large Pages, from which Normal allocations are produced.
    ///
    /// The minimum this size can be is 4096, as the header of 512 bytes cannot exceed 1/8th of the page.
    const LARGE_PAGE_SIZE: PowerOf2;

    /// The size of Huge Pages, from which Large allocations are produced.
    ///
    /// Allocations from the Platform are always requested as multiple of this page size.
    const HUGE_PAGE_SIZE: PowerOf2;
}

/// Properties
///
/// Properties of a given Configuration.
///
/// Work-around for the inability to implement static methods directly on a trait.
pub struct Properties<C>(C);

impl<C> Properties<C>
    where
        C: Configuration
{
    /// Returns the minimum allocation size.
    pub fn minimum_allocation_size() -> AllocationSize { ClassSize::minimum_allocation_size() }

    /// Returns the threshold of Normal allocations.
    ///
    /// Allocations for a size less than or equal to the threshold are of the Normal category.
    pub fn normal_threshold() -> AllocationSize { AllocationSize::new(C::LARGE_PAGE_SIZE.value() / 16 * 7) }

    /// Returns the threshold of Large allocations.
    ///
    /// Allocations for a size less than or equal to the threshold, yet too large to be of the Normal category, are of
    /// the Large category.
    pub fn large_threshold() -> AllocationSize {
        AllocationSize::new(C::HUGE_PAGE_SIZE.value() - C::LARGE_PAGE_SIZE.value())
    }

    /// Returns the category of a pointer, based on its alignment.
    pub fn category_of_pointer(ptr: NonNull<u8>) -> Category {
        let ptr = ptr.as_ptr() as usize;

        if ptr % C::LARGE_PAGE_SIZE != 0 {
            Category::Normal
        } else if ptr % C::HUGE_PAGE_SIZE != 0 {
            Category::Large
        } else {
            Category::Huge
        }
    }

    /// Returns the category of an allocation, based on its size.
    pub fn category_of_size(size: usize) -> Category {
        debug_assert!(size > 0);

        if size <= Self::normal_threshold().value() {
            Category::Normal
        } else if size <= Self::large_threshold().value() {
            Category::Large
        } else {
            Category::Huge
        }
    }

    /// Returns the class size of an allocation, if Normal.
    pub fn class_size_of_size(size: usize) -> Option<ClassSize> {
        if size == 0 || size > Self::normal_threshold().value() {
            return None;
        }

        //  Safety:
        //  -   Not 0.
        let size = unsafe { num::NonZeroUsize::new_unchecked(size) };
    
        Some(ClassSize::from_size(size))
    }

    /// Returns the allocation size and alignment of an allocation, based on its size.
    pub fn layout_of_size(size: usize) -> Layout {
        match Self::category_of_size(size) {
            Category::Normal => Self::class_size_of_size(size).expect("Normal").layout(),
            Category::Large => Self::page_layout(C::LARGE_PAGE_SIZE, size),
            Category::Huge => Self::page_layout(C::HUGE_PAGE_SIZE, size),
        }
    }

    fn page_layout(page_size: PowerOf2, size: usize) -> Layout {
        let size = page_size.round_up(size);
        let align = page_size.value();
    
        //  Safety:
        //  -   `size` is a multiple of `align`.
        //  -   `align` is a power of 2.
        unsafe { Layout::from_size_align_unchecked(size, align) }
    }
}

#[cfg(test)]
mod tests {

use super::*;

struct TestConfiguration;

impl Configuration for TestConfiguration {
    const LARGE_PAGE_SIZE: PowerOf2 = unsafe { PowerOf2::new_unchecked(1 << 11) };
    const HUGE_PAGE_SIZE: PowerOf2 = unsafe { PowerOf2::new_unchecked(1 << 20) };
}

type TestProperties = Properties<TestConfiguration>;

#[test]
fn properties_normal_threshold() {
    fn threshold() -> usize {
        TestProperties::normal_threshold().value()
    }

    assert_eq!(896, threshold());
}

#[test]
fn properties_large_threshold() {
    fn threshold() -> usize {
        TestProperties::large_threshold().value()
    }

    assert_eq!(1_046_528, threshold());
}

#[test]
fn properties_category_of_pointer() {
    fn category(ptr: usize) -> Category {
        TestProperties::category_of_pointer(NonNull::new(ptr as *mut u8).unwrap())
    }

    assert_eq!(Category::Normal, category(1 << 0));
    assert_eq!(Category::Normal, category(1 << 1));
    assert_eq!(Category::Normal, category(1 << 2));
    assert_eq!(Category::Normal, category(1 << 9));
    assert_eq!(Category::Normal, category(1 << 10));

    assert_eq!(Category::Large, category(1 << 11));
    assert_eq!(Category::Large, category(1 << 12));
    assert_eq!(Category::Large, category(1 << 18));
    assert_eq!(Category::Large, category(1 << 19));

    assert_eq!(Category::Huge, category(1 << 20));
    assert_eq!(Category::Huge, category(1 << 21));
}

#[test]
fn properties_category_of_size() {
    fn category(size: usize) -> Category {
        TestProperties::category_of_size(size)
    }

    assert_eq!(Category::Normal, category(1));
    assert_eq!(Category::Normal, category(2));
    assert_eq!(Category::Normal, category(895));
    assert_eq!(Category::Normal, category(896));
    
    assert_eq!(Category::Large, category(897));
    assert_eq!(Category::Large, category(898));
    assert_eq!(Category::Large, category(1_046_527));
    assert_eq!(Category::Large, category(1_046_528));

    assert_eq!(Category::Huge, category(1_046_529));
    assert_eq!(Category::Huge, category(1 << 20));
    assert_eq!(Category::Huge, category(1 << 21));
    assert_eq!(Category::Huge, category(1 << 22));
}

#[test]
fn properties_class_size_of_size() {
    fn class_size(size: usize) -> Option<usize> {
        TestProperties::class_size_of_size(size).map(|c| c.value())
    }

    let minimum_allocation_size = TestProperties::minimum_allocation_size().value();

    assert_eq!(None, class_size(0));

    assert_eq!(Some(0), class_size(1));
    assert_eq!(Some(0), class_size(minimum_allocation_size));
    assert_eq!(Some(1), class_size(minimum_allocation_size + 1));
    assert_eq!(Some(19), class_size(895));
    assert_eq!(Some(19), class_size(896));

    assert_eq!(None, class_size(897));
    assert_eq!(None, class_size(898));
}

#[test]
fn properties_layout_of_size() {
    fn layout(size: usize) -> (usize, usize) {
        let layout = TestProperties::layout_of_size(size);
        (layout.size(), layout.align())
    }

    let min = TestProperties::minimum_allocation_size().value();

    assert_eq!((min, min), layout(1));
    assert_eq!((min, min), layout(2));
    assert_eq!((768, 256), layout(768));
    assert_eq!((896, 128), layout(896));

    assert_eq!((2048, 2048), layout(897));
    assert_eq!((2048, 2048), layout(2048));
    assert_eq!((4096, 2048), layout(2049));
    assert_eq!((4096, 2048), layout(4096));
    assert_eq!((1_046_528, 2048), layout(1_046_527));
    assert_eq!((1_046_528, 2048), layout(1_046_528));

    assert_eq!((1 << 20, 1 << 20), layout(1_046_529));
    assert_eq!((1 << 20, 1 << 20), layout(1 << 20));
    assert_eq!((1 << 21, 1 << 20), layout(1 << 21));
    assert_eq!((1 << 22, 1 << 20), layout(1 << 22));
}

}
