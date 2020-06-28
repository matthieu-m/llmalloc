//! Description of various properties of the allocations.

use core::{alloc, mem, num};

pub use alloc::Layout;
pub use crate::utils::PowerOf2;

/// AllocationSize
///
/// The effective size of a given allocation.
#[derive(Debug, Default, Clone, Copy, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct AllocationSize(usize);

impl AllocationSize {
    /// Creates a new instance with a specific value.
    pub const fn new(value: usize) -> Self { Self(value) }

    /// Returns the underlying value.
    pub const fn value(&self) -> usize { self.0 }
}

/// Category
///
/// The Category of an allocation.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub enum Category {
    /// Normal.
    ///
    /// Normal allocations are fulfilled by slabs, cached in ThreadLocal, and lazily refilled.
    Normal,
    /// Large.
    ///
    /// Large allocations are fulfilled by bitmaps, stored in SocketLocal, and eagerly refilled.
    Large,
    /// Huge.
    ///
    /// Huge allocations are fullfilled by the Platform instance.
    Huge,
}

/// ClassSize
///
/// The class size of a Normal allocation.
///
/// Large and Huge allocations are not bucketed by size, and therefore it is meaningless for them.
#[derive(Debug, Default, Clone, Copy, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct ClassSize(usize);

impl ClassSize {
    /// Returns the minimum allocation size.
    ///
    /// llmalloc stores meta-data within freed allocations, requiring a minimum size.
    pub const fn minimum_allocation_size() -> AllocationSize { AllocationSize(MIN_ALLOCATION_SIZE) }

    /// Returns the number of class sizes for a given Normal page size.
    pub const fn number_classes(page_size: PowerOf2) -> usize {
        //  On a 64 bits architecture:
        //  -   32 -> 4
        //  -   64 -> 8
        //  -   128 -> 12
        //  ...
        let span = MIN_ALLOCATION_SIZE.leading_zeros() - page_size.value().leading_zeros();
        (span * 4) as usize
    }

    /// Creates a new instance.
    pub const fn new(value: usize) -> Self { Self(value) }

    /// Creates an instance based on the requested size of the allocation.
    pub fn from_size(size: num::NonZeroUsize) -> Self {
        //  On a 64 bits architecture:
        //  -    1-16 -> 0
        //
        //  -   17-20 -> 1
        //  -   21-24 -> 2
        //  -   25-28 -> 3
        //  -   29-32 -> 4
        //
        //  -   33-40 -> 5
        //  -   41-48 -> 6
        //  -   49-56 -> 7
        //  -   57-64 -> 8
        //  ...
        let size = size.get() - 1;

        if size < MIN_ALLOCATION_SIZE {
            return Self(0);
        }

        //  Index of highest bit of (original size - 1).
        let highbit = USIZE_BITS - size.leading_zeros() as usize;

        //  Next two highest bits of (original size - 1).
        let nextduo = (size >> (highbit - 3)) & 0b11;

        //  highbit 6: index 1-4
        //  highbit 7: index 5-8
        //  highbit 8: index 9-12
        let base_index = (highbit - 6) * 4 + 1;

        Self(base_index + nextduo)
    }

    /// Returns the underlying value.
    pub const fn value(&self) -> usize { self.0 }

    /// Returns the Layout for such an allocation.
    pub const fn layout(&self) -> alloc::Layout {
        let (size, alignment) = self.properties();

        //  Safety:
        //  -   `size` is a multiple of alignment.
        //  -   `alignment` is a power of 2.
        unsafe { alloc::Layout::from_size_align_unchecked(size, alignment) }
    }

    /// Returns the number of elements of this class size fitting in a given number of bytes.
    pub fn number_elements(&self, bytes: usize) -> usize {
        //  What is going on here?
        //
        //  Integer divisions are amongst the slowest CPU operations, this implementation attempts to eschew them.
        //
        //  There are only a handful of sizes, which can easily be calculated. From `self.properties()` we see:
        //
        //  ```
        //  let size = MIN_ALLOCATION_SIZE / 4 * (4 + rem) << main;
        //  ```
        //
        //  Therefore, we "reverse" the division by `size`:
        //
        //  ```
        //      bytes / size
        //  <=> bytes / (MIN_ALLOCATION_SIZE / 4 * (4 + rem) << main)
        //  <=> (bytes >> main) / (MIN_ALLOCATION_SIZE / 4 * (4 + rem))
        //  <=> (bytes >> main) / MIN_ALLOCATION_SIZE * 4 / (4 + rem)
        //  <=> (bytes >> MIN_SIZE_SHIFT << 2 >> main) / (4 + rem)
        //  <=> (bytes >> ADJUSTED_SIZE_SHIFT >> main) / (4 + rem)
        //  ```
        //
        //  And rather than dividing by `4 + rem`, which may not be optimized out correctly, the algorithm instead
        //  branches on the 4 possible values of `rem` and hardcodes the divisor, leaving it up to the optimizer to
        //  apply peephole optimizations.
        const MIN_SIZE_SHIFT: usize = MIN_ALLOCATION_SIZE.trailing_zeros() as usize;
        const ADJUSTED_SIZE_SHIFT: usize = MIN_SIZE_SHIFT - 2;

        let (main, rem) = (self.0 / 4, self.0 % 4);

        let intermediate = bytes >> ADJUSTED_SIZE_SHIFT >> main;

        match rem {
            0 => intermediate / 4,
            1 => intermediate / 5,
            2 => intermediate / 6,
            3 => intermediate / 7,
            //  Safety:
            //  -   X % 4 is always between 0 and 3.
            _ => unsafe { core::hint::unreachable_unchecked() },
        }
    }

    /// Returns the size and alignment for such an allocation.
    ///
    /// Guarantees that:
    /// -   `size` is a multiple of `alignment`.
    /// -   `alignment` is a power of 2.
    const fn properties(&self) -> (usize, usize) {
        //  On a 64 bits architecture:
        //  0 -> (16, 16)
        //  1 -> (20, 4)
        //  2 -> (24, 8)
        //  3 -> (28, 4)
        //
        //  4 -> (32, 32)
        //  5 -> (40, 8)
        //  6 -> (48, 16)
        //  7 -> (56, 8)
        //
        //  8 -> (64, 64)
        //  ...
        let (main, rem) = (self.0 / 4, self.0 % 4);

        let size = {
            let quarter = MIN_ALLOCATION_SIZE / 4;
            let adjusted = quarter * (4 + rem);
            adjusted << main
        };

        let alignment = {
            let quarter = MIN_ALLOCATION_SIZE / 4;
            let adjusted = quarter * Self::alignment_multiplier(rem);
            adjusted << main
        };

        (size, alignment)
    }

    /// Only defined for remainder in [0, 4].
    ///
    /// Returns 4 for 0, 2 for 2, 1 for 1 and 3.
    const fn alignment_multiplier(remainder: usize) -> usize {
        let odd_mask = remainder & 1;
        (4 - remainder) * (1 - odd_mask) + odd_mask
    }
}

//
//  Implementation Details.
//

//  The minimum allocation size is tailored so that:
//  -   A `CellForeign` fits within the memory.
//  -   A `CellForeign` is correctly aligned on said memory.
const MIN_ALLOCATION_SIZE: usize = mem::size_of::<usize>() * 4;
const USIZE_BITS: usize = mem::size_of::<usize>() * 8;

#[cfg(test)]
mod tests {

use super::*;

#[test]
fn assumptions() {
    assert!(MIN_ALLOCATION_SIZE >= 4);
    assert_eq!(1, MIN_ALLOCATION_SIZE.count_ones());
    assert_eq!(1, USIZE_BITS.count_ones());
}

#[test]
fn class_size_number_classes() {
    fn number_classes(page_size: usize) -> usize {
        let page_size = PowerOf2::new(page_size).expect("Power of 2");
        ClassSize::number_classes(page_size)
    }

    assert_eq!(0, number_classes(MIN_ALLOCATION_SIZE));
    assert_eq!(4, number_classes(2 * MIN_ALLOCATION_SIZE));
    assert_eq!(8, number_classes(4 * MIN_ALLOCATION_SIZE));
    assert_eq!(12, number_classes(8 * MIN_ALLOCATION_SIZE));
    assert_eq!(16, number_classes(16 * MIN_ALLOCATION_SIZE));
    assert_eq!(20, number_classes(32 * MIN_ALLOCATION_SIZE));
    assert_eq!(24, number_classes(64 * MIN_ALLOCATION_SIZE));
}

#[cfg(target_pointer_width = "32")]
#[test]
fn class_size_from_size() {
    fn from_size(size: usize) -> usize {
        let size = num::NonZeroUsize::new(size).expect("Not 0");
        let class_size = ClassSize::from_size(size);
        class_size.value()
    }

    assert_eq!(0, from_size(1));
    assert_eq!(0, from_size(16));

    assert_eq!(1, from_size(17));
    assert_eq!(1, from_size(20));
    assert_eq!(2, from_size(21));
    assert_eq!(2, from_size(24));
    assert_eq!(3, from_size(25));
    assert_eq!(3, from_size(28));
    assert_eq!(4, from_size(29));
    assert_eq!(4, from_size(32));

    assert_eq!(5, from_size(33));

    assert_eq!(8, from_size(1 << 6));
    assert_eq!(12, from_size(1 << 7));
    assert_eq!(16, from_size(1 << 8));
    assert_eq!(20, from_size(1 << 9));
    assert_eq!(24, from_size(1 << 10));
    assert_eq!(64, from_size(1 << 20));
    assert_eq!(104, from_size(1 << 30));
    assert_eq!(108, from_size(1 << 31));
    assert_eq!(112, from_size(usize::MAX));
}

#[cfg(target_pointer_width = "64")]
#[test]
fn class_size_from_size() {
    fn from_size(size: usize) -> usize {
        let size = num::NonZeroUsize::new(size).expect("Not 0");
        let class_size = ClassSize::from_size(size);
        class_size.value()
    }

    assert_eq!(0, from_size(1));
    assert_eq!(0, from_size(32));

    assert_eq!(1, from_size(33));
    assert_eq!(1, from_size(40));
    assert_eq!(2, from_size(41));
    assert_eq!(2, from_size(48));
    assert_eq!(3, from_size(49));
    assert_eq!(3, from_size(56));
    assert_eq!(4, from_size(57));
    assert_eq!(4, from_size(64));

    assert_eq!(5, from_size(65));

    assert_eq!(8, from_size(1 << 7));
    assert_eq!(12, from_size(1 << 8));
    assert_eq!(16, from_size(1 << 9));
    assert_eq!(20, from_size(1 << 10));
    assert_eq!(60, from_size(1 << 20));
    assert_eq!(100, from_size(1 << 30));
    assert_eq!(140, from_size(1 << 40));
    assert_eq!(180, from_size(1 << 50));
    assert_eq!(220, from_size(1 << 60));
    assert_eq!(232, from_size(1 << 63));
    assert_eq!(236, from_size(usize::MAX));
}

#[test]
fn class_size_layout() {
    fn layout(class_size: usize) -> (usize, usize) {
        let class_size = ClassSize::new(class_size);
        let layout = class_size.layout();
        (layout.size(), layout.align())
    }

    let base = MIN_ALLOCATION_SIZE;

    assert_eq!((4 * base / 4, 4 * base / 4), layout(0));
    assert_eq!((5 * base / 4, 1 * base / 4), layout(1));
    assert_eq!((6 * base / 4, 2 * base / 4), layout(2));
    assert_eq!((7 * base / 4, 1 * base / 4), layout(3));

    assert_eq!((4 * base / 2, 4 * base / 2), layout(4));
    assert_eq!((5 * base / 2, 1 * base / 2), layout(5));
    assert_eq!((6 * base / 2, 2 * base / 2), layout(6));
    assert_eq!((7 * base / 2, 1 * base / 2), layout(7));

    assert_eq!((4 * base / 1, 4 * base / 1), layout(8));
}

#[test]
fn class_size_number_elements() {
    fn number_elements(size: usize, bytes: usize) -> usize {
        let size = num::NonZeroUsize::new(size).expect("Not 0");
        let class_size = ClassSize::from_size(size);
        assert_eq!(size.get(), class_size.layout().size());

        class_size.number_elements(bytes)
    }

    assert_eq!(0, number_elements(32, 31));
    assert_eq!(1, number_elements(32, 32));
    assert_eq!(1, number_elements(32, 63));
    assert_eq!(2, number_elements(32, 64));
    assert_eq!(2, number_elements(32, 95));
    assert_eq!(3, number_elements(32, 96));

    assert_eq!(0, number_elements(40, 39));
    assert_eq!(1, number_elements(40, 40));
    assert_eq!(1, number_elements(40, 79));
    assert_eq!(2, number_elements(40, 80));
    assert_eq!(2, number_elements(40, 119));
    assert_eq!(3, number_elements(40, 120));

    assert_eq!(0, number_elements(48, 47));
    assert_eq!(1, number_elements(48, 48));
    assert_eq!(1, number_elements(48, 95));
    assert_eq!(2, number_elements(48, 96));
    assert_eq!(2, number_elements(48, 143));
    assert_eq!(3, number_elements(48, 144));

    assert_eq!(0, number_elements(56, 55));
    assert_eq!(1, number_elements(56, 56));
    assert_eq!(1, number_elements(56, 111));
    assert_eq!(2, number_elements(56, 112));
    assert_eq!(2, number_elements(56, 167));
    assert_eq!(3, number_elements(56, 168));
}

#[test]
fn class_size_properties() {
    fn properties(class_size: usize) -> (usize, usize) {
        let class_size = ClassSize::new(class_size);
        class_size.properties()
    }

    let base = MIN_ALLOCATION_SIZE;

    assert_eq!((4 * base / 4, 4 * base / 4), properties(0));
    assert_eq!((5 * base / 4, 1 * base / 4), properties(1));
    assert_eq!((6 * base / 4, 2 * base / 4), properties(2));
    assert_eq!((7 * base / 4, 1 * base / 4), properties(3));

    assert_eq!((4 * base / 2, 4 * base / 2), properties(4));
    assert_eq!((5 * base / 2, 1 * base / 2), properties(5));
    assert_eq!((6 * base / 2, 2 * base / 2), properties(6));
    assert_eq!((7 * base / 2, 1 * base / 2), properties(7));

    assert_eq!((4 * base / 1, 4 * base / 1), properties(8));
}

#[test]
fn class_size_alignment_multiplier() {
    assert_eq!(4, ClassSize::alignment_multiplier(0));
    assert_eq!(1, ClassSize::alignment_multiplier(1));
    assert_eq!(2, ClassSize::alignment_multiplier(2));
    assert_eq!(1, ClassSize::alignment_multiplier(3));
}

}
