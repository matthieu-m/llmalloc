//! A collection of utilities.

use core::ptr::NonNull;

mod power_of_2;

pub use power_of_2::PowerOf2;

/// Returns whether the pointer is sufficiently aligned for the given alignment.
pub(crate) fn is_sufficiently_aligned_for(ptr: NonNull<u8>, alignment: PowerOf2) -> bool {
    (ptr.as_ptr() as usize) % alignment == 0
}

//  The Prefetch Guard is used to prevent pre-fetching on a previous page from accidentally causing false-sharing with
//  the thread currently using the LargePage.
#[repr(align(128))]
#[derive(Default)]
pub(crate) struct PrefetchGuard(u8);

#[cfg(test)]
mod tests {

use crate::PowerOf2;

use super::*;

#[test]
fn is_sufficiently_aligned_for() {
    fn is_aligned_for(ptr: usize, alignment: usize) -> bool {
        let alignment = PowerOf2::new(alignment).unwrap();
        let ptr = NonNull::new(ptr as *mut u8).unwrap();
        super::is_sufficiently_aligned_for(ptr, alignment)
    }

    fn is_not_aligned_for(ptr: usize, alignment: usize) -> bool {
        !is_aligned_for(ptr, alignment)
    }

    assert!(is_aligned_for(1, 1));
    assert!(is_aligned_for(2, 1));
    assert!(is_aligned_for(3, 1));
    assert!(is_aligned_for(4, 1));

    assert!(is_not_aligned_for(1, 2));
    assert!(is_aligned_for(2, 2));
    assert!(is_not_aligned_for(3, 2));
    assert!(is_aligned_for(4, 2));
}

}
