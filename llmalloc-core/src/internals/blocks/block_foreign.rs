//! A Block of memory of a Foreign thread, only accessed by the local thread.

use core::{
    cell::Cell,
    ptr::{self, NonNull},
};

use crate::{PowerOf2, utils};

use super::BlockPtr;

/// BlockForeign.
///
/// A BlockForeign points to memory not local to the current ThreadLocal, but is still only manipulated by the current
/// thread.
#[repr(C)]
#[derive(Default)]
pub(crate) struct BlockForeign {
    //  Pointer to the next cell, linked-list style, if any.
    pub(crate) next: BlockPtr<BlockForeign>,
    //  Length of linked-list starting at the next cell in CellAtomicForeignPtr.
    //  Only accurate for the head.
    pub(crate) length: Cell<usize>,
    //  Tail of the list, only used by BlockForeignList.
    pub(crate) tail: BlockPtr<BlockForeign>,
}

impl BlockForeign {
    /// In-place constructs a `CellAtomicForeign`.
    ///
    /// #   Safety
    ///
    /// -   Assumes that access to the memory location is exclusive.
    /// -   Assumes that there is sufficient memory available.
    /// -   Assumes that the pointer is correctly aligned.
    #[allow(clippy::cast_ptr_alignment)]
    pub(crate) unsafe fn initialize(at: NonNull<u8>) -> NonNull<Self> {
        debug_assert!(utils::is_sufficiently_aligned_for(at, PowerOf2::align_of::<Self>()));

        //  Safety:
        //  -   `at` is assumed to be sufficiently aligned.
        let ptr = at.as_ptr() as *mut Self;

        //  Safety:
        //  -   Access to the memory location is exclusive.
        //  -   `at` is assumed to be sufficiently sized.
        ptr::write(ptr, Self::default());

        at.cast()
    }
}

#[cfg(test)]
mod tests {

use core::mem::MaybeUninit;

use super::*;

#[test]
fn block_foreign_initialize() {
    let mut block = MaybeUninit::<BlockForeign>::uninit();

    //  Safety:
    //  -   Access to the memory location is exclusive.
    unsafe { ptr::write_bytes(block.as_mut_ptr(), 0xfe, 1) };

    //  Safety:
    //  -   Access to the memory location is exclusive.
    //  -   The memory location is sufficiently sized and aligned for `BlockForeign`.
    unsafe { BlockForeign::initialize(NonNull::from(&block).cast()) };

    //  Safety:
    //  -   Initialized!
    let block = unsafe { block.assume_init() };

    assert!(block.next.get().is_none());
    assert_eq!(0, block.length.get());
    assert!(block.tail.get().is_none());
}    

} // mod tests
