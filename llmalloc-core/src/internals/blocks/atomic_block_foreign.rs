//! A Block of Foreign memory, accessed from multiple threads.

use core::{
    cell::Cell,
    mem,
    ptr::NonNull,
};

use super::{AtomicLength, AtomicPtr, BlockForeign, BlockPtr};

/// AtomicBlockForeign.
///
/// A AtomicBlockForeign points to memory not local to the current ThreadLocal.
#[repr(C)]
#[derive(Default)]
pub(crate) struct AtomicBlockForeign {
    //  Pointer to the next block, linked-list style, if any.
    pub(crate) next: AtomicPtr<AtomicBlockForeign>,
    //  Length of linked-list starting at the next block in AtomicBlockForeignPtr.
    //  Only accurate for the head.
    pub(crate) length: AtomicLength,
}

impl AtomicBlockForeign {
    /// In-place reinterpret a `BlockForeign` as a `AtomicBlockForeign`.
    ///
    /// #   Safety
    ///
    /// -   Assumes that access to the block, and all tail blocks, is exclusive.
    /// -   Assumes that a Release atomic fence was called after the last write to the `BlockForeign` list.
    pub(crate) unsafe fn from(foreign: NonNull<BlockForeign>) -> NonNull<AtomicBlockForeign> {
        //  Safety:
        //  -   The layout are checked to be compatible below.
        let atomic = foreign.cast();

        debug_assert!(Self::are_layout_compatible(foreign, atomic));

        atomic
    }

    //  Returns whether the layout of BlockForeign and AtomicBlockForeign are compatible.
    //
    //  The layout are compatible if:
    //  -   BlockPtr<BlockForeign> and AtomicBlockForeignPtr are both plain pointers, size-wise.
    //  -   Block<usize> and AtomicLength are both plain usize, size-wise.
    //  -   AtomicBlockForeign::next and BlockForeign::next are placed at the same offset.
    //  -   AtomicBlockForeign::length and BlockForeign::length are placed at the same offset.
    fn are_layout_compatible(foreign: NonNull<BlockForeign>, atomic: NonNull<AtomicBlockForeign>) -> bool {
        const PTR_SIZE: usize = mem::size_of::<*const u8>();
        const USIZE_SIZE: usize = mem::size_of::<usize>();

        if mem::size_of::<BlockPtr<BlockForeign>>() != PTR_SIZE || mem::size_of::<AtomicPtr<AtomicBlockForeign>>() != PTR_SIZE {
            return false;
        }

        if mem::size_of::<Cell<usize>>() != USIZE_SIZE || mem::size_of::<AtomicLength>() != USIZE_SIZE {
            return false;
        }

        let (foreign_next_offset, foreign_length_offset) = {
            let address = foreign.as_ptr() as usize;
            //  Safety:
            //  -   Bounded lifetime.
            let next_address = unsafe { &foreign.as_ref().next as *const _ as usize };
            let length_address = unsafe { &foreign.as_ref().length as *const _ as usize };
            (next_address - address, length_address - address)
        };

        let (atomic_next_offset, atomic_length_offset) = {
            let address = atomic.as_ptr() as usize;
            //  Safety:
            //  -   Bounded lifetime.
            let next_address = unsafe { &atomic.as_ref().next as *const _ as usize };
            let length_address = unsafe { &atomic.as_ref().length as *const _ as usize };
            (next_address - address, length_address - address)
        };

        foreign_next_offset == atomic_next_offset && foreign_length_offset == atomic_length_offset
    }
}
