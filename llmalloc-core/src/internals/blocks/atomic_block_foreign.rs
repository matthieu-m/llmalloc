//! A Block of Foreign memory, accessed from multiple threads.

use core::{
    cell::Cell,
    mem,
    ops::Deref,
    ptr::NonNull,
    sync::atomic::{self, Ordering},
};

use super::{AtomicLength, AtomicPtr, BlockForeign, BlockForeignList, BlockPtr};

/// AtomicBlockForeign.
///
/// A AtomicBlockForeign points to memory not local to the current ThreadLocal.
#[repr(C)]
#[derive(Default)]
pub(crate) struct AtomicBlockForeign {
    //  Pointer to the next block, linked-list style, if any.
    pub(crate) next: AtomicBlockForeignPtr,
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

        if mem::size_of::<BlockPtr<BlockForeign>>() != PTR_SIZE || mem::size_of::<AtomicBlockForeignPtr>() != PTR_SIZE {
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

/// AtomicBlockForeignPtr.
#[derive(Default)]
pub(crate) struct AtomicBlockForeignPtr(AtomicPtr<AtomicBlockForeign>);

impl AtomicBlockForeignPtr {
    /// Returns the length of the tail list.
    pub(crate) fn len(&self) -> usize {
        let head = self.0.load();

        head.map(|head| unsafe { head.as_ref() }.length.load() + 1)
            .unwrap_or(0)
    }

    /// Steals the content of the list.
    pub(crate) fn steal(&self) -> Option<NonNull<AtomicBlockForeign>> { self.0.exchange(None) }

    /// Extends the tail-list pointed to by prepending `list`, atomically.
    ///
    /// Returns the size of the new list.
    ///
    /// #   Safety
    ///
    /// -   Assumes the list is not empty.
    pub(crate) fn extend(&self, list: &BlockForeignList) -> usize {
        debug_assert!(!list.is_empty());

        let additional_length = list.len();

        //  Safety:
        //  -   The list is assumed not to be empty.
        let (head, tail) = unsafe { list.steal() };

        atomic::fence(Ordering::Release);

        //  Safety:
        //  -   Access to the list blocks is exclusive.
        //  -   A Release atomic fence was called after the last write to the `BlockForeign` list.
        let (head, tail) = unsafe { (AtomicBlockForeign::from(head), AtomicBlockForeign::from(tail)) };

        let mut current = self.0.load();

        loop {
            let current_ptr = match current {
                Some(ptr) => ptr,
                None =>
                    match self.0.compare_exchange(current, Some(head)) {
                        Ok(_) => return additional_length,
                        Err(new_current) => {
                            current = new_current;
                            //  Not None, otherwise the exchange would have succeeded.
                            current.unwrap()
                        },
                    },
            };

            //  Safety:
            //  -   `current` is not null.
            let current_length = unsafe { current_ptr.as_ref().length.load() };

            //  Safety:
            //  -   Bounded lifetime.
            unsafe {
                tail.as_ref().next.store(current);
                tail.as_ref().length.store(current_length);
            }

            //  Safety:
            //  -   Bounded lifetime.
            unsafe {
                head.as_ref().length.store(current_length + additional_length);
            }

            match self.0.compare_exchange(current, Some(head)) {
                Ok(_) => return current_length + additional_length + 1,
                Err(new_current) => current = new_current,
            }
        }
    }
}

impl Deref for AtomicBlockForeignPtr {
    type Target = AtomicPtr<AtomicBlockForeign>;

    fn deref(&self) -> &Self::Target { &self.0 }
}

#[cfg(test)]
mod tests {

use super::*;
use super::super::{AlignedArray, BlockLocal, BlockLocalStack};

#[test]
fn block_foreign_ptr_extend_steal() {
    let array = AlignedArray::<BlockForeign>::default();
    let (a, b, c) = (array.get(0), array.get(1), array.get(2));
    let (x, y, z) = (array.get(10), array.get(11), array.get(12));

    let list = BlockForeignList::default();

    list.push(c);
    list.push(b);
    list.push(a);

    let foreign = AtomicBlockForeignPtr::default();
    assert_eq!(0, foreign.len());

    foreign.extend(&list);
    assert_eq!(3, foreign.len());

    list.push(z);
    list.push(y);
    list.push(x);

    assert_eq!(6, foreign.extend(&list));
    assert_eq!(6, foreign.len());

    let head = foreign.steal();
    assert_eq!(0, foreign.len());

    //  Double-check the list!
    let local = BlockLocalStack::default();
    unsafe { local.refill(BlockLocal::from_atomic(head.unwrap())) };

    assert_eq!(Some(x.cast()), local.pop());
    assert_eq!(Some(y.cast()), local.pop());
    assert_eq!(Some(z.cast()), local.pop());
    assert_eq!(Some(a.cast()), local.pop());
    assert_eq!(Some(b.cast()), local.pop());
    assert_eq!(Some(c.cast()), local.pop());
    assert_eq!(None, local.pop());
}

} // mod tests
