//! Local data, only accessible from the local thread.

use core::{
    cell::Cell,
    ptr::NonNull,
};

use crate::{
    PowerOf2,
    internals::blocks::{BlockForeignList, BlockLocal, BlockLocalStack},
    utils,
};

//  Local data. Only accessible from the local thread.
#[repr(align(128))]
pub(crate) struct Local {
    //  Stack of free blocks.
    next: BlockLocalStack,
    //  Pointer to the beginning of the uncarved area of the page.
    //
    //  Linking all the cells together when creating the page would take too long, so instead only the first block is
    //  prepared, and the `watermark` and `end` are initialized to denote the area of the page which can freely be
    //  carved into further cells.
    watermark: Cell<NonNull<u8>>,
    //  Pointer to the end of the page; when `watermark == end`, the entire page has been carved.
    //
    //  When the entire page has been carved, acquiring new cells from the page is only possible through `foreign.freed`.
    end: NonNull<u8>,
    //  Size, in bytes, of the cells.
    block_size: usize,
}

impl Local {
    /// Creates a new instance of `Local`.
    ///
    /// #   Safety
    ///
    /// -   `begin` and `end` are assumed to be correctly aligned and sized for a `BlockForeign` pointer.
    /// -   `end - begin` is assumed to be a multiple of `block_size`.
    pub(crate) unsafe fn new(block_size: usize, begin: NonNull<u8>, end: NonNull<u8>) -> Self {
        debug_assert!(block_size >= 1);
        debug_assert!((end.as_ptr() as usize - begin.as_ptr() as usize) % block_size == 0,
            "block_size: {}, begin: {:x}, end: {:x}", block_size, begin.as_ptr() as usize, end.as_ptr() as usize);

        let next = BlockLocalStack::from_raw(begin);
        let watermark = Cell::new(NonNull::new_unchecked(begin.as_ptr().add(block_size)));

        Self { next, watermark, end, block_size, }
    }

    /// Allocates one cell from the page, if any.
    ///
    /// Returns a null pointer is no cell is available.
    pub(crate) fn allocate(&self) -> Option<NonNull<u8>> {
        //  Fast Path.
        if let Some(block) = self.next.pop() {
            return Some(block.cast());
        }

        //  Cruise path.
        if self.watermark.get() == self.end {
            return None;
        }

        //  Expansion path.
        let result = self.watermark.get();

        //  Safety:
        //  -   `self.block_size` matches the size of the cells.
        //  -   `self.watermark` is still within bounds.
        unsafe { self.watermark.set(NonNull::new_unchecked(result.as_ptr().add(self.block_size))) };

        Some(result.cast())
    }

    /// Deallocates one cell from the page.
    ///
    /// #   Safety
    ///
    /// -   Assumes that `ptr` is sufficiently sized.
    /// -   Assumes that `ptr` is sufficiently aligned.
    pub(crate) unsafe fn deallocate(&self, ptr: NonNull<u8>) {
        debug_assert!(utils::is_sufficiently_aligned_for(ptr, PowerOf2::align_of::<BlockLocal>()));

        //  Safety:
        //  -   `ptr` is assumed to be sufficiently sized.
        //  -   `ptr` is assumed to be sufficiently aligned.
        let block = ptr.cast();

        self.next.push(block);
    }

    /// Extends the local list from a foreign list.
    ///
    /// #   Safety
    ///
    /// -   Assumes that the access to the linked cells, is exclusive.
    pub(crate) unsafe fn extend(&self, list: &BlockForeignList) {
        debug_assert!(!list.is_empty());

        //  Safety:
        //  -   It is assumed that access to the cell, and all linked cells, is exclusive.
        self.next.extend(list);
    }

    /// Refills the local list from a foreign list.
    ///
    /// #   Safety
    ///
    /// -   Assumes that access to the cell, and all linked cells, is exclusive.
    pub(crate) unsafe fn refill(&self, list: NonNull<BlockLocal>) {
        //  Safety:
        //  -   It is assumed that access to the cell, and all linked cells, is exclusive.
        self.next.refill(list);
    }

    /// Returns the size of the blocks.
    #[cfg(test)]
    pub(crate) fn block_size(&self) -> usize { self.block_size }

    /// Returns the end of the watermark area.
    #[cfg(test)]
    pub(crate) fn end(&self) -> NonNull<u8> { self.end }
}

#[cfg(test)]
mod tests {

use super::*;
use super::super::test::{BlockStore, BLOCK_SIZE};

#[test]
fn local_new() {
    //  This test actually tests `block_store.create_local` more than anything.
    //  Since further tests will depend on it correctly initializing `Local`, it is better to validate it early.
    let mut block_store = BlockStore::default();
    let end_store = block_store.end();

    {
        let local = unsafe { block_store.create_local(BLOCK_SIZE * 2) };
        assert_eq!(block_store.get(0), local.next.peek().unwrap().cast());
        assert_eq!(block_store.get(8), local.watermark.get());
        assert_eq!(end_store, local.end);
    }

    {
        let local = unsafe { block_store.create_local(BLOCK_SIZE * 2 + BLOCK_SIZE / 2) };
        assert_eq!(block_store.get(6), local.next.peek().unwrap().cast());
        assert_eq!(block_store.get(16), local.watermark.get());
        assert_eq!(end_store, local.end);
    }
}

#[test]
fn local_allocate_expansion() {
    let mut block_store = BlockStore::default();
    let local = unsafe { block_store.create_local(BLOCK_SIZE) };

    //  Bump watermark until it is no longer possible.
    for i in 0..64 {
        assert_eq!(block_store.get(4 * i), local.allocate().unwrap());
    }

    assert_eq!(local.end, local.watermark.get());

    assert_eq!(None, local.allocate());
}

#[test]
fn local_allocate_deallocate_ping_pong() {
    let mut block_store = BlockStore::default();
    let local = unsafe { block_store.create_local(BLOCK_SIZE) };

    let ptr = local.allocate();

    for _ in 0..10 {
        unsafe { local.deallocate(ptr.unwrap()) };
        assert_eq!(ptr, local.allocate());
    }

    assert_eq!(block_store.get(4), local.watermark.get());
}

#[test]
fn local_extend() {
    let mut block_store = BlockStore::default();
    let local = unsafe { block_store.create_local(BLOCK_SIZE) };

    //  Allocate all.
    while let Some(_) = local.allocate() {}

    let foreign_list = unsafe { block_store.create_foreign_list(&local, 3..7) };

    unsafe { local.extend(&foreign_list) };
    assert!(foreign_list.is_empty());

    for i in 3..7 {
        assert_eq!(block_store.get(4 * i), local.allocate().unwrap());
    }
}

#[test]
fn local_refill() {
    let mut block_store = BlockStore::default();
    let local = unsafe { block_store.create_local(BLOCK_SIZE) };

    //  Allocate all.
    while let Some(_) = local.allocate() {}

    let foreign = unsafe { block_store.create_foreign_stack(&local, 3..7) };

    unsafe { local.refill(BlockLocal::from_atomic(foreign)) };

    for i in 3..7 {
        assert_eq!(block_store.get(4 * i), local.allocate().unwrap());
    }
}

} // mod tests
