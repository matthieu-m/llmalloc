//! A Helper for tests.

use core::{
    mem,
    ops::Range,
    ptr::NonNull,
};

use crate::internals::blocks::{AtomicBlockForeign, AtomicBlockForeignList, BlockForeign, BlockForeignList};

use super::local::Local;

pub(crate) const BLOCK_SIZE: usize = 32;

pub(crate) struct BlockStore([usize; 256]);

impl BlockStore {
    pub(crate) const CAPACITY: usize = 256;

    pub(crate) fn get(&self, index: usize) -> NonNull<u8> { NonNull::from(&self.0[index]).cast() }

    pub(crate) fn end(&self) -> NonNull<u8> {
        let pointer = unsafe { self.get(0).as_ptr().add(BLOCK_SIZE / 4 * BlockStore::CAPACITY) };
        NonNull::new(pointer).unwrap()
    }

    /// Borrows self, outside of the compiler's overview.
    pub(crate) unsafe fn create_local(&self, block_size: usize) -> Local {
        assert!(block_size >= mem::size_of::<BlockForeign>());

        let (begin, end) = self.begin_end(block_size);

        Local::new(block_size, begin, end)
    }

    /// Creates a `BlockForeignList` containing the specified range of cells.
    ///
    /// #   Safety
    ///
    /// -   `local` should have been created from this instance.
    /// -   The cells should not _also_ be available through `local.next`.
    pub(crate) unsafe fn create_foreign_list(&self, local: &Local, blocks: Range<usize>) -> BlockForeignList {
        assert!(blocks.start <= blocks.end);

        let block_size = local.block_size();

        let (begin, end) = self.begin_end(block_size);
        assert_eq!(end, local.end());
        assert!(blocks.end <= (end.as_ptr() as usize - begin.as_ptr() as usize) / block_size);

        let list = BlockForeignList::default();

        for index in blocks.rev() {
            let block_address = begin.as_ptr().add(index * block_size);
            let block = NonNull::new(block_address as *mut BlockForeign).unwrap();
            list.push(block);
        }

        list
    }

    /// Creates a stack of `BlockForeign` containing the specified range of blocks.
    ///
    /// #   Safety
    ///
    /// -   `local` should have been created from this instance.
    /// -   The blocks should not _also_ be available through `local.next`.
    pub(crate) unsafe fn create_foreign_stack(&self, local: &Local, blocks: Range<usize>) -> NonNull<AtomicBlockForeign> {
        let list = self.create_foreign_list(local, blocks);

        let block = AtomicBlockForeignList::default();
        block.extend(&list);

        block.steal().unwrap()
    }

    //  Internal: Compute begin and end for a given `block_size`.
    unsafe fn begin_end(&self, block_size: usize) -> (NonNull<u8>, NonNull<u8>) {
        let begin = self as *const Self as *mut Self as *mut u8;
        let end = begin.add(mem::size_of::<Self>());

        let number_elements = (end as usize - begin as usize) / block_size;
        let begin = end.sub(number_elements * block_size);

        (NonNull::new(begin).unwrap(), NonNull::new(end).unwrap())
    }
}

impl Default for BlockStore {
    fn default() -> Self {
        let result: Self = unsafe { mem::zeroed() };
        assert_eq!(Self::CAPACITY, result.0.len());

        result
    }
}
