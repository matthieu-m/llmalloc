//! A Block of memory from the Local thread, only accessed by the local thread.

use core::{
    mem,
    ptr::{self, NonNull},
};

use crate::{PowerOf2, utils};

use super::{AtomicBlockForeign, AtomicBlockForeignList, BlockForeign, BlockForeignList, BlockPtr};

/// BlockLocal.
///
/// A BlockLocal points to memory local to the current ThreadLocal.
#[repr(C)]
#[derive(Default)]
pub(crate) struct BlockLocal {
    next: BlockLocalStack,
}

impl BlockLocal {
    /// In-place constructs a `BlockLocal`.
    ///
    /// #   Safety
    ///
    /// -   Assumes that access to the memory location is exclusive.
    /// -   Assumes that there is sufficient memory available.
    /// -   Assumes that the pointer is correctly aligned.
    #[allow(clippy::cast_ptr_alignment)]
    pub(crate) unsafe fn initialize(at: NonNull<u8>) -> NonNull<BlockLocal> {
        debug_assert!(utils::is_sufficiently_aligned_for(at, PowerOf2::align_of::<BlockLocal>()));

        //  Safety:
        //  -   `at` is assumed to be sufficiently aligned.
        let ptr = at.as_ptr() as *mut BlockLocal;

        //  Safety:
        //  -   Access to the memory location is exclusive.
        //  -   `ptr` is assumed to be sufficiently sized.
        ptr::write(ptr, BlockLocal::default());

        at.cast()
    }

    /// In-place reinterpret a `AtomicBlockForeign` as a `BlockLocal`.
    ///
    /// #   Safety
    ///
    /// -   Assumes that access to the block, and all tail blocks, is exclusive.
    pub(crate) unsafe fn from_atomic(foreign: NonNull<AtomicBlockForeign>) -> NonNull<BlockLocal> {
        //  Safety:
        //  -   The layout are checked to be compatible below.
        let local = foreign.cast();

        debug_assert!(Self::are_layout_compatible(foreign, local));

        local
    }

    /// In-place reinterpret a `BlockForeign` as a `BlockLocal`.
    ///
    /// #   Safety
    ///
    /// -   Assumes that access to the block, and all tail blocks, is exclusive.
    pub(crate) unsafe fn from_foreign(foreign: NonNull<BlockForeign>) -> NonNull<BlockLocal> {
        //  Safety:
        //  -   The layout are checked to be compatible below.
        let atomic: NonNull<AtomicBlockForeign> = foreign.cast();

        //  Safety:
        //  -   The layout are checked to be compatible below.
        let local = atomic.cast();

        debug_assert!(Self::are_layout_compatible(atomic, local));

        local
    }

    //  Returns whether the layout of AtomicBlockForeign and BlockLocal are compatible.
    //
    //  The layout are compatible if:
    //  -   BlockLocalStack and AtomicBlockForeignList are both plain pointers, size-wise.
    //  -   BlockLocal::next and AtomicBlockForeign::next are placed at the same offset.
    fn are_layout_compatible(foreign: NonNull<AtomicBlockForeign>, local: NonNull<BlockLocal>) -> bool {
        const PTR_SIZE: usize = mem::size_of::<*const u8>();

        if mem::size_of::<BlockLocalStack>() != PTR_SIZE || mem::size_of::<AtomicBlockForeignList>() != PTR_SIZE {
            return false;
        }

        let foreign_offset = {
            let address = foreign.as_ptr() as usize;
            //  Safety:
            //  -   Bounded lifetime.
            let next_address = unsafe { &foreign.as_ref().next as *const _ as usize };
            next_address - address
        };

        let local_offset = {
            let address = local.as_ptr() as usize;
            //  Safety:
            //  -   Bounded lifetime.
            let next_address = unsafe { &local.as_ref().next as *const _ as usize };
            next_address - address
        };

        foreign_offset == local_offset
    }
}

/// BlockLocalStack.
#[derive(Default)]
pub(crate) struct BlockLocalStack(BlockPtr<BlockLocal>);

impl BlockLocalStack {
    /// Creates an instance.
    pub(crate) fn new(ptr: Option<NonNull<BlockLocal>>) -> Self { Self(BlockPtr::new(ptr)) }

    /// Creates an instance from a raw pointer.
    ///
    /// #   Safety
    ///
    /// -   Assumes that access to the memory location is exclusive.
    /// -   Assumes that there is sufficient memory available.
    /// -   Assumes that the pointer is correctly aligned.
    pub(crate) unsafe fn from_raw(ptr: NonNull<u8>) -> Self { Self::new(Some(BlockLocal::initialize(ptr))) }

    /// Returns whether the stack is empty, or not.
    pub(crate) fn is_empty(&self) -> bool { self.get().is_none() }

    /// Pops the head of the tail-list, if any.
    pub(crate) fn pop(&self) -> Option<NonNull<BlockLocal>> {
        let result = self.get()?;

        //  Safety:
        //  -   Non-null, and valid instance.
        let next = unsafe { result.as_ref().next.get() };
        self.set(next);

        Some(result)
    }

    /// Prepends the block to the head of the tail-list.
    pub(crate) fn push(&self, block: NonNull<BlockLocal>) {
        unsafe {
            //  Safety:
            //  -   Bounded lifetime.
            block.as_ref().next.set(self.get());
        }

        self.set(Some(block));
    }

    /// Refills the list from a BlockForeign.
    ///
    /// #   Safety
    ///
    /// -   Assumes that access to the memory location, and any tail location, is exclusive.
    pub(crate) unsafe fn refill(&self, block: NonNull<BlockLocal>) {
        debug_assert!(self.is_empty());

        self.set(Some(block))
    }

    /// Extends the tail-list pointed to by prepending `list`.
    ///
    /// #   Safety
    ///
    /// -   Assumes that the access to the tail blocks, is exclusive.
    /// -   Assumes that the list is not empty.
    pub(crate) unsafe fn extend(&self, list: &BlockForeignList) {
        debug_assert!(!list.is_empty());

        //  Safety:
        //  -   `list` is assumed not to be empty.
        let (head, tail) = list.steal();

        //  Link the tail.
        let tail = BlockLocal::from_foreign(tail);

        //  Safety:
        //  -   Boundded lifetime.
        tail.as_ref().next.set(self.get());

        //  Set the head.
        let head = BlockLocal::from_foreign(head);
        self.set(Some(head));
    }

    /// Returns the pointer, possibly null.
    #[cfg(test)]
    pub(crate) fn peek(&self) -> Option<NonNull<BlockLocal>> { self.get() }

    fn get(&self) -> Option<NonNull<BlockLocal>> { self.0.get() }

    fn set(&self, value: Option<NonNull<BlockLocal>>) { self.0.set(value); }
}

#[cfg(test)]
mod tests {

use core::mem::MaybeUninit;

use super::*;
use super::super::AlignedArray;

#[test]
fn block_local_initialize() {
    let mut block = MaybeUninit::<BlockLocal>::uninit();

    //  Safety:
    //  -   Access to the memory location is exclusive.
    unsafe { ptr::write_bytes(block.as_mut_ptr(), 0xfe, 1) };

    //  Safety:
    //  -   Access to the memory location is exclusive.
    //  -   The memory location is sufficiently sized and aligned for `BlockLocal`.
    unsafe { BlockLocal::initialize(NonNull::from(&block).cast()) };

    //  Safety:
    //  -   Initialized!
    let block = unsafe { block.assume_init() };

    assert!(block.next.is_empty());
}

#[test]
fn block_local_from() {
    let array = AlignedArray::<BlockForeign>::default();

    let (head, tail) = (array.get(1), array.get(2));

    //  Safety:
    //  -   Bounded lifetime.
    unsafe {
        head.as_ref().next.set(Some(tail));
        head.as_ref().length.set(1);
    }

    //  Safety:
    //  -   Access to the blocks is exclusive.
    let block = unsafe { BlockLocal::from_foreign(head) };

    //  Safety:
    //  -  Bounded lifetime.
    let next = unsafe { block.as_ref().next.peek() };

    assert_eq!(Some(tail.cast()), next);
}

#[test]
fn block_local_stack_is_empty() {
    let array = AlignedArray::<BlockLocal>::default();
    let block = array.get(1);

    let stack = BlockLocalStack::new(None);
    assert!(stack.is_empty());

    let stack = BlockLocalStack::new(Some(block));
    assert!(!stack.is_empty());
}

#[test]
fn block_local_stack_peek() {
    let array = AlignedArray::<BlockLocal>::default();
    let block = array.get(1);

    let stack = BlockLocalStack::new(None);
    assert_eq!(None, stack.peek());

    let stack = BlockLocalStack::new(Some(block));
    assert_eq!(Some(block), stack.peek());
}

#[test]
fn block_local_stack_pop_push() {
    let array = AlignedArray::<BlockLocal>::default();
    let (a, b) = (array.get(1), array.get(2));

    let stack = BlockLocalStack::default();
    assert_eq!(None, stack.peek());
    assert_eq!(None, stack.pop());

    stack.push(a);

    assert_eq!(Some(a), stack.peek());
    assert_eq!(Some(a), stack.pop());

    assert_eq!(None, stack.peek());
    assert_eq!(None, stack.pop());

    stack.push(b);
    stack.push(a);

    assert_eq!(Some(a), stack.peek());
    assert_eq!(Some(a), stack.pop());

    assert_eq!(Some(b), stack.peek());
    assert_eq!(Some(b), stack.pop());

    assert_eq!(None, stack.peek());
    assert_eq!(None, stack.pop());
}

#[test]
fn block_local_stack_refill() {
    let array = AlignedArray::<BlockLocal>::default();
    let (head, tail) = (array.get(1), array.get(2));

    //  Safety:
    //  -   Bounded lifetime.
    unsafe {
        head.as_ref().next.set(Some(tail));
    }

    let stack = BlockLocalStack::default();

    unsafe { stack.refill(head) };

    assert_eq!(Some(head.cast()), stack.pop());
    assert_eq!(Some(tail.cast()), stack.pop());
    assert_eq!(None, stack.pop());
}

#[test]
fn block_local_stack_extend_empty() {
    let array = AlignedArray::<BlockForeign>::default();
    let (head, tail) = (array.get(1), array.get(2));

    let list = BlockForeignList::default();
    list.push(tail);
    list.push(head);

    let stack = BlockLocalStack::default();

    unsafe { stack.extend(&list) };

    assert_eq!(Some(head.cast()), stack.pop());
    assert_eq!(Some(tail.cast()), stack.pop());
    assert_eq!(None, stack.pop());
}

#[test]
fn block_local_stack_extend_existing() {
    let array = AlignedArray::<BlockForeign>::default();
    let (head, tail) = (array.get(1), array.get(2));

    let local = BlockLocal::default();
    let local = NonNull::from(&local);

    let list = BlockForeignList::default();
    list.push(tail);
    list.push(head);

    let stack = BlockLocalStack::new(Some(local));

    unsafe { stack.extend(&list) };

    assert_eq!(Some(head.cast()), stack.pop());
    assert_eq!(Some(tail.cast()), stack.pop());
    assert_eq!(Some(local), stack.pop());
    assert_eq!(None, stack.pop());
}

} // mod tests
