//! List of Foreign Blocks of memory.

use core::{
    hint,
    ptr::NonNull,
};

use crate::Configuration;

use super::{BlockForeign, BlockPtr};

/// BlockForeignList.
#[derive(Default)]
pub(crate) struct BlockForeignList(BlockPtr<BlockForeign>);

impl BlockForeignList {
    /// Returns whether the list is empty.
    pub(crate) fn is_empty(&self) -> bool { self.0.get().is_none() }

    /// Returns the length of the list.
    pub(crate) fn len(&self) -> usize {
        self.head()
            //  Safety:
            //  -   `head` is valid.
            .map(|head| unsafe { head.as_ref().length.get() + 1 })
            .unwrap_or(0)
    }

    /// Returns the head of the list, it may be None.
    pub(crate) fn head(&self) -> Option<NonNull<BlockForeign>> { self.0.get() }

    /// Returns true if either the list is empty or it contains a Block within the same page.
    pub(crate) fn is_compatible<C>(&self, block: NonNull<BlockForeign>) -> bool
        where
            C: Configuration
    {
        let head = if let Some(head) = self.head() {
            head
        } else {
            return true;
        };

        let head = head.as_ptr() as usize;
        let block = block.as_ptr() as usize;

        let page_size = C::LARGE_PAGE_SIZE;

        page_size.round_down(head) == page_size.round_down(block)
    }

    /// Prepends the block to the head of the tail-list.
    ///
    /// Returns the length of the list after the operation.
    pub(crate) fn push(&self, block: NonNull<BlockForeign>) -> usize {
        match self.head() {
            None => {
                //  Safety:
                //  -   Bounded lifetime.
                unsafe {
                    block.as_ref().next.set(None);
                    block.as_ref().length.set(0);
                    block.as_ref().tail.set(Some(block));
                }

                self.0.set(Some(block));

                1
            },
            Some(head) => {
                //  Safety:
                //  -   The pointer is valid.
                let length = unsafe { head.as_ref().length.get() };
                let tail = unsafe { head.as_ref().tail.get() };

                {
                    //  Safety:
                    //  -   Bounded lifetime.
                    let block = unsafe { block.as_ref() };

                    block.next.set(Some(head));
                    block.length.set(length + 1);
                    block.tail.set(tail);
                }

                self.0.set(Some(block));

                //  +1 as the length incremented.
                //  +1 as length is the length of the _tail_.
                length + 2
            },
        }
    }

    //  Steals the content of the list.
    // 
    //  Returns the head and tail, in this order.
    // 
    //  After the call, the list is empty.
    // 
    //  #   Safety
    //
    //  -   Assumes the list is not empty.
    pub(crate) unsafe fn steal(&self) -> (NonNull<BlockForeign>, NonNull<BlockForeign>) {
        debug_assert!(!self.is_empty());

        let head = force_unwrap(self.0.replace_with_null());

        //  Safety:
        //  -   `head` is not null, as the list is not empty.
        let tail = head.as_ref().tail.replace_with_null();

        (head, force_unwrap(tail))
    }
}

//
//  Implementation
//

#[inline(always)]
unsafe fn force_unwrap<T>(t: Option<T>) -> T {
    match t {
        None => {
            debug_assert!(false, "Unexpectedly empty Option");
            hint::unreachable_unchecked()
        },
        Some(t) => t,
    }
}

#[cfg(test)]
mod tests {

use crate::PowerOf2;

use super::*;
use super::super::AlignedArray;

#[test]
fn block_foreign_list_default() {
    let list = BlockForeignList::default();

    assert!(list.is_empty());
    assert_eq!(0, list.len());
}

#[test]
fn block_foreign_list_is_compatible() {
    struct C;

    impl Configuration for C {
        const LARGE_PAGE_SIZE: PowerOf2 = unsafe { PowerOf2::new_unchecked(1 << 8) };
        const HUGE_PAGE_SIZE: PowerOf2 = unsafe { PowerOf2::new_unchecked(1 << 16) };
    }

    let array = AlignedArray::<BlockForeign>::default();

    //  The array is aligned on a 256 bytes boundaries, and contains 16-bytes aligned elements.
    //  Hence the page break is at element 16.
    let (a, b, c) = (array.get(14), array.get(15), array.get(16));

    let list = BlockForeignList::default();

    assert!(list.is_compatible::<C>(a));
    assert!(list.is_compatible::<C>(b));
    assert!(list.is_compatible::<C>(c));

    list.push(a);

    assert!(list.is_compatible::<C>(b));
    assert!(!list.is_compatible::<C>(c));
}

#[test]
fn block_foreign_list_push_steal() {
    let array = AlignedArray::<BlockForeign>::default();
    let (a, b, c) = (array.get(0), array.get(1), array.get(2));

    let list = BlockForeignList::default();
    assert_eq!(0, list.len());

    list.push(c);
    assert_eq!(1, list.len());

    list.push(b);
    assert_eq!(2, list.len());

    list.push(a);
    assert_eq!(3, list.len());

    //  Safety:
    //  -   `list` is not empty.
    let (head, tail) = unsafe { list.steal() };

    assert_eq!(a, head);
    assert_eq!(c, tail);
}

} // mod tests
