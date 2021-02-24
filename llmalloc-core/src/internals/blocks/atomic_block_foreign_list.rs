//! Intrusive Linked List of AtomicBlockForeign.

use core::{
    ops::Deref,
    ptr::NonNull,
    sync::atomic::{self, Ordering},
};

use super::{AtomicPtr, AtomicBlockForeign, BlockForeignList};

/// AtomicBlockForeignList.
#[derive(Default)]
pub(crate) struct AtomicBlockForeignList(AtomicPtr<AtomicBlockForeign>);

impl AtomicBlockForeignList {
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

impl Deref for AtomicBlockForeignList {
    type Target = AtomicPtr<AtomicBlockForeign>;

    fn deref(&self) -> &Self::Target { &self.0 }
}

#[cfg(test)]
mod tests {

use super::*;
use super::super::{AlignedArray, BlockForeign, BlockForeignList, BlockLocal, BlockLocalStack};

#[test]
fn block_foreign_ptr_extend_steal() {
    let array = AlignedArray::<BlockForeign>::default();
    let (a, b, c) = (array.get(0), array.get(1), array.get(2));
    let (x, y, z) = (array.get(10), array.get(11), array.get(12));

    let list = BlockForeignList::default();

    list.push(c);
    list.push(b);
    list.push(a);

    let foreign = AtomicBlockForeignList::default();
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
