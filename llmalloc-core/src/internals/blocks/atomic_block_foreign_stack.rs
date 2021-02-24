//! A thread-safe stack of Foreign Blocks of memory.

use core::ptr::NonNull;

use super::{AtomicBlockForeign, AtomicPtr};

#[derive(Default)]
pub(crate) struct AtomicBlockForeignStack(AtomicPtr<AtomicBlockForeign>);

impl AtomicBlockForeignStack {
    /// Pops the top of the stack, if any.
    pub(crate) fn pop(&self) -> Option<NonNull<AtomicBlockForeign>> {
        let mut current = self.0.load();

        loop {
            let head_ptr = current?;

            //  Safety:
            //  -   `head_ptr` points to properly aligned memory.
            //  -   `head`'s lifetime is bounded.
            let head = unsafe { head_ptr.as_ref() };

            let next = head.next.load();

            match self.0.compare_exchange(current, next) {
                Ok(_) => return Some(head_ptr),
                Err(new_current) => current = new_current,
            }
        }
    }

    /// Pushes onto the stack.
    pub(crate) fn push(&self, block: NonNull<AtomicBlockForeign>) {
        let mut current = self.0.load();

        loop {
            let current_ptr = match current {
                Some(ptr) => ptr,
                None =>
                    match self.0.compare_exchange(current, Some(block)) {
                        Ok(_) => return,
                        Err(new_current) => {
                            current = new_current;
                            //  new_current is non-None, otherwise the compare exchange would have succeeded.
                            current.unwrap()
                        },
                    },
            };

            //  Safety:
            //  -   `current_ptr` points to properly aligned memory.
            //  -   `next`'s lifetime is bounded.
            let next = unsafe { current_ptr.as_ref() };

            let new_length = next.length.load() + 1;

            //  Safety:
            //  -   Short lifetime
            unsafe {
                block.as_ref().next.store(current);
                block.as_ref().length.store(new_length);
            }

            match self.0.compare_exchange(current, Some(block)) {
                Ok(_) => return,
                Err(new_current) => current = new_current,
            }
        }
    }
}

#[cfg(test)]
mod tests {

use super::*;
use super::super::AlignedArray;

#[test]
fn atomic_block_foreign_stack_pop_push() {
    let array = AlignedArray::<AtomicBlockForeign>::default();
    let (a, b) = (array.get(0), array.get(1));

    let stack = AtomicBlockForeignStack::default();

    assert_eq!(None, stack.pop());

    stack.push(a);
    stack.push(b);

    assert_eq!(Some(b), stack.pop());
    assert_eq!(Some(a), stack.pop());
    assert_eq!(None, stack.pop());
}

} // mod tests
