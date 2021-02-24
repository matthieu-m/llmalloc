//! Pointer to Block, unsynchronized.

use core::{
    cell::Cell,
    ptr::NonNull,
};

/// BlockPtr
///
/// A simple block over a potentially null pointer to T.
pub(crate) struct BlockPtr<T>(Cell<Option<NonNull<T>>>);

impl<T> BlockPtr<T> {
    /// Creates an instance.
    pub(crate) fn new(ptr: Option<NonNull<T>>) -> Self { Self(Cell::new(ptr)) }

    /// Returns the inner pointer, possibly null.
    pub(crate) fn get(&self) -> Option<NonNull<T>> { self.0.get() }

    /// Sets the inner pointer.
    pub(crate) fn set(&self, ptr: Option<NonNull<T>>) { self.0.set(ptr); }

    /// Sets the inner pointer to null and return the previous value, possibly null.
    pub(crate) fn replace_with_null(&self) -> Option<NonNull<T>> { self.0.replace(None) }
}

impl<T> Default for BlockPtr<T> {
    fn default() -> Self { Self::new(None) }
}

#[cfg(test)]
mod tests {

use super::*;

#[test]
fn block_ptr_new() {
    let a = 1u8;
    let a = Some(NonNull::from(&a));

    let block = BlockPtr::<u8>::new(None);
    assert_eq!(None, block.get());

    let block = BlockPtr::new(a);
    assert_eq!(a, block.get());
}

#[test]
fn block_ptr_get_set() {
    let (a, b) = (1u8, 2u8);
    let (a, b) = (Some(NonNull::from(&a)), Some(NonNull::from(&b)));

    let block = BlockPtr::new(None);

    block.set(a);
    assert_eq!(a, block.get());

    block.set(b);
    assert_eq!(b, block.get());
}

#[test]
fn block_ptr_replace_with_null() {
    let a = 1u8;
    let a = Some(NonNull::from(&a));

    let block = BlockPtr::new(None);

    assert_eq!(None, block.replace_with_null());
    assert_eq!(None, block.get());

    block.set(a);

    assert_eq!(a, block.replace_with_null());
    assert_eq!(None, block.get());
}

} // mod tests
