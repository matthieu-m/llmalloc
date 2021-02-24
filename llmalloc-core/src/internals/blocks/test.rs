//! Test utilities;

use core::ptr::NonNull;

#[repr(align(256))]
#[derive(Default)]
pub(crate) struct AlignedArray<T>([AlignedElement<T>; 32]);

impl<T> AlignedArray<T> {
    pub(crate) fn get(&self, index: usize) -> NonNull<T> {
        let cell = &self.0[index];
        NonNull::from(cell).cast()
    }
}

#[repr(align(16))]
#[derive(Default)]
struct AlignedElement<T>(T);
