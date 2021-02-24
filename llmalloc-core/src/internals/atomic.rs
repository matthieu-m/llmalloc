//! Building brick for List and Stack.

use core::{
    ptr::{self, NonNull},
    sync::atomic::{self, Ordering},
};

//  Automatically uses Acquire/Release, to synchronize before CellLocal conversion.
#[derive(Default)]
pub(crate) struct AtomicLength(atomic::AtomicUsize);

impl AtomicLength {
    pub(crate) fn load(&self) -> usize { self.0.load(Ordering::Acquire) }

    pub(crate) fn store(&self, len: usize) { self.0.store(len, Ordering::Release) }
}

//  Automatically uses Acquire/Release, to synchronize before CellLocal conversion.
pub(crate) struct AtomicPtr<T>(atomic::AtomicPtr<T>);

impl<T> AtomicPtr<T> {
    pub(crate) fn load(&self) -> Option<NonNull<T>> { NonNull::new(self.0.load(Ordering::Acquire)) }

    pub(crate) fn store(&self, ptr: Option<NonNull<T>>) { self.0.store(into_raw(ptr), Ordering::Release) }

    pub(crate) fn exchange(&self, ptr: Option<NonNull<T>>) -> Option<NonNull<T>> {
        NonNull::new(self.0.swap(into_raw(ptr), Ordering::AcqRel))
    }

    pub(crate) fn compare_exchange(&self, current: Option<NonNull<T>>, new: Option<NonNull<T>>)
        -> Result<Option<NonNull<T>>, Option<NonNull<T>>>
    {
        self.0.compare_exchange(into_raw(current), into_raw(new), Ordering::AcqRel, Ordering::Acquire)
            .map(NonNull::new)
            .map_err(NonNull::new)
    }
}

impl<T> Default for AtomicPtr<T> {
    fn default() -> Self { Self(atomic::AtomicPtr::new(ptr::null_mut())) }
}

//
//  Implementation
//

#[inline(always)]
fn into_raw<T>(ptr: Option<NonNull<T>>) -> *mut T {
    ptr.map(|t| t.as_ptr())
        .unwrap_or(ptr::null_mut())
}
