//! Cells
//!
//! A Cell represent a unit of allocation.
//!
//! Whilst allocated, the content of the cell is purely in the hands of the user. Whilst deallocated, however, the Cell
//! storage is reused to store meta-data used to manage the memory.
//!
//! Specifically, the cells within a LargePage are maintained into a tail-list structure.
//!
//! Note: Cells are never _constructed_, instead raw memory is reinterpreted as cells.

use core::{cell, mem, ptr, sync::atomic};

use crate::{Configuration, PowerOf2};
use crate::utils;

/// CellPtr
///
/// A simple cell over a potentially null pointer to T.
pub(crate) struct CellPtr<T>(cell::Cell<*mut T>);

impl<T> CellPtr<T> {
    /// Creates an instance.
    pub(crate) fn new(ptr: *mut T) -> Self { Self(cell::Cell::new(ptr)) }

    /// Returns the inner pointer, possibly null.
    pub(crate) fn get(&self) -> *mut T { self.0.get() }

    /// Sets the inner pointer.
    pub(crate) fn set(&self, ptr: *mut T) { self.0.set(ptr); }

    /// Sets the inner pointer to null and return the previous value, possibly null.
    pub(crate) fn replace_with_null(&self) -> *mut T { self.0.replace(ptr::null_mut()) }
}

impl<T> Default for CellPtr<T> {
    fn default() -> Self { Self(cell::Cell::new(ptr::null_mut())) }
}

/// CellLocal.
///
/// A CellLocal points to memory local to the current ThreadLocal.
#[repr(C)]
#[derive(Default)]
pub(crate) struct CellLocal {
    next: CellLocalPtr,
}

impl CellLocal {
    /// In-place constructs a `CellLocal`.
    ///
    /// #   Safety
    ///
    /// -   Assumes that access to the memory location is exclusive.
    /// -   Assumes that there is sufficient memory available.
    /// -   Assumes that the pointer is correctly aligned.
    #[allow(clippy::cast_ptr_alignment)]
    pub(crate) unsafe fn initialize(at: *mut u8) -> ptr::NonNull<CellLocal> {
        debug_assert!(utils::is_sufficiently_aligned_for(at, PowerOf2::align_of::<CellLocal>()));

        //  Safety:
        //  -   `at` is assumed to be sufficiently aligned.
        let at = at as *mut CellLocal;

        //  Safety:
        //  -   Access to the memory location is exclusive.
        //  -   `at` is assumed to be sufficiently sized.
        ptr::write(at, CellLocal::default());

        //  Safety:
        //  -   Not null.
        ptr::NonNull::new_unchecked(at)
    }

    /// In-place reinterpret a `CellAtomicForeign` as a `CellLocal`.
    ///
    /// #   Safety
    ///
    /// -   Assumes that access to the cell, and all tail cells, is exclusive.
    pub(crate) unsafe fn from_atomic(foreign: ptr::NonNull<CellAtomicForeign>) -> ptr::NonNull<CellLocal> {
        //  Safety:
        //  -   The layout are checked to be compatible below.
        let local = foreign.cast();

        debug_assert!(Self::are_layout_compatible(foreign, local));

        local
    }

    /// In-place reinterpret a `CellForeign` as a `CellLocal`.
    ///
    /// #   Safety
    ///
    /// -   Assumes that access to the cell, and all tail cells, is exclusive.
    pub(crate) unsafe fn from_foreign(foreign: ptr::NonNull<CellForeign>) -> ptr::NonNull<CellLocal> {
        //  Safety:
        //  -   The layout are checked to be compatible below.
        let atomic: ptr::NonNull<CellAtomicForeign> = foreign.cast();

        //  Safety:
        //  -   The layout are checked to be compatible below.
        let local = atomic.cast();

        debug_assert!(Self::are_layout_compatible(atomic, local));

        local
    }

    //  Returns whether the layout of CellAtomicForeign and CellLocal are compatible.
    //
    //  The layout are compatible if:
    //  -   CellLocalPtr and CellAtomicForeignPtr are both plain pointers, size-wise.
    //  -   CellLocal::next and CellAtomicForeign::next are placed at the same offset.
    fn are_layout_compatible(foreign: ptr::NonNull<CellAtomicForeign>, local: ptr::NonNull<CellLocal>) -> bool {
        const PTR_SIZE: usize = mem::size_of::<*const u8>();

        if mem::size_of::<CellLocalPtr>() != PTR_SIZE || mem::size_of::<CellAtomicForeignPtr>() != PTR_SIZE {
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

/// CellLocalPtr.
#[derive(Default)]
pub(crate) struct CellLocalPtr(CellPtr<CellLocal>);

impl CellLocalPtr {
    /// Creates an instance.
    pub(crate) fn new(ptr: *mut CellLocal) -> Self { Self(CellPtr::new(ptr)) }

    /// Creates an instance from a raw pointer.
    ///
    /// #   Safety
    ///
    /// -   Assumes that access to the memory location is exclusive.
    /// -   Assumes that there is sufficient memory available.
    /// -   Assumes that the pointer is correctly aligned.
    pub(crate) unsafe fn from_raw(ptr: *mut u8) -> Self { Self::new(CellLocal::initialize(ptr).as_ptr()) }

    /// Pops the head of the tail-list, if any.
    pub(crate) fn pop(&self) -> Option<ptr::NonNull<CellLocal>> {
        let result = self.get();

        if !result.is_null() {
            //  Safety:
            //  -   Non-null, and valid instance.
            let next = unsafe { (*result).next.get() };
            self.set(next);
        }

        ptr::NonNull::new(result)
    }

    /// Prepends the cell to the head of the tail-list.
    pub(crate) fn push(&self, cell: ptr::NonNull<CellLocal>) {
        unsafe {
            //  Safety:
            //  -   Bounded lifetime.
            let cell = cell.as_ref();
            cell.next.set(self.get());
        }

        self.set(cell.as_ptr());
    }

    /// Refills the list from a CellForeign.
    ///
    /// #   Safety
    ///
    /// -   Assumes that access to the memory location, and any tail location, is exclusive.
    pub(crate) unsafe fn refill(&self, cell: ptr::NonNull<CellLocal>) {
        self.set(cell.as_ptr())
    }

    /// Extends the tail-list pointed to by prepending `list`.
    ///
    /// #   Safety
    ///
    /// -   Assumes that the access to the tail cells, is exclusive.
    /// -   Assumes that the list is not empty.
    pub(crate) unsafe fn extend(&self, list: &CellForeignList) {
        debug_assert!(!list.is_empty());

        //  Safety:
        //  -   `list` is assumed not to be empty.
        let (head, tail) = list.steal();

        //  Link the tail.
        let tail = CellLocal::from_foreign(tail);

        //  Safety:
        //  -   Boundded lifetime.
        tail.as_ref().next.set(self.get());

        //  Set the head.
        let head = CellLocal::from_foreign(head);
        self.set(head.as_ptr());
    }

    /// Returns whether the cell is null, or not.
    #[cfg(test)]
    pub(crate) fn is_null(&self) -> bool { self.get().is_null() }

    /// Returns the pointer, possibly null.
    #[cfg(test)]
    pub(crate) fn peek(&self) -> Option<ptr::NonNull<CellLocal>> { ptr::NonNull::new(self.get()) }

    fn get(&self) -> *mut CellLocal { self.0.get() }

    fn set(&self, value: *mut CellLocal) { self.0.set(value); }
}

/// CellForeign.
///
/// A CellForeign points to memory not local to the current ThreadLocal, but is still only manipulated by the current
/// thread.
#[repr(C)]
#[derive(Default)]
pub(crate) struct CellForeign {
    //  Pointer to the next cell, linked-list style, if any.
    next: CellPtr<CellForeign>,
    //  Length of linked-list starting at the next cell in CellAtomicForeignPtr.
    //  Only accurate for the head.
    length: cell::Cell<usize>,
    //  Tail of the list, only used by CellForeignList.
    tail: CellPtr<CellForeign>,
}

impl CellForeign {
    /// In-place constructs a `CellAtomicForeign`.
    ///
    /// #   Safety
    ///
    /// -   Assumes that access to the memory location is exclusive.
    /// -   Assumes that there is sufficient memory available.
    /// -   Assumes that the pointer is correctly aligned.
    #[allow(clippy::cast_ptr_alignment)]
    pub(crate) unsafe fn initialize(at: *mut u8) -> ptr::NonNull<Self> {
        debug_assert!(utils::is_sufficiently_aligned_for(at, PowerOf2::align_of::<Self>()));

        //  Safety:
        //  -   `at` is assumed to be sufficiently aligned.
        let at = at as *mut Self;

        //  Safety:
        //  -   Access to the memory location is exclusive.
        //  -   `at` is assumed to be sufficiently sized.
        ptr::write(at, Self::default());

        //  Safety:
        //  -   Not null.
        ptr::NonNull::new_unchecked(at)
    }
}

/// CellForeignList.
#[derive(Default)]
pub(crate) struct CellForeignList(CellPtr<CellForeign>);

impl CellForeignList {
    /// Returns whether the list is empty.
    pub(crate) fn is_empty(&self) -> bool { self.0.get().is_null() }

    /// Returns the length of the list.
    pub(crate) fn len(&self) -> usize {
        let head = self.0.get();

        if head.is_null() {
            0
        } else {
            //  Safety:
            //  -   The pointer is valid.
            unsafe { (*head).length.get() + 1 }
        }
    }

    /// Returns the head of the list, it may be None.
    pub(crate) fn head(&self) -> *mut CellForeign { self.0.get() }

    /// Returns true if either the list is empty or it contains a Cell within the same page.
    pub(crate) fn is_compatible<C>(&self, cell: ptr::NonNull<CellForeign>) -> bool
        where
            C: Configuration
    {
        if self.0.get().is_null() {
            return true;
        }

        let head = self.0.get() as usize;
        let cell = cell.as_ptr() as usize;

        let page_size = C::LARGE_PAGE_SIZE;

        page_size.round_down(head) == page_size.round_down(cell)
    }

    /// Prepends the cell to the head of the tail-list.
    ///
    /// Returns the length of the list after the operation.
    pub(crate) fn push(&self, cell: ptr::NonNull<CellForeign>) -> usize {
        let ptr = cell.as_ptr();

        if self.is_empty() {
            //  Safety:
            //  -   Bounded lifetime.
            unsafe {
                cell.as_ref().length.set(0);
                cell.as_ref().tail.set(ptr);
            }

            self.0.set(ptr);

            return 1;
        }

        let head = self.0.get();

        //  Safety:
        //  -   The pointer is valid.
        let length = unsafe { (*head).length.get() };
        let tail = unsafe { (*head).tail.get() };

        {
            //  Safety:
            //  -   Bounded lifetime.
            let cell = unsafe { cell.as_ref() };

            cell.next.0.set(head);
            cell.length.set(length + 1);
            cell.tail.set(tail);
        }

        self.0.set(ptr);

        //  +1 as the length incremented.
        //  +1 as length is the length of the _tail_.
        length + 2
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
    unsafe fn steal(&self) -> (ptr::NonNull<CellForeign>, ptr::NonNull<CellForeign>) {
        debug_assert!(!self.is_empty());

        let head = self.0.replace_with_null();
        debug_assert!(!head.is_null());

        //  Safety:
        //  -   `head` is not null, as the list is not empty.
        let tail = (*head).tail.replace_with_null();

        (
            ptr::NonNull::new_unchecked(head),
            ptr::NonNull::new_unchecked(tail),
        )
    }
}

/// CellAtomicForeign.
///
/// A CellAtomicForeign points to memory not local to the current ThreadLocal.
#[repr(C)]
#[derive(Default)]
pub(crate) struct CellAtomicForeign {
    //  Pointer to the next cell, linked-list style, if any.
    next: CellAtomicForeignPtr,
    //  Length of linked-list starting at the next cell in CellAtomicForeignPtr.
    //  Only accurate for the head.
    length: AtomicLength,
}

impl CellAtomicForeign {
    /// In-place reinterpret a `CellForeign` as a `CellAtomicForeign`.
    ///
    /// #   Safety
    ///
    /// -   Assumes that access to the cell, and all tail cells, is exclusive.
    /// -   Assumes that a Release atomic fence was called after the last write to the `CellForeign` list.
    pub(crate) unsafe fn from(foreign: ptr::NonNull<CellForeign>) -> ptr::NonNull<CellAtomicForeign> {
        //  Safety:
        //  -   The layout are checked to be compatible below.
        let atomic = foreign.cast();

        debug_assert!(Self::are_layout_compatible(foreign, atomic));

        atomic
    }

    //  Returns whether the layout of CellForeign and CellAtomicForeign are compatible.
    //
    //  The layout are compatible if:
    //  -   CellPtr<CellForeign> and CellAtomicForeignPtr are both plain pointers, size-wise.
    //  -   Cell<usize> and AtomicLength are both plain usize, size-wise.
    //  -   CellAtomicForeign::next and CellForeign::next are placed at the same offset.
    //  -   CellAtomicForeign::length and CellForeign::length are placed at the same offset.
    fn are_layout_compatible(foreign: ptr::NonNull<CellForeign>, atomic: ptr::NonNull<CellAtomicForeign>) -> bool {
        const PTR_SIZE: usize = mem::size_of::<*const u8>();
        const USIZE_SIZE: usize = mem::size_of::<usize>();

        if mem::size_of::<CellPtr<CellForeign>>() != PTR_SIZE || mem::size_of::<CellAtomicForeignPtr>() != PTR_SIZE {
            return false;
        }

        if mem::size_of::<cell::Cell<usize>>() != USIZE_SIZE || mem::size_of::<AtomicLength>() != USIZE_SIZE {
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

/// CellAtomicForeignPtr.
#[derive(Default)]
pub(crate) struct CellAtomicForeignPtr(AtomicPtr<CellAtomicForeign>);

impl CellAtomicForeignPtr {
    /// Returns the length of the tail list.
    pub(crate) fn len(&self) -> usize {
        let head = self.0.load();

        if head.is_null() {
            0
        } else {
            //  Safety:
            //  -   The pointer is valid.
            unsafe { (*head).length.load() + 1 }
        }
    }

    /// Steals the content of the list.
    pub(crate) fn steal(&self) -> *mut CellAtomicForeign { self.0.exchange(ptr::null_mut()) }

    /// Extends the tail-list pointed to by prepending `list`, atomically.
    ///
    /// Returns the size of the new list.
    ///
    /// #   Safety
    ///
    /// -   Assumes the list is not empty.
    pub(crate) fn extend(&self, list: &CellForeignList) -> usize {
        debug_assert!(!list.is_empty());

        let additional_length = list.len();

        //  Safety:
        //  -   The list is assumed not to be empty.
        let (head, tail) = unsafe { list.steal() };

        atomic::fence(atomic::Ordering::Release);

        //  Safety:
        //  -   Access to the list cells is exclusive.
        //  -   A Release atomic fence was called after the last write to the `CellForeign` list.
        let (head, tail) = unsafe { (CellAtomicForeign::from(head), CellAtomicForeign::from(tail)) };

        let mut current = self.0.load();

        loop {
            if current.is_null() {
                match self.0.compare_exchange(current, head.as_ptr()) {
                    Ok(_) => return additional_length,
                    Err(new_current) => current = new_current,
                }
            }

            //  If current was null, then the compare-exchange would have succeeded.
            debug_assert!(!current.is_null());

            //  Safety:
            //  -   `current` is not null.
            let current_length = unsafe { (*current).length.load() };

            //  Safety:
            //  -   Bounded lifetime.
            unsafe {
                tail.as_ref().next.0.store(current);
                tail.as_ref().length.store(current_length);
            }

            //  Safety:
            //  -   Bounded lifetime.
            unsafe {
                head.as_ref().length.store(current_length + additional_length);
            }

            match self.0.compare_exchange(current, head.as_ptr()) {
                Ok(_) => return current_length + additional_length + 1,
                Err(new_current) => current = new_current,
            }
        }
    }
}

#[derive(Default)]
pub(crate) struct CellAtomicForeignStack(AtomicPtr<CellAtomicForeign>);

impl CellAtomicForeignStack {
    /// Pops the top of the stack, if any.
    pub(crate) fn pop(&self) -> Option<ptr::NonNull<CellAtomicForeign>> {
        let mut current = self.0.load();

        loop {
            if current.is_null() {
                return None;
            }

            //  If current was null, then the compare-exchange would have succeeded.
            debug_assert!(!current.is_null());

            //  Safety:
            //  -   `current` is not null.
            let head = unsafe { &*current };

            let next = head.next.0.load();

            match self.0.compare_exchange(current, next) {
                //  Safety:
                //  -   `current` is not null.
                Ok(_) => return Some(unsafe { ptr::NonNull::new_unchecked(current) }),
                Err(new_current) => current = new_current,
            }
        }
    }

    /// Pushes onto the stack.
    pub(crate) fn push(&self, cell: ptr::NonNull<CellAtomicForeign>) {
        let mut current = self.0.load();

        loop {
            if current.is_null() {
                match self.0.compare_exchange(current, cell.as_ptr()) {
                    Ok(_) => return,
                    Err(new_current) => current = new_current,
                }
            }

            //  If current was null, then the compare-exchange would have succeeded.
            debug_assert!(!current.is_null());

            //  Safety:
            //  -   `current` is not null.
            let next = unsafe { &*current };

            let new_length = next.length.load() + 1;

            //  Safety:
            //  -   Short lifetime
            unsafe {
                cell.as_ref().next.0.store(current);
                cell.as_ref().length.store(new_length);
            }

            match self.0.compare_exchange(current, cell.as_ptr()) {
                Ok(_) => return,
                Err(new_current) => current = new_current,
            }
        }
    }
}

//
//  Implementation Details
//

//  Automatically uses Acquire/Release, to synchronize before CellLocal conversion.
#[derive(Default)]
struct AtomicLength(atomic::AtomicUsize);

impl AtomicLength {
    fn load(&self) -> usize { self.0.load(atomic::Ordering::Acquire) }

    fn store(&self, len: usize) { self.0.store(len, atomic::Ordering::Release) }
}

//  Automatically uses Acquire/Release, to synchronize before CellLocal conversion.
struct AtomicPtr<T>(atomic::AtomicPtr<T>);

impl<T> AtomicPtr<T> {
    fn load(&self) -> *mut T { self.0.load(atomic::Ordering::Acquire) }

    fn store(&self, ptr: *mut T) { self.0.store(ptr, atomic::Ordering::Release) }

    fn exchange(&self, ptr: *mut T) -> *mut T { self.0.swap(ptr, atomic::Ordering::AcqRel) }

    fn compare_exchange(&self, current: *mut T, new: *mut T) -> Result<*mut T, *mut T> {
        self.0.compare_exchange(current, new, atomic::Ordering::AcqRel, atomic::Ordering::Acquire)
    }
}

impl<T> Default for AtomicPtr<T> {
    fn default() -> Self { Self(atomic::AtomicPtr::new(ptr::null_mut())) }
}

#[cfg(test)]
mod tests {

use core::mem::MaybeUninit;

use super::*;

#[derive(Default)]
struct CellSize([usize; 3]);

const NULL: *mut u8 = ptr::null_mut();
const NULL_LOCAL: *mut CellLocal = ptr::null_mut();

#[repr(align(256))]
#[derive(Default)]
struct AlignedArray<T>([AlignedElement<T>; 32]);

impl<T> AlignedArray<T> {
    fn get(&self, index: usize) -> ptr::NonNull<T> {
        let cell = &self.0[index];
        ptr::NonNull::new(cell as *const _ as *mut _).unwrap()
    }
}

#[repr(align(16))]
#[derive(Default)]
struct AlignedElement<T>(T);

#[test]
fn cell_ptr_new() {
    let a = 1u8;
    let a = &a as *const u8 as *mut u8;

    let cell = CellPtr::new(NULL);
    assert_eq!(NULL, cell.get());

    let cell = CellPtr::new(a);
    assert_eq!(a, cell.get());
}

#[test]
fn cell_ptr_get_set() {
    let (a, b) = (1u8, 2u8);
    let (a, b) = (&a as *const u8 as *mut u8, &b as *const u8 as *mut u8);

    let cell = CellPtr::new(NULL);

    cell.set(a);
    assert_eq!(a, cell.get());

    cell.set(b);
    assert_eq!(b, cell.get());
}

#[test]
fn cell_ptr_replace_with_null() {
    let a = 1u8;
    let a = &a as *const u8 as *mut u8;

    let cell = CellPtr::new(NULL);

    assert_eq!(NULL, cell.replace_with_null());
    assert_eq!(NULL, cell.get());

    cell.set(a);

    assert_eq!(a, cell.replace_with_null());
    assert_eq!(NULL, cell.get());
}

#[test]
fn cell_local_initialize() {
    let mut cell = MaybeUninit::<CellLocal>::uninit();

    //  Safety:
    //  -   Access to the memory location is exclusive.
    unsafe { ptr::write_bytes(cell.as_mut_ptr(), 0xfe, 1) };

    //  Safety:
    //  -   Access to the memory location is exclusive.
    //  -   The memory location is sufficiently sized and aligned for `CellLocal`.
    unsafe { CellLocal::initialize(cell.as_mut_ptr() as *mut u8) };

    //  Safety:
    //  -   Initialized!
    let cell = unsafe { cell.assume_init() };

    assert!(cell.next.is_null());
}

#[test]
fn cell_local_from() {
    let array = AlignedArray::<CellForeign>::default();

    let (head, tail) = (array.get(1), array.get(2));

    //  Safety:
    //  -   Bounded lifetime.
    unsafe {
        head.as_ref().next.0.set(tail.as_ptr());
        head.as_ref().length.set(1);
    }

    //  Safety:
    //  -   Access to the cells is exclusive.
    let cell = unsafe { CellLocal::from_foreign(head) };

    //  Safety:
    //  -  Bounded lifetime.
    let next = unsafe { cell.as_ref().next.peek() };

    assert_eq!(Some(tail.cast()), next);
}

#[test]
fn cell_local_ptr_is_null() {
    let array = AlignedArray::<CellLocal>::default();
    let cell = array.get(1).as_ptr();

    let ptr = CellLocalPtr::new(NULL_LOCAL);
    assert!(ptr.is_null());

    let ptr = CellLocalPtr::new(cell);
    assert!(!ptr.is_null());
}

#[test]
fn cell_local_ptr_peek() {
    let array = AlignedArray::<CellLocal>::default();
    let cell = array.get(1);

    let ptr = CellLocalPtr::new(ptr::null_mut());
    assert_eq!(None, ptr.peek());

    let ptr = CellLocalPtr::new(cell.as_ptr());
    assert_eq!(Some(cell), ptr.peek());
}

#[test]
fn cell_local_ptr_pop_push() {
    let array = AlignedArray::<CellLocal>::default();
    let (a, b) = (array.get(1), array.get(2));

    let ptr = CellLocalPtr::default();
    assert_eq!(None, ptr.peek());
    assert_eq!(None, ptr.pop());

    ptr.push(a);

    assert_eq!(Some(a), ptr.peek());
    assert_eq!(Some(a), ptr.pop());

    assert_eq!(None, ptr.peek());
    assert_eq!(None, ptr.pop());

    ptr.push(b);
    ptr.push(a);

    assert_eq!(Some(a), ptr.peek());
    assert_eq!(Some(a), ptr.pop());

    assert_eq!(Some(b), ptr.peek());
    assert_eq!(Some(b), ptr.pop());

    assert_eq!(None, ptr.peek());
    assert_eq!(None, ptr.pop());
}

#[test]
fn cell_local_ptr_refill() {
    let array = AlignedArray::<CellLocal>::default();
    let (head, tail) = (array.get(1), array.get(2));

    //  Safety:
    //  -   Bounded lifetime.
    unsafe {
        head.as_ref().next.0.set(tail.as_ptr());
    }

    let ptr = CellLocalPtr::default();

    unsafe { ptr.refill(head) };

    assert_eq!(Some(head.cast()), ptr.pop());
    assert_eq!(Some(tail.cast()), ptr.pop());
    assert_eq!(None, ptr.pop());
}

#[test]
fn cell_local_ptr_extend_empty() {
    let array = AlignedArray::<CellForeign>::default();
    let (head, tail) = (array.get(1), array.get(2));

    let list = CellForeignList::default();
    list.push(tail);
    list.push(head);

    let ptr = CellLocalPtr::default();

    unsafe { ptr.extend(&list) };

    assert_eq!(Some(head.cast()), ptr.pop());
    assert_eq!(Some(tail.cast()), ptr.pop());
    assert_eq!(None, ptr.pop());
}

#[test]
fn cell_local_ptr_extend_existing() {
    let array = AlignedArray::<CellForeign>::default();
    let (head, tail) = (array.get(1), array.get(2));

    let local = CellLocal::default();
    let local = ptr::NonNull::new(&local as *const _ as *mut CellLocal).unwrap();

    let list = CellForeignList::default();
    list.push(tail);
    list.push(head);

    let ptr = CellLocalPtr::new(local.as_ptr());

    unsafe { ptr.extend(&list) };

    assert_eq!(Some(head.cast()), ptr.pop());
    assert_eq!(Some(tail.cast()), ptr.pop());
    assert_eq!(Some(local), ptr.pop());
    assert_eq!(None, ptr.pop());
}

#[test]
fn cell_foreign_initialize() {
    let mut cell = MaybeUninit::<CellForeign>::uninit();

    //  Safety:
    //  -   Access to the memory location is exclusive.
    unsafe { ptr::write_bytes(cell.as_mut_ptr(), 0xfe, 1) };

    //  Safety:
    //  -   Access to the memory location is exclusive.
    //  -   The memory location is sufficiently sized and aligned for `CellForeign`.
    unsafe { CellForeign::initialize(cell.as_mut_ptr() as *mut u8) };

    //  Safety:
    //  -   Initialized!
    let cell = unsafe { cell.assume_init() };

    assert!(cell.next.get().is_null());
    assert_eq!(0, cell.length.get());
    assert!(cell.tail.get().is_null());
}

#[test]
fn cell_foreign_stack_pop_push() {
    let array = AlignedArray::<CellAtomicForeign>::default();
    let (a, b) = (array.get(0), array.get(1));

    let stack = CellAtomicForeignStack::default();

    assert_eq!(None, stack.pop());

    stack.push(a);
    stack.push(b);

    assert_eq!(Some(b), stack.pop());
    assert_eq!(Some(a), stack.pop());
    assert_eq!(None, stack.pop());
}

#[test]
fn cell_foreign_list_default() {
    let list = CellForeignList::default();

    assert!(list.is_empty());
    assert_eq!(0, list.len());
}

#[test]
fn cell_foreign_list_is_compatible() {
    struct C;

    impl Configuration for C {
        const LARGE_PAGE_SIZE: PowerOf2 = unsafe { PowerOf2::new_unchecked(1 << 8) };
        const HUGE_PAGE_SIZE: PowerOf2 = unsafe { PowerOf2::new_unchecked(1 << 16) };
    }

    let array = AlignedArray::<CellForeign>::default();

    //  The array is aligned on a 256 bytes boundaries, and contains 16-bytes aligned elements.
    //  Hence the page break is at element 16.
    let (a, b, c) = (array.get(14), array.get(15), array.get(16));

    let list = CellForeignList::default();

    assert!(list.is_compatible::<C>(a));
    assert!(list.is_compatible::<C>(b));
    assert!(list.is_compatible::<C>(c));

    list.push(a);

    assert!(list.is_compatible::<C>(b));
    assert!(!list.is_compatible::<C>(c));
}

#[test]
fn cell_foreign_list_push_steal() {
    let array = AlignedArray::<CellForeign>::default();
    let (a, b, c) = (array.get(0), array.get(1), array.get(2));

    let list = CellForeignList::default();
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

#[test]
fn cell_foreign_ptr_extend_steal() {
    let array = AlignedArray::<CellForeign>::default();
    let (a, b, c) = (array.get(0), array.get(1), array.get(2));
    let (x, y, z) = (array.get(10), array.get(11), array.get(12));

    let list = CellForeignList::default();

    list.push(c);
    list.push(b);
    list.push(a);

    let foreign = CellAtomicForeignPtr::default();
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
    let local = CellLocalPtr::default();
    unsafe { local.refill(CellLocal::from_atomic(ptr::NonNull::new(head).unwrap())) };

    assert_eq!(Some(x.cast()), local.pop());
    assert_eq!(Some(y.cast()), local.pop());
    assert_eq!(Some(z.cast()), local.pop());
    assert_eq!(Some(a.cast()), local.pop());
    assert_eq!(Some(b.cast()), local.pop());
    assert_eq!(Some(c.cast()), local.pop());
    assert_eq!(None, local.pop());
}

}
