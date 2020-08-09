#![no_std]
#![deny(missing_docs)]

//! Exposition of LLAllocator API via a C ABI.

use core::alloc::Layout;

use llmalloc::LLAllocator;

/// Prepares the socket-local and thread-local structures for allocation.
///
/// Returns 0 on success, and a negative value otherwise.
///
/// Failure to warm up the current thread may occur if:
///
/// -   The socket-local structure is not ready, and the underlying `Platform` cannot allocate one.
/// -   The socket-local structure cannot allocate a thread-local structure.
#[cold]
#[no_mangle]
pub extern fn ll_warm_up() -> i32 { if ALLOCATOR.warm_up().is_ok() { 0 } else { -1 } }

/// Ensures that at least `target` `HugePage` are allocated on the socket.
///
/// Returns the minimum of the currently allocated number of pages and `target`.
///
/// Failure to meet the `target` may occur if:
///
/// -   The maximum number of `HugePage` a `socket` may contain has been reached.
/// -   The underlying `Platform` is failing to allocate more `HugePage`.
#[cold]
#[no_mangle]
pub extern fn ll_reserve(target: usize) -> usize { ALLOCATOR.reserve(target) }

/// Allocates `size` bytes of memory, generally suitably aligned.
///
/// If the allocation fails, the returned pointer may be NULL.
///
/// If the allocation succeeds, the pointer is aligned on the greatest power of 2 which divides `size`, or 1 if `size`
/// is 0; this guarantees that the pointer is suitably aligned:
///
/// -   The alignment of the type for which memory is allocated must be a power of 2.
/// -   The size of the type for which memory is allocated must be a multiple of its alignment.
/// -   Therefore, the greatest power of 2 which divides `size` is greater than the required alignment.
pub extern fn ll_malloc(size: usize) -> *mut u8 {
    let shift = size.trailing_zeros();
    let alignment = 1usize << shift;

    //  Safety:
    //  -   `alignment` is non-zero.
    //  -   `alignment` is a power of 2.
    //  -   `size` is a multiple of `alignment`.
    let layout = unsafe { Layout::from_size_align_unchecked(size, alignment) };

    ALLOCATOR.allocate(layout)
}

/// Allocates `size` bytes of memory, aligned as specified.
///
/// If the allocation fails, the returned pointer may be NULL.
///
/// #   Safety
///
/// -   Assumes that `alignment` is non-zero.
/// -   Assumes that `alignment` is a power of 2.
/// -   Assumes that `size` is a multiple of `alignment`.
pub unsafe extern fn ll_aligned_malloc(size: usize, alignment: usize) -> *mut u8 {
        //  Safety:
    //  -   `alignment` is non-zero.
    //  -   `alignment` is a power of 2.
    //  -   `size` is a multiple of `alignment`.
    let layout = Layout::from_size_align_unchecked(size, alignment);

    ALLOCATOR.allocate(layout)
}

/// Deallocates the memory located at `pointer`.
///
/// #   Safety
///
/// -   Assumes `pointer` has been returned by a prior call to `allocate`.
/// -   Assumes `pointer` has not been deallocated since its allocation.
/// -   Assumes the memory pointed by `pointer` is no longer in use.
pub unsafe extern fn ll_free(pointer: *mut u8) { ALLOCATOR.deallocate(pointer) }

//
//  Implementation
//

static ALLOCATOR: LLAllocator = LLAllocator::new();
