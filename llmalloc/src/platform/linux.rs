//! Implementation of Linux specific calls.

use core::{alloc::Layout, marker, ptr, sync::atomic};

use llmalloc_core::{self, PowerOf2};

use super::{NumaNodeIndex, Configuration, Platform, ThreadLocal};

/// Implementation of the Configuration trait, for Linux.
#[derive(Default)]
pub(crate) struct LLConfiguration;

impl Configuration for LLConfiguration {
    //  2 MB
    const LARGE_PAGE_SIZE: PowerOf2 = unsafe { PowerOf2::new_unchecked(2 * 1024 * 1024) };

    //  1 GB
    const HUGE_PAGE_SIZE: PowerOf2 = unsafe { PowerOf2::new_unchecked(1024 * 1024 * 1024) };
}

/// Implementation of the Platform trait, for Linux.
#[derive(Default)]
pub(crate) struct LLPlatform;

impl LLPlatform {
    /// Creates an instance.
    pub(crate) const fn new() -> Self { Self }
}

impl llmalloc_core::Platform for LLPlatform {
    unsafe fn allocate(&self, layout: Layout) -> *mut u8 {
        const HUGE_PAGE_SIZE: PowerOf2 = LLConfiguration::HUGE_PAGE_SIZE;

        assert!(layout.size() % HUGE_PAGE_SIZE == 0,
            "Incorrect size: {} % {} != 0", layout.size(), HUGE_PAGE_SIZE.value());
        assert!(layout.align() <= HUGE_PAGE_SIZE.value(),
            "Incorrect alignment: {} > {}", layout.align(), HUGE_PAGE_SIZE.value());

        let candidate = mmap_huge(layout.size())
            .or_else(|| mmap_exact(layout.size()))
            .or_else(|| mmap_over(layout.size()))
            .map(|pointer| pointer.as_ptr())
            .unwrap_or(ptr::null_mut());

        debug_assert!(candidate as usize % HUGE_PAGE_SIZE == 0,
            "Incorrect alignment of allocation: {:x} % {:x} != 0", candidate as usize, HUGE_PAGE_SIZE.value());

        candidate
    }

    unsafe fn deallocate(&self, pointer: *mut u8, layout: Layout) {
        munmap_deallocate(pointer, layout.size());
    }
}

impl Platform for LLPlatform {
    #[cold]
    #[inline(never)]
    fn current_node(&self) -> NumaNodeIndex {
        let cpu = unsafe { sched_getcpu() };
        let node = unsafe { numa_node_of_cpu(cpu) };

        assert!(cpu >= 0, "Expected CPU index, got {}", cpu);
        assert!(node >= 0, "Expected NUMA Node index, got {}", node);

        select_node(NumaNodeIndex::new(node as u32))
    }
}

/// Implementation of the ThreadLocal trait, for Linux.
pub(crate) struct LLThreadLocal<T> {
    key: atomic::AtomicI64,
    destructor: *const u8,
    _marker: marker::PhantomData<*const T>,
}

impl<T> LLThreadLocal<T> {
    const UNINITIALIZED: i64 = -1;
    const UNDER_INITIALIZATION: i64 = -2;

    /// Creates an uninitialized instance.
    pub(crate) const fn new(destructor: *const u8) -> Self {
        let key = atomic::AtomicI64::new(-1);
        let destructor = destructor;
        let _marker = marker::PhantomData;

        LLThreadLocal { key, destructor, _marker }
    }

    #[inline(always)]
    fn get_key(&self) -> u32 {
        let key = self.key.load(atomic::Ordering::Relaxed);
        if key >= 0 { key as u32} else { unsafe { self.initialize() } }
    }

    #[cold]
    #[inline(never)]
    unsafe fn initialize(&self) -> u32 {
        let mut key = self.key.load(atomic::Ordering::Relaxed);

        if self.key.compare_and_swap(Self::UNINITIALIZED, Self::UNDER_INITIALIZATION, atomic::Ordering::Relaxed)
            == Self::UNINITIALIZED
        {
            key = self.create_key();
            self.key.store(key, atomic::Ordering::Relaxed);
        }

        while key < 0 {
            pthread_yield();
            key = self.key.load(atomic::Ordering::Relaxed);
        }

        key as u32
    }

    #[cold]
    unsafe fn create_key(&self) -> i64 {
        let mut key = 0u32;

        let result = pthread_key_create(&mut key as *mut _, self.destructor);
        assert!(result == 0, "Could not create thread-local key: {}", result);

        key as i64
    }
}

impl<T> ThreadLocal<T> for LLThreadLocal<T> {
    fn get(&self) -> *mut T {
        let key = self.key.load(atomic::Ordering::Relaxed);

        //  If key is not initialized, then a null pointer is returned.
        unsafe { pthread_getspecific(key as u32) as *mut T }
    }

    #[cold]
    #[inline(never)]
    fn set(&self, value: *mut T) {
        let key = self.get_key();

        let result = unsafe { pthread_setspecific(key, value as *mut u8) };
        assert!(result == 0, "Could not set thread-local value for {}: {}", key, result);
    }
}

unsafe impl<T> Sync for LLThreadLocal<T> {}

//  Selects the "best" node.
//
//  The Linux kernel sometimes distinguishes nodes even though their distance is 11, when the distance to self is 10.
//  This may lead to over-allocation, hence it is judged best to "cluster" the nodes together.
//
//  This function will therefore return the smallest node number whose distance to the `original` is less than or
//  equal to 11.
fn select_node(original: NumaNodeIndex) -> NumaNodeIndex {
    let original = original.value() as i32;

    for current in 0..original {
        if unsafe { numa_distance(current, original) } <= 11 {
            return NumaNodeIndex::new(current as u32);
        }
    }

    NumaNodeIndex::new(original as u32)
}

//  Attempts to allocate the required size in Huge Pages.
//
//  If non-null, the result is aligned on `HUGE_PAGE_SIZE`.
fn mmap_huge(size: usize) -> Option<ptr::NonNull<u8>> {
    const MAP_HUGE_SHIFT: u8 = 26;

    const MAP_HUGETLB: i32 = 0x40000;
    const MAP_HUGE_1GB: i32 = 30 << MAP_HUGE_SHIFT;

    mmap_allocate(size, MAP_HUGETLB | MAP_HUGE_1GB)
        .and_then(|pointer| unsafe { mmap_check(pointer, size) })
}

//  Attempts to allocate the required size in Normal (or Large) Pages.
//
//  If non-null, the result is aligned on `HUGE_PAGE_SIZE`.
fn mmap_exact(size: usize) -> Option<ptr::NonNull<u8>> {
    mmap_allocate(size, 0)
        .and_then(|pointer| unsafe { mmap_check(pointer, size) })
}

//  Attempts to allocate the required size in Normal (or Large) Pages.
//
//  Ensures the alignment is met by over-allocated then trimming front and back.
fn mmap_over(size: usize) -> Option<ptr::NonNull<u8>> {
    const ALIGNMENT: PowerOf2 = LLConfiguration::HUGE_PAGE_SIZE;

    let over_size = size + ALIGNMENT.value();
    let front_pointer = mmap_allocate(over_size, 0)?;

    let back_size = (front_pointer.as_ptr() as usize) % ALIGNMENT;
    let front_size = ALIGNMENT.value() - back_size;

    debug_assert!(front_size <= ALIGNMENT.value(), "{} > {}", front_size, ALIGNMENT.value());
    debug_assert!(back_size < ALIGNMENT.value(), "{} >= {}", back_size, ALIGNMENT.value());
    debug_assert!(front_size + size + back_size == over_size,
        "{} + {} + {} != {}", front_size, size, back_size, over_size);

    //  Safety:
    //  -   `front_size` is less than `over_size`, hence the result is within the allocated block.
    let aligned_pointer = unsafe { front_pointer.as_ptr().add(front_size) };

    debug_assert!(aligned_pointer as usize % ALIGNMENT == 0,
        "{:x} not {:x}-aligned!", aligned_pointer as usize, ALIGNMENT.value());

    //  Safety:
    //  -   `front_size + size` is less than `over_size`, hence the result is within the allocated block,
    //      or pointing to its end.
    let back_pointer = unsafe { aligned_pointer.add(size) };

    if front_size > 0 {
        //  Safety:
        //  -   `front_pointer` points to a `mmap`ed area of at least `front_size` bytes.
        //  -   `[front_pointer, front_pointer + front_size)` is no longer in use.
        unsafe { munmap_deallocate(front_pointer.as_ptr(), front_size) };
    }

    if back_size > 0 {
        //  Safety:
        //  -   `back_pointer` points to a `mmap`ed area of at least `back_size` bytes.
        //  -   `[back_pointer, back_pointer + back_size)` is no longer in use.
        unsafe { munmap_deallocate(back_pointer, back_size) };
    }

    //  Safety:
    //  -   `aligned_pointer` is not null.
    Some(unsafe { ptr::NonNull::new_unchecked(aligned_pointer) })
}

//  `mmap` alignment checker.
//
//  Returns a non-null pointer if suitably aligned, and None otherwise.
//  If none is returned, the memory has been unmapped.
//
//  #   Safety
//
//  -   Assumes that `pointer` points to a `mmap`ed area of at least `size` bytes.
//  -   Assumes that `pointer` is no longer in use, unless returned.
unsafe fn mmap_check(pointer: ptr::NonNull<u8>, size: usize) -> Option<ptr::NonNull<u8>> {
    const ALIGNMENT: PowerOf2 = LLConfiguration::HUGE_PAGE_SIZE;

    if pointer.as_ptr() as usize % ALIGNMENT == 0 {
        Some(pointer)
    } else {
        //  Safety:
        //  -   `pointer` points to a `mmap`ed area of at least `size` bytes.
        //  -   `[pointer, pointer + size)` is no longer in use.
        munmap_deallocate(pointer.as_ptr(), size);
        None
    }
}

//  Wrapper around `mmap`.
//
//  Returns a pointer to `size` bytes of memory; does not guarantee any alignment.
fn mmap_allocate(size: usize, extra_flags: i32) -> Option<ptr::NonNull<u8>> {
    const FAILURE: *mut u8 = !0 as *mut u8;

    const PROT_READ: i32 = 1;
    const PROT_WRITE: i32 = 2;

    const MAP_PRIVATE: i32 = 0x2;
    const MAP_ANONYMOUS: i32 = 0x20;

    let length = size;
    let prot = PROT_READ | PROT_WRITE;
    let flags = MAP_PRIVATE | MAP_ANONYMOUS | extra_flags;

    //  No specific address hint.
    let addr = ptr::null_mut();
    //  When used in conjunction with MAP_ANONYMOUS, fd is mandated to be -1 on some implementations.
    let fd = -1;
    //  When used in conjunction with MAP_ANONYMOUS, offset is mandated to be 0 on some implementations.
    let offset = 0;

    //  Safety:
    //  -   `addr`, `fd`, and `offset` are suitable for MAP_ANONYMOUS.
    let result = unsafe { mmap(addr, length, prot, flags, fd, offset) };

    let result = if result != FAILURE { result } else { ptr::null_mut() };
    ptr::NonNull::new(result)
}

//  Wrapper around `munmap`.
//
//  #   Panics
//
//  If `munmap` returns a non-0 result.
//
//  #   Safety
//
//  -   Assumes that `addr` points to a `mmap`ed area of at least `size` bytes.
//  -   Assumes that the range `[addr, addr + size)` is no longer in use.
unsafe fn munmap_deallocate(addr: *mut u8, size: usize) {
    let result = munmap(addr, size);
    assert!(result == 0, "Could not munmap {:x}, {}: {}", addr as usize, size, result);
}

#[link(name = "c")]
extern "C" {
    //  Returns the current index of the CPU on which the thread is executed.
    //
    //  The only possible error is ENOSYS, if the kernel does not implement getcpu.
    fn sched_getcpu() -> i32;

    //  Refer to: https://man7.org/linux/man-pages/man2/mmap.2.html
    fn mmap(addr: *mut u8, length: usize, prot: i32, flags: i32, fd: i32, offset: isize) -> *mut u8;

    //  Refer to: https://man7.org/linux/man-pages/man2/mmap.2.html
    fn munmap(addr: *mut u8, length: usize) -> i32;
}

#[link(name = "numa")]
extern "C" {
    //  Returns the NUMA node corresponding to a CPU, or -1 if the CPU is invalid.
    fn numa_node_of_cpu(cpu: i32) -> i32;

    //  Returns the distance between two NUMA nodes.
    //
    //  A node has a distance 10 to itself; factors should be multiples of 10, although 11 and 21 has been observed.
    fn numa_distance(left: i32, right: i32) -> i32;
}

#[link(name = "pthread")]
extern "C" {
    //  Initializes the value of the thread-local key.
    //
    //  Errors:
    //  -   EAGAIN: if the system lacked the necessary resources.
    //  -   ENOMEM: if insufficient memory exists to create the key.
    fn pthread_key_create(key: *mut u32, destructor: *const u8) -> i32;

    //  Gets the pointer to the thread-local value stored for key, or null.
    fn pthread_getspecific(key: u32) -> *mut u8;

    //  Sets the pointer to the thread-local value stored for key.
    //
    //  Errors:
    //  -   ENOMEM: if insufficient memory exists to associate the value with the key.
    //  -   EINVAL: if the key value is invalid.
    fn pthread_setspecific(key: u32, value: *mut u8) -> i32;

    //  Yields.
    //
    //  Errors:
    //  -   None known.
    fn pthread_yield() -> i32;
}
