//! Implementation of Linux specific calls.

use core::{alloc::Layout, marker, mem, ptr, sync::atomic};

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
        let cpu = unsafe { libc::sched_getcpu() };
        assert!(cpu >= 0, "Expected CPU index, got {}", cpu);

        let node = unsafe { numa_node_of_cpu(cpu) };

        //  If libnuma cannot find the appropriate node (such as under WSL), then use 0 as fallback.
        if node < 0 {
            return NumaNodeIndex::new(0);
        }

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
    ///
    /// #   Safety
    ///
    /// -   Assumes that `destructor` points to an `unsafe extern "C" fn(*mut c_void)` function, or compatible.
    pub(crate) const unsafe fn new(destructor: *const u8) -> Self {
        let key = atomic::AtomicI64::new(-1);
        let _marker = marker::PhantomData;

        LLThreadLocal { key, destructor, _marker }
    }

    #[inline(always)]
    fn get_key(&self) -> libc::pthread_key_t {
        let key = self.key.load(atomic::Ordering::Relaxed);
        if key >= 0 { key as libc::pthread_key_t} else { unsafe { self.initialize() } }
    }

    #[cold]
    #[inline(never)]
    unsafe fn initialize(&self) -> libc::pthread_key_t {
        const RELAXED: atomic::Ordering = atomic::Ordering::Relaxed;

        let mut key = self.key.load(RELAXED);

        if let Ok(_) = self.key.compare_exchange(Self::UNINITIALIZED, Self::UNDER_INITIALIZATION, RELAXED, RELAXED)
        {
            key = self.create_key();
            self.key.store(key, RELAXED);
        }

        while key < 0 {
            libc::sched_yield();
            key = self.key.load(RELAXED);
        }

        key as libc::pthread_key_t
    }

    #[cold]
    unsafe fn create_key(&self) -> i64 {
        let mut key: libc::pthread_key_t = 0;

        //  Safety:
        //  -   fn pointers are just pointers.
        let destructor = mem::transmute::<_, Destructor>(self.destructor);
        let result = libc::pthread_key_create(&mut key as *mut _, Some(destructor));
        assert!(result == 0, "Could not create thread-local key: {}", result);

        key as i64
    }
}

impl<T> ThreadLocal<T> for LLThreadLocal<T> {
    fn get(&self) -> *mut T {
        let key = self.key.load(atomic::Ordering::Relaxed);

        //  If key is not initialized, then a null pointer is returned.
        unsafe { libc::pthread_getspecific(key as libc::pthread_key_t) as *mut T }
    }

    #[cold]
    #[inline(never)]
    fn set(&self, value: *mut T) {
        let key = self.get_key();

        let result = unsafe { libc::pthread_setspecific(key, value as *mut libc::c_void) };
        assert!(result == 0, "Could not set thread-local value for {}: {}", key, result);
    }
}

unsafe impl<T> Sync for LLThreadLocal<T> {}

type Destructor = unsafe extern "C" fn(*mut libc::c_void);

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

    const MAP_HUGE_1GB: libc::c_int = 30 << MAP_HUGE_SHIFT;

    mmap_allocate(size, libc::MAP_HUGETLB | MAP_HUGE_1GB)
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
    let length = size;
    let prot = libc::PROT_READ | libc::PROT_WRITE;
    let flags = libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | extra_flags;

    //  No specific address hint.
    let addr = ptr::null_mut();
    //  When used in conjunction with MAP_ANONYMOUS, fd is mandated to be -1 on some implementations.
    let fd = -1;
    //  When used in conjunction with MAP_ANONYMOUS, offset is mandated to be 0 on some implementations.
    let offset = 0;

    //  Safety:
    //  -   `addr`, `fd`, and `offset` are suitable for MAP_ANONYMOUS.
    let result = unsafe { libc::mmap(addr, length, prot, flags, fd, offset) };

    let result = if result != libc::MAP_FAILED { result as *mut u8 } else { ptr::null_mut() };
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
    let result = libc::munmap(addr as *mut libc::c_void, size);
    assert!(result == 0, "Could not munmap {:x}, {}: {}", addr as usize, size, result);
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
