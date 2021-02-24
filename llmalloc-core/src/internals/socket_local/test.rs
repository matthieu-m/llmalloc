//! Test helpers for socket_local.

use core::{
    alloc::Layout,
    cell::Cell,
    mem,
    ptr::NonNull,
};

use crate::{Configuration, Platform, PowerOf2};
use super::{HugeAllocator, HugePagesManager};

/// Test Huge Allocator
pub(crate) type TestHugeAllocator = HugeAllocator<TestConfiguration, TestPlatform>;

//  Test Huge Pages Manager
pub(crate) type TestHugePagesManager = HugePagesManager<TestConfiguration, TestPlatform>;

/// Test configuration
pub(crate) struct TestConfiguration;

impl Configuration for TestConfiguration {
    const LARGE_PAGE_SIZE: PowerOf2 = unsafe { PowerOf2::new_unchecked(4 * 1024) };
    const HUGE_PAGE_SIZE: PowerOf2 = unsafe { PowerOf2::new_unchecked(8 * 1024) };
}

/// Test Platform
pub(crate) struct TestPlatform([Cell<Option<NonNull<u8>>>; 32]);

impl TestPlatform {
    pub(crate) const HUGE_PAGE_SIZE: usize = TestConfiguration::HUGE_PAGE_SIZE.value();

    pub(crate) unsafe fn new(store: &HugePageStore) -> TestPlatform {
        let stores: [Cell<Option<NonNull<u8>>>; 32] = Default::default();

        for (i, cell) in stores.iter().enumerate() {
            cell.set(NonNull::new(store.as_ptr().add(i * Self::HUGE_PAGE_SIZE)));
        }

        TestPlatform(stores)
    }

    //  Creates a TestHugeAllocator.
    pub(crate) unsafe fn allocator(store: &HugePageStore) -> TestHugeAllocator {
        TestHugeAllocator::new(Self::new(store))
    }

    //  Shrink the number of allocations to at most n.
    pub(crate) fn shrink(&self, n: usize) {
        for ptr in &self.0[n..] {
            ptr.set(None);
        }
    }

    //  Exhausts the HugePagesManager.
    pub(crate) fn exhaust(&self, manager: &TestHugePagesManager) {
        let owner = 0x1234 as *mut ();
        let platform = TestPlatform::default();

        loop {
            let large = unsafe { manager.allocate_large(LARGE_PAGE_LAYOUT, owner, &platform) };

            if large.is_none() { break; }
        }
    }

    //  Returns the number of allocated pages.
    pub(crate) fn allocated(&self) -> usize { self.0.len() - self.available() }

    //  Returns the number of available pages.
    pub(crate) fn available(&self) -> usize { self.0.iter().filter(|p| p.get().is_some()).count() }
}

impl Platform for TestPlatform {
    unsafe fn allocate(&self, layout: Layout) -> Option<NonNull<u8>> {
        assert_eq!(Self::HUGE_PAGE_SIZE, layout.size());
        assert_eq!(Self::HUGE_PAGE_SIZE, layout.align());

        for ptr in &self.0[..] {
            if ptr.get().is_none() {
                continue;
            }

            return ptr.replace(None);
        }

        None
    }

    unsafe fn deallocate(&self, pointer: NonNull<u8>, layout: Layout) {
        assert_eq!(Self::HUGE_PAGE_SIZE, layout.size());
        assert_eq!(Self::HUGE_PAGE_SIZE, layout.align());

        for ptr in &self.0[..] {
            if ptr.get().is_some() {
                continue;
            }

            ptr.set(Some(pointer));
            return;
        }
    }
}

impl Default for TestPlatform {
    fn default() -> Self { unsafe { mem::zeroed() } }
}

/// Store of Huge Pages.
pub(crate) struct HugePageStore(Vec<HugePageCell>); // 128K

impl HugePageStore {
    pub(crate) fn as_ptr(&self) -> *mut u8 { self.0.as_ptr() as *const u8 as *mut _ }
}

impl Default for HugePageStore {
    fn default() -> Self {
        let mut vec = vec!();
        //  256K worth of memory
        vec.resize(256 * 1024 / mem::size_of::<HugePageCell>(), HugePageCell::default());

        Self(vec)
    }
}

//
//  Implementation
//

const LARGE_PAGE_SIZE: usize = TestConfiguration::LARGE_PAGE_SIZE.value();
const LARGE_PAGE_LAYOUT: Layout = unsafe { Layout::from_size_align_unchecked(LARGE_PAGE_SIZE, LARGE_PAGE_SIZE) };

#[repr(align(8192))]
#[derive(Clone, Default)]
struct HugePageCell(u8);
