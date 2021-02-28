//! Foreign data, accessible both from the local thread and foreign threads.

use core::ptr::NonNull;

use crate::internals::blocks::{AtomicBlockForeignList, BlockForeignList, BlockLocal};

use super::{
    adrift::Adrift,
    local::Local,
};

//  Foreign data. Accessible both from the local thread and foreign threads, at the cost of synchronization.
#[repr(align(128))]
pub(crate) struct Foreign {
    //  List of cells returned by other threads.
    freed: AtomicBlockForeignList,
    //  The adrift marker.
    //
    //  A page that is too full is marked as "adrift", and cast away. As cells are freed -- as denoted by next.length
    //  -- the adrift page will be caught, ready to be used for allocations again.
    adrift: Adrift,
    //  When the number of freed cells exceeds this threshold, the page should be caught.
    catch_threshold: usize,
}

impl Foreign {
    /// Creates a new instance of `Foreign`.
    pub(crate) fn new(catch_threshold: usize) -> Self {
        debug_assert!(catch_threshold >= 1);

        let freed = AtomicBlockForeignList::default();
        let adrift = Adrift::default();

        Self { freed, adrift, catch_threshold, }
    }

    /// Attempts to refill `local` from foreign, and allocate.
    ///
    /// In case of failure, the page is cast adrift and null is returned.
    pub(crate) unsafe fn allocate(&self, local: &Local) -> Option<NonNull<u8>> {
        //  If the number of freed cells is sufficient to catch the page, immediately recycle them.
        if self.freed.len() >= self.catch_threshold {
            return self.recycle_allocate(local);
        }

        //  Cast the page adrift.
        let generation = self.adrift.cast_adrift();
        debug_assert!(generation % 2 != 0);

        //  There is an inherent race-condition, above, as a foreign thread may have extended the freed list beyond
        //  the catch threshold and yet seen a non-adrift page. The current thread therefore needs to check again.
        if self.freed.len() < self.catch_threshold {
            return None;
        }

        //  A race-condition DID occur! Let's attempt to catch the adrift page then!
        if self.adrift.catch(generation) {
            //  The page was successfully caught, it is the current thread's unique property again.
            self.recycle_allocate(local)
        } else {
            //  The page was caught by another thread, it can no longer be used by this thread.
            None
        }
    }

    /// Refills foreign.
    ///
    /// Returns true if the page was adrift and has been caught, false otherwise.
    ///
    /// #   Safety
    ///
    /// -   Assumes that the linked cells are not empty.
    /// -   Assumes that the linked cells actually belong to the page!
    pub(crate) unsafe fn refill(&self, list: &BlockForeignList) -> bool {
        //  Safety:
        //  -   `list` is assumed not be empty.
        let len = self.freed.extend(list);

        if len < self.catch_threshold {
            return false;
        }

        if let Some(adrift) = self.adrift.is_adrift() {
            self.adrift.catch(adrift)
        } else {
            false
        }
    }

    /// Returns the catch threshold of the page.
    #[cfg(test)]
    pub(crate) fn catch_threshold(&self) -> usize { self.catch_threshold }

    /// Returns the number of freed blocks.
    #[cfg(test)]
    pub(crate) fn freed(&self) -> usize { self.freed.len() }

    /// Returns whether the page is adrift (and if so, its generation).
    #[cfg(test)]
    pub(crate) fn is_adrift(&self) -> Option<u64> { self.adrift.is_adrift() }

    /// Attempts to the catch the adrift page, returns whether it succeeded.
    #[cfg(test)]
    pub(crate) fn catch_adrift(&self, generation: u64) -> bool { self.adrift.catch(generation) }

    //  Internal: Recycles the freed cells and immediately allocate.
    //
    //  #   Safety
    //
    //  -   Assumes that the current LargePage is owned by the current thread.
    unsafe fn recycle_allocate(&self, local: &Local) -> Option<NonNull<u8>> {
        debug_assert!(self.freed.len() >= self.catch_threshold);

        //  Safety:
        //  -   It is assumed that the current thread owns the LargePage, otherwise access to `local` is racy.
        let list = self.freed.steal()?;

        //  Safety:
        //  -   Access to `list` is exclusive.
        local.refill(BlockLocal::from_atomic(list));

        local.allocate()
    }
}

#[cfg(test)]
mod tests {

use std::{
    ops::Range,
    sync::atomic::{AtomicBool, AtomicU64, Ordering},
};

use llmalloc_test::BurstyBuilder;

use super::*;
use super::super::test::{BlockStore, BLOCK_SIZE};

#[test]
fn foreign_refill() {
    let block_store = BlockStore::default();
    let local = unsafe { block_store.create_local(BLOCK_SIZE) };

    //  Allocate all.
    while let Some(_) = local.allocate() {}

    let foreign = Foreign::new(16);

    //  Insufficient number of elements.
    let list = unsafe { block_store.create_foreign_list(&local, 3..7) };
    assert!(unsafe { !foreign.refill(&list) });

    //  Sufficient number, but not adrift.
    let list = unsafe { block_store.create_foreign_list(&local, 7..32) };
    assert!(unsafe { !foreign.refill(&list) });

    //  Number already sufficient, and now was adrift.
    foreign.adrift.cast_adrift();
    assert_eq!(Some(1), foreign.adrift.is_adrift());

    let list = unsafe { block_store.create_foreign_list(&local, 0..3) };
    assert!(unsafe { foreign.refill(&list) });
}

#[test]
fn foreign_allocate() {
    let block_store = BlockStore::default();
    let local = unsafe { block_store.create_local(BLOCK_SIZE) };

    //  Allocate all.
    while let Some(_) = local.allocate() {}

    let foreign = Foreign::new(16);

    //  Enough elements, immediate recycling.
    let list = unsafe { block_store.create_foreign_list(&local, 0..32) };
    assert!(unsafe { !foreign.refill(&list) });

    assert_eq!(block_store.get(0), unsafe { foreign.allocate(&local).unwrap() });

    //  Local was refilled with 32 elements (one already consumed above).
    for i in 1..32 {
        assert_eq!(block_store.get(4 * i), local.allocate().unwrap());
    }

    assert_eq!(None, local.allocate());

    //  Not enough elements, cast the page adrift.
    let list = unsafe { block_store.create_foreign_list(&local, 32..40) };
    assert!(unsafe { !foreign.refill(&list) });

    assert_eq!(None, unsafe { foreign.allocate(&local) });
    assert_eq!(Some(1), foreign.adrift.is_adrift());

    //  The race-condition cannot be tested in single-threaded code.
}

struct Global {
    victim: Foreign,
    adrift: AtomicU64,
    allocated: AtomicBool,
    caught: [AtomicBool; 4],
    local: Local,
    //  Need to be stable in memory.
    store: Box<BlockStore>,
}

impl Global {
    fn new(catch_threshold: usize) -> Self {
        let victim = Foreign::new(catch_threshold);
        let adrift = AtomicU64::new(0);
        let allocated = AtomicBool::new(false);
        let caught = Default::default();
        let store = Box::new(BlockStore::default());
        let local = unsafe { store.create_local(BLOCK_SIZE) };

        Self { victim, adrift, allocated, caught, local, store, }
    }

    fn reset(&self, thread: usize) {
        self.victim.freed.steal();
        self.allocated.store(false, Ordering::Relaxed);
        self.caught[thread].store(false, Ordering::Relaxed);

        if thread == 0 {
            if let Some(adrift) = self.victim.adrift.is_adrift() {
                self.victim.adrift.catch(adrift);
            }

            self.adrift.store(self.victim.adrift.value(), Ordering::Relaxed);
        }
    }

    fn exaust_local(&self, thread: usize) {
        if thread == 0 {
            while let Some(_) = self.local.allocate() {}
        }
    }

    fn cast_adrift(&self, thread: usize) {
        if thread == 0 {
            if self.victim.adrift.is_adrift().is_none() {
                self.victim.adrift.cast_adrift();
            }
        }
    }

    fn was_adrift(&self) -> bool {
        self.adrift.load(Ordering::Relaxed) != self.victim.adrift.value()
    }

    //  Creates a foreign list.
    //
    //  #   Safety
    //
    //  -   Assumes that `blocks` does not overlap with any live range...
    unsafe fn create_foreign_list(&self, blocks: Range<usize>) -> BlockForeignList {
        self.store.create_foreign_list(&self.local, blocks)
    }

    //  Verify that the number of freed blocks matches expectations.
    #[track_caller]
    fn verify_freed(&self, expected: usize) {
        assert_eq!(expected, self.victim.freed.len());

    }

    //  Verify that a single thread caught the page.
    #[track_caller]
    fn verify_catcher(&self) -> usize {
        let catchers: Vec<_> = self.caught.iter()
            .enumerate()
            .filter(|tup| tup.1.load(Ordering::Relaxed))
            .map(|tup| tup.0)
            .collect();

        assert_eq!(1, catchers.len(), "{:?}", catchers);

        catchers[0]
    }

    #[track_caller]
    fn verify_no_catcher(&self) {
        let caught: usize = self.caught.iter()
            .filter(|caught| caught.load(Ordering::Relaxed))
            .count();

        assert_eq!(0, caught, "{:?}", self.caught);
    }
}

//  Lying through my teeth, careful there.
unsafe impl Send for Global {}
unsafe impl Sync for Global {}

#[test]
fn foreign_concurrent_refill_uncaught_fuzzing() {
    //  This test aims at testing that refill can be called concurrently, without attempting to catch.
    //
    //  To do so:
    //  -   Reset the victim, not adrift.
    //  -   Each thread creates a list and calls refill.
    //      No thread catches the page -- it's not adrift.
    //  -   Verify that the number of blocks matches the total refilled.
    const NUMBER_BLOCKS: usize = 3;

    let mut builder = BurstyBuilder::new(Global::new(NUMBER_BLOCKS - 1), vec!(0, 1, 2, 3));

    //  Step 1: reset.
    builder.add_simple_step(|| |global: &Global, local: &mut usize| {
        global.reset(*local);
    });

    //  Step 2: refill.
    builder.add_complex_step(|| {
        let prep = |global: &Global, local: &mut usize| {
            let range = (*local * NUMBER_BLOCKS) .. ((*local + 1) * NUMBER_BLOCKS);
            //  Safety:
            //  -   Disjoint ranges.
            unsafe { global.create_foreign_list(range) } 
        };
        let step = |global: &Global, local: &mut usize, list: BlockForeignList| {
            let caught = unsafe { global.victim.refill(&list) };
            assert!(!caught, "Thread {}", *local);
        };

        (prep, step)
    });

    //  Step 3: verify.
    builder.add_simple_step(|| |global: &Global, _: &mut usize| {
        global.verify_freed(4 * NUMBER_BLOCKS);
    });

    builder.launch(100);
}

#[test]
fn foreign_concurrent_refill_caught_fuzzing() {
    //  This test aims at testing that refill can be called concurrently, and only one will succeed in catching.
    //
    //  To do so:
    //  -   Reset the victim, not adrift.
    //  -   Each thread creates a list and calls refill.
    //      No thread catches the page -- it's not adrift.
    //  -   Verify that the number of blocks matches the total refilled.
    const NUMBER_BLOCKS: usize = 3;

    //  Only the third will be able to catch.
    let mut builder = BurstyBuilder::new(Global::new(NUMBER_BLOCKS * 3 - 1), vec!(0, 1, 2, 3));

    //  Step 1: reset.
    builder.add_simple_step(|| |global: &Global, local: &mut usize| {
        global.reset(*local);
        global.cast_adrift(*local);
    });

    //  Step 2: refill.
    builder.add_complex_step(|| {
        let prep = |global: &Global, local: &mut usize| {
            let range = (*local * NUMBER_BLOCKS) .. ((*local + 1) * NUMBER_BLOCKS);
            //  Safety:
            //  -   Disjoint ranges.
            unsafe { global.create_foreign_list(range) } 
        };
        let step = |global: &Global, local: &mut usize, list: BlockForeignList| {
            let caught = unsafe { global.victim.refill(&list) };

            if caught {
                global.caught[*local].store(true, Ordering::Relaxed);
            }
        };

        (prep, step)
    });

    //  Step 3: verify.
    builder.add_simple_step(|| |global: &Global, _: &mut usize| {
        global.verify_freed(4 * NUMBER_BLOCKS);
        global.verify_catcher();
    });

    builder.launch(100);
}

#[test]
fn foreign_concurrent_allocate_refill_fuzzing() {
    //  This test aims at testing that allocate and refill can be called concurrently.
    //
    //  Depending on the order of execution, the scenario may go a few different ways:
    //
    //  -   The refills occur prior to allocate: Local is refilled, the allocation succeeds.
    //  -   Allocate occurs prior to the refill, casting the page adrift:
    //      -   The refills occur prior to the page being adrift, allocate will catch it, and the allocation will
    //          succeed.
    //      -   The refills occur after the page being adrift, one will catch it.
    //
    //  In the end, this means:
    //
    //  -   The page should NOT be adrift.
    //  -   The page may have been cast adrift at most once.
    //  -   If the page was cast adrift, a single thread should have caught it.
    //  -   If the local thread caught it, the allocation has succeeded.
    const NUMBER_BLOCKS: usize = 3;

    let mut builder = BurstyBuilder::new(Global::new(NUMBER_BLOCKS - 1), vec!(0, 1, 2, 3));

    //  Step 1: reset.
    builder.add_simple_step(|| |global: &Global, local: &mut usize| {
        global.reset(*local);
        global.exaust_local(*local);
    });

    //  Step 2: refill.
    builder.add_complex_step(|| {
        let prep = |global: &Global, local: &mut usize| {
            let range = (*local * NUMBER_BLOCKS) .. ((*local + 1) * NUMBER_BLOCKS);
            //  Safety:
            //  -   Disjoint ranges.
            unsafe { global.create_foreign_list(range) } 
        };
        let step = |global: &Global, local: &mut usize, list: BlockForeignList| {
            let caught = if *local == 0 {
                if let Some(_) = unsafe { global.victim.allocate(&global.local) } {
                    global.allocated.store(true, Ordering::Relaxed);
                    global.was_adrift()
                } else {
                    false
                }
            } else {
                unsafe { global.victim.refill(&list) }
            };

            if caught {
                global.caught[*local].store(caught, Ordering::Relaxed);
            }
        };

        (prep, step)
    });

    //  Step 3: verify.
    builder.add_simple_step(|| |global: &Global, _: &mut usize| {
        //  Not adrift.
        assert_eq!(None, global.victim.adrift.is_adrift(), "{:?}", global.victim.adrift.value());

        //  If it was adrift, it was caught once, otherwise, it wasn't caught.
        if global.was_adrift() {
            assert_eq!(global.adrift.load(Ordering::Relaxed) + 2, global.victim.adrift.value());

            let catcher = global.verify_catcher();

            //  If it was adrift _and_ the allocation succeed, then that catcher is 0.
            //
            //  (Otherwise, it would have allocated without setting it adrift)
            if global.allocated.load(Ordering::Relaxed) {
                assert_eq!(0, catcher);
            }
        } else {
            global.verify_no_catcher();

            //  If it was not set adrift, then the allocating succeeded.
            assert!(global.allocated.load(Ordering::Relaxed));
        }

        //  The number of freed elements should be a multiple of NUMBER_BLOCKS.
        let freed = global.victim.freed.len();
        assert_eq!(0, freed % NUMBER_BLOCKS, "{:?}", freed);

        //  If the allocation succeeded, there should be only 0, 1, or 2 * NUMBER_BLOCKS elements in global.freed,
        //  otherwise, there should be all 3.
        if global.allocated.load(Ordering::Relaxed) {
            let normalized = freed / NUMBER_BLOCKS;
            assert!([0, 1, 2].contains(&normalized), "{:?}", freed);
        } else {
            assert_eq!(3 * NUMBER_BLOCKS, freed);
        }
    });

    builder.launch(100);
}

} // mod tests
