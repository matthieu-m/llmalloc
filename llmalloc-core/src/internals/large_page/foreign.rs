//! Foreign data, accessible both from the local thread and foreign threads.

use core::ptr::NonNull;

use crate::internals::{
    atomic::AtomicPtr,
    blocks::{AtomicBlockForeignList, BlockForeignList, BlockLocal},
};

use super::{
    adrift::Adrift,
    local::Local,
};

//  Foreign data. Accessible both from the local thread and foreign threads, at the cost of synchronization.
#[repr(align(128))]
pub(crate) struct Foreign {
    //  Pointer to the next LargePage, type-erased, for use by LargePageStack.
    pub(crate) next: AtomicPtr<()>,
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

        let next = AtomicPtr::default();
        let freed = AtomicBlockForeignList::default();
        let adrift = Adrift::default();

        Self { next, freed, adrift, catch_threshold, }
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

use super::*;
use super::super::test::{BlockStore, BLOCK_SIZE};

#[test]
fn foreign_refill() {
    let mut block_store = BlockStore::default();
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
    let mut block_store = BlockStore::default();
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

} // mod tests
