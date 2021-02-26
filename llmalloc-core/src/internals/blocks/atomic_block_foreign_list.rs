//! Intrusive Linked List of AtomicBlockForeign.

use core::{
    ops::Deref,
    ptr::NonNull,
    sync::atomic::{self, Ordering},
};

use super::{AtomicPtr, AtomicBlockForeign, BlockForeignList};

/// AtomicBlockForeignList.
#[derive(Default)]
pub(crate) struct AtomicBlockForeignList(AtomicPtr<AtomicBlockForeign>);

impl AtomicBlockForeignList {
    /// Returns the length of the tail list.
    pub(crate) fn len(&self) -> usize {
        let head = self.0.load();

        head.map(|head| unsafe { head.as_ref() }.length.load() + 1)
            .unwrap_or(0)
    }

    /// Steals the content of the list.
    pub(crate) fn steal(&self) -> Option<NonNull<AtomicBlockForeign>> { self.0.exchange(None) }

    /// Extends the tail-list pointed to by prepending `list`, atomically.
    ///
    /// Returns the size of the new list.
    ///
    /// #   Safety
    ///
    /// -   Assumes the list is not empty.
    pub(crate) unsafe fn extend(&self, list: &BlockForeignList) -> usize {
        debug_assert!(!list.is_empty());

        let additional_length = list.len();

        //  Safety:
        //  -   The list is assumed not to be empty.
        let (head, tail) = list.steal();

        atomic::fence(Ordering::Release);

        //  Safety:
        //  -   Access to the list blocks is exclusive.
        //  -   A Release atomic fence was called after the last write to the `BlockForeign` list.
        let (head, tail) = (AtomicBlockForeign::from(head), AtomicBlockForeign::from(tail));

        let mut current = self.0.load();

        loop {
            let current_length = if let Some(current_ptr) = current {
                //  Safety:
                //  -   `current` is not null.
                current_ptr.as_ref().length.load() + 1
            } else {
                0
            };

            //  Safety:
            //  -   Bounded lifetime.
            tail.as_ref().next.store(current);
            tail.as_ref().length.store(current_length);

            //  Safety:
            //  -   Bounded lifetime.
            head.as_ref().length.store(current_length + additional_length - 1);

            //  Technical ABA.
            //
            //  This CAS exhibits an ABA issue:
            //  1.  Thread A reads `current` above, sets it as the tail, pauses.
            //  2.  Thread B steals `current`, then reintroduces it.
            //  3.  Thread A performs the CAS successfully, but returns the wrong length.
            //
            //  Practically speaking, though, due to AtomicBlockForeignList only being used in LargePage::Foreign, this
            //  is rather implausible.
            //
            //  Specifically, step (2) actually requires thread B, the owner of the LargePage, to recycle the blocks,
            //  allocate it, transfer it to another thread C, which deallocates it, overflows its own local collection,
            //  and returns it to thread B's LargePage::Foreign.
            //
            //  Furthermore, even if it _did_ happen, the list itself wouldn't be incorrect. Only its _length_ would be.
            match self.0.compare_exchange(current, Some(head)) {
                Ok(_) => return current_length + additional_length,
                Err(new_current) => current = new_current,
            }
        }
    }
}

impl Deref for AtomicBlockForeignList {
    type Target = AtomicPtr<AtomicBlockForeign>;

    fn deref(&self) -> &Self::Target { &self.0 }
}

#[cfg(test)]
mod tests {

use std::ops::Range;

use llmalloc_test::BurstyBuilder;

use super::*;
use super::super::{AlignedArray, BlockForeign, BlockForeignList, BlockLocal, BlockLocalStack};

fn create_list(array: &AlignedArray<BlockForeign>, range: Range<usize>) -> BlockForeignList {
    let list = BlockForeignList::default();

    for index in range.rev() {
        list.push(array.get(index));
    }

    list
}

#[test]
fn block_foreign_ptr_extend_steal() {
    let array = AlignedArray::<BlockForeign>::default();
    let (a, b, c) = (array.get(0), array.get(1), array.get(2));
    let (x, y, z) = (array.get(10), array.get(11), array.get(12));

    let list = BlockForeignList::default();

    list.push(c);
    list.push(b);
    list.push(a);

    let foreign = AtomicBlockForeignList::default();
    assert_eq!(0, foreign.len());

    assert_eq!(3, unsafe { foreign.extend(&list) });
    assert_eq!(3, foreign.len());

    list.push(z);
    list.push(y);
    list.push(x);

    assert_eq!(6, unsafe { foreign.extend(&list) });
    assert_eq!(6, foreign.len());

    let head = foreign.steal();
    assert_eq!(0, foreign.len());

    //  Double-check the list!
    let local = BlockLocalStack::default();
    unsafe { local.refill(BlockLocal::from_atomic(head.unwrap())) };

    assert_eq!(Some(x.cast()), local.pop());
    assert_eq!(Some(y.cast()), local.pop());
    assert_eq!(Some(z.cast()), local.pop());
    assert_eq!(Some(a.cast()), local.pop());
    assert_eq!(Some(b.cast()), local.pop());
    assert_eq!(Some(c.cast()), local.pop());
    assert_eq!(None, local.pop());
}

#[test]
fn atomic_block_foreign_list_concurrent_steal_fuzzing() {
    //  This test aims at validating that multiple threads can steal concurrently.
    //
    //  To do so:
    //  -   The list is prepared with a list.
    //  -   Each thread attempts to steal from it, the successful one signals it succeeded.
    //  -   The fact that one, and only one, thread succeeded is checked.
    struct Global {
        victim: AtomicBlockForeignList,
        pointer: NonNull<AtomicBlockForeign>,
        thieves: atomic::AtomicUsize,
    }

    unsafe impl Send for Global {}
    unsafe impl Sync for Global {}

    let array = AlignedArray::<BlockForeign>::default();

    let pointer = {
        let list = create_list(&array, 0..3);

        let head = unsafe { list.steal().0 };

        atomic::fence(Ordering::Release);

        unsafe { AtomicBlockForeign::from(head) }
    };

    let global = Global { victim: AtomicBlockForeignList::default(), pointer, thieves: atomic::AtomicUsize::default() };

    let mut builder = BurstyBuilder::new(global, vec!(true, false, false, false));

    //  Step 1: setup the list.
    builder.add_simple_step(|| |global: &Global, local: &mut bool| {
        if *local {
            global.victim.store(Some(global.pointer));
            global.thieves.store(0, Ordering::Relaxed);
        }
    });

    //  Step 2: attempt to steal from all threads.
    builder.add_simple_step(|| |global: &Global, _: &mut bool| {
        if let Some(stolen) = global.victim.steal() {
            assert_eq!(global.pointer, stolen);

            global.thieves.fetch_add(1, Ordering::Relaxed);
        }
    });

    //  Step 3: validate one, and only one, would-be-thief succeeded.
    builder.add_simple_step(|| |global: &Global, local: &mut bool| {
        if *local {
            global.victim.store(Some(global.pointer));
            global.thieves.store(0, Ordering::Relaxed);
        }
    });

    builder.launch(100);
}

#[test]
fn atomic_block_foreign_list_concurrent_extend_fuzzing() {
    //  This test aims at validating that multiple threads can extend concurrently.
    //
    //  To do so:
    //  -   Each thread creates a BlockForeignList, from a dedicated range.
    //  -   Each thread attempts at extending the AtomicBlockForeignList, storing the maximum length globally.
    //  -   The maximum length is checked.
    #[derive(Default)]
    struct Global {
        victim: AtomicBlockForeignList,
        maximum_length: atomic::AtomicUsize,
    }

    struct Local {
        array: NonNull<AlignedArray<BlockForeign>>,
        range: Range<usize>,
    }

    impl Local {
        fn new(array: &AlignedArray<BlockForeign>, range: Range<usize>) -> Self {
            Local { array: NonNull::from(array), range, }
        }
    }

    //  Safety:
    //  -   We are going to be very careful with what we do. Promise.
    unsafe impl Send for Local {}

    let array = AlignedArray::<BlockForeign>::default();

    let mut builder = BurstyBuilder::new(Global::default(),
        vec!(Local::new(&array, 0..3), Local::new(&array, 3..6), Local::new(&array, 6..9), Local::new(&array, 9..12)));

    //  Step 1: Reset.
    builder.add_simple_step(|| |global: &Global, _: &mut Local| {
        global.victim.store(None);
        global.maximum_length.store(0, Ordering::Relaxed);
    });

    //  Step 2: Prepare list and extend.
    builder.add_complex_step(|| {
        let prep = |_: &Global, local: &mut Local| {
            create_list(unsafe { local.array.as_ref() }, local.range.clone())
        };
        let step = |global: &Global, _: &mut Local, list: BlockForeignList| {
            let length = unsafe { global.victim.extend(&list) };
            global.maximum_length.fetch_max(length, Ordering::Relaxed);
        };
        (prep, step)
    });

    //  Step 3: Check maximum length.
    builder.add_simple_step(|| |global: &Global, _: &mut Local| {
        assert_eq!(12, global.maximum_length.load(Ordering::Relaxed));
    });

    builder.launch(100);
}

#[test]
fn atomic_block_foreign_list_concurrent_extend_steal_fuzzing() {
    //  This test aims at validating that multiple threads can extend _and_ steal concurrently.
    //
    //  To do so:
    //  -   Each thread is designed as either producer, or consumer, based on whether the range they are given is empty.
    //  -   Each producer thread attempts to extend the list, whilst each consumer thread attempts to steal from it.
    //      Additionally, consumer threads total how much they stole.
    //  -   The total of list-length and stolen is checked.
    #[derive(Default)]
    struct Global {
        victim: AtomicBlockForeignList,
        stolen: atomic::AtomicUsize,
    }

    struct Local {
        array: NonNull<AlignedArray<BlockForeign>>,
        range: Range<usize>,
    }

    impl Local {
        fn new(array: &AlignedArray<BlockForeign>, range: Range<usize>) -> Self {
            Local { array: NonNull::from(array), range, }
        }
    }

    //  Safety:
    //  -   We are going to be very careful with what we do. Promise.
    unsafe impl Send for Local {}

    let array = AlignedArray::<BlockForeign>::default();

    let mut builder = BurstyBuilder::new(Global::default(),
        vec!(Local::new(&array, 0..6), Local::new(&array, 6..6), Local::new(&array, 6..12), Local::new(&array, 12..12)));

    //  Step 1: Reset.
    builder.add_simple_step(|| |global: &Global, _: &mut Local| {
        global.victim.store(None);
        global.stolen.store(0, Ordering::Relaxed);
    });

    //  Step 2: Prepare list and extend.
    builder.add_complex_step(|| {
        let prep = |_: &Global, local: &mut Local| {
            create_list(unsafe { local.array.as_ref() }, local.range.clone())
        };
        let step = |global: &Global, _: &mut Local, list: BlockForeignList| {
            if list.is_empty() {
                //  Consumer
                if let Some(stolen) = global.victim.steal() {
                    let total = unsafe { stolen.as_ref().length.load() + 1 };
                    global.stolen.fetch_add(total, Ordering::Relaxed);
                }
            } else {
                //  Producer
                unsafe { global.victim.extend(&list) };
            }
        };
        (prep, step)
    });

    //  Step 3: Check maximum length.
    builder.add_simple_step(|| |global: &Global, _: &mut Local| {
        let remaining = global.victim.len();
        let stolen = global.stolen.load(Ordering::Relaxed);

        assert_eq!(12, remaining + stolen,
            "remaining: {}, stolen: {}", remaining, stolen);
    });

    builder.launch(100);
}

} // mod tests
