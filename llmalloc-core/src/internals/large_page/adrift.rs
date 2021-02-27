//! The Adrift flag.

use core::sync::atomic::{AtomicU64, Ordering};

//  Adrift "boolean"
// 
//  A memory allocator needs to track pages that are empty, partially used, or completely used.
// 
//  Tracking the latter is really unfortunate, moving them and out of concurrent lists means contention with other
//  other threads, for pages that are completely unusable to fulfill allocation requests.
// 
//  Enters the Anchored/Adrift mechanism!
// 
//  Rather than keeping track of full pages within a special list, they are instead "cast adrift".
// 
//  The trick is to catch and anchor them back when the user deallocates memory, for unlike a leaked page, a page that
//  is cast adrift is still pointed to: by the user's allocations.
// 
//  Essentially, thus, the page is cast adrift when full, and caught back when "sufficiently" empty.
// 
//  To avoid ABA issues with a boolean, a counter is used instead:
//  -   An even value means "anchored".
//  -   An odd value means "adrift".
pub(crate) struct Adrift(AtomicU64);

impl Adrift {
    //  Creates an anchored instance.
    pub(crate) fn new() -> Self { Self(AtomicU64::new(0)) }

    //  Checks whether the value is adrift.
    //
    //  If adrift, returns the current value, otherwise returns None.
    pub(crate) fn is_adrift(&self) -> Option<u64> {
        let current = self.load();
        if current % 2 != 0 { Some(current) } else { None }
    }

    //  Casts the value adrift, incrementing the counter.
    //
    //  Returns the (new) current value.
    pub(crate) fn cast_adrift(&self) -> u64 {
        let before = self.0.fetch_add(1, Ordering::AcqRel);
        debug_assert!(before % 2 == 0, "before: {}", before);

        before + 1
    }

    //  Attempts to catch the value, returns true if it succeeds.
    pub(crate) fn catch(&self, current: u64) -> bool { 
        debug_assert!(current % 2 != 0);

        self.0.compare_exchange(current, current + 1, Ordering::AcqRel, Ordering::Relaxed).is_ok()
    }

    //  Internal: load.
    fn load(&self) -> u64 { self.0.load(Ordering::Acquire) }
}

impl Default for Adrift {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {

use std::sync::atomic::{AtomicU64, Ordering};

use llmalloc_test::BurstyBuilder;

use super::*;

#[test]
fn adrift() {
    let adrift = Adrift::default();
    assert_eq!(None, adrift.is_adrift());

    assert_eq!(1, adrift.cast_adrift());
    assert_eq!(Some(1), adrift.is_adrift());

    assert!(!adrift.catch(3));
    assert_eq!(Some(1), adrift.is_adrift());

    assert!(adrift.catch(1));
    assert_eq!(None, adrift.is_adrift());

    assert_eq!(3, adrift.cast_adrift());
    assert_eq!(Some(3), adrift.is_adrift());
}

#[test]
fn adrift_concurrent_catch_fuzzing() {
    //  This test aims at testing that a single thread can catch an adrift page.
    //
    //  To do so:
    //  -   Adrift is cast adrift.
    //  -   Each thread attempts to catch it, recording whether it did.
    //  -   A check is made that a single thread caught it.
    #[derive(Default)]
    struct Global {
        victim: Adrift,
        cast: AtomicU64,
        caught: [AtomicU64; 4],
    }

    impl Global {
        fn reset(&self, index: usize) {
            if index == 0 {
                if let Some(current) = self.victim.is_adrift() {
                    assert!(self.victim.catch(current));
                }
                self.cast.store(self.victim.cast_adrift(), Ordering::Relaxed);
            }

            self.caught[index].store(0, Ordering::Relaxed);
        }

        fn verify(&self) {
            let cast = self.cast.load(Ordering::Relaxed);

            let caught: Vec<_> = self.caught.iter()
                .map(|caught| caught.load(Ordering::Relaxed))
                .filter(|caught| *caught != 0)
                .collect();

            assert_eq!(vec!(cast), caught, "{:?}", self.caught);
        }
    }

    let mut builder = BurstyBuilder::new(Global::default(), vec!(0usize, 1, 2, 3));

    //  Step 1: reset.
    builder.add_simple_step(|| |global: &Global, local: &mut usize| {
        global.reset(*local);
    });

    //  Step 2: catch, if you can!
    builder.add_simple_step(|| |global: &Global, local: &mut usize| {
        let cast = global.cast.load(Ordering::Relaxed);

        if global.victim.catch(cast) {
            global.caught[*local].store(cast, Ordering::Relaxed);
        }
    });

    //  Step 3: verify one, and only one, thread caught it.
    builder.add_simple_step(|| |global: &Global, _: &mut usize| {
        global.verify();
    });

    builder.launch(100);
}

} // mod tests
