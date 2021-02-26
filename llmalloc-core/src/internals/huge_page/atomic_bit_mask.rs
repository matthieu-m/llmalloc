//! An atomic bit mask representing the occupation (or not) of a block.

use core::sync::atomic::{AtomicU64, Ordering};

use crate::PowerOf2;

pub(crate) struct AtomicBitMask(AtomicU64);

impl AtomicBitMask {
    //  Safety:
    //  -   64 is a power of 2.
    pub(crate) const CAPACITY: PowerOf2 = unsafe { PowerOf2::new_unchecked(64) };

    /// Initializes the AtomicBitMask with the given mask.
    pub(crate) fn initialize(&self, mask: u64) { self.0.store(mask, Ordering::Relaxed); }

    /// Claims a 0 bit, returns its index or None if all bits are claimed.
    pub(crate) fn claim_single(&self) -> Option<usize> {
        loop {
            //  Locate first 0 bit.
            let current_mask = self.0.load(Ordering::Acquire);
            let candidate = (!current_mask).trailing_zeros() as usize;

            //  All bits are ones, move on.
            if candidate == Self::capacity() { return None; }

            let candidate_mask = 1u64 << candidate;
            let before = self.0.fetch_or(candidate_mask, Ordering::AcqRel);

            //  This thread claimed the bit first, victory!
            if before & candidate_mask == 0 {
                return Some(candidate);
            }
        }
    }

    /// Claims up to `number` bits, returns the index of the lowest claimed and the number of bits claimed, or None.
    ///
    /// This functions scans the bits from highest to lowest.
    ///
    /// If the number of returned bits is less than `number`, then the index is 0, allowing extending the allocation
    /// to the correct number by claiming bits from the previous AtomicBitMask.
    pub(crate) fn claim_multiple(&self, number: usize, align: PowerOf2) -> Option<(usize, usize)> {
        //  Do a single pass over the bits, attempting to find `number` consecutive ones.
        let mut progress_mask = 0;

        loop {
            //  Locate last 0 bit.
            let current_mask = self.0.load(Ordering::Acquire) | progress_mask;

            //  Potential number of available bits.
            let potential = Self::capacity() - (!current_mask).leading_zeros() as usize;

            //  Not enough potential bits available, try to claim as many low-bits as possible.
            if potential < number { break; }

            //  Highest lower index which allows `number` bits.
            let candidate = align.round_down(potential - number);

            let candidate_mask = Self::low(number) << candidate;

            //  Optimize the claim attempt by not even attempting it if the mask wouldn't fit.
            let is_free = current_mask & candidate_mask == 0;

            if is_free && self.claim(candidate_mask) {
                return Some((candidate, number));
            }

            //  Force searches to discard previously checked bits.
            //
            //  Otherwise, it could enter an infinite loop attempting to claim n bits when only 1 is available.
            progress_mask = Self::high(Self::capacity() - potential + 1);
        }

        //  Claim as many low-bits as possible, as long as they are aligned.
        {
            let current_mask = self.0.load(Ordering::Acquire);

            let potential = align.round_down(current_mask.trailing_zeros() as usize);

            if potential == 0 {
                return None;
            }

            let candidate_mask = Self::low(potential);

            if self.claim(candidate_mask) {
                Some((0, potential))
            } else {
                None
            }
        }
    }

    /// Claims the exact number of bits at the exact index.
    ///
    /// Returns true on success, false on failure.
    pub(crate) fn claim_at(&self, inner: usize, number: usize) -> bool {
        let mask = Self::low(number) << inner;

        self.claim(mask)
    }

    /// Releases bit at given index.
    pub(crate) fn release_single(&self, inner: usize) {
        debug_assert!(inner < Self::capacity());

        let inner_mask = 1u64 << inner;

        self.release(inner_mask);
    }

    /// Releases `number` bits starting at the given index.
    pub(crate) fn release_multiple(&self, inner: usize, number: usize) {
        debug_assert!(inner + number <= Self::capacity());

        let mask = AtomicBitMask::low(number) << inner;

        self.release(mask);
    }

    /// Returns the capacity as usize.
    pub(crate) fn capacity() -> usize { Self::CAPACITY.value() }

    //  Internal: Claims the bits, returns true on success, false on failure.
    //
    //  On failure, unclaims bits that were erroneously claimed.
    fn claim(&self, mask: u64) -> bool {
        let before = self.0.fetch_or(mask, Ordering::AcqRel);

        //  Success!
        if before & mask == 0 {
            return true;
        } 

        //  Not all bits were claimed before, release the ones that were not.
        //
        //  Example:
        //  -   mask:           0111.
        //  -   before:         1010.
        //  -   before & mask:  0010.
        //  -   !before & mask: 0101.
        if before & mask != mask {
            self.release(!before & mask)
        }

        false
    }

    //  Internal: Releases the bits.
    fn release(&self, mask: u64) {
        let _before = self.0.fetch_and(!mask, Ordering::AcqRel);
        debug_assert!(_before & mask == mask,
            "before: {:b}, mask: {:b}, before & mask: {:b}", _before, mask, _before & mask);
    }

    //  Internal: Computes a mask with the `number` high bits set, and all others unset.
    fn high(number: usize) -> u64 {
        debug_assert!(number <= Self::capacity());

        !Self::low(Self::capacity() - number)
    }

    //  Internal: Computes a mask with the `number` low bits set, and all others unset.
    fn low(number: usize) -> u64 {
        debug_assert!(number <= Self::capacity());

        if number == 64 {
            u64::MAX
        } else {
            (1u64 << number) - 1
        }
    }
}

#[cfg(test)]
impl Clone for AtomicBitMask {
    fn clone(&self) -> Self {
        let result = AtomicBitMask::default();
        result.initialize(self.0.load(Ordering::Relaxed));
        result
    }
}

impl Default for AtomicBitMask {
    fn default() -> Self { Self(AtomicU64::new(0)) }
}

#[cfg(test)]
pub(crate) fn load_bitmask(bitmask: &AtomicBitMask) -> u64 { bitmask.0.load(Ordering::Relaxed) }

#[cfg(test)]
mod tests {

use std::ops::Range;

use llmalloc_test::BurstyBuilder;

use super::*;

#[test]
fn atomic_bitmask_initialize() {
    fn initialize(mask: u64) -> u64 {
        let bitmask = AtomicBitMask::default();
        bitmask.initialize(mask);
        load_bitmask(&bitmask)
    }

    assert_eq!(0, initialize(0));
    assert_eq!(3, initialize(3));
    assert_eq!(42, initialize(42));
    assert_eq!(u64::MAX, initialize(u64::MAX));
}

#[test]
fn atomic_bitmask_claim_single() {
    //  Initializes bitmask with `mask`, call claim_single, return single + new state of bitmask
    fn claim_single(mask: u64) -> (Option<usize>, u64) {
        let bitmask = AtomicBitMask::default();
        bitmask.initialize(mask);

        let claimed = bitmask.claim_single();

        (claimed, load_bitmask(&bitmask))
    }

    assert_eq!((None, u64::MAX), claim_single(u64::MAX));

    //  Check the next is claimed with a contigous pattern starting from beginning
    for i in 0..63 {
        let before = (1u64 << i) - 1;
        let after = (1u64 << (i + 1)) - 1;

        assert_eq!((Some(i), after), claim_single(before));
    }

    //  Check the one available bit is claimed if a single is available.
    for i in 0..64 {
        let mask = u64::MAX - (1u64 << i);

        assert_eq!((Some(i), u64::MAX), claim_single(mask));
    }
}

#[test]
fn atomic_bitmask_claim_at() {
    //  Initializes bitmask with `mask`, call claim_at with `index` and `number`, return result + new state of bitmask
    fn claim_at(mask: u64, index: usize, number: usize) -> (bool, u64) {
        let bitmask = AtomicBitMask::default();
        bitmask.initialize(mask);

        let result = bitmask.claim_at(index, number);

        (result, load_bitmask(&bitmask))
    }

    for i in 0..64 {
        assert_eq!((true, 1u64 << i), claim_at(0, i, 1));
    }

    for i in 0..64 {
        assert_eq!((false, 1u64 << i), claim_at(1u64 << i, i, 1));
    }

    assert_eq!((true, u64::MAX), claim_at(0, 0, 64));
    assert_eq!((false, 1), claim_at(1, 0, 64));
}

//  Initializes bitmask with `mask`, call claim_multiple with `number`, return result + new state of bitmask
fn claim_multiple(mask: u64, number: usize) -> (Option<usize>, Option<usize>, u64) {
    let bitmask = AtomicBitMask::default();
    bitmask.initialize(mask);

    if let Some(claimed) = bitmask.claim_multiple(number, PowerOf2::ONE) {
        (Some(claimed.0), Some(claimed.1), load_bitmask(&bitmask))
    } else {
        (None, None, load_bitmask(&bitmask))
    }
}

#[test]
fn atomic_bitmask_claim_multiple_none() {
    assert_eq!((None, None, u64::MAX), claim_multiple(u64::MAX, 1));

    for i in 1..63 {
        let mask = AtomicBitMask::low(i);
        assert_eq!((None, None, mask), claim_multiple(mask, 65 - i));
    }
}

#[test]
fn atomic_bitmask_claim_multiple_complete() {
    //  Claim multiple always starts from the _high_ bits.
    for number in 1..=64 {
        let after = AtomicBitMask::low(number) << (64 - number);
        assert_eq!((Some(64 - number), Some(number), after), claim_multiple(0, number));
    }
}

#[test]
fn atomic_bitmask_claim_multiple_partial() {
    //  Claim multiple should be able to claim N (< number) low bits.
    for number in 1..=63 {
        let before = AtomicBitMask::high(number);
        assert_eq!((Some(0), Some(64 - number), u64::MAX), claim_multiple(before, 64));
    }
}

#[test]
fn atomic_bitmask_claim_multiple_pockmarked() {
    let mask = {
        AtomicBitMask::low(1) << 63 |
        // 7 0-bits
        AtomicBitMask::low(8) << 48 |
        // 8 0-bits
        AtomicBitMask::low(8) << 32 |
        // 9 0-bits
        AtomicBitMask::low(7) << 16 |
        // 10 0-bits
        AtomicBitMask::low(5) << 1
        // 1 0-bits
    };

    let after = mask | AtomicBitMask::low(7) << 56;
    assert_eq!((Some(56), Some(7), after), claim_multiple(mask, 7));

    let after = mask | AtomicBitMask::low(8) << 40;
    assert_eq!((Some(40), Some(8), after), claim_multiple(mask, 8));

    let after = mask | AtomicBitMask::low(9) << 23;
    assert_eq!((Some(23), Some(9), after), claim_multiple(mask, 9));

    let after = mask | AtomicBitMask::low(10) << 6;
    assert_eq!((Some(6), Some(10), after), claim_multiple(mask, 10));

    let after = mask | AtomicBitMask::low(1);
    assert_eq!((Some(0), Some(1), after), claim_multiple(mask, 11));
}

//  Initializes bitmask with `mask`, call claim_multiple with `number`, return result + new state of bitmask
fn claim_multiple_aligned(mask: u64, number: usize, align: usize) -> (Option<usize>, Option<usize>, u64) {
    let bitmask = AtomicBitMask::default();
    bitmask.initialize(mask);

    if let Some(claimed) = bitmask.claim_multiple(number, PowerOf2::new(align).expect("Power of 2")) {
        (Some(claimed.0), Some(claimed.1), load_bitmask(&bitmask))
    } else {
        (None, None, load_bitmask(&bitmask))
    }
}

#[test]
fn atomic_bitmask_claim_multiple_aligned_none() {
    let tests = [
        ( 2, 0b10011001_10011001_10011001_10011001_10011001_10011001_10011001_10011001u64),
        ( 4, 0b10000001_10000001_10000001_10000001_10000001_10000001_10000001_10000001u64),
        ( 8, 0b10000000_00000001_10000000_00000001_10000000_00000001_10000000_00000001u64),
        (16, 0b10000000_00000000_00000000_00000001_10000000_00000000_00000000_00000001u64),
        (32, 0b10000000_00000000_00000000_00000000_00000000_00000000_00000000_00000001u64),
        (64, AtomicBitMask::high(1)),
    ];

    for &(number, mask) in &tests {

        assert_eq!((None, None, mask), claim_multiple_aligned(mask, number, number),
            "mask: {:b}, number: {}", mask, number);
    }
}

#[test]
fn atomic_bitmask_claim_multiple_aligned_pockmarked() {
    let mask = {
        AtomicBitMask::low(1) << 63 |
        //  2 0-bits
        AtomicBitMask::low(1) << 60 |
        AtomicBitMask::low(1) << 59 |
        //  4 0-bits
        AtomicBitMask::low(1) << 54 |
        //  8 0-bits
        AtomicBitMask::low(1) << 45 |
        //  16 0-bits
        AtomicBitMask::low(1) << 28
        //  29 0-bits
    };

    let after = mask | AtomicBitMask::low(2) << 56;
    assert_eq!((Some(56), Some(2), after), claim_multiple_aligned(mask, 2, 2));

    let after = mask | AtomicBitMask::low(4) << 48;
    assert_eq!((Some(48), Some(4), after), claim_multiple_aligned(mask, 4, 4));

    let after = mask | AtomicBitMask::low(8) << 32;
    assert_eq!((Some(32), Some(8), after), claim_multiple_aligned(mask, 8, 8));

    let after = mask | AtomicBitMask::low(16);
    assert_eq!((Some(0), Some(16), after), claim_multiple_aligned(mask, 16, 16));
}

#[test]
fn atomic_bitmask_release_single() {
    //  Initializes bitmask with `mask`, call release_single with `index`, return new state of bitmask
    fn release_single(mask: u64, index: usize) -> u64 {
        let bitmask = AtomicBitMask::default();
        bitmask.initialize(mask);

        bitmask.release_single(index);

        load_bitmask(&bitmask)
    }

    for i in 0..64 {
        assert_eq!(0, release_single(1u64 << i, i));
    }

    for i in 0..64 {
        assert_eq!(u64::MAX - (1u64 << i), release_single(u64::MAX, i));
    }
}

#[test]
fn atomic_bitmask_release_multiple() {
    //  Initializes bitmask with `mask`, call release_multiple with `index` and `number`, return new state of bitmask
    fn release_multiple(mask: u64, index: usize, number: usize) -> u64 {
        let bitmask = AtomicBitMask::default();
        bitmask.initialize(mask);

        bitmask.release_multiple(index, number);

        load_bitmask(&bitmask)
    }

    assert_eq!(0, release_multiple(u64::MAX, 0, 64));

    for i in 0..64 {
        assert_eq!(0, release_multiple(1u64 << i, i, 1));
    }

    for i in 0..64 {
        assert_eq!(u64::MAX - (1u64 << i), release_multiple(u64::MAX, i, 1));
    }
}

#[derive(Default)]
struct Global {
    victim: AtomicBitMask,
    claimed: [AtomicU64; 4],
    released: [AtomicU64; 4],
}

impl Global {
    fn mask(&self) -> u64 { self.victim.0.load(Ordering::Relaxed) }

    fn claimed(&self) -> impl Iterator<Item = u64> + '_ {
        self.claimed.iter()
            .map(|claimed| claimed.load(Ordering::Relaxed))
    }

    fn released(&self) -> impl Iterator<Item = u64> + '_ {
        self.released.iter()
            .map(|released| released.load(Ordering::Relaxed))
    }

    fn claimed_mask(&self) -> u64 {
        self.claimed().sum()
    }

    fn released_mask(&self) -> u64 {
        self.released().sum()
    }

    fn claimed_exact(&self) -> Vec<u64> {
        let mut claimed: Vec<_> = self.claimed()
            .filter(|claimed| *claimed != 0)
            .collect();
        claimed.sort();
        claimed
    }

    fn released_exact(&self) -> Vec<u64> {
        let mut released: Vec<_> = self.released()
            .filter(|released| *released != 0)
            .collect();
        released.sort();
        released
    }

    fn print_claims(&self) {
        for (index, claimed) in self.claimed.iter().enumerate() {
            let claimed = claimed.load(Ordering::Relaxed);
            println!("index: {}, claimed: {:?} ({:b})", index, Self::as_range(claimed), claimed);
        }
    }

    fn as_range(claimed: u64) -> Range<usize> {
        let start = claimed.trailing_zeros() as usize;
        let end = 64 - claimed.leading_zeros() as usize;
        let number = claimed.count_ones() as usize;

        assert_eq!(number, end - start, "claimed: {:b}, number: {}, start: {}, end: {}", claimed, number, start, end);

        start..end
    }
}

#[test]
fn atomic_bit_mask_concurrent_claim_single_success_fuzzing() {
    //  This test aims at validating that multiple threads can call claim_single concurrently.
    //
    //  To do so:
    //  -   The mask is reset to 0.
    //  -   Each thread claims, and register its claim in Global state.
    //  -   A check is made that the claims are as expected.
    let mut builder = BurstyBuilder::new(Global::default(), vec!(0usize, 1, 2, 3));

    //  Step 1: Reset.
    builder.add_simple_step(|| |global: &Global, index: &mut usize| {
        global.victim.initialize(0);
        global.claimed[*index].store(0, Ordering::Relaxed);
    });

    //  Step 2: Claim.
    builder.add_simple_step(|| |global: &Global, index: &mut usize| {
        let claimed = global.victim.claim_single().unwrap();
        global.claimed[*index].store(1u64 << claimed, Ordering::Relaxed);
    });

    //  Step 3: Verify claims.
    builder.add_simple_step(|| |global: &Global, _: &mut usize| {
        assert_eq!(vec!(1, 2, 4, 8), global.claimed_exact());

        //  The successes should match the victim.
        let claimed = global.claimed_mask();
        let actual = global.mask();
        assert_eq!(claimed, actual, "claimed: {:b}, actual: {:b}", claimed, actual);
    });

    builder.launch(100);
}

#[test]
fn atomic_bit_mask_concurrent_claim_single_failure_fuzzing() {
    //  This test aims at validating that multiple threads can call claim_single concurrently.
    //
    //  To do so:
    //  -   The mask is reset to a pattern with only bits 18 and 52 non-claimed.
    //  -   Each thread attempts to claims, and register its claim in Global state.
    //  -   A check is made that the claims are as expected: 2 success (18 and 52) and 2 failures (MAX).
    const INITIAL: u64 = 0b11111111_11101111_11111111_11111111_11111111_11111011_11111111_11111111u64;

    let mut builder = BurstyBuilder::new(Global::default(), vec!(0usize, 1, 2, 3));

    //  Step 1: Reset.
    builder.add_simple_step(|| |global: &Global, index: &mut usize| {
        global.victim.initialize(INITIAL);
        global.claimed[*index].store(0, Ordering::Relaxed);
    });

    //  Step 2: Claim.
    builder.add_simple_step(|| |global: &Global, index: &mut usize| {
        if let Some(claimed) = global.victim.claim_single() {
            global.claimed[*index].store(1u64 << claimed, Ordering::Relaxed);
        }
    });

    //  Step 3: Verify claims.
    builder.add_simple_step(|| |global: &Global, _: &mut usize| {
        assert_eq!(vec!(1u64 << 18, 1u64 << 52), global.claimed_exact());
        
        //  The successes should match the victim.
        let claimed = global.claimed_mask();
        let actual = global.mask();
        assert_eq!(INITIAL | claimed, actual,
            "initial: {:b}, claimed: {:b}, actual: {:b}", INITIAL, claimed, actual);
    });

    builder.launch(100);
}

#[test]
fn atomic_bit_mask_concurrent_independent_claim_single_release_fuzzing() {
    //  This test aims at validating that multiple threads can call claim_single and release(_single/multi) concurrently.
    //
    //  To do so:
    //  -   The mask is reset to a specific pattern, with a few free low bits.
    //  -   2 threads attempt to claim, while 2 threads release what they have claimed.
    //      Due to the releases happening at _high_ bits, there is no interference.
    //  -   A check is made that the claims are as expected.
    const INITIAL: u64 = 0b11111111_11111111_11111111_11111111_11111111_00000000_11111111_11101111u64;
    
    let mut builder = BurstyBuilder::new(Global::default(), vec!(0usize, 1, 2, 3));

    //  Step 1: Reset.
    builder.add_simple_step(|| |global: &Global, index: &mut usize| {
        global.victim.initialize(INITIAL);
        global.claimed[*index].store(0, Ordering::Relaxed);
        global.released[*index].store(0, Ordering::Relaxed);
    });

    //  Step 2: Claim or release.
    builder.add_simple_step(|| |global: &Global, index: &mut usize| {
        if *index % 2 == 0 {
            if let Some(claimed) = global.victim.claim_single() {
                global.claimed[*index].store(1u64 << claimed, Ordering::Relaxed);
            }
        } else {
            //  Release bits 63 and 61.
            let released = 64 - *index;

            global.victim.release_single(released);
            global.released[*index].store(1u64 << released, Ordering::Relaxed);
        }
    });

    //  Step 3: Verify claims.
    builder.add_simple_step(|| |global: &Global, _: &mut usize| {
        let released = global.released_exact();
        assert_eq!(vec!(1u64 << 61, 1u64 << 63), released);

        let claimed = global.claimed_exact();
        assert_eq!(vec!(1u64 << 4, 1u64 << 16), claimed);

        //  The successes should match the victim.
        let claimed = global.claimed_mask();
        let released = global.released_mask();

        let actual = global.mask();
        assert_eq!((INITIAL & !released) | claimed, actual,
            "initial: {:b}, released: {:b}, claimed: {:b}, actual: {:b}", INITIAL, released, claimed, actual);
    });

    builder.launch(100);
}

#[test]
fn atomic_bit_mask_concurrent_overlapping_claim_single_release_fuzzing() {
    //  This test aims at validating that multiple threads can call claim_single and release(_single/multi) concurrently.
    //
    //  To do so:
    //  -   The mask is reset to a specific pattern, with a single (52) free bit.
    //  -   2 threads attempt to claim, while 2 threads release what they have claimed.
    //      Either 1 or 2 claims are made, depending on whether the second success manages to recycle a released item.
    //  -   A check is made that the claims are as expected.
    const INITIAL: u64 = 0b11111111_11101111_11111111_11111111_11111111_11111111_11111111_11111111u64;
    
    let mut builder = BurstyBuilder::new(Global::default(), vec!(0usize, 1, 2, 3));

    //  Step 1: Reset.
    builder.add_simple_step(|| |global: &Global, index: &mut usize| {
        global.victim.initialize(INITIAL);
        global.claimed[*index].store(0, Ordering::Relaxed);
        global.released[*index].store(0, Ordering::Relaxed);
    });

    //  Step 2: Claim or release.
    builder.add_simple_step(|| |global: &Global, index: &mut usize| {
        if *index % 2 == 0 {
            if let Some(claimed) = global.victim.claim_single() {
                global.claimed[*index].store(1u64 << claimed, Ordering::Relaxed);
            }
        } else {
            //  Release bits 63 and 61.
            let released = 64 - *index;

            global.victim.release_single(released);
            global.released[*index].store(1u64 << released, Ordering::Relaxed);
        }
    });

    //  Step 3: Verify claims.
    builder.add_simple_step(|| |global: &Global, _: &mut usize| {
        let released = global.released_exact();
        assert_eq!(vec!(1u64 << 61, 1u64 << 63), released);

        let claimed = global.claimed_exact();
        assert!(claimed.contains(&(1u64 << 52)), "{:?}", claimed);

        if claimed.len() > 1 {
            assert_eq!(2, claimed.len());
            assert!(claimed.contains(&released[0]) || claimed.contains(&released[1]), "{:?}", claimed);
        }

        //  The successes should match the victim.
        let claimed = global.claimed_mask();
        let released = global.released_mask();

        let actual = global.mask();
        assert_eq!((INITIAL & !released) | claimed, actual,
            "initial: {:b}, released: {:b}, claimed: {:b}, actual: {:b}", INITIAL, released, claimed, actual);
    });

    builder.launch(100);
}

struct LocalAt {
    index: usize,
    range: Range<usize>,
}

impl LocalAt {
    fn new(index: usize, range: Range<usize>) -> Self { Self { index, range, } }

    fn at(&self) -> usize { self.range.start }

    fn number(&self) -> usize { self.range.end - self.range.start }

    fn mask(&self) -> u64 { AtomicBitMask::low(self.number()) << self.at() }
}

#[test]
fn atomic_bit_mask_concurrent_claim_at_success_fuzzing() {
    //  This test aims at validating that multiple threads can call claim_at concurrently.
    //
    //  To do so:
    //  -   The mask is reset to 0.
    //  -   Each thread claims a range, and registers the claimed range.
    //  -   A check is made that the claims are as expected.
    let mut builder = BurstyBuilder::new(Global::default(),
        vec!(LocalAt::new(0, 0..3), LocalAt::new(1, 3..6), LocalAt::new(2, 10..13), LocalAt::new(3, 13..16)));

    //  Step 1: Reset.
    builder.add_simple_step(|| |global: &Global, local: &mut LocalAt| {
        global.victim.initialize(0);
        global.claimed[local.index].store(0, Ordering::Relaxed);
    });

    //  Step 2: Claim.
    builder.add_simple_step(|| |global: &Global, local: &mut LocalAt| {
        if global.victim.claim_at(local.at(), local.number()) {
            global.claimed[local.index].store(local.mask(), Ordering::Relaxed);
        }
    });

    //  Step 3: Verify claims.
    builder.add_simple_step(|| |global: &Global, _: &mut LocalAt| {
        let claimed = global.claimed_mask();

        //  The success should match the expectations.
        assert_eq!((AtomicBitMask::low(6) << 10) | AtomicBitMask::low(6), claimed);

        //  The successes should match the victim.
        let actual = global.mask();
        assert_eq!(claimed, actual, "claimed: {:b}, actual: {:b}", claimed, actual);
    });

    builder.launch(100);
}

#[test]
fn atomic_bit_mask_concurrent_claim_at_failure_fuzzing() {
    //  This test aims at validating that multiple threads can call claim_at concurrently.
    //
    //  To do so:
    //  -   The mask is reset to 0.
    //  -   Each thread attempts to claim a range, and registers the claimed range.
    //      At least 1 will succeed, and perhaps a 2nd.
    //  -   A check is made that the claims are as expected.
    let mut builder = BurstyBuilder::new(Global::default(),
        vec!(LocalAt::new(0, 0..4), LocalAt::new(1, 3..7), LocalAt::new(2, 6..10), LocalAt::new(3, 9..13)));

    //  Step 1: Reset.
    builder.add_simple_step(|| |global: &Global, local: &mut LocalAt| {
        global.victim.initialize(0);
        global.claimed[local.index].store(0, Ordering::Relaxed);
    });

    //  Step 2: Claim.
    builder.add_simple_step(|| |global: &Global, local: &mut LocalAt| {
        if global.victim.claim_at(local.at(), local.number()) {
            global.claimed[local.index].store(local.mask(), Ordering::Relaxed);
        }
    });

    //  Step 3: Verify claims.
    builder.add_simple_step(|| |global: &Global, _: &mut LocalAt| {
        let claimed = global.claimed_mask();

        //  At least one should succeed.
        assert!(claimed.count_ones() >= 4, "{:b}", claimed);

        //  At most two should succeed.
        assert!(claimed.count_ones() <= 8, "{:b}", claimed);

        //  The successes should match the victim.
        let actual = global.mask();
        assert_eq!(claimed, actual, "claimed: {:b}, actual: {:b}", claimed, actual);
    });

    builder.launch(100);
}

#[test]
fn atomic_bit_mask_concurrent_independent_claim_at_release_fuzzing() {
    //  This test aims at validating that multiple threads can call claim_at and release concurrently.
    //
    //  To do so:
    //  -   The mask is set to a specific pattern.
    //  -   Two threads attempt to claim at specific spots, whilst the other 2 release at specific spots.
    //      As the spots are independent, all should succeed.
    //  -   A check is made that the claims are as expected.
    const INITIAL: u64 = 0b00000000_11111111_00000000_11111111_00000000_11111111_11100000_00111000u64;

    let mut builder = BurstyBuilder::new(Global::default(),
        vec!(LocalAt::new(0, 0..3), LocalAt::new(1, 3..6), LocalAt::new(2, 10..13), LocalAt::new(3, 13..16)));

    //  Step 1: Reset.
    builder.add_simple_step(|| |global: &Global, local: &mut LocalAt| {
        global.victim.initialize(INITIAL);
        global.claimed[local.index].store(0, Ordering::Relaxed);
        global.released[local.index].store(0, Ordering::Relaxed);
    });

    //  Step 2: Claim or release.
    builder.add_simple_step(|| |global: &Global, local: &mut LocalAt| {
        if local.index % 2 == 0 {
            if global.victim.claim_at(local.at(), local.number()) {
                global.claimed[local.index].store(local.mask(), Ordering::Relaxed);
            }
        } else {
            global.victim.release_multiple(local.at(), local.number());
            global.released[local.index].store(local.mask(), Ordering::Relaxed);
        }
    });

    //  Step 3: Verify claims.
    builder.add_simple_step(|| |global: &Global, local: &mut LocalAt| {
        let released = global.released_exact();
        let claimed = global.claimed_exact();

        if local.index % 2 == 0 {
            assert!(claimed.contains(&local.mask()), "{:?} should contain {:b} ({})", claimed, local.mask(), local.mask());
        } else {
            assert!(released.contains(&local.mask()), "{:?} should contain {:b} ({})", released, local.mask(), local.mask());
        }

        //  The claims should match the victim.
        let claimed = global.claimed_mask();
        let released = global.released_mask();

        let actual = global.mask();
        assert_eq!((INITIAL & !released) | claimed, actual,
            "initial: {:b}, released: {:b}, claimed: {:b}, actual: {:b}", INITIAL, released, claimed, actual);
    });

    builder.launch(100);
}

#[test]
fn atomic_bit_mask_concurrent_overlapping_claim_at_release_fuzzing() {
    //  This test aims at validating that multiple threads can call claim_at and release concurrently.
    //
    //  To do so:
    //  -   The mask is set to a specific pattern.
    //  -   Two threads attempt to claim at specific spots, whilst the other 2 release at specific spots.
    //      As the spots are overlapping, the claims may fail, in various ways...
    //  -   A check is made that the claims are as expected.
    const INITIAL: u64 = 0b00000000_11111111_00000000_11111111_00000000_11111111_11100000_00111000u64;

    let mut builder = BurstyBuilder::new(Global::default(),
        vec!(LocalAt::new(0, 0..4), LocalAt::new(1, 3..6), LocalAt::new(2, 10..14), LocalAt::new(3, 13..16)));

    //  Step 1: Reset.
    builder.add_simple_step(|| |global: &Global, local: &mut LocalAt| {
        global.victim.initialize(INITIAL);
        global.claimed[local.index].store(0, Ordering::Relaxed);
        global.released[local.index].store(0, Ordering::Relaxed);
    });

    //  Step 2: Claim or release.
    builder.add_simple_step(|| |global: &Global, local: &mut LocalAt| {
        if local.index % 2 == 0 {
            if global.victim.claim_at(local.at(), local.number()) {
                global.claimed[local.index].store(local.mask(), Ordering::Relaxed);
            }
        } else {
            global.victim.release_multiple(local.at(), local.number());
            global.released[local.index].store(local.mask(), Ordering::Relaxed);
        }
    });

    //  Step 3: Verify claims.
    builder.add_simple_step(|| |global: &Global, local: &mut LocalAt| {
        let released = global.released_exact();
        let claimed = global.claimed_exact();

        if local.index % 2 != 0 {
            assert!(released.contains(&local.mask()), "{:?} should contain {:b} ({})", released, local.mask(), local.mask());
        }

        if claimed.len() == 2 && local.index % 2 == 0  {
            assert!(claimed.contains(&local.mask()), "{:?} should contain {:b} ({})", claimed, local.mask(), local.mask());
        }

        //  The claims should match the victim.
        let claimed = global.claimed_mask();
        let released = global.released_mask();

        let actual = global.mask();
        assert_eq!((INITIAL & !released) | claimed, actual,
            "initial: {:b}, released: {:b}, claimed: {:b}, actual: {:b}", INITIAL, released, claimed, actual);
    });

    builder.launch(100);
}

struct LocalMulti {
    index: usize,
    number: usize,
    alignment: PowerOf2,
}

impl LocalMulti {
    fn new(index: usize, number: usize, alignment: usize) -> Self {
        Self { index, number, alignment: PowerOf2::new(alignment).unwrap(), }
    }

    fn at(&self) -> usize { self.alignment.value() as usize }

    fn number(&self) -> usize { self.number }

    fn mask(&self) -> u64 { AtomicBitMask::low(self.number()) << self.at() }
}

#[test]
fn atomic_bit_mask_concurrent_claim_multiple_full_success_fuzzing() {
    //  This test aims at validating that multiple threads can call claim_multiple concurrently.
    //
    //  To do so:
    //  -   The mask is reset to 0.
    //  -   Each thread claims multiple (contiguous) bits, and registers the claimed range.
    //  -   A check is made that the claims are as expected.
    let mut builder = BurstyBuilder::new(Global::default(),
        vec!(LocalMulti::new(0, 4, 2), LocalMulti::new(1, 4, 4), LocalMulti::new(2, 8, 2), LocalMulti::new(3, 8, 8)));

    //  Step 1: Reset.
    builder.add_simple_step(|| |global: &Global, local: &mut LocalMulti| {
        global.victim.initialize(0);
        global.claimed[local.index].store(0, Ordering::Relaxed);
    });

    //  Step 2: Claim.
    builder.add_simple_step(|| |global: &Global, local: &mut LocalMulti| {
        if let Some((index, n)) = global.victim.claim_multiple(local.number, local.alignment) {
            global.claimed[local.index].store(AtomicBitMask::low(n) << index, Ordering::Relaxed);

            assert_eq!(0, index % local.alignment, "local: {}, index: {}", local.index, index);
        }
    });

    //  Step 3: Verify claims; they should not overlap...
    builder.add_simple_step(|| |global: &Global, local: &mut LocalMulti| {
        let claimed = global.claimed_mask();

        if claimed.count_ones() != 24 && local.index == 0 {
            global.print_claims();
        }

        //  All should succeed, but their may be gaps, depending on the order in which claims are made, as claims are
        //  optimistic, with a rollback in case of failure, leading to the possibility of observing a dirty state.
        //
        //  Example of gap:
        //  -   Thread 0 claims 0..4.
        //  -   Thread 1 and 2 attempts to claim 4..n, Thread 3 attempts to claim 8..16.
        //  -   Thread 3 wins the race, securing 8..16.
        //  -   Thread 2 wins the race over thread 1, thread 1 fails, both roll back.
        //  -   Both of them look to secure claims past 16, leaving the range 4..8 unclaimed.
        assert_eq!(24, claimed.count_ones(), "{:b}", claimed);

        //  The successes should match the victim.
        let actual = global.mask();
        assert_eq!(claimed, actual, "claimed: {:b}, actual: {:b}", claimed, actual);
    });

    builder.launch(100);
}

#[test]
fn atomic_bit_mask_concurrent_claim_multiple_partial_success_fuzzing() {
    //  This test aims at validating that multiple threads can call claim_multiple concurrently.
    //
    //  To do so:
    //  -   The mask is reset to 0.
    //  -   Each thread claims multiple (contiguous) bits, and registers the claimed range.
    //      One thread will fail to fully claim a range.
    //  -   A check is made that the claims are as expected.
    let mut builder = BurstyBuilder::new(Global::default(),
        vec!(LocalMulti::new(0, 20, 4), LocalMulti::new(1, 20, 4), LocalMulti::new(2, 20, 4), LocalMulti::new(3, 20, 4)));

    //  Step 1: Reset.
    builder.add_simple_step(|| |global: &Global, local: &mut LocalMulti| {
        global.victim.initialize(0);
        global.claimed[local.index].store(0, Ordering::Relaxed);
    });

    //  Step 2: Claim.
    builder.add_simple_step(|| |global: &Global, local: &mut LocalMulti| {
        if let Some((index, n)) = global.victim.claim_multiple(local.number, local.alignment) {
            global.claimed[local.index].store(AtomicBitMask::low(n) << index, Ordering::Relaxed);

            assert_eq!(0, index % local.alignment, "Thread: {}, index: {}", local.index, index);

            let expected = if index == 0 { 64 - 3 * local.number } else { local.number };
            assert_eq!(expected, n, "Thread: {}, index: {}, n: {}", local.index, index, n);
        }
    });

    //  Step 3: Verify claims; they should not overlap...
    builder.add_simple_step(|| |global: &Global, local: &mut LocalMulti| {
        let claimed = global.claimed_mask();


        if claimed.count_ones() != 64 && local.index == 0 {
            global.print_claims();
        }

        //  At least one should succeed.
        assert_eq!(64, claimed.count_ones(), "{:b}", claimed);

        //  The successes should match the victim.
        let actual = global.mask();
        assert_eq!(claimed, actual, "claimed: {:b}, actual: {:b}", claimed, actual);
    });

    builder.launch(100);
}

#[test]
fn atomic_bit_mask_concurrent_claim_multiple_failure_fuzzing() {
    //  This test aims at validating that multiple threads can call claim_multiple concurrently.
    //
    //  To do so:
    //  -   The mask is reset to a specific bit pattern with only 2 fitting ranges.
    //  -   Each thread attempts to claim multiple (contiguous) bits, and registers the claimed range.
    //      2 threads will fail.
    //  -   A check is made that the claims are as expected.
    const INITIAL: u64 = 0b11111111_11110000_11111111_11111111_11111100_00000011_11111111_11111111u64;

    let mut builder = BurstyBuilder::new(Global::default(),
        vec!(LocalMulti::new(0, 4, 16), LocalMulti::new(1, 4, 16), LocalMulti::new(2, 8, 2), LocalMulti::new(3, 8, 2)));

    //  Step 1: Reset.
    builder.add_simple_step(|| |global: &Global, local: &mut LocalMulti| {
        global.victim.initialize(INITIAL);
        global.claimed[local.index].store(0, Ordering::Relaxed);
    });

    //  Step 2: Claim.
    builder.add_simple_step(|| |global: &Global, local: &mut LocalMulti| {
        if let Some((index, n)) = global.victim.claim_multiple(local.number, local.alignment) {
            global.claimed[local.index].store(AtomicBitMask::low(n) << index, Ordering::Relaxed);

            assert_eq!(0, index % local.alignment, "Thread: {}, index: {}", local.index, index);
            assert_eq!(n, local.number, "Thread: {}", local.index);
        }
    });

    //  Step 3: Verify claims.
    builder.add_simple_step(|| |global: &Global, local: &mut LocalMulti| {
        assert_eq!(vec!(AtomicBitMask::low(8) << 18, AtomicBitMask::low(4) << 48), global.claimed_exact());

        let claimed = global.claimed_mask();

        if claimed != !INITIAL && local.index == 0 {
            global.print_claims();
        }

        //  One of each of (0, 1) and (2, 3) should succeed, as each pair competes for a spot the other pair judges
        //  unsuitable.
        assert_eq!(!INITIAL, claimed, "{:b}", claimed);

        //  The successes should match the victim.
        let actual = global.mask();
        assert_eq!(INITIAL | claimed, actual, "initial: {:b}, claimed: {:b}, actual: {:b}", INITIAL, claimed, actual);
    });

    builder.launch(100);
}

#[test]
fn atomic_bit_mask_concurrent_independent_claim_multiple_release_fuzzing() {
    //  This test aims at validating that multiple threads can call claim_multiple and release concurrently.
    //
    //  To do so:
    //  -   The mask is set to a specific pattern.
    //  -   Two threads attempt to claim multiple bits, whilst the other 2 release at specific spots.
    //      As the spots are independent, all should succeed.
    //  -   A check is made that the claims are as expected.
    const INITIAL: u64 = 0b00001111_11111111_00000000_11111111_11111111_11111111_11111111_11111111u64;

    let mut builder = BurstyBuilder::new(Global::default(),
        vec!(LocalMulti::new(0, 4, 4), LocalMulti::new(1, 8, 16), LocalMulti::new(2, 8, 8), LocalMulti::new(3, 8, 8)));

    //  Step 1: Reset.
    builder.add_simple_step(|| |global: &Global, local: &mut LocalMulti| {
        global.victim.initialize(INITIAL);
        global.claimed[local.index].store(0, Ordering::Relaxed);
        global.released[local.index].store(0, Ordering::Relaxed);
    });

    //  Step 2: Claim or release.
    builder.add_simple_step(|| |global: &Global, local: &mut LocalMulti| {
        if local.index % 2 == 0 {
            if let Some((index, n)) = global.victim.claim_multiple(local.number, local.alignment) {
                global.claimed[local.index].store(AtomicBitMask::low(n) << index, Ordering::Relaxed);
            }
        } else {
            global.victim.release_multiple(local.at(), local.number());
            global.released[local.index].store(local.mask(), Ordering::Relaxed);
        }
    });

    //  Step 3: Verify claims.
    builder.add_simple_step(|| |global: &Global, local: &mut LocalMulti| {
        let released = global.released_exact();
        let claimed = global.claimed_exact();

        if local.index % 2 != 0 {
            assert!(released.contains(&local.mask()), "{:?} should contain {:b} ({})", released, local.mask(), local.mask());
        }

        if local.index % 2 == 0  {
            assert_eq!(vec!(AtomicBitMask::low(8) << 40, AtomicBitMask::low(4) << 60), claimed);
        }

        //  The claims should match the victim.
        let claimed = global.claimed_mask();
        let released = global.released_mask();

        let actual = global.mask();
        assert_eq!((INITIAL & !released) | claimed, actual,
            "initial: {:b}, released: {:b}, claimed: {:b}, actual: {:b}", INITIAL, released, claimed, actual);
    });

    builder.launch(100);
}

#[test]
fn atomic_bit_mask_concurrent_overlapping_claim_multiple_release_fuzzing() {
    //  This test aims at validating that multiple threads can call claim_multiple and release concurrently.
    //
    //  To do so:
    //  -   The mask is set to a specific pattern.
    //  -   Two threads attempt to claim multiple bits, whilst the other 2 release at specific spots.
    //      As the spots are overlapping, the claims may fail, in various ways...
    //  -   A check is made that the claims are as expected.
    const INITIAL: u64 = 0b11001111_11111111_00110000_11001111_11111111_11111111_11111111_11111111u64;

    let mut builder = BurstyBuilder::new(Global::default(),
        vec!(LocalMulti::new(0, 4, 16), LocalMulti::new(1, 4, 32), LocalMulti::new(2, 8, 8), LocalMulti::new(3, 8, 16)));

    //  Step 1: Reset.
    builder.add_simple_step(|| |global: &Global, local: &mut LocalMulti| {
        global.victim.initialize(INITIAL);
        global.claimed[local.index].store(0, Ordering::Relaxed);
        global.released[local.index].store(0, Ordering::Relaxed);
    });

    //  Step 2: Claim or release.
    builder.add_simple_step(|| |global: &Global, local: &mut LocalMulti| {
        if local.index % 2 == 0 {
            if let Some((index, n)) = global.victim.claim_multiple(local.number, local.alignment) {
                global.claimed[local.index].store(AtomicBitMask::low(n) << index, Ordering::Relaxed);
            }
        } else {
            global.victim.release_multiple(local.at(), local.number());
            global.released[local.index].store(local.mask(), Ordering::Relaxed);
        }
    });

    //  Step 3: Verify claims.
    builder.add_simple_step(|| |global: &Global, local: &mut LocalMulti| {
        let released = global.released_exact();
        let claimed = global.claimed_exact();

        if local.index % 2 != 0 {
            assert!(released.contains(&local.mask()), "{:?} should contain {:b} ({})", released, local.mask(), local.mask());
        }

        if claimed.len() == 2 && local.index % 2 == 0  {
            assert_eq!(vec!(AtomicBitMask::low(8) << 16, AtomicBitMask::low(4) << 32), claimed);
        }

        //  The claims should match the victim.
        let claimed = global.claimed_mask();
        let released = global.released_mask();

        let actual = global.mask();
        assert_eq!((INITIAL & !released) | claimed, actual,
            "initial: {:b}, released: {:b}, claimed: {:b}, actual: {:b}", INITIAL, released, claimed, actual);
    });

    builder.launch(100);
}

} // mod tests
