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

            if self.claim(candidate_mask) {
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
        debug_assert!(_before & mask == mask);
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

} // mod tests
