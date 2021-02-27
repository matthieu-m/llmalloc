//! An atomic bit mask representing the occupation (or not) of a block.

use core::cmp;

use crate::PowerOf2;

use super::{AtomicBitMask, NumberPages, PageIndex};

//  Page Tokens.
//
//  A bitmap of which Large Pages are available, and which are not, where 0 means available and 1 occupied.
//
//  The first bit is always 1, as the first Large Page is always occupied by the Huge Page header itself.
pub(crate) struct PageTokens([AtomicBitMask; 8]);

impl PageTokens {
    /// Initialize the tokens based on the number of pages.
    pub(crate) fn new(number_pages: NumberPages) -> Self {
        let result = PageTokens(Default::default());

        debug_assert!(number_pages.0 < result.capacity());

        //  The first Large Page is always reserved:
        //  -   for the header of the Huge Page.
        //  -   and optionally to store data in the remaining buffer area.
        result.0[0].initialize(1);

        //  The trailing pages are also reserved, based on how many actual pages are available.
        let mut trailing_ones = result.capacity() - 1 - number_pages.0;
        let mut index = result.0.len() - 1;

        while trailing_ones >= 64 {
            result.0[index].initialize(u64::MAX);
            trailing_ones -= 64;
            index -= 1;
        }

        if trailing_ones > 0 {
            let zeroes = {
                let shift = 64 - trailing_ones;
                (1u64 << shift) - 1
            };

            let ones = u64::MAX - zeroes + if index > 0 { 0 } else { 1 };
            result.0[index].initialize(ones);
        }

        result
    }

    /// Allocates 1 Large Page, returns its index or none if it could not allocate.
    pub(crate) fn fast_allocate(&self) -> Option<PageIndex> {
        for (outer, bits) in self.0.iter().enumerate() {
            if let Some(inner) = bits.claim_single() {
                return PageIndex::new(outer * AtomicBitMask::CAPACITY + inner);
            }
        }

        None
    }

    /// Allocates multiple Large Pages, return the index of the first page, or none if it could not allocate.
    ///
    /// The index returned is a multiple of `align_pages`.
    pub(crate) fn flexible_allocate(&self, number_pages: NumberPages, align_pages: PowerOf2) -> Option<PageIndex> {
        const OUTER_1: &[usize] = &[7, 6, 5, 4, 3, 2];
        const OUTER_2: &[usize] = &[7, 5, 3];
        const OUTER_4: &[usize] = &[7];
        const OUTER: &[&[usize]] = &[&[], OUTER_1, OUTER_2, &[], OUTER_4];

        //  Safety:
        //  -   `align_pages / AtomicBitMask::CAPACITY` is either 0 or a power of 2.
        //  -   1 is a power of 2.
        //  -   Hence the maximum is a power of 2.
        let align_outer = align_pages / AtomicBitMask::CAPACITY;
        debug_assert!(align_outer == 0 || self.0.len() % align_outer == 0);
        debug_assert!(self.0.len() == 8, "{} != 8 => review `match`!", self.0.len());

        //  Do a single pass over the bits, attempting to find `number` consecutive ones.
        //
        //  Iterate in reverse order to avoid interfering with the fast single-page allocation path.
        let mut outer = self.0.len() as isize - 1;

        match align_outer {
            0 => while outer >= 0 {
                //  Safety:
                //  -   `outer` is within bounds.
                let index = unsafe { self.flexible_allocate_backward_from(outer as usize, number_pages, align_pages) };

                outer = match index {
                    Ok(index) => return Some(index),
                    Err(next) => next,
                };
            },
            _ => for outer in unsafe { *OUTER.get_unchecked(align_outer) } {
                //  Safety:
                //  -   `outer` is within bounds.
                let index = unsafe { self.flexible_allocate_backward_from(*outer, number_pages, AtomicBitMask::CAPACITY) };

                if let Ok(index) = index {
                    return Some(index);
                }
            },
        }

        None
    }

    /// Deallocates the large page at the specified `index`.
    ///
    /// #   Safety
    ///
    /// -   Assumes that `index` is within bounds.
    pub(crate) unsafe fn fast_deallocate(&self, index: PageIndex) {
        let (outer, inner) = (index.value() / AtomicBitMask::capacity(), index.value() % AtomicBitMask::capacity());
        debug_assert!(outer < self.0.len());

        self.0.get_unchecked(outer).release_single(inner);
    }

    /// Deallocates `number_pages` Large Pages starting from `index`.
    ///
    /// #   Safety
    ///
    /// -   Assumes that `index` is within bounds.
    /// -   Assumes that `index + number_pages` is within bounds.
    pub(crate) unsafe fn flexible_deallocate(&self, index: PageIndex, number_pages: NumberPages) {
        let (outer, inner) = (index.value() / AtomicBitMask::capacity(), index.value() % AtomicBitMask::capacity());
        debug_assert!(outer < self.0.len());

        let (head_bits, middle_atomics, tail_bits) = Self::split(inner, number_pages.0);
        debug_assert!(outer + middle_atomics < self.0.len());
        debug_assert!(tail_bits == 0 || outer + middle_atomics + 1 < self.0.len());

        self.0.get_unchecked(outer).release_multiple(inner, head_bits);

        for n in 0..middle_atomics {
            self.0.get_unchecked(outer + n + 1).release_multiple(0, AtomicBitMask::capacity());
        }

        if tail_bits != 0 {
            self.0.get_unchecked(outer + middle_atomics + 1).release_multiple(0, tail_bits);
        }
    }

    //  Internal: Returns the number of bits.
    fn capacity(&self) -> usize { self.0.len() * AtomicBitMask::capacity() }

    //  Internal: Allocates number_pages, starting from outer.
    //
    //  In case of failure, all internally claimed pages are released and the index of the next outer index to try is
    //  returned.
    //
    //  #   Safety
    //
    //  -   `outer` is within bounds.
    unsafe fn flexible_allocate_backward_from(&self, outer: usize, number_pages: NumberPages, align_pages: PowerOf2)
        -> Result<PageIndex, isize>
    {
        debug_assert!(number_pages.0 % align_pages == 0, "{} % {} != 0", number_pages.0, align_pages.value());

        //  The maximum capacity, accounting for the fact that `self.0` has 1 always reserved page.
        let maximum_capacity = outer * AtomicBitMask::capacity() + AtomicBitMask::capacity() - 1;

        let number_pages = number_pages.0;

        //  If the number of pages to allocate is greater than the maximum capacity, abandon.
        if number_pages > maximum_capacity {
            return Err(-1);
        }

        let (inner, tail) =
            match self.0.get_unchecked(outer).claim_multiple(cmp::min(number_pages, AtomicBitMask::capacity()), align_pages) {
                Some(tuple) => tuple,
                None => return Err(outer as isize - 1),
            };

        if tail == number_pages {
            return PageIndex::new(outer * AtomicBitMask::capacity() + inner).ok_or(-1);
        }

        debug_assert!(inner == 0, "{} != 0", inner);
        debug_assert!(tail % align_pages == 0, "{} % {} != 0", tail, align_pages.value());

        //  If outer is 0, it is not possible to go further back.
        if outer == 0 {
            self.flexible_allocate_rewind(outer, outer, tail);
            return Err(-1);
        }

        let remaining_capacity = maximum_capacity - AtomicBitMask::capacity();
        let remaining_pages = number_pages - tail;
        debug_assert!(remaining_pages % align_pages == 0, "{} % {} != 0", remaining_pages, align_pages.value());

        //  There may have been capacity if the AtomicBitMask at `outer` could allocate more, but it failed to.
        if remaining_pages > remaining_capacity {
            self.flexible_allocate_rewind(outer, outer, tail);
            return Err(-1);
        }

        let (head, middle) = (remaining_pages % AtomicBitMask::capacity(), remaining_pages / AtomicBitMask::capacity());
        debug_assert!(outer > middle);
        debug_assert!(head % align_pages == 0, "{} % {} != 0", head, align_pages.value());

        let middle_outer = outer - middle;
        debug_assert!(middle_outer > 0);

        for i in (middle_outer..outer).rev() {
            if !self.0.get_unchecked(i).claim_at(0, AtomicBitMask::capacity()) {
                //  The current AtomicBitMask `i` was released by claim_at, so only the other BitMasks need releasing.
                self.flexible_allocate_rewind(i + 1, outer, tail);
                return Err(i as isize - 1);
            }
        }

        if head == 0 {
            return PageIndex::new(middle_outer * AtomicBitMask::capacity()).ok_or(-1);
        }

        let head_outer = middle_outer - 1;

        if !self.0.get_unchecked(head_outer).claim_at(AtomicBitMask::capacity() - head, head) {
            //  The current AtomicBitMask `middle_outer - 1` was released by claim_at, so only the other BitMasks need
            //  releasing.
            self.flexible_allocate_rewind(middle_outer, outer, tail);
            return Err(head_outer as isize);
        }

        PageIndex::new(head_outer * AtomicBitMask::capacity() + AtomicBitMask::capacity() - head).ok_or(-1)
    }

    //  Internal: Releases the tentatively claimed pages.
    //
    //  -   [from, to): Completely claimed AtomicBitMask.
    //  -   to: first "tail" bits are claimed.
    unsafe fn flexible_allocate_rewind(&self, from: usize, to: usize, tail: usize) {
        debug_assert!(from <= to);
        debug_assert!(to < self.0.len());

        for i in from..to {
            self.0.get_unchecked(i).release_multiple(0, AtomicBitMask::capacity());
        }

        self.0.get_unchecked(to).release_multiple(0, tail);
    }

    //  Internal: Returns the number of head bits, middle atomics, tail bits.
    fn split(inner: usize, number_pages: usize) -> (usize, usize, usize) {
        let head_bits = AtomicBitMask::capacity() - inner;

        if head_bits >= number_pages {
            return (number_pages, 0, 0);
        }

        let number_pages = number_pages - head_bits;
        (head_bits, number_pages / AtomicBitMask::CAPACITY, number_pages % AtomicBitMask::CAPACITY)
    }
}

//
//  Implementation
//

#[cfg(test)]
mod tests {

use std::sync::atomic::{AtomicUsize, Ordering};

use llmalloc_test::BurstyBuilder;

use super::*;
use super::super::atomic_bit_mask::load_bitmask;

//  Internal: Computes a mask with the `number` high bits set, and all others unset.
fn high(number: usize) -> u64 {
    debug_assert!(number <= AtomicBitMask::capacity());

    !low(AtomicBitMask::capacity() - number)
}

//  Internal: Computes a mask with the `number` low bits set, and all others unset.
fn low(number: usize) -> u64 {
    debug_assert!(number <= AtomicBitMask::capacity());

    if number == 64 {
        u64::MAX
    } else {
        (1u64 << number) - 1
    }
}

type RawPageTokens = [u64; 8];

fn load_page_tokens(page_tokens: &PageTokens) -> RawPageTokens {
    assert_eq!(512, page_tokens.capacity());

    [
        load_bitmask(&page_tokens.0[0]),
        load_bitmask(&page_tokens.0[1]),
        load_bitmask(&page_tokens.0[2]),
        load_bitmask(&page_tokens.0[3]),
        load_bitmask(&page_tokens.0[4]),
        load_bitmask(&page_tokens.0[5]),
        load_bitmask(&page_tokens.0[6]),
        load_bitmask(&page_tokens.0[7]),
    ] 
}

fn create_page_tokens(tokens: RawPageTokens) -> PageTokens {
    assert_eq!(1, tokens[0] & 1);

    let page_tokens = PageTokens::new(NumberPages(511));

    for i in 0..tokens.len() {
        page_tokens.0[i].initialize(tokens[i]);
    }

    page_tokens
}

#[track_caller]
fn check_tokens(tokens: RawPageTokens, expected: RawPageTokens) {
    assert_eq!(tokens.len(), expected.len());

    for (index, (expected, token)) in expected.iter().zip(tokens.iter()).enumerate() {
        assert_eq!(expected, token,
            "Index {} - expected {:b} got {:b}", index, expected, token);
    }
}

#[test]
fn page_tokens_new() {
    fn new(number_pages: usize) -> RawPageTokens {
        let page_tokens = PageTokens::new(NumberPages(number_pages));
        load_page_tokens(&page_tokens)
    }

    const FULL: u64 = u64::MAX;

    check_tokens(new(511), [low(1), 0, 0, 0, 0, 0, 0, 0]);

    for i in 1..=64 {
        check_tokens(new(511 - i), [low(1), 0, 0, 0, 0, 0, 0, high(i)]);
    }

    for i in 1..=64 {
        check_tokens(new(511 - 64 - i), [low(1), 0, 0, 0, 0, 0, high(i), FULL]);
    }

    for i in 1..=64 {
        check_tokens(new(127 - i), [low(1), high(i), FULL, FULL, FULL, FULL, FULL, FULL]);
    }

    for i in 1..=62 {
        check_tokens(new(63 - i), [low(1) + high(i), FULL, FULL, FULL, FULL, FULL, FULL, FULL]);
    }
}

#[test]
fn page_tokens_fast_allocate() {
    fn fast_allocate(initial: RawPageTokens) -> Option<usize> {
        let tokens = create_page_tokens(initial);

        tokens.fast_allocate().map(|x| x.value())
    }

    let full = u64::MAX;

    assert_eq!(Some(1), fast_allocate([1, 0, 0, 0, 0, 0, 0, 0]));
    assert_eq!(None, fast_allocate([full, full, full, full, full, full, full, full]));

    for i in 0..64 {
        assert_eq!(Some(64 + i), fast_allocate([full, low(i), full, full, full, full, full, full]));
    }

    for i in 0..64 {
        assert_eq!(Some(64 * 7 + i), fast_allocate([full, full, full, full, full, full, full, low(i)]));
    }
}

#[test]
fn page_tokens_fast_deallocate() {
    fn fast_deallocate(index: usize, initial: RawPageTokens) -> RawPageTokens {
        let page_tokens = create_page_tokens(initial);
        unsafe { page_tokens.fast_deallocate(PageIndex::new(index).unwrap()) };
        load_page_tokens(&page_tokens)
    }

    let full = u64::MAX;

    assert_eq!(
        [1, 0, 0, 0, 0, 0, 0, 0],
        fast_deallocate(1, [3, 0, 0, 0, 0, 0, 0, 0])
    );

    assert_eq!(
        [1, 0, 0, 0, 0, 0, 0, 0],
        fast_deallocate(64 * 4 + 5, [1, 0, 0, 0, 1u64 << 5, 0, 0, 0])
    );

    assert_eq!(
        [1 + high(62), 0, 0, 0, 0, 0, 0, 0],
        fast_deallocate(1, [full, 0, 0, 0, 0, 0, 0, 0])
    );
}

fn flexible_allocate(number: usize, initial: RawPageTokens, expected: RawPageTokens) -> Option<usize> {
    let tokens = create_page_tokens(initial);
    let index = tokens.flexible_allocate(NumberPages(number), PowerOf2::ONE).map(|x| x.value());

    check_tokens(load_page_tokens(&tokens), expected);

    index
}

#[test]
fn page_tokens_flexible_allocate_success() {
    let full = u64::MAX;
    let all_empty = [1, 0, 0, 0, 0, 0, 0, 0];
    let all_full = [full, full, full, full, full, full, full, full];

    assert_eq!(Some(510), flexible_allocate(2, all_empty, [1, 0, 0, 0, 0, 0, 0, high(2)]));
    assert_eq!(Some(1), flexible_allocate(511, all_empty, all_full));

    for number in 2..=64 {
        let expected = [1, 0, 0, 0, 0, 0, 0, high(number)];
        assert_eq!(Some(512 - number), flexible_allocate(number, all_empty, expected));
    }

    for number in 1..=64 {
        let expected = [1, 0, 0, 0, 0, 0, high(number), full];
        assert_eq!(Some(512 - 64 - number), flexible_allocate(64 + number, all_empty, expected));
    }
}

#[test]
fn page_tokens_flexible_allocate_failure() {
    let full = u64::MAX;
    let all_full = [full, full, full, full, full, full, full, full];

    assert_eq!(None, flexible_allocate(2, all_full, all_full));
    assert_eq!(None, flexible_allocate(511, all_full, all_full));

    for number in 2..=64 {
        let mask = low(65 - number);
        let tokens = [mask, mask, mask, mask, mask, mask, mask, mask];
        assert_eq!(None, flexible_allocate(number, tokens, tokens));
    }

    for number in 1..=64 {
        let mask = low(65 - number);
        let tokens = [mask, 0, mask, 0, mask, 0, mask, 0];
        assert_eq!(None, flexible_allocate(64 + number, tokens, tokens));
    }
}

#[test]
fn page_tokens_flexible_allocate_straddling() {
    let full = u64::MAX;

    for number in 2..=64 {
        let initial = [1, 0, 0, 0, 0, 0, 0, high(63)];
        let expected = [1, 0, 0, 0, 0, 0, high(number - 1), full];
        assert_eq!(Some(512 - 63 - number), flexible_allocate(number, initial, expected));
    }

    for number in 1..=64 {
        let initial = [1, 0, 0, 0, 0, 0, 0, high(63)];
        let expected = [1, 0, 0, 0, 0, high(number - 1), full, full];
        assert_eq!(Some(512 - 63 - 64 - number), flexible_allocate(64 + number, initial, expected));
    }
}

fn flexible_allocate_aligned(number: usize, alignment: usize, initial: RawPageTokens, expected: RawPageTokens)
    -> Option<usize>
{
    let alignment = PowerOf2::new(alignment).expect("Power of 2");
    let tokens = create_page_tokens(initial);
    let index = tokens.flexible_allocate(NumberPages(number), alignment).map(|x| x.value());

    check_tokens(load_page_tokens(&tokens), expected);

    index
}

#[test]
fn page_tokens_flexible_allocate_aligned_small_success() {
    let full = u64::MAX;
    let all_empty = [1, 0, 0, 0, 0, 0, 0, 0];

    assert_eq!(Some(510), flexible_allocate_aligned(2, 2, all_empty, [1, 0, 0, 0, 0, 0, 0, high(2)]));
    assert_eq!(Some(512 - 64), flexible_allocate_aligned(64, 64, all_empty, [1, 0, 0, 0, 0, 0, 0, full]));

    let before = [1, 0, 0, 0, 0, 0, 0, high(1)];
    let tests = [
        ( 2, [1, 0, 0, 0, 0, 0, 0, high(1) + (low(2) << 60)]),
        ( 4, [1, 0, 0, 0, 0, 0, 0, high(1) + (low(4) << 56)]),
        ( 8, [1, 0, 0, 0, 0, 0, 0, high(1) + (low(8) << 48)]),
        (16, [1, 0, 0, 0, 0, 0, 0, high(1) + (low(16) << 32)]),
        (32, [1, 0, 0, 0, 0, 0, 0, high(1) + low(32)]),
        (64, [1, 0, 0, 0, 0, 0, full, high(1)]),
    ];

    for &(number, expected) in &tests {
        assert_eq!(Some(512 - number * 2), flexible_allocate_aligned(number, number, before, expected));
    }

    let before = [1, 0, 0, 0, 0, 0, high(1), full];
    for number in 1..=31 {
        let number = number * 2;
        let expected = [1, 0, 0, 0, 0, 0, high(number + 2) - (high(1) >> 1), full];
        assert_eq!(Some(512 - 64 - number - 2), flexible_allocate_aligned(number, 2, before, expected));
    }
}

#[test]
fn page_tokens_flexible_allocate_aligned_small_failure() {
    let full = u64::MAX;
    let all_full = [full, full, full, full, full, full, full, full];

    assert_eq!(None, flexible_allocate_aligned(2, 2, all_full, all_full));
    assert_eq!(None, flexible_allocate_aligned(256, 256, all_full, all_full));
}

#[test]
fn page_tokens_flexible_allocate_aligned_128() {
    let full = u64::MAX;
    let one = low(1) << 11;
    let two = low(1) << 33;
    let all_empty = [1, 0, 0, 0, 0, 0, 0, 0];

    assert_eq!(Some(384), flexible_allocate_aligned(128, 128, all_empty, [1, 0, 0, 0, 0, 0, full, full]));
    assert_eq!(Some(256), flexible_allocate_aligned(256, 128, all_empty, [1, 0, 0, 0, full, full, full, full]));
    assert_eq!(Some(128), flexible_allocate_aligned(384, 128, all_empty, [1, 0, full, full, full, full, full, full]));

    let before = [1, 0, 0, 0, 0, two, one, 0];
    assert_eq!(Some(128), flexible_allocate_aligned(128, 128, before, [1, 0, full, full, 0, two, one, 0]));

    let before = [1, 0, 0, 0, 0, 0, 0, one];
    assert_eq!(Some(128), flexible_allocate_aligned(256, 128, before, [1, 0, full, full, full, full, 0, one]));
}

#[test]
fn page_tokens_flexible_allocate_aligned_256() {
    let full = u64::MAX;
    let all_empty = [1, 0, 0, 0, 0, 0, 0, 0];

    assert_eq!(Some(256), flexible_allocate_aligned(256, 256, all_empty, [1, 0, 0, 0, full, full, full, full]));

    for outer in 0..=3 {
        for inner in 0..=63 {
            let mut before = all_empty;
            let mut after = [1, 0, 0, 0, full, full, full, full];

            before[outer] |= low(1) << inner;
            after[outer] |= low(1) << inner;

            assert_eq!(Some(256), flexible_allocate_aligned(256, 256, before, after));
        }
    }

    for outer in 4..=7 {
        for inner in 0..=63 {
            let mut mask = all_empty;
            mask[outer] |= low(1) << inner;

            assert_eq!(None, flexible_allocate_aligned(256, 256, mask, mask));
        }
    }
}

#[test]
fn page_tokens_flexible_deallocate() {
    fn flexible_deallocate(index: usize, number: usize, initial: RawPageTokens) -> RawPageTokens {
        let page_tokens = create_page_tokens(initial);
        unsafe { page_tokens.flexible_deallocate(PageIndex::new(index).unwrap(), NumberPages(number)) };

        load_page_tokens(&page_tokens)
    }

    let full = u64::MAX;

    assert_eq!(
        [1, 0, 0, 0, 0, 0, 0, 0],
        flexible_deallocate(510, 2, [1, 0, 0, 0, 0, 0, 0, high(2)])
    );

    assert_eq!(
        [1, 0, 0, 0, 0, 0, 0, 0],
        flexible_deallocate(1, 511, [full, full, full, full, full, full, full, full])
    );

    assert_eq!(
        [1, 0, 0, 0, low(3) + high(40), 0, 0, 0],
        flexible_deallocate(64 * 4 + 3, 21, [1, 0, 0, 0, full, 0, 0, 0])
    );

    assert_eq!(
        [1, 0, 0, low(37), high(40), 0, 0, 0],
        flexible_deallocate(64 * 3 + 37, 51, [1, 0, 0, full, full, 0, 0, 0])
    );

    assert_eq!(
        [1, 0, low(37), 0, 0, high(40), 0, 0],
        flexible_deallocate(64 * 2 + 37, 179, [1, 0, full, full, full, full, 0, 0])
    );
}

struct Global {
    victim: PageTokens,
    page_indexes: [AtomicUsize; 4],
}

impl Global {
    fn new(number_pages: usize) -> Self {
        let victim = PageTokens::new(NumberPages(number_pages));
        let page_indexes = Default::default();

        Self { victim, page_indexes }
    }

    fn reset(&self, expected_free_pages: usize) {
        let actual_free_pages = load_page_tokens(&self.victim).iter()
            .map(|page| page.count_zeros() as usize)
            .sum();
        assert_eq!(expected_free_pages, actual_free_pages);

        for page_index in &self.page_indexes {
            page_index.store(0, Ordering::Relaxed);
        }
    }

    fn set(&self, thread: usize, page_index: PageIndex) {
        self.page_indexes[thread].store(page_index.value(), Ordering::Relaxed);
    }

    fn page_indexes(&self) -> Vec<usize> {
        let mut page_indexes: Vec<_> = self.page_indexes.iter()
            .map(|page_index| page_index.load(Ordering::Relaxed))
            .filter(|page_index| *page_index != 0)
            .collect();
        page_indexes.sort();
        page_indexes
    }

    #[track_caller]
    fn verify_alignment(&self, alignment: PowerOf2) {
        let indexes = self.page_indexes();
        for index in &indexes {
            assert_eq!(0, *index % alignment, "{:?}", indexes);
        }
    }

    #[track_caller]
    fn verify_non_overlapping(&self, number_pages: NumberPages) {
        let indexes = self.page_indexes();

        for pair in indexes.windows(2) {
            assert!(pair[1] - pair[0] >= number_pages.0, "{:?}", indexes);
        }
    }
}

struct Local {
    index: usize,
    allocated: Option<PageIndex>,
}

impl Local {
    fn new(index: usize) -> Self { Self { index, allocated: None, } }
}

#[test]
fn page_tokens_concurrent_fast_allocate_deallocate_success_fuzzing() {
    //  This test aims at validating that fast_allocate can be called concurrently.
    //
    //  The test is primed with an empty PageTokens of sufficient capacity.
    //  -   The global state is reset.
    //  -   Each thread calls fast_allocate, stores the result locally _and_ globally.
    //  -   Each thread calls fast_deallocate.
    //  -   The globally stored indices are checked.
    let mut builder = BurstyBuilder::new(Global::new(4),
        vec!(Local::new(0), Local::new(1), Local::new(2), Local::new(3)));

    //  Step 1: reset.
    builder.add_simple_step(|| |global: &Global, local: &mut Local| {
        global.reset(4);
        local.allocated = None;
    });

    //  Step 2: allocate.
    builder.add_simple_step(|| |global: &Global, local: &mut Local| {
        let allocated = global.victim.fast_allocate();

        if let Some(allocated) = allocated {
            local.allocated = Some(allocated);
            global.set(local.index, allocated);
        }
    });

    //  Step 3: deallocate.
    builder.add_simple_step(|| |global: &Global, local: &mut Local| {
        if let Some(allocated) = local.allocated {
            unsafe { global.victim.fast_deallocate(allocated) };
        }
    });

    //  Step 4: check allocations.
    builder.add_simple_step(|| |global: &Global, _: &mut Local| {
        assert_eq!(vec!(1, 2, 3, 4), global.page_indexes());
    });

    builder.launch(100);
}

#[test]
fn page_tokens_concurrent_fast_allocate_deallocate_failure_fuzzing() {
    //  This test aims at validating that fast_allocate can be called concurrently.
    //
    //  The test is primed with an empty PageTokens of sufficient capacity.
    //  -   The global state is reset.
    //  -   Each thread calls fast_allocate, stores the result locally _and_ globally.
    //  -   Each thread calls fast_deallocate.
    //  -   The globally stored indices are checked.
    let mut builder = BurstyBuilder::new(Global::new(3),
        vec!(Local::new(0), Local::new(1), Local::new(2), Local::new(3)));

    //  Step 1: reset.
    builder.add_simple_step(|| |global: &Global, local: &mut Local| {
        global.reset(3);
        local.allocated = None;
    });

    //  Step 2: allocate.
    builder.add_simple_step(|| |global: &Global, local: &mut Local| {
        let allocated = global.victim.fast_allocate();

        if let Some(allocated) = allocated {
            local.allocated = Some(allocated);
            global.set(local.index, allocated);
        }
    });

    //  Step 3: deallocate.
    builder.add_simple_step(|| |global: &Global, local: &mut Local| {
        if let Some(allocated) = local.allocated {
            unsafe { global.victim.fast_deallocate(allocated) };
        }
    });

    //  Step 4: check allocations.
    builder.add_simple_step(|| |global: &Global, _: &mut Local| {
        assert_eq!(vec!(1, 2, 3), global.page_indexes());
    });

    builder.launch(100);
}

#[test]
fn page_tokens_concurrent_flexible_allocate_deallocate_success_fuzzing() {
    //  This test aims at validating that flexible_allocate can be called concurrently.
    //
    //  The test is primed with an empty PageTokens of sufficient capacity.
    //  -   The global state is reset.
    //  -   Each thread calls flexible_allocate, stores the result locally _and_ globally.
    //  -   Each thread calls flexible_deallocate.
    //  -   The globally stored indices are checked.
    const NUMBER_PAGES: NumberPages = NumberPages(22);
    const ALIGNMENT: PowerOf2 = unsafe { PowerOf2::new_unchecked(2) };

    //  Due to collisions & partial commits, part of the available pages will be attempted, skipped, and yet released
    //  during rollback, leaving them free in the end. Hence, some overhead is needed.
    const INITIAL_PAGES: usize = NUMBER_PAGES.0 * 5 + 1;

    let mut builder = BurstyBuilder::new(Global::new(INITIAL_PAGES),
        vec!(Local::new(0), Local::new(1), Local::new(2), Local::new(3)));

    //  Step 1: reset.
    builder.add_simple_step(|| |global: &Global, local: &mut Local| {
        global.reset(INITIAL_PAGES);
        local.allocated = None;
    });

    //  Step 2: allocate.
    builder.add_simple_step(|| |global: &Global, local: &mut Local| {
        let allocated = global.victim.flexible_allocate(NUMBER_PAGES, ALIGNMENT);

        if let Some(allocated) = allocated {
            local.allocated = Some(allocated);
            global.set(local.index, allocated);
        }
    });

    //  Step 3: deallocate.
    builder.add_simple_step(|| |global: &Global, local: &mut Local| {
        if let Some(allocated) = local.allocated {
            unsafe { global.victim.flexible_deallocate(allocated, NUMBER_PAGES) };
        }
    });

    //  Step 4: check allocations.
    builder.add_simple_step(|| |global: &Global, _: &mut Local| {
        let indexes = global.page_indexes();
        assert_eq!(4, indexes.len(), "{:?}", indexes);

        global.verify_alignment(ALIGNMENT);
        global.verify_non_overlapping(NUMBER_PAGES);
    });

    builder.launch(100);
}

#[test]
fn page_tokens_concurrent_flexible_allocate_deallocate_failure_fuzzing() {
    //  This test aims at validating that flexible_allocate can be called concurrently.
    //
    //  The test is primed with an empty PageTokens of sufficient capacity.
    //  -   The global state is reset.
    //  -   Each thread calls flexible_allocate, stores the result locally _and_ globally.
    //  -   Each thread calls flexible_deallocate.
    //  -   The globally stored indices are checked.
    const NUMBER_PAGES: NumberPages = NumberPages(22);
    const ALIGNMENT: PowerOf2 = unsafe { PowerOf2::new_unchecked(2) };

    //  It's expected that only 2 are likely to succeed, but maybe 3 will...
    const INITIAL_PAGES: usize = NUMBER_PAGES.0 * 3 + 1;

    let mut builder = BurstyBuilder::new(Global::new(INITIAL_PAGES),
        vec!(Local::new(0), Local::new(1), Local::new(2), Local::new(3)));

    //  Step 1: reset.
    builder.add_simple_step(|| |global: &Global, local: &mut Local| {
        global.reset(INITIAL_PAGES);
        local.allocated = None;
    });

    //  Step 2: allocate.
    builder.add_simple_step(|| |global: &Global, local: &mut Local| {
        let allocated = global.victim.flexible_allocate(NUMBER_PAGES, ALIGNMENT);

        if let Some(allocated) = allocated {
            local.allocated = Some(allocated);
            global.set(local.index, allocated);
        }
    });

    //  Step 3: deallocate.
    builder.add_simple_step(|| |global: &Global, local: &mut Local| {
        if let Some(allocated) = local.allocated {
            unsafe { global.victim.flexible_deallocate(allocated, NUMBER_PAGES) };
        }
    });

    //  Step 4: check allocations.
    builder.add_simple_step(|| |global: &Global, _: &mut Local| {
        let indexes = global.page_indexes();
        assert!(indexes.len() >= 2, "{:?}", indexes);

        global.verify_alignment(ALIGNMENT);
        global.verify_non_overlapping(NUMBER_PAGES);
    });

    builder.launch(100);
}

} // mod tests
