//! An integer guaranteed to be a PowerOf2.

use core::{mem, num, ops};

/// PowerOf2
///
/// An integral guaranteed to be non-zero and a power of 2.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct PowerOf2(num::NonZeroUsize);

impl PowerOf2 {
    /// 1 as a PowerOf2 instance.
    //  Safety:
    //  -   1 is a power of 2.
    pub const ONE: PowerOf2 = unsafe { PowerOf2::new_unchecked(1) };

    /// Creates a new instance of PowerOf2.
    ///
    /// Or nothing if the value is not a power of 2.
    pub fn new(value: usize) -> Option<PowerOf2> {
        if value.count_ones() == 1 {
            //  Safety:
            //  -   Value is a power of 2, as per the if check.
            Some(unsafe { PowerOf2::new_unchecked(value) })
        } else {
            None
        }
    }

    /// Creates a new instance of PowerOf2.
    ///
    /// #   Safety
    ///
    /// Assumes that the value is a power of 2.
    pub const unsafe fn new_unchecked(value: usize) -> PowerOf2 {
        //  Safety:
        //  -   A power of 2 cannot be 0.
        PowerOf2(num::NonZeroUsize::new_unchecked(value))
    }

    /// Creates a PowerOf2 matching the alignment of a type.
    pub const fn align_of<T>() -> PowerOf2 {
        //  Safety:
        //  -   Alignment is always a power of 2, and never 0.
        unsafe { PowerOf2::new_unchecked(mem::align_of::<T>()) }
    }

    /// Returns the inner value.
    pub const fn value(&self) -> usize { self.0.get() }

    /// Rounds the value up to the nearest higher multiple of `self`.
    pub const fn round_up(&self, n: usize) -> usize {
        let mask = self.mask();

        (n + mask) & !mask
    }

    /// Rounds the value down to the nearest lower multiple of `self`.
    pub const fn round_down(&self, n: usize) -> usize { n & !self.mask() }

    const fn bit_index(&self) -> usize { self.value().trailing_zeros() as usize }

    const fn mask(&self) -> usize { self.value() - 1 }
}

impl ops::Div for PowerOf2 {
    //  Cannot be PowerOf2, because it could yield 0.
    type Output = usize;

    fn div(self, rhs: PowerOf2) -> usize { self.value() / rhs }
}

impl ops::Div<PowerOf2> for usize {
    type Output = usize;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: PowerOf2) -> usize { self >> rhs.bit_index() }
}

impl ops::Mul for PowerOf2 {
    type Output = PowerOf2;

    fn mul(self, rhs: PowerOf2) -> PowerOf2 { unsafe { PowerOf2::new_unchecked(self.value() * rhs) } }
}

impl ops::Mul<usize> for PowerOf2 {
    type Output = usize;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn mul(self, rhs: usize) -> usize { rhs << self.bit_index() }
}

impl ops::Mul<PowerOf2> for usize {
    type Output = usize;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn mul(self, rhs: PowerOf2) -> usize { self << rhs.bit_index() }
}

impl ops::Rem<PowerOf2> for usize {
    type Output = usize;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn rem(self, rhs: PowerOf2) -> usize { self & rhs.mask() }
}

#[cfg(test)]
mod tests {

use super::*;

#[test]
fn power_of_2_new() {
    fn new(value: usize) -> Option<usize> {
        PowerOf2::new(value).map(|p| p.value())
    }

    assert_eq!(None, new(0));
    assert_eq!(Some(1), new(1));
    assert_eq!(Some(2), new(2));
    assert_eq!(None, new(3));
    assert_eq!(Some(4), new(4));
    assert_eq!(None, new(5));
    assert_eq!(None, new(6));
    assert_eq!(None, new(7));
    assert_eq!(Some(8), new(8));
    assert_eq!(None, new(9));
}

#[test]
fn power_of_2_div() {
    fn div(pow2: usize, n: usize) -> usize {
        n / PowerOf2::new(pow2).expect("Power of 2")
    }

    assert_eq!(0, div(1, 0));
    assert_eq!(1, div(1, 1));
    assert_eq!(2, div(1, 2));
    assert_eq!(3, div(1, 3));

    assert_eq!(0, div(2, 1));
    assert_eq!(1, div(2, 2));
    assert_eq!(1, div(2, 3));
    assert_eq!(2, div(2, 4));

    assert_eq!(0, div(4, 3));
    assert_eq!(1, div(4, 4));
    assert_eq!(1, div(4, 7));
    assert_eq!(2, div(4, 8));
}

#[test]
fn power_of_2_mul() {
    fn mul(pow2: usize, n: usize) -> usize {
        n * PowerOf2::new(pow2).expect("Power of 2")
    }

    assert_eq!(0, mul(1, 0));
    assert_eq!(1, mul(1, 1));
    assert_eq!(2, mul(1, 2));
    assert_eq!(3, mul(1, 3));

    assert_eq!(2, mul(2, 1));
    assert_eq!(4, mul(2, 2));
    assert_eq!(6, mul(2, 3));
    assert_eq!(8, mul(2, 4));

    assert_eq!(12, mul(4, 3));
    assert_eq!(16, mul(4, 4));
    assert_eq!(28, mul(4, 7));
    assert_eq!(32, mul(4, 8));
}

#[test]
fn power_of_2_rem() {
    fn rem(pow2: usize, n: usize) -> usize {
        n % PowerOf2::new(pow2).expect("Power of 2")
    }

    assert_eq!(0, rem(1, 0));
    assert_eq!(0, rem(1, 1));
    assert_eq!(0, rem(1, 2));
    assert_eq!(0, rem(1, 3));

    assert_eq!(0, rem(2, 0));
    assert_eq!(1, rem(2, 1));
    assert_eq!(0, rem(2, 2));
    assert_eq!(1, rem(2, 3));

    assert_eq!(0, rem(4, 0));
    assert_eq!(1, rem(4, 1));
    assert_eq!(2, rem(4, 2));
    assert_eq!(3, rem(4, 3));
    assert_eq!(0, rem(4, 4));
    assert_eq!(1, rem(4, 5));
    assert_eq!(2, rem(4, 6));
    assert_eq!(3, rem(4, 7));
    assert_eq!(0, rem(4, 8));
}

#[test]
fn power_of_2_round_up() {
    fn round_up(pow2: usize, n: usize) -> usize {
        PowerOf2::new(pow2).expect("Power of 2").round_up(n)
    }

    assert_eq!(0, round_up(1, 0));
    assert_eq!(1, round_up(1, 1));
    assert_eq!(2, round_up(1, 2));
    assert_eq!(3, round_up(1, 3));

    assert_eq!(0, round_up(2, 0));
    assert_eq!(2, round_up(2, 1));
    assert_eq!(2, round_up(2, 2));
    assert_eq!(4, round_up(2, 3));
    assert_eq!(4, round_up(2, 4));
    assert_eq!(6, round_up(2, 5));

    assert_eq!(0, round_up(4, 0));
    assert_eq!(4, round_up(4, 1));
    assert_eq!(4, round_up(4, 4));
    assert_eq!(8, round_up(4, 5));
    assert_eq!(8, round_up(4, 8));
    assert_eq!(12, round_up(4, 9));
}

#[test]
fn power_of_2_round_down() {
    fn round_down(pow2: usize, n: usize) -> usize {
        PowerOf2::new(pow2).expect("Power of 2").round_down(n)
    }

    assert_eq!(0, round_down(1, 0));
    assert_eq!(1, round_down(1, 1));
    assert_eq!(2, round_down(1, 2));
    assert_eq!(3, round_down(1, 3));

    assert_eq!(0, round_down(2, 1));
    assert_eq!(2, round_down(2, 2));
    assert_eq!(2, round_down(2, 3));
    assert_eq!(4, round_down(2, 4));
    assert_eq!(4, round_down(2, 5));
    assert_eq!(6, round_down(2, 6));

    assert_eq!(0, round_down(4, 3));
    assert_eq!(4, round_down(4, 4));
    assert_eq!(4, round_down(4, 7));
    assert_eq!(8, round_down(4, 8));
    assert_eq!(8, round_down(4, 11));
    assert_eq!(12, round_down(4, 12));
}

}
