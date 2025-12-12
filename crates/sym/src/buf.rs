//! Module for a collection of symbolic bits whose count are known at compile-time.
use std::mem::MaybeUninit;
use std::ops::Deref;

use crate::bit::{FALSE, SymbolicBit};

/// An array of symbolic bits.
#[derive(PartialEq, Eq, Debug, Clone)]
pub struct SymbolicBitBuf<const N: usize> {
    bits: [SymbolicBit; N],
}

impl<const N: usize> IntoIterator for SymbolicBitBuf<N> {
    type Item = SymbolicBit;
    type IntoIter = std::array::IntoIter<SymbolicBit, N>;

    fn into_iter(self) -> Self::IntoIter {
        self.bits.into_iter()
    }
}

/// An 8-bit byte of symbolic bits.
pub type SymbolicByte = SymbolicBitBuf<8>;

impl<const N: usize> std::ops::Deref for SymbolicBitBuf<N> {
    type Target = [SymbolicBit];

    fn deref(&self) -> &Self::Target {
        &self.bits
    }
}

impl<const N: usize, T> AsRef<T> for SymbolicBitBuf<N>
where
    T: ?Sized,
    <SymbolicBitBuf<N> as std::ops::Deref>::Target: AsRef<T>,
{
    fn as_ref(&self) -> &T {
        self.deref().as_ref()
    }
}

impl<const N: usize> From<SymbolicBit> for SymbolicBitBuf<N> {
    fn from(bit: SymbolicBit) -> Self {
        let mut result = Self::default();
        result[0] = bit;
        result
    }
}

impl<const N: usize> From<[SymbolicBit; N]> for SymbolicBitBuf<N> {
    fn from(bits: [SymbolicBit; N]) -> Self {
        Self { bits }
    }
}

impl<const N: usize> From<SymbolicBitBuf<N>> for [SymbolicBit; N] {
    fn from(buf: SymbolicBitBuf<N>) -> Self {
        buf.bits
    }
}

impl<const N: usize> Default for SymbolicBitBuf<N> {
    fn default() -> Self {
        Self { bits: [FALSE; N] }
    }
}

impl<const N: usize> std::ops::Index<usize> for SymbolicBitBuf<N> {
    type Output = SymbolicBit;
    fn index(&self, index: usize) -> &Self::Output {
        &self.bits[index]
    }
}

impl<const N: usize> std::ops::IndexMut<usize> for SymbolicBitBuf<N> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.bits[index]
    }
}

impl<const N: usize> From<SymbolicBitBuf<N>> for Vec<SymbolicByte> {
    fn from(mut value: SymbolicBitBuf<N>) -> Self {
        const {
            assert!(N.is_multiple_of(8));
        }

        let mut result = Vec::with_capacity(N / 8);
        let (buf_bytes, _) = value.bits.as_chunks_mut::<8>();
        for buf_byte in buf_bytes {
            let mut byte = [FALSE; 8];
            std::mem::swap(&mut byte, buf_byte);
            result.push(SymbolicByte::from(byte));
        }

        result
    }
}

impl<const N: usize> TryFrom<Vec<SymbolicByte>> for SymbolicBitBuf<N> {
    type Error = String;

    fn try_from(value: Vec<SymbolicByte>) -> Result<Self, Self::Error> {
        if N == 8 * value.len() {
            let initializer = |uninit_bits: &mut [MaybeUninit<SymbolicBit>]| {
                value
                    .into_iter()
                    .flat_map(|byte| byte.into_inner().into_iter())
                    .enumerate()
                    .for_each(|(i, bit)| {
                        uninit_bits[i].write(bit);
                    })
            };

            // SAFETY: All bits are initialized
            unsafe { Ok(SymbolicBitBuf::<N>::initialize(initializer)) }
        } else {
            Err(format!(
                "value has {num_bits} bits, expected {N} bits",
                num_bits = 8 * value.len(),
            ))
        }
    }
}

impl<const N: usize> TryFrom<Vec<&SymbolicByte>> for SymbolicBitBuf<N> {
    type Error = String;

    fn try_from(value: Vec<&SymbolicByte>) -> Result<Self, Self::Error> {
        if N == 8 * value.len() {
            let initializer = |uninit_bits: &mut [MaybeUninit<SymbolicBit>]| {
                value
                    .into_iter()
                    .flat_map(|byte| byte.inner().iter())
                    .cloned()
                    .enumerate()
                    .for_each(|(i, bit)| {
                        uninit_bits[i].write(bit);
                    })
            };

            // SAFETY: All bits are initialized
            unsafe { Ok(SymbolicBitBuf::<N>::initialize(initializer)) }
        } else {
            Err(format!(
                "value has {num_bits} bits, expected {N} bits",
                num_bits = 8 * value.len(),
            ))
        }
    }
}

enum ShiftDirection {
    Left,
    Right,
}

impl<const N: usize> SymbolicBitBuf<N> {
    /// Create a new `SymbolicBitBuf` with the bits initialized by the callback.
    ///
    /// # Safety
    ///
    /// All of the bits must be initialized by the initializer callback.
    #[inline]
    unsafe fn initialize<F>(initializer: F) -> Self
    where
        F: FnOnce(&mut [MaybeUninit<SymbolicBit>]),
    {
        let mut bits: [std::mem::MaybeUninit<SymbolicBit>; N] =
            unsafe { std::mem::MaybeUninit::uninit().assume_init() };

        initializer(&mut bits);

        // SAFETY: Assumes all bits are initialized
        let bits = unsafe { (&bits as *const _ as *const [SymbolicBit; N]).read() };

        Self { bits }
    }

    #[must_use]
    pub fn into_inner(self) -> [SymbolicBit; N] {
        self.bits
    }

    #[must_use]
    pub fn inner(&self) -> &[SymbolicBit; N] {
        &self.bits
    }

    #[must_use]
    pub fn equals(self, rhs: Self) -> SymbolicBit {
        self.bits
            .into_iter()
            .zip(rhs.bits)
            .map(|(lhs, rhs)| lhs.equals(rhs))
            .fold(SymbolicBit::Literal(true), |lhs, rhs| lhs & rhs)
    }

    /// Concatenates this buffer with another, resulting in a single buffer containing `[self:rhs]`.
    ///
    /// # Future
    ///
    /// Once [generic_const_exprs](https://github.com/rust-lang/rust/issues/76560) is stabilized,
    /// the value of `O` can be computed from `N` and `M` and will no longer be part of the type
    /// signature.
    #[must_use]
    pub fn concat<const M: usize, const O: usize>(
        self,
        rhs: SymbolicBitBuf<M>,
    ) -> SymbolicBitBuf<O> {
        const {
            assert!(N + M == O, "output size must be sum of buffer sizes");
        }

        let initializer = |uninit_bits: &mut [MaybeUninit<SymbolicBit>]| {
            self.bits
                .into_iter()
                .chain(rhs.bits)
                .enumerate()
                .for_each(|(i, bit)| {
                    uninit_bits[i].write(bit);
                })
        };

        // SAFETY: All bits are initialized
        unsafe { SymbolicBitBuf::<O>::initialize(initializer) }
    }

    fn shift(&mut self, amount: usize, shift_in: SymbolicBit, direction: ShiftDirection) {
        match direction {
            ShiftDirection::Left => {
                // [ 0 1 2 3 4 5 6 7 ] << 3
                // [ x x x 0 1 2 3 4 ]
                for i in (amount..N).rev() {
                    self.bits.swap(i, i - amount);
                }
                for i in 0..usize::min(N, amount) {
                    self.bits[i] = shift_in.clone();
                }
            }
            ShiftDirection::Right => {
                // [ 0 1 2 3 4 5 6 7 ] >> 3
                // [ 3 4 5 6 7 x x x ]
                for i in amount..N {
                    self.bits.swap(i, i - amount);
                }
                for i in 0..usize::min(N, amount) {
                    self.bits[N - 1 - i] = shift_in.clone();
                }
            }
        }
    }

    fn mux(&mut self, rhs: Self, selector: SymbolicBit) {
        rhs.bits.into_iter().enumerate().for_each(|(i, rhs)| {
            let lhs = std::mem::take(&mut self.bits[i]);
            self.bits[i] = selector.clone().select(lhs, rhs);
        });
    }
}

impl<const N: usize> std::ops::Not for SymbolicBitBuf<N> {
    type Output = Self;

    fn not(mut self) -> Self::Output {
        for i in 0..N {
            let bit = std::mem::take(&mut self.bits[i]);
            self.bits[i] = !bit;
        }

        self
    }
}

impl<const N: usize> std::ops::BitAndAssign for SymbolicBitBuf<N> {
    fn bitand_assign(&mut self, rhs: Self) {
        rhs.bits.into_iter().enumerate().for_each(|(i, rhs)| {
            let lhs = std::mem::take(&mut self.bits[i]);
            self.bits[i] = lhs & rhs;
        });
    }
}

impl<const N: usize> std::ops::BitOrAssign for SymbolicBitBuf<N> {
    fn bitor_assign(&mut self, rhs: Self) {
        rhs.bits.into_iter().enumerate().for_each(|(i, rhs)| {
            let lhs = std::mem::take(&mut self.bits[i]);
            self.bits[i] = lhs | rhs;
        });
    }
}

impl<const N: usize> std::ops::BitXorAssign for SymbolicBitBuf<N> {
    fn bitxor_assign(&mut self, rhs: Self) {
        rhs.bits.into_iter().enumerate().for_each(|(i, rhs)| {
            let lhs = std::mem::take(&mut self.bits[i]);
            self.bits[i] = lhs ^ rhs;
        });
    }
}

impl<const N: usize> std::ops::BitAnd for SymbolicBitBuf<N> {
    type Output = Self;

    fn bitand(mut self, rhs: Self) -> Self::Output {
        self &= rhs;
        self
    }
}

impl<const N: usize> std::ops::BitOr for SymbolicBitBuf<N> {
    type Output = Self;

    fn bitor(mut self, rhs: Self) -> Self::Output {
        self |= rhs;
        self
    }
}

impl<const N: usize> std::ops::BitXor for SymbolicBitBuf<N> {
    type Output = Self;

    fn bitxor(mut self, rhs: Self) -> Self::Output {
        self ^= rhs;
        self
    }
}

impl<const N: usize> std::ops::ShlAssign for SymbolicBitBuf<N> {
    fn shl_assign(&mut self, rhs: Self) {
        for (i, shift_bit) in rhs.bits.into_iter().enumerate() {
            let mut shifted_value = self.clone();
            shifted_value.shift(1 << i, FALSE, ShiftDirection::Left);
            self.mux(shifted_value, !shift_bit);
        }
    }
}

impl<const N: usize> std::ops::Shl for SymbolicBitBuf<N> {
    type Output = Self;

    fn shl(mut self, rhs: Self) -> Self::Output {
        self <<= rhs;
        self
    }
}

impl<const N: usize> std::ops::ShlAssign<usize> for SymbolicBitBuf<N> {
    fn shl_assign(&mut self, rhs: usize) {
        self.shift(rhs, FALSE, ShiftDirection::Left);
    }
}

impl<const N: usize> std::ops::Shl<usize> for SymbolicBitBuf<N> {
    type Output = Self;

    fn shl(mut self, rhs: usize) -> Self::Output {
        self <<= rhs;
        self
    }
}

/// Performs an **unsigned** right shift.
impl<const N: usize> std::ops::ShrAssign for SymbolicBitBuf<N> {
    fn shr_assign(&mut self, rhs: Self) {
        for (i, shift_bit) in rhs.bits.into_iter().enumerate() {
            let mut shifted_value = self.clone();
            shifted_value.shift(1 << i, FALSE, ShiftDirection::Right);
            self.mux(shifted_value, !shift_bit);
        }
    }
}

impl<const N: usize> std::ops::Shr for SymbolicBitBuf<N> {
    type Output = Self;

    fn shr(mut self, rhs: Self) -> Self::Output {
        self >>= rhs;
        self
    }
}

impl<const N: usize> std::ops::ShrAssign<usize> for SymbolicBitBuf<N> {
    fn shr_assign(&mut self, rhs: usize) {
        self.shift(rhs, FALSE, ShiftDirection::Right);
    }
}

impl<const N: usize> std::ops::Shr<usize> for SymbolicBitBuf<N> {
    type Output = Self;

    fn shr(mut self, rhs: usize) -> Self::Output {
        self >>= rhs;
        self
    }
}
