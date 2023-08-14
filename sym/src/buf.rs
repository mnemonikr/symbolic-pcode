use std::mem::MaybeUninit;

use crate::SymbolicBit;

#[derive(Clone)]
pub struct SymbolicBitBuf<const N: usize> {
    bits: [SymbolicBit; N],
}

pub type SymbolicByte = SymbolicBitBuf<8>;

impl<const N: usize> std::ops::Deref for SymbolicBitBuf<N> {
    type Target = [SymbolicBit];

    fn deref(&self) -> &Self::Target {
        &self.bits
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
        const DEFAULT: SymbolicBit = SymbolicBit::Literal(false);
        let bits = [DEFAULT; N];
        Self { bits }
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
            std::mem::MaybeUninit::uninit().assume_init();

        initializer(&mut bits);

        // SAFETY: Assumes all bits are initialized
        let bits = (&bits as *const _ as *const [SymbolicBit; N]).read();

        Self { bits }
    }

    pub fn into_inner(self) -> [SymbolicBit; N] {
        self.bits
    }

    pub fn equals(self, rhs: Self) -> SymbolicBit {
        self.bits
            .into_iter()
            .zip(rhs.bits.into_iter())
            .map(|(lhs, rhs)| lhs.equals(rhs))
            .fold(SymbolicBit::Literal(true), |lhs, rhs| lhs & rhs)
    }

    pub fn concat<const M: usize, const O: usize>(
        self,
        rhs: SymbolicBitBuf<M>,
    ) -> SymbolicBitBuf<O> {
        assert!(N + M == O);
        let initializer = |uninit_bits: &mut [MaybeUninit<SymbolicBit>]| {
            self.bits
                .into_iter()
                .chain(rhs.bits.into_iter())
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
                for i in 0..amount {
                    self.bits[i] = shift_in.clone();
                }
            }
            ShiftDirection::Right => {
                // [ 0 1 2 3 4 5 6 7 ] >> 3
                // [ 3 4 5 6 7 x x x ]
                for i in amount..N {
                    self.bits.swap(i, i - amount);
                }
                for i in 0..amount {
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
            shifted_value.shift(1 << i, SymbolicBit::Literal(false), ShiftDirection::Left);
            self.mux(shifted_value, shift_bit);
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

/// Performs an **unsigned** right shift.
impl<const N: usize> std::ops::ShrAssign for SymbolicBitBuf<N> {
    fn shr_assign(&mut self, rhs: Self) {
        for (i, shift_bit) in rhs.bits.into_iter().enumerate() {
            let mut shifted_value = self.clone();
            shifted_value.shift(1 << i, SymbolicBit::Literal(false), ShiftDirection::Right);
            self.mux(shifted_value, shift_bit);
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
