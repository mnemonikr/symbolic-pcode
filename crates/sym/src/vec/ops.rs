use super::{ShiftDirection, SymbolicBitVec};
use crate::bit::{FALSE, SymbolicBit};

impl std::ops::Index<usize> for SymbolicBitVec {
    type Output = SymbolicBit;

    fn index(&self, index: usize) -> &Self::Output {
        &self.bits[index]
    }
}

impl std::ops::Not for SymbolicBitVec {
    type Output = Self;

    fn not(self) -> Self::Output {
        Self {
            bits: self.bits.into_iter().map(|bit| !bit).collect(),
        }
    }
}

impl std::ops::BitAnd for SymbolicBitVec {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        assert_eq!(self.bits.len(), rhs.bits.len());
        Self {
            bits: self
                .bits
                .into_iter()
                .zip(rhs.bits)
                .map(|(lhs, rhs)| lhs & rhs)
                .collect(),
        }
    }
}

impl std::ops::BitOr for SymbolicBitVec {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        assert_eq!(self.bits.len(), rhs.bits.len());
        Self {
            bits: self
                .bits
                .into_iter()
                .zip(rhs.bits)
                .map(|(lhs, rhs)| lhs | rhs)
                .collect(),
        }
    }
}

impl std::ops::BitXor for SymbolicBitVec {
    type Output = Self;

    fn bitxor(self, rhs: Self) -> Self::Output {
        assert_eq!(self.bits.len(), rhs.bits.len());
        Self {
            bits: self
                .bits
                .into_iter()
                .zip(rhs.bits)
                .map(|(lhs, rhs)| lhs ^ rhs)
                .collect(),
        }
    }
}

impl std::ops::Shl<usize> for SymbolicBitVec {
    type Output = Self;

    fn shl(mut self, rhs: usize) -> Self::Output {
        self <<= rhs;
        self
    }
}

impl std::ops::Shl for SymbolicBitVec {
    type Output = Self;

    fn shl(mut self, rhs: Self) -> Self::Output {
        self <<= rhs;
        self
    }
}

impl std::ops::ShlAssign<usize> for SymbolicBitVec {
    fn shl_assign(&mut self, rhs: usize) {
        self.shift_mut(rhs, FALSE, ShiftDirection::Left);
    }
}

impl std::ops::ShlAssign for SymbolicBitVec {
    fn shl_assign(&mut self, rhs: Self) {
        for (i, shift_bit) in rhs.bits.into_iter().enumerate() {
            let mut shifted_value = self.clone();
            shifted_value.shift_mut(1 << i, FALSE, ShiftDirection::Left);
            self.mux_mut(shifted_value, !shift_bit);
        }
    }
}

/// Performs an _unsigned_ right shift.
impl std::ops::ShrAssign for SymbolicBitVec {
    fn shr_assign(&mut self, rhs: Self) {
        for (i, shift_bit) in rhs.bits.into_iter().enumerate() {
            let mut shifted_value = self.clone();
            shifted_value.shift_mut(1 << i, FALSE, ShiftDirection::Right);
            self.mux_mut(shifted_value, !shift_bit);
        }
    }
}

impl std::ops::ShrAssign<usize> for SymbolicBitVec {
    fn shr_assign(&mut self, rhs: usize) {
        self.shift_mut(rhs, FALSE, ShiftDirection::Right);
    }
}

impl std::ops::Shr for SymbolicBitVec {
    type Output = Self;

    fn shr(mut self, rhs: Self) -> Self::Output {
        self >>= rhs;
        self
    }
}

impl std::ops::Shr<usize> for SymbolicBitVec {
    type Output = Self;

    fn shr(mut self, rhs: usize) -> Self::Output {
        self >>= rhs;
        self
    }
}

impl std::ops::Add for SymbolicBitVec {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        assert_eq!(self.bits.len(), rhs.bits.len());
        // The carry bit size is actually N+1 in order to track whether an overflow has occurred.
        // The addition does not care about the overflow so remove this bit
        let (sum, _) = self.addition_with_carry(rhs);
        sum
    }
}

impl std::ops::Neg for SymbolicBitVec {
    type Output = Self;

    fn neg(self) -> Self::Output {
        let num_bits = self.bits.len();
        !self + SymbolicBitVec::constant(1, num_bits)
    }
}

impl std::ops::Sub for SymbolicBitVec {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        assert_eq!(self.bits.len(), rhs.bits.len());
        self + (-rhs)
    }
}

impl std::ops::Mul for SymbolicBitVec {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        // The output size is the sum of the number of bits. Clippy is overly zealous here flagging
        // the addition as erroneous. https://github.com/rust-lang/rust-clippy/issues/16247
        #[allow(clippy::suspicious_arithmetic_impl)]
        let output_size = self.len() + rhs.len();

        self.multiply(rhs, output_size)
    }
}
