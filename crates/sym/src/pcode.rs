use crate::{SymbolicBit, SymbolicBitVec, SymbolicByte};

pub trait PcodeOps: BitwisePcodeOps {
    type Bit: BitwisePcodeOps + TryInto<bool>;

    fn add(self, other: Self) -> Self;
    fn unsigned_carry(self, other: Self) -> Self::Bit;
    fn signed_carry(self, other: Self) -> Self::Bit;
    fn negate(self) -> Self;
    fn subtract(self, other: Self) -> Self;
    fn borrow(self, other: Self) -> Self::Bit;
    fn multiply(self, other: Self) -> Self;
    fn unsigned_divide(self, other: Self) -> Self;
    fn signed_divide(self, other: Self) -> Self;
    fn unsigned_remainder(self, other: Self) -> Self;
    fn signed_remainder(self, other: Self) -> Self;
    fn zero_extend(self, amount: usize) -> Self;
    fn sign_extend(self, amount: usize) -> Self;
    fn piece(self, lsb: Self) -> Self;

    // Subpiece is implemented as a combination of these truncation operations
    fn truncate_leading_bytes(self, amount: u64) -> Self;
    fn truncate_trailing_bytes(self, amount: u64) -> Self;

    fn lsb(self) -> Self::Bit;

    fn shift_left(self, other: Self) -> Self;
    fn unsigned_shift_right(self, other: Self) -> Self;
    fn signed_shift_right(self, other: Self) -> Self;

    fn equals(self, other: Self) -> Self::Bit;
    fn unsigned_less_than(self, other: Self) -> Self::Bit;
    fn signed_less_than(self, other: Self) -> Self::Bit;

    fn not_equals(self, other: Self) -> Self::Bit;
    fn unsigned_less_than_or_equals(self, other: Self) -> Self::Bit;
    fn signed_less_than_or_equals(self, other: Self) -> Self::Bit;

    fn signed_greater_than(self, other: Self) -> Self::Bit;
    fn signed_greater_than_or_equals(self, other: Self) -> Self::Bit;
    fn unsigned_greater_than(self, other: Self) -> Self::Bit;
    fn unsigned_greater_than_or_equals(self, other: Self) -> Self::Bit;
}

pub trait BitwisePcodeOps {
    fn and(self, other: Self) -> Self;
    fn negate(self) -> Self;
    fn or(self, other: Self) -> Self;
    fn xor(self, other: Self) -> Self;
}

impl PcodeOps for SymbolicBitVec {
    type Bit = SymbolicBit;

    fn add(self, other: Self) -> Self {
        self + other
    }

    fn unsigned_carry(self, other: Self) -> Self::Bit {
        self.unsigned_addition_overflow(other)
    }

    fn signed_carry(self, other: Self) -> Self::Bit {
        self.signed_addition_overflow(other)
    }

    fn negate(self) -> Self {
        -self
    }

    fn subtract(self, other: Self) -> Self {
        self - other
    }

    fn borrow(self, other: Self) -> Self::Bit {
        self.subtraction_with_borrow(other).1
    }

    fn multiply(self, other: Self) -> Self {
        let output_size = self.len();
        self.multiply(other, output_size)
    }

    fn unsigned_divide(self, other: Self) -> Self {
        self.unsigned_divide(other).0
    }

    fn signed_divide(self, other: Self) -> Self {
        self.signed_divide(other).0
    }

    fn unsigned_remainder(self, other: Self) -> Self {
        self.unsigned_divide(other).1
    }

    fn signed_remainder(self, other: Self) -> Self {
        self.signed_divide(other).1
    }

    fn zero_extend(self, amount: usize) -> Self {
        self.zero_extend(8 * amount)
    }

    fn sign_extend(self, amount: usize) -> Self {
        self.sign_extend(8 * amount)
    }

    fn piece(self, lsb: Self) -> Self {
        lsb.concat(self)
    }

    fn truncate_leading_bytes(self, amount: u64) -> Self {
        let truncate_count = amount.try_into().unwrap_or(usize::MAX);
        let truncate_count = truncate_count.saturating_mul(8);
        self.truncate_msb(truncate_count)
    }

    fn truncate_trailing_bytes(self, amount: u64) -> Self {
        let truncate_count = amount.try_into().unwrap_or(usize::MAX);
        let truncate_count = truncate_count.saturating_mul(8);
        self.truncate_lsb(truncate_count)
    }

    fn shift_left(self, other: Self) -> Self {
        self << other
    }

    fn unsigned_shift_right(self, other: Self) -> Self {
        self >> other
    }

    fn signed_shift_right(self, other: Self) -> Self {
        self.signed_shift_right(other)
    }

    fn equals(self, other: Self) -> Self::Bit {
        self.equals(other)
    }

    fn not_equals(self, other: Self) -> Self::Bit {
        !self.equals(other)
    }

    fn unsigned_less_than(self, other: Self) -> Self::Bit {
        self.less_than(other)
    }

    fn signed_less_than(self, other: Self) -> Self::Bit {
        self.signed_less_than(other)
    }

    fn unsigned_less_than_or_equals(self, other: Self) -> Self::Bit {
        self.less_than_eq(other)
    }

    fn signed_less_than_or_equals(self, other: Self) -> Self::Bit {
        self.signed_less_than_eq(other)
    }

    fn unsigned_greater_than(self, other: Self) -> Self::Bit {
        self.greater_than(other)
    }

    fn signed_greater_than(self, other: Self) -> Self::Bit {
        self.signed_greater_than(other)
    }

    fn unsigned_greater_than_or_equals(self, other: Self) -> Self::Bit {
        self.greater_than_eq(other)
    }

    fn signed_greater_than_or_equals(self, other: Self) -> Self::Bit {
        self.signed_greater_than_eq(other)
    }

    fn lsb(self) -> Self::Bit {
        self.into_iter().next().unwrap()
    }
}

impl<T> BitwisePcodeOps for T
where
    T: std::ops::BitAnd<Output = T>
        + std::ops::BitOr<Output = T>
        + std::ops::BitXor<Output = T>
        + std::ops::Not<Output = T>,
{
    fn and(self, other: Self) -> Self {
        self & other
    }

    fn negate(self) -> Self {
        !self
    }

    fn or(self, other: Self) -> Self {
        self | other
    }

    fn xor(self, other: Self) -> Self {
        self ^ other
    }
}
