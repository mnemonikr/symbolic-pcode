use crate::{SymbolicBit, SymbolicBitVec, SymbolicByte};
use pcode_ops::PcodeOps;

impl PcodeOps for SymbolicBitVec {
    type Byte = SymbolicByte;
    type Bit = SymbolicBit;

    fn num_bytes(&self) -> usize {
        self.num_bytes()
    }

    fn into_le_bytes(self) -> impl ExactSizeIterator<Item = Self::Byte> {
        self.into_bytes().into_iter()
    }

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
        other.unsigned_divide(self).0
    }

    fn signed_divide(self, other: Self) -> Self {
        other.signed_divide(self).0
    }

    fn unsigned_remainder(self, other: Self) -> Self {
        other.unsigned_divide(self).1
    }

    fn signed_remainder(self, other: Self) -> Self {
        other.signed_divide(self).1
    }

    fn zero_extend(self, new_size: usize) -> Self {
        let num_bits_extension = (u8::BITS as usize * new_size).saturating_sub(self.len());
        if num_bits_extension > 0 {
            self.zero_extend(num_bits_extension)
        } else {
            self
        }
    }

    fn sign_extend(self, new_size: usize) -> Self {
        let num_bits_extension = (u8::BITS as usize * new_size).saturating_sub(self.len());
        if num_bits_extension > 0 {
            self.sign_extend(num_bits_extension)
        } else {
            self
        }
    }

    fn piece(self, lsb: Self) -> Self {
        lsb.concat(self)
    }

    fn truncate_to_size(self, new_size: usize) -> Self {
        if self.num_bytes() > new_size {
            let num_extra_bits = 8 * self.num_bytes() - self.len();
            let trimmed = self.truncate_msb(num_extra_bits);

            let trimmed_len = trimmed.len();
            trimmed.truncate_msb(trimmed_len - 8 * new_size)
        } else {
            self
        }
    }

    fn truncate_trailing_bytes(self, amount: u64) -> Self {
        let truncate_count = amount.try_into().unwrap_or(usize::MAX);
        let truncate_count = truncate_count.saturating_mul(8);
        self.truncate_lsb(truncate_count)
    }

    fn lsb(self) -> Self::Bit {
        self.into_iter().next().unwrap()
    }

    fn popcount(self) -> Self {
        self.popcount()
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

    fn signed_greater_than(self, other: Self) -> Self::Bit {
        self.signed_greater_than(other)
    }

    fn signed_greater_than_or_equals(self, other: Self) -> Self::Bit {
        self.signed_greater_than_eq(other)
    }

    fn unsigned_greater_than(self, other: Self) -> Self::Bit {
        self.greater_than(other)
    }

    fn unsigned_greater_than_or_equals(self, other: Self) -> Self::Bit {
        self.greater_than_eq(other)
    }

    fn predicated_on(self, condition: Self::Bit) -> Self {
        std::iter::repeat(!condition)
            .take(self.len())
            .collect::<Self>()
            | self
    }

    fn assert(self, condition: Self::Bit) -> Self {
        std::iter::repeat(condition)
            .take(self.len())
            .collect::<Self>()
            & self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pcode_ops::validate;

    #[test]
    fn validate_pcode() -> validate::Result {
        validate::Validator::<SymbolicBitVec>::validate()
    }
}
