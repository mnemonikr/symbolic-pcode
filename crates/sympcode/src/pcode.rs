use pcode_ops::{BitwisePcodeOps, PcodeOps};
use symbit::{SymbolicBit, SymbolicBitVec, SymbolicByte};

#[repr(transparent)]
#[derive(Debug)]
pub struct SymPcode(SymbolicBitVec);

impl From<SymbolicBitVec> for SymPcode {
    fn from(value: SymbolicBitVec) -> Self {
        Self(value)
    }
}

impl From<SymPcode> for SymbolicBitVec {
    fn from(value: SymPcode) -> Self {
        value.0
    }
}

impl SymPcode {
    pub fn with_variables(vars: impl IntoIterator<Item = usize>) -> Self {
        Self(vars.into_iter().map(SymbolicBit::Variable).collect())
    }

    pub fn into_inner(self) -> SymbolicBitVec {
        self.0
    }
}

impl FromIterator<SymByte> for SymPcode {
    fn from_iter<T: IntoIterator<Item = SymByte>>(iter: T) -> Self {
        Self(iter.into_iter().map(|b| b.0).collect())
    }
}

#[repr(transparent)]
#[derive(Clone)]
pub struct SymByte(SymbolicByte);

impl From<SymBit> for SymByte {
    fn from(value: SymBit) -> Self {
        Self(value.0.into())
    }
}

impl TryFrom<SymByte> for u8 {
    type Error = <Self as TryFrom<SymbolicByte>>::Error;

    fn try_from(value: SymByte) -> Result<Self, Self::Error> {
        value.0.try_into()
    }
}

impl From<u8> for SymByte {
    fn from(value: u8) -> Self {
        Self(value.into())
    }
}

#[repr(transparent)]
#[derive(Clone, Debug)]
pub struct SymBit(SymbolicBit);

impl From<SymBit> for SymbolicBit {
    fn from(value: SymBit) -> Self {
        value.0
    }
}

impl From<bool> for SymBit {
    fn from(value: bool) -> Self {
        Self(value.into())
    }
}

impl BitwisePcodeOps for SymBit {
    fn and(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }

    fn not(self) -> Self {
        Self(!self.0)
    }

    fn or(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }

    fn xor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}

impl TryFrom<SymBit> for bool {
    type Error = <Self as TryFrom<SymbolicBit>>::Error;

    fn try_from(value: SymBit) -> Result<Self, Self::Error> {
        value.0.try_into()
    }
}

impl PcodeOps for SymPcode {
    type Byte = SymByte;
    type Bit = SymBit;

    fn num_bytes(&self) -> usize {
        self.0.num_bytes()
    }

    fn into_le_bytes(self) -> impl ExactSizeIterator<Item = Self::Byte> {
        self.0.into_bytes().into_iter().map(SymByte)
    }

    fn add(self, other: Self) -> Self {
        Self(self.0 + other.0)
    }

    fn unsigned_carry(self, other: Self) -> Self::Bit {
        SymBit(self.0.unsigned_addition_overflow(other.0))
    }

    fn signed_carry(self, other: Self) -> Self::Bit {
        SymBit(self.0.signed_addition_overflow(other.0))
    }

    fn negate(self) -> Self {
        Self(-self.0)
    }

    fn subtract(self, other: Self) -> Self {
        SymPcode(self.0 - other.0)
    }

    fn borrow(self, other: Self) -> Self::Bit {
        SymBit(self.0.subtraction_with_borrow(other.0).1)
    }

    fn multiply(self, other: Self) -> Self {
        let output_size = self.0.len();
        Self(self.0.multiply(other.0, output_size))
    }

    fn unsigned_divide(self, other: Self) -> Self {
        Self(other.0.unsigned_divide(self.0).0)
    }

    fn signed_divide(self, other: Self) -> Self {
        Self(other.0.signed_divide(self.0).0)
    }

    fn unsigned_remainder(self, other: Self) -> Self {
        Self(other.0.unsigned_divide(self.0).1)
    }

    fn signed_remainder(self, other: Self) -> Self {
        Self(other.0.signed_divide(self.0).1)
    }

    fn zero_extend(self, new_size: usize) -> Self {
        let num_bits_extension = (u8::BITS as usize * new_size).saturating_sub(self.0.len());
        if num_bits_extension > 0 {
            Self(self.0.zero_extend(num_bits_extension))
        } else {
            self
        }
    }

    fn sign_extend(self, new_size: usize) -> Self {
        let num_bits_extension = (u8::BITS as usize * new_size).saturating_sub(self.0.len());
        if num_bits_extension > 0 {
            Self(self.0.sign_extend(num_bits_extension))
        } else {
            self
        }
    }

    fn piece(self, lsb: Self) -> Self {
        Self(lsb.0.concat(self.0))
    }

    fn truncate_to_size(self, new_size: usize) -> Self {
        if self.0.num_bytes() > new_size {
            let num_extra_bits = 8 * self.0.num_bytes() - self.0.len();
            let trimmed = self.0.truncate_msb(num_extra_bits);

            let trimmed_len = trimmed.len();
            Self(trimmed.truncate_msb(trimmed_len - 8 * new_size))
        } else {
            self
        }
    }

    fn truncate_trailing_bytes(self, amount: u64) -> Self {
        let truncate_count = amount.try_into().unwrap_or(usize::MAX);
        let truncate_count = truncate_count.saturating_mul(8);
        Self(self.0.truncate_lsb(truncate_count))
    }

    fn lsb(self) -> Self::Bit {
        SymBit(self.0.into_iter().next().unwrap())
    }

    fn popcount(self) -> Self {
        Self(self.0.popcount())
    }

    fn shift_left(self, other: Self) -> Self {
        Self(self.0 << other.0)
    }

    fn unsigned_shift_right(self, other: Self) -> Self {
        Self(self.0 >> other.0)
    }

    fn signed_shift_right(self, other: Self) -> Self {
        Self(self.0.signed_shift_right(other.0))
    }

    fn equals(self, other: Self) -> Self::Bit {
        SymBit(self.0.equals(other.0))
    }

    fn not_equals(self, other: Self) -> Self::Bit {
        SymBit(!self.0.equals(other.0))
    }

    fn unsigned_less_than(self, other: Self) -> Self::Bit {
        SymBit(self.0.less_than(other.0))
    }

    fn signed_less_than(self, other: Self) -> Self::Bit {
        SymBit(self.0.signed_less_than(other.0))
    }

    fn unsigned_less_than_or_equals(self, other: Self) -> Self::Bit {
        SymBit(self.0.less_than_eq(other.0))
    }

    fn signed_less_than_or_equals(self, other: Self) -> Self::Bit {
        SymBit(self.0.signed_less_than_eq(other.0))
    }

    fn signed_greater_than(self, other: Self) -> Self::Bit {
        SymBit(self.0.signed_greater_than(other.0))
    }

    fn signed_greater_than_or_equals(self, other: Self) -> Self::Bit {
        SymBit(self.0.signed_greater_than_eq(other.0))
    }

    fn unsigned_greater_than(self, other: Self) -> Self::Bit {
        SymBit(self.0.greater_than(other.0))
    }

    fn unsigned_greater_than_or_equals(self, other: Self) -> Self::Bit {
        SymBit(self.0.greater_than_eq(other.0))
    }

    fn fill_bytes_with(bit: Self::Bit, num_bytes: usize) -> Self {
        Self(std::iter::repeat_n(bit.0, u8::BITS as usize * num_bytes).collect())
    }
}

impl BitwisePcodeOps for SymPcode {
    fn and(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }

    fn not(self) -> Self {
        Self(!self.0)
    }

    fn or(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }

    fn xor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pcode_ops::validate;

    #[test]
    fn validate_pcode() -> validate::Result {
        validate::Validator::<SymPcode>::validate()
    }
}
