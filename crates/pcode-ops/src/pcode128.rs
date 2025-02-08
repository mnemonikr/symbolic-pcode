use crate::{BitwisePcodeOps, PcodeOps};

/// Representation of a value with a bit size no greater than 128.
#[derive(Copy, Clone, Debug)]
pub struct Pcode128 {
    value: u128,
    valid_bits: u32,
}

/// Default value is a 0 byte (8 bits).
impl Default for Pcode128 {
    fn default() -> Self {
        Self {
            value: 0,
            valid_bits: 8,
        }
    }
}

impl Pcode128 {
    const MAX_BYTES: u32 = const { u128::BITS / u8::BITS };

    pub fn new(value: u128, valid_bits: u32) -> Self {
        assert!(valid_bits <= u128::BITS && valid_bits > 0);
        let mask = u128::MAX >> (u128::BITS - valid_bits);

        Self {
            value: value & mask,
            valid_bits,
        }
    }

    fn bitmask(&self) -> u128 {
        u128::MAX >> (u128::BITS - self.valid_bits)
    }

    fn map(&self, f: impl Fn(u128) -> u128) -> Self {
        Self::new(f(self.value), self.valid_bits)
    }

    fn signed_map(&self, f: impl Fn(i128) -> i128) -> Self {
        Self::new(f(self.signed_value()) as u128, self.valid_bits)
    }

    pub fn value(&self) -> u128 {
        self.value
    }

    pub fn signed_value(&self) -> i128 {
        if self.valid_bits == u128::BITS {
            return self.value as i128;
        }

        let sign_bit_mask = 1 << (self.valid_bits - 1);
        let value = if self.value & sign_bit_mask > 0 {
            self.value | (u128::MAX << self.valid_bits)
        } else {
            self.value
        };

        value as i128
    }
}

impl From<u128> for Pcode128 {
    fn from(value: u128) -> Self {
        Pcode128::new(value, u128::BITS)
    }
}

impl From<Pcode128> for u128 {
    fn from(pcode_value: Pcode128) -> Self {
        pcode_value.value()
    }
}

macro_rules! impl_from_value {
    ($type:ty) => {
        impl From<$type> for Pcode128 {
            fn from(value: $type) -> Self {
                Pcode128::new(value.into(), <$type>::BITS)
            }
        }

        impl TryFrom<Pcode128> for $type {
            type Error = std::num::TryFromIntError;
            fn try_from(value: Pcode128) -> Result<Self, Self::Error> {
                <$type>::try_from(value.value())
            }
        }
    };
}

impl_from_value!(u64);
impl_from_value!(u32);
impl_from_value!(u16);
impl_from_value!(u8);

impl BitwisePcodeOps for Pcode128 {
    fn and(self, rhs: Self) -> Self {
        self.map(|value| value & rhs.value())
    }

    fn not(self) -> Self {
        self.map(|value| !value)
    }

    fn or(self, rhs: Self) -> Self {
        self.map(|value| value | rhs.value())
    }

    fn xor(self, rhs: Self) -> Self {
        self.map(|value| value ^ rhs.value())
    }
}

impl PcodeOps for Pcode128 {
    type Bit = bool;
    type Byte = u8;

    fn num_bytes(&self) -> usize {
        self.valid_bits.div_ceil(u8::BITS) as usize
    }

    fn add(self, rhs: Self) -> Self {
        assert_eq!(self.valid_bits, rhs.valid_bits);
        self.map(|value| value.wrapping_add(rhs.value()))
    }

    fn unsigned_carry(self, rhs: Self) -> Self::Bit {
        assert_eq!(self.valid_bits, rhs.valid_bits);
        let value = self.value().checked_add(rhs.value());
        if let Some(value) = value {
            value != Pcode128::new(value, self.valid_bits).value()
        } else {
            true
        }
    }

    fn signed_carry(self, rhs: Self) -> Self::Bit {
        assert_eq!(self.valid_bits, rhs.valid_bits);
        let value = self.signed_value().checked_add(rhs.signed_value());
        if let Some(value) = value {
            value != Pcode128::new(value as u128, self.valid_bits).signed_value()
        } else {
            true
        }
    }

    fn negate(self) -> Self {
        self.signed_map(|value| -value)
    }

    fn subtract(self, rhs: Self) -> Self {
        assert_eq!(self.valid_bits, rhs.valid_bits);
        self.map(|value| value.wrapping_sub(rhs.value()))
    }

    fn borrow(self, rhs: Self) -> Self::Bit {
        assert_eq!(self.valid_bits, rhs.valid_bits);
        let value = self.signed_value().checked_sub(rhs.signed_value());
        if let Some(value) = value {
            value != Pcode128::new(value as u128, self.valid_bits).signed_value()
        } else {
            true
        }
    }

    fn multiply(self, rhs: Self) -> Self {
        assert_eq!(self.valid_bits, rhs.valid_bits);
        self.map(|value| value.wrapping_mul(rhs.value()))
    }

    fn unsigned_divide(self, rhs: Self) -> Self {
        assert_eq!(self.valid_bits, rhs.valid_bits);
        self.map(|value| value.wrapping_div(rhs.value()))
    }

    fn signed_divide(self, rhs: Self) -> Self {
        assert_eq!(self.valid_bits, rhs.valid_bits);
        self.signed_map(|value| value.wrapping_div(rhs.signed_value()))
    }

    fn unsigned_remainder(self, rhs: Self) -> Self {
        assert_eq!(self.valid_bits, rhs.valid_bits);
        self.map(|value| value.wrapping_rem(rhs.value()))
    }

    fn signed_remainder(self, rhs: Self) -> Self {
        assert_eq!(self.valid_bits, rhs.valid_bits);
        self.signed_map(|value| value.wrapping_rem(rhs.signed_value()))
    }

    fn zero_extend(self, new_size: usize) -> Self {
        Pcode128::new(self.value(), u8::BITS * new_size as u32)
    }

    fn sign_extend(self, new_size: usize) -> Self {
        Pcode128::new(self.signed_value() as u128, u8::BITS * new_size as u32)
    }

    fn piece(self, lsb: Self) -> Self {
        Pcode128::new(
            (self.value() << lsb.valid_bits) | lsb.value(),
            self.valid_bits + lsb.valid_bits,
        )
    }

    fn truncate_to_size(self, new_size: usize) -> Self {
        Pcode128::new(self.value(), u8::BITS * new_size as u32)
    }

    fn truncate_trailing_bytes(self, amount: u64) -> Self {
        let invalid_bits = u8::BITS * amount as u32;
        Pcode128::new(self.value() >> invalid_bits, self.valid_bits - invalid_bits)
    }

    fn lsb(self) -> Self::Bit {
        self.value() & 0x1 > 0
    }

    fn popcount(self) -> Self {
        Pcode128::new(
            self.value().count_ones() as u128,
            const { u128::BITS.ilog2() },
        )
    }

    fn shift_left(self, rhs: Self) -> Self {
        self.map(|value| value << rhs.value())
    }

    fn unsigned_shift_right(self, rhs: Self) -> Self {
        self.map(|value| value >> rhs.value())
    }

    fn signed_shift_right(self, rhs: Self) -> Self {
        self.signed_map(|value| value >> rhs.value())
    }

    fn equals(self, rhs: Self) -> Self::Bit {
        self.value() == rhs.value()
    }

    fn not_equals(self, rhs: Self) -> Self::Bit {
        self.value() != rhs.value()
    }

    fn unsigned_less_than(self, rhs: Self) -> Self::Bit {
        self.value() < rhs.value()
    }

    fn signed_less_than(self, rhs: Self) -> Self::Bit {
        self.signed_value() < rhs.signed_value()
    }

    fn unsigned_less_than_or_equals(self, rhs: Self) -> Self::Bit {
        self.value() <= rhs.value()
    }

    fn signed_less_than_or_equals(self, rhs: Self) -> Self::Bit {
        self.signed_value() <= rhs.signed_value()
    }

    fn signed_greater_than(self, rhs: Self) -> Self::Bit {
        self.signed_value() > rhs.signed_value()
    }

    fn signed_greater_than_or_equals(self, rhs: Self) -> Self::Bit {
        self.signed_value() >= rhs.signed_value()
    }

    fn unsigned_greater_than(self, rhs: Self) -> Self::Bit {
        self.value() > rhs.value()
    }

    fn unsigned_greater_than_or_equals(self, rhs: Self) -> Self::Bit {
        self.value() >= rhs.value()
    }

    fn fill_bytes_with(bit: Self::Bit, num_bytes: usize) -> Self {
        std::iter::repeat(if bit { u8::MAX } else { 0 })
            .take(num_bytes)
            .collect()
    }

    fn into_le_bytes(self) -> impl ExactSizeIterator<Item = Self::Byte> {
        self.value()
            .to_le_bytes()
            .into_iter()
            .take(self.num_bytes())
    }
}

impl FromIterator<u8> for Pcode128 {
    fn from_iter<T: IntoIterator<Item = u8>>(iter: T) -> Self {
        let mut buffer = [0u8; Pcode128::MAX_BYTES as usize];
        let mut valid_bits = 0;
        for (i, byte) in iter.into_iter().enumerate() {
            buffer[i] = byte;
            valid_bits += 8;
        }
        Pcode128::new(u128::from_le_bytes(buffer), valid_bits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validate::*;

    #[test]
    fn validate_pcode128_impl() -> Result {
        Validator::<Pcode128>::validate()
    }
}
