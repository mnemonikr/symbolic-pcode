use pcode_ops::{BitwisePcodeOps, PcodeOps};

#[derive(Copy, Clone, Debug)]
pub struct ConcreteValue {
    value: u128,
    valid_bits: u32,
}

impl Default for ConcreteValue {
    fn default() -> Self {
        Self {
            value: 0,
            valid_bits: 8,
        }
    }
}

impl ConcreteValue {
    const MAX_BYTES: u32 = u128::BITS / u8::BITS;

    pub fn new(value: u128, valid_bits: u32) -> Self {
        assert!(valid_bits <= u128::BITS);
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

macro_rules! impl_from_value {
    ($type:ty) => {
        impl From<$type> for ConcreteValue {
            fn from(value: $type) -> Self {
                ConcreteValue::new(value.into(), <$type>::BITS)
            }
        }

        impl TryFrom<ConcreteValue> for $type {
            type Error = std::num::TryFromIntError;
            fn try_from(value: ConcreteValue) -> Result<Self, Self::Error> {
                <$type>::try_from(value.value())
            }
        }
    };
}

impl_from_value!(u64);
impl_from_value!(u32);
impl_from_value!(u16);
impl_from_value!(u8);

impl BitwisePcodeOps for ConcreteValue {
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

impl PcodeOps for ConcreteValue {
    type Bit = bool;
    type Byte = u8;

    fn num_bytes(&self) -> usize {
        ((self.valid_bits + u8::BITS - 1) / u8::BITS) as usize
    }

    fn add(self, rhs: Self) -> Self {
        assert_eq!(self.valid_bits, rhs.valid_bits);
        self.map(|value| value.wrapping_add(rhs.value()))
    }

    fn unsigned_carry(self, rhs: Self) -> Self::Bit {
        assert_eq!(self.valid_bits, rhs.valid_bits);
        let value = self.value().checked_add(rhs.value());
        if let Some(value) = value {
            value != ConcreteValue::new(value, self.valid_bits).value()
        } else {
            true
        }
    }

    fn signed_carry(self, rhs: Self) -> Self::Bit {
        let value = self.signed_value().checked_add(rhs.signed_value());
        if let Some(value) = value {
            value != ConcreteValue::new(value as u128, self.valid_bits).signed_value()
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
            value != ConcreteValue::new(value as u128, self.valid_bits).signed_value()
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
        ConcreteValue::new(self.value(), u8::BITS * new_size as u32)
    }

    fn sign_extend(self, new_size: usize) -> Self {
        ConcreteValue::new(self.signed_value() as u128, u8::BITS * new_size as u32)
    }

    fn piece(self, lsb: Self) -> Self {
        ConcreteValue::new(
            (self.value() << lsb.valid_bits) | lsb.value(),
            self.valid_bits + lsb.valid_bits,
        )
    }

    fn truncate_to_size(self, new_size: usize) -> Self {
        ConcreteValue::new(self.value(), u8::BITS * new_size as u32)
    }

    fn truncate_trailing_bytes(self, amount: u64) -> Self {
        let invalid_bits = u8::BITS * amount as u32;
        ConcreteValue::new(self.value() >> invalid_bits, self.valid_bits - invalid_bits)
    }

    fn lsb(self) -> Self::Bit {
        self.value() & 0x1 > 0
    }

    fn popcount(self) -> Self {
        ConcreteValue::new(
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

    fn predicated_on(self, condition: Self::Bit) -> Self {
        if condition {
            self
        } else {
            ConcreteValue::new(self.bitmask(), self.valid_bits)
        }
    }

    fn assert(self, condition: Self::Bit) -> Self {
        if condition {
            self
        } else {
            ConcreteValue::new(0, self.valid_bits)
        }
    }

    fn into_le_bytes(self) -> impl ExactSizeIterator<Item = Self::Byte> {
        self.value()
            .to_le_bytes()
            .into_iter()
            .take(self.num_bytes())
    }
}

impl FromIterator<u8> for ConcreteValue {
    fn from_iter<T: IntoIterator<Item = u8>>(iter: T) -> Self {
        let mut buffer = [0u8; ConcreteValue::MAX_BYTES as usize];
        let mut valid_bits = 0;
        for (i, byte) in iter.into_iter().enumerate() {
            buffer[i] = byte;
            valid_bits += 8;
        }
        ConcreteValue::new(u128::from_le_bytes(buffer), valid_bits)
    }
}

#[derive(Debug, Copy, Clone, Default)]
pub struct SymbolicValue {}

impl From<u8> for SymbolicValue {
    fn from(_value: u8) -> Self {
        Self::default()
    }
}

impl TryFrom<SymbolicValue> for u8 {
    type Error = &'static str;
    fn try_from(_value: SymbolicValue) -> Result<Self, Self::Error> {
        Err("symbolic value")
    }
}

impl TryFrom<SymbolicValue> for bool {
    type Error = &'static str;
    fn try_from(_value: SymbolicValue) -> Result<Self, Self::Error> {
        Err("symbolic value")
    }
}

impl From<bool> for SymbolicValue {
    fn from(_value: bool) -> Self {
        Self::default()
    }
}

impl From<u64> for SymbolicValue {
    fn from(_value: u64) -> Self {
        Self::default()
    }
}

impl TryFrom<SymbolicValue> for u64 {
    type Error = &'static str;
    fn try_from(_value: SymbolicValue) -> Result<Self, Self::Error> {
        Err("symbolic value")
    }
}

impl FromIterator<SymbolicValue> for SymbolicValue {
    fn from_iter<T: IntoIterator<Item = SymbolicValue>>(_iter: T) -> Self {
        Self::default()
    }
}

impl PcodeOps for SymbolicValue {
    type Byte = SymbolicValue;
    type Bit = SymbolicValue;

    fn into_le_bytes(self) -> impl ExactSizeIterator<Item = Self::Byte> {
        std::iter::once(self)
    }

    fn num_bytes(&self) -> usize {
        1
    }

    fn add(self, _rhs: Self) -> Self {
        self
    }

    fn unsigned_carry(self, _rhs: Self) -> Self::Bit {
        self
    }

    fn signed_carry(self, _rhs: Self) -> Self::Bit {
        self
    }

    fn negate(self) -> Self {
        self
    }

    fn subtract(self, _rhs: Self) -> Self {
        self
    }

    fn borrow(self, _rhs: Self) -> Self::Bit {
        self
    }

    fn multiply(self, _rhs: Self) -> Self {
        self
    }

    fn unsigned_divide(self, _rhs: Self) -> Self {
        self
    }

    fn signed_divide(self, _rhs: Self) -> Self {
        self
    }

    fn unsigned_remainder(self, _rhs: Self) -> Self {
        self
    }

    fn signed_remainder(self, _rhs: Self) -> Self {
        self
    }

    fn zero_extend(self, _new_size: usize) -> Self {
        self
    }

    fn sign_extend(self, _new_size: usize) -> Self {
        self
    }

    fn piece(self, _lsb: Self) -> Self {
        self
    }

    fn truncate_to_size(self, _new_size: usize) -> Self {
        self
    }

    fn truncate_trailing_bytes(self, _amount: u64) -> Self {
        self
    }

    fn lsb(self) -> Self::Bit {
        self
    }

    fn popcount(self) -> Self {
        self
    }

    fn shift_left(self, _rhs: Self) -> Self {
        self
    }

    fn unsigned_shift_right(self, _rhs: Self) -> Self {
        self
    }

    fn signed_shift_right(self, _rhs: Self) -> Self {
        self
    }

    fn equals(self, _rhs: Self) -> Self::Bit {
        self
    }

    fn not_equals(self, _rhs: Self) -> Self::Bit {
        self
    }

    fn unsigned_less_than(self, _rhs: Self) -> Self::Bit {
        self
    }

    fn signed_less_than(self, _rhs: Self) -> Self::Bit {
        self
    }

    fn unsigned_less_than_or_equals(self, _rhs: Self) -> Self::Bit {
        self
    }

    fn signed_less_than_or_equals(self, _rhs: Self) -> Self::Bit {
        self
    }

    fn signed_greater_than(self, _rhs: Self) -> Self::Bit {
        self
    }

    fn signed_greater_than_or_equals(self, _rhs: Self) -> Self::Bit {
        self
    }

    fn unsigned_greater_than(self, _rhs: Self) -> Self::Bit {
        self
    }

    fn unsigned_greater_than_or_equals(self, _rhs: Self) -> Self::Bit {
        self
    }

    fn predicated_on(self, _condition: Self::Bit) -> Self {
        self
    }

    fn assert(self, _condition: Self::Bit) -> Self {
        self
    }
}

impl BitwisePcodeOps for SymbolicValue {
    fn and(self, _rhs: Self) -> Self {
        self
    }

    fn not(self) -> Self {
        self
    }

    fn or(self, _rhs: Self) -> Self {
        self
    }

    fn xor(self, _rhs: Self) -> Self {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pcode_ops::validate::*;

    #[test]
    fn concrete_value_pcode_validation() -> Result {
        Validator::<ConcreteValue>::validate()
    }
}
