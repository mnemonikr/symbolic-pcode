use pcode_ops::{BitwisePcodeOps, PcodeOps};

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
