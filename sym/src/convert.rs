use crate::{buf::SymbolicByte, sym::*, SymbolicBitBuf};

impl From<bool> for SymbolicBit {
    fn from(value: bool) -> Self {
        SymbolicBit::Literal(value)
    }
}

impl TryFrom<SymbolicBit> for bool {
    type Error = String;

    fn try_from(value: SymbolicBit) -> Result<Self, Self::Error> {
        if let SymbolicBit::Literal(value) = value {
            Ok(value)
        } else {
            Err(format!("Symbolic bit is not a literal: {value:?}"))
        }
    }
}

impl TryFrom<SymbolicBitBuf<1>> for u8 {
    type Error = ConcretizationError<u8>;
    fn try_from(value: SymbolicBitBuf<1>) -> Result<Self, Self::Error> {
        concretize_bit_iter(value.into_inner().iter())
    }
}
impl TryFrom<SymbolicBitBuf<2>> for u8 {
    type Error = ConcretizationError<u8>;
    fn try_from(value: SymbolicBitBuf<2>) -> Result<Self, Self::Error> {
        concretize_bit_iter(value.into_inner().iter())
    }
}
impl TryFrom<SymbolicBitBuf<3>> for u8 {
    type Error = ConcretizationError<u8>;
    fn try_from(value: SymbolicBitBuf<3>) -> Result<Self, Self::Error> {
        concretize_bit_iter(value.into_inner().iter())
    }
}
impl TryFrom<SymbolicBitBuf<4>> for u8 {
    type Error = ConcretizationError<u8>;
    fn try_from(value: SymbolicBitBuf<4>) -> Result<Self, Self::Error> {
        concretize_bit_iter(value.into_inner().iter())
    }
}
impl TryFrom<SymbolicBitBuf<5>> for u8 {
    type Error = ConcretizationError<u8>;
    fn try_from(value: SymbolicBitBuf<5>) -> Result<Self, Self::Error> {
        concretize_bit_iter(value.into_inner().iter())
    }
}
impl TryFrom<SymbolicBitBuf<6>> for u8 {
    type Error = ConcretizationError<u8>;
    fn try_from(value: SymbolicBitBuf<6>) -> Result<Self, Self::Error> {
        concretize_bit_iter(value.into_inner().iter())
    }
}

impl TryFrom<SymbolicBitBuf<7>> for u8 {
    type Error = ConcretizationError<u8>;
    fn try_from(value: SymbolicBitBuf<7>) -> Result<Self, Self::Error> {
        concretize_bit_iter(value.into_inner().iter())
    }
}

impl TryFrom<SymbolicByte> for u8 {
    type Error = ConcretizationError<u8>;
    fn try_from(value: SymbolicByte) -> Result<Self, Self::Error> {
        concretize_bit_iter(value.into_inner().iter())
    }
}

impl TryFrom<&SymbolicByte> for u8 {
    type Error = ConcretizationError<u8>;
    fn try_from(value: &SymbolicByte) -> Result<Self, Self::Error> {
        concretize_bit_iter(value.iter())
    }
}

impl TryFrom<SymbolicBitVec> for u16 {
    type Error = ConcretizationError<u16>;
    fn try_from(value: SymbolicBitVec) -> Result<Self, Self::Error> {
        concretize_bit_iter(value.iter())
    }
}

impl TryFrom<&SymbolicBitVec> for u16 {
    type Error = ConcretizationError<u16>;
    fn try_from(value: &SymbolicBitVec) -> Result<Self, Self::Error> {
        concretize_bit_iter(value.iter())
    }
}

impl TryFrom<SymbolicBitVec> for u32 {
    type Error = ConcretizationError<u32>;
    fn try_from(value: SymbolicBitVec) -> Result<Self, Self::Error> {
        concretize_bit_iter(value.iter())
    }
}

impl TryFrom<&SymbolicBitVec> for u32 {
    type Error = ConcretizationError<u32>;
    fn try_from(value: &SymbolicBitVec) -> Result<Self, Self::Error> {
        concretize_bit_iter(value.iter())
    }
}

impl TryFrom<SymbolicBitVec> for u64 {
    type Error = ConcretizationError<u64>;
    fn try_from(value: SymbolicBitVec) -> Result<Self, Self::Error> {
        concretize_bit_iter(value.iter())
    }
}

impl TryFrom<&SymbolicBitVec> for u64 {
    type Error = ConcretizationError<u64>;
    fn try_from(value: &SymbolicBitVec) -> Result<Self, Self::Error> {
        concretize_bit_iter(value.iter())
    }
}

impl TryFrom<SymbolicBitVec> for usize {
    type Error = ConcretizationError<usize>;
    fn try_from(value: SymbolicBitVec) -> Result<Self, Self::Error> {
        concretize_bit_iter(value.iter())
    }
}

impl TryFrom<&SymbolicBitVec> for usize {
    type Error = ConcretizationError<usize>;
    fn try_from(value: &SymbolicBitVec) -> Result<Self, Self::Error> {
        concretize_bit_iter(value.iter())
    }
}

impl From<u8> for SymbolicByte {
    fn from(value: u8) -> Self {
        let mut byte = SymbolicByte::default();
        let mut mask = 1u8;
        for i in 0..8 {
            byte[i] = SymbolicBit::Literal(value & mask != 0u8);
            mask <<= 1;
        }

        byte
    }
}

impl From<u8> for SymbolicBitVec {
    fn from(value: u8) -> Self {
        into_symbitvec(value)
    }
}

impl From<u16> for SymbolicBitVec {
    fn from(value: u16) -> Self {
        into_symbitvec(value)
    }
}

impl From<u32> for SymbolicBitVec {
    fn from(value: u32) -> Self {
        into_symbitvec(value)
    }
}

impl From<u64> for SymbolicBitVec {
    fn from(value: u64) -> Self {
        into_symbitvec(value)
    }
}

impl From<usize> for SymbolicBitVec {
    fn from(value: usize) -> Self {
        into_symbitvec(value)
    }
}

impl From<&[u8]> for SymbolicBitVec {
    fn from(value: &[u8]) -> Self {
        value
            .iter()
            .copied()
            .map(Into::<SymbolicBitVec>::into)
            .collect::<Vec<_>>()
            .into()
    }
}

impl From<Vec<SymbolicBitVec>> for SymbolicBitVec {
    fn from(value: Vec<SymbolicBitVec>) -> Self {
        Self {
            bits: value
                .into_iter()
                .map(|bitvec| bitvec.bits.into_iter())
                .flatten()
                .collect(),
        }
    }
}

fn into_symbitvec<T>(value: T) -> SymbolicBitVec
where
    T: std::ops::BitAnd + std::ops::ShlAssign + PartialEq + From<u8> + Copy,
    <T as std::ops::BitAnd>::Output: PartialEq + From<u8>,
{
    let mut bits = Vec::with_capacity(std::mem::size_of::<T>());
    let mut mask: T = 1u8.into();
    while mask != 0u8.into() {
        bits.push(SymbolicBit::Literal(value & mask != 0u8.into()));
        mask <<= 1u8.into();
    }

    SymbolicBitVec { bits }
}

impl std::ops::Deref for SymbolicBitVec {
    type Target = [SymbolicBit];

    fn deref(&self) -> &Self::Target {
        &self.bits.as_slice()
    }
}

impl TryFrom<SymbolicBitVec> for SymbolicBit {
    type Error = String;

    fn try_from(mut value: SymbolicBitVec) -> Result<Self, Self::Error> {
        if value.len() == 1 {
            Ok(value.bits.pop().unwrap())
        } else {
            Err(format!("value has {num_bits} bits", num_bits = value.len()))
        }
    }
}

impl From<&[SymbolicBit]> for SymbolicBitVec {
    fn from(bits: &[SymbolicBit]) -> Self {
        Self {
            bits: bits.to_vec(),
        }
    }
}

impl From<Vec<SymbolicBit>> for SymbolicBitVec {
    fn from(bits: Vec<SymbolicBit>) -> Self {
        Self { bits }
    }
}

impl TryFrom<SymbolicBitVec> for u8 {
    type Error = ConcretizationError<u8>;
    fn try_from(value: SymbolicBitVec) -> Result<Self, Self::Error> {
        concretize_bit_iter(value.iter())
    }
}

impl TryFrom<&SymbolicBitVec> for u8 {
    type Error = ConcretizationError<u8>;
    fn try_from(value: &SymbolicBitVec) -> Result<Self, Self::Error> {
        concretize_bit_iter(value.iter())
    }
}

impl From<Vec<SymbolicByte>> for SymbolicBitVec {
    fn from(value: Vec<SymbolicByte>) -> Self {
        Self {
            bits: value
                .into_iter()
                .map(|byte| <[SymbolicBit; 8]>::from(byte).into_iter())
                .flatten()
                .collect(),
        }
    }
}
