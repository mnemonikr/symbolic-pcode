use crate::buf::{SymbolicBitBuf, SymbolicByte};
use crate::sym::{self, ConcretizationError, SymbolicBit, SymbolicBitVec};

/// Value that is the little-endian portion of a larger value
pub struct LittleEndian<T: Copy>(T);

impl From<bool> for SymbolicBit {
    fn from(value: bool) -> Self {
        SymbolicBit::Literal(value)
    }
}

impl From<SymbolicBit> for SymbolicBitVec {
    fn from(value: SymbolicBit) -> Self {
        Self { bits: vec![value] }
    }
}

impl TryFrom<SymbolicBit> for bool {
    type Error = ConcretizationError;

    fn try_from(value: SymbolicBit) -> Result<Self, Self::Error> {
        if let SymbolicBit::Literal(value) = value {
            Ok(value)
        } else {
            Err(ConcretizationError::NonLiteralBit {
                bit_index: 0,
                byte_index: 0,
            })
        }
    }
}

impl TryFrom<SymbolicBitBuf<1>> for u8 {
    type Error = ConcretizationError;
    fn try_from(value: SymbolicBitBuf<1>) -> Result<Self, Self::Error> {
        concretize_bits_u8((&value).iter())
    }
}
impl TryFrom<SymbolicBitBuf<2>> for u8 {
    type Error = ConcretizationError;
    fn try_from(value: SymbolicBitBuf<2>) -> Result<Self, Self::Error> {
        concretize_bits_u8((&value).iter())
    }
}
impl TryFrom<SymbolicBitBuf<3>> for u8 {
    type Error = ConcretizationError;
    fn try_from(value: SymbolicBitBuf<3>) -> Result<Self, Self::Error> {
        concretize_bits_u8((&value).iter())
    }
}
impl TryFrom<SymbolicBitBuf<4>> for u8 {
    type Error = ConcretizationError;
    fn try_from(value: SymbolicBitBuf<4>) -> Result<Self, Self::Error> {
        concretize_bits_u8((&value).iter())
    }
}
impl TryFrom<SymbolicBitBuf<5>> for u8 {
    type Error = ConcretizationError;
    fn try_from(value: SymbolicBitBuf<5>) -> Result<Self, Self::Error> {
        concretize_bits_u8((&value).iter())
    }
}
impl TryFrom<SymbolicBitBuf<6>> for u8 {
    type Error = ConcretizationError;
    fn try_from(value: SymbolicBitBuf<6>) -> Result<Self, Self::Error> {
        concretize_bits_u8((&value).iter())
    }
}

impl TryFrom<SymbolicBitBuf<7>> for u8 {
    type Error = ConcretizationError;
    fn try_from(value: SymbolicBitBuf<7>) -> Result<Self, Self::Error> {
        concretize_bits_u8((&value).iter())
    }
}

impl TryFrom<SymbolicBitBuf<16>> for u16 {
    type Error = ConcretizationError;
    fn try_from(value: SymbolicBitBuf<16>) -> Result<Self, Self::Error> {
        concretize_bits_u16((&value).iter())
    }
}

impl From<u64> for SymbolicBitBuf<64> {
    fn from(value: u64) -> Self {
        value.to_le_bytes().into()
    }
}

impl<const BYTES: usize, const BITS: usize> From<[u8; BYTES]> for SymbolicBitBuf<BITS> {
    fn from(bytes: [u8; BYTES]) -> Self {
        let mut bits = [sym::FALSE; BITS];
        assert!(8 * BYTES == BITS);
        for i in 0..bytes.len() {
            for b in 0..8 {
                bits[8 * i + b] = SymbolicBit::Literal((bytes[i] & (1 << b)) > 0);
            }
        }

        bits.into()
    }
}

impl TryFrom<SymbolicByte> for u8 {
    type Error = ConcretizationError;
    fn try_from(value: SymbolicByte) -> Result<Self, Self::Error> {
        concretize_into_u8(std::iter::once(value))
    }
}

impl TryFrom<&SymbolicByte> for u8 {
    type Error = ConcretizationError;
    fn try_from(value: &SymbolicByte) -> Result<Self, Self::Error> {
        concretize_u8(std::iter::once(value))
    }
}

impl TryFrom<SymbolicBitVec> for u16 {
    type Error = ConcretizationError;
    fn try_from(value: SymbolicBitVec) -> Result<Self, Self::Error> {
        concretize_bits_u16(value.iter())
    }
}

impl TryFrom<&SymbolicBitVec> for u16 {
    type Error = ConcretizationError;
    fn try_from(value: &SymbolicBitVec) -> Result<Self, Self::Error> {
        concretize_bits_u16(value.iter())
    }
}

impl TryFrom<SymbolicBitVec> for u32 {
    type Error = ConcretizationError;
    fn try_from(value: SymbolicBitVec) -> Result<Self, Self::Error> {
        concretize_bits_u32(value.iter())
    }
}

impl TryFrom<&SymbolicBitVec> for u32 {
    type Error = ConcretizationError;
    fn try_from(value: &SymbolicBitVec) -> Result<Self, Self::Error> {
        concretize_bits_u32(value.iter())
    }
}

impl TryFrom<SymbolicBitVec> for u64 {
    type Error = ConcretizationError;
    fn try_from(value: SymbolicBitVec) -> Result<Self, Self::Error> {
        concretize_bits_u64(value.iter())
    }
}

impl TryFrom<&SymbolicBitVec> for u64 {
    type Error = ConcretizationError;
    fn try_from(value: &SymbolicBitVec) -> Result<Self, Self::Error> {
        concretize_bits_u64(value.iter())
    }
}

impl TryFrom<SymbolicBitVec> for usize {
    type Error = ConcretizationError;
    fn try_from(value: SymbolicBitVec) -> Result<Self, Self::Error> {
        concretize_bits_usize(value.iter())
    }
}

impl TryFrom<&SymbolicBitVec> for usize {
    type Error = ConcretizationError;
    fn try_from(value: &SymbolicBitVec) -> Result<Self, Self::Error> {
        concretize_bits_usize(value.iter())
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

impl FromIterator<LittleEndian<u8>> for SymbolicBitVec {
    fn from_iter<T: IntoIterator<Item = LittleEndian<u8>>>(iter: T) -> Self {
        iter.into_iter()
            .map(|x| x.0)
            .map(Into::<SymbolicBitVec>::into)
            .collect()
    }
}

impl FromIterator<SymbolicBitVec> for SymbolicBitVec {
    fn from_iter<T: IntoIterator<Item = SymbolicBitVec>>(iter: T) -> Self {
        Self {
            bits: iter
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
    let num_bits = 8 * std::mem::size_of::<T>();
    let mut bits = Vec::with_capacity(num_bits);
    let mut mask: T = 1u8.into();
    for _ in 0..num_bits {
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

impl TryFrom<SymbolicBitVec> for u8 {
    type Error = ConcretizationError;
    fn try_from(value: SymbolicBitVec) -> Result<Self, Self::Error> {
        concretize_bits_u8(value.iter())
    }
}

impl TryFrom<&SymbolicBitVec> for u8 {
    type Error = ConcretizationError;
    fn try_from(value: &SymbolicBitVec) -> Result<Self, Self::Error> {
        concretize_bits_u8(value.iter())
    }
}

impl FromIterator<SymbolicBit> for SymbolicBitVec {
    fn from_iter<T: IntoIterator<Item = SymbolicBit>>(iter: T) -> Self {
        Self {
            bits: iter.into_iter().collect(),
        }
    }
}

impl FromIterator<SymbolicByte> for SymbolicBitVec {
    fn from_iter<T: IntoIterator<Item = SymbolicByte>>(iter: T) -> Self {
        Self {
            bits: iter
                .into_iter()
                .map(|byte| byte.into_inner().into_iter())
                .flatten()
                .collect(),
        }
    }
}

pub fn concretize_into_u8(
    iter: impl IntoIterator<Item = SymbolicByte>,
) -> std::result::Result<u8, ConcretizationError> {
    concretize_into(iter, u8::from_le_bytes)
}

pub fn concretize_into_u16(
    iter: impl IntoIterator<Item = SymbolicByte>,
) -> std::result::Result<u16, ConcretizationError> {
    concretize_into(iter, u16::from_le_bytes)
}

pub fn concretize_into_u32(
    iter: impl IntoIterator<Item = SymbolicByte>,
) -> std::result::Result<u32, ConcretizationError> {
    concretize_into(iter, u32::from_le_bytes)
}

pub fn concretize_into_u64(
    iter: impl IntoIterator<Item = SymbolicByte>,
) -> std::result::Result<u64, ConcretizationError> {
    concretize_into(iter, u64::from_le_bytes)
}

pub fn concretize_into_u128(
    iter: impl IntoIterator<Item = SymbolicByte>,
) -> std::result::Result<u128, ConcretizationError> {
    concretize_into(iter, u128::from_le_bytes)
}

pub fn concretize_into_usize(
    iter: impl IntoIterator<Item = SymbolicByte>,
) -> std::result::Result<usize, ConcretizationError> {
    concretize_into(iter, usize::from_le_bytes)
}

pub fn concretize_u8<'a>(
    iter: impl Iterator<Item = &'a SymbolicByte>,
) -> std::result::Result<u8, ConcretizationError> {
    concretize(iter, u8::from_le_bytes)
}

pub fn concretize_u16<'a>(
    iter: impl Iterator<Item = &'a SymbolicByte>,
) -> std::result::Result<u16, ConcretizationError> {
    concretize(iter, u16::from_le_bytes)
}

pub fn concretize_u32<'a>(
    iter: impl Iterator<Item = &'a SymbolicByte>,
) -> std::result::Result<u32, ConcretizationError> {
    concretize(iter, u32::from_le_bytes)
}

pub fn concretize_u64<'a>(
    iter: impl Iterator<Item = &'a SymbolicByte>,
) -> std::result::Result<u64, ConcretizationError> {
    concretize(iter, u64::from_le_bytes)
}

pub fn concretize_u128<'a>(
    iter: impl Iterator<Item = &'a SymbolicByte>,
) -> std::result::Result<u128, ConcretizationError> {
    concretize(iter, u128::from_le_bytes)
}

pub fn concretize_usize<'a>(
    iter: impl Iterator<Item = &'a SymbolicByte>,
) -> std::result::Result<usize, ConcretizationError> {
    concretize(iter, usize::from_le_bytes)
}

// TODO Use an appropriate error here
pub fn concretize<'a, T, F, const N: usize>(
    iter: impl Iterator<Item = &'a SymbolicByte>,
    from_le_bytes: F,
) -> std::result::Result<T, ConcretizationError>
where
    F: FnOnce([u8; N]) -> T,
{
    // TODO Once we can use this directly in the function signature we can remove N
    assert_eq!(std::mem::size_of::<T>(), N);

    concretize_bits(
        iter.map(|byte| byte.as_ref())
            .map(|bits| bits.iter())
            .flatten(),
        from_le_bytes,
    )
}

// TODO Use an appropriate error here
pub fn concretize_into<T, F, const N: usize>(
    iter: impl IntoIterator<Item = SymbolicByte>,
    from_le_bytes: F,
) -> std::result::Result<T, ConcretizationError>
where
    F: FnOnce([u8; N]) -> T,
{
    // TODO Once we can use this directly in the function signature we can remove N
    assert_eq!(std::mem::size_of::<T>(), N);

    concretize_bits_into(
        iter.into_iter()
            .map(|byte| byte.into_inner())
            .map(|bits| bits.into_iter())
            .flatten(),
        from_le_bytes,
    )
}

pub fn concretize_bits_u8<'a>(
    iter: impl Iterator<Item = &'a SymbolicBit>,
) -> std::result::Result<u8, ConcretizationError> {
    concretize_bits(iter, u8::from_le_bytes)
}

pub fn concretize_bits_u16<'a>(
    iter: impl Iterator<Item = &'a SymbolicBit>,
) -> std::result::Result<u16, ConcretizationError> {
    concretize_bits(iter, u16::from_le_bytes)
}

pub fn concretize_bits_u32<'a>(
    iter: impl Iterator<Item = &'a SymbolicBit>,
) -> std::result::Result<u32, ConcretizationError> {
    concretize_bits(iter, u32::from_le_bytes)
}

pub fn concretize_bits_u64<'a>(
    iter: impl Iterator<Item = &'a SymbolicBit>,
) -> std::result::Result<u64, ConcretizationError> {
    concretize_bits(iter, u64::from_le_bytes)
}

pub fn concretize_bits_u128<'a>(
    iter: impl Iterator<Item = &'a SymbolicBit>,
) -> std::result::Result<u128, ConcretizationError> {
    concretize_bits(iter, u128::from_le_bytes)
}

pub fn concretize_bits_usize<'a>(
    iter: impl Iterator<Item = &'a SymbolicBit>,
) -> std::result::Result<usize, ConcretizationError> {
    concretize_bits(iter, usize::from_le_bytes)
}

pub fn concretize_bits<'a, T, F, const N: usize>(
    iter: impl Iterator<Item = &'a SymbolicBit>,
    from_le_bytes: F,
) -> std::result::Result<T, ConcretizationError>
where
    F: FnOnce([u8; N]) -> T,
{
    // TODO Once we can use this directly in the function signature we can remove N
    assert_eq!(std::mem::size_of::<T>(), N);

    let mut bytes = [0u8; N];
    let mut byte_index = 0;
    let mut bit_index = 0;
    for bit in iter {
        if byte_index >= N {
            return Err(ConcretizationError::Overflow { max_bytes: N });
        }

        match *bit {
            SymbolicBit::Literal(true) => {
                bytes[byte_index] |= 1 << bit_index;
            }
            SymbolicBit::Literal(false) => (),
            _ => {
                return Err(ConcretizationError::NonLiteralBit {
                    bit_index,
                    byte_index,
                })
            }
        };

        bit_index += 1;
        if bit_index == 8 {
            bit_index = 0;
            byte_index += 1;
        }
    }

    Ok(from_le_bytes(bytes))
}

pub fn concretize_bits_into<T, F, const N: usize>(
    iter: impl IntoIterator<Item = SymbolicBit>,
    from_le_bytes: F,
) -> std::result::Result<T, ConcretizationError>
where
    F: FnOnce([u8; N]) -> T,
{
    // TODO Once we can use this directly in the function signature we can remove N
    assert_eq!(std::mem::size_of::<T>(), N);

    let mut bytes = [0u8; N];
    let mut byte_index = 0;
    let mut bit_index = 0;
    for bit in iter {
        if byte_index >= N {
            return Err(ConcretizationError::Overflow { max_bytes: N });
        }

        match bit {
            SymbolicBit::Literal(true) => {
                bytes[byte_index] |= 1 << bit_index;
            }
            SymbolicBit::Literal(false) => (),
            _ => {
                return Err(ConcretizationError::NonLiteralBit {
                    bit_index,
                    byte_index,
                })
            }
        };

        bit_index += 1;
        if bit_index == 8 {
            bit_index = 0;
            byte_index += 1;
        }
    }

    Ok(from_le_bytes(bytes))
}
