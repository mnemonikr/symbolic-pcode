use crate::buf::{SymbolicBitBuf, SymbolicByte};
use crate::sym::{self, ConcretizationError, SymbolicBit, SymbolicBitVec};

impl std::ops::Deref for SymbolicBitVec {
    type Target = [SymbolicBit];

    fn deref(&self) -> &Self::Target {
        &self.bits.as_slice()
    }
}

fn symbolize<const BYTES: usize, T: FromBytes<BYTES>>(
    value: ConcreteValue<BYTES, T>,
) -> Vec<SymbolicBit> {
    let mut bits = Vec::with_capacity(8 * BYTES);
    let bytes = value.into_inner().to_le_bytes();

    for i in 0..BYTES {
        for b in 0..8 {
            bits.push(SymbolicBit::Literal((bytes[i] & (1 << b)) > 0));
        }
    }

    bits
}

fn symbolize_bytes<const BYTES: usize, const BITS: usize>(
    bytes: [u8; BYTES],
) -> [SymbolicBit; BITS] {
    assert_eq!(8 * BYTES, BITS);
    let mut bits = [sym::FALSE; BITS];

    for i in 0..BYTES {
        for b in 0..8 {
            bits[8 * i + b] = SymbolicBit::Literal((bytes[i] & (1 << b)) > 0);
        }
    }

    bits
}

impl<const BYTES: usize, const BITS: usize> From<[u8; BYTES]> for SymbolicBitBuf<BITS> {
    fn from(bytes: [u8; BYTES]) -> Self {
        symbolize_bytes(bytes).into()
    }
}

// Note: This cannot make use of symbolize_bytes because the BITS parameter cannot be computed. See
// TODO issue for more details.
impl<const N: usize> From<[u8; N]> for SymbolicBitVec {
    fn from(bytes: [u8; N]) -> Self {
        let mut bits = Vec::with_capacity(8 * N);

        for i in 0..N {
            for b in 0..8 {
                bits.push(SymbolicBit::Literal((bytes[i] & (1 << b)) > 0));
            }
        }

        SymbolicBitVec { bits }
    }
}

impl From<bool> for SymbolicBit {
    fn from(value: bool) -> Self {
        SymbolicBit::Literal(value)
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

impl From<SymbolicBit> for SymbolicBitVec {
    fn from(value: SymbolicBit) -> Self {
        Self { bits: vec![value] }
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

#[repr(transparent)]
struct ConcreteValue<const N: usize, T: FromBytes<N>> {
    value: T,
}

impl<const N: usize, T: FromBytes<N>> ConcreteValue<N, T> {
    pub fn new(value: T) -> Self {
        Self { value }
    }

    pub fn into_inner(self) -> T {
        self.value
    }
}
impl<const N: usize, T: FromBytes<N>> From<T> for ConcreteValue<N, T> {
    fn from(value: T) -> Self {
        ConcreteValue { value }
    }
}

// TODO: This is equivalent to the below macros. However it cannot operate on the target type
// directly since that type does not name BYTES. Instead it must operate on the wrapper
// ConcreteValue.
impl<const BYTES: usize, T: FromBytes<BYTES>, const BITS: usize> TryFrom<SymbolicBitBuf<BITS>>
    for ConcreteValue<BYTES, T>
{
    type Error = ConcretizationError;
    fn try_from(value: SymbolicBitBuf<BITS>) -> Result<Self, Self::Error> {
        concretize_bits_into::<T, BYTES>(value).map(ConcreteValue::from)
    }
}

impl<const BYTES: usize, T: FromBytes<BYTES>, const BITS: usize> From<ConcreteValue<BYTES, T>>
    for SymbolicBitBuf<BITS>
{
    fn from(value: ConcreteValue<BYTES, T>) -> Self {
        assert_eq!(8 * BYTES, BITS);
        let vec = symbolize(value);
        let array: [SymbolicBit; BITS] = vec.try_into().unwrap();
        array.into()
    }
}

impl<const BYTES: usize, T: FromBytes<BYTES>> TryFrom<SymbolicBitVec> for ConcreteValue<BYTES, T> {
    type Error = ConcretizationError;
    fn try_from(value: SymbolicBitVec) -> Result<Self, Self::Error> {
        concretize_bits::<T, BYTES>(value.iter()).map(ConcreteValue::from)
    }
}

impl<const BYTES: usize, T: FromBytes<BYTES>> From<ConcreteValue<BYTES, T>> for SymbolicBitVec {
    fn from(value: ConcreteValue<BYTES, T>) -> Self {
        SymbolicBitVec {
            bits: symbolize(value),
        }
    }
}

/// Try to convert from SymbolicBitBuf or &SymbolicBitBuf to given target type.
///
/// If the number of bits for SymbolicBitBuf is not provided, then the number of bits required to
/// represent the target type is used.
macro_rules! tryfrom_bitbuf {
    ($target:ty) => {
        tryfrom_bitbuf!($target, { 8 * std::mem::size_of::<$target>() });
    };
    ($target:ty, $size:expr) => {
        impl TryFrom<SymbolicBitBuf<$size>> for $target {
            type Error = ConcretizationError;
            fn try_from(value: SymbolicBitBuf<$size>) -> Result<Self, Self::Error> {
                concretize_bits_into(value)
            }
        }

        impl TryFrom<&SymbolicBitBuf<$size>> for $target {
            type Error = ConcretizationError;
            fn try_from(value: &SymbolicBitBuf<$size>) -> Result<Self, Self::Error> {
                concretize_bits(value.iter())
            }
        }
    };
}

tryfrom_bitbuf!(u8);
tryfrom_bitbuf!(u8, 1);
tryfrom_bitbuf!(u8, 2);
tryfrom_bitbuf!(u8, 3);
tryfrom_bitbuf!(u8, 4);
tryfrom_bitbuf!(u8, 5);
tryfrom_bitbuf!(u8, 6);
tryfrom_bitbuf!(u8, 7);
tryfrom_bitbuf!(u16);
tryfrom_bitbuf!(u32);
tryfrom_bitbuf!(u64);
tryfrom_bitbuf!(u128);
tryfrom_bitbuf!(usize);

/// Convert from the given source type into a SymbolicBitBuf. The number of bits in the
/// SymbolicBitBuf is the number of bits required to represent the source type.
macro_rules! bitbuf_from {
    ($source:ty) => {
        impl From<$source> for SymbolicBitBuf<{ 8 * std::mem::size_of::<$source>() }> {
            fn from(value: $source) -> Self {
                value.to_le_bytes().into()
            }
        }
    };
}

bitbuf_from!(u8);
bitbuf_from!(u16);
bitbuf_from!(u32);
bitbuf_from!(u64);
bitbuf_from!(u128);
bitbuf_from!(usize);

/// Convert from SymbolicBitVec or &SymbolicBitVec into the given target type.
macro_rules! tryfrom_bitvec {
    ($target:ty) => {
        impl TryFrom<SymbolicBitVec> for $target {
            type Error = ConcretizationError;
            fn try_from(value: SymbolicBitVec) -> Result<Self, Self::Error> {
                concretize_bits(value.iter())
            }
        }

        impl TryFrom<&SymbolicBitVec> for $target {
            type Error = ConcretizationError;
            fn try_from(value: &SymbolicBitVec) -> Result<Self, Self::Error> {
                concretize_bits(value.iter())
            }
        }
    };
}

tryfrom_bitvec!(u8);
tryfrom_bitvec!(u16);
tryfrom_bitvec!(u32);
tryfrom_bitvec!(u64);
tryfrom_bitvec!(u128);
tryfrom_bitvec!(usize);

/// Convert from the given target type into SymbolicBitVec
macro_rules! from_bitvec {
    ($target:ty) => {
        impl From<$target> for SymbolicBitVec {
            fn from(value: $target) -> Self {
                value.to_le_bytes().into()
            }
        }
    };
}

from_bitvec!(u8);
from_bitvec!(u16);
from_bitvec!(u32);
from_bitvec!(u64);
from_bitvec!(u128);
from_bitvec!(usize);

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

pub fn concretize<'a, T, const N: usize>(
    iter: impl Iterator<Item = &'a SymbolicByte>,
) -> std::result::Result<T, ConcretizationError>
where
    T: FromBytes<N>,
{
    // TODO Once we can use this directly in the function signature we can remove N
    assert_eq!(std::mem::size_of::<T>(), N);

    concretize_bits(
        iter.map(|byte| byte.as_ref())
            .map(|bits| bits.iter())
            .flatten(),
    )
}

pub fn concretize_into<T, const N: usize>(
    iter: impl IntoIterator<Item = SymbolicByte>,
) -> std::result::Result<T, ConcretizationError>
where
    T: FromBytes<N>,
{
    // TODO Once we can use this directly in the function signature we can remove N
    assert_eq!(std::mem::size_of::<T>(), N);

    concretize_bits_into(
        iter.into_iter()
            .map(|byte| byte.into_inner())
            .map(|bits| bits.into_iter())
            .flatten(),
    )
}

pub fn concretize_bits_cow<'a, T, const N: usize>(
    iter: impl Iterator<Item = std::borrow::Cow<'a, SymbolicBit>>,
) -> std::result::Result<T, ConcretizationError>
where
    T: FromBytes<N>,
{
    // TODO Once we can use this directly in the function signature we can remove N
    assert_eq!(std::mem::size_of::<T>(), N);

    let mut bytes = [0u8; N];
    let mut byte_index = 0;
    let mut bit_index = 0;
    for cow_bit in iter {
        if byte_index >= N {
            return Err(ConcretizationError::Overflow { max_bytes: N });
        }

        let bit = match cow_bit {
            std::borrow::Cow::Owned(SymbolicBit::Literal(bit)) => bit,
            std::borrow::Cow::Borrowed(SymbolicBit::Literal(bit)) => *bit,
            _ => {
                return Err(ConcretizationError::NonLiteralBit {
                    bit_index,
                    byte_index,
                })
            }
        };

        if bit {
            bytes[byte_index] |= 1 << bit_index;
        }

        bit_index += 1;
        if bit_index == 8 {
            bit_index = 0;
            byte_index += 1;
        }
    }

    Ok(T::from_le_bytes(bytes))
}

pub fn concretize_bits<'a, T, const N: usize>(
    iter: impl Iterator<Item = &'a SymbolicBit>,
) -> std::result::Result<T, ConcretizationError>
where
    T: FromBytes<N>,
{
    concretize_bits_cow(iter.map(std::borrow::Cow::Borrowed))
}

pub fn concretize_bits_into<T, const N: usize>(
    iter: impl IntoIterator<Item = SymbolicBit>,
) -> std::result::Result<T, ConcretizationError>
where
    T: FromBytes<N>,
{
    concretize_bits_cow(iter.into_iter().map(std::borrow::Cow::Owned))
}

/// Create an object from bytes. The object must be constructable from a known, fixed set of bytes.
/// TODO Rename this trait since it also supports to bytes
pub trait FromBytes<const N: usize> {
    /// Create an instance of the object from a little-endian byte representation.
    fn from_le_bytes(bytes: [u8; N]) -> Self;

    fn to_le_bytes(self) -> [u8; N];
}

macro_rules! impl_frombytes {
    ($target:ty) => {
        impl FromBytes<{ std::mem::size_of::<$target>() }> for $target {
            fn from_le_bytes(bytes: [u8; std::mem::size_of::<$target>()]) -> Self {
                Self::from_le_bytes(bytes)
            }

            fn to_le_bytes(self) -> [u8; std::mem::size_of::<$target>()] {
                self.to_le_bytes()
            }
        }
    };
}

impl_frombytes!(usize);
impl_frombytes!(u128);
impl_frombytes!(u64);
impl_frombytes!(u32);
impl_frombytes!(u16);
impl_frombytes!(u8);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn concretize_buf() {
        assert_eq!(0, u128::try_from(SymbolicBitBuf::<128>::default()).unwrap());
        assert_eq!(0, u64::try_from(SymbolicBitBuf::<64>::default()).unwrap());
        assert_eq!(0, u32::try_from(SymbolicBitBuf::<32>::default()).unwrap());
        assert_eq!(0, u16::try_from(SymbolicBitBuf::<16>::default()).unwrap());
        assert_eq!(0, u8::try_from(SymbolicBitBuf::<8>::default()).unwrap());
        assert_eq!(
            0,
            usize::try_from(SymbolicBitBuf::<{ 8 * std::mem::size_of::<usize>() }>::default())
                .unwrap()
        );
    }

    #[test]
    fn symbolicbitbuf_conversions() {
        let value: u8 = 0x5A;
        assert_eq!(value, u8::try_from(SymbolicBitBuf::from(value)).unwrap());

        let value: u16 = 0xBEEF;
        assert_eq!(value, u16::try_from(SymbolicBitBuf::from(value)).unwrap());

        let value: u32 = 0xDEADBEEF;
        assert_eq!(value, u32::try_from(SymbolicBitBuf::from(value)).unwrap());

        let value: u64 = 0xFEEDBEEF_0BADF00D;
        assert_eq!(value, u64::try_from(SymbolicBitBuf::from(value)).unwrap());

        let value: u128 = 0xFEEDBEEF_0BADF00D_DEADBEEF_DEAD4BED;
        assert_eq!(value, u128::try_from(SymbolicBitBuf::from(value)).unwrap());
    }

    #[test]
    fn symbolicbitvec_conversions() {
        let value: u8 = 0x5A;
        assert_eq!(value, u8::try_from(SymbolicBitVec::from(value)).unwrap());

        let value: u16 = 0xBEEF;
        assert_eq!(value, u16::try_from(SymbolicBitVec::from(value)).unwrap());

        let value: u32 = 0xDEADBEEF;
        assert_eq!(value, u32::try_from(SymbolicBitVec::from(value)).unwrap());

        let value: u64 = 0xFEEDBEEF_0BADF00D;
        assert_eq!(value, u64::try_from(SymbolicBitVec::from(value)).unwrap());

        let value: u128 = 0xFEEDBEEF_0BADF00D_DEADBEEF_DEAD4BED;
        assert_eq!(value, u128::try_from(SymbolicBitVec::from(value)).unwrap());
    }
}
