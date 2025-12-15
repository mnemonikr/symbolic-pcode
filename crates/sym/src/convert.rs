use crate::bit::{FALSE, SymbolicBit};
use crate::buf::{SymbolicBitBuf, SymbolicByte};
use crate::vec::SymbolicBitVec;

#[derive(thiserror::Error, Debug)]
pub enum ConcretizationError {
    #[error("non-literal bit at index {bit_index}")]
    NonLiteralBit { bit_index: usize },

    #[error("value exceeded maximum number of bytes ({max_bytes})")]
    Overflow { max_bytes: usize },
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
            Err(ConcretizationError::NonLiteralBit { bit_index: 0 })
        }
    }
}

impl From<SymbolicBit> for SymbolicBitVec {
    fn from(value: SymbolicBit) -> Self {
        std::iter::once(value).collect()
    }
}

/// A wrapper around a value that implements [LittleEndian]. This wrapper implements conversions to
/// and from [SymbolicBitVec] and [SymbolicBitBuf]. Use the macro [concrete_type!] to automatically
/// implement the conversions for a type using this wrapper.
///
/// # Examples
///
/// ```
/// # use symbit::{ConcreteValue, SymbolicBitBuf};
/// let value = ConcreteValue::new(0xDEADBEEFu32);
/// let buf = SymbolicBitBuf::<32>::from(value);
/// assert_eq!(value, buf.try_into().unwrap());
/// ```
#[repr(transparent)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct ConcreteValue<const N: usize, T: LittleEndian<N>> {
    value: T,
}

impl<const N: usize, T: LittleEndian<N>> ConcreteValue<N, T> {
    pub fn new(value: T) -> Self {
        Self { value }
    }

    pub fn into_inner(self) -> T {
        self.value
    }

    pub fn symbolize(self) -> Vec<SymbolicBit> {
        symbolize(self.value.into_words()).collect()
    }
}

impl<const N: usize, T: LittleEndian<N>> From<T> for ConcreteValue<N, T> {
    fn from(value: T) -> Self {
        ConcreteValue { value }
    }
}

impl<const BYTES: usize, T: LittleEndian<BYTES>, const BITS: usize> TryFrom<SymbolicBitBuf<BITS>>
    for ConcreteValue<BYTES, T>
{
    type Error = ConcretizationError;
    fn try_from(value: SymbolicBitBuf<BITS>) -> Result<Self, Self::Error> {
        concretize_bits_into::<T, BYTES>(value).map(ConcreteValue::from)
    }
}

impl<const BYTES: usize, T: LittleEndian<BYTES>, const BITS: usize> TryFrom<&SymbolicBitBuf<BITS>>
    for ConcreteValue<BYTES, T>
{
    type Error = ConcretizationError;
    fn try_from(value: &SymbolicBitBuf<BITS>) -> Result<Self, Self::Error> {
        concretize_bits::<T, BYTES>(value.iter()).map(ConcreteValue::from)
    }
}

impl<const BYTES: usize, T: LittleEndian<BYTES>, const BITS: usize> From<ConcreteValue<BYTES, T>>
    for SymbolicBitBuf<BITS>
{
    fn from(value: ConcreteValue<BYTES, T>) -> Self {
        const {
            assert!(8 * BYTES == BITS, "8 BITS must be 1 BYTE");
        }

        // SAFETY: Asserted BITS agrees with BYTES, so the vec length is BITS
        let array: [SymbolicBit; BITS] = unsafe { value.symbolize().try_into().unwrap_unchecked() };

        array.into()
    }
}

impl<const BYTES: usize, T: LittleEndian<BYTES>> TryFrom<SymbolicBitVec>
    for ConcreteValue<BYTES, T>
{
    type Error = ConcretizationError;
    fn try_from(value: SymbolicBitVec) -> Result<Self, Self::Error> {
        concretize_bits_into::<T, BYTES>(value).map(ConcreteValue::from)
    }
}

impl<const BYTES: usize, T: LittleEndian<BYTES>> TryFrom<&SymbolicBitVec>
    for ConcreteValue<BYTES, T>
{
    type Error = ConcretizationError;
    fn try_from(value: &SymbolicBitVec) -> Result<Self, Self::Error> {
        concretize_bits::<T, BYTES>(value.iter()).map(ConcreteValue::from)
    }
}

impl<const BYTES: usize, T: LittleEndian<BYTES>> From<ConcreteValue<BYTES, T>> for SymbolicBitVec {
    fn from(value: ConcreteValue<BYTES, T>) -> Self {
        value.symbolize().into_iter().collect()
    }
}

#[macro_export]
macro_rules! concrete_type {
    ($target:ty) => {
        impl TryFrom<SymbolicBitVec> for $target {
            type Error = ConcretizationError;
            fn try_from(value: SymbolicBitVec) -> Result<Self, Self::Error> {
                ConcreteValue::try_from(value).map(ConcreteValue::into_inner)
            }
        }

        impl TryFrom<&SymbolicBitVec> for $target {
            type Error = ConcretizationError;
            fn try_from(value: &SymbolicBitVec) -> Result<Self, Self::Error> {
                ConcreteValue::try_from(value).map(ConcreteValue::into_inner)
            }
        }

        impl From<$target> for SymbolicBitVec {
            fn from(value: $target) -> Self {
                ConcreteValue::new(value).into()
            }
        }

        impl From<$target> for SymbolicBitBuf<{ 8 * std::mem::size_of::<$target>() }> {
            fn from(value: $target) -> Self {
                ConcreteValue::new(value).into()
            }
        }

        concrete_type!($target, { 8 * std::mem::size_of::<$target>() });
    };
    ($target:ty, $size:expr) => {
        impl TryFrom<SymbolicBitBuf<$size>> for $target {
            type Error = ConcretizationError;
            fn try_from(value: SymbolicBitBuf<$size>) -> Result<Self, Self::Error> {
                ConcreteValue::try_from(value).map(ConcreteValue::into_inner)
            }
        }

        impl TryFrom<&SymbolicBitBuf<$size>> for $target {
            type Error = ConcretizationError;
            fn try_from(value: &SymbolicBitBuf<$size>) -> Result<Self, Self::Error> {
                ConcreteValue::try_from(value).map(ConcreteValue::into_inner)
            }
        }
    };
}

concrete_type!(u8);
concrete_type!(u8, 1);
concrete_type!(u8, 2);
concrete_type!(u8, 3);
concrete_type!(u8, 4);
concrete_type!(u8, 5);
concrete_type!(u8, 6);
concrete_type!(u8, 7);
concrete_type!(u16);
concrete_type!(u32);
concrete_type!(u64);
concrete_type!(u128);
concrete_type!(usize);

impl FromIterator<SymbolicBitVec> for SymbolicBitVec {
    fn from_iter<T: IntoIterator<Item = SymbolicBitVec>>(iter: T) -> Self {
        iter.into_iter()
            .flat_map(|bitvec| bitvec.into_iter())
            .collect()
    }
}

impl FromIterator<SymbolicByte> for SymbolicBitVec {
    fn from_iter<T: IntoIterator<Item = SymbolicByte>>(iter: T) -> Self {
        iter.into_iter()
            .flat_map(|byte| byte.into_inner().into_iter())
            .collect()
    }
}

impl FromIterator<u8> for SymbolicBitVec {
    fn from_iter<T: IntoIterator<Item = u8>>(iter: T) -> Self {
        symbolize(iter).collect()
    }
}

pub fn symbolize(iter: impl IntoIterator<Item = u8>) -> impl Iterator<Item = SymbolicBit> {
    iter.into_iter().flat_map(|byte| {
        let mut bits = [FALSE; 8];

        for (index, bit) in bits.iter_mut().enumerate() {
            *bit = SymbolicBit::Literal((byte & (1 << index)) > 0);
        }

        bits
    })
}

pub fn concretize<'a, T, const N: usize>(
    iter: impl Iterator<Item = &'a SymbolicByte>,
) -> std::result::Result<T, ConcretizationError>
where
    T: LittleEndian<N>,
{
    concretize_bits(iter.map(|byte| byte.as_ref()).flat_map(|bits| bits.iter()))
}

pub fn concretize_into<T, const N: usize>(
    iter: impl IntoIterator<Item = SymbolicByte>,
) -> std::result::Result<T, ConcretizationError>
where
    T: LittleEndian<N>,
{
    concretize_bits_into(
        iter.into_iter()
            .map(|byte| byte.into_inner())
            .flat_map(|bits| bits.into_iter()),
    )
}

pub fn concretize_bits_cow<'a, T, const N: usize>(
    iter: impl Iterator<Item = std::borrow::Cow<'a, SymbolicBit>>,
) -> std::result::Result<T, ConcretizationError>
where
    T: LittleEndian<N>,
{
    // TODO Once we can use this directly in the function signature we can remove N
    //
    // Tracking issue: https://github.com/mnemonikr/symbolic-pcode/issues/108
    const {
        assert!(std::mem::size_of::<T>() == N, "N must match size of T");
    };

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
                    bit_index: 8 * byte_index + bit_index,
                });
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

    Ok(T::from_words(bytes))
}

pub fn concretize_bits<'a, T, const N: usize>(
    iter: impl Iterator<Item = &'a SymbolicBit>,
) -> std::result::Result<T, ConcretizationError>
where
    T: LittleEndian<N>,
{
    concretize_bits_cow(iter.map(std::borrow::Cow::Borrowed))
}

pub fn concretize_bits_into<T, const N: usize>(
    iter: impl IntoIterator<Item = SymbolicBit>,
) -> std::result::Result<T, ConcretizationError>
where
    T: LittleEndian<N>,
{
    concretize_bits_cow(iter.into_iter().map(std::borrow::Cow::Owned))
}

/// Little-endian representation of an object with a known size at compile-time. The default word
/// size is a byte.
pub trait LittleEndian<const N: usize, T = u8> {
    /// Create an instance of this object from a little-endian representation.
    fn from_words(words: [T; N]) -> Self;

    /// Convert self into an array of words.
    fn into_words(self) -> [T; N];
}

macro_rules! impl_little_endian {
    ($target:ty) => {
        impl_little_endian!($target, { std::mem::size_of::<$target>() });
    };
    ($target:ty, $size:expr) => {
        impl LittleEndian<$size> for $target {
            fn from_words(bytes: [u8; $size]) -> Self {
                Self::from_le_bytes(bytes)
            }

            fn into_words(self) -> [u8; $size] {
                self.to_le_bytes()
            }
        }
    };
}

impl_little_endian!(usize);
impl_little_endian!(u128);
impl_little_endian!(u64);
impl_little_endian!(u32);
impl_little_endian!(u16);
impl_little_endian!(u8);
