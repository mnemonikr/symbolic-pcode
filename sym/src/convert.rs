use crate::buf::{SymbolicBitBuf, SymbolicByte};
use crate::sym::{self, ConcretizationError, SymbolicBit, SymbolicBitVec};

impl std::ops::Deref for SymbolicBitVec {
    type Target = [SymbolicBit];

    fn deref(&self) -> &Self::Target {
        &self.bits.as_slice()
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

/// A wrapper around a value that implements [LittleEndian]. This wrapper implements conversions to
/// and from [SymbolicBitVec] and [SymbolicBitBuf]. Use the macro [concrete_type!] to automatically
/// implement the conversions for a type using this wrapper.
///
/// ```
/// # use sym::{ConcreteValue, SymbolicBitBuf};
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
        assert_eq!(8 * BYTES, BITS);

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
        SymbolicBitVec {
            bits: value.symbolize(),
        }
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
        Self {
            bits: iter
                .into_iter()
                .flat_map(|bitvec| bitvec.bits.into_iter())
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
                .flat_map(|byte| byte.into_inner().into_iter())
                .collect(),
        }
    }
}

pub fn symbolize(iter: impl IntoIterator<Item = u8>) -> impl Iterator<Item = SymbolicBit> {
    iter.into_iter().flat_map(|byte| {
        let mut bits = [sym::FALSE; 8];

        for b in 0..8 {
            bits[b] = SymbolicBit::Literal((byte & (1 << b)) > 0);
        }

        bits
    })
}

#[must_use]
pub fn concretize<'a, T, const N: usize>(
    iter: impl Iterator<Item = &'a SymbolicByte>,
) -> std::result::Result<T, ConcretizationError>
where
    T: LittleEndian<N>,
{
    concretize_bits(iter.map(|byte| byte.as_ref()).flat_map(|bits| bits.iter()))
}

#[must_use]
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

#[must_use]
pub fn concretize_bits_cow<'a, T, const N: usize>(
    iter: impl Iterator<Item = std::borrow::Cow<'a, SymbolicBit>>,
) -> std::result::Result<T, ConcretizationError>
where
    T: LittleEndian<N>,
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

    Ok(T::from_words(bytes))
}

#[must_use]
pub fn concretize_bits<'a, T, const N: usize>(
    iter: impl Iterator<Item = &'a SymbolicBit>,
) -> std::result::Result<T, ConcretizationError>
where
    T: LittleEndian<N>,
{
    concretize_bits_cow(iter.map(std::borrow::Cow::Borrowed))
}

#[must_use]
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
        let buf = SymbolicBitBuf::from(value);
        assert_eq!(value, u8::try_from(&buf).unwrap());
        assert_eq!(value, u8::try_from(buf).unwrap());

        let value: u16 = 0xBEEF;
        let buf = SymbolicBitBuf::from(value);
        assert_eq!(value, u16::try_from(&buf).unwrap());
        assert_eq!(value, u16::try_from(buf).unwrap());

        let value: u32 = 0xDEADBEEF;
        let buf = SymbolicBitBuf::from(value);
        assert_eq!(value, u32::try_from(&buf).unwrap());
        assert_eq!(value, u32::try_from(buf).unwrap());

        let value: u64 = 0xFEEDBEEF_0BADF00D;
        let buf = SymbolicBitBuf::from(value);
        assert_eq!(value, u64::try_from(&buf).unwrap());
        assert_eq!(value, u64::try_from(buf).unwrap());

        let value: u128 = 0xFEEDBEEF_0BADF00D_DEADBEEF_DEAD4BED;
        let buf = SymbolicBitBuf::from(value);
        assert_eq!(value, u128::try_from(&buf).unwrap());
        assert_eq!(value, u128::try_from(buf).unwrap());
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

    #[test]
    fn bool_happy() {
        assert!(bool::try_from(SymbolicBit::from(true)).unwrap());
    }

    #[test]
    fn bool_error() {
        let err = bool::try_from(SymbolicBit::Variable(0)).unwrap_err();
        assert!(matches!(
            err,
            ConcretizationError::NonLiteralBit {
                byte_index: 0,
                bit_index: 0,
            }
        ));
    }

    #[test]
    fn bit_bitvec_happy() {
        let bit = SymbolicBit::Variable(0);
        let vec = SymbolicBitVec::from(bit.clone());
        assert_eq!(bit, vec.try_into().unwrap());
    }

    #[test]
    fn bitvec_bit_err() {
        let err = SymbolicBit::try_from(SymbolicBitVec::constant(2, 2)).unwrap_err();
        assert_eq!("value has 2 bits", err);
    }

    #[test]
    fn combine_bitvecs() {
        let zero = SymbolicBitVec::constant(0, 1);
        let one = SymbolicBitVec::constant(1, 1);
        let array = [zero, one];

        // Little endian format means this should be 0b10 when combined
        let combined: SymbolicBitVec = array.into_iter().collect();
        assert_eq!(0b10u8, combined.try_into().unwrap());
    }

    #[test]
    fn combine_symbytes_into_vec() {
        let zero = SymbolicByte::from(0u8);
        let one = SymbolicByte::from(1u8);
        let array = [zero, one];

        // Little endian format means this should be 0x0100 when combined
        let combined: SymbolicBitVec = array.into_iter().collect();
        assert_eq!(0x0100u16, combined.try_into().unwrap());
    }

    #[test]
    fn concretize_happy() {
        let zero = SymbolicByte::from(0u8);
        let one = SymbolicByte::from(1u8);
        let array = [zero, one];

        assert_eq!(0x0100u16, concretize(array.iter()).unwrap());
        assert_eq!(0x0100u16, concretize_into(array).unwrap());
    }

    #[test]
    fn concretize_overflow_err() {
        let zero = SymbolicByte::from(0u8);
        let one = SymbolicByte::from(1u8);
        let array = [zero, one];

        let err = concretize::<u8, 1>(array.iter()).unwrap_err();
        assert!(matches!(
            err,
            ConcretizationError::Overflow { max_bytes: 1 }
        ));

        let err = concretize_into::<u8, 1>(array).unwrap_err();
        assert!(matches!(
            err,
            ConcretizationError::Overflow { max_bytes: 1 }
        ));
    }

    #[test]
    fn concretize_symbolic_err() {
        let byte = SymbolicByte::from(SymbolicBit::Variable(0));
        let array = [byte];
        let err = concretize::<u8, 1>(array.iter()).unwrap_err();
        assert!(matches!(
            err,
            ConcretizationError::NonLiteralBit {
                byte_index: 0,
                bit_index: 0
            }
        ));

        let err = concretize_into::<u8, 1>(array).unwrap_err();
        assert!(matches!(
            err,
            ConcretizationError::NonLiteralBit {
                byte_index: 0,
                bit_index: 0
            }
        ));
    }

    #[test]
    fn concrete_value_derives() {
        let x = ConcreteValue::new(1u8);

        // Clone
        let y = x.clone();

        // PartialEq
        assert_eq!(x, y);

        // Debug
        assert_eq!("ConcreteValue { value: 1 }", format!("{x:?}"));
    }
}
