use std::cmp::Ordering;
use std::ops::Deref;

use crate::PcodeOps;

/// Error while converting from [PcodeValue]
#[derive(thiserror::Error, Debug)]
pub enum TryFromPcodeValueError {
    #[error("pcode value does not match size expectations of target type")]
    InvalidSize,

    #[error("failed to convert byte at index {index}")]
    InvalidByte { index: usize },
}

/// Wrapper around a value that implements [PcodeOps]. This wrapper supports conversions of pcode
/// values to primitive types.
///
/// ## Example type conversion
///
/// ```
/// # use pcode_ops::{convert::PcodeValue, Pcode128};
/// let pcode128 = Pcode128::from(u64::MAX);
/// let pcode_value = PcodeValue::from(pcode128);
/// let value: u64 = pcode_value.try_into().unwrap();
/// assert_eq!(value, u64::MAX);
/// ```
#[repr(transparent)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct PcodeValue<T: PcodeOps> {
    inner: T,
}

impl<T: PcodeOps> PcodeValue<T> {
    /// Consumes the object and returns the inner value
    pub fn into_inner(self) -> T {
        self.inner
    }

    /// Create a value from a byte
    pub fn from_byte(byte: T::Byte) -> Self {
        Self {
            inner: std::iter::once(byte).collect(),
        }
    }

    /// Create a value from a bit
    pub fn from_bit(bit: T::Bit) -> Self {
        let mut byte = std::array::from_fn(|_| T::Bit::from(false));
        byte[0] = bit;
        Self::from_byte(byte.into())
    }
}

impl<T: PcodeOps> TryFrom<PcodeValue<T>> for Vec<u8> {
    type Error = TryFromPcodeValueError;

    fn try_from(pcode_value: PcodeValue<T>) -> Result<Self, Self::Error> {
        pcode_value
            .inner
            .into_le_bytes()
            .enumerate()
            .map(|(index, byte)| {
                byte.try_into()
                    .map_err(|_| TryFromPcodeValueError::InvalidByte { index })
            })
            .collect::<Result<Vec<_>, _>>()
    }
}

impl<const N: usize, T: PcodeOps> TryFrom<PcodeValue<T>> for [u8; N] {
    type Error = TryFromPcodeValueError;

    fn try_from(pcode_value: PcodeValue<T>) -> Result<Self, Self::Error> {
        let bytes: Vec<u8> = pcode_value.try_into()?;
        bytes
            .try_into()
            .map_err(|_| TryFromPcodeValueError::InvalidSize)
    }
}

impl<const N: usize, T: PcodeOps> From<[u8; N]> for PcodeValue<T> {
    fn from(value: [u8; N]) -> Self {
        value.into_iter().collect()
    }
}

impl<T: PcodeOps> From<T> for PcodeValue<T> {
    fn from(value: T) -> Self {
        Self { inner: value }
    }
}

impl<T: PcodeOps> Deref for PcodeValue<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<T: PcodeOps> FromIterator<u8> for PcodeValue<T> {
    fn from_iter<I: IntoIterator<Item = u8>>(iter: I) -> Self {
        PcodeValue::from(iter.into_iter().map(T::Byte::from).collect::<T>())
    }
}

macro_rules! impl_tryfrom_pcodevalue {
    ($target:ty) => {
        impl_tryfrom_pcodevalue!($target, { std::mem::size_of::<$target>() }, {
            stringify!($target).starts_with("i")
        });
    };
    ($target:ty, $size:expr, $signed:expr) => {
        impl<T: PcodeOps> From<$target> for PcodeValue<T> {
            fn from(value: $target) -> PcodeValue<T> {
                let pcode: T = value.to_le_bytes().into_iter().map(T::Byte::from).collect();
                PcodeValue::from(pcode)
            }
        }

        impl<T: PcodeOps> TryFrom<PcodeValue<T>> for $target {
            type Error = TryFromPcodeValueError;

            fn try_from(pcode_value: PcodeValue<T>) -> Result<Self, Self::Error> {
                const BYTES: usize = (<$target>::BITS / u8::BITS) as usize;
                let pcode_value = match usize::cmp(&pcode_value.num_bytes(), &BYTES) {
                    Ordering::Less => {
                        if $signed {
                            pcode_value.inner.sign_extend(BYTES).into()
                        } else {
                            pcode_value.inner.zero_extend(BYTES).into()
                        }
                    }
                    Ordering::Equal => pcode_value,
                    Ordering::Greater => return Err(TryFromPcodeValueError::InvalidSize),
                };

                let bytes = pcode_value
                    .inner
                    .into_le_bytes()
                    .enumerate()
                    .map(|(index, byte)| {
                        byte.try_into()
                            .map_err(|_| TryFromPcodeValueError::InvalidByte { index })
                    })
                    .collect::<Result<Vec<u8>, _>>()?;

                let bytes = bytes
                    .try_into()
                    .map_err(|_| TryFromPcodeValueError::InvalidSize)?;
                Ok(<$target>::from_le_bytes(bytes))
            }
        }
    };
}

impl_tryfrom_pcodevalue!(usize);
impl_tryfrom_pcodevalue!(u128);
impl_tryfrom_pcodevalue!(u64);
impl_tryfrom_pcodevalue!(u32);
impl_tryfrom_pcodevalue!(u16);
impl_tryfrom_pcodevalue!(u8);
impl_tryfrom_pcodevalue!(isize);
impl_tryfrom_pcodevalue!(i128);
impl_tryfrom_pcodevalue!(i64);
impl_tryfrom_pcodevalue!(i32);
impl_tryfrom_pcodevalue!(i16);
impl_tryfrom_pcodevalue!(i8);
