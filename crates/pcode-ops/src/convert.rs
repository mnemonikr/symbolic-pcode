use std::ops::Deref;

#[derive(thiserror::Error, Debug)]
pub enum TryFromPcodeValueError {
    #[error("pcode value exceeds size of target type")]
    SizeExceeded,

    #[error("failed to convert byte at index {index}")]
    InvalidByte { index: usize },
}

/// Little-endian representation of an object with a known size at compile-time. The default word
/// size is a byte.
pub trait LittleEndian<const N: usize, T = u8> {
    /// Create an instance of this object from a little-endian representation.
    fn from_words(words: [T; N]) -> Self;

    /// Convert self into an array of words.
    fn into_words(self) -> [T; N];
}

#[repr(transparent)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct PcodeValue<T: crate::PcodeOps> {
    inner: T,
}

impl<T: crate::PcodeOps> From<T> for PcodeValue<T> {
    fn from(value: T) -> Self {
        Self { inner: value }
    }
}

impl<T: crate::PcodeOps> Deref for PcodeValue<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

macro_rules! impl_tryfrom_pcodevalue {
    ($target:ty) => {
        impl_tryfrom_pcodevalue!($target, { std::mem::size_of::<$target>() }, {
            stringify!($target).starts_with("i")
        });
    };
    ($target:ty, $size:expr, $signed:expr) => {
        impl<T: crate::PcodeOps> TryFrom<PcodeValue<T>> for $target {
            type Error = TryFromPcodeValueError;

            fn try_from(pcode_value: PcodeValue<T>) -> Result<Self, Self::Error> {
                const BYTES: usize = (<$target>::BITS / u8::BITS) as usize;
                let pcode_value = match usize::cmp(&pcode_value.num_bytes(), &BYTES) {
                    std::cmp::Ordering::Less => {
                        if $signed {
                            pcode_value.inner.sign_extend(BYTES).into()
                        } else {
                            pcode_value.inner.zero_extend(BYTES).into()
                        }
                    }
                    std::cmp::Ordering::Equal => pcode_value,
                    std::cmp::Ordering::Greater => {
                        return Err(TryFromPcodeValueError::SizeExceeded)
                    }
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
                    .map_err(|_| TryFromPcodeValueError::SizeExceeded)?;
                Ok(<$target>::from_le_bytes(bytes))
            }
        }
    };
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

impl_little_endian!(usize);
impl_little_endian!(u128);
impl_little_endian!(u64);
impl_little_endian!(u32);
impl_little_endian!(u16);
impl_little_endian!(u8);
impl_little_endian!(isize);
impl_little_endian!(i128);
impl_little_endian!(i64);
impl_little_endian!(i32);
impl_little_endian!(i16);
impl_little_endian!(i8);
