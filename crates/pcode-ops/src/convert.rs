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
impl_little_endian!(isize);
impl_little_endian!(i128);
impl_little_endian!(i64);
impl_little_endian!(i32);
impl_little_endian!(i16);
impl_little_endian!(i8);
