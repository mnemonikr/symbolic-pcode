/// This trait is used to implement Sleigh pcode operations. While the representation of the value
/// is not prescribed, several of the operations assume the ability to reason about the bytes of
/// the value. In some cases the value may be interpreted as a boolean value as the result of a
/// condition evaluation.
pub trait PcodeOps: BitwisePcodeOps + FromIterator<Self::Byte> {
    /// A representation of a byte (8-bit) value.
    type Byte: From<u8> + TryInto<u8> + Clone + From<Self::Bit>;

    /// A representation of a single bit.
    type Bit: BitwisePcodeOps + From<bool> + TryInto<bool> + std::fmt::Debug + Clone;

    /// Create a value that is the given bit repeated to fill the specified number of bytes.
    fn fill_bytes_with(bit: Self::Bit, num_bytes: usize) -> Self;

    /// Returns the number of bytes used to represent this value.
    fn num_bytes(&self) -> usize;

    /// Converts value that has little-endian byte representation.
    fn from_le<const N: usize, T: crate::convert::LittleEndian<N>>(value: T) -> Self {
        value
            .into_words()
            .into_iter()
            .map(Self::Byte::from)
            .collect()
    }

    /// Returns an iterator of bytes in a little endian sequence.
    fn into_le_bytes(self) -> impl ExactSizeIterator<Item = Self::Byte>;

    /// This is standard integer addition. It works for either unsigned or signed interpretations
    /// of the integer encoding (twos complement). Size of both inputs and output must be the same.
    /// The addition is of course performed modulo this size. Overflow and carry conditions are
    /// calculated by other operations. See INT_CARRY and INT_SCARRY.
    fn add(self, rhs: Self) -> Self;

    /// This operation checks for unsigned addition overflow or carry conditions. If the result of
    /// adding input0 and input1 as unsigned integers overflows the size of the varnodes, output is
    /// assigned true. Both inputs must be the same size, and output must be size 1.
    fn unsigned_carry(self, rhs: Self) -> Self::Bit;

    /// This operation checks for signed addition overflow or carry conditions. If the result of
    /// adding input0 and input1 as signed integers overflows the size of the varnodes, output is
    /// assigned true. Both inputs must be the same size, and output must be size 1.
    fn signed_carry(self, rhs: Self) -> Self::Bit;

    /// This is the twos complement or arithmetic negation operation. Treating input0 as a signed
    /// integer, the result is the same integer value but with the opposite sign. This is equivalent
    /// to doing a bitwise negation of input0 and then adding one. Both input0 and output must be
    /// the same size.
    fn negate(self) -> Self;

    /// This is standard integer subtraction. It works for either unsigned or signed
    /// interpretations of the integer encoding (twos complement). Size of both inputs and output
    /// must be the same. The subtraction is of course performed modulo this size. Overflow and
    /// borrow conditions are calculated by other operations. See INT_SBORROW and INT_LESS.
    fn subtract(self, rhs: Self) -> Self;

    /// This operation checks for signed subtraction overflow or borrow conditions. If the result of
    /// subtracting input1 from input0 as signed integers overflows the size of the varnodes, output
    /// is assigned true. Both inputs must be the same size, and output must be size 1. Note that
    /// the equivalent unsigned subtraction overflow condition is INT_LESS.
    fn borrow(self, rhs: Self) -> Self::Bit;

    /// This is an integer multiplication operation. The result of multiplying input0 and input1,
    /// viewed as integers, is stored in output. Both inputs and output must be the same size. The
    /// multiplication is performed modulo the size, and the result is true for either a signed or
    /// unsigned interpretation of the inputs and output. To get extended precision results, the
    /// inputs must first by zero-extended or sign-extended to the desired size.
    fn multiply(self, rhs: Self) -> Self;

    /// This is an unsigned integer division operation. Divide input0 by input1, truncating the
    /// result to the nearest integer, and store the result in output. Both inputs and output must
    /// be the same size. There is no handling of division by zero. To simulate a processor's
    /// handling of a division-by-zero trap, other operations must be used before the INT_DIV.
    fn unsigned_divide(self, rhs: Self) -> Self;

    /// This is a signed integer division operation. The resulting integer is the one closest to
    /// the rational value input0/input1 but which is still smaller in absolute value. Both inputs
    /// and output must be the same size. There is no handling of division by zero. To simulate a
    /// processor's handling of a division-by-zero trap, other operations must be used before the
    /// INT_SDIV.
    fn signed_divide(self, rhs: Self) -> Self;

    /// This is an unsigned integer remainder operation. The remainder of performing the unsigned
    /// integer division of input0 and input1 is put in output. Both inputs and output must be the
    /// same size. If q = input0/input1, using the INT_DIV operation defined above, then output
    /// satisfies the equation q*input1 + output = input0, using the INT_MULT and INT_ADD
    /// operations.
    fn unsigned_remainder(self, rhs: Self) -> Self;

    /// This is a signed integer remainder operation. The remainder of performing the signed
    /// integer division of input0 and input1 is put in output. Both inputs and output must be the
    /// same size. If q = input0 s/ input1, using the INT_SDIV operation defined above, then output
    /// satisfies the equation q*input1 + output = input0, using the INT_MULT and INT_ADD
    /// operations.
    fn signed_remainder(self, rhs: Self) -> Self;

    /// Zero-extend the data in input0 and store the result in output. Copy all the data from input0
    /// into the least significant positions of output. Fill out any remaining space in the most
    /// significant bytes of output with zero. The size of output must be strictly bigger than the
    /// size of input.
    fn zero_extend(self, new_size: usize) -> Self;

    /// Sign-extend the data in input0 and store the result in output. Copy all the data from input0
    /// into the least significant positions of output. Fill out any remaining space in the most
    /// significant bytes of output with either zero or all ones (0xff) depending on the most
    /// significant bit of input0. The size of output must be strictly bigger than the size of
    /// input0.
    fn sign_extend(self, new_size: usize) -> Self;

    /// This is a concatenation operator that understands the endianess of the data. The size of
    /// input0 and input1 must add up to the size of output. The data from the inputs is
    /// concatenated in such a way that, if the inputs and output are considered integers, the first
    /// input makes up the most significant part of the output.
    fn piece(self, lsb: Self) -> Self;

    /// This is a truncation operator understands the endianness of the data. Reduces the value size
    /// to the specified number of bytes. If this size is is smaller than the current size, then the
    /// additional most significant bytes are truncated.
    fn truncate_to_size(self, new_size: usize) -> Self;

    /// This is a truncation operator that understands the endianess of the data. Discards the
    /// specified number of least significant bytes of the value.
    fn truncate_trailing_bytes(self, amount: u64) -> Self;

    /// Reduces the value to its least significant bit. Primarily used for operations involving
    /// conditions such as branches.
    fn lsb(self) -> Self::Bit;

    /// This is a bit count (population count) operator. Within the binary representation of the
    /// value contained in the input varnode, the number of 1 bits are counted and then returned
    /// in the output varnode. A value of 0 returns 0, a 4-byte varnode containing the value
    /// 2<sup>32</sup>-1 (all bits set) returns 32, for instance. The input and output varnodes can
    /// have any size. The resulting count is zero extended into the output varnode.
    fn popcount(self) -> Self;

    /// This operation performs a left shift on input0. The value given by input1, interpreted as an
    /// unsigned integer, indicates the number of bits to shift. The vacated (least significant)
    /// bits are filled with zero. If input1 is zero, no shift is performed and input0 is copied
    /// into output. If input1 is larger than the number of bits in output, the result is zero. Both
    /// input0 and output must be the same size. Input1 can be any size.
    fn shift_left(self, rhs: Self) -> Self;

    /// This operation performs an unsigned (logical) right shift on input0. The value given by
    /// input1, interpreted as an unsigned integer, indicates the number of bits to shift. The
    /// vacated (most significant) bits are filled with zero. If input1 is zero, no shift is
    /// performed and input0 is copied into output. If input1 is larger than the number of bits in
    /// output, the result is zero. Both input0 and output must be the same size. Input1 can be any
    /// size.
    fn unsigned_shift_right(self, rhs: Self) -> Self;

    /// This operation performs a signed (arithmetic) right shift on input0. The value given by
    /// input1, interpreted as an unsigned integer, indicates the number of bits to shift. The
    /// vacated bits are filled with the original value of the most significant (sign) bit of
    /// input0. If input1 is zero, no shift is performed and input0 is copied into output. If input1
    /// is larger than the number of bits in output, the result is zero or all 1-bits (-1),
    /// depending on the original sign of input0. Both input0 and output must be the same size.
    /// Input1 can be any size.
    fn signed_shift_right(self, rhs: Self) -> Self;

    /// This is the integer equality operator. Output is assigned true, if input0 equals input1. It
    /// works for signed, unsigned, or any contiguous data where the match must be down to the bit.
    /// Both inputs must be the same size, and the output must have a size of 1.
    fn equals(self, rhs: Self) -> Self::Bit;

    /// This is the integer inequality operator. Output is assigned true, if input0 does not equal
    /// input1. It works for signed, unsigned, or any contiguous data where the match must be down
    /// to the bit. Both inputs must be the same size, and the output must have a size of 1.
    fn not_equals(self, rhs: Self) -> Self::Bit;

    /// This is an unsigned integer comparison operator. If the unsigned integer input0 is strictly
    /// less than the unsigned integer input1, output is set to true. Both inputs must be the same
    /// size, and the output must have a size of 1.
    fn unsigned_less_than(self, rhs: Self) -> Self::Bit;

    /// This is a signed integer comparison operator. If the signed integer input0 is strictly less
    /// than the signed integer input1, output is set to true. Both inputs must be the same size,
    /// and the output must have a size of 1.
    fn signed_less_than(self, rhs: Self) -> Self::Bit;

    /// This is an unsigned integer comparison operator. If the unsigned integer input0 is less than
    /// or equal to the unsigned integer input1, output is set to true. Both inputs must be the same
    /// size, and the output must have a size of 1.
    fn unsigned_less_than_or_equals(self, rhs: Self) -> Self::Bit;

    /// This is a signed integer comparison operator. If the signed integer input0 is less than or
    /// equal to the signed integer input1, output is set to true. Both inputs must be the same
    /// size, and the output must have a size of 1.
    fn signed_less_than_or_equals(self, rhs: Self) -> Self::Bit;

    /// This is a signed integer comparison operator. If the signed integer input0 is greater than
    /// the signed integer input1, output is set to true. Both inputs must be the same size, and the
    /// output must have a size of 1.
    fn signed_greater_than(self, rhs: Self) -> Self::Bit;

    /// This is a signed integer comparison operator. If the signed integer input0 is greater than
    /// or equal to the signed integer input1, output is set to true. Both inputs must be the same
    /// size, and the output must have a size of 1.
    fn signed_greater_than_or_equals(self, rhs: Self) -> Self::Bit;

    /// This is an unsigned integer comparison operator. If the unsigned integer input0 is greater
    /// than the unsigned integer input1, output is set to true. Both inputs must be the same size,
    /// and the output must have a size of 1.
    fn unsigned_greater_than(self, rhs: Self) -> Self::Bit;

    /// This is an unsigned integer comparison operator. If the unsigned integer input0 is greater
    /// than or equal to the unsigned integer input1, output is set to true. Both inputs must be the
    /// same size, and the output must have a size of 1.
    fn unsigned_greater_than_or_equals(self, rhs: Self) -> Self::Bit;
}

/// Bitwise operations supported by pcode values.
pub trait BitwisePcodeOps {
    /// This operation performs a Logical-And on the bits of input0 and input1. Both inputs and
    /// output must be the same size.
    fn and(self, rhs: Self) -> Self;

    /// This is the bitwise negation operation. Output is the result of taking every bit of input0
    /// and flipping it. Both input0 and output must be the same size.
    fn not(self) -> Self;

    /// This operation performs a Logical-Or on the bits of input0 and input1. Both inputs and
    /// output must be the same size.
    fn or(self, rhs: Self) -> Self;

    /// This operation performs a logical Exclusive-Or on the bits of input0 and input1. Both
    /// inputs and output must be the same size.
    fn xor(self, rhs: Self) -> Self;
}

impl<T> BitwisePcodeOps for T
where
    T: std::ops::BitAnd<Output = T>
        + std::ops::BitOr<Output = T>
        + std::ops::BitXor<Output = T>
        + std::ops::Not<Output = T>,
{
    fn and(self, other: Self) -> Self {
        self & other
    }

    fn not(self) -> Self {
        !self
    }

    fn or(self, other: Self) -> Self {
        self | other
    }

    fn xor(self, other: Self) -> Self {
        self ^ other
    }
}
