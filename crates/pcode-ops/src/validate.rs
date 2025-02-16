use std::fmt::Debug;
use std::marker::PhantomData;

use crate::PcodeOps;

/// Error returned when validation fails
#[derive(Debug)]
pub enum ValidationError {
    IncorrectValue {
        op: Operation,
        expected: u64,
        actual: u64,
    },
    IncorrectBit {
        op: BitOperation,
        expected: bool,
        actual: bool,
    },
    ValueConversionFailure {
        op: Operation,
        expected: u64,
        err: String,
    },
    BitConversionFailure {
        op: BitOperation,
        expected: bool,
        err: String,
    },
}

/// Validation result
pub type Result = std::result::Result<(), ValidationError>;

#[non_exhaustive]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Operation {
    And(u64, u64),
    Or(u64, u64),
    Xor(u64, u64),
    Not(u8),
    ShiftLeft(u8, u32),
    SignedShiftRight(u8, u32),
    UnsignedShiftRight(u8, u32),
    Add(u64, u64),
    Subtract(i64, i64),
    Multiply(i64, i64),
    UnsignedDivide(u64, u64),
    SignedDivide(i64, i64),
    UnsignedRemainder(u64, u64),
    SignedRemainder(i64, i64),
    Negate(i64),
    ZeroExtend(u8, usize),
    SignExtend(u8, usize),
    Popcount(u64),
    Piece(u8, u8),
    TruncateToSize(u16, usize),
    TruncateTrailingBytes(u16, usize),
}

#[non_exhaustive]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BitOperation {
    UnsignedCarry(u64, u64),
    SignedCarry(i64, i64),
    Borrow(i64, i64),
    Lsb(u64),
    Equals(u64, u64),
    NotEquals(u64, u64),
    UnsignedLessThan(u8, u8),
    UnsignedGreaterThan(u8, u8),
    UnsignedLessThanOrEquals(u8, u8),
    UnsignedGreaterThanOrEquals(u8, u8),
    SignedLessThan(i8, i8),
    SignedGreaterThan(i8, i8),
    SignedLessThanOrEquals(i8, i8),
    SignedGreaterThanOrEquals(i8, i8),
}

impl Operation {
    fn evaluate<T: PcodeOps>(&self) -> T {
        match *self {
            Self::Popcount(x) => T::from_le(x).popcount(),
            Self::And(x, y) => T::from_le(x).and(T::from_le(y)),
            Self::Or(x, y) => T::from_le(x).or(T::from_le(y)),
            Self::Xor(x, y) => T::from_le(x).xor(T::from_le(y)),
            Self::Not(x) => T::from_le(x).not(),
            Self::ShiftLeft(x, y) => T::from_le(x).shift_left(T::from_le(y)),
            Self::SignedShiftRight(x, y) => T::from_le(x).signed_shift_right(T::from_le(y)),
            Self::UnsignedShiftRight(x, y) => T::from_le(x).unsigned_shift_right(T::from_le(y)),
            Self::Add(x, y) => T::from_le(x).add(T::from_le(y)),
            Self::Subtract(x, y) => T::from_le(x).subtract(T::from_le(y)),
            Self::Multiply(x, y) => T::from_le(x).multiply(T::from_le(y)),
            Self::UnsignedDivide(x, y) => T::from_le(x).unsigned_divide(T::from_le(y)),
            Self::SignedDivide(x, y) => T::from_le(x).signed_divide(T::from_le(y)),
            Self::UnsignedRemainder(x, y) => T::from_le(x).unsigned_remainder(T::from_le(y)),
            Self::SignedRemainder(x, y) => T::from_le(x).signed_remainder(T::from_le(y)),
            Self::Negate(x) => T::from_le(x).negate(),
            Self::ZeroExtend(x, y) => T::from_le(x).zero_extend(y),
            Self::SignExtend(x, y) => T::from_le(x).sign_extend(y),
            Self::Piece(x, y) => T::from_le(x).piece(T::from_le(y)),
            Self::TruncateToSize(x, y) => T::from_le(x).truncate_to_size(y),
            Self::TruncateTrailingBytes(x, y) => T::from_le(x).truncate_trailing_bytes(y as u64),
        }
    }
}

impl BitOperation {
    fn evaluate<T: PcodeOps>(self) -> T::Bit {
        match self {
            BitOperation::UnsignedCarry(lhs, rhs) => {
                T::from_le(lhs).unsigned_carry(T::from_le(rhs))
            }
            BitOperation::SignedCarry(lhs, rhs) => T::from_le(lhs).signed_carry(T::from_le(rhs)),
            BitOperation::Borrow(lhs, rhs) => T::from_le(lhs).borrow(T::from_le(rhs)),
            BitOperation::Lsb(x) => T::from_le(x).lsb(),
            BitOperation::Equals(lhs, rhs) => T::from_le(lhs).equals(T::from_le(rhs)),
            BitOperation::NotEquals(lhs, rhs) => T::from_le(lhs).not_equals(T::from_le(rhs)),
            BitOperation::UnsignedLessThan(lhs, rhs) => {
                T::from_le(lhs).unsigned_less_than(T::from_le(rhs))
            }
            BitOperation::UnsignedGreaterThan(lhs, rhs) => {
                T::from_le(lhs).unsigned_greater_than(T::from_le(rhs))
            }
            BitOperation::UnsignedLessThanOrEquals(lhs, rhs) => {
                T::from_le(lhs).unsigned_less_than_or_equals(T::from_le(rhs))
            }
            BitOperation::UnsignedGreaterThanOrEquals(lhs, rhs) => {
                T::from_le(lhs).unsigned_greater_than_or_equals(T::from_le(rhs))
            }
            BitOperation::SignedLessThan(lhs, rhs) => {
                T::from_le(lhs).signed_less_than(T::from_le(rhs))
            }
            BitOperation::SignedGreaterThan(lhs, rhs) => {
                T::from_le(lhs).signed_greater_than(T::from_le(rhs))
            }
            BitOperation::SignedLessThanOrEquals(lhs, rhs) => {
                T::from_le(lhs).signed_less_than_or_equals(T::from_le(rhs))
            }
            BitOperation::SignedGreaterThanOrEquals(lhs, rhs) => {
                T::from_le(lhs).signed_greater_than_or_equals(T::from_le(rhs))
            }
        }
    }
}

/// Validation functions for the implementation of [PcodeOps] for a given type.
pub struct Validator<T: PcodeOps> {
    // No subtyping (invariant), !Send + !Sync
    _phantom: PhantomData<*mut T>,
}

impl<T: PcodeOps + std::fmt::Debug> Validator<T>
where
    <T as TryInto<u64>>::Error: Debug,
    <<T as PcodeOps>::Bit as TryInto<bool>>::Error: Debug,
{
    /// Validate all of the [PcodeOps] operations.
    pub fn validate() -> Result {
        // Shift ops
        Self::shift_left()?;
        Self::signed_shift_right()?;
        Self::unsigned_shift_right()?;

        // Size manipulation ops
        Self::zero_extend()?;
        Self::sign_extend()?;
        Self::piece()?;
        Self::truncate_to_size()?;
        Self::truncate_trailing_bytes()?;

        // Arithmetic
        Self::add()?;
        Self::unsigned_carry()?;
        Self::signed_carry()?;
        Self::negate()?;
        Self::subtract()?;
        Self::borrow()?;
        Self::multiplty()?;
        Self::unsigned_divide()?;
        Self::signed_divide()?;
        Self::unsigned_remainder()?;
        Self::signed_remainder()?;

        // Bitwise ops
        Self::and()?;
        Self::or()?;
        Self::xor()?;
        Self::not()?;

        // Comparison ops
        Self::signed_less_than()?;
        Self::unsigned_less_than()?;
        Self::signed_greater_than()?;
        Self::unsigned_greater_than()?;
        Self::signed_less_than_or_equals()?;
        Self::unsigned_less_than_or_equals()?;
        Self::signed_greater_than_or_equals()?;
        Self::unsigned_greater_than_or_equals()?;
        Self::equals()?;
        Self::not_equals()?;

        // Other
        Self::popcount()?;

        Ok(())
    }

    fn and() -> Result {
        let test_values = [
            (0b0, 0b0, 0b0),
            (0b0, 0b1, 0b0),
            (0b1, 0b0, 0b0),
            (0b1, 0b1, 0b1),
            (0xFF, 0xA5, 0xA5),
            (0x00, 0xA5, 0x00),
        ];

        for (lhs, rhs, expected) in test_values {
            expect_op::<T>(Operation::And(lhs, rhs), expected)?;
        }

        Ok(())
    }

    fn or() -> Result {
        let test_values = [
            (0b0, 0b0, 0b0),
            (0b0, 0b1, 0b1),
            (0b1, 0b0, 0b1),
            (0b1, 0b1, 0b1),
            (0xFF, 0xA5, 0xFF),
            (0x00, 0xA5, 0xA5),
        ];

        for (lhs, rhs, expected) in test_values {
            expect_op::<T>(Operation::Or(lhs, rhs), expected)?;
        }

        Ok(())
    }

    fn xor() -> Result {
        let test_values = [
            (0b0, 0b0, 0b0),
            (0b0, 0b1, 0b1),
            (0b1, 0b0, 0b1),
            (0b1, 0b1, 0b0),
            (0xFF, 0xA5, 0x5A),
            (0x00, 0xA5, 0xA5),
        ];

        for (lhs, rhs, expected) in test_values {
            expect_op::<T>(Operation::Xor(lhs, rhs), expected)?;
        }

        Ok(())
    }

    fn not() -> Result {
        let test_values = [(0xFF, 0x00), (0x00, 0xFF), (0xA5, 0x5A)];

        for (lhs, expected) in test_values {
            expect_op::<T>(Operation::Not(lhs), expected)?;
        }

        Ok(())
    }

    fn equals() -> Result {
        let test_values = [(0xFF, 0x00, false), (0x00, 0xFF, false), (0x00, 0x00, true)];

        for (lhs, rhs, expected) in test_values {
            expect_bit_op::<T>(BitOperation::Equals(lhs, rhs), expected)?;
        }

        Ok(())
    }

    fn not_equals() -> Result {
        let test_values = [(0xFF, 0x00, true), (0x00, 0xFF, true), (0x00, 0x00, false)];

        for (lhs, rhs, expected) in test_values {
            expect_bit_op::<T>(BitOperation::NotEquals(lhs, rhs), expected)?;
        }

        Ok(())
    }

    fn unsigned_greater_than() -> Result {
        let test_values = [(0xFF, 0x00, true), (0x00, 0xFF, false), (0x00, 0x00, false)];

        for (lhs, rhs, expected) in test_values {
            expect_bit_op::<T>(BitOperation::UnsignedGreaterThan(lhs, rhs), expected)?;
        }

        Ok(())
    }

    fn unsigned_less_than() -> Result {
        let test_values = [(0xFF, 0x00, false), (0x00, 0xFF, true), (0x00, 0x00, false)];

        for (lhs, rhs, expected) in test_values {
            expect_bit_op::<T>(BitOperation::UnsignedLessThan(lhs, rhs), expected)?;
        }

        Ok(())
    }

    fn unsigned_greater_than_or_equals() -> Result {
        let test_values = [(0xFF, 0x00, true), (0x00, 0xFF, false), (0x00, 0x00, true)];

        for (lhs, rhs, expected) in test_values {
            expect_bit_op::<T>(
                BitOperation::UnsignedGreaterThanOrEquals(lhs, rhs),
                expected,
            )?;
        }

        Ok(())
    }

    fn unsigned_less_than_or_equals() -> Result {
        let test_values = [(0, 0, true), (1, 0, false), (0, 1, true)];

        for (lhs, rhs, expected) in test_values {
            expect_bit_op::<T>(BitOperation::UnsignedLessThanOrEquals(lhs, rhs), expected)?;
        }

        Ok(())
    }

    fn signed_greater_than() -> Result {
        let test_values = [
            (-1, 0, false),
            (0, -1, true),
            (0, 0, false),
            (1, 0, true),
            (0, 1, false),
        ];

        for (lhs, rhs, expected) in test_values {
            expect_bit_op::<T>(BitOperation::SignedGreaterThan(lhs, rhs), expected)?;
        }

        Ok(())
    }

    fn signed_less_than() -> Result {
        let test_values = [
            (-1, 0, true),
            (0, -1, false),
            (0, 0, false),
            (1, 0, false),
            (0, 1, true),
        ];

        for (lhs, rhs, expected) in test_values {
            expect_bit_op::<T>(BitOperation::SignedLessThan(lhs, rhs), expected)?;
        }

        Ok(())
    }

    fn signed_greater_than_or_equals() -> Result {
        let test_values = [
            (-1, 0, false),
            (0, -1, true),
            (0, 0, true),
            (1, 0, true),
            (0, 1, false),
        ];

        for (lhs, rhs, expected) in test_values {
            expect_bit_op::<T>(BitOperation::SignedGreaterThanOrEquals(lhs, rhs), expected)?;
        }

        Ok(())
    }

    fn signed_less_than_or_equals() -> Result {
        let test_values = [
            (-1, 0, true),
            (0, -1, false),
            (0, 0, true),
            (1, 0, false),
            (0, 1, true),
        ];

        for (lhs, rhs, expected) in test_values {
            expect_bit_op::<T>(BitOperation::SignedLessThanOrEquals(lhs, rhs), expected)?;
        }

        Ok(())
    }

    fn popcount() -> Result {
        let test_values = [(0b0, 0b0), (0b1, 0b1), (0xFF, 8), (0x00, 0), (0xA5, 4)];
        for (lhs, expected) in test_values {
            expect_op::<T>(Operation::Popcount(lhs), expected)?;
        }

        Ok(())
    }

    fn shift_left() -> Result {
        let test_values = [
            (0x00, 0x00, 0x00),
            (0x01, 0x00, 0x01),
            (0x01, 0x01, 0x02),
            (0x01, 0x02, 0x04),
            (0x01, 0x03, 0x08),
            (0x01, 0x04, 0x10),
            (0x01, 0x05, 0x20),
            (0x01, 0x06, 0x40),
            (0x01, 0x07, 0x80),
            (0x01, 0x08, 0x00),
            // If RHS exceeds number of bits in LHS then result is always 0
            (0x01, 0x09, 0x00),
        ];

        for (lhs, rhs, expected) in test_values {
            expect_op::<T>(Operation::ShiftLeft(lhs, rhs), expected)?;
        }

        Ok(())
    }

    fn signed_shift_right() -> Result {
        let test_values = [
            (0x00, 0x00, 0x00),
            // Positive tests
            (0x7F, 0x00, 0x7F),
            (0x7F, 0x01, 0x3F),
            (0x7F, 0x02, 0x1F),
            (0x7F, 0x03, 0x0F),
            (0x7F, 0x04, 0x07),
            (0x7F, 0x05, 0x03),
            (0x7F, 0x06, 0x01),
            (0x7F, 0x07, 0x00),
            (0x7F, 0x08, 0x00),
            // If RHS exceeds number of bits in LHS then result is 0 when positive
            (0x7F, 0x09, 0x00),
            // Negative tests
            (0x80, 0x00, 0x80),
            (0x80, 0x01, 0xC0),
            (0x80, 0x02, 0xE0),
            (0x80, 0x03, 0xF0),
            (0x80, 0x04, 0xF8),
            (0x80, 0x05, 0xFC),
            (0x80, 0x06, 0xFE),
            (0x80, 0x07, 0xFF),
            (0x80, 0x08, 0xFF),
            // If RHS exceeds number of bits in LHS then result is -1 when negative
            (0x80, 0x09, 0xFF),
        ];
        for (lhs, rhs, expected) in test_values {
            expect_op::<T>(Operation::SignedShiftRight(lhs, rhs), expected)?;
        }
        Ok(())
    }

    fn unsigned_shift_right() -> Result {
        let test_values = [
            (0x00, 0x00, 0x00),
            // Positive tests
            (0xFF, 0x00, 0xFF),
            (0xFF, 0x01, 0x7F),
            (0xFF, 0x02, 0x3F),
            (0xFF, 0x03, 0x1F),
            (0xFF, 0x04, 0x0F),
            (0xFF, 0x05, 0x07),
            (0xFF, 0x06, 0x03),
            (0xFF, 0x07, 0x01),
            (0xFF, 0x08, 0x00),
            // If RHS exceeds number of bits in LHS then result is 0 when positive
            (0xFF, 0x09, 0x00),
        ];
        for (lhs, rhs, expected) in test_values {
            expect_op::<T>(Operation::UnsignedShiftRight(lhs, rhs), expected)?;
        }
        Ok(())
    }

    fn zero_extend() -> Result {
        let test_values = [(0x00, 0x00), (0xFF, 0xFF)];
        for (lhs, expected) in test_values {
            expect_op::<T>(Operation::ZeroExtend(lhs, 2), expected)?;
        }
        Ok(())
    }

    fn sign_extend() -> Result {
        let test_values = [(0x00, 0x00), (0xFF, 0xFFFF)];
        for (lhs, expected) in test_values {
            expect_op::<T>(Operation::SignExtend(lhs, 2), expected)?;
        }
        Ok(())
    }

    fn add() -> Result {
        let test_values = [
            (0, 0, 0),
            (0, 1, 1),
            (1, 1, 2),
            (0xFEDCBA9876543210, 0x0123456789ABCDEF, u64::MAX),
            (u64::MAX, 1, 0),
        ];
        for (lhs, rhs, expected) in test_values {
            expect_op::<T>(Operation::Add(lhs, rhs), expected)?;
        }
        Ok(())
    }

    fn unsigned_carry() -> Result {
        let values = [
            (0xFEDCBA9876543210, 0x0123456789ABCDEF, false),
            (u64::MAX, 1, true),
        ];

        for (lhs, rhs, expected) in values {
            expect_bit_op::<T>(BitOperation::UnsignedCarry(lhs, rhs), expected)?;
        }

        Ok(())
    }

    fn signed_carry() -> Result {
        let values = [
            (0xFEDCBA9876543210u64 as i64, 0x0123456789ABCDEF, false),
            (i64::MAX, 1i64, true),
            (i64::MIN, i64::MIN, true),
            (i64::MIN, -1, true),
            ((u64::MAX as i64), 1, false),
        ];

        for (lhs, rhs, expected) in values {
            expect_bit_op::<T>(BitOperation::SignedCarry(lhs, rhs), expected)?;
        }

        Ok(())
    }

    fn negate() -> Result {
        let test_values = [
            (-0x0123456789ABCDEFi64, 0x0123456789ABCDEFi64),
            (0x0123456789ABCDEF, -0x0123456789ABCDEF),
            (i64::MIN, i64::MIN),
        ];

        for (lhs, expected) in test_values {
            expect_op::<T>(Operation::Negate(lhs), expected as u64)?;
        }

        Ok(())
    }

    fn subtract() -> Result {
        let test_values = [(0, 0, 0), (0, 1, -1), (1, 0, 1), (0, i64::MIN, i64::MIN)];

        for (lhs, rhs, expected) in test_values {
            expect_op::<T>(Operation::Subtract(lhs, rhs), expected as u64)?;
        }

        Ok(())
    }

    fn borrow() -> Result {
        let values = [
            (0xFEDCBA9876543210u64 as i64, 0x0123456789ABCDEF, false),
            (i64::MAX, i64::MIN, true),
            (i64::MIN, i64::MAX, true),
            (i64::MIN, 1, true),
            ((u64::MAX as i64), 1, false),
        ];

        for (lhs, rhs, expected) in values {
            expect_bit_op::<T>(BitOperation::Borrow(lhs, rhs), expected)?;
        }

        Ok(())
    }

    fn multiplty() -> Result {
        let test_values = [
            (0, 0, 0),
            (1, 1, 1),
            (-1, -1, 1),
            (-1, 1, -1),
            (2, 0, 0),
            (2, 2, 4),
            (i64::MIN, -1, i64::MIN),
        ];

        for (lhs, rhs, expected) in test_values {
            expect_op::<T>(Operation::Multiply(lhs, rhs), expected as u64)?;
        }

        Ok(())
    }

    fn unsigned_divide() -> Result {
        let test_values = [
            (1, u64::MAX, 0),
            (0, u64::MAX, 0),
            (u64::MAX, 1, u64::MAX),
            (5, 2, 2),
            (2, 2, 1),
            (u64::MAX, u64::MAX, 1),
        ];

        for (lhs, rhs, expected) in test_values {
            expect_op::<T>(Operation::UnsignedDivide(lhs, rhs), expected)?;
        }

        Ok(())
    }

    fn signed_divide() -> Result {
        let test_values = [
            (1, i64::MAX, 0),
            (0, i64::MAX, 0),
            (i64::MAX, 1, i64::MAX),
            (5, 2, 2),
            (-2, -2, 1),
            (i64::MAX, i64::MAX, 1),
            (i64::MIN, -1, i64::MIN),
        ];

        for (lhs, rhs, expected) in test_values {
            expect_op::<T>(Operation::SignedDivide(lhs, rhs), expected as u64)?;
        }

        Ok(())
    }

    fn unsigned_remainder() -> Result {
        let test_values = [
            (1, u64::MAX, 1),
            (0, u64::MAX, 0),
            (u64::MAX, 1, 0),
            (5, 2, 1),
            (2, 2, 0),
            (u64::MAX, u64::MAX, 0),
        ];

        for (lhs, rhs, expected) in test_values {
            expect_op::<T>(Operation::UnsignedRemainder(lhs, rhs), expected)?;
        }

        Ok(())
    }

    fn signed_remainder() -> Result {
        let test_values = [
            (1, i64::MAX, 1),
            (0, i64::MAX, 0),
            (i64::MAX, 1, 0),
            (5, 2, 1),
            (2, 2, 0),
            (i64::MAX, i64::MAX, 0),
            (i64::MIN, -1, 0),
        ];

        for (lhs, rhs, expected) in test_values {
            expect_op::<T>(Operation::SignedRemainder(lhs, rhs), expected as u64)?;
        }

        Ok(())
    }

    fn piece() -> Result {
        let test_values = [(0xAB, 0xCD, 0xABCD), (0xF0, 0x0F, 0xF00F)];

        for (lhs, rhs, expected) in test_values {
            expect_op::<T>(Operation::Piece(lhs, rhs), expected)?;
        }

        Ok(())
    }

    fn truncate_to_size() -> Result {
        let test_values = [(0xABCD, 1, 0xCD), (0xABCD, 2, 0xABCD), (0xABCD, 3, 0xABCD)];

        for (lhs, rhs, expected) in test_values {
            expect_op::<T>(Operation::TruncateToSize(lhs, rhs), expected)?;
        }

        Ok(())
    }

    fn truncate_trailing_bytes() -> Result {
        // NOTE: Truncation exceeding the number of bytes in an input is prohibited
        let test_values = [(0xABCD, 1, 0xAB), (0xABCD, 0, 0xABCD)];

        for (lhs, rhs, expected) in test_values {
            expect_op::<T>(Operation::TruncateTrailingBytes(lhs, rhs), expected)?;
        }

        Ok(())
    }
}

fn expect_op<T: PcodeOps + Debug>(op: Operation, expected: u64) -> Result
where
    <T as TryInto<u64>>::Error: Debug,
{
    let actual = op.evaluate::<T>();
    let actual_str = format!("{actual:?}");
    let actual = actual
        .try_into()
        .map_err(|err| ValidationError::ValueConversionFailure {
            op,
            expected,
            err: format!("Failed to convert {actual_str}: {err:?}"),
        })?;
    if actual == expected {
        Ok(())
    } else {
        Err(ValidationError::IncorrectValue {
            op,
            actual,
            expected,
        })
    }
}

fn expect_bit_op<T: PcodeOps>(op: BitOperation, expected: bool) -> Result
where
    <<T as PcodeOps>::Bit as TryInto<bool>>::Error: Debug,
{
    let actual = op.evaluate::<T>();
    let actual_str = format!("{actual:?}");
    let actual = actual
        .try_into()
        .map_err(|err| ValidationError::BitConversionFailure {
            op,
            expected,
            err: format!("Failed to convert {actual_str}: {err:?}"),
        })?;
    if actual == expected {
        Ok(())
    } else {
        Err(ValidationError::IncorrectBit {
            op,
            actual,
            expected,
        })
    }
}
