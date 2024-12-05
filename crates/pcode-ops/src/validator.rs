use crate::PcodeOps;

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
    },
    BitConversionFailure {
        op: BitOperation,
        expected: bool,
    },
}

fn expect_op<T: PcodeOps>(op: Operation, expected: u64) -> Result {
    let actual: u64 = op
        .evaluate::<T>()
        .try_into()
        .map_err(|_| ValidationError::ValueConversionFailure { op, expected })?;
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

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BitOperation {
    UnsignedCarry(u64, u64),
    SignedCarry(i64, i64),
    Borrow(i64, i64),
}

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
}

impl Operation {
    pub fn evaluate<T: PcodeOps>(&self) -> T {
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
        }
    }
}

pub type Result = std::result::Result<(), ValidationError>;

pub struct Validator<T: PcodeOps> {
    // No subtyping (invariant), !Send + !Sync
    _phantom: std::marker::PhantomData<*mut T>,
}

impl<T: PcodeOps> Validator<T> {
    /// Validate all of the [PcodeOps] operations.
    pub fn validate() -> Result {
        // Shift ops
        Self::shift_left()?;
        Self::signed_shift_right()?;
        Self::unsigned_shift_right()?;

        // Extension ops
        Self::zero_extend()?;
        Self::sign_extend()?;

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

        Self::popcount()?;

        // Bitwise ops
        Self::and()?;
        Self::or()?;
        Self::xor()?;
        Self::not()?;

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
            let op = BitOperation::UnsignedCarry(lhs, rhs);
            let lhs = T::from_le(lhs);
            let rhs = T::from_le(rhs);
            let actual = lhs.unsigned_carry(rhs);
            Self::check_bit(op, actual, expected)?;
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
            let op = BitOperation::SignedCarry(lhs, rhs);
            let lhs = T::from_le(lhs);
            let rhs = T::from_le(rhs);
            let actual = lhs.signed_carry(rhs);
            Self::check_bit(op, actual, expected)?;
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
            let op = BitOperation::Borrow(lhs, rhs);
            let lhs = T::from_le(lhs);
            let rhs = T::from_le(rhs);
            let actual = lhs.borrow(rhs);
            Self::check_bit(op, actual, expected)?;
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

    fn check_bit(op: BitOperation, actual: <T as PcodeOps>::Bit, expected: bool) -> Result {
        let actual: bool = actual
            .try_into()
            .map_err(|_| ValidationError::BitConversionFailure { op, expected })?;
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
}
