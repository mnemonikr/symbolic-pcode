use crate::{SymbolicBit, SymbolicBitVec};

#[test]
fn add_bytes_no_carry() {
    let x: SymbolicBitVec = 0xAAu8.into();
    let y: SymbolicBitVec = 0x55u8.into();
    let sum: u8 = (x + y).try_into().expect("failed byte converison");
    assert_eq!(sum, 0xFF);
}

#[test]
fn one_plus_one() {
    let x: SymbolicBitVec = 0x01u8.into();
    let y: SymbolicBitVec = 0x01u8.into();
    let sum: u8 = (x + y).try_into().expect("failed byte converison");
    assert_eq!(sum, 0x02);
}

#[test]
fn maximal_carry_addition() {
    let x: SymbolicBitVec = 0x7Fu8.into();
    let y: SymbolicBitVec = 0x7Fu8.into();
    let sum: u8 = (x + y).try_into().expect("failed byte converison");
    assert_eq!(sum, 0xFE);
}

#[test]
fn negative_one() {
    let one: SymbolicBitVec = 0x01u8.into();
    let negative_one: u8 = (-one).try_into().expect("failed byte converison");
    assert_eq!(negative_one, u8::MAX);
}

#[test]
fn addition_overflow() {
    let x: SymbolicBitVec = u8::MAX.into();
    let sum: u8 = (x + 1u8.into()).try_into().expect("failed byte converison");
    assert_eq!(sum, 0x00);
}

#[test]
fn addition_overflow_carry() {
    let x: SymbolicBitVec = u8::MAX.into();
    let y: SymbolicBitVec = 1u8.into();
    let carry = x.addition_carry_bits(y);
    assert_eq!(carry[8], SymbolicBit::Literal(true));

    let x: SymbolicBitVec = (u8::MAX - 1).into();
    let y: SymbolicBitVec = 1u8.into();
    let carry = x.addition_carry_bits(y);
    assert_eq!(carry[8], SymbolicBit::Literal(false));
}

#[test]
fn subtraction() {
    for n in u8::MIN..=u8::MAX {
        let x: SymbolicBitVec = n.into();
        let y: SymbolicBitVec = n.into();
        let diff: u8 = (x - y).try_into().expect("failed byte converison");
        assert_eq!(diff, 0);
    }
}

#[test]
fn subtraction_borrows() {
    let test_data = (u8::MIN..=u8::MAX)
        .map(|value| (value, value, false))
        .chain(vec![
            (0x00, 0x80, true),  // 0 - (-128) != -128
            (0x01, 0x81, true),  // 1 - (-127) != -128
            (0x80, 0x00, false), // -128 - 0 = -128
            (0x80, 0x01, true),  // -128 - 1 != 127
        ])
        .collect::<Vec<_>>();

    for (lhs, rhs, expected_result) in test_data {
        let x: SymbolicBitVec = lhs.into();
        let y: SymbolicBitVec = rhs.into();
        let (_, borrows) = x.subtraction_with_borrow(y);
        assert_eq!(
            borrows,
            SymbolicBit::Literal(expected_result),
            "expected {lhs:#02x} - {rhs:#02x} borrow to be {expected_result}"
        );
    }
}

#[test]
fn left_shift() {
    for n in u8::MIN..u8::MAX {
        let x: SymbolicBitVec = 1u8.into();
        let result: u8 = (x << n as usize)
            .try_into()
            .expect("failed byte conversion");

        // Rust panics if shifting more than byte size
        let expected_result = if n < 8 { 1u8 << n } else { 0 };
        assert_eq!(result, expected_result);
    }
}

#[test]
fn symbolic_concrete_conversions() {
    for n in 0..u8::MAX {
        let symbolic_byte: SymbolicBitVec = n.into();
        let concrete_byte: u8 = symbolic_byte
            .try_into()
            .expect("failed to convert back into byte");
        assert_eq!(concrete_byte, n);
    }
}

#[test]
fn sign_extension() {
    let x: SymbolicBitVec = u8::MAX.into();
    let sext: u16 = x
        .sign_extend(8)
        .try_into()
        .expect("failed concrete conversion");
    assert_eq!(sext, 0xFFFFu16);
}

#[test]
fn zero_extension() {
    let x: SymbolicBitVec = u8::MAX.into();
    let zext: u16 = x
        .zero_extend(8)
        .try_into()
        .expect("failed concrete conversion");
    assert_eq!(zext, 0x00FFu16);
}

#[test]
fn concatenation() {
    let x: SymbolicBitVec = 0xADu8.into();
    let y: SymbolicBitVec = 0xDEu8.into();
    let concat: u16 = x.concat(y).try_into().expect("failed type conversion");
    assert_eq!(concat, 0xDEADu16);
}

#[test]
fn truncation_lsb() {
    let x: SymbolicBitVec = 0xDEADu16.into();
    let truncated: u8 = x
        .truncate_lsb(8)
        .try_into()
        .expect("failed type converison");
    assert_eq!(truncated, 0xDEu8);
}

#[test]
fn truncation_msb() {
    let x: SymbolicBitVec = 0xDEADu16.into();
    let truncated: u8 = x
        .truncate_msb(8)
        .try_into()
        .expect("failed type converison");
    assert_eq!(truncated, 0xADu8);
}

#[test]
fn split_into_bytes() {
    let x: SymbolicBitVec = 0xDEADu16.into();
    let split = x.into_parts(8);
    assert_eq!(split.len(), 2);

    let byte: u8 = (&split[0]).try_into().expect("failed type conversion");
    assert_eq!(byte, 0xADu8);

    let byte: u8 = (&split[1]).try_into().expect("failed type conversion");
    assert_eq!(byte, 0xDEu8);
}

#[test]
fn one_bit_equality() {
    let x = SymbolicBitVec::constant(0, 1);
    let y = SymbolicBitVec::constant(0, 1);
    let eq = x.equals(y);
    assert_eq!(eq, SymbolicBit::Literal(true));

    let x = SymbolicBitVec::constant(0, 1);
    let y = SymbolicBitVec::constant(1, 1);
    let eq = x.equals(y);
    assert_eq!(eq, SymbolicBit::Literal(false));
}

#[test]
fn equals() {
    let x: SymbolicBitVec = 0xFEEDF00Du32.into();
    let y: SymbolicBitVec = 0xDEEDF00Du32.into();
    let eq = x.clone().equals(x.clone());
    let neq = x.equals(y);
    assert_eq!(eq, SymbolicBit::Literal(true));
    assert_eq!(neq, SymbolicBit::Literal(false));
}

#[test]
fn less_than() {
    for x in 0..16u8 {
        for y in 0..16u8 {
            let sym_x = SymbolicBitVec::constant(x.into(), 4);
            let sym_y = SymbolicBitVec::constant(y.into(), 4);
            let result = sym_x.less_than(sym_y);
            assert_eq!(result, SymbolicBit::Literal(x < y), "failed {x} < {y}");
        }
    }
}

#[test]
fn greater_than() {
    for x in 0..16u8 {
        for y in 0..16u8 {
            let sym_x = SymbolicBitVec::constant(x.into(), 4);
            let sym_y = SymbolicBitVec::constant(y.into(), 4);
            let result = sym_x.greater_than(sym_y);
            assert_eq!(result, SymbolicBit::Literal(x > y), "failed {x} > {y}");
        }
    }
}

#[test]
fn signed_less_than() {
    let neg_one: SymbolicBitVec = 0xFFu8.into();
    let neg_two: SymbolicBitVec = 0xFEu8.into();
    let pos_one: SymbolicBitVec = 0x01u8.into();
    let pos_two: SymbolicBitVec = 0x02u8.into();

    assert_eq!(
        neg_two.clone().signed_less_than(neg_one.clone()),
        SymbolicBit::Literal(true),
        "Expect -2 < -1 (signed comparison)",
    );
    assert_eq!(
        neg_one.clone().signed_less_than(pos_one.clone()),
        SymbolicBit::Literal(true),
        "Expect -1 < 1 (signed comparison)",
    );
    assert_eq!(
        pos_one.clone().signed_less_than(pos_two.clone()),
        SymbolicBit::Literal(true),
        "Expect 1 < 2 (signed comparison)",
    );
    assert_eq!(
        pos_one.clone().signed_less_than(pos_one.clone()),
        SymbolicBit::Literal(false),
        "Expect 1 < 1 to be false (signed comparison)",
    );
    assert_eq!(
        neg_one.clone().signed_less_than(neg_one.clone()),
        SymbolicBit::Literal(false),
        "Expect -1 < -1 to be false (signed comparison)",
    );
}

#[test]
fn signed_greater_than() {
    let neg_one: SymbolicBitVec = 0xFFu8.into();
    let neg_two: SymbolicBitVec = 0xFEu8.into();
    let pos_one: SymbolicBitVec = 0x01u8.into();
    let pos_two: SymbolicBitVec = 0x02u8.into();

    assert_eq!(
        pos_two.clone().signed_greater_than(pos_one.clone()),
        SymbolicBit::Literal(true),
        "Expect 2 > 1 (signed comparison)",
    );
    assert_eq!(
        pos_one.clone().signed_greater_than(neg_one.clone()),
        SymbolicBit::Literal(true),
        "Expect 1 > -1 (signed comparison)",
    );
    assert_eq!(
        neg_one.clone().signed_greater_than(neg_two.clone()),
        SymbolicBit::Literal(true),
        "Expect -1 > -2 (signed comparison)",
    );
    assert_eq!(
        pos_one.clone().signed_greater_than(pos_one.clone()),
        SymbolicBit::Literal(false),
        "Expect 1 > 1 to be false (signed comparison)",
    );
    assert_eq!(
        neg_one.clone().signed_greater_than(neg_one.clone()),
        SymbolicBit::Literal(false),
        "Expect -1 > -1 to be false (signed comparison)",
    );
}

#[test]
fn signed_less_than_eq() {
    let neg_one: SymbolicBitVec = 0xFFu8.into();
    let neg_two: SymbolicBitVec = 0xFEu8.into();
    let pos_one: SymbolicBitVec = 0x01u8.into();
    let pos_two: SymbolicBitVec = 0x02u8.into();

    assert_eq!(
        neg_two.clone().signed_less_than_eq(neg_one.clone()),
        SymbolicBit::Literal(true),
        "Expect -2 < -1 (signed comparison)",
    );
    assert_eq!(
        neg_one.clone().signed_less_than_eq(pos_one.clone()),
        SymbolicBit::Literal(true),
        "Expect -1 < 1 (signed comparison)",
    );
    assert_eq!(
        pos_one.clone().signed_less_than_eq(pos_two.clone()),
        SymbolicBit::Literal(true),
        "Expect 1 < 2 (signed comparison)",
    );
    assert_eq!(
        pos_one.clone().signed_less_than_eq(pos_one.clone()),
        SymbolicBit::Literal(true),
        "Expect 1 < 1 (signed comparison)",
    );
    assert_eq!(
        neg_one.clone().signed_less_than_eq(neg_one.clone()),
        SymbolicBit::Literal(true),
        "Expect -1 < -1 (signed comparison)",
    );
}

#[test]
fn signed_greater_than_eq() {
    let neg_one: SymbolicBitVec = 0xFFu8.into();
    let neg_two: SymbolicBitVec = 0xFEu8.into();
    let pos_one: SymbolicBitVec = 0x01u8.into();
    let pos_two: SymbolicBitVec = 0x02u8.into();

    assert_eq!(
        pos_two.clone().signed_greater_than_eq(pos_one.clone()),
        SymbolicBit::Literal(true),
        "Expect 2 >= 1 (signed comparison)",
    );
    assert_eq!(
        pos_one.clone().signed_greater_than_eq(neg_one.clone()),
        SymbolicBit::Literal(true),
        "Expect 1 >= -1 (signed comparison)",
    );
    assert_eq!(
        neg_one.clone().signed_greater_than_eq(neg_two.clone()),
        SymbolicBit::Literal(true),
        "Expect -1 >= -2 (signed comparison)",
    );
    assert_eq!(
        pos_one.clone().signed_greater_than_eq(pos_one.clone()),
        SymbolicBit::Literal(true),
        "Expect 1 >= 1 (signed comparison)",
    );
    assert_eq!(
        neg_one.clone().signed_greater_than_eq(neg_one.clone()),
        SymbolicBit::Literal(true),
        "Expect -1 >= -1 (signed comparison)",
    );
}

#[test]
fn popcount() {
    for n in 0..=8 {
        let value = SymbolicBitVec::constant((1 << n) - 1, 8);
        let popcount = value.popcount();
        let popcount: u8 = popcount.try_into().expect("failed converison");
        assert_eq!(popcount, n);
    }
}

#[test]
fn shift_left() {
    for n in 0..=8 {
        let value = SymbolicBitVec::constant(0b0000_0001, 8);
        let shift_amount = SymbolicBitVec::constant(n, 8);
        let expected = if n < 8 { 1 << n } else { 0 };
        let result = value << shift_amount;
        assert_eq!(result.len(), 8);

        let result: u8 = result.try_into().expect("failed conversion");
        assert_eq!(result, expected, "failed 1 << {n}");
    }
}

#[test]
fn shift_right() {
    for n in 0..=8 {
        let value = SymbolicBitVec::constant(0b1000_0000, 8);
        let shift_amount = SymbolicBitVec::constant(n, 8);
        let expected = if n < 8 { 0x80 >> n } else { 0 };
        let result: u8 = (value >> shift_amount)
            .try_into()
            .expect("failed conversion");
        assert_eq!(result, expected, "failed 0x80 >> {n}");
    }
}

#[test]
fn signed_shift_right_negative() {
    for n in 0..=8 {
        let value = SymbolicBitVec::constant(0b1000_0000, 8);
        let shift_amount = SymbolicBitVec::constant(n, 8);
        let expected = if n < 8 { (-128 as i8 >> n) as u8 } else { 0xFF };
        let result: u8 = value
            .signed_shift_right(shift_amount)
            .try_into()
            .expect("failed conversion");
        assert_eq!(result, expected, "failed signed shift 0x80 >> {n}");
    }
}

#[test]
fn signed_shift_right_positive() {
    for n in 0..=8 {
        let value = SymbolicBitVec::constant(0b0111_1111, 8);
        let shift_amount = SymbolicBitVec::constant(n, 8);
        let expected = if n < 8 { 0x7F >> n } else { 0 };
        let result: u8 = value
            .signed_shift_right(shift_amount)
            .try_into()
            .expect("failed conversion");
        assert_eq!(result, expected, "failed signed shift 0x7F >> {n}");
    }
}

#[test]
fn multiply() {
    let output_bits = 8;
    for n in 0..16u8 {
        let value = SymbolicBitVec::constant(n.into(), 4);
        let square = value.clone() * value;
        assert_eq!(
            square.len(),
            output_bits,
            "expected {output_bits} bits in result"
        );

        let square: u8 = square.try_into().expect("failed conversion");
        assert_eq!(square, n * n, "failed {n} * {n}");
    }
}

#[test]
fn signed_multiply() {
    let output_bits = 8;
    for n in 0..16u8 {
        let value = SymbolicBitVec::constant(n.into(), 4);
        let value = value.sign_extend(4);
        let n: u8 = value.clone().try_into().expect("failed conversion");
        let square = value.clone().multiply(value, output_bits);
        assert_eq!(
            square.len(),
            output_bits,
            "expected {output_bits} bits in result"
        );

        let square: u8 = square.try_into().expect("failed conversion");

        let n = n as i8;
        assert_eq!(square as i8, n * n, "failed {n} * {n}");
    }
}

#[test]
fn single_bit_multiply() {
    let output_bits = 1;
    for n in 0..16u8 {
        let expected = (n * n) & 1;
        let value = SymbolicBitVec::constant(n.into(), 4);
        let value = value.sign_extend(4);
        let n: u8 = value.clone().try_into().expect("failed conversion");
        let result = value.clone().multiply(value, output_bits);
        assert_eq!(
            result.len(),
            output_bits,
            "expected {output_bits} bits in result"
        );

        let result: u8 = result.try_into().expect("failed conversion");
        assert_eq!(result, expected, "failed to select lsb of {n} * {n}");
    }
}

#[test]
fn unsigned_divide() {
    for dividend in 0..16u8 {
        for divisor in 1..16u8 {
            let sym_dividend = SymbolicBitVec::constant(dividend.into(), 4);
            let sym_divisor = SymbolicBitVec::constant(divisor.into(), 4);
            let (quotient, remainder) = sym_divisor.unsigned_divide(sym_dividend);

            let quotient: u8 = quotient.try_into().expect("failed quotient conversion");
            let remainder: u8 = remainder.try_into().expect("failed remainder conversion");

            let expected_quotient = dividend / divisor;
            let expected_remainder = dividend % divisor;
            assert_eq!(
                quotient, expected_quotient,
                "invalid quotient for {dividend} / {divisor}"
            );
            assert_eq!(
                remainder, expected_remainder,
                "invalid remainder for {dividend} / {divisor}"
            );
        }
    }
}

#[test]
fn signed_divide() {
    for dividend in 0..16u8 {
        for divisor in 1..16u8 {
            let sym_dividend = SymbolicBitVec::constant(dividend.into(), 4).sign_extend(4);
            let sym_divisor = SymbolicBitVec::constant(divisor.into(), 4).sign_extend(4);
            let dividend: u8 = sym_dividend.clone().try_into().unwrap();
            let dividend = dividend as i8;
            let divisor: u8 = sym_divisor.clone().try_into().unwrap();
            let divisor = divisor as i8;

            let (quotient, remainder) = sym_divisor.signed_divide(sym_dividend);

            let quotient: u8 = quotient.try_into().expect("failed quotient conversion");
            let quotient = quotient as i8;
            let remainder: u8 = remainder.try_into().expect("failed remainder conversion");
            let remainder = remainder as i8;

            let expected_quotient = dividend / divisor;
            let expected_remainder = dividend % divisor;
            assert_eq!(
                quotient, expected_quotient,
                "invalid quotient for {dividend} / {divisor}"
            );
            assert_eq!(
                remainder, expected_remainder,
                "invalid remainder for {dividend} / {divisor}"
            );
        }
    }
}
