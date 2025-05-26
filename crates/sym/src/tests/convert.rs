use crate::*;

#[test]
fn concretize_buf() {
    assert_eq!(0, u128::try_from(SymbolicBitBuf::<128>::default()).unwrap());
    assert_eq!(0, u64::try_from(SymbolicBitBuf::<64>::default()).unwrap());
    assert_eq!(0, u32::try_from(SymbolicBitBuf::<32>::default()).unwrap());
    assert_eq!(0, u16::try_from(SymbolicBitBuf::<16>::default()).unwrap());
    assert_eq!(0, u8::try_from(SymbolicBitBuf::<8>::default()).unwrap());
    assert_eq!(
        0,
        usize::try_from(SymbolicBitBuf::<{ 8 * std::mem::size_of::<usize>() }>::default()).unwrap()
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
        ConcretizationError::NonLiteralBit { bit_index: 0 }
    ));
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
        ConcretizationError::NonLiteralBit { bit_index: 0 }
    ));

    let err = concretize_into::<u8, 1>(array).unwrap_err();
    assert!(matches!(
        err,
        ConcretizationError::NonLiteralBit { bit_index: 0 }
    ));
}

#[test]
fn concrete_value_derives() {
    let x = ConcreteValue::new(1u8);

    // Clone
    #[expect(clippy::clone_on_copy)]
    let y = x.clone();

    // PartialEq
    assert_eq!(x, y);

    // Debug
    assert_eq!("ConcreteValue { value: 1 }", format!("{x:?}"));
}
