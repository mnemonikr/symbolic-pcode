use crate::{FALSE, SymbolicBitBuf, TRUE};

#[test]
fn ops_not() {
    let x: SymbolicBitBuf<1> = [FALSE].into();
    assert_eq!(u8::try_from(!x.clone()).unwrap(), 0x1);
    assert_eq!(u8::try_from(!!x).unwrap(), 0x0);
}

#[test]
fn ops_bitor() {
    let x: SymbolicBitBuf<4> = [FALSE, FALSE, TRUE, TRUE].into();
    let y: SymbolicBitBuf<4> = [FALSE, TRUE, FALSE, TRUE].into();
    assert_eq!(u8::try_from(x | y).unwrap(), 0x0E);
}

#[test]
fn ops_bitand() {
    let x: SymbolicBitBuf<4> = [FALSE, FALSE, TRUE, TRUE].into();
    let y: SymbolicBitBuf<4> = [FALSE, TRUE, FALSE, TRUE].into();
    assert_eq!(u8::try_from(x & y).unwrap(), 0x08);
}

#[test]
fn ops_shl_usize() {
    let x: SymbolicBitBuf<4> = [TRUE, FALSE, FALSE, FALSE].into();
    let result = u8::try_from(x << 1).unwrap();
    assert_eq!(result, 0x2);
}

#[test]
fn ops_shl_sym() {
    let x: SymbolicBitBuf<4> = [TRUE, FALSE, FALSE, FALSE].into();
    let result = u8::try_from(x.clone() << x).unwrap();
    assert_eq!(result, 0x2);
}

#[test]
fn ops_shr_usize() {
    let x: SymbolicBitBuf<4> = [FALSE, TRUE, FALSE, FALSE].into();
    let result = u8::try_from(x >> 1).unwrap();
    assert_eq!(result, 0x1);
}

#[test]
fn ops_shr_sym() {
    let x: SymbolicBitBuf<4> = [FALSE, TRUE, FALSE, FALSE].into();
    let y: SymbolicBitBuf<4> = [TRUE, FALSE, FALSE, FALSE].into();
    let result = u8::try_from(x >> y).unwrap();
    assert_eq!(result, 0x1);
}
