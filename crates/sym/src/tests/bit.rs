use crate::*;

#[test]
fn bit_equality() {
    let x = SymbolicBit::Literal(true);
    assert_eq!(x.clone().equals(x), SymbolicBit::Literal(true));

    let x = SymbolicBit::Literal(false);
    assert_eq!(x.clone().equals(x), SymbolicBit::Literal(true));
}

#[test]
fn double_negation() {
    let x = SymbolicBit::Variable(0);
    assert_eq!(!!x.clone(), x);
}

#[test]
fn conjunction_with_false() {
    let x = SymbolicBit::Variable(0);
    assert_eq!(
        x.clone() & SymbolicBit::Literal(false),
        SymbolicBit::Literal(false),
    );
    assert_eq!(
        SymbolicBit::Literal(false) & x.clone(),
        SymbolicBit::Literal(false),
    );
}

#[test]
fn conjunction_with_true() {
    let x = SymbolicBit::Variable(0);
    assert_eq!(x.clone() & SymbolicBit::Literal(true), x);
    assert_eq!(SymbolicBit::Literal(true) & x.clone(), x);
}

#[test]
fn conjunction_with_negated_self() {
    let x = SymbolicBit::Variable(0);
    assert_eq!(x.clone() & !x.clone(), SymbolicBit::Literal(false));
    assert_eq!(!x.clone() & x.clone(), SymbolicBit::Literal(false));
}

#[test]
fn disjunction_with_false() {
    let x = SymbolicBit::Variable(0);
    assert_eq!(x.clone() | SymbolicBit::Literal(false), x);
    assert_eq!(SymbolicBit::Literal(false) | x.clone(), x);
}

#[test]
fn disjunction_with_true() {
    let x = SymbolicBit::Variable(0);
    assert_eq!(
        x.clone() | SymbolicBit::Literal(true),
        SymbolicBit::Literal(true)
    );
    assert_eq!(
        SymbolicBit::Literal(true) | x.clone(),
        SymbolicBit::Literal(true),
    );
}

#[test]
fn exclusive_or_with_same_variable() {
    let x = SymbolicBit::Variable(0);
    assert_eq!(x.clone() ^ x.clone(), SymbolicBit::Literal(false));
}

#[test]
fn exclusive_or_with_complex_self() {
    let x = complex_bit();
    assert_eq!(x.clone() ^ x.clone(), SymbolicBit::Literal(false));
}

#[test]
fn exclusive_or_with_zero() {
    let x = SymbolicBit::Variable(0);
    assert_eq!(x.clone() ^ SymbolicBit::Literal(false), x);
    assert_eq!(SymbolicBit::Literal(false) ^ x.clone(), x);
}

fn complex_bit() -> SymbolicBit {
    let mut bit = SymbolicBit::Literal(true);
    for i in 0..5 {
        bit = SymbolicBit::Variable(i) & bit;
    }
    bit
}
