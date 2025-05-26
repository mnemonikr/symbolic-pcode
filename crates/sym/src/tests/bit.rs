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
fn exclusive_or_with_self() {
    let x = SymbolicBit::Variable(0);
    assert_eq!(x.clone() ^ x.clone(), SymbolicBit::Literal(false));
}

#[test]
fn exclusive_or_with_zero() {
    let x = SymbolicBit::Variable(0);
    assert_eq!(x.clone() ^ SymbolicBit::Literal(false), x);
    assert_eq!(SymbolicBit::Literal(false) ^ x.clone(), x);
}

#[test]
fn evaluate_literal() {
    assert!(TRUE.evaluate(|_id| false));
    assert!(!FALSE.evaluate(|_id| true));
}

#[test]
fn evaluate_variable() {
    let x = SymbolicBit::Variable(0);
    let y = SymbolicBit::Variable(1);
    assert!(x.evaluate(|id| id == 0));
    assert!(!y.evaluate(|id| id == 0));
}

#[test]
fn evaluate_expression() {
    let x = SymbolicBit::Variable(0);
    let y = SymbolicBit::Variable(1);
    let z = x & y;
    assert!(z.evaluate(|_id| true));
}

#[test]
fn evaluate_variable_lookup() {
    use std::collections::BTreeMap;
    let mut concrete_values = BTreeMap::default();
    concrete_values.insert(0, true);
    concrete_values.insert(1, false);
    let x = SymbolicBit::Variable(0);
    let y = SymbolicBit::Variable(1);
    let z = x ^ y;
    assert!(z.evaluate(|id| concrete_values.get(&id).copied().unwrap()));
}
