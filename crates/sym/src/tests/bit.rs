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
    let mut eval = Evaluator::default();
    assert!(eval.evaluate(&TRUE));
    assert!(!eval.evaluate(&FALSE));
}

#[test]
fn evaluate_variable() {
    let x = SymbolicBit::Variable(0);
    let y = SymbolicBit::Variable(1);
    let mut eval = Evaluator::from_iter([(0, true), (1, false)]);
    assert!(eval.evaluate(&x));
    assert!(!eval.evaluate(&y));
}

#[test]
fn evaluate_expression() {
    let x = SymbolicBit::Variable(0);
    let y = SymbolicBit::Variable(1);
    let mut eval = Evaluator::from_iter([(0, true), (1, false)]);
    let z = x ^ y;
    assert!(eval.evaluate(&z));
}

#[test]
fn evaluator_cache() {
    let x = SymbolicBit::Variable(0);
    let y = SymbolicBit::Variable(1);
    let mut eval = Evaluator::from_iter([(0, true), (1, true)]);
    let z = x.clone() & y.clone();
    let w = y & x;

    // This will not trigger the cache because individual AND gates are not shared.
    // assert!(eval.evaluate(&z));
    // assert!(eval.evaluate(&w));

    // However this WILL trigger the cache because of the reuse of the same AND gate
    // The cache is used even though the operands are swapped.
    assert!(eval.evaluate(&(z.clone() & w.clone())));
    assert!(eval.evaluate(&(w & z)));
}
