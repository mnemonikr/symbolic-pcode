use crate::*;

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
    let mut eval = Evaluator::new(VariableMapping::from_iter([(0, true), (1, false)]));
    assert!(eval.evaluate(&x));
    assert!(!eval.evaluate(&y));
}

#[test]
fn evaluate_expression() {
    let x = SymbolicBit::Variable(0);
    let y = SymbolicBit::Variable(1);
    let mut eval = Evaluator::new(VariableMapping::from_iter([(0, true), (1, false)]));
    let z = x ^ y;
    assert!(eval.evaluate(&z));
}

#[test]
fn evaluator_cache() {
    let x = SymbolicBit::Variable(0);
    let y = SymbolicBit::Variable(1);
    let mut eval = Evaluator::new(VariableMapping::from_iter([(0, true), (1, true)]));
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
