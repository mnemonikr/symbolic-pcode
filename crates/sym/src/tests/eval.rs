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
    let mut eval = Evaluator::new(VariableAssignments::from_iter([(0, true), (1, false)]));
    assert!(eval.evaluate(&x));
    assert!(!eval.evaluate(&y));
}

#[test]
fn evaluate_expression() {
    let x = SymbolicBit::Variable(0);
    let y = SymbolicBit::Variable(1);
    let mut eval = Evaluator::new(VariableAssignments::from_iter([(0, true), (1, false)]));
    let z = x ^ y;
    assert!(eval.evaluate(&z));
}

#[test]
fn evaluator_cache() {
    let x = SymbolicBit::Variable(0);
    let y = SymbolicBit::Variable(1);
    let mut eval = Evaluator::new(VariableAssignments::from_iter([(0, true), (1, false)]));
    let z = x.clone() & !y.clone();
    let w = !y & x;

    // This will not trigger the cache because individual AND gates are not shared.
    // assert!(eval.evaluate(&z));
    // assert!(eval.evaluate(&w));

    // However this WILL trigger the cache because of the reuse of the same AND gate
    // The cache is used even though the operands are swapped.
    assert!(eval.evaluate(&(z.clone() & w.clone())));
    assert!(eval.evaluate(&(w & z)));
}

#[test]
fn var_assignments_from_bitvecs() {
    let variables = SymbolicBitVec::from_iter([SymbolicBit::Variable(0), SymbolicBit::Variable(1)]);
    let literals = SymbolicBitVec::from_iter([TRUE, FALSE]);
    let assignments = VariableAssignments::from_bitvecs(&variables, &literals);
    assert!(assignments.get(0).unwrap());
    assert!(!assignments.get(1).unwrap());
}

#[test]
fn var_assignments_from_swapped_bitvecs() {
    let variables = SymbolicBitVec::from_iter([SymbolicBit::Variable(0), SymbolicBit::Variable(1)]);
    let literals = SymbolicBitVec::from_iter([TRUE, FALSE]);

    // Swapped the literals and the variables
    let assignments = VariableAssignments::from_bitvecs(&literals, &variables);

    assert!(assignments.get(0).is_none());
    assert!(assignments.get(1).is_none());
}

#[test]
fn var_assignments_from_bitvecs_without_literals() {
    let variables = SymbolicBitVec::from_iter([SymbolicBit::Variable(0), SymbolicBit::Variable(1)]);

    // Swapped the literals and the variables
    let assignments = VariableAssignments::from_bitvecs(&variables, &variables);

    assert!(assignments.get(0).is_none());
    assert!(assignments.get(1).is_none());
}

#[test]
fn evaluator_from_assignments() {
    let variables = SymbolicBitVec::from_iter([SymbolicBit::Variable(0), SymbolicBit::Variable(1)]);
    let literals = SymbolicBitVec::from_iter([TRUE, FALSE]);
    let assignments = VariableAssignments::from_bitvecs(&variables, &literals);
    let mut evaluator = Evaluator::from(assignments);
    assert!(evaluator.evaluate(&SymbolicBit::Variable(0)));
    assert!(!evaluator.evaluate(&SymbolicBit::Variable(1)));
}
