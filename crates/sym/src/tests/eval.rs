use std::rc::Rc;

use crate::*;

#[test]
fn evaluate_literal() {
    let eval = Evaluator::default();
    assert!(eval.evaluate(&TRUE).response.unwrap());
    assert!(!eval.evaluate(&FALSE).response.unwrap());
}

#[test]
fn evaluate_variable() {
    let x = SymbolicBit::Variable(0);
    let y = SymbolicBit::Variable(1);
    let evaluator = Evaluator::new(VariableAssignments::from_iter([(0, true), (1, false)]));
    let x_eval = evaluator.evaluate(&x);

    assert!(x_eval.response.unwrap(), "evaluation should be true");
    assert!(x_eval.used_variables.get(&0).unwrap(), "x should be true");
    assert!(
        !x_eval.used_variables.contains_key(&1),
        "y should not be used"
    );

    let y_eval = evaluator.evaluate(&y);
    assert!(!y_eval.response.unwrap(), "evaluation should be false");
    assert!(!y_eval.used_variables.get(&1).unwrap(), "y should be false");
    assert!(
        !y_eval.used_variables.contains_key(&0),
        "x should not be used"
    );
}

#[test]
fn evaluate_expression() {
    let x = SymbolicBit::Variable(0);
    let y = SymbolicBit::Variable(1);
    let evaluator = Evaluator::new(VariableAssignments::from_iter([(0, true), (1, false)]));
    let z = x ^ y;
    let evaluation = evaluator.evaluate(&z);

    assert!(
        evaluation.response.unwrap(),
        "true xor false should be true"
    );
    assert!(
        evaluation.used_variables.get(&0).unwrap(),
        "x should be true"
    );
    assert!(
        !evaluation.used_variables.get(&1).unwrap(),
        "y should be false"
    );
}

#[test]
fn evaluate_symbolic_expression() {
    let x = SymbolicBit::Variable(0);
    let y = SymbolicBit::Variable(1);
    let evaluator = Evaluator::new(VariableAssignments::from_iter(std::iter::empty()));
    let z = x ^ y;
    let evaluation = evaluator.evaluate(&z);

    assert!(
        evaluation.response.is_none(),
        "evaluation should not have a concrete value"
    );

    // One of x or y may not be considered unassigned due to short-circuiting of the evaluation
    assert!(
        evaluation.unassigned_variables.contains(&0)
            || evaluation.unassigned_variables.contains(&1),
        "either x or y (or both) should be unassigned"
    );
}

#[test]
fn evaluate_false_and_symbolic() {
    let x = SymbolicBit::Variable(0);
    let y = SymbolicBit::Variable(1);
    let z = SymbolicBit::And(Rc::new(x), Rc::new(y));
    let evaluator = Evaluator::new(VariableAssignments::from_iter([(0, false)]));
    let evaluation = evaluator.evaluate(&z);

    assert!(!evaluation.response.unwrap(), "evaluation should be false");
    assert!(
        evaluation.unassigned_variables.is_empty(),
        "all evaluated variables should be assigned"
    );
}

#[test]
fn evaluate_true_and_symbolic() {
    let x = SymbolicBit::Variable(0);
    let y = SymbolicBit::Variable(1);
    let z = SymbolicBit::And(Rc::new(x), Rc::new(y));
    let evaluator = Evaluator::new(VariableAssignments::from_iter([(0, true)]));
    let evaluation = evaluator.evaluate(&z);

    assert!(
        evaluation.response.is_none(),
        "evaluation should be symbolic"
    );
    assert!(
        evaluation.unassigned_variables.contains(&1),
        "y should be unassigned"
    );
}

#[test]
fn evaluate_symbolic_and_false() {
    let x = SymbolicBit::Variable(0);
    let y = SymbolicBit::Variable(1);
    let z = SymbolicBit::And(Rc::new(x), Rc::new(y));
    let evaluator = Evaluator::new(VariableAssignments::from_iter([(1, false)]));
    let evaluation = evaluator.evaluate(&z);

    assert!(!evaluation.response.unwrap(), "evaluation should be false");
    assert!(
        evaluation.unassigned_variables.is_empty(),
        "all evaluated variables should be assigned"
    );
}

#[test]
fn evaluate_symbolic_and_true() {
    let x = SymbolicBit::Variable(0);
    let y = SymbolicBit::Variable(1);
    let z = SymbolicBit::And(Rc::new(x), Rc::new(y));
    let evaluator = Evaluator::new(VariableAssignments::from_iter([(1, true)]));
    let evaluation = evaluator.evaluate(&z);

    assert!(
        evaluation.response.is_none(),
        "evaluation should be symbolic"
    );
    assert!(
        evaluation.unassigned_variables.contains(&0),
        "x should be unassigned"
    );
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
    let evaluator = Evaluator::from(assignments);
    assert!(
        evaluator
            .evaluate(&SymbolicBit::Variable(0))
            .response
            .unwrap()
    );
    assert!(
        !evaluator
            .evaluate(&SymbolicBit::Variable(1))
            .response
            .unwrap()
    );
}
