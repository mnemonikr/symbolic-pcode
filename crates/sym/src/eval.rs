use std::collections::{BTreeMap, BTreeSet};

use crate::{SymbolicBit, SymbolicBitVec};

/// A simple evaluator that evaluates a [SymbolicBit] given a [VariableAssignments]. Portions of the
/// evaluation may be cached since the variable assignments are fixed for a given evaluator.
#[derive(Clone, Debug, Default)]
pub struct Evaluator {
    assignments: VariableAssignments,
}

#[derive(Clone, Default, PartialEq, Eq, Debug)]
pub struct Evaluation {
    /// The evaluation response. This may not be populated if there was a symbolic bit without a
    /// known concrete value
    pub response: Option<bool>,

    /// Variables that were used and their assigned value
    pub used_variables: BTreeMap<usize, bool>,

    /// Variables that were used in the evaluation but did not have an assignment
    pub unassigned_variables: BTreeSet<usize>,
}

impl Evaluator {
    /// Create a new instance using the given [VariableAssignments]. The assignments are fixed for
    /// the lifetime of this evaluator.
    pub fn new(assignments: impl Into<VariableAssignments>) -> Self {
        Self {
            assignments: assignments.into(),
        }
    }

    pub fn evaluate(&self, bit: &SymbolicBit) -> Evaluation {
        match bit {
            SymbolicBit::Literal(x) => Evaluation {
                response: Some(*x),
                ..Default::default()
            },
            SymbolicBit::Variable(id) => {
                let response = self.assignments.get(*id);
                if let Some(x) = response {
                    Evaluation {
                        response,
                        used_variables: std::iter::once((*id, x)).collect(),
                        ..Default::default()
                    }
                } else {
                    Evaluation {
                        response,
                        unassigned_variables: std::iter::once(*id).collect(),
                        ..Default::default()
                    }
                }
            }
            SymbolicBit::Not(bit) => {
                let mut evaluation = self.evaluate(bit);
                evaluation.response = evaluation.response.map(|x| !x);
                evaluation
            }
            SymbolicBit::And(lhs, rhs) => {
                let mut lhs = self.evaluate(lhs);
                if let Some(lhs_response) = lhs.response {
                    if !lhs_response {
                        // LHS is false, do not need to evaluate RHS
                        lhs
                    } else {
                        // LHS is true, evaluate RHS
                        let mut rhs = self.evaluate(rhs);
                        if let Some(rhs_response) = rhs.response {
                            // RHS is concrete value
                            lhs.response = Some(lhs_response && rhs_response);
                            lhs.used_variables.append(&mut rhs.used_variables);
                            lhs
                        } else {
                            // RHS is symbolic
                            rhs
                        }
                    }
                } else {
                    // LHS is not concrete. Should still evaluate RHS in case it is false
                    let rhs = self.evaluate(rhs);
                    if matches!(rhs.response, Some(false)) {
                        // RHS is false, ignore LHS
                        rhs
                    } else {
                        // RHS is either true or symbolic but LHS is already symbolic
                        lhs
                    }
                }
            }
        }
    }
}

impl From<VariableAssignments> for Evaluator {
    fn from(value: VariableAssignments) -> Self {
        Self::new(value)
    }
}

/// Mapping [SymbolicBit::Variable] identifiers to [SymbolicBit::Literal] values.
#[derive(Clone, Debug, Default)]
pub struct VariableAssignments {
    assignments: BTreeMap<usize, bool>,
}

impl VariableAssignments {
    /// Create variable assignments given a [SymbolicBitVec] containing only [SymbolicBit::Variable]
    /// variants and another [SymbolicBitVec] containing only [SymbolicBit::Literal] variants.
    /// Incorrect variants in either will be ignored.
    ///
    /// Both bit vectors should have the same length. In the event they are not equal, the shorter
    /// length will be used for both.
    pub fn from_bitvecs(variables: &SymbolicBitVec, literals: &SymbolicBitVec) -> Self {
        let iter =
            std::iter::zip(variables.iter(), literals.iter()).filter_map(|(variable, literal)| {
                if let SymbolicBit::Variable(variable) = variable
                    && let SymbolicBit::Literal(literal) = literal
                {
                    return Some((*variable, *literal));
                }

                None
            });
        Self::from_iter(iter)
    }

    pub fn get(&self, variable_id: usize) -> Option<bool> {
        self.assignments.get(&variable_id).copied()
    }
}

impl<I: IntoIterator<Item = (usize, bool)>> From<I> for VariableAssignments {
    fn from(iter: I) -> Self {
        iter.into_iter().collect()
    }
}

impl FromIterator<(usize, bool)> for VariableAssignments {
    fn from_iter<T: IntoIterator<Item = (usize, bool)>>(iter: T) -> Self {
        Self {
            assignments: iter.into_iter().collect(),
        }
    }
}
