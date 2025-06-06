use std::collections::BTreeMap;
use std::rc::Rc;

use crate::{SymbolicBit, SymbolicBitVec};

/// A simple evaluator that evaluates a [SymbolicBit] given a [VariableAssignments]. Portions of the
/// evaluation may be cached since the variable assignments are fixed for a given evaluator.
#[derive(Clone, Debug, Default)]
pub struct Evaluator {
    assignments: VariableAssignments,
    and_gates: std::collections::HashMap<(usize, usize), bool>,
}

impl Evaluator {
    /// Create a new instance using the given [VariableAssignments]. The assignments are fixed for
    /// the lifetime of this evaluator.
    pub fn new(assignments: VariableAssignments) -> Self {
        Self {
            assignments,
            and_gates: Default::default(),
        }
    }

    /// Evaluate the value of this [SymbolicBit].
    pub fn evaluate(&mut self, bit: &SymbolicBit) -> bool {
        match bit {
            SymbolicBit::Literal(x) => *x,
            SymbolicBit::Variable(id) => self
                .assignments
                .get(*id)
                .expect("variable should be defined in cache"),
            SymbolicBit::Not(bit) => !self.evaluate(bit),
            SymbolicBit::And(lhs, rhs) => {
                let lhs_addr = Rc::as_ptr(lhs).addr();
                let rhs_addr = Rc::as_ptr(rhs).addr();
                let cache_key = (
                    usize::min(lhs_addr, rhs_addr),
                    usize::max(lhs_addr, rhs_addr),
                );

                if let Some(value) = self.and_gates.get(&cache_key).copied() {
                    value
                } else {
                    let value = self.evaluate(lhs) && self.evaluate(rhs);
                    self.and_gates.insert(cache_key, value);
                    value
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
                if let SymbolicBit::Variable(variable) = variable {
                    if let SymbolicBit::Literal(literal) = literal {
                        return Some((*variable, *literal));
                    }
                }

                None
            });
        Self::from_iter(iter)
    }

    pub fn get(&self, variable_id: usize) -> Option<bool> {
        self.assignments.get(&variable_id).copied()
    }
}

impl FromIterator<(usize, bool)> for VariableAssignments {
    fn from_iter<T: IntoIterator<Item = (usize, bool)>>(iter: T) -> Self {
        Self {
            assignments: iter.into_iter().collect(),
        }
    }
}
