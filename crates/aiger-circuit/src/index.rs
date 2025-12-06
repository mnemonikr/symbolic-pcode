use std::collections::HashMap;

use crate::circuit::AigerCircuit;
use crate::model::{AigerLiteral, FALSE, TRUE};

/// A mapping of identifiers to indexes. This is an internal structure used for computing the
/// indexes of variables and gates.
#[derive(Default)]
pub struct Indexes {
    /// Mapping variable id to index
    variables: HashMap<usize, usize>,

    /// Mapping `AndId` to index. Note that this index is the internal and gate index. The global
    /// index can only be calculated once all variable indexes have also been stored.
    ands: HashMap<AndId, usize>,
}

impl Indexes {
    /// Create a new index
    pub fn new() -> Self {
        Default::default()
    }

    /// Insert all components of the [AigerCircuit] into the index. Components already inserted are
    /// ignored.
    pub fn insert_indexes<B>(&mut self, bit: &B)
    where
        for<'a> &'a B: Into<AigerCircuit<'a, B>>,
    {
        match bit.into() {
            AigerCircuit::Variable(id) => {
                let index = self.variables.len() + 1;
                self.variables.entry(id).or_insert(index);
            }
            AigerCircuit::And(x, y) => {
                let id = AndId::new(x, y);
                if !self.ands.contains_key(&id) {
                    self.insert_indexes(x);
                    self.insert_indexes(y);

                    // Index this gate at the end to ensure its index is greater than the index of
                    // anything contained in either the lhs or rhs
                    self.ands.insert(id, self.ands.len() + 1);
                }
            }
            AigerCircuit::Not(x) => {
                self.insert_indexes(x);
            }
            AigerCircuit::Literal(_) => (),
        }
    }

    /// Get the index for the given [AigerCircuit].
    ///
    /// # Panics
    ///
    /// Will panic if the `bit` is not indexed
    pub fn index<B>(&self, bit: &B) -> usize
    where
        for<'a> &'a B: Into<AigerCircuit<'a, B>>,
    {
        match bit.into() {
            AigerCircuit::Variable(id) => *self.variables.get(&id).unwrap(),
            AigerCircuit::And(x, y) => {
                let id = AndId::new(x, y);
                *self.ands.get(&id).unwrap() + self.num_input_literals()
            }
            AigerCircuit::Literal(_) => panic!("literal bits are not indexed"),
            AigerCircuit::Not(_) => panic!("negated bits are not indexed"),
        }
    }

    /// Get the number of input literals indexed.
    pub fn num_input_literals(&self) -> usize {
        self.variables.len()
    }

    pub fn variables(&self) -> impl Iterator<Item = (AigerLiteral, usize)> {
        self.variables
            .iter()
            .map(|(&variable_id, &index)| (AigerLiteral::new(index), variable_id))
    }

    /// Get the AIGER literal for this [AigerCircuit].
    ///
    /// # Panics
    ///
    /// Will panic if this bit is an unindexed variable or and gate, or the negation of such a bit.
    pub fn literal<B>(&self, bit: &B) -> AigerLiteral
    where
        for<'a> &'a B: Into<AigerCircuit<'a, B>>,
    {
        match bit.into() {
            AigerCircuit::Literal(false) => FALSE,
            AigerCircuit::Literal(true) => TRUE,
            AigerCircuit::Variable(_) | AigerCircuit::And(_, _) => {
                AigerLiteral::new(self.index(bit))
            }
            AigerCircuit::Not(x) => self.literal(x).negated(),
        }
    }
}

/// The identifier for an and gate
#[derive(Debug, PartialEq, Eq, Hash)]
struct AndId(usize, usize);

impl AndId {
    /// Construct a new and gate identifier from the inputs of a [AigerCircuit::And] gate. The
    /// addresses of the reference counters are used to ensure that cloned gates will have the same
    /// identifier. This does mean that gates that are equivalent but were not cloned will have
    /// different identifiers.
    pub fn new<T>(lhs: *const T, rhs: *const T) -> Self {
        let lhs = lhs as usize;
        let rhs = rhs as usize;
        if lhs < rhs {
            Self(lhs, rhs)
        } else {
            Self(rhs, lhs)
        }
    }
}
