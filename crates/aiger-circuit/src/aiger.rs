//! Convert circuit of [AigerCircuit]s into [AIGER format](https://github.com/arminbiere/aiger/blob/master/FORMAT).

use std::collections::{BTreeMap, HashMap};

use crate::circuit::AigerCircuit;
use crate::index::Indexes;
use crate::model::{AigerGate, AigerLiteral};

/// Object capable of coverting [AigerCircuit]s into AIGER format.
pub struct Aiger {
    inputs: HashMap<AigerLiteral, usize>,
    outputs: Vec<AigerLiteral>,
    gates: Vec<AigerGate>,
}

impl Aiger {
    /// Create an Aiger object from the given output bits. This will walk the circuit and gather the
    /// corresponding Aiger data. Identical references in [AigerCircuit::And] and [AigerCircuit::Not]
    /// are deduplicated, but otherwise no equivalence checking is performed for circuit reduction.
    pub fn from_bits<B>(bits: impl IntoIterator<Item = B>) -> Self
    where
        for<'a> &'a B: Into<AigerCircuit<'a, B>>,
    {
        let mut indexes = Indexes::new();
        let bits = bits.into_iter().collect::<Vec<_>>();
        for bit in bits.iter() {
            indexes.insert_indexes(bit);
        }

        let mut gates = Default::default();
        let mut outputs = Vec::with_capacity(bits.len());

        for bit in bits {
            Self::insert_gates(&bit, &indexes, &mut gates);
            outputs.push(indexes.literal(&bit));
        }
        let gates = gates.into_values().collect::<Vec<_>>();
        Self {
            inputs: indexes.variables().collect(),
            outputs,
            gates,
        }
    }

    /// Insert all [AigerCircuit::And] gates into the tree. The tree is a mapping from the gate
    /// index to the [AigerGate] composed of [AigerLiteral]s.
    fn insert_gates<B>(bit: &B, indexes: &Indexes, gates: &mut BTreeMap<usize, AigerGate>)
    where
        for<'a> &'a B: Into<AigerCircuit<'a, B>>,
    {
        match bit.into() {
            AigerCircuit::And(x, y) => {
                let index = indexes.index(bit);

                if gates.get(&index).is_none() {
                    gates.insert(
                        index,
                        AigerGate::new(
                            indexes.literal(bit),
                            (indexes.literal(x), indexes.literal(y)),
                        ),
                    );

                    // Only insert children if this is a new insertion. If this gate has already
                    // been inserted then its children have also been inserted as well
                    Self::insert_gates(x, indexes, gates);
                    Self::insert_gates(y, indexes, gates);
                }
            }
            AigerCircuit::Not(x) => {
                Self::insert_gates(x, indexes, gates);
            }

            // No other type can lead to an and gate
            _ => (),
        }
    }

    /// Iterator over all Aiger input literals.
    pub fn inputs(&self) -> impl Iterator<Item = AigerLiteral> {
        (1..=self.inputs.len()).map(AigerLiteral::new)
    }

    /// Get the symbolic library input variable id for this literal.
    ///
    /// Returns `None` if the given literal does not map to an input variable.
    pub fn input_variable_id(&self, input: AigerLiteral) -> Option<usize> {
        self.inputs.get(&input).copied()
    }

    /// Iterator over all Aiger output literals. These literals correspond to the bits provided in
    /// [Aiger::from_bits].
    pub fn outputs(&self) -> impl Iterator<Item = AigerLiteral> + '_ {
        self.outputs.iter().copied()
    }

    /// Iterator over all Aiger gate literals.
    pub fn gates(&self) -> impl Iterator<Item = AigerGate> + '_ {
        self.gates.iter().copied()
    }

    /// Serialize the circuit into the AIGER binary format. The notable difference between the
    /// binary and ASCII formats:
    ///
    /// * Inputs are omitted
    /// * And gates are encoded in a binary format
    pub fn serialize_binary(&self) -> Vec<u8> {
        let header = format!(
            "aag {M} {I} {L} {O} {A}\n",
            M = self.inputs.len() + self.gates.len(),
            I = self.inputs.len(),
            L = 0,
            O = self.outputs.len(),
            A = self.gates.len()
        );
        let mut serialized = Vec::from(header.as_bytes());
        self.outputs()
            .for_each(|output| serialized.extend_from_slice(format!("{output}\n").as_bytes()));
        self.gates()
            .map(|gate| gate.serialize_binary())
            .for_each(|mut bytes| serialized.append(&mut bytes));

        serialized
    }
}
