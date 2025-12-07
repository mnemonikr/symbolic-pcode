//! Convert circuit of [AigerCircuit]s into [AIGER format](https://github.com/arminbiere/aiger/blob/master/FORMAT).

use std::collections::{BTreeMap, HashMap};

use crate::circuit::{AigerCircuit, AsAigerCircuit};
use crate::index::Indexes;
use crate::model::{AigerGate, AigerLiteral};

/// The core object for generating AIGER formatted data. For inspecting the AST see [Aiger::inputs],
/// [Aiger::gates], and [Aiger::outputs].
///
/// ## Example
///
/// ```
/// # use aiger_circuit::Aiger;
/// # use aiger_circuit::circuit::SimpleCircuit;
/// # let output = SimpleCircuit::lit(false);
/// // Assume output variable is the output bit of a complex circuit
/// let aiger = Aiger::with_output(&output);
/// for input in aiger.inputs() {
///     // Iterates over all variables used in the circuit
/// }
///
/// for gate in aiger.gates() {
///     // Iterates over all unique gates in the circuit
/// }
///
/// for output in aiger.outputs() {
///     // Iterate over all outputs
/// }
///
/// // Serialize data into AIGER binary format
/// let aiger_binary_data = aiger.serialize_binary();
/// ```
pub struct Aiger {
    inputs: HashMap<AigerLiteral, usize>,
    outputs: Vec<AigerLiteral>,
    gates: Vec<AigerGate>,
}

impl Aiger {
    /// Create an Aiger object from the given output. See [Aiger::with_outputs].
    pub fn with_output<'a, T>(output: &'a T) -> Self
    where
        T: AsAigerCircuit<'a>,
    {
        let outputs = [output];
        Self::with_outputs(outputs)
    }

    /// Create an Aiger object from the given output bits. This will walk the circuit and gather the
    /// corresponding Aiger data. Identical references in [AigerCircuit::And] and [AigerCircuit::Not]
    /// are deduplicated, but otherwise no equivalence checking is performed for circuit reduction.
    pub fn with_outputs<'a, T>(outputs: impl IntoIterator<Item = &'a T>) -> Self
    where
        T: AsAigerCircuit<'a> + 'a,
    {
        let mut indexes = Indexes::new();
        let outputs = outputs.into_iter().collect::<Vec<_>>();
        for output in outputs.iter() {
            indexes.insert_indexes(*output);
        }

        let mut gates = Default::default();
        let mut aiger_outputs = Vec::with_capacity(outputs.len());

        for output in outputs {
            Self::insert_gates(output, &indexes, &mut gates);
            aiger_outputs.push(indexes.literal(output));
        }
        let gates = gates.into_values().collect::<Vec<_>>();
        Self {
            inputs: indexes.variables().collect(),
            outputs: aiger_outputs,
            gates,
        }
    }

    /// Insert all [AigerCircuit::And] gates into the tree. The tree is a mapping from the gate
    /// index to the [AigerGate] composed of [AigerLiteral]s.
    fn insert_gates<'a, T>(bit: &'a T, indexes: &Indexes, gates: &mut BTreeMap<usize, AigerGate>)
    where
        T: AsAigerCircuit<'a>,
    {
        match bit.as_circuit() {
            AigerCircuit::And(x, y) => {
                let index = indexes.index(bit);

                if gates.get(&index).is_none() {
                    gates.insert(
                        index,
                        AigerGate::new(
                            indexes.literal(bit),
                            (indexes.literal(x.value), indexes.literal(y.value)),
                        ),
                    );

                    // Only insert children if this is a new insertion. If this gate has already
                    // been inserted then its children have also been inserted as well
                    Self::insert_gates(x.value, indexes, gates);
                    Self::insert_gates(y.value, indexes, gates);
                }
            }
            AigerCircuit::Not(x) => {
                Self::insert_gates(x, indexes, gates);
            }

            // No other type can lead to an and-gate
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

    /// Iterator over all Aiger output literals. These literals correspond to the values provided in
    /// [Aiger::with_outputs].
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
    /// * And-gates are encoded in a binary format
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
