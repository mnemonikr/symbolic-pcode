//! Convert circuit of [SymbolicBit]s into [AIGER format](https://github.com/arminbiere/aiger/blob/master/FORMAT).

use std::{
    collections::{BTreeMap, HashMap},
    rc::Rc,
};

use crate::SymbolicBit;

/// False AIGER literal
pub const FALSE_LITERAL: AigerLiteral = AigerLiteral::new(0);

/// True AIGER literal
pub const TRUE_LITERAL: AigerLiteral = FALSE_LITERAL.negated();

/// Object capable of coverting [SymbolicBit]s into AIGER format.
pub struct Aiger {
    num_inputs: usize,
    outputs: Vec<AigerLiteral>,
    gates: Vec<AigerGate>,
}

impl Aiger {
    /// Create an Aiger object from the given output bits
    pub fn from_bits(bits: impl IntoIterator<Item = SymbolicBit>) -> Self {
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
            num_inputs: indexes.num_input_literals(),
            outputs,
            gates,
        }
    }

    /// Insert all [SymbolicBit::And] gates into the tree. The tree is a mapping from the gate
    /// index to the [AigerGate] composed of [AigerLiteral]s.
    fn insert_gates(bit: &SymbolicBit, indexes: &Indexes, gates: &mut BTreeMap<usize, AigerGate>) {
        match bit {
            SymbolicBit::And(x, y) => {
                let index = indexes.index(bit);

                if gates.get(&index).is_none() {
                    gates.insert(
                        index,
                        AigerGate::new(
                            indexes.literal(bit),
                            (indexes.literal(x.as_ref()), indexes.literal(y.as_ref())),
                        ),
                    );

                    // Only insert children if this is a new insertion. If this gate has already
                    // been inserted then its children have also been inserted as well
                    Self::insert_gates(x.as_ref(), indexes, gates);
                    Self::insert_gates(y.as_ref(), indexes, gates);
                }
            }
            SymbolicBit::Not(x) => {
                Self::insert_gates(x, indexes, gates);
            }

            // No other type can lead to an and gate
            _ => (),
        }
    }

    /// Iterator of input literals
    pub fn inputs(&self) -> impl Iterator<Item = AigerLiteral> {
        (1..=self.num_inputs).into_iter().map(AigerLiteral::new)
    }

    /// Iterator of output literals
    pub fn outputs(&self) -> impl Iterator<Item = AigerLiteral> + '_ {
        self.outputs.iter().copied()
    }

    /// Iterator of and gates literals
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
            M = self.num_inputs + self.gates.len(),
            I = self.num_inputs,
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

/// An AIGER literal. These are used to represent inputs, outputs, and encode and gates.
#[repr(transparent)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AigerLiteral(usize);

impl std::fmt::Display for AigerLiteral {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl AigerLiteral {
    /// Create a new literal from the given index. Note that this index is a global for the
    /// circuit, meaning the number of inputs can influence the index of an and gate.
    pub const fn new(index: usize) -> Self {
        Self(index << 1)
    }

    /// The negation of a literal is encoded by toggling the least significant bit.
    pub const fn negated(self) -> Self {
        Self(self.0 ^ 0b1)
    }

    /// The index of this literal. Note that the 0 index is reserved for the literal false value.
    pub const fn index(&self) -> usize {
        self.0 >> 1
    }

    /// Determine whether this literal is negated
    pub const fn is_negated(&self) -> bool {
        (self.0 & 0b1) == 0b1
    }
}

/// The encoding for an AIGER and gate. The gate is encoded in parts:
///
/// 1. The LHS is the literal identifying this gate
/// 2. The RHS is a pair of literals representing the gate inputs
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct AigerGate {
    lhs: AigerLiteral,
    rhs: (AigerLiteral, AigerLiteral),
}

impl AigerGate {
    /// Create a new AIGER gate. The `lhs` is the literal for the gate, and the `rhs` are the
    /// inputs to the gate.
    pub fn new(lhs: AigerLiteral, rhs: (AigerLiteral, AigerLiteral)) -> Self {
        // Normalize so first RHS is always greater than second. Therefore LHS > RHS.0 >= RHS.1
        if rhs.0 > rhs.1 {
            Self { lhs, rhs }
        } else {
            Self {
                lhs,
                rhs: (rhs.1, rhs.0),
            }
        }
    }

    /// Serialize the aiger gate into binary format. In this the delta values `lhs - rhs.0` and
    /// `rhs.0 - rhs.1` are LEB128 encoded. Note that `rhs` literals may be internally swapped to
    /// ensure that `rhs.0 > rhs.1`
    pub fn serialize_binary(&self) -> Vec<u8> {
        let deltas = (self.lhs.0 - self.rhs.0 .0, self.rhs.0 .0 - self.rhs.1 .0);
        let mut encodings = ([0u8; Self::max_leb_size()], [0u8; Self::max_leb_size()]);
        let lens = (
            Self::leb128_encode(deltas.0, &mut encodings.0),
            Self::leb128_encode(deltas.1, &mut encodings.1),
        );

        encodings
            .0
            .into_iter()
            .take(lens.0)
            .chain(encodings.1.into_iter().take(lens.1))
            .collect()
    }

    /// Encode the given value using LEB128 format. This encoding is a little-endian format that
    /// uses 7 bits per byte. If this is not the final byte in the sequence, then the byte's
    /// most significant bit is set.
    fn leb128_encode(mut value: usize, output: &mut [u8; Self::max_leb_size()]) -> usize {
        const MAX_WORD_VALUE: usize = 0x7F;

        let mut i = 0;
        loop {
            if value <= MAX_WORD_VALUE {
                // SAFETY: index is guaranteed to be within range
                unsafe {
                    // Last byte does not have MSB set
                    *output.get_unchecked_mut(i) = value as u8;
                }
                break;
            } else {
                // SAFETY: index is guaranteed to be within range
                unsafe {
                    // Get lower 7 bits, set MSB
                    *output.get_unchecked_mut(i) = (value & 0x7F) as u8 | 0x80;
                }
                value >>= 7;
                i += 1;
            }
        }

        i + 1
    }

    /// The maximum number of bytes required to represent a `usize` value when LEB128 encoded.
    const fn max_leb_size() -> usize {
        let num_bits = 8 * std::mem::size_of::<usize>();

        // LEB encoding can hold 7 bits of data. Add 6 to round up instead of down via division
        (num_bits + 6) / 7
    }
}

/// The identifier for an and gate
#[derive(Debug, PartialEq, Eq, Hash)]
struct AndId(usize, usize);

impl AndId {
    /// Construct a new and gate identifier from the inputs of a [SymbolicBit::And] gate. The
    /// addresses of the reference counters are used to ensure that cloned gates will have the same
    /// identifier. This does mean that gates that are equivalent but were not cloned will have
    /// different identifiers.
    pub fn new(lhs: &Rc<SymbolicBit>, rhs: &Rc<SymbolicBit>) -> Self {
        let lhs = Rc::as_ptr(lhs) as usize;
        let rhs = Rc::as_ptr(rhs) as usize;
        if lhs < rhs {
            Self(lhs, rhs)
        } else {
            Self(rhs, lhs)
        }
    }
}

/// A mapping of identifiers to indexes.
#[derive(Default)]
struct Indexes {
    /// Mapping variable id to index
    variables: HashMap<usize, usize>,

    /// Mapping `AndId` to index. Note that this index is the internal and gate index. The global
    /// index can only be calculated once all indexes have been stored
    ands: HashMap<AndId, usize>,
}

impl Indexes {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn insert_indexes(&mut self, bit: &SymbolicBit) {
        match bit {
            SymbolicBit::Variable(id) => {
                let index = self.variables.len() + 1;
                self.variables.entry(*id).or_insert(index);
            }
            SymbolicBit::And(x, y) => {
                self.insert_indexes(x.as_ref());
                self.insert_indexes(y.as_ref());
                let id = AndId::new(x, y);
                let index = self.ands.len() + 1;
                self.ands.entry(id).or_insert(index);
            }
            SymbolicBit::Not(x) => {
                self.insert_indexes(x.as_ref());
            }
            SymbolicBit::Literal(_) => (),
        }
    }

    /// Get the index for the given [SymbolicBit].
    ///
    /// # Panics
    ///
    /// Will panic if the `bit` is not indexed
    pub fn index(&self, bit: &SymbolicBit) -> usize {
        match bit {
            SymbolicBit::Variable(id) => *self.variables.get(id).unwrap(),
            SymbolicBit::And(x, y) => {
                let id = AndId::new(x, y);
                *self.ands.get(&id).unwrap() + self.num_input_literals()
            }
            SymbolicBit::Literal(_) => panic!("literal bits are not indexed"),
            SymbolicBit::Not(_) => panic!("negated bits are not indexed"),
        }
    }

    /// Get the number of input literals indexed.
    pub fn num_input_literals(&self) -> usize {
        self.variables.len()
    }

    /// Get the AIGER literal for this [SymbolicBit].
    ///
    /// # Panics
    ///
    /// Will panic if this bit is an unindexed variable or and gate, or the negation of such a bit.
    pub fn literal(&self, bit: &SymbolicBit) -> AigerLiteral {
        match bit {
            SymbolicBit::Literal(false) => FALSE_LITERAL,
            SymbolicBit::Literal(true) => TRUE_LITERAL,
            SymbolicBit::Variable(_) | SymbolicBit::And(_, _) => AigerLiteral::new(self.index(bit)),
            SymbolicBit::Not(x) => self.literal(x.as_ref()).negated(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn output_false() {
        let aiger = Aiger::from_bits(std::iter::once(SymbolicBit::Literal(false)));
        let inputs = aiger.inputs().collect::<Vec<_>>();
        assert!(inputs.is_empty());

        let gates = aiger.gates().collect::<Vec<_>>();
        assert!(gates.is_empty());

        let outputs = aiger.outputs().collect::<Vec<_>>();
        assert_eq!(outputs, vec![FALSE_LITERAL]);

        assert_eq!(
            aiger.serialize_binary(),
            Vec::from("aag 0 0 0 1 0\n0\n".as_bytes())
        );
    }

    #[test]
    fn output_true() {
        let aiger = Aiger::from_bits(std::iter::once(SymbolicBit::Literal(true)));
        let inputs = aiger.inputs().collect::<Vec<_>>();
        assert!(inputs.is_empty());

        let gates = aiger.gates().collect::<Vec<_>>();
        assert!(gates.is_empty());

        let outputs = aiger.outputs().collect::<Vec<_>>();
        assert_eq!(outputs, vec![TRUE_LITERAL]);

        assert_eq!(
            aiger.serialize_binary(),
            Vec::from("aag 0 0 0 1 0\n1\n".as_bytes())
        );
    }

    #[test]
    fn output_variable() {
        let aiger = Aiger::from_bits(std::iter::once(SymbolicBit::Variable(0)));
        let inputs = aiger.inputs().collect::<Vec<_>>();
        assert_eq!(inputs, vec![AigerLiteral::new(1)]);

        let gates = aiger.gates().collect::<Vec<_>>();
        assert!(gates.is_empty());

        let outputs = aiger.outputs().collect::<Vec<_>>();
        assert_eq!(outputs, vec![AigerLiteral::new(1)]);

        assert_eq!(
            aiger.serialize_binary(),
            Vec::from("aag 1 1 0 1 0\n2\n".as_bytes())
        );
    }

    #[test]
    fn output_negated_variable() {
        let aiger = Aiger::from_bits(std::iter::once(SymbolicBit::Not(Rc::new(
            SymbolicBit::Variable(0),
        ))));
        let inputs = aiger.inputs().collect::<Vec<_>>();
        assert_eq!(inputs, vec![AigerLiteral::new(1)]);

        let gates = aiger.gates().collect::<Vec<_>>();
        assert!(gates.is_empty());

        let outputs = aiger.outputs().collect::<Vec<_>>();
        assert_eq!(outputs, vec![AigerLiteral::new(1).negated()]);

        assert_eq!(
            aiger.serialize_binary(),
            Vec::from("aag 1 1 0 1 0\n3\n".as_bytes())
        );
    }

    #[test]
    fn output_and_variables() {
        let aiger = Aiger::from_bits(std::iter::once(SymbolicBit::And(
            Rc::new(SymbolicBit::Variable(0)),
            Rc::new(SymbolicBit::Variable(1)),
        )));
        let inputs = aiger.inputs().collect::<Vec<_>>();
        assert_eq!(inputs, vec![AigerLiteral::new(1), AigerLiteral::new(2)]);

        let gates = aiger.gates().collect::<Vec<_>>();
        assert_eq!(
            gates,
            vec![AigerGate::new(
                AigerLiteral::new(3),
                (AigerLiteral::new(1), AigerLiteral::new(2))
            )]
        );

        let outputs = aiger.outputs().collect::<Vec<_>>();
        assert_eq!(outputs, vec![AigerLiteral::new(3)]);

        let mut expected = Vec::from("aag 3 2 0 1 1\n6\n".as_bytes());
        expected.push(0x02);
        expected.push(0x02);
        assert_eq!(aiger.serialize_binary(), expected);
    }

    #[test]
    fn output_or_variables() {
        let aiger = Aiger::from_bits(std::iter::once(SymbolicBit::Not(Rc::new(
            SymbolicBit::And(
                Rc::new(SymbolicBit::Not(Rc::new(SymbolicBit::Variable(0)))),
                Rc::new(SymbolicBit::Not(Rc::new(SymbolicBit::Variable(1)))),
            ),
        ))));
        let inputs = aiger.inputs().collect::<Vec<_>>();
        assert_eq!(inputs, vec![AigerLiteral::new(1), AigerLiteral::new(2)]);

        let gates = aiger.gates().collect::<Vec<_>>();
        assert_eq!(
            gates,
            vec![AigerGate::new(
                AigerLiteral::new(3),
                (
                    AigerLiteral::new(1).negated(),
                    AigerLiteral::new(2).negated()
                )
            )]
        );

        let outputs = aiger.outputs().collect::<Vec<_>>();
        assert_eq!(outputs, vec![AigerLiteral::new(3).negated()]);

        let mut expected = Vec::from("aag 3 2 0 1 1\n7\n".as_bytes());
        expected.push(0x01);
        expected.push(0x02);
        assert_eq!(aiger.serialize_binary(), expected);
    }

    #[test]
    fn half_adder() {
        let both = SymbolicBit::Variable(0) & SymbolicBit::Variable(1);
        let neither = !SymbolicBit::Variable(0) & !SymbolicBit::Variable(1);
        let bits = vec![!both.clone() & !neither, both];
        let aiger = Aiger::from_bits(bits);

        let inputs = aiger.inputs().collect::<Vec<_>>();
        assert_eq!(inputs, vec![AigerLiteral::new(1), AigerLiteral::new(2)]);

        let gates = aiger.gates().collect::<Vec<_>>();
        assert_eq!(
            gates,
            vec![
                AigerGate::new(
                    AigerLiteral::new(3),
                    (AigerLiteral::new(1), AigerLiteral::new(2))
                ),
                AigerGate::new(
                    AigerLiteral::new(4),
                    (
                        AigerLiteral::new(1).negated(),
                        AigerLiteral::new(2).negated()
                    )
                ),
                AigerGate::new(
                    AigerLiteral::new(5),
                    (
                        AigerLiteral::new(3).negated(),
                        AigerLiteral::new(4).negated()
                    )
                ),
            ]
        );

        let outputs = aiger.outputs().collect::<Vec<_>>();
        assert_eq!(outputs, vec![AigerLiteral::new(5), AigerLiteral::new(3)]);

        let mut expected = Vec::from("aag 5 2 0 2 3\n10\n6\n".as_bytes());
        expected.extend_from_slice(&[0x02, 0x02, 0x03, 0x02, 0x01, 0x02]);
        assert_eq!(aiger.serialize_binary(), expected);
    }

    #[test]
    fn gate_serialization() {
        let serialized = AigerGate::new(
            AigerLiteral::new(2),
            (
                AigerLiteral::new(1).negated(),
                AigerLiteral::new(1).negated(),
            ),
        )
        .serialize_binary();
        assert_eq!(serialized, vec![0x1, 0x0]);

        let serialized = AigerGate::new(
            AigerLiteral::new(0x80 >> 1),
            (AigerLiteral::new(0), AigerLiteral::new(0)),
        )
        .serialize_binary();
        assert_eq!(serialized, vec![0x80, 0x01, 0x0]);

        let serialized = AigerGate::new(
            AigerLiteral::new(0x7F >> 1).negated(),
            (AigerLiteral::new(0), AigerLiteral::new(0)),
        )
        .serialize_binary();
        assert_eq!(serialized, vec![0x7F, 0x0]);

        let serialized = AigerGate::new(
            AigerLiteral::new(0x102 >> 1),
            (AigerLiteral::new(0), AigerLiteral::new(0)),
        )
        .serialize_binary();
        assert_eq!(serialized, vec![0x82, 0x02, 0x0]);

        // 2^14 - 1
        let serialized = AigerGate::new(
            AigerLiteral::new((0b1_0000000_0000000 - 1) >> 1).negated(),
            (AigerLiteral::new(0), AigerLiteral::new(0)),
        )
        .serialize_binary();
        assert_eq!(serialized, vec![0xFF, 0x7F, 0x0]);

        // 2^14 + 3
        let serialized = AigerGate::new(
            AigerLiteral::new((0b1_0000000_0000000 + 3) >> 1).negated(),
            (AigerLiteral::new(0), AigerLiteral::new(0)),
        )
        .serialize_binary();
        assert_eq!(serialized, vec![0x83, 0x80, 0x01, 0x0]);

        // 2^28 - 1
        let serialized = AigerGate::new(
            AigerLiteral::new((0b1_0000000_0000000_0000000_0000000 - 1) >> 1).negated(),
            (AigerLiteral::new(0), AigerLiteral::new(0)),
        )
        .serialize_binary();
        assert_eq!(serialized, vec![0xFF, 0xFF, 0xFF, 0x7F, 0x00]);

        // 2^28 + 7
        let serialized = AigerGate::new(
            AigerLiteral::new((0b1_0000000_0000000_0000000_0000000 + 7) >> 1).negated(),
            (AigerLiteral::new(0), AigerLiteral::new(0)),
        )
        .serialize_binary();
        assert_eq!(serialized, vec![0x87, 0x80, 0x80, 0x80, 0x01, 0x00]);
    }
}
