//! Convert circuit of [SymbolicBit]s into [AIGER format](https://github.com/arminbiere/aiger/blob/master/FORMAT).

use std::collections::{BTreeMap, HashMap};

/// A representation of a symbolic bit. It is assumed the symbolic library being used already
/// provides its own symbolic bit implementation. It is recommended that a conversion to this type
/// be implemented.
///
/// ## Example
///
/// ```
/// # use std::rc::Rc;
/// # use sym::aiger;
/// // The library symbolic bit
/// enum LibSymBit {
///     Literal(bool),
///     Variable(usize),
///     Not(Rc<Self>),
///     And(Rc<Self>, Rc<Self>),
/// }
///
/// impl From<&LibSymBit> for aiger::SymbolicBit<LibSymBit> {
///     fn from(value: &LibSymBit) -> Self {
///         match value {
///             LibSymBit::Literal(b) => aiger::SymbolicBit::Literal(*b),
///             LibSymBit::Variable(id) => aiger::SymbolicBit::Variable(*id),
///             LibSymBit::Not(x) => aiger::SymbolicBit::Not(Rc::as_ptr(x)),
///             LibSymBit::And(x, y) => aiger::SymbolicBit::And(Rc::as_ptr(x), Rc::as_ptr(y)),
///         }
///     }
/// }
/// ```
pub enum SymbolicBit<T> {
    And(*const T, *const T),
    Not(*const T),
    Variable(usize),
    Literal(bool),
}

/// False AIGER literal
pub const FALSE: AigerLiteral = AigerLiteral::new(0);

/// True AIGER literal
pub const TRUE: AigerLiteral = FALSE.negated();

/// Object capable of coverting [SymbolicBit]s into AIGER format.
pub struct Aiger {
    inputs: HashMap<AigerLiteral, usize>,
    outputs: Vec<AigerLiteral>,
    gates: Vec<AigerGate>,
}

impl Aiger {
    /// Create an Aiger object from the given output bits. This will walk the circuit and gather the
    /// corresponding Aiger data. Identical pointers in [SymbolicBit::And] and [SymbolicBit::Not]
    /// are deduplicated, but otherwise no equivalence checking is performed for circuit reduction.
    pub fn from_bits<B>(bits: impl IntoIterator<Item = B>) -> Self
    where
        for<'a> &'a B: Into<SymbolicBit<B>>,
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
            inputs: indexes
                .variables
                .iter()
                .map(|(&variable_id, &index)| (AigerLiteral::new(index), variable_id))
                .collect(),
            outputs,
            gates,
        }
    }

    /// Insert all [SymbolicBit::And] gates into the tree. The tree is a mapping from the gate
    /// index to the [AigerGate] composed of [AigerLiteral]s.
    fn insert_gates<B>(bit: &B, indexes: &Indexes, gates: &mut BTreeMap<usize, AigerGate>)
    where
        for<'a> &'a B: Into<SymbolicBit<B>>,
    {
        match bit.into() {
            SymbolicBit::And(x, y) => {
                let index = indexes.index(bit);
                let x = unsafe { &*x };
                let y = unsafe { &*y };

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
            SymbolicBit::Not(x) => {
                Self::insert_gates(unsafe { &*x }, indexes, gates);
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

    /// Determine whether this literal is either TRUE or FALSE
    pub const fn is_const(&self) -> bool {
        self.0 >> 1 == 0
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
pub struct AigerGate {
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

    pub fn gate_literal(&self) -> AigerLiteral {
        self.lhs
    }

    pub fn input_lhs(&self) -> AigerLiteral {
        self.rhs.0
    }

    pub fn input_rhs(&self) -> AigerLiteral {
        self.rhs.1
    }

    /// Serialize the aiger gate into binary format. In this the delta values `lhs - rhs.0` and
    /// `rhs.0 - rhs.1` are LEB128 encoded. Note that `rhs` literals may be internally swapped to
    /// ensure that `rhs.0 > rhs.1`
    pub fn serialize_binary(&self) -> Vec<u8> {
        let deltas = (self.lhs.0 - self.rhs.0.0, self.rhs.0.0 - self.rhs.1.0);
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

/// A mapping of identifiers to indexes. This is an internal structure used for computing the
/// indexes of variables and gates.
#[derive(Default)]
struct Indexes {
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

    /// Insert all components of the [SymbolicBit] into the index. Components already inserted are
    /// ignored.
    pub fn insert_indexes<B>(&mut self, bit: &B)
    where
        for<'a> &'a B: Into<SymbolicBit<B>>,
    {
        match bit.into() {
            SymbolicBit::Variable(id) => {
                let index = self.variables.len() + 1;
                self.variables.entry(id).or_insert(index);
            }
            SymbolicBit::And(x, y) => {
                let id = AndId::new(x, y);
                if !self.ands.contains_key(&id) {
                    self.insert_indexes(unsafe { &*x });
                    self.insert_indexes(unsafe { &*y });

                    // Index this gate at the end to ensure its index is greater than the index of
                    // anything contained in either the lhs or rhs
                    self.ands.insert(id, self.ands.len() + 1);
                }
            }
            SymbolicBit::Not(x) => {
                self.insert_indexes(unsafe { &*x });
            }
            SymbolicBit::Literal(_) => (),
        }
    }

    /// Get the index for the given [SymbolicBit].
    ///
    /// # Panics
    ///
    /// Will panic if the `bit` is not indexed
    pub fn index<B>(&self, bit: &B) -> usize
    where
        for<'a> &'a B: Into<SymbolicBit<B>>,
    {
        match bit.into() {
            SymbolicBit::Variable(id) => *self.variables.get(&id).unwrap(),
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
    pub fn literal<B>(&self, bit: &B) -> AigerLiteral
    where
        for<'a> &'a B: Into<SymbolicBit<B>>,
    {
        match bit.into() {
            SymbolicBit::Literal(false) => FALSE,
            SymbolicBit::Literal(true) => TRUE,
            SymbolicBit::Variable(_) | SymbolicBit::And(_, _) => AigerLiteral::new(self.index(bit)),
            SymbolicBit::Not(x) => self.literal(unsafe { &*x }).negated(),
        }
    }
}
