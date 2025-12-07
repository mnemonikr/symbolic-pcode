/// False AIGER literal
pub const FALSE: AigerLiteral = AigerLiteral::new(0);

/// True AIGER literal
pub const TRUE: AigerLiteral = FALSE.negated();

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

/// The encoding for an AIGER and gate. The gate is encoded in LHS and RHS. The LHS is the literal
/// identifying this gate. The RHS is a pair of literals representing the gate inputs.
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

    /// The Aiger literal for this gate.
    pub fn gate_literal(&self) -> AigerLiteral {
        self.lhs
    }

    /// The Aiger literal for the LHS gate input.
    pub fn input_lhs(&self) -> AigerLiteral {
        self.rhs.0
    }

    /// The Aiger literal for the RHS gate input.
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

        // LEB encoding can hold 7 bits of data. Round up to nearest LEB byte
        num_bits.div_ceil(7)
    }
}
