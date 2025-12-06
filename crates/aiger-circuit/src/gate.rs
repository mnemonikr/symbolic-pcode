use crate::literal::AigerLiteral;

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

        // LEB encoding can hold 7 bits of data. Round up to nearest LEB byte
        num_bits.div_ceil(7)
    }
}
