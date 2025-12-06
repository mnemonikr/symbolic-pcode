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

