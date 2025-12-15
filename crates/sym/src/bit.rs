use std::rc::Rc;

pub const FALSE: SymbolicBit = SymbolicBit::Literal(false);
pub const TRUE: SymbolicBit = SymbolicBit::Literal(true);

/// A value that can be used to represent a variable bit, possibly with constraints on its value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SymbolicBit {
    /// A literal `true` or `false` value.
    Literal(bool),

    /// A variable value. The parameter is the identifier for this variable. Two variables with the
    /// same identifier are equivalent.
    Variable(usize),

    /// The negation of a symbolic bit. The `!` operator should be preferred to this, as it has the
    /// opportunity to perform simplications where a direct construction does not.
    Not(Rc<Self>),

    /// The conjunction of two symbolic bits. The `&` operator should be preferred to this, as it
    /// has the opportunity to perform simpliciations where a direct construction does not.
    And(Rc<Self>, Rc<Self>),
}

impl SymbolicBit {
    pub fn maybe_literal(&self) -> Option<bool> {
        match self {
            Self::Literal(b) => Some(*b),
            _ => None,
        }
    }

    pub fn maybe_variable(&self) -> Option<usize> {
        match self {
            Self::Variable(id) => Some(*id),
            _ => None,
        }
    }

    pub fn equals(self, rhs: Self) -> Self {
        (self.clone() & rhs.clone()) | (!self & !rhs)
    }

    pub fn select(self, lhs: Self, rhs: Self) -> Self {
        (self.clone() & lhs) | (!self & rhs)
    }

    pub fn is_identical(&self, rhs: &Self) -> bool {
        match self {
            Self::Literal(x) => {
                if let Self::Literal(y) = rhs {
                    return *x == *y;
                }
            }
            Self::Variable(x) => {
                if let Self::Variable(y) = rhs {
                    return *x == *y;
                }
            }
            Self::Not(x) => {
                if let Self::Not(y) = rhs {
                    if Rc::ptr_eq(x, y) {
                        return true;
                    } else if let Self::Variable(x) = **x
                        && let Self::Variable(y) = **y
                    {
                        // Check if same variable
                        return x == y;
                    }
                }
            }
            Self::And(x, y) => {
                if let Self::And(u, v) = rhs {
                    if Rc::ptr_eq(x, u) && Rc::ptr_eq(y, v) || Rc::ptr_eq(x, v) && Rc::ptr_eq(y, u)
                    {
                        return true;
                    } else if let Self::Variable(x) = **x
                        && let Self::Variable(y) = **y
                        && let Self::Variable(u) = **u
                        && let Self::Variable(v) = **v
                    {
                        // Check if same variables
                        return x == u && y == v || x == v && y == u;
                    }
                }
            }
        }

        false
    }
}

impl Default for SymbolicBit {
    fn default() -> Self {
        FALSE
    }
}

impl std::ops::Not for SymbolicBit {
    type Output = Self;

    fn not(self) -> Self::Output {
        match self {
            SymbolicBit::Literal(false) => SymbolicBit::Literal(true),
            SymbolicBit::Literal(true) => SymbolicBit::Literal(false),
            SymbolicBit::Not(y) => Rc::unwrap_or_clone(y),
            _ => SymbolicBit::Not(Rc::new(self)),
        }
    }
}

impl std::ops::BitAnd for SymbolicBit {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        if self.is_identical(&rhs) {
            return self;
        }

        match self {
            SymbolicBit::Literal(false) => return SymbolicBit::Literal(false),
            SymbolicBit::Literal(true) => return rhs,
            SymbolicBit::Not(z) if z.is_identical(&rhs) => return SymbolicBit::Literal(false),
            _ => (),
        }

        match rhs {
            SymbolicBit::Literal(false) => return SymbolicBit::Literal(false),
            SymbolicBit::Literal(true) => return self,
            SymbolicBit::Not(z) if z.is_identical(&self) => return SymbolicBit::Literal(false),
            _ => (),
        }

        SymbolicBit::And(Rc::new(self), Rc::new(rhs))
    }
}

impl std::ops::BitOr for SymbolicBit {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        !(!self & !rhs)
    }
}

impl std::ops::BitXor for SymbolicBit {
    type Output = Self;

    fn bitxor(self, rhs: Self) -> Self::Output {
        (self.clone() & !rhs.clone()) | (!self & rhs)
    }
}
