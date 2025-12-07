use std::{
    ops::{BitAnd, Not},
    rc::Rc,
};

/// Implement this trait to convert from another circuit type to an AigerCircuit. This allows the
/// [crate::Aiger] object to traverse the circuit to convert it into the AIGER format. If you do
/// not already have a circuit object you can use [SimpleCircuit].
pub trait AsAigerCircuit<'a> {
    /// The inner type of the AigerCircuit. Generally this will match the implementing type.
    type Inner: AsAigerCircuit<'a>;

    fn as_circuit(&'a self) -> AigerCircuit<'a, Self::Inner>;
}

/// A representation of the output of a circuit. Consumers of symbolic libraries with their own
/// representations of a circuit output should implement a conversion to this type via
/// [AsAigerCircuit]. See [SimpleCircuit] as an example.
#[derive(Debug)]
pub enum AigerCircuit<'a, T> {
    And(AndOperand<'a, T>, AndOperand<'a, T>),
    Not(&'a T),
    Variable(usize),
    Literal(bool),
}

/// Operand for [AigerCircuit::And].
#[derive(Debug)]
pub struct AndOperand<'a, T> {
    pub id: usize,
    pub value: &'a T,
}

/// The bare minimum constraint representation needed to be used in an [AigerCircuit]. This is an
/// example of how integration can be performed, but can also be used if a more fully featured
/// constraint representation is unavailable.
#[repr(transparent)]
#[derive(Clone, Debug)]
pub struct SimpleCircuit(Rc<SimpleConstraint>);

impl BitAnd for SimpleCircuit {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        Self(Rc::new(SimpleConstraint::And(self.0, rhs.0)))
    }
}

impl Not for SimpleCircuit {
    type Output = Self;

    fn not(self) -> Self::Output {
        Self(Rc::new(SimpleConstraint::Not(self.0)))
    }
}

impl SimpleCircuit {
    /// A new variable bit. The `id` is used to distinguish this variable from other variables.
    pub fn var(id: usize) -> Self {
        Self(Rc::new(SimpleConstraint::Variable(id)))
    }

    /// A new boolean literal.
    pub fn lit(val: bool) -> Self {
        Self(Rc::new(SimpleConstraint::Literal(val)))
    }
}

/// The inner value for a [SimpleCircuit].
#[derive(Clone, Debug)]
pub enum SimpleConstraint {
    And(Rc<Self>, Rc<Self>),
    Not(Rc<Self>),
    Variable(usize),
    Literal(bool),
}

impl<'a> AsAigerCircuit<'a> for SimpleCircuit {
    type Inner = SimpleConstraint;

    fn as_circuit(&'a self) -> AigerCircuit<'a, Self::Inner> {
        match self.0.as_ref() {
            SimpleConstraint::And(x, y) => AigerCircuit::And(
                AndOperand {
                    id: Rc::as_ptr(x) as usize,
                    value: x.as_ref(),
                },
                AndOperand {
                    id: Rc::as_ptr(y) as usize,
                    value: y.as_ref(),
                },
            ),
            SimpleConstraint::Not(x) => AigerCircuit::Not(x.as_ref()),
            SimpleConstraint::Variable(id) => AigerCircuit::Variable(*id),
            SimpleConstraint::Literal(v) => AigerCircuit::Literal(*v),
        }
    }
}

impl<'a> AsAigerCircuit<'a> for SimpleConstraint {
    type Inner = SimpleConstraint;

    fn as_circuit(&'a self) -> AigerCircuit<'a, Self::Inner> {
        match self {
            SimpleConstraint::And(x, y) => AigerCircuit::And(
                AndOperand {
                    id: Rc::as_ptr(x) as usize,
                    value: x.as_ref(),
                },
                AndOperand {
                    id: Rc::as_ptr(y) as usize,
                    value: y.as_ref(),
                },
            ),
            SimpleConstraint::Not(x) => AigerCircuit::Not(x.as_ref()),
            SimpleConstraint::Variable(id) => AigerCircuit::Variable(*id),
            SimpleConstraint::Literal(v) => AigerCircuit::Literal(*v),
        }
    }
}
