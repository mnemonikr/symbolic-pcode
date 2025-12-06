/// A representation of a symbolic bit. It is assumed the symbolic library being used already
/// provides its own symbolic bit implementation. It is recommended that a conversion to this type
/// be implemented.
///
/// ## Example
///
/// ```
/// # use std::rc::Rc;
/// # use aiger_circuit;
/// // The library symbolic bit
/// enum LibSymBit {
///     Literal(bool),
///     Variable(usize),
///     Not(Rc<Self>),
///     And(Rc<Self>, Rc<Self>),
/// }
///
/// impl<'a> From<&'a LibSymBit> for aiger_circuit::AigerCircuit<'a, LibSymBit> {
///     fn from(value: &'a LibSymBit) -> Self {
///         match value {
///             LibSymBit::Literal(b) => aiger_circuit::AigerCircuit::Literal(*b),
///             LibSymBit::Variable(id) => aiger_circuit::AigerCircuit::Variable(*id),
///             LibSymBit::Not(x) => aiger_circuit::AigerCircuit::Not(x.as_ref()),
///             LibSymBit::And(x, y) => aiger_circuit::AigerCircuit::And(x.as_ref(), y.as_ref()),
///         }
///     }
/// }
/// ```
pub enum AigerCircuit<'a, T> {
    And(&'a T, &'a T),
    Not(&'a T),
    Variable(usize),
    Literal(bool),
}
