#![doc=include_str!("../README.md")]

mod bit;
mod buf;
mod convert;
mod eval;
mod vec;

pub use crate::bit::*;
pub use crate::buf::*;
pub use crate::convert::ConcreteValue;
pub use crate::convert::ConcretizationError;
pub use crate::convert::concretize;
pub use crate::convert::concretize_into;
pub use crate::eval::*;
pub use crate::vec::*;

#[cfg(feature = "aiger")]
mod aiger;

#[cfg(test)]
mod tests;
