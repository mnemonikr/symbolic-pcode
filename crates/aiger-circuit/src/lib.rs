#![doc=include_str!("../README.md")]

pub mod circuit;
pub mod model;

mod aiger;
mod index;

pub use aiger::Aiger;

#[cfg(test)]
mod tests;
