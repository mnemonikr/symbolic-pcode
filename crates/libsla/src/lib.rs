//! This crate includes Rust bindings to the
//! [Ghidra](https://github.com/NationalSecurityAgency/ghidra) Sleigh library libsla for translating
//! native code to p-code.
//!
//! Sleigh is a processor specification language developed for the Ghidra used to describe
//! microprocessors with enough detail to facilitate disassembly and decompilation. The
//! processor-specific instructions are translated to **p-code**, which captures the instruction
//! semantics independent of the specific processor. The details on how to perform this translation
//! are captured by the compiled Sleigh specification for the processor.

mod opcodes;
mod sleigh;

pub use opcodes::*;
pub use sleigh::*;

#[cfg(test)]
mod tests;
