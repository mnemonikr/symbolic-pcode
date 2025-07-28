//! This crate is a system crate that builds the
//! [Ghidra](https://github.com/NationalSecurityAgency/ghidra) Sleigh library libsla for translating
//! native code to p-code. It is not recommended to depend on this crate directly unless you are
//! adding Rust bindings to interface with the library.

pub mod api;
pub mod rust;
pub mod sys;

// Reexport cxx crate for interfacing
pub use cxx;
