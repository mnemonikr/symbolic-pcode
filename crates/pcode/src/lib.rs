//! This crate enables the execution of pcode using symbolic represenations. The resulting
//! execution trace forms a set of constraints that can then either be evaluated or solved using a
//! constraint solver.
//!
//! ### Processor
//!
//! The processor holds a reference to the [sla::Sleigh] object for translating native code to
//! pcode.
//!
//! The [processor::ProcessorResponseHandler] trait must be implemented for the processor to handle
//! changes that occur in response to state changes. For example, an x86 processor updates the
//! instruction pointer prior to execution of the instruction. Since this requires knowing the size
//! of the instruction to be executed, the [processor::ProcessorResponseHandler::decoded] function
//! would be used to update the register state prior to execution.

/// Pcode emulation module.
pub mod emulator;

/// Module for storing the result of [pcode_ops::PcodeOps] at an [sla::Address] in an [sla::AddressSpace].
pub mod mem;

/// Module for tracking execution of a series of instructions. Supports branch exploration when
/// branching predicate cannot be evaluted to a boolean value.
pub mod processor;

#[cfg(test)]
mod tests;
