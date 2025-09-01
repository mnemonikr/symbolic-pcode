# Overview

This crate provides Rust bindings to the Sleigh library libsla found in [NSA's Ghidra](https://github.com/NationalSecurityAgency/ghidra),
which disassembles processor instructions into p-code. This enables binary analysis programs to
analyze arbitrary programs by targeting p-code instead of specific instruction set architectures.

# Configuration

Building a Sleigh instance requires a _compiled sleigh specification_ (.sla) and a
_processor specification_ (.pspec). These can be obtained from the
[sleigh-config](https://crates.io/crates/sleigh-config) crate.

Processor specification files are responsible for filling in context data defined in sla files. For
example, `addrsize` is variable context defined in the x86 sla file. The x86-64 pspec defines this
as `2` for 64-bit addressing while the x86 pspec defines this as `1` for 32-bit addressing. Note the
sla file is responsible for interpreting the meaning of these values.

## Custom Sleigh Specification

Custom sleigh specification files can be compiled from Rust using the
[sleigh-compiler](https://crates.io/crates/sleigh-compiler) crate. Alternatively the original
compiler can be built from the
[Ghidra decompiler source](https://github.com/NationalSecurityAgency/ghidra/blob/stable/Ghidra/Features/Decompiler/src/decompile/cpp)
using `make sleigh_opt`.

# Example

```rust
// Build Sleigh with configuration files from sleigh-config crate
let sleigh = GhidraSleigh::builder()
    .processor_spec(sleigh_config::processor_x86::PSPEC_X86_64)?
    .build(sleigh_config::processor_x86::SLA_X86_64)?;

// The instruction reader is defined by the user and implements the LoadImage trait.
let instruction_reader = InstructionReader::new();

// Instruction to decode from the reader.
let instruction_offset = 0x800000;
let address_space = sleigh.default_code_space();
let instruction_address = Address::new(instruction_offset, address_space);

// Disassemble!
let pcode_disassembly = sleigh.disassemble_pcode(&instruction_reader, instruction_address)?;
```
