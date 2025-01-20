# Overview

This crate provides Rust bindings to the Sleigh library libsla found in [NSA's Ghidra](https://github.com/NationalSecurityAgency/ghidra),
which disassembles processor instructions into p-code. This enables binary analysis programs to
analyze arbitrary programs by targeting p-code instead of specific instruction set architectures.

# Requirements

A Sleigh instance requires a _compiled sleigh specification_ (.sla) and a _processor specification_
(.pspec).

## Sleigh Specification (.sla)

The relevant .slaspec file must be compiled using the Sleigh compiler to generate the appropriate
.sla file for the target architecture. Existing .slaspec files can be found in the
[Ghidra processors repository](https://github.com/NationalSecurityAgency/ghidra/tree/stable/Ghidra/Processors).

## Sleigh Compiler

The sleigh compiler must be built from the
[Ghidra decompiler source](https://github.com/NationalSecurityAgency/ghidra/blob/stable/Ghidra/Features/Decompiler/src/decompile/cpp)
using `make sleigh_opt`.

## Processor Specification (.pspec)

Processor specification files can be found in
[Ghidra processors repository](https://github.com/NationalSecurityAgency/ghidra/tree/stable/Ghidra/Processors).
These are responsible for filling in context data defined in sla files. For example, `addrsize` is
variable context defined in the x86 sla file. The x86-64 pspec defines this as `2` for 64-bit
addressing while the x86 pspec defines this as `1` for 32-bit addressing. Note the sla file is 
responsible for interpreting the meaning of these values.

# Example

This gives an overview of the general structure for disassembling code into p-code. For a working
example using x86-64 see the [src/sleigh.rs](sleigh unit tests);

```rust
// Construct new sleigh instance
let mut sleigh = GhidraSleigh::new();

// Compiled from x86-64.slaspec in Ghidra repository
let slaspec = std::fs::read_to_string("x86-64.sla");

// Located in Ghidra repository. No compilation necessary.
let pspec = std::fs::read_to_string("x86-64.pspec");

// Initialize sleigh. Required before decoding can be performed.
sleigh.initialize(&slaspec, &pspec).expect("failed to initialize sleigh");

// The instruction reader is defined by the user and implements the LoadImage trait.
let instruction_reader = InstructionReader::new();

// Instruction to decode from the reader.
let instruction_offset = 0x800000;
let address_space = sleigh.default_code_space();
let instruction_address = Address::new(instruction_offset, address_space);

// Disassemble!
let pcode_disassembly = sleigh.disassemble_pcode(&instruction_reader, instruction_address).expect("disassembly failed");
```

