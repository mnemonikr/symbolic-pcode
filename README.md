Emulate instructions for Ghidra-supported processors using concrete or symbolic execution of Ghidra p-code.

# Usage

TODO

# P-code operations

This section document which p-code operations are currently supported by the emulator. Emulation of instructions with unsupported p-code operations will trigger a panic.

| Opcode        | Status | Opcode           | Status | Opcode            | Status |
| ------------- | ------ | ---------------- | ------ | ----------------  | ------ |
| `COPY`        | ✅     | `INT_LESS`       | ⬜     | `FLOAT_LESS`      | ⬜     |
| `LOAD`        | ✅     | `INT_LESSEQUAL`  | ⬜     | `FLOAT_LESSEQUAL` | ⬜     |
| `STORE`       | ✅     | `INT_ADD`        | ✅     | `FLOAT_ADD`       | ⬜     |
| `BRANCH`      | ⬜     | `INT_SUB`        | ✅     | `FLOAT_SUB`       | ⬜     |
| `CBRANCH`     | ⬜     | `INT_MULT`       | ⬜     | `FLOAT_MULT`      | ⬜     |
| `BRANCHIND`   | ✅     | `INT_DIV`        | ⬜     | `FLOAT_DIV`       | ⬜     |
| `CALL`        | ⬜     | `INT_2COMP`      | ✅     | `FLOAT_NEG`       | ⬜     |
| `CALLIND`     | ⬜     | `INT_ZEXT`       | ✅     | `FLOAT_ABS`       | ⬜     |
| `RETURN`      | ✅     | `INT_SEXT`       | ✅     | `FLOAT_SQRT`      | ⬜     |
| `PIECE`       | ⬜     | `INT_SBORROW`    | ⬜     | `FLOAT_CEIL`      | ⬜     |
| `SUBPIECE`    | ✅     | `INT_CARRY`      | ✅     | `FLOAT_FLOOR`     | ⬜     |
| `POPCOUNT`    | ✅     | `INT_SCARRY`     | ✅     | `FLOAT_ROUND`     | ⬜     |
| `BOOL_NEGATE` | ✅     | `INT_NEGATE`     | ✅     | `FLOAT_NAN`       | ⬜     |
| `BOOL_XOR`    | ✅     | `INT_XOR`        | ✅     | `INT2FLOAT`       | ⬜     |
| `BOOL_AND`    | ✅     | `INT_AND`        | ✅     | `FLOAT2INT`       | ⬜     |
| `BOOL_OR`     | ✅     | `INT_OR`         | ✅     | `TRUNC`           | ⬜     |
|               |        | `INT_LEFT`       | ⬜     |                   |        |
|               |        | `INT_RIGHT`      | ⬜     |  
|               |        | `INT_SRIGHT`     | ⬜     |
|               |        | `INT_EQUAL`      | ✅     |
|               |        | `INT_NOTEQUAL`   | ✅     |
|               |        | `INT_SLESS`      | ✅     |
|               |        | `INT_SLESSEQUAL` | ⬜     |
|               |        | `INT_REM`        | ⬜     |
|               |        | `INT_SREM`       | ⬜     |
|               |        | `INT_SDIV`       | ⬜     |

# How it works

## Native code to p-code

This crate includes Rust bindings to the Ghidra Sleigh library libsla for translating native code to p-code.

Sleigh is a processor specification language developed for the [NSA's Ghidra project](https://github.com/NationalSecurityAgency/ghidra) used to describe microprocessors with enough detail to facilitate disassembly and decompilation. The processor-specific instructions are translated to **p-code**, which captures the instruction semantics independent of the specific processor. The details on how to perform this translation are captured by the compiled Sleigh specification for the processor.

# Sleigh Specifications

A _compiled_ Sleigh specification (.sla file) is required to translate native code to p-code.

## Sleigh Compiler

The sleigh compiler must be built from the [Ghidra decompiler source](https://github.com/NationalSecurityAgency/ghidra/blob/stable/Ghidra/Features/Decompiler/src/decompile/cpp) using `make sleigh_opt`.

## Compile Sleigh Processor Specification (.slaspec)

The relevant .slaspec file must be compiled using the Sleigh compiler to generate the appropriate .sla file for the target architecture. Existing .slaspec files can be found in the [Ghidra processors repository](https://github.com/NationalSecurityAgency/ghidra/tree/stable/Ghidra/Processors).
