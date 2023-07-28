Emulate instructions for Ghidra-supported processors using concrete or symbolic execution of Ghidra p-code.

# Usage

TODO

# P-code operations

This section document which p-code operations are currently supported by the emulator. Emulation of instructions with unsupported p-code operations will trigger a panic.

## Memory operations

| Opcode      | Status |
| ----------- | ------ |
| `COPY`      | ✅     |
| `LOAD`      | ✅     |
| `STORE`     | ✅     |

## Control flow operations

| Opcode      | Status |
| ----------- | ------ |
| `BRANCH`    | ⬜     |
| `CBRANCH`   | ⬜     |
| `BRANCHIND` | ✅     |
| `CALL`      | ⬜     |
| `CALLIND`   | ⬜     |
| `RETURN`    | ✅     |


## General operations

| Opcode     | Status |
| ---------- | ------ |
| `PIECE`    | ⬜     |
| `SUBPIECE` | ✅     |
| `POPCOUNT` | ✅     |

## Integer operations

| Opcode           | Status |
| ---------------- | ------ |
| `INT_EQUAL`      | ✅     |
| `INT_NOTEQUAL`   | ✅     |
| `INT_LESS`       | ⬜     |
| `INT_SLESS`      | ✅     |
| `INT_LESSEQUAL`  | ⬜     |
| `INT_SLESSEQUAL` | ⬜     |
| `INT_ZEXT`       | ✅     |
| `INT_SEXT`       | ✅     |
| `INT_ADD`        | ✅     |
| `INT_SUB`        | ✅     |
| `INT_CARRY`      | ✅     |
| `INT_SCARRY`     | ✅     |
| `INT_SBORROW`    | ⬜     |
| `INT_2COMP`      | ✅     |
| `INT_NEGATE`     | ✅     |
| `INT_XOR`        | ✅     |
| `INT_AND`        | ✅     |
| `INT_OR`         | ✅     |
| `INT_LEFT`       | ⬜     |
| `INT_RIGHT`      | ⬜     |
| `INT_SRIGHT`     | ⬜     |
| `INT_MULT`       | ⬜     |
| `INT_DIV`        | ⬜     |
| `INT_REM`        | ⬜     |
| `INT_SDIV`       | ⬜     |
| `INT_SREM`       | ⬜     |

## Boolean operations

| Opcode        | Status |
| ------------- | ------ |
| `BOOL_NEGATE` | ✅     |
| `BOOL_XOR`    | ✅     |
| `BOOL_AND`    | ✅     |
| `BOOL_OR`     | ✅     |

## Floating point operations

| Opcode            | Status  |
| ----------------- | ------- |
| `FLOAT_LESS`      | ⬜      |
| `FLOAT_LESSEQUAL` | ⬜      |
| `FLOAT_ADD`       | ⬜      |
| `FLOAT_SUB`       | ⬜      |
| `FLOAT_MULT`      | ⬜      |
| `FLOAT_DIV`       | ⬜      |
| `FLOAT_NEG`       | ⬜      |
| `FLOAT_ABS`       | ⬜      |
| `FLOAT_SQRT`      | ⬜      |
| `FLOAT_CEIL`      | ⬜      |
| `FLOAT_FLOOR`     | ⬜      |
| `FLOAT_ROUND`     | ⬜      |
| `FLOAT_NAN`       | ⬜      |
| `INT2FLOAT`       | ⬜      |
| `FLOAT2INT`       | ⬜      |
| `TRUNC`           | ⬜      |

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
