Emulate instructions for Ghidra-supported processors using concrete or symbolic execution of Ghidra p-code.

# Usage

TODO

# P-code operations

This section document which p-code operations are currently supported by the emulator. Emulation of instructions with unsupported p-code operations will trigger a panic.

|        | Opcode        |        | Opcode           |        | Opcode           |        | Opcode            |
| ------ | ------------- | ------ | ---------------- | ------ | ---------------- | ------ | ----------------  |
| ✅     | `COPY`        | ✅     | `INT_SLESS`      | ⬜     | `INT_LESS`       | ⬜     | `FLOAT_LESS`      |
| ✅     | `LOAD`        | ⬜     | `INT_SLESSEQUAL` | ⬜     | `INT_LESSEQUAL`  | ⬜     | `FLOAT_LESSEQUAL` |
| ✅     | `STORE`       | ✅     | `INT_SCARRY`     | ✅     | `INT_CARRY`      | ⬜     | `FLOAT_ABS`       |
| ✅     | `BOOL_NEGATE` | ✅     | `INT_NEGATE`     | ✅     | `INT_2COMP`      | ⬜     | `FLOAT_NEG`       |
| ✅     | `BOOL_XOR`    | ✅     | `INT_XOR`        | ✅     | `INT_ADD`        | ⬜     | `FLOAT_ADD`       |
| ✅     | `BOOL_AND`    | ✅     | `INT_OR`         | ✅     | `INT_SUB`        | ⬜     | `FLOAT_SUB`       |
| ✅     | `BOOL_OR`     | ✅     | `INT_AND`        | ⬜     | `INT_MULT`       | ⬜     | `FLOAT_MULT`      |
| ⬜     | `BRANCH`      | ⬜     | `INT_SDIV`       | ⬜     | `INT_DIV`        | ⬜     | `FLOAT_DIV`       |
| ⬜     | `CBRANCH`     | ✅     | `INT_EQUAL`      | ✅     | `INT_NOTEQUAL`   | ⬜     | `FLOAT_SQRT`      |
| ✅     | `BRANCHIND`   | ⬜     | `INT_SREM`       | ⬜     | `INT_REM`        | ⬜     | `FLOAT_CEIL`      |
| ⬜     | `CALL`        | ✅     | `INT_SEXT`       | ✅     | `INT_ZEXT`       | ⬜     | `FLOAT_FLOOR`     |
| ⬜     | `CALLIND`     | ⬜     | `INT_SBORROW`    | ⬜     | `INT_LEFT`       | ⬜     | `FLOAT_ROUND`     |
| ✅     | `RETURN`      | ⬜     | `INT_SRIGHT`     | ⬜     | `INT_RIGHT`      | ⬜     | `FLOAT_NAN`       |
| ⬜     | `PIECE`       |        |                  |        |                  | ⬜     | `INT2FLOAT`       |
| ✅     | `SUBPIECE`    |        |                  |        |                  | ⬜     | `FLOAT2INT`       |
| ✅     | `POPCOUNT`    |        |                  |        |                  | ⬜     | `TRUNC`           |

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
