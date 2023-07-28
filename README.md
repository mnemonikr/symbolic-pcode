Emulate instructions for Ghidra-supported processors using concrete or symbolic execution of Ghidra p-code.

# Usage

TODO

# P-code operations

This section document which p-code operations are currently supported by the emulator. Emulation of instructions with unsupported p-code operations will trigger a panic.

| Opcode                             | Opcode                                | Opcode                               | Opcode                                 |
| ---------------------------------- | ------------------------------------- | -------------------------------------| -------------------------------------  |
| :white_check_mark:   `COPY`        | :white_check_mark:   `INT_SLESS`      | :white_large_square: `INT_LESS`      | :white_large_square: `FLOAT_LESS`      |
| :white_check_mark:   `LOAD`        | :white_large_square: `INT_SLESSEQUAL` | :white_large_square: `INT_LESSEQUAL` | :white_large_square: `FLOAT_LESSEQUAL` |
| :white_check_mark:   `STORE`       | :white_check_mark:   `INT_SCARRY`     | :white_check_mark:   `INT_CARRY`     | :white_large_square: `FLOAT_ABS`       |
| :white_check_mark:   `BOOL_NEGATE` | :white_check_mark:   `INT_NEGATE`     | :white_check_mark:   `INT_2COMP`     | :white_large_square: `FLOAT_NEG`       |
| :white_check_mark:   `BOOL_XOR`    | :white_check_mark:   `INT_XOR`        | :white_check_mark:   `INT_ADD`       | :white_large_square: `FLOAT_ADD`       |
| :white_check_mark:   `BOOL_AND`    | :white_check_mark:   `INT_OR`         | :white_check_mark:   `INT_SUB`       | :white_large_square: `FLOAT_SUB`       |
| :white_check_mark:   `BOOL_OR`     | :white_check_mark:   `INT_AND`        | :white_large_square: `INT_MULT`      | :white_large_square: `FLOAT_MULT`      |
| :white_large_square: `BRANCH`      | :white_large_square: `INT_SDIV`       | :white_large_square: `INT_DIV`       | :white_large_square: `FLOAT_DIV`       |
| :white_large_square: `CBRANCH`     | :white_check_mark:   `INT_EQUAL`      | :white_check_mark:   `INT_NOTEQUAL`  | :white_large_square: `FLOAT_SQRT`      |
| :white_check_mark:   `BRANCHIND`   | :white_large_square: `INT_SREM`       | :white_large_square: `INT_REM`       | :white_large_square: `FLOAT_CEIL`      |
| :white_large_square: `CALL`        | :white_check_mark:   `INT_SEXT`       | :white_check_mark:   `INT_ZEXT`      | :white_large_square: `FLOAT_FLOOR`     |
| :white_large_square: `CALLIND`     | :white_large_square: `INT_SBORROW`    | :white_large_square: `INT_LEFT`      | :white_large_square: `FLOAT_ROUND`     |
| :white_check_mark:   `RETURN`      | :white_large_square: `INT_SRIGHT`     | :white_large_square: `INT_RIGHT`     | :white_large_square: `FLOAT_NAN`       |
| :white_large_square: `PIECE`       |                                       | :white_large_square: `INT2FLOAT`     |
| :white_check_mark:   `SUBPIECE`    |                                       | :white_large_square: `FLOAT2INT`     |
| :white_check_mark:   `POPCOUNT`    |                                       | :white_large_square: `TRUNC`         |

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
