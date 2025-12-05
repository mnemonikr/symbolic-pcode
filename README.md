Execute instructions for Ghidra-supported processors using concrete or symbolic execution of Ghidra p-code.

# Usage

The public API for this library has not yet been finalized. See [x86-64 integration tests](tests/x86_64_emulator.rs) for examples using library primitives.

# P-code operations

This section document which p-code operations are currently supported by the emulator.

| Opcode                             | Opcode                                | Opcode                             | Opcode                                 |
| ---------------------------------- | ------------------------------------- | -----------------------------------| -------------------------------------  |
| :white_check_mark: `COPY`        | :white_check_mark: `INT_SLESS`      | :white_check_mark: `INT_LESS`      | :white_large_square: `FLOAT_LESS`      |
| :white_check_mark: `LOAD`        | :white_check_mark: `INT_SLESSEQUAL` | :white_check_mark: `INT_LESSEQUAL` | :white_large_square: `FLOAT_LESSEQUAL` |
| :white_check_mark: `STORE`       | :white_check_mark: `INT_EQUAL`      | :white_check_mark: `INT_NOTEQUAL`  | :white_large_square: `FLOAT_ABS`       |
| :white_check_mark: `BOOL_NEGATE` | :white_check_mark: `INT_NEGATE`     | :white_check_mark: `INT_2COMP`     | :white_large_square: `FLOAT_NEG`       |
| :white_check_mark: `BOOL_XOR`    | :white_check_mark: `INT_XOR`        | :white_check_mark: `INT_ADD`       | :white_large_square: `FLOAT_ADD`       |
| :white_check_mark: `BOOL_AND`    | :white_check_mark: `INT_OR`         | :white_check_mark: `INT_SUB`       | :white_large_square: `FLOAT_SUB`       |
| :white_check_mark: `BOOL_OR`     | :white_check_mark: `INT_AND`        | :white_check_mark: `INT_MULT`      | :white_large_square: `FLOAT_MULT`      |
| :white_check_mark: `BRANCH`      | :white_check_mark: `INT_SDIV`       | :white_check_mark: `INT_DIV`       | :white_large_square: `FLOAT_DIV`       |
| :white_check_mark: `CBRANCH`     | :white_check_mark: `INT_SREM`       | :white_check_mark: `INT_REM`       | :white_large_square: `FLOAT_SQRT`      |
| :white_check_mark: `BRANCHIND`   | :white_check_mark: `INT_SCARRY`     | :white_check_mark: `INT_CARRY`     | :white_large_square: `FLOAT_CEIL`      |
| :white_check_mark: `CALL`        | :white_check_mark: `INT_SEXT`       | :white_check_mark: `INT_ZEXT`      | :white_large_square: `FLOAT_FLOOR`     |
| :white_check_mark: `CALLIND`     | :white_check_mark: `INT_SBORROW`    | :white_check_mark: `INT_LEFT`      | :white_large_square: `FLOAT_ROUND`     |
| :white_check_mark: `RETURN`      | :white_check_mark: `INT_SRIGHT`     | :white_check_mark: `INT_RIGHT`     | :white_large_square: `FLOAT_NAN`       |
| :white_check_mark: `PIECE`       |                                     |                                    | :white_large_square: `INT2FLOAT`       |
| :white_check_mark: `SUBPIECE`    |                                     |                                    | :white_large_square: `FLOAT2INT`       |
| :white_check_mark: `POPCOUNT`    |                                     |                                    | :white_large_square: `TRUNC`           |

# How it works

## Native code to p-code

This uses [libsla](https://crates.io/crates/libsla) to interface with Ghidra's SLEIGH library for translating native code to p-code. SLEIGH is a processor specification language developed for the [NSA's Ghidra project](https://github.com/NationalSecurityAgency/ghidra) used to describe microprocessors with enough detail to facilitate disassembly and decompilation. The processor-specific instructions are translated to **p-code**, which captures the instruction semantics independent of the specific processor.

The details on how to perform this translation are captured by the compiled SLEIGH specification for the processor. Precompiled specifications from Ghidra can be found in the [sleigh-config](https://crates.io/crates/sleigh-config) crate.

# Building tests

## Test Fixtures

The test fixtures are Rust components built targeting specific architectures. Make sure you have the following targets installed via `rustup target`:

* x86_64-unknown-none
* x86_64-unknown-linux-musl
* aarch64-unknown-none
* aarch64-unknown-linux-musl. This also requires the aarch64-linux-gnu-gcc compiler to be installed for cross-compilation.

The Linux musl variants are used by integration tests with a Linux kernel model.

## Z3

The integration tests use [Z3](https://github.com/Z3Prover/z3) to perform constraint solving and expects the Z3 shared library to be installed. Alternatively the integration test [Cargo.toml](./tests/Cargo.toml) can be updated to build and statically link Z3. See the comments in that file for more details.
